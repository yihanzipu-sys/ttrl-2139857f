#!/bin/bash
# =============================================================================
# TTRL minimal reproduction driver (arXiv:2504.16084)
#
# Goal: run the TTRL process (majority-vote pseudo-labels -> GRPO, NO ground
# truth) on a FRESH BASE model, Qwen/Qwen3.5-0.8B-Base, on a single GPU, and
# show a large margin over the untrained base model.
#
# Qwen3.5 is a 2026 hybrid linear-attention multimodal arch (`qwen3_5`). The
# TTRL fork's verl is pinned to vllm<=0.8.5 which DOES NOT support it, so the
# stack is upgraded:
#   - transformers (has qwen3_5 modeling code)
#   - vLLM nightly  (has Qwen3_5ForConditionalGeneration in its registry)
#
# Staged + FAIL-FAST so a bad combo fails cheap (timeboxed). The first stage is
# a cheap loadability gate: if vLLM cannot even serve qwen3_5, we stop before
# paying for the heavy verl training install.
#
# Stages (set STAGE env to stop early; default = all):
#   smoke : install minimal stack, confirm vLLM loads+generates from the model
#   eval  : measure baseline (no-TTRL) pass@1 / maj@n  -> the control
#   train : run TTRL GRPO training, then re-eval  -> the treatment
# =============================================================================
set -uo pipefail

MODEL_ID="Qwen/Qwen3.5-0.8B-Base"
STAGE="${STAGE:-all}"
# Benchmark: MATH-L1 (easiest tier, 43 problems) — a 0.8B model has enough prior
# here for majority voting to produce signal (AIME is too hard: no consensus).
TASK="${TASK:-MATH-L1-TTT}"
ART=".openresearch/artifacts"
mkdir -p "$ART"
ROOT="$(cd "$(dirname "$0")" && pwd)"

log()  { echo "[run.sh $(date +%H:%M:%S)] $*"; }
fail() { echo "FAILED: $*" | tee -a "$ART/status.txt"; cp -f "$ART/EVAL.md" EVAL.md 2>/dev/null || true; exit 1; }

# Always leave an EVAL.md so the run is interpretable even on early exit.
write_eval() { cp -f "$1" EVAL.md; [ "$1" != "$ART/EVAL.md" ] && cp -f "$1" "$ART/EVAL.md" || true; }
cat > "$ART/EVAL.md" <<EOF
# TTRL on ${MODEL_ID} — in progress
Run started; no results yet. See \`status.txt\` and \`smoke.txt\` for stage progress.
EOF
write_eval "$ART/EVAL.md"

# -----------------------------------------------------------------------------
log "Python / GPU info"
python3 --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || fail "no GPU visible"

export HF_HUB_ENABLE_HF_TRANSFER=1
export PIP_ROOT_USER_ACTION=ignore
# Skip the FlashInfer GDN linear-attention prefill JIT compile (hangs on first
# run for this qwen3_5 hybrid-mamba model); use the triton backend instead.
export VLLM_GDN_PREFILL_BACKEND=triton
PY=python3
$PY -m pip install --quiet --upgrade pip hf-transfer 2>&1 | tail -2

# =============================================================================
# STAGE 1: smoke — can the upgraded stack load+serve qwen3_5 at all?
# =============================================================================
log "STAGE smoke: installing minimal inference stack (transformers + vLLM nightly)"

# transformers new enough to know qwen3_5; let pip resolve a compatible torch.
# Also light deps needed by preprocess (datasets/pandas/pyarrow) and the
# standalone evaluator's grader (math_verify) — none pull torch.
$PY -m pip install --quiet "transformers>=4.57" accelerate "numpy<2.0.0" \
  datasets pandas "pyarrow>=15.0.0" math-verify latex2sympy2_extended 2>&1 | tail -3

# vLLM nightly carries Qwen3_5ForConditionalGeneration in its model registry.
# Prebuilt wheel (no source compile) from the nightly index.
$PY -m pip install --quiet --pre vllm \
  --extra-index-url https://wheels.vllm.ai/nightly 2>&1 | tail -5

# FAST_TRAIN=1 skips the (already-proven) smoke gate AND the (already-measured)
# baseline eval, jumping straight to TTRL training. The qwen3_5 vLLM stack and
# the base AIME numbers (pass@1=1.67, maj@16=6.67) are established; re-running
# them every iteration just burns GPU. Set FAST_TRAIN=0 to do the full pipeline.
# Default OFF here: MATH-L1 is a new benchmark, so measure its real baseline
# (the AIME baseline doesn't apply) then train then re-eval on the same set.
# L1 baseline already measured (pass@1=76.6, maj@16=90.7) — reuse it and jump
# straight to training so the run finishes within the instance's time budget.
FAST_TRAIN="${FAST_TRAIN:-1}"
BASE_PASS1="${BASE_PASS1:-76.6}"
BASE_MAJ="${BASE_MAJ:-90.7}"

if [ "$FAST_TRAIN" = "1" ]; then
  log "FAST_TRAIN=1: skipping smoke gate + baseline eval (already established)"
else
  log "verify qwen3_5 loads in vLLM, then generate (run from a FILE so vLLM spawn works)"
  $PY scripts/ttrl_smoke.py "$MODEL_ID" 2>&1 | tee "$ART/smoke.txt"
  SMOKE_RC=${PIPESTATUS[0]}
  if [ "$SMOKE_RC" -ne 0 ] || ! grep -q SMOKE_PASS "$ART/smoke.txt"; then
    cat > "$ART/EVAL.md" <<EOF
# TTRL on ${MODEL_ID} — BLOCKED at smoke stage

vLLM could not load/serve \`${MODEL_ID}\` (\`qwen3_5\` hybrid-attention arch).
This is a **stack-support blocker**, not a TTRL result. See \`smoke.txt\` for the traceback.
EOF
    write_eval "$ART/EVAL.md"
    fail "smoke stage: vLLM cannot serve qwen3_5"
  fi
  log "STAGE smoke: PASS"
  [ "$STAGE" = "smoke" ] && { echo "stopping after smoke (STAGE=smoke)"; exit 0; }
fi

# =============================================================================
# Eval helper — vLLM pass@1/maj@n on the TASK test set. The CONTROL + treatment.
# (No parquet preprocessing needed; the standalone trainer/eval read the json.)
# =============================================================================
TEST_JSON="verl/data/${TASK}/test.json"
log "TASK=${TASK}  test set: ${TEST_JSON}"
run_eval () {  # $1 = model path/id, $2 = label, $3 = out json
  $PY scripts/ttrl_eval.py --model "$1" --label "$2" --out "$3" \
      --data "$TEST_JSON" --n 16 --max-tokens 1024 2>&1 | tail -20
}

if [ "$FAST_TRAIN" = "1" ]; then
  log "FAST_TRAIN=1: reusing seeded baseline from \$BASE_PASS1/\$BASE_MAJ"
  cat > "$ART/eval_base.json" <<EOF
{"label":"base","model":"${MODEL_ID}","n_samples":16,"n_problems":0,"pass@1":${BASE_PASS1:-0},"maj@16":${BASE_MAJ:-0},"per_problem":[]}
EOF
else
  log "STAGE eval: baseline (untrained ${MODEL_ID}) on ${TASK}"
  run_eval "$MODEL_ID" "base" "$ART/eval_base.json" || fail "baseline eval failed"
  [ "$STAGE" = "eval" ] && { log "stopping after eval (STAGE=eval)"; $PY scripts/ttrl_report.py "$ART"; write_eval "$ART/EVAL.md"; exit 0; }
fi

# =============================================================================
# STAGE 3: train — TTRL GRPO (majority-vote rewards, NO ground truth), re-eval.
# Self-contained HF-transformers trainer (verl's vLLM rollout is incompatible
# with vLLM 0.23 for qwen3_5). Same TTRL mechanism: vote -> binary reward -> GRPO.
# =============================================================================
HFDIR="$ROOT/ttrl_ckpt_hf"
log "STAGE train: standalone TTRL GRPO (no verl) — test-time RL on ${TASK} test set"
# TTRL = test-time RL: train (label-free) on the SAME test set we evaluate.
$PY -u scripts/ttrl_grpo.py --model "$MODEL_ID" \
    --data "$TEST_JSON" \
    --out "$HFDIR" --artifacts "$ART" \
    --steps "${TTRL_STEPS:-20}" --group-size 8 --prompts-per-step 8 \
    --max-new-tokens 1024 --lr 1e-6 --kl-coef 0.0 2>&1 | tee "$ART/train.log" | tail -80
TRAIN_RC=${PIPESTATUS[0]}
[ "$TRAIN_RC" -ne 0 ] && log "training returned non-zero ($TRAIN_RC); see train.log"

if [ -f "$HFDIR/config.json" ]; then
  log "STAGE eval: TTRL-trained model at $HFDIR"
  run_eval "$HFDIR" "ttrl" "$ART/eval_ttrl.json" || log "ttrl eval failed"
else
  log "no TTRL checkpoint saved; see train.log"
fi

# =============================================================================
# Final report -> EVAL.md
# =============================================================================
$PY scripts/ttrl_report.py "$ART"
write_eval "$ART/EVAL.md"
log "DONE"
