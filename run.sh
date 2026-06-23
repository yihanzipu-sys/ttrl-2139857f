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
PY=python3
$PY -m pip install --quiet --upgrade pip hf-transfer 2>&1 | tail -2

# =============================================================================
# STAGE 1: smoke — can the upgraded stack load+serve qwen3_5 at all?
# =============================================================================
log "STAGE smoke: installing minimal inference stack (transformers + vLLM nightly)"

# transformers new enough to know qwen3_5; let pip resolve a compatible torch.
$PY -m pip install --quiet "transformers>=4.57" accelerate "numpy<2.0.0" 2>&1 | tail -3

# vLLM nightly carries Qwen3_5ForConditionalGeneration in its model registry.
# Prebuilt wheel (no source compile) from the nightly index.
$PY -m pip install --quiet --pre vllm \
  --extra-index-url https://wheels.vllm.ai/nightly 2>&1 | tail -5

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

# =============================================================================
# Data prep: AIME-TTT json -> parquet (verl format). Tiny (30 problems).
# =============================================================================
log "preprocess AIME-TTT -> parquet"
$PY - <<'PYEOF' 2>&1 | tail -5
import os, datasets
src = "verl/data/AIME-TTT"
def mk(split):
    def fn(ex, idx):
        return {"data_source":"aime",
                "prompt":[{"role":"user","content":ex["prompt"]+"\nPlease reason step by step, and put your final answer within \\boxed{}."}],
                "ability":"math",
                "reward_model":{"style":"rule","ground_truth":ex["answer"]},
                "extra_info":{"split":split,"index":f"aime-{idx}"}}
    return fn
for split,f in [("train","train.json"),("test","test.json")]:
    ds = datasets.load_dataset("json", data_files=os.path.join(src,f), split="train")
    ds = ds.map(mk(split), with_indices=True, remove_columns=ds.column_names)
    ds.to_parquet(os.path.join(src, f.replace(".json",".parquet")))
    print(split, len(ds), "->", os.path.join(src, f.replace('.json','.parquet')))
PYEOF

# =============================================================================
# STAGE 2: eval — baseline (no-TTRL) pass@1 / maj@n via vLLM. The CONTROL.
# This is also the metric we reuse after training (treatment).
# =============================================================================
run_eval () {  # $1 = model path/id, $2 = label, $3 = out json
  $PY scripts/ttrl_eval.py --model "$1" --label "$2" --out "$3" \
      --data verl/data/AIME-TTT/test.json --n 16 --max-tokens 3072 2>&1 | tail -20
}

log "STAGE eval: baseline (untrained ${MODEL_ID})"
run_eval "$MODEL_ID" "base" "$ART/eval_base.json" || fail "baseline eval failed"
[ "$STAGE" = "eval" ] && { log "stopping after eval (STAGE=eval)"; $PY scripts/ttrl_report.py "$ART"; write_eval "$ART/EVAL.md"; exit 0; }

# =============================================================================
# STAGE 3: train — TTRL GRPO (majority-vote rewards, NO ground truth), re-eval.
# =============================================================================
log "STAGE train: install verl training stack"
$PY -m pip install --quiet ray codetiming hydra-core tensordict pylatexenc \
    "latex2sympy2_extended" "math-verify" liger-kernel dill torchdata 2>&1 | tail -3
$PY -m pip install --quiet -e verl 2>&1 | tail -3

OUTDIR="$ROOT/ttrl_ckpt"
log "STAGE train: launch TTRL trainer"
bash scripts/ttrl_train.sh "$MODEL_ID" "$OUTDIR" 2>&1 | tee "$ART/train.log" | tail -40
TRAIN_RC=${PIPESTATUS[0]}
if [ "$TRAIN_RC" -ne 0 ]; then
  log "training returned non-zero ($TRAIN_RC); see train.log"
fi

# Merge the latest FSDP checkpoint to HF format, then re-eval the TTRL model.
STEP_DIR=$(ls -d "$OUTDIR"/global_step_*/actor 2>/dev/null | sort -V | tail -1)
if [ -n "$STEP_DIR" ]; then
  HFDIR="$OUTDIR/hf_merged"
  log "merge FSDP checkpoint $STEP_DIR -> $HFDIR"
  $PY -m verl.model_merger merge --backend fsdp \
      --local_dir "$STEP_DIR" --target_dir "$HFDIR" 2>&1 | tail -10
  # vLLM needs tokenizer/processor files alongside weights; copy from base if missing.
  for f in tokenizer.json tokenizer_config.json vocab.json merges.txt config.json \
           preprocessor_config.json video_preprocessor_config.json generation_config.json; do
    [ -f "$HFDIR/$f" ] || $PY - "$MODEL_ID" "$HFDIR" "$f" <<'PYEOF' 2>/dev/null || true
import sys, shutil
from huggingface_hub import hf_hub_download
mid, dst, fn = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    p = hf_hub_download(mid, fn); shutil.copy(p, dst + "/" + fn); print("copied", fn)
except Exception as e:
    pass
PYEOF
  done
  if [ -d "$HFDIR" ] && [ -n "$(ls -A "$HFDIR" 2>/dev/null)" ]; then
    log "STAGE eval: TTRL-trained model at $HFDIR"
    run_eval "$HFDIR" "ttrl" "$ART/eval_ttrl.json" || log "ttrl eval failed"
  fi
else
  log "no FSDP checkpoint found; EVAL will report training-only signal from train.log"
fi

# =============================================================================
# Final report -> EVAL.md
# =============================================================================
$PY scripts/ttrl_report.py "$ART"
write_eval "$ART/EVAL.md"
log "DONE"
