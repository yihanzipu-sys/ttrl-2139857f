#!/bin/bash
# Minimal single-GPU TTRL GRPO training, adapted from
# verl/examples/ttrl/Qwen2.5-Math/aime.sh. Majority-vote pseudo-labels, NO GT.
# $1 = model path/id, $2 = output dir
set -uo pipefail
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1

MODEL="${1:?model}"
OUTDIR="${2:?outdir}"

TASK="AIME-TTT"
DATA_LOCAL_DIR="verl/data"

# --- minimal sizing for one GPU ---
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=3072
N_VOTES_PER_PROMPT=16        # paper: 64. minimal: 16 votes -> pseudo-label
N_SAMPLES_PER_PROMPT=8       # paper: 32. minimal: 8 rollouts trained on
DATA_TRAIN_BATCH_SIZE=4
MINI_BATCH_SIZE=1
MICRO_BATCH_SIZE=1
EPISODE="${TTRL_EPISODES:-30}"

python -m verl.trainer.main_ppo \
  --config-name='ppo_trainer_ttrl.yaml' \
  data.train_files=["$DATA_LOCAL_DIR/$TASK/train.parquet"] \
  data.val_files=["$DATA_LOCAL_DIR/$TASK/test.parquet"] \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.n=$N_SAMPLES_PER_PROMPT \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=8 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.adv_estimator=grpo \
  custom_reward_function.path="./verl/utils/reward_score/ttrl_math/__init__.py" \
  custom_reward_function.name=reward_func \
  ttrl.enable=True \
  ttrl.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
  ttrl.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  trainer.logger=['console'] \
  trainer.project_name=TTRL-qwen35 \
  trainer.experiment_name="ttrl-qwen35-0.8b-aime" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq="${TTRL_SAVE_FREQ:-10}" \
  trainer.test_freq=5 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.default_local_dir="$OUTDIR" \
  trainer.total_epochs=$EPISODE "${@:3}"
