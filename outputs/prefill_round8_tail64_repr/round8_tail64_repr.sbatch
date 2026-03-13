#!/bin/bash
#SBATCH --job-name=prefill-r8-tail64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/scratch/murphy/outputs/cot-loop-detection/logs/%x-%j.out

set -euo pipefail

export HF_HOME=/data/scratch/murphy/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data/scratch/murphy/cache/huggingface/hub
export TRANSFORMERS_CACHE=/data/scratch/murphy/cache/huggingface/transformers
export VLLM_CACHE_ROOT=/data/scratch/murphy/cache/vllm
export TORCH_HOME=/data/scratch/murphy/cache/torch
export XDG_CACHE_HOME=/data/scratch/murphy/cache
export WANDB_MODE=offline

PYTHON_BIN=/data/scratch/murphy/venvs/cot-loop-detection-runtime-py310/bin/python
REPO_ROOT=/home/murphy/projects/worktrees/cot-loop-detection-runtime
OUT_ROOT=/data/scratch/murphy/outputs/cot-loop-detection
DATA_ROOT=/data/scratch/murphy/datasets/cot-loop-detection

TRAIN_POOL_JSONL="${DATA_ROOT}/k5_multisource_train_pool_v2.jsonl"
EVAL_POOL_JSONL="${DATA_ROOT}/k5_eval_math500_aime2425_v1.jsonl"
REF_TRAIN_DATA="${OUT_ROOT}/probe_data/prefill_round4_sourcectrl_windowviews"
REF_MATCHED_DATA="${OUT_ROOT}/probe_data/prefill_round4_eval_lenmatched_windowviews"
ROUND8_TRAIN_DATA="${OUT_ROOT}/probe_data/prefill_round8_tail64_sourcectrl"
ROUND8_MATCHED_DATA="${OUT_ROOT}/probe_data/prefill_round8_eval_lenmatched_tail64"
ROUND8_RUN_ROOT="${OUT_ROOT}/probe_runs/prefill_round8_tail64_repr"

cd "${REPO_ROOT}"

rm -rf "${ROUND8_TRAIN_DATA}" "${ROUND8_MATCHED_DATA}" "${ROUND8_RUN_ROOT}"
mkdir -p "${OUT_ROOT}/logs"

common_rebuild_args=(
  --train-pool-jsonl "${TRAIN_POOL_JSONL}"
  --eval-pool-jsonl "${EVAL_POOL_JSONL}"
  --prompt-field problem
  --model-id open-thoughts/OpenThinker3-1.5B
  --trust-remote-code
  --prefill-batch-size 2
  --feature-key prefill_lasttok_layers_concat
  --feature-pooling last_token_all_layers_concat
  --feature-layer -1
  --extra-feature-view prefill_tail64_strided_final:tail64_strided_concat:-1
)

"${PYTHON_BIN}" scripts/rebuild_prefill_views_fixed_splits.py \
  --reference-data-dir "${REF_TRAIN_DATA}" \
  --out-dir "${ROUND8_TRAIN_DATA}" \
  "${common_rebuild_args[@]}"

"${PYTHON_BIN}" scripts/rebuild_prefill_views_fixed_splits.py \
  --reference-data-dir "${REF_MATCHED_DATA}" \
  --out-dir "${ROUND8_MATCHED_DATA}" \
  "${common_rebuild_args[@]}"

common_train_args=(
  --train-data-dir "${ROUND8_TRAIN_DATA}"
  --matched-eval-data-dir "${ROUND8_MATCHED_DATA}"
  --train-pool-jsonl "${TRAIN_POOL_JSONL}"
  --eval-pool-jsonl "${EVAL_POOL_JSONL}"
  --prompt-field problem
  --tokenizer-model-id open-thoughts/OpenThinker3-1.5B
  --metadata-feature-set legacy
  --selection-split val
  --selection-metric pr_auc
  --tie-breaker macro_f1
  --holdout-ratio 0.15
  --holdout-seed 0
  --seeds 0 1 2
)

"${PYTHON_BIN}" scripts/train_metadata_residual_probe.py \
  "${common_train_args[@]}" \
  --feature-key prefill_lasttok_layers_concat \
  --out-dir "${ROUND8_RUN_ROOT}/anchor"

"${PYTHON_BIN}" scripts/train_metadata_residual_probe.py \
  "${common_train_args[@]}" \
  --feature-key prefill_tail64_strided_final \
  --out-dir "${ROUND8_RUN_ROOT}/tail64_strided_final"

"${PYTHON_BIN}" scripts/train_metadata_residual_probe.py \
  "${common_train_args[@]}" \
  --feature-key prefill_lasttok_layers_concat \
  --extra-feature-key prefill_tail64_strided_final \
  --low-rank-dim 256 \
  --out-dir "${ROUND8_RUN_ROOT}/anchor_plus_tail64_lowrank256"

echo "Round 8 tail64 representation sweep completed."
