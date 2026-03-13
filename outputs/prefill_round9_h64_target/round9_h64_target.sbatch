#!/bin/bash
#SBATCH --job-name=prefill-r9-h64
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
export PIP_CACHE_DIR=/data/scratch/murphy/cache/pip
export WANDB_MODE=offline

PYTHON_BIN=/home/murphy/miniforge3/envs/swe311/bin/python
REPO_ROOT=/home/murphy/projects/worktrees/cot-loop-detection-runtime
OUT_ROOT=/data/scratch/murphy/outputs/cot-loop-detection
DATA_ROOT=/data/scratch/murphy/datasets/cot-loop-detection
SUBSET_ROOT="${OUT_ROOT}/probe_data/prefill_round9_horizon_prompt_subsets"

TRAIN_POOL_JSONL="${DATA_ROOT}/k5_multisource_train_pool_v2.jsonl"
EVAL_POOL_JSONL="${DATA_ROOT}/k5_eval_math500_aime2425_v1.jsonl"
SOURCECTRL_SUBSET_ROOT="${SUBSET_ROOT}/sourcectrl"
MATCHED_SUBSET_ROOT="${SUBSET_ROOT}/matched"

ROUND9_TRAIN_DATA="${OUT_ROOT}/probe_data/prefill_round9_h64_sourcectrl"
ROUND9_MATCHED_DATA="${OUT_ROOT}/probe_data/prefill_round9_h64_matched"
ROUND9_RUN_ROOT="${OUT_ROOT}/probe_runs/prefill_round9_h64_target"

cd "${REPO_ROOT}"

for required in \
  "${SOURCECTRL_SUBSET_ROOT}/train.jsonl" \
  "${SOURCECTRL_SUBSET_ROOT}/test.jsonl" \
  "${MATCHED_SUBSET_ROOT}/train.jsonl" \
  "${MATCHED_SUBSET_ROOT}/test.jsonl"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required Round 9 subset file: ${required}" >&2
    exit 1
  fi
done

rm -rf "${ROUND9_TRAIN_DATA}" "${ROUND9_MATCHED_DATA}" "${ROUND9_RUN_ROOT}"
mkdir -p "${OUT_ROOT}/logs"

common_build_args=(
  --prompt-field problem
  --model-id open-thoughts/OpenThinker3-1.5B
  --trust-remote-code
  --tp 1
  --dp 1
  --max-tokens 64
  --prefill-batch-size 2
  --label-target loop_by_horizon
  --label-horizon 64
  --feature-key prefill_lasttok_layers_concat
  --feature-pooling last_token_all_layers_concat
  --feature-layer -1
  --extra-feature-view prefill_tail64_strided_final:tail64_strided_concat:-1
)

"${PYTHON_BIN}" scripts/build_probe_dataset.py \
  "${common_build_args[@]}" \
  --train-dataset "${SOURCECTRL_SUBSET_ROOT}/train.jsonl" \
  --test-dataset "${SOURCECTRL_SUBSET_ROOT}/test.jsonl" \
  --out-dir "${ROUND9_TRAIN_DATA}"

"${PYTHON_BIN}" scripts/build_probe_dataset.py \
  "${common_build_args[@]}" \
  --train-dataset "${MATCHED_SUBSET_ROOT}/train.jsonl" \
  --test-dataset "${MATCHED_SUBSET_ROOT}/test.jsonl" \
  --out-dir "${ROUND9_MATCHED_DATA}"

common_train_args=(
  --train-data-dir "${ROUND9_TRAIN_DATA}"
  --matched-eval-data-dir "${ROUND9_MATCHED_DATA}"
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
  --out-dir "${ROUND9_RUN_ROOT}/h64_anchor"

"${PYTHON_BIN}" scripts/train_metadata_residual_probe.py \
  "${common_train_args[@]}" \
  --feature-key prefill_tail64_strided_final \
  --out-dir "${ROUND9_RUN_ROOT}/h64_tail64"

echo "Round 9 loop_by_64 target sweep completed."
