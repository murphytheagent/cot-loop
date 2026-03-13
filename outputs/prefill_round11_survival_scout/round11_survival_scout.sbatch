#!/bin/bash
#SBATCH --job-name=prefill-r11-survival
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
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
export VLLM_GPU_MEMORY_UTILIZATION=0.20

PYTHON_BIN=/home/murphy/miniforge3/envs/swe311/bin/python
REPO_ROOT=/home/murphy/projects/worktrees/cot-loop-detection-runtime
OUT_ROOT=/data/scratch/murphy/outputs/cot-loop-detection
SUBSET_JSONL="${OUT_ROOT}/probe_data/prefill_round9_horizon_prompt_subsets/sourcectrl/train.jsonl"
SCOUT_OUT="${OUT_ROOT}/analysis/prefill_round11_survival_scout/sourcectrl_train256_h30000.json"

cd "${REPO_ROOT}"
mkdir -p "$(dirname "${SCOUT_OUT}")"

"${PYTHON_BIN}" scripts/scan_horizon_label_prevalence.py \
  --dataset "${SUBSET_JSONL}" \
  --prompt-field problem \
  --out-json "${SCOUT_OUT}" \
  --sample-size 256 \
  --sample-seed 0 \
  --num-length-bins 3 \
  --horizons 2048 4096 8192 16384 30000 \
  --model-id open-thoughts/OpenThinker3-1.5B \
  --trust-remote-code \
  --tp 1 \
  --dp 1 \
  --max-tokens 30000 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --loop-n 30 \
  --loop-k 20

echo "Round 11 survival scout completed."
