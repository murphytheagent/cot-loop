#!/bin/bash
#SBATCH --job-name=collect-model-stats
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs outputs/model_stats

# Prefer an explicit conda env when requested; otherwise fall back to local venv.
if [[ -n "${CONDA_ENV:-}" ]]; then
  CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[collect-model-stats] conda activation script not found: ${CONDA_SH}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
elif [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
else
  echo "[collect-model-stats] no runtime env found (.venv missing and CONDA_ENV not set)" >&2
  exit 1
fi

# Load user-level runtime secrets when available (e.g., HF_TOKEN / WANDB_API_KEY).
if [[ -f "${HOME}/.env" ]]; then
  # shellcheck disable=SC1091
  set -a
  source "${HOME}/.env"
  set +a
fi

export VLLM_SUPPRESS_SEM_UNLINK_ERRORS=1
export HF_HUB_CONNECT_TIMEOUT=30
export HF_HUB_READ_TIMEOUT=120

DEFAULT_CACHE_BASE="${HOME}/.cache"
if [[ -d "/data/users/${USER}" ]]; then
  DEFAULT_CACHE_BASE="/data/users/${USER}/cache"
elif [[ -d "/data/scratch/${USER}" ]]; then
  DEFAULT_CACHE_BASE="/data/scratch/${USER}/cache"
elif [[ -d "/data/scratch" ]]; then
  DEFAULT_CACHE_BASE="/data/scratch/${USER}/cache"
fi
mkdir -p "${DEFAULT_CACHE_BASE}"

export HF_HOME="${HF_HOME:-${DEFAULT_CACHE_BASE}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${DEFAULT_CACHE_BASE}/datasets}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${DEFAULT_CACHE_BASE}/vllm}"
export TORCH_HOME="${TORCH_HOME:-${DEFAULT_CACHE_BASE}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${DEFAULT_CACHE_BASE}}"

if [[ -n "${PINNED_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${PINNED_VISIBLE_DEVICES}"
fi

mkdir -p \
  "${HF_HOME}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${VLLM_CACHE_ROOT}" \
  "${TORCH_HOME}" \
  "${XDG_CACHE_HOME}"

echo "[collect-model-stats] HF_HOME=${HF_HOME}"
echo "[collect-model-stats] HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}"
echo "[collect-model-stats] TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"
echo "[collect-model-stats] HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "[collect-model-stats] VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"
echo "[collect-model-stats] TORCH_HOME=${TORCH_HOME}"
echo "[collect-model-stats] XDG_CACHE_HOME=${XDG_CACHE_HOME}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[collect-model-stats] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

TASK_KIND="${TASK_KIND:-math_freeform}"
DATASET="${DATASET:-data/aime_2024_2025.jsonl}"
DATASET_CONFIG="${DATASET_CONFIG:-}"
SPLIT="${SPLIT:-test}"
QUESTION_FIELD="${QUESTION_FIELD:-}"
ANSWER_FIELD="${ANSWER_FIELD:-}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B}"
STATISTICS="${STATISTICS:-success_fraction,loop_fraction,avg_generation_length,avg_loop_generation_length,avg_first_loop_prefix_length,max_length_hit_fraction,loop_max_length_hit_fraction,max_length_hit_loop_fraction,generation_length_variance,max_length_hit_success_fraction,loop_success_fraction,avg_correct_generation_length,avg_wrong_generation_length}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
TP="${TP:-1}"
DP="${DP:-1}"
MAX_TOKENS="${MAX_TOKENS:-81920}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-0}"
LOOP_N="${LOOP_N:-30}"
LOOP_K="${LOOP_K:-20}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
RELEASE_VERSION="${RELEASE_VERSION:-release_v6}"
LIVECODEBENCH_REPO="${LIVECODEBENCH_REPO:-}"
LM_STYLE_OVERRIDE="${LM_STYLE_OVERRIDE:-}"
OUT="${OUT:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ "${TASK_KIND}" == "multiple_choice_gpqa" ]]; then
  DATASET_IS_LOCAL_FILE=0
  if [[ -f "${DATASET}" ]]; then
    DATASET_IS_LOCAL_FILE=1
  fi
  if [[ "${DATASET_IS_LOCAL_FILE}" -eq 0 && -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required for gated GPQA datasets, but local GPQA files can run without it." >&2
    exit 1
  fi
fi

if [[ "${TASK_KIND}" == "livecodebench_codegen" ]]; then
  if [[ -z "${HF_DATASETS_CACHE:-}" ]]; then
    echo "HF_DATASETS_CACHE is required for TASK_KIND=livecodebench_codegen." >&2
    exit 1
  fi
  if [[ -z "${LIVECODEBENCH_REPO}" ]]; then
    echo "LIVECODEBENCH_REPO is required for TASK_KIND=livecodebench_codegen." >&2
    exit 1
  fi
  if [[ ! -d "${LIVECODEBENCH_REPO}" ]]; then
    echo "LIVECODEBENCH_REPO does not exist: ${LIVECODEBENCH_REPO}" >&2
    exit 1
  fi
fi

ARGS=(
  --task-kind "$TASK_KIND"
  --dataset "$DATASET"
  --split "$SPLIT"
  --model-id "$MODEL_ID"
  --statistics "$STATISTICS"
  --tp "$TP"
  --dp "$DP"
  --max-tokens "$MAX_TOKENS"
  --max-model-len "$MAX_MODEL_LEN"
  --dtype "$DTYPE"
  --seed "$SEED"
  --loop-n "$LOOP_N"
  --loop-k "$LOOP_K"
  --release-version "$RELEASE_VERSION"
)

if [[ -n "${DATASET_CONFIG}" ]]; then
  ARGS+=(--dataset-config "$DATASET_CONFIG")
fi

if [[ -n "${QUESTION_FIELD}" ]]; then
  ARGS+=(--question-field "$QUESTION_FIELD")
fi

if [[ -n "${ANSWER_FIELD}" ]]; then
  ARGS+=(--answer-field "$ANSWER_FIELD")
fi

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "$MAX_SAMPLES")
fi

if [[ -n "${MAX_NUM_SEQS}" ]]; then
  ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
fi

if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
  ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi

if [[ -n "${LIVECODEBENCH_REPO}" ]]; then
  ARGS+=(--livecodebench-repo "$LIVECODEBENCH_REPO")
fi

if [[ -n "${LM_STYLE_OVERRIDE}" ]]; then
  ARGS+=(--lm-style-override "$LM_STYLE_OVERRIDE")
fi

if [[ -n "${OUT}" ]]; then
  ARGS+=(--out "$OUT")
fi

python scripts/collect_model_stats.py "${ARGS[@]}" ${EXTRA_ARGS}
