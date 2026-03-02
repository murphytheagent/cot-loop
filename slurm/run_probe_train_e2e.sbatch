#!/bin/bash
#SBATCH --job-name=probe-train-e2e
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# SLURM starts jobs in the submit directory, so we're already in the right place
mkdir -p logs outputs

# Prefer local venv if present; otherwise support explicit conda env activation.
if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
elif [[ -n "${CONDA_ENV:-}" ]]; then
  CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[probe-e2e] conda activation script not found: ${CONDA_SH}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
else
  echo "[probe-e2e] no runtime env found (.venv missing and CONDA_ENV not set)" >&2
  exit 1
fi

# Load user-level runtime secrets when available (e.g., WANDB_API_KEY).
if [[ -f "${HOME}/.env" ]]; then
  # shellcheck disable=SC1091
  set -a
  source "${HOME}/.env"
  set +a
fi
export HF_HUB_CONNECT_TIMEOUT="${HF_HUB_CONNECT_TIMEOUT:-30}"
export HF_HUB_READ_TIMEOUT="${HF_HUB_READ_TIMEOUT:-120}"

DEFAULT_CACHE_BASE="${HOME}/.cache"
if [[ -d "/data/users/${USER}" ]]; then
  DEFAULT_CACHE_BASE="/data/users/${USER}/cache"
elif [[ -d "/data/scratch" ]]; then
  DEFAULT_CACHE_BASE="/data/scratch/${USER}/cache"
fi
mkdir -p "${DEFAULT_CACHE_BASE}"

export HF_HOME="${HF_HOME:-${DEFAULT_CACHE_BASE}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

if [[ -z "${VLLM_CACHE_ROOT:-}" ]]; then
  export VLLM_CACHE_ROOT="${DEFAULT_CACHE_BASE}/vllm"
fi

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${VLLM_CACHE_ROOT}"
echo "[probe-e2e] HF_HOME=${HF_HOME}"
echo "[probe-e2e] HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}"
echo "[probe-e2e] TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"
echo "[probe-e2e] VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"

MODEL_PRESET="${MODEL_PRESET:-openthinker3_1p5b}"
PROMPT_FIELD="${PROMPT_FIELD:-problem}"

TRAIN_DATASET="${TRAIN_DATASET:-HuggingFaceH4/MATH-500}"
TRAIN_CONFIG="${TRAIN_CONFIG:-}"
TRAIN_SPLIT="${TRAIN_SPLIT:-test}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-}"

TEST_DATASET="${TEST_DATASET:-}"
TEST_CONFIG="${TEST_CONFIG:-}"
TEST_SPLIT="${TEST_SPLIT:-test}"
TEST_MAX_SAMPLES="${TEST_MAX_SAMPLES:-}"

SPLIT_RATIO="${SPLIT_RATIO:-0.1}"
SEED="${SEED:-}"
SEEDS="${SEEDS:-}"
DATASET_SEED="${DATASET_SEED:-}"

MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_TOKENS="${MAX_TOKENS:-}"
PREFILL_BATCH_SIZE="${PREFILL_BATCH_SIZE:-32}"

LOOP_N="${LOOP_N:-30}"
LOOP_K="${LOOP_K:-10}"
SHARD_SIZE="${SHARD_SIZE:-2048}"

OUT_DATA_DIR="${OUT_DATA_DIR:-outputs/probe_data/${MODEL_PRESET}}"
RUN_NAME="${RUN_NAME:-${MODEL_PRESET}-${SLURM_JOB_ID}}"
OUT_RUN_DIR="${OUT_RUN_DIR:-outputs/probe_runs/${RUN_NAME}}"

WANDB_PROJECT="${WANDB_PROJECT:-cot-loop-probe}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
PROBE_PRESET="${PROBE_PRESET:-mlp}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-auto}"
EVAL_EVERY="${EVAL_EVERY:-1}"
LOG_EVERY="${LOG_EVERY:-5}"

TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

declare -a SEED_LIST=()
if [[ -n "${SEEDS}" ]]; then
  read -r -a SEED_LIST <<< "${SEEDS}"
elif [[ -n "${SEED}" ]]; then
  SEED_LIST=("${SEED}")
else
  SEED_LIST=(0 1 2)
fi

if [[ "${#SEED_LIST[@]}" -eq 0 ]]; then
  echo "[probe-e2e] no seeds resolved from SEEDS/SEED" >&2
  exit 1
fi

if [[ -z "${DATASET_SEED}" ]]; then
  DATASET_SEED="${SEED_LIST[0]}"
fi

echo "[probe-e2e] model_preset=${MODEL_PRESET} train=${TRAIN_DATASET}:${TRAIN_SPLIT} test=${TEST_DATASET}:${TEST_SPLIT}"
echo "[probe-e2e] probe_preset=${PROBE_PRESET}"
echo "[probe-e2e] resolved seeds: ${SEED_LIST[*]}"

BUILD_ARGS=(
  --train-dataset "${TRAIN_DATASET}"
  --train-split "${TRAIN_SPLIT}"
  --prompt-field "${PROMPT_FIELD}"
  --split-ratio "${SPLIT_RATIO}"
  --seed "${DATASET_SEED}"
  --loop-n "${LOOP_N}"
  --loop-k "${LOOP_K}"
  --shard-size "${SHARD_SIZE}"
  --prefill-batch-size "${PREFILL_BATCH_SIZE}"
  --out-dir "${OUT_DATA_DIR}"
)

if [[ -n "${MODEL_PRESET}" ]]; then
  BUILD_ARGS+=(--model-preset "${MODEL_PRESET}")
fi
if [[ -n "${TEST_DATASET}" ]]; then
  BUILD_ARGS+=(--test-dataset "${TEST_DATASET}" --test-split "${TEST_SPLIT}")
fi

if [[ -n "${TRAIN_CONFIG}" ]]; then
  BUILD_ARGS+=(--train-config "${TRAIN_CONFIG}")
fi
if [[ -n "${TRAIN_MAX_SAMPLES}" ]]; then
  BUILD_ARGS+=(--train-max-samples "${TRAIN_MAX_SAMPLES}")
fi
if [[ -n "${TEST_DATASET}" && -n "${TEST_CONFIG}" ]]; then
  BUILD_ARGS+=(--test-config "${TEST_CONFIG}")
fi
if [[ -n "${TEST_DATASET}" && -n "${TEST_MAX_SAMPLES}" ]]; then
  BUILD_ARGS+=(--test-max-samples "${TEST_MAX_SAMPLES}")
fi
if [[ -n "${MAX_NUM_SEQS}" ]]; then
  BUILD_ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
fi
if [[ -n "${MAX_TOKENS}" ]]; then
  BUILD_ARGS+=(--max-tokens "${MAX_TOKENS}")
fi

echo "[probe-e2e] validating/reusing dataset at ${OUT_DATA_DIR} with dataset_seed=${DATASET_SEED}"
python scripts/build_probe_dataset.py "${BUILD_ARGS[@]}" --reuse-if-compatible

for RUN_SEED in "${SEED_LIST[@]}"; do
  if [[ "${#SEED_LIST[@]}" -gt 1 ]]; then
    SEED_RUN_DIR="${OUT_RUN_DIR}/seed_${RUN_SEED}"
    SEED_WANDB_RUN_NAME="${WANDB_RUN_NAME}-seed${RUN_SEED}"
  else
    SEED_RUN_DIR="${OUT_RUN_DIR}"
    SEED_WANDB_RUN_NAME="${WANDB_RUN_NAME}"
  fi

  TRAIN_ARGS=(
    --data-dir "${OUT_DATA_DIR}"
    --out-dir "${SEED_RUN_DIR}"
    --probe-preset "${PROBE_PRESET}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --lr "${LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --num-workers "${NUM_WORKERS}"
    --device "${DEVICE}"
    --seed "${RUN_SEED}"
    --eval-every "${EVAL_EVERY}"
    --log-every "${LOG_EVERY}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-run-name "${SEED_WANDB_RUN_NAME}"
  )

  echo "[probe-e2e] seed=${RUN_SEED}: train probe data_dir=${OUT_DATA_DIR} out_dir=${SEED_RUN_DIR} probe_preset=${PROBE_PRESET}"
  python scripts/train_probe.py "${TRAIN_ARGS[@]}" ${TRAIN_EXTRA_ARGS}
done

if [[ "${#SEED_LIST[@]}" -gt 1 ]]; then
  PLOT_OUT="${OUT_RUN_DIR}/probe_multiseed_curves.png"
  echo "[probe-e2e] plotting multi-seed curves from ${OUT_RUN_DIR} -> ${PLOT_OUT}"
  python scripts/plot_probe_multiseed.py \
    --run-dir "${OUT_RUN_DIR}" \
    --out "${PLOT_OUT}"
fi
