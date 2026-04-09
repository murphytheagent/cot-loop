#!/bin/bash
#SBATCH --job-name=prompt-profile-length-mech
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs outputs

# Prefer an explicit conda env when requested; otherwise fall back to local venv.
if [[ -n "${CONDA_ENV:-}" ]]; then
  CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[prompt-profile-length-mech] conda activation script not found: ${CONDA_SH}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
elif [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
else
  echo "[prompt-profile-length-mech] no runtime env found (.venv missing and CONDA_ENV not set)" >&2
  exit 1
fi

if [[ -f "${HOME}/.env" ]]; then
  # shellcheck disable=SC1091
  set -a
  source "${HOME}/.env"
  set +a
fi

if [[ -d "/data/users/${USER}" ]]; then
  CACHE_BASE="/data/users/${USER}/cache"
elif [[ -d "/data/scratch/${USER}" ]]; then
  CACHE_BASE="/data/scratch/${USER}/cache"
else
  echo "[prompt-profile-length-mech] expected a data-volume cache root outside HOME" >&2
  exit 1
fi

mkdir -p "${CACHE_BASE}"
export HF_HOME="${HF_HOME:-${CACHE_BASE}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${CACHE_BASE}/vllm}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_BASE}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_BASE}}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${VLLM_CACHE_ROOT}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

REGRESSION_ROOT="${REGRESSION_ROOT:-/data/scratch/${USER}/outputs/cot-loop-detection/prompt_profile_regression_natural_rerun_20260405}"
OUT_ROOT="${OUT_ROOT:-/data/scratch/${USER}/outputs/cot-loop-detection/prompt_profile_length_mechanism_20260408}"
DATASET_LIST="${DATASET_LIST:-gpqa aime math500 mmlu_pro livecodebench}"
SELECTION_KIND="${SELECTION_KIND:-best_loss}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-256}"

read -r -a DATASETS <<< "${DATASET_LIST}"

mkdir -p "${OUT_ROOT}"

echo "[prompt-profile-length-mech] REGRESSION_ROOT=${REGRESSION_ROOT}"
echo "[prompt-profile-length-mech] OUT_ROOT=${OUT_ROOT}"
echo "[prompt-profile-length-mech] DATASETS=${DATASET_LIST}"
echo "[prompt-profile-length-mech] SELECTION_KIND=${SELECTION_KIND}"
echo "[prompt-profile-length-mech] HF_HOME=${HF_HOME}"
echo "[prompt-profile-length-mech] VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"

for dataset in "${DATASETS[@]}"; do
  archive_manifest="${REGRESSION_ROOT}/${dataset}/shared_archive/manifest.json"
  if [[ ! -f "${archive_manifest}" ]]; then
    echo "[prompt-profile-length-mech] missing manifest: ${archive_manifest}" >&2
    exit 1
  fi
done

AUDIT_CMD=(
  "${PYTHON_BIN}" scripts/analyze_prompt_profile_length_mechanism.py
  --regression-root "${REGRESSION_ROOT}"
  --out-dir "${OUT_ROOT}"
  --selection-kind "${SELECTION_KIND}"
  --batch-size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --datasets
)

for dataset in "${DATASETS[@]}"; do
  AUDIT_CMD+=("${dataset}")
done

printf '$'
for arg in "${AUDIT_CMD[@]}"; do
  printf ' %q' "${arg}"
done
printf '\n'

"${AUDIT_CMD[@]}"
