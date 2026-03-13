#!/bin/bash
#SBATCH --job-name=probe-k5-threeview-ablate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs outputs

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
elif [[ -n "${CONDA_ENV:-}" ]]; then
  CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[k5-threeview-ablate] conda activation script not found: ${CONDA_SH}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
else
  echo "[k5-threeview-ablate] no runtime env found (.venv missing and CONDA_ENV not set)" >&2
  exit 1
fi

if [[ -f "${HOME}/.env" ]]; then
  # shellcheck disable=SC1091
  set -a
  source "${HOME}/.env"
  set +a
fi

DATA_DIR="${DATA_DIR:-outputs/probe_data/k5_tok15000_three_view_train5000_eval_math500_aime}"
RUN_ROOT="${RUN_ROOT:-outputs/probe_runs/k5_tok15000_threeview_ablation}"
SUMMARY_DIR="${SUMMARY_DIR:-${RUN_ROOT}/summary}"
WANDB_PROJECT="${WANDB_PROJECT:-cot-loop-probe}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-15}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_EVERY="${LOG_EVERY:-10}"

export DATA_DIR RUN_ROOT SUMMARY_DIR

mkdir -p "${RUN_ROOT}" "${SUMMARY_DIR}"

if [[ ! -f "${DATA_DIR}/manifest.json" ]]; then
  echo "[k5-threeview-ablate] missing dataset manifest: ${DATA_DIR}/manifest.json" >&2
  exit 1
fi

echo "[k5-threeview-ablate] data_dir=${DATA_DIR}"
echo "[k5-threeview-ablate] run_root=${RUN_ROOT}"
echo "[k5-threeview-ablate] seeds=${SEEDS}"

run_config() {
  local view_key="$1"
  local run_tag="$2"
  local hidden_dim="$3"
  local depth="$4"
  local lr="$5"
  local weight_decay="$6"
  local dropout="$7"
  local batch_size="$8"

  local run_base="${RUN_ROOT}/${run_tag}"
  mkdir -p "${run_base}"

  local run_dirs=()
  for seed in ${SEEDS}; do
    local run_dir="${run_base}/seed_${seed}"
    run_dirs+=("${run_dir}")
    local run_name="${run_tag}-seed${seed}"

    echo "[k5-threeview-ablate] train ${run_name} view=${view_key} hidden=${hidden_dim} depth=${depth}"
    python scripts/train_probe.py \
      --data-dir "${DATA_DIR}" \
      --out-dir "${run_dir}" \
      --probe-preset mlp \
      --feature-key "${view_key}" \
      --mlp-hidden-dim "${hidden_dim}" \
      --mlp-depth "${depth}" \
      --mlp-dropout "${dropout}" \
      --epochs "${EPOCHS}" \
      --batch-size "${batch_size}" \
      --lr "${lr}" \
      --weight-decay "${weight_decay}" \
      --num-workers "${NUM_WORKERS}" \
      --device auto \
      --seed "${seed}" \
      --eval-every 1 \
      --log-every "${LOG_EVERY}" \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-run-name "${run_name}"
  done

  python scripts/aggregate_probe_runs.py \
    --run-dirs "${run_dirs[@]}" \
    --selection-metric macro_f1 \
    --tie-breaker positive_f1 \
    --out-json "${run_base}/seed_summary.json" \
    --out-csv "${run_base}/seed_summary.csv"
}

# Preemptive study: posterior (rollout completion) first.
run_config "rollout_lasttok_layers_mean" "completion_mean_h128_d1" 128 1 1e-4 0.05 0.1 256
run_config "rollout_lasttok_layers_mean" "completion_mean_h256_d2" 256 2 8e-5 0.03 0.1 256

# Prefill ablations (aggregated vs disaggregated all layers).
run_config "prefill_lasttok_layers_mean" "prefill_layers_mean_h256_d2" 256 2 1e-4 0.05 0.1 256
run_config "prefill_lasttok_layers_concat" "prefill_layers_concat_h512_d2" 512 2 1e-4 0.03 0.1 64

python - <<'PY'
import csv
import json
import math
import os

run_root = os.environ["RUN_ROOT"]
summary_dir = os.environ["SUMMARY_DIR"]
os.makedirs(summary_dir, exist_ok=True)

rows = []
for run_tag in sorted(os.listdir(run_root)):
    run_dir = os.path.join(run_root, run_tag)
    if not os.path.isdir(run_dir):
        continue
    summary_json = os.path.join(run_dir, "seed_summary.json")
    if not os.path.isfile(summary_json):
        continue
    with open(summary_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    aggregate = payload.get("aggregate", {})
    if isinstance(aggregate, dict) and isinstance(aggregate.get("metrics"), dict):
        # Backward compatibility for any old nested shape.
        metrics = aggregate.get("metrics", {})
    elif isinstance(aggregate, dict):
        metrics = aggregate
    else:
        metrics = {}
    rows.append(
        {
            "run_tag": run_tag,
            "macro_f1_mean": metrics.get("macro_f1", {}).get("mean"),
            "macro_f1_std": metrics.get("macro_f1", {}).get("std"),
            "positive_precision_mean": metrics.get("positive_precision", {}).get("mean"),
            "positive_recall_mean": metrics.get("positive_recall", {}).get("mean"),
            "positive_f1_mean": metrics.get("positive_f1", {}).get("mean"),
            "pr_auc_mean": metrics.get("pr_auc", {}).get("mean"),
            "roc_auc_mean": metrics.get("roc_auc", {}).get("mean"),
            "accuracy_mean": metrics.get("accuracy", {}).get("mean"),
        }
    )

def macro_f1_rank_key(row):
    value = row.get("macro_f1_mean")
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return (1, float("inf"))
    if math.isnan(value_f):
        return (1, float("inf"))
    return (0, -value_f)

rows.sort(key=macro_f1_rank_key)

out_json = os.path.join(summary_dir, "ablation_ranked_summary.json")
out_csv = os.path.join(summary_dir, "ablation_ranked_summary.csv")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)

if rows:
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

print(f"wrote {out_json}")
print(f"wrote {out_csv}")
PY

echo "[k5-threeview-ablate] done"
