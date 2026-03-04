#!/bin/bash
#SBATCH --job-name=probe-k5-threeview-data
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs outputs

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
elif [[ -n "${CONDA_ENV:-}" ]]; then
  CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[k5-threeview-data] conda activation script not found: ${CONDA_SH}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
else
  echo "[k5-threeview-data] no runtime env found (.venv missing and CONDA_ENV not set)" >&2
  exit 1
fi

if [[ -f "${HOME}/.env" ]]; then
  # shellcheck disable=SC1091
  set -a
  source "${HOME}/.env"
  set +a
fi

if [[ -d "/data/users/${USER}" ]]; then
  DEFAULT_CACHE_BASE="/data/users/${USER}/cache"
elif [[ -d "/data/scratch" ]]; then
  DEFAULT_CACHE_BASE="/data/scratch/${USER}/cache"
else
  echo "[k5-threeview-data] no supported non-home cache root found (/data/users or /data/scratch)" >&2
  exit 1
fi
mkdir -p "${DEFAULT_CACHE_BASE}"

export HF_HOME="${HF_HOME:-${DEFAULT_CACHE_BASE}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${DEFAULT_CACHE_BASE}/vllm}"
export TORCH_HOME="${TORCH_HOME:-${DEFAULT_CACHE_BASE}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${DEFAULT_CACHE_BASE}}"

canonicalize_path() {
  python - "$1" <<'PY'
import os
import sys

path = sys.argv[1]
print(os.path.realpath(os.path.abspath(os.path.expanduser(path))))
PY
}

HOME_REALPATH="$(canonicalize_path "${HOME}")"

for cache_var in HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE VLLM_CACHE_ROOT TORCH_HOME XDG_CACHE_HOME; do
  cache_val="${!cache_var}"
  cache_realpath="$(canonicalize_path "${cache_val}")"
  if [[ "${cache_realpath}" == "${HOME_REALPATH}" || "${cache_realpath}" == "${HOME_REALPATH}/"* ]]; then
    echo "[k5-threeview-data] ${cache_var} resolves under HOME (${cache_val} -> ${cache_realpath}); refusing to run" >&2
    exit 1
  fi
done

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${VLLM_CACHE_ROOT}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

echo "[k5-threeview-data] HF_HOME=${HF_HOME}"
echo "[k5-threeview-data] HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}"
echo "[k5-threeview-data] TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"
echo "[k5-threeview-data] VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"

DATA_ROOT="/data/scratch/${USER}/datasets/cot-loop-detection"
if [[ -d "/data/users/${USER}" ]]; then
  DATA_ROOT="/data/users/${USER}/datasets/cot-loop-detection"
fi
mkdir -p "${DATA_ROOT}"

TRAIN_POOL_JSONL="${DATA_ROOT}/k5_multisource_train_pool_v2.jsonl"
TRAIN_POOL_META="${DATA_ROOT}/k5_multisource_train_pool_v2.meta.json"
EVAL_POOL_JSONL="${DATA_ROOT}/k5_eval_math500_aime2425_v1.jsonl"
EVAL_POOL_META="${DATA_ROOT}/k5_eval_math500_aime2425_v1.meta.json"

OUT_DATA_DIR="${OUT_DATA_DIR:-outputs/probe_data/k5_tok15000_three_view_train5000_eval_math500_aime}"
OUT_SUMMARY_JSON="${OUT_SUMMARY_JSON:-outputs/probe_data/k5_tok15000_three_view_train5000_eval_math500_aime_summary.json}"

export TRAIN_POOL_JSONL TRAIN_POOL_META EVAL_POOL_JSONL EVAL_POOL_META OUT_DATA_DIR OUT_SUMMARY_JSON

python - <<'PY'
import json
import os
import random
from collections import Counter

from datasets import load_dataset

train_pool_jsonl = os.environ["TRAIN_POOL_JSONL"]
train_pool_meta = os.environ["TRAIN_POOL_META"]

aime_path = os.path.join("data", "aime_2024_2025.jsonl")
eval_pool_jsonl = os.environ["EVAL_POOL_JSONL"]
eval_pool_meta = os.environ["EVAL_POOL_META"]

os.makedirs(os.path.dirname(train_pool_jsonl), exist_ok=True)

sources = [
    {
        "name": "SuperSecureHuman/competition_math_hf_dataset",
        "config": None,
        "split": "train",
        "prompt_field": "problem",
        "take": 7500,
    },
    {
        "name": "open-r1/OpenR1-Math-220k",
        "config": None,
        "split": "train",
        "prompt_field": "problem",
        "take": 16000,
    },
    {
        "name": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "prompt_field": "question",
        "take": 6000,
    },
]

rows = []
for spec in sources:
    split_expr = f"{spec['split']}[:{spec['take']}]"
    ds = load_dataset(spec["name"], spec["config"], split=split_expr)
    field = spec["prompt_field"]
    if field not in ds.column_names:
        raise SystemExit(
            f"{spec['name']} missing field '{field}' in columns {ds.column_names}"
        )

    loaded = 0
    for ex in ds:
        prompt = ex.get(field)
        if prompt is None:
            continue
        prompt = str(prompt).strip()
        if not prompt:
            continue
        rows.append(
            {
                "problem": prompt,
                "source": spec["name"],
                "source_split": spec["split"],
            }
        )
        loaded += 1
    print(f"loaded {loaded} prompts from {spec['name']} ({split_expr})")

random.Random(0).shuffle(rows)
seen = set()
deduped = []
for row in rows:
    key = row["problem"]
    if key in seen:
        continue
    seen.add(key)
    deduped.append(row)

with open(train_pool_jsonl, "w", encoding="utf-8") as f:
    for row in deduped:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")

train_meta = {
    "path": train_pool_jsonl,
    "seed": 0,
    "num_rows": len(deduped),
    "sources": sources,
    "source_counts_after_dedup": dict(Counter(row["source"] for row in deduped)),
}
with open(train_pool_meta, "w", encoding="utf-8") as f:
    json.dump(train_meta, f, indent=2)

print(f"wrote train pool: {train_pool_jsonl} rows={len(deduped)}")

# Eval pool: natural distribution = MATH-500 test + AIME24/25 local file.
eval_rows = []
math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
for ex in math_ds:
    prompt = str(ex.get("problem", "")).strip()
    if not prompt:
        continue
    eval_rows.append(
        {
            "problem": prompt,
            "source": "HuggingFaceH4/MATH-500",
            "source_split": "test",
        }
    )

if not os.path.isfile(aime_path):
    raise SystemExit(f"Missing local AIME file: {aime_path}")
with open(aime_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        prompt = str(row.get("question", "")).strip()
        if not prompt:
            continue
        eval_rows.append(
            {
                "problem": prompt,
                "source": "local/aime_2024_2025",
                "source_split": "test",
            }
        )

with open(eval_pool_jsonl, "w", encoding="utf-8") as f:
    for row in eval_rows:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")

eval_meta = {
    "path": eval_pool_jsonl,
    "num_rows": len(eval_rows),
    "source_counts": dict(Counter(row["source"] for row in eval_rows)),
}
with open(eval_pool_meta, "w", encoding="utf-8") as f:
    json.dump(eval_meta, f, indent=2)

print(f"wrote eval pool: {eval_pool_jsonl} rows={len(eval_rows)}")
PY

python scripts/build_probe_dataset.py \
  --train-dataset "${TRAIN_POOL_JSONL}" \
  --train-split train \
  --train-max-samples 5000 \
  --test-dataset "${EVAL_POOL_JSONL}" \
  --test-split train \
  --prompt-field problem \
  --seed 0 \
  --model-preset openthinker3_1p5b \
  --dp 8 \
  --max-tokens 15000 \
  --max-num-seqs 4 \
  --loop-n 30 \
  --loop-k 5 \
  --prefill-batch-size 16 \
  --completion-batch-size 1 \
  --feature-key rollout_lasttok_layers_mean \
  --feature-pooling rollout_last_token_all_layers_mean \
  --feature-layer -1 \
  --extra-feature-view prefill_lasttok_layers_mean:last_token_all_layers_mean:-1 \
  --extra-feature-view prefill_lasttok_layers_concat:last_token_all_layers_concat:-1 \
  --balance-train downsample \
  --balance-test none \
  --balance-seed 0 \
  --reuse-if-compatible \
  --out-dir "${OUT_DATA_DIR}"

python - <<'PY'
import json
import os

out_data_dir = os.environ["OUT_DATA_DIR"]
out_summary_json = os.environ["OUT_SUMMARY_JSON"]
manifest_path = os.path.join(out_data_dir, "manifest.json")

if not os.path.exists(manifest_path):
    raise SystemExit(f"Missing manifest: {manifest_path}")

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

summary = {
    "out_data_dir": out_data_dir,
    "train_pool_jsonl": os.environ["TRAIN_POOL_JSONL"],
    "train_pool_meta": os.environ["TRAIN_POOL_META"],
    "eval_pool_jsonl": os.environ["EVAL_POOL_JSONL"],
    "eval_pool_meta": os.environ["EVAL_POOL_META"],
    "default_feature_key": manifest.get("default_feature_key"),
    "feature_keys": sorted((manifest.get("feature_views") or {}).keys()),
    "train": manifest.get("train"),
    "test": manifest.get("test"),
    "balancing": manifest.get("balancing"),
    "loop_detector": manifest.get("loop_detector"),
    "rollout_config": manifest.get("rollout_config"),
}

os.makedirs(os.path.dirname(out_summary_json), exist_ok=True)
with open(out_summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"wrote summary: {out_summary_json}")
PY

echo "[k5-threeview-data] done"
