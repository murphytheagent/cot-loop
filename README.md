# CoT Loop Detection via Probe Classifiers

This repository trains probe classifiers to predict whether a language model will enter a repetitive loop during chain-of-thought reasoning, based solely on prefill activations.

## Overview

The core hypothesis: Can we detect whether a model will loop **before generation begins**, using only the hidden states from the prefill pass?

**Workflow:**
1. Build model-formatted chat prompts (shared `utils.build_prompt` source)
2. Extract pooled prefill activations from those prompts (configurable pooling + layer)
3. Generate rollout trajectories and label them (looped vs not-looped)
4. Train a binary probe classifier on the prefill features
5. Evaluate the probe's ability to predict looping behavior

## Quick Start

### Installation

Requires Python >= 3.10.

```bash
uv sync
```

### Environment Setup

Create a `.env` file with your W&B API key:

```bash
echo "WANDB_API_KEY=your_key_here" > .env
```

### Build Probe Dataset

Extract features and labels from train/test datasets:

```bash
python scripts/build_probe_dataset.py \
  --train-dataset my_org/my_dataset \
  --train-split train \
  --test-dataset my_org/my_dataset \
  --test-split test \
  --prompt-field prompt \
  --model-preset qwq_32b \
  --out-dir outputs/probe_data
```

If `--test-dataset` is omitted, it defaults to local `data/aime_2024_2025.jsonl`:

```bash
python scripts/build_probe_dataset.py \
  --train-dataset my_org/my_dataset \
  --train-split train \
  --prompt-field prompt \
  --model-preset openthinker3_7b \
  --out-dir outputs/probe_data
```

For this default local test file, loader behavior is hardcoded to use
`question` as prompt text and require `answer` on every row.

Optional: if you want a random split of one dataset, pass identical train/test specs
and use `--split-ratio`.

### Train Probe

```bash
python scripts/train_probe.py \
  --data-dir outputs/probe_data \
  --out-dir outputs/probe_runs/run1 \
  --probe-preset linear \
  --wandb-project cot-loop-probe \
  --epochs 10 \
  --batch-size 256
```

Available probe presets:
- `linear` (default)
- `mlp` (one hidden layer; shape is defined in `src/loop_probe/configs.py`)

### SLURM End-to-End Probe Job

Submit one job that builds the probe dataset and trains the probe:

```bash
sbatch slurm/run_probe_train_e2e.sbatch
```

Default dataset/model settings in this job:
- `MODEL_PRESET=openthinker3_1p5b`
- `#SBATCH --gres=gpu:8` (job requests 8 GPUs by default)
- rollout `tp/dp` comes from `src/loop_probe/configs.py` preset defaults
- optional rollout concurrency override: `MAX_NUM_SEQS=...`
- `TRAIN_DATASET=HuggingFaceH4/MATH-500`, `TRAIN_SPLIT=test`
- `TEST_DATASET` omitted (defaults to local `data/aime_2024_2025.jsonl` in `build_probe_dataset.py`)
- `TEST_SPLIT=test`
- `PROMPT_FIELD=problem`
- `PROBE_PRESET=mlp`

Prompt formatting note:
- Probe dataset build and generation/eval scripts share the same chat prompt constructor: `scripts/utils.py:build_prompt()`.

## Model Presets

Predefined configurations for common models (see `src/loop_probe/configs.py`):

| Preset | Model | TP | DP | Temperature | Max Tokens |
|--------|-------|----|----|-------------|------------|
| `qwq_32b` | Qwen/QwQ-32B | 8 | 1 | 0.0 | 30000 |
| `openthinker3_7b` | open-thoughts/OpenThinker3-7B | 1 | 8 | 0.0 | 30000 |
| `openthinker3_1p5b` | open-thoughts/OpenThinker3-1.5B | 1 | 8 | 0.0 | 30000 |

Override any preset field via CLI:

```bash
--model-preset qwq_32b --temperature 0.3 --max-tokens 20000
```

Or skip presets entirely:

```bash
--model-id Qwen/QwQ-32B --tp 4 --temperature 0.0 --max-tokens 30000
```

## Loop Detection

A sequence is labeled as "looped" if any 30-gram appears ≥20 times in the generated token IDs. These defaults (`--loop-n 30`, `--loop-k 20`) can be adjusted during dataset building.

## Outputs

**Dataset build:**
- `{out_dir}/train/shard-*.pt` - Training shards (features + labels)
- `{out_dir}/test/shard-*.pt` - Test shards
- `{out_dir}/manifest.json` - Dataset metadata and configuration

**Training:**
- `{out_dir}/best.pt` - Best checkpoint (by ROC-AUC, then macro-F1)
- `{out_dir}/last.pt` - Final epoch checkpoint
- `{out_dir}/metrics.jsonl` - Per-epoch evaluation metrics
- `{out_dir}/best_metrics.json` - Best eval row summary for this run

**Multi-seed SLURM summary:**
- `${OUT_RUN_DIR}/seed_summary.json` - Per-seed best rows + aggregate mean/std
- `${OUT_RUN_DIR}/seed_summary.csv` - Aggregate mean/std table

**Multi-seed E2E training (`slurm/run_probe_train_e2e.sbatch`):**
- `{out_run_dir}/seed_*/metrics.jsonl` - Per-seed train/eval metrics
- `{out_run_dir}/probe_multiseed_curves.png` - Aggregated train/eval curves across seeds

Manual plotting command (if needed):
```bash
python scripts/plot_probe_multiseed.py \
  --run-dir outputs/probe_runs/<run_name> \
  --out outputs/probe_runs/<run_name>/probe_multiseed_curves.png
```

## Repository Structure

```
cot-loop/
├── src/loop_probe/          # Core probe training library
│   ├── configs.py           # Model presets and configuration
│   ├── hf_data.py           # Hugging Face dataset loading
│   ├── prefill.py           # Prefill activation extraction
│   ├── rollout.py           # vLLM trajectory generation
│   ├── labeling.py          # Loop detection and labeling
│   ├── dataloader.py        # PyTorch dataset/dataloader
│   ├── probes/              # Probe architectures
│   └── train_utils.py       # Training utilities
├── scripts/
│   ├── build_probe_dataset.py  # Extract features & labels
│   ├── train_probe.py          # Train probe classifier
│   ├── aggregate_probe_runs.py # Multi-seed mean/std summary
│   └── [loop analysis scripts]
├── slurm/                   # SLURM batch scripts
├── data/                    # Input datasets
└── outputs/                 # Generated artifacts
```

## Advanced Usage

### Custom Model Configuration

```bash
python scripts/build_probe_dataset.py \
  --train-dataset my_org/prompts \
  --prompt-field text \
  --model-id meta-llama/Llama-3.1-8B \
  --tp 2 \
  --dtype float16 \
  --max-model-len 16384 \
  --temperature 0.0 \
  --max-tokens 10000 \
  --out-dir outputs/llama_probe
```

### Training Hyperparameters

```bash
python scripts/train_probe.py \
  --data-dir outputs/probe_data \
  --out-dir outputs/probe_runs/run1 \
  --wandb-project my-project \
  --probe-preset mlp \
  --epochs 20 \
  --batch-size 512 \
  --lr 1e-3 \
  --weight-decay 0.01 \
  --eval-every 2
```

### Class Imbalance

The training script automatically applies `pos_weight` in BCEWithLogitsLoss based on the train split's positive/negative ratio.

## Additional Scripts

For rollout-loop analysis on existing generations, see the `scripts/` directory.  
The Figure 1 plotting and generation scripts remain for archival comparisons and are not the primary detector training path.

## Technical Details

See [src/loop_probe/README.md](src/loop_probe/README.md) for detailed module documentation.

## Notes

- Prefill feature extraction uses Transformers and loads the full model on GPU
- Rollout generation uses vLLM for efficient batched inference
- Training requires `WANDB_API_KEY` in `.env` or environment
- GPU memory requirements depend on model size and batch size
