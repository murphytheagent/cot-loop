# CoT Loop Detection via Probe Classifiers

This repository trains probe classifiers to predict whether a language model will enter a repetitive loop during chain-of-thought reasoning from configurable activation views. The current workflow supports both prompt-prefill features and rollout-completion features, including shared-dataset multi-view experiments.

## Overview

The core question is no longer limited to prefill-only detection. The current experiments compare whether loop risk is most detectable from prompt-prefill activations, rollout-completion activations, or a combination of both.

Latest status:
- Round G (k=5, three-view) found the completion-view feature set outperforming the tested prefill variants.

**Workflow:**
1. Build model-formatted chat prompts (shared `utils.build_prompt` source)
2. Extract configurable activation views from prompt-prefill states and/or rollout-completion states (one or more configurable pooling/layer views)
3. Generate rollout trajectories and label them (looped vs not-looped)
4. Train a binary probe classifier on the precomputed features
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

To build one shared rollout-label dataset that can be reused by both
`last_token_final` and `mean_pool_final`:

```bash
python scripts/build_probe_dataset.py \
  --train-dataset HuggingFaceH4/MATH-500 \
  --train-split test \
  --test-dataset data/aime_2024_2025.jsonl \
  --test-split test \
  --prompt-field problem \
  --model-preset openthinker3_1p5b \
  --feature-key last_token_final \
  --feature-pooling last_token \
  --feature-layer -1 \
  --extra-feature-view mean_pool_final:mean_pool:-1 \
  --out-dir outputs/probe_data/openthinker3_1p5b_shared_final_views
```

To build a three-view dataset with all-layer prefill features plus a
rollout-completion feature in one rollout-label pass:

```bash
python scripts/build_probe_dataset.py \
  --train-dataset my_org/train_pool \
  --train-split train \
  --train-max-samples 5000 \
  --test-dataset my_org/eval_pool \
  --test-split test \
  --prompt-field problem \
  --model-preset openthinker3_1p5b \
  --max-tokens 15000 \
  --loop-k 5 \
  --feature-key rollout_lasttok_layers_mean \
  --feature-pooling rollout_last_token_all_layers_mean \
  --feature-layer -1 \
  --extra-feature-view prefill_lasttok_layers_mean:last_token_all_layers_mean:-1 \
  --extra-feature-view prefill_lasttok_layers_concat:last_token_all_layers_concat:-1 \
  --balance-train downsample \
  --balance-test none \
  --completion-batch-size 1 \
  --out-dir outputs/probe_data/openthinker3_three_view_k5
```

For balanced train/test probes after label construction:

```bash
python scripts/build_probe_dataset.py \
  --train-dataset SuperSecureHuman/competition_math_hf_dataset \
  --train-split train \
  --train-max-samples 1800 \
  --test-dataset SuperSecureHuman/competition_math_hf_dataset \
  --test-split test \
  --test-max-samples 600 \
  --prompt-field problem \
  --model-preset openthinker3_1p5b \
  --feature-key last_token_final \
  --extra-feature-view mean_pool_final:mean_pool:-1 \
  --balance-train downsample \
  --balance-test downsample \
  --out-dir outputs/probe_data/openthinker3_balanced_shared_final_views
```

### Train Probe

```bash
python scripts/train_probe.py \
  --data-dir outputs/probe_data \
  --out-dir outputs/probe_runs/run1 \
  --probe-preset linear \
  --feature-key mean_pool_final \
  --wandb-project cot-loop-probe \
  --epochs 10 \
  --batch-size 256
```

`--feature-key` is optional. If omitted, the manifest default view is used.

Available probe presets:
- `linear` (default)
- `mlp` (configurable width/depth; defaults are in `src/loop_probe/configs.py`)

Optional MLP overrides:
- `--mlp-hidden-dim <int>`
- `--mlp-depth <int>`
- `--mlp-dropout <float>`

### Train RFM-lite Probe

```bash
python scripts/train_rfm_probe.py \
  --data-dir outputs/probe_data/openthinker3_balanced_shared_final_views \
  --feature-key last_token_final \
  --out-dir outputs/probe_runs/rfm_last_token/seed_0 \
  --seed 0 \
  --random-features 2048 \
  --bandwidth 1.0 \
  --ridge 0.1 \
  --rfm-steps 1 \
  --grad-weight 1.0
```

### Evaluate Saved Checkpoints on Another Split/Dataset

Torch probe checkpoint (`best.pt` / `last.pt`):

```bash
python scripts/eval_probe_checkpoint.py \
  --checkpoint outputs/probe_runs/run1/best.pt \
  --data-dir outputs/probe_data/other_eval_dataset \
  --feature-key last_token_final \
  --split test
```

RFM-lite checkpoint (`best_model.pt`):

```bash
python scripts/eval_rfm_checkpoint.py \
  --checkpoint outputs/probe_runs/rfm_last_token/seed_0/best_model.pt \
  --data-dir outputs/probe_data/other_eval_dataset \
  --feature-key last_token_final \
  --split test
```

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
- optional completion-feature extraction throughput override: `COMPLETION_BATCH_SIZE=...` (default: `1`)
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
- `{out_dir}/features/<feature_key>/{train,test}/shard-*.pt` - Additional feature views (when `--extra-feature-view` is used)

**Training:**
- `{out_dir}/best.pt` - Best checkpoint (by ROC-AUC, then macro-F1)
- `{out_dir}/last.pt` - Final epoch checkpoint
- `{out_dir}/metrics.jsonl` - Per-epoch evaluation metrics
- `{out_dir}/best_metrics.json` - Best eval row summary for this run

**RFM-lite training (`scripts/train_rfm_probe.py`):**
- `{out_dir}/best_model.pt` - Serialized RFM-lite pipeline checkpoint
- `{out_dir}/model_summary.json` - RFM-lite hyperparameters and selected step
- `{out_dir}/metrics.jsonl` - Per-step train/eval metrics
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
│   ├── train_rfm_probe.py      # Train RFM-lite probe
│   ├── eval_probe_checkpoint.py # Evaluate torch probe checkpoints
│   ├── eval_rfm_checkpoint.py  # Evaluate RFM-lite checkpoints
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
