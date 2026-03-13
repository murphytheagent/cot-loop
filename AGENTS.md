# Repository Guidelines

## Project Focus
This repository builds a chain-of-thought (CoT) loop detector from configurable activation views. The current focus is a mixed-view workflow: completion features remain the strongest overall arm, while the 2026-03-13 prefill follow-up is stress-testing metadata-aware prefill residual signal and anchor-plus-boundary augmentations on the active PR branch.

## Project Structure & Module Organization
- `src/loop_probe/`: Core library for prompt loading, prefill extraction, rollout generation, loop labeling, probe architectures, and training utilities.
- `scripts/`: CLI entry points for data/building, probe training, analysis, and plotting.
- `data/`: Local datasets and documentation. `data/README.md` defines the expected JSONL schema for non-HF local files.
- `slurm/`: SLURM batch scripts for probe pipeline, generation, and prefill-stability experiments.
- `outputs/`: Generated artifacts (prefill shards, checkpoints, metrics CSVs, figures).
- `pyproject.toml` + `uv.lock`: Dependency definitions (Python >= 3.10).

## Build, Test, and Development Commands
- Install dependencies: `uv sync`.
- Build probe dataset shards:
  `python scripts/build_probe_dataset.py --train-dataset <dataset-or-jsonl> --train-split <split> --prompt-field <field> --model-preset openthinker3_1p5b --out-dir outputs/probe_data`.
- Train probe:
  `python scripts/train_probe.py --data-dir outputs/probe_data --out-dir outputs/probe_runs/run1 --probe-preset linear --wandb-project cot-loop-probe`.
- Run probe end-to-end on SLURM:
  `sbatch slurm/run_probe_train_e2e.sbatch`.
- Generate rollout data for labeling/analysis (optional):
  `python scripts/run_vllm_generate.py --model-id Qwen/QwQ-32B --data data/aime_2024_2025.jsonl --metrics-out outputs/qwq32b_metrics.csv --tp 8`.
- Run on SLURM for generation:
  `sbatch slurm/run_vllm_generate.sbatch`.
- Summarize loop metrics from generations:
  `python scripts/compute_metrics.py --generations path/to.jsonl --out outputs/metrics.csv`.
- Run plotting scripts as needed:
  `python scripts/plot_accuracy_vs_temperature.py --metrics outputs/*_metrics.csv --out outputs/accuracy_plot.png`.

## Coding Style & Naming Conventions
- Python-only codebase; use 4-space indentation and keep functions small and single-purpose.
- Prefer explicit CLI args and type hints similar to existing scripts.
- Outputs keep detector naming: checkpoints live in run folders, metrics in `_metrics.csv`, and figures in `outputs/`.
- No formatter/linter is configured; follow PEP 8 conventions for consistency.

## Testing Guidelines
- No automated test suite is present.
- Use small-scale local checks for smoke validation (small `--train-max-samples`, `--test-max-samples`, and short rollout limits in generation scripts) before long jobs.
- Validate local data schema with `data/README.md` and `scripts/run_vllm_generate.py`/`scripts/build_probe_dataset.py` behavior.

## Commit & Pull Request Guidelines
- Git history uses short, lowercase, imperative-style messages (e.g., `added dp`, `fixed probe batching`). Keep messages concise.
- PRs should include a brief purpose, exact reproduction command(s), and expected artifacts (checkpoints/metrics/plots).
