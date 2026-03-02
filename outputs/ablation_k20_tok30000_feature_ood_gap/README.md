# k20/tok30000 MLP Feature OOD Gap Ablation

## Setup
- Model preset: `openthinker3_1p5b`
- Loop labeling: `n=30`, `k=20`
- Decode horizon: `max_tokens=30000`
- Probe: `mlp`
- Train split size: `256`
- Eval size per run: `60`
- Feature variants:
  - `last_token_final` (`pooling=last_token`, `layer=-1`)
  - `mean_pool_final` (`pooling=mean_pool`, `layer=-1`)
  - `last_token_layer14` (`pooling=last_token`, `layer=14`)
- Split regimes:
  - `ood`: train `MATH-500:test` vs test `AIME`
  - `id`: disjoint train/test subsets from `MATH-500:test` (`256/60`)

## ROC-AUC Summary
- `last_token_final`: OOD `0.6521` vs ID `0.5370` (ID-OOD `-0.1151`)
- `mean_pool_final`: OOD `0.4603` vs ID `0.7818` (ID-OOD `+0.3215`)
- `last_token_layer14`: OOD `0.6105` vs ID `0.7310` (ID-OOD `+0.1205`)

Raw tables:
- `summary_rows.csv`
- `summary_ood_gap.csv`
- `summary.json`

## Caveat
- Label prevalence differs across runs because each dataset build regenerated rollouts:
  - OOD test positives: `18-21/60`
  - ID test positives: `3-6/60`
- The very low positive count on ID test splits increases variance in ROC-AUC estimates.
