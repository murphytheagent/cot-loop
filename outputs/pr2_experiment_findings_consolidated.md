# CoT Loop Detection PR #2: Consolidated Experiments and Findings

Last updated: 2026-03-05 17:18 UTC
Scope covered: 2026-03-01 19:00 UTC through 2026-03-05 17:18 UTC
PR: https://github.com/Zhi0467/cot-loop/pull/2

## Goal
Predict whether a (model, prompt) pair will enter a loop (n=30-gram repeating >= k=20) using prefill activations before generation.

## Experiment Inventory

### A) Initial architecture/threshold matrix (short horizon, exploratory)
- Runs: `635-638` (k=10/k=20 x MLP/Linear), `max_tokens=2048`.
- Key issue found: k=20 test split had `0/60` positives, so ROC-AUC for k=20 was undefined and not interpretable.
- Outcome: Treat as diagnostic only; do not use for model-selection conclusions.

### B) Corrective matched-horizon controls (same setup, `max_tokens=30000`)
- Runs: `639-640` (k=20, MLP/Linear), `642-643` (k=10, MLP/Linear).
- k=20 labels recovered to `14/60` positives; k=10 had `23/60` positives.
- ROC-AUC:
  - k=20: MLP `0.6599` vs Linear `0.4410`
  - k=10: MLP `0.6052` vs Linear `0.5088`
- Finding: after fixing decode horizon, MLP consistently outperformed Linear on AUC.

### C) Feature OOD-gap ablation on locked setup (`k=20`, `max_tokens=30000`, MLP)
- Runs: `656-661`.
- Feature variants:
  - `last_token_final`
  - `mean_pool_final`
  - `last_token_layer14` (later deprioritized)
- ROC-AUC summary:
  - `last_token_final`: OOD `0.6521`, ID `0.5370`, ID-OOD `-0.1151`
  - `mean_pool_final`: OOD `0.4603`, ID `0.7818`, ID-OOD `+0.3215`
  - `last_token_layer14`: OOD `0.6105`, ID `0.7310`, ID-OOD `+0.1205`
- Important caveat: each feature run rebuilt labels independently; prevalence drifted (OOD positives `18-21/60`, ID positives `3-6/60`).

### D) Metric audit + matched-label dual-view comparison (focus narrowed to final-layer views)
- Scope: `last_token_final` vs `mean_pool_final`, class-imbalance-aware reporting.
- Pipeline updates:
  - Added PR-AUC, positive precision/recall/F1, prevalence logging.
  - Added shared multi-view dataset build so one rollout-label pass serves both feature views.
- Matched-label run: `722` (shared labels for both views, prevalence `17/60`).
- Multi-seed (`n=3`) means +/- std:
  - `last_token_final`: ROC-AUC `0.5723 +/- 0.0909`, PR-AUC `0.3684 +/- 0.0803`, macro-F1 `0.3866 +/- 0.0621`
  - `mean_pool_final`: ROC-AUC `0.5695 +/- 0.0553`, PR-AUC `0.3892 +/- 0.0701`, macro-F1 `0.3176 +/- 0.1369`
- Finding: ROC-AUC is effectively tied on identical labels; `last_token_final` is more stable and better on macro-F1, while `mean_pool_final` was slightly higher on PR-AUC.

### E) Data-first scope correction + official RFM comparison
- Completed in the same PR2 branch with ID-balanced and OOD-natural evaluations.
- ID balanced dataset: `22/44` positives.
- OOD dataset: `10/200` positives (prevalence `0.05`; majority baseline accuracy `0.95`).
- Multi-seed (`n=3`) summary:
  - ID (`22/44` positives):
    - MLP + `last_token_final`: ROC-AUC `0.6550 +/- 0.1643`, PR-AUC `0.6860 +/- 0.1511`
    - MLP + `mean_pool_final`: ROC-AUC `0.6453 +/- 0.0969`, PR-AUC `0.6974 +/- 0.1077`
    - official-RFM + `last_token_final`: ROC-AUC `0.5124 +/- 0.0000`, PR-AUC `0.5509 +/- 0.0000`
    - official-RFM + `mean_pool_final`: ROC-AUC `0.3623 +/- 0.0012`, PR-AUC `0.4238 +/- 0.0005`
  - OOD (`10/200` positives):
    - MLP + `last_token_final`: ROC-AUC `0.5933 +/- 0.0301`, PR-AUC `0.0842 +/- 0.0107`
    - MLP + `mean_pool_final`: ROC-AUC `0.4747 +/- 0.0267`, PR-AUC `0.0598 +/- 0.0045`
    - official-RFM + `last_token_final`: ROC-AUC `0.6270 +/- 0.0003`, PR-AUC `0.0936 +/- 0.0001`
    - official-RFM + `mean_pool_final`: ROC-AUC `0.4474 +/- 0.0000`, PR-AUC `0.0564 +/- 0.0006`
- Finding: under current label/data regime, OOD detection remains weak overall; MLP + `last_token_final` remains the strongest practical arm for continuing ablations.

### F) Larger multi-source rebuild (in progress, then explicitly stopped)
- Run: `743`.
- Built/deduplicated multi-source prompt pool (`29,482` prompts), started large rebuild, then canceled due explicit human `!stop` at 2026-03-03 22:41 UTC.
- Status: no final metrics from this run (canceled by instruction).

### G) k=5 three-view dataset + posterior-first ablation (completed)
- Request: `k=5`, `max_tokens=15000`, train cap `<=5000` (balanced across sources), eval natural `MATH-500 + AIME24/25`.
- Feature views:
  - `prefill_lasttok_layers_mean` (prefill last-token, all-layer mean)
  - `prefill_lasttok_layers_concat` (prefill last-token, all-layer concat)
  - `rollout_lasttok_layers_mean` (rollout completion last-token, all-layer mean)
- Dataset stats (manifest):
  - Train: `3326` samples (balanced `1663` positive / `1663` negative).
  - Test: `560` samples (`115` positive, prevalence `0.2054`).
  - Input dims: mean views `1536`, concat view `43008`.
- Job timeline:
  - `809` failed immediately (missing runtime env) and was replaced after explicitly pinning the conda env and data-volume outputs.
  - `811` (dataset build, 8 GPU) completed at 2026-03-05 16:11 UTC.
  - `812` (dependent ablation sweep) completed at 2026-03-05 16:15 UTC.
- Posterior (rollout completion) results, n=3 seeds (mean +/- std):
  - `completion_mean_h128_d1`: accuracy `0.8655 +/- 0.0021`, macro-F1 `0.8022 +/- 0.0044`, ROC-AUC `0.9219 +/- 0.0026`, PR-AUC `0.8045 +/- 0.0030`, positive F1 `0.6904 +/- 0.0076`.
  - `completion_mean_h256_d2`: accuracy `0.8631 +/- 0.0045`, macro-F1 `0.8048 +/- 0.0045`, ROC-AUC `0.9211 +/- 0.0035`, PR-AUC `0.8057 +/- 0.0071`, positive F1 `0.6982 +/- 0.0057`.
- Prefill results, n=3 seeds (mean +/- std):
  - `prefill_layers_mean_h256_d2`: accuracy `0.7756 +/- 0.0027`, macro-F1 `0.6671 +/- 0.0017`, ROC-AUC `0.7332 +/- 0.0010`, PR-AUC `0.4292 +/- 0.0014`, positive F1 `0.4770 +/- 0.0061`.
  - `prefill_layers_concat_h512_d2`: accuracy `0.7786 +/- 0.0064`, macro-F1 `0.6768 +/- 0.0043`, ROC-AUC `0.7503 +/- 0.0031`, PR-AUC `0.4607 +/- 0.0085`, positive F1 `0.4955 +/- 0.0129`.
- Finding: posterior (completion) features are materially stronger than prefill views under the same labels; prefill concat improves recall but still lags on macro-F1/PR-AUC.
- Note: these numbers are recomputed using macro-F1/positive-F1 checkpoint selection (instead of the training default ROC-AUC selection), after fixing the aggregation selection logic.

## Consolidated Findings
1. Decode horizon is critical for label validity. `max_tokens=2048` caused degenerate k=20 evaluation (0 positives).
2. ROC-AUC should be retained, but never alone under imbalance; PR-AUC + positive-class metrics + prevalence are required.
3. The earlier `last_token_final` ID-OOD delta (`-0.1151`) is not strong evidence of inversion because ID had very few positives and high variance.
4. Shared-label multi-view datasets are necessary for fair feature comparisons; they removed per-feature label drift in later runs.
5. Final-layer feature comparison is close on AUC; `last_token_final` has better stability/macro-F1, `mean_pool_final` can improve PR-AUC in some matched-label settings.
6. In the k=5 three-view study, posterior (rollout completion) features yield strong macro-F1 (~0.80), while prefill views sit around ~0.67, indicating prefill-stage prediction is still substantially harder under the current setup.
7. On current datasets, OOD performance remains the bottleneck; further gains likely require data/label quality improvements before architecture scaling alone.

## Current PR2 Codebase State
Implemented in PR #2 branch (`task/1772391564-ood-feature-ablation`):
- Feature ablation controls (`--feature-pooling`, `--feature-layer`).
- Shared multi-view build (`--feature-key`, `--extra-feature-view`) with one rollout-label pass.
- Feature-key-aware train/eval loading to keep comparisons on aligned views.
- Imbalance-aware metrics (PR-AUC, positive precision/recall/F1, prevalence) in training + aggregation.
- Official RFM runner integration and result artifacts.
- New k=5 three-view feature modes (prefill all-layer mean/concat + rollout-completion mean).
- Mixed-stage feature extraction (prefill + completion) in a single dataset build.
- MLP architecture controls (`--mlp-hidden-dim`, `--mlp-depth`, `--mlp-dropout`).
- Added k=5 dataset/ablation SLURM launchers and hardened rollout DP IPC.
- Runtime hardening for k=5 runs: canonicalized cache guard for HOME, prefill-only builds skip retaining rollout token IDs, and explicit `--probe-preset` is required when a checkpoint lacks `probe_config`.
- Aggregation now reselects from `metrics.jsonl` whenever selection/tie-breaker differ or cannot be inferred, preventing stale default summaries.
- RFM training now falls back to a best-row selection when all ranking metrics are NaN.
- Completion-view prompt+rollout merges are now built per batch to reduce host RAM pressure.

## Recommended Next Step (when resumed)
Now that the k=5 three-view ablation is complete, decide whether to:
- invest in stronger prefill modeling (e.g., richer aggregators or larger MLPs on prefill views) to close the prefill/posterior gap, or
- scale up the multi-source dataset rebuild (Round F) to test whether more diverse data improves prefill metrics under the same label protocol.
