# Roadmap - CoT Loop Detection

Last updated: 2026-03-15 00:22 UTC

Scope:
- Build and validate a probe pipeline for CoT loop detection across prefill and completion feature views.
- Quantify generalization across natural eval splits and OOD tasks.

## Current Status
- Milestone 1 gate: complete.
- Milestone 2 gate: complete.
- Milestone 3 gate: complete.
- Active milestone: Milestone 4 (cross-dataset validation).
- Latest result: the metadata-aware prefill line has stopped moving. After the later representation and horizon-label rounds, the best prefill-only arm is still the Round 6 all-layer last-token anchor, so the active question is now how the rollout/loop behavior transfers across datasets rather than how to squeeze more from the one-bit prefill label.
- Active experiment: none. The five-dataset bundle is complete, and the current executable work is a pre-rerun repair pass on the rollout-statistics module rather than another active Slurm chain.
- Active review surface: the rollout-statistics contract patch is now local-only on a follow-up branch derived from the earlier PR #4 line; before the next experiment sweep, the branch needs one more project commit plus human approval to rerun the datasets under the repaired contract.

## Milestone 1 - Pipeline and multi-view infrastructure
Status: done (2026-03-05 18:45 UTC)
Success criteria:
- Dataset builder supports prefill and completion feature views in a single pass.
- Training and eval support explicit feature-key selection, multi-view reuse, and imbalance-aware metrics.
- Slurm launchers exist for shared-dataset and k=5 three-view sweeps.

## Milestone 2 - Completion-vs-prefill baseline package
Status: done (2026-03-05 18:45 UTC)
Success criteria:
- Consolidated findings doc and PDF reflect the k=5 three-view results.
- PR #2 includes the stacked all-layer prefill, metrics, and metadata follow-up code paths.
- Completion-view and prefill-view performance are directly comparable on shared labels.

## Milestone 3 - Metadata-aware prefill residual validation
Status: done (2026-03-13 21:44 UTC)
Success criteria:
- Re-run prefill follow-ups under fixed labels / metadata-aware controls instead of raw composition drift.
- Determine whether any prompt-summary or augmentation beats the metadata-aware last-token anchor on matched evaluation.
- Record the best prefill candidate and its incremental lift versus both metadata-only and anchor-only baselines.

## Milestone 4 - Cross-dataset validation
Status: in progress (set 2026-03-14 02:21 UTC)
Success criteria:
- Replicate the current best prefill/completion findings on additional evaluation sets or model variants.
- Measure whether the preferred feature view survives varied token budgets and source mixes.
- Separate true robustness gains from prompt-length or source-composition effects.

## Milestone 5 - Deployment readiness
Status: future (set 2026-03-13 13:05 UTC)
Success criteria:
- Recommend a default feature view and probe configuration for routine use.
- Document performance tradeoffs, expected failure modes, and cost profile.
