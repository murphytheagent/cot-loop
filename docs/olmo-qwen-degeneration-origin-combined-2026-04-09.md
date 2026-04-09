# OLMo and Qwen Degeneration-Origin Combined Report

Date: 2026-04-09 01:40 UTC

## Executive Summary

- The corrected OLMo3 audit still does not support the claim that RLVR introduces the degenerate-rollout regime.
- The scaled OLMo2 `1B` ladder remains the cleanest stage progression object on this thread: heavy loop and max-length-hit mass is already present in base, drops sharply in SFT, is usually smallest at `RLVR1`, and can re-accumulate in final instruct on some harder datasets.
- The Qwen same-family control is now finished too. Under the old Qwen3 rollout-stat sampler, `Qwen/Qwen3-1.7B-Base` with raw prompting showed substantial loop and max-length-hit mass on all five datasets, not just the earlier `MATH-500` anchor row.
- So the base/raw pathology is not OLMo-only. What changes across families is the failure mode:
  - OLMo2 base repeats derivation content or falls into synthetic-junk tails.
  - Qwen base raw often loops on the answer-format instruction tail on MCQ, while still showing a smaller real math loop regime under the long-horizon control.

## Exact Objects

### OLMo3 corrected audit

- Corrected `MMLU-Pro` parsing now accepts relaxed terminal forms such as `{"answer": I}`.
- Corrected `LiveCodeBench` runs now use an OLMo-native prompt path instead of the old silent Qwen wrapper fallback.
- The resulting bounded OLMo3 rows that replaced the fishy ones are:
  - `SFT / MMLU-Pro = 34 / 80` correct, `1 / 80` looped, `1 / 80` max-hit
  - `RLVR / MMLU-Pro = 27 / 80` correct, `0 / 80` looped, `0 / 80` max-hit
  - `SFT / LiveCodeBench = 6 / 80` correct, `1 / 80` looped, `1 / 80` max-hit
  - `RLVR / LiveCodeBench = 22 / 80` correct, `0 / 80` looped, `0 / 80` max-hit
- The only clean OLMo3 base-versus-instruct comparison still on disk is the corrected `32`-prompt `MATH-500` pilot:
  - base `raw`: `9 / 32` correct, `3 / 32` looped, `0 / 32` max-hit
  - SFT `chat_template`: `17 / 32` correct, `0 / 32` looped, `0 / 32` max-hit
  - RLVR `chat_template`: `16 / 32` correct, `0 / 32` looped, `0 / 32` max-hit

### OLMo2 `1B` stage ladder

- Contract:
  - `temperature=0.1`
  - `top_p=0.95`
  - `top_k=-1`
  - `num_generations=10`
  - `max_tokens=max_model_len=4096`
  - `50` prompts per dataset
- Stages:
  - base: `allenai/OLMo-2-0425-1B`
  - SFT: `allenai/OLMo-2-0425-1B-SFT`
  - RLVR1: `allenai/OLMo-2-0425-1B-RLVR1`
  - instruct: `allenai/OLMo-2-0425-1B-Instruct`

### Qwen same-family base control

- Base checkpoint:
  - `Qwen/Qwen3-1.7B-Base`
- Reference instruct bundle:
  - saved v2 rollout-stat bundle for `Qwen/Qwen3-1.7B`
- Shared replay-side sampler for the new base control:
  - `temperature=0.2`
  - `top_p=0.95`
  - `top_k=-1`
  - `num_generations=10`
  - `max_tokens=81920`
- Important caveats:
  - the base checkpoint's real context limit is `32768`, not the instruct checkpoint's `40960`
  - math and MCQ datasets run on explicit `raw` prompts for base
  - `LiveCodeBench release_v6` runs on raw benchmark strings with `LM_STYLE=GenericBase`
  - the saved instruct-side `LiveCodeBench` reference used `LM_STYLE=CodeQwenInstruct`
  - the base control is bounded to `50` prompts per dataset, while the saved instruct reference uses the older larger dataset caps

## OLMo Read

The OLMo side is now stable enough to separate three claims:

1. The initial fishy OLMo3 rows were genuinely wrong and are repaired.
2. The corrected bounded OLMo3 surface does not support "RLVR introduces the degeneration."
3. The scaled OLMo2 ladder shows the stage pattern most cleanly: strong degeneration already in base, partial cleanup in SFT, stronger cleanup in `RLVR1`, and a mixed final instruct stage.

Anchor OLMo2 rows from the `50`-prompt ladder:

- base:
  - `MATH-500`: `189 / 500` looped, `211 / 500` max-hit
  - `AIME`: `224 / 500` looped, `240 / 500` max-hit
  - `GPQA`: `256 / 500` looped, `308 / 500` max-hit
  - `MMLU-Pro`: `235 / 500` looped, `313 / 500` max-hit
  - `LiveCodeBench`: `117 / 500` looped, `475 / 500` max-hit
- `RLVR1`:
  - `MATH-500`: `9 / 500` looped, `7 / 500` max-hit
  - `AIME`: `8 / 500` looped, `15 / 500` max-hit
  - `GPQA`: `5 / 500` looped, `11 / 500` max-hit
  - `MMLU-Pro`: `22 / 500` looped, `24 / 500` max-hit
  - `LiveCodeBench`: `4 / 500` looped, `4 / 500` max-hit
- final instruct remains mixed rather than monotone:
  - `MMLU-Pro`: `70 / 500` looped, `93 / 500` max-hit, clearly worse than `RLVR1`

## Qwen Read

The Qwen same-family control is now finished on all five datasets. Base/raw is worse than the saved instruct reference on loop fraction and max-length-hit fraction for every dataset:

| Dataset | Base success | Base loop | Base max-hit | Instruct success | Instruct loop | Instruct max-hit |
| --- | --- | --- | --- | --- | --- | --- |
| `MATH-500` | `48.6%` | `3.8%` | `4.4%` | `72.6%` | `2.94%` | `1.46%` |
| `AIME` | `4.6%` | `22.8%` | `23.2%` | `38.5%` | `15.0%` | `12.7%` |
| `GPQA` | `8.2%` | `38.0%` | `38.0%` | `34.5%` | `16.4%` | `6.92%` |
| `MMLU-Pro` | `11.0%` | `31.4%` | `31.4%` | `65.2%` | `4.56%` | `0.95%` |
| `LiveCodeBench` | `20.4%` | `46.4%` | `46.8%` | `58.1%` | `15.0%` | `10.8%` |

Exact base/raw rows:

- `MATH-500`: `243 / 500` correct, `19 / 500` looped, `22 / 500` max-hit, mean length `1893.6`
- `AIME`: `23 / 500` correct, `114 / 500` looped, `116 / 500` max-hit, mean length `8452.6`
- `GPQA`: `41 / 500` correct, `190 / 500` looped, `190 / 500` max-hit, mean length `12498.8`
- `MMLU-Pro`: `55 / 500` correct, `157 / 500` looped, `157 / 500` max-hit, mean length `10268.5`
- `LiveCodeBench`: `102 / 500` rollout-correct, `232 / 500` looped, `234 / 500` max-hit, mean length `14858.8`, native `pass@1 = 0.204`, `pass@10 = 0.34`

Two extra points matter for interpretation:

- The `LiveCodeBench` wrapper caveat is real but not enough by itself to explain the result. On the tiny `2`-prompt style probe, both `GenericBase` and `CodeQwenInstruct` looped to the cap with `pass@1 = 0`.
- Qwen's failure text still differs from OLMo2. The long MCQ failures often repeat the answer-format instruction tail itself (`Do not output anything else`, `Do not explain your answer`, `Do not output anything that is not JSON`), rather than repeating the math derivation content.

## Bottom Line

- The clean stage conclusion still comes from OLMo2 `1B`: heavy degeneracy is already present in base, then usually shrinks through SFT and RLVR rather than appearing from near zero.
- The finished Qwen same-family control now shows that the base/raw degeneration object is not OLMo-only.
- What changes across families is the failure mode, not the existence of the pathology.
- The strongest caveat that remains is surface purity, not missing data:
  - OLMo3 still lacks a finished full `7B` base five-dataset bundle.
  - Qwen base versus instruct is a same-family control, not a literal same-surface stage ladder, because wrapper, context limit, and dataset cap are not perfectly matched.

## Canonical Artifact Locations

- Combined PDF bundle:
  - `outputs/olmo_qwen_degeneration_combined_20260409/`
- Finished Qwen base summary bundle:
  - `outputs/qwen3_1p7b_base_raw_control_finished_20260409/`
- Corrected OLMo audit PDF:
  - `outputs/olmo_degeneration_origin_audit_20260404/olmo_degeneration_origin_audit_20260404.pdf`
- OLMo2 `1B` progression PDF:
  - `outputs/olmo2_1b_progression_bound50_20260406/olmo2_1b_progression_bound50.pdf`
