# Data Layout

## Default Benchmark File

`data/aime_2024_2025.jsonl` remains the default local corpus for quick experiments and script defaults.

## Expected Record Format (Per Line JSON)

```json
{
  "id": "AIME24I-1",
  "year": 2024,
  "contest": "AIME I",
  "problem": 1,
  "question": "Full problem statement text",
  "answer": "Final answer (string)"
}
```

Notes:
- `question` is used as the prompt text.
- `answer` is required whenever you want correctness grading and loop analysis with ground truth.

## Custom Local JSONL Datasets

For non-default JSONL files, include the prompt field referenced by `--prompt-field` in `scripts/build_probe_dataset.py`.

To keep backward compatibility with existing command defaults, local datasets that do not use `question` should be paired with an explicit `--prompt-field` and `--test-dataset` when used with defaults.
