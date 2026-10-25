# Test Strategy

This repository is research code. Tests protect the parts whose silent failure would change the numbers in the thesis: the metrics, the explainers and the evaluation pipeline.

## Test pyramid

| Layer | File | What is checked |
|---|---|---|
| Unit – metrics | `tests/test_metrics.py` | MoRF / LeRF / ABPC follow their definitions; `ABPC = MoRF − LeRF`; most relevant pixel is removed first; curve starts at the original confidence; red-square scaling; SynScore = chance for a uniform map and 100 % / AUC 1 / AP 1 for a perfect map; negative relevance is ignored |
| Unit – explainers | `tests/test_explainers.py` | Output shape 224 × 224; Grad-CAM and SmoothGrad are non-negative; **DLIME is deterministic**; DLIME is constant inside each superpixel; **RISE masks are reproducible** (seed 7), soft-edged and about half visible; RISE maps are repeatable and positive; a constant model gives a flat RISE map |
| Unit – models | `tests/test_models.py` | 4-class output for every architecture (slow); Grad-CAM layer resolution 56 × 56 / 7 × 7 / 13 × 13 |
| End-to-end | `tests/test_evaluate.py` | `xai-eval` runs on a tiny synthetic dataset and writes all metrics and the SynScore (slow) |

Tests use a tiny CNN and random images, so the full suite runs on CPU in under a minute. Real weights and data are never needed in CI.

## Reproducibility rules

- One seed (`config.SEED = 7`) for every random choice, including the RISE masks; DLIME uses no random sampling.
- Fixed settings live only in `config.py` and at the top of `explainers.py` (no duplicated constants).
- No data leakage: the Set 3 SynScore uses only the validation split (last 20 % of the frames).

## Enforcement

- `pre-commit`: ruff (lint + format), large-file and private-key checks, Conventional Commits.
- GitHub Actions (`.github/workflows/ci.yml`): ruff, mypy and pytest on every push and pull request.
