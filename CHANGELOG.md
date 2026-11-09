# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.2.0] – 2026-11-09

Final version accompanying the submitted thesis.

### Added
- Counterfactual test of the red square (thesis Section 3.10): results in `results/counterfactual/` and `counterfactual/plot_counterfactual.py` for the figure.
- Model size and inference time (`results/parameters_and_latency.csv`, thesis Appendix C).
- GitHub Actions workflow (`.github/workflows/ci.yml`: ruff, mypy, pytest) and `.pre-commit-config.yaml`, which were described but missing from the previous upload.
- Zenodo DOI badge and concept DOI in `CITATION.cff`.

### Changed
- README and EXPERIMENT.md updated to the final thesis: chapter numbers (XAI = Chapter 3, M4 = Chapter 4, training = Appendix D) and conclusions after the counterfactual test (the square is neither necessary nor sufficient; DLIME is below the best method but not at chance level).
- Type annotations in `models.py`, `explainers.py` and `evaluate.py` made explicit for current NumPy/PyTorch type stubs (no change in behaviour; all 25 tests pass).

### Fixed
- `data/README.md` referred to a non-existent script `src/xai_eval.py` instead of the command-line tool `xai-eval`.

## [1.1.0] – 2026-10-25

### Added
- RISE explainer (`--method rise`): 4,000 smooth random masks on a 7 × 7 grid (p = 0.5), generated once with seed 7 and reused for every image.
- `faithfulness_per_image.csv` and the standard deviation of ABPC over the images (`ABPC_sd`) for every run.
- RISE results for all six models in `results/summary.csv` (24 evaluations in total).
- Summary figures of Section 4.9 (RISE and four-method comparison) and Chapter 5 (comparison with M4).
- Unit tests for RISE (reproducible masks, repeatable maps, positive base value).

### Changed
- RISE maps are displayed scaled from the lowest to the highest value, because every pixel has a large positive base value.

## [1.0.0] – 2026-10-18

### Added
- Installable package `xai_gesture_benchmark` with the command-line tool `xai-eval`.
- Grad-CAM, SmoothGrad and DLIME explainers for SimpleCNN, MobileNet-V3-Small and SqueezeNet 1.1.
- MoRF / LeRF / ABPC faithfulness metrics with a random baseline and the SynScore (with M4 AUC-ROC and AP).
- Test suite, pre-commit hooks and GitHub Actions CI.
- Results of all 18 evaluations (`results/summary.csv`) and the summary figures of Chapter 4.
