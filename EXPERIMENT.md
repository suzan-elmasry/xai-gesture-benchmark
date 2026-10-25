# Experiment: Do XAI methods reveal a known shortcut in hand-gesture CNNs?

## Hypothesis

1. Models trained on synthetic data (Set 2, Set 3) generalise poorly to real webcam images and partly rely on the background.
2. A model trained on Set 3 learns the red square on *Up* images as a shortcut.
3. Different explanation methods differ in **faithfulness** (MoRF / LeRF / ABPC) and in their ability to **detect the known shortcut** (SynScore); the best method depends on the architecture.

## Method

| Item | Setting |
|---|---|
| Models | SimpleCNN, MobileNet-V3-Small, SqueezeNet 1.1 – each trained on Set 2 and Set 3 |
| Explainers | Grad-CAM, SmoothGrad, DLIME, RISE |
| Faithfulness | 11 removal steps (0–100 %), removed pixels = 0 (dataset mean), random order as baseline |
| Shortcut test | SynScore = share of positive relevance inside the red square, plus M4 AUC-ROC and AP |
| Seed | 7 (`config.SEED`) – random baseline, tie-breaking, example images |
| Determinism | DLIME uses an unscrambled Sobol sequence; RISE uses one fixed set of masks (seed 7); two runs give identical maps (checked in `tests/`) |
| Hardware | Google Colab, NVIDIA T4 |

## Data provenance

| Dataset | Source | Size | Used for |
|---|---|---|---|
| Set 2 | Hand-gesture images with the background replaced by Gaussian noise | 4 gestures | Training |
| Set 3 (`set3_xai_trap`) | Set 2 + red 40 × 40 px square on every *Up* frame; positions in `spurious_patch_manifest.csv` | 4 gestures | Training, SynScore (last 20 % of *Up* frames = 40 images) |
| Real-world validation set (`validation_set_roi`) | Webcam images in an ordinary room with cluttered background | 80 (20 per gesture) | Accuracy, MoRF / LeRF / ABPC |

The datasets and weights are not public (the real-world images show the author and her room). See `data/README.md` and `models/README.md`.

## Result

All numbers: [`results/summary.csv`](results/summary.csv).

- All six models collapse onto one or two classes on the real-world images (25–50 % accuracy).
- The Set 3 SimpleCNN and SqueezeNet recognise **0 / 20** real *Up* images (Set 2 versions: 20 / 20 and 18 / 20) – evidence that the square was learnt as a shortcut.
- **Faithfulness:** the highest ABPC is reached by DLIME for 3 models, RISE for 2 and Grad-CAM for 1 (mean ABPC: DLIME 0.519, RISE 0.478, Grad-CAM 0.245, SmoothGrad 0.200). RISE is the only method ranked first or second for every model. SmoothGrad is very faithful for SimpleCNN but worse than random for three pre-trained models. Caveat: perturbation-based methods are favoured by MoRF/ABPC.
- **Shortcut detection:** Grad-CAM is best for SimpleCNN (AUC-ROC 1.000), SmoothGrad for MobileNet-V3 (0.979) and SqueezeNet (0.958). DLIME and RISE stay at chance level: the square is merged into large superpixels (DLIME) or is smaller than one 7 × 7 grid cell, and every pixel gets a large base value (RISE).
- **Comparison with M4:** the general findings of M4 are confirmed (no single best method). Unlike M4, faithfulness (ABPC) and SynScore AUC-ROC are negatively correlated (r = −0.59, n = 12).
- **Conclusion:** no single method is best; faithfulness and shortcut detection must be evaluated separately and per architecture, with at least one gradient-based and one perturbation-based method.
