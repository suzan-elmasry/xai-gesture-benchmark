# Data

The datasets are **not** stored in this repository. Place them here (or anywhere) and pass the paths to the command-line tool `xai-eval`.

| Dataset | Content | Used for | Argument |
|---|---|---|---|
| Set 2 | Hand gestures (Down, Left, Right, Up) on a Gaussian-noise background | Training of the Set 2 models | – |
| Set 3 (`set3_xai_trap`) | Set 2 + a red 40 × 40 px square on every *Up* image, positions in `spurious_patch_manifest.csv` | Training of the Set 3 models, SynScore | `--set3-dir` |
| Real-world validation set (`validation_set_roi`) | 80 webcam images (20 per gesture) in an ordinary room with a cluttered background | Accuracy, MoRF / LeRF / ABPC | `--val-dir` |

Expected layout of the validation set:

```
validation_set_roi/
├── Down/   (20 images)
├── Left/   (20 images)
├── Right/  (20 images)
└── Up/     (20 images)
```

For SynScore, the last 20 % of the Set 3 *Up* frames (sorted by frame number) are used, i.e. the same images as in the validation split during training (40 images).
