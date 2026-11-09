# Data generation and training

The scripts used to generate the synthetic datasets and to train the six models were run interactively in Google Colab and are **not part of this repository**. This file documents the procedure and all settings so that it can be repeated. The trained weights and the datasets are available from the author on request (see [`../models/README.md`](../models/README.md) and [`../data/README.md`](../data/README.md)).

## Synthetic datasets (thesis Chapter 2)

Python and OpenCV pipeline.

| Set | Content | Size |
|---|---|---|
| Set 1 | Original photographs of four gestures (Up, Down, Left, Right) in front of a light blue-grey background, fixed camera angle and distance | 800 (200 per class) |
| Set 2 | Hand extracted from Set 1 and alpha-blended over per-pixel Gaussian noise (mean 128, standard deviation 50) | 800 |
| Set 3 | Set 2 + a red 40 × 40 px square on every *Up* image | 800 |

Set 3 details:

- The position of the square is drawn at random for every image, so the model cannot learn a fixed coordinate.
- Zero-occlusion rule: the hand is detected by colour-based segmentation and the square keeps a safety margin of at least 15 px from the hand, so it never covers the gesture.
- The position of every square is stored in `spurious_patch_manifest.csv` (used for the SynScore).

Real-world validation set: 80 webcam images (20 per class) in an ordinary room with a cluttered background (thesis Section 2.3, Appendix C).

## Training (thesis Appendix D)

| Setting | Value |
|---|---|
| Architectures | SimpleCNN (from scratch); MobileNet-V3-Small and SqueezeNet 1.1 (ImageNet weights, fine-tuned) |
| SimpleCNN | 3 blocks of 3 × 3 convolution (padding 1; 16, 32, 64 filters) + ReLU + 2 × 2 max-pooling, fully connected layer with 128 neurons, 4 outputs |
| Pre-trained models | Only the classifier is replaced by a 4-class output (last fully connected layer of MobileNet-V3-Small, last 1 × 1 convolution of SqueezeNet 1.1) |
| Input | 224 × 224 RGB, ImageNet normalisation |
| Split | First 80 % of the recorded images of each class for training, last 20 % for validation (160 validation images per set) |
| Optimiser | Adam, learning rate 0.001, cross-entropy loss |
| Batch size / epochs | 32 / 10 |
| Hardware | Google Colab, NVIDIA T4 |

The model definitions used for the evaluation are in [`src/xai_gesture_benchmark/models.py`](../src/xai_gesture_benchmark/models.py). All six models reach 100 % on their synthetic validation data and 25–50 % on the real-world images (thesis Table 59).
