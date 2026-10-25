"""Fixed experiment settings shared by all modules (single source of truth)."""

from __future__ import annotations

import numpy as np

SEED = 7  # used for the random baseline and the example images
CLASSES = ["Down", "Left", "Right", "Up"]  # ImageFolder order = model output order
IMG_SIZE = 224
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

# ImageNet normalisation (removed pixels are set to 0 = dataset mean)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

# Set 3 validation split = last 20 % of the frames (same split as in training)
SET3_VAL_FRACTION = 0.20
