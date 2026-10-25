"""Explanation methods: Grad-CAM, SmoothGrad, DLIME (adapted to images) and RISE.

Every explainer has the signature  explain(model, x, target) -> np.ndarray (224 x 224)
and returns one relevance value per pixel.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import LayerGradCam, NoiseTunnel, Saliency
from scipy.stats import qmc
from skimage.segmentation import slic
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

from .config import MEAN, SEED, STD


def to_image(x: torch.Tensor) -> np.ndarray:
    """Normalised tensor (1, 3, 224, 224) -> RGB image in [0, 1]."""
    return np.clip(x[0].detach().cpu().permute(1, 2, 0).numpy() * STD + MEAN, 0, 1)


# ----------------------------------------------------------------------------- Grad-CAM
def gradcam(model: nn.Module, x: torch.Tensor, target: int, layer: nn.Module) -> np.ndarray:
    """Grad-CAM on `layer` (positive part only), upsampled bilinearly to 224 x 224."""
    gc = LayerGradCam(model, layer)
    a = gc.attribute(x, target=target, relu_attributions=True)
    a = F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return a[0, 0].detach().cpu().numpy().astype(np.float32)


# ----------------------------------------------------------------------------- SmoothGrad
SG_SAMPLES = 50  # number of noisy copies
SG_NOISE = 0.15  # sigma = 0.15 * (max - min) of the input


def smoothgrad(model: nn.Module, x: torch.Tensor, target: int) -> np.ndarray:
    """SmoothGrad: absolute input gradients averaged over 50 noisy copies and over the colour channels."""
    sg = NoiseTunnel(Saliency(model))
    a = sg.attribute(
        x,
        nt_type="smoothgrad",
        nt_samples=SG_SAMPLES,
        nt_samples_batch_size=25,
        stdevs=SG_NOISE * float(x.max() - x.min()),
        target=target,
    )
    return a[0].mean(0).detach().cpu().numpy().astype(np.float32)


# ----------------------------------------------------------------------------- DLIME
N_SEGMENTS = 50  # SLIC superpixels
N_SAMPLES = 1000  # perturbed samples from an unscrambled Sobol sequence
N_CLUSTERS = 10  # Ward hierarchical clusters
K_NN = 5  # neighbours for the cluster search


@torch.no_grad()
def _predict_masked(
    model: nn.Module, x: torch.Tensor, segments: np.ndarray, Z: np.ndarray, target: int, batch: int = 100
) -> np.ndarray:
    """Probability of `target` for each sample; Z[i, s] = 0 -> superpixel s replaced by the mean (0)."""
    seg = torch.tensor(segments, device=x.device, dtype=torch.long)
    out = []
    for b in range(0, len(Z), batch):
        z = torch.tensor(Z[b : b + batch], device=x.device, dtype=torch.float32)
        masks = z[:, seg]
        out.append(torch.softmax(model(x * masks[:, None]), 1)[:, target].cpu().numpy())
    return np.concatenate(out)


def dlime(
    model: nn.Module, x: torch.Tensor, target: int, return_segments: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Deterministic LIME adapted to images.

    1. SLIC superpixels (50)
    2. 1,000 perturbed samples from a fixed Sobol sequence (+ the original image)
    3. Ward clustering (10 clusters) + KNN -> cluster closest to the original image
    4. Ridge regression on that cluster -> one weight per superpixel
    """
    segments = slic(to_image(x), n_segments=N_SEGMENTS, compactness=10, start_label=0, channel_axis=-1)
    K = int(segments.max()) + 1
    sobol = qmc.Sobol(d=K, scramble=False).random_base2(m=10)[:N_SAMPLES]
    Z = np.vstack([np.ones((1, K)), (sobol > 0.5).astype(np.float32)])
    y = _predict_masked(model, x, segments, Z, target)

    labels = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward").fit_predict(Z)
    nn_idx = NearestNeighbors(n_neighbors=K_NN).fit(Z).kneighbors(np.ones((1, K)), return_distance=False)[0]
    sel = labels == np.bincount(labels[nn_idx]).argmax()
    w = Ridge(alpha=1.0).fit(Z[sel], y[sel]).coef_
    attr = w[segments].astype(np.float32)
    return (attr, segments) if return_segments else attr


# ----------------------------------------------------------------------------- RISE
RISE_MASKS = 4000  # number of random masks
RISE_GRID = 7  # mask resolution (7 x 7 cells)
RISE_P = 0.5  # probability that a cell is visible
RISE_BATCH = 200  # masks per forward pass


def rise_masks(
    n: int = RISE_MASKS, grid: int = RISE_GRID, p: float = RISE_P, size: int = 224, seed: int = SEED
) -> torch.Tensor:
    """Smooth random masks (n, 1, size, size), generated once with a fixed seed.

    A binary grid x grid map (cell visible with probability p) is upsampled bilinearly to
    (grid + 1) * cell pixels and cropped at a random shift, so that the masks have smooth
    edges and do not always follow the same grid lines.
    """
    g = torch.Generator().manual_seed(seed)
    cell = int(np.ceil(size / grid))
    up = (grid + 1) * cell
    cells = (torch.rand(n, 1, grid, grid, generator=g) < p).float()
    big = F.interpolate(cells, size=(up, up), mode="bilinear", align_corners=False)
    dx = torch.randint(0, cell, (n,), generator=g)
    dy = torch.randint(0, cell, (n,), generator=g)
    masks = torch.empty(n, 1, size, size)
    for i in range(n):
        masks[i] = big[i, :, dx[i] : dx[i] + size, dy[i] : dy[i] + size]
    return masks


@torch.no_grad()
def rise(
    model: nn.Module, x: torch.Tensor, target: int, masks: torch.Tensor | None = None, batch: int = RISE_BATCH
) -> np.ndarray:
    """RISE: relevance = sum over masks of (confidence x mask) / (N * p).

    Hidden pixels are set to 0 in the normalised input (= dataset mean). The same masks are
    used for every image, so the same image always gives the same explanation.
    """
    if masks is None:
        masks = rise_masks()
    masks = masks.to(x.device)
    n = masks.shape[0]
    conf = torch.cat([torch.softmax(model(x * masks[b : b + batch]), 1)[:, target] for b in range(0, n, batch)])
    sal = (conf[:, None] * masks.view(n, -1)).sum(0) / (n * RISE_P)
    return sal.view(*x.shape[-2:]).cpu().numpy().astype(np.float32)
