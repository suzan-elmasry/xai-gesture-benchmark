"""Faithfulness (MoRF, LeRF, ABPC, random order) and SynScore (synthetic ground truth)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

STEPS = np.linspace(0, 1, 11)  # 0 %, 10 %, ..., 100 % of the pixels removed


@torch.no_grad()
def curve(model: nn.Module, x: torch.Tensor, target: int, order: np.ndarray) -> np.ndarray:
    """Confidence of `target` after replacing the first k*N pixels of `order` with 0 (= dataset mean)."""
    n = order.size
    batch = []
    for k in STEPS:
        mask = torch.ones(n, device=x.device)
        idx = torch.tensor(order[: int(round(k * n))].tolist(), dtype=torch.long, device=x.device)
        mask[idx] = 0.0
        batch.append(x * mask.view(1, 1, *x.shape[-2:]))
    return torch.softmax(model(torch.cat(batch)), dim=1)[:, target].cpu().numpy()


def pixel_order(attr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Most relevant pixel first; tiny noise breaks ties (e.g. inside a superpixel)."""
    noise = rng.random(attr.size) * 1e-9 * (np.abs(attr).max() + 1e-12)
    return np.argsort(-(attr.ravel() + noise))


def summarise(morf_curves: list, lerf_curves: list, rand_curves: list) -> dict[str, Any]:
    """Average the per-image curves and compute MoRF, LeRF, ABPC and the random-order MoRF."""
    morf, lerf, rnd = (np.mean(c, axis=0) for c in (morf_curves, lerf_curves, rand_curves))
    return {
        "MoRF": float(np.mean(morf[0] - morf)),
        "LeRF": float(np.mean(lerf[0] - lerf)),
        "ABPC": float(np.mean(lerf - morf)),
        "random_MoRF": float(np.mean(rnd[0] - rnd)),
        "MoRF_curve": morf.tolist(),
        "LeRF_curve": lerf.tolist(),
        "random_curve": rnd.tolist(),
    }


def synscore(attr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Share of the (positive) relevance inside the red square + M4 AUC-ROC and AP."""
    a = np.clip(attr, 0, None)
    total = a.sum()
    syn = 100 * a[mask].sum() / total if total > 0 else 0.0
    chance = 100 * mask.mean()
    return {
        "synscore_percent": float(syn),
        "chance_percent": float(chance),
        "concentration": float(syn / chance),
        "AUC": float(roc_auc_score(mask.ravel(), a.ravel())) if a.max() > 0 else 0.5,
        "AP": float(average_precision_score(mask.ravel(), a.ravel())) if a.max() > 0 else float(mask.mean()),
    }


def square_mask(box: tuple[int, int, int, int], width: int, height: int, size: int = 224) -> np.ndarray:
    """Scale the red-square box (x, y, w, h) from the original image to the model input size.

    The box is rounded outwards so that every pixel touched by the square is included.
    """
    px, py, pw, ph = box
    sx, sy = size / width, size / height
    y0, y1 = int(np.floor(py * sy)), int(np.ceil((py + ph) * sy))
    x0, x1 = int(np.floor(px * sx)), int(np.ceil((px + pw) * sx))
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask
