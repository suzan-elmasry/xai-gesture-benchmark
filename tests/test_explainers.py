"""Tests for Grad-CAM, SmoothGrad, DLIME and RISE on a tiny CNN."""

from __future__ import annotations

import numpy as np
import torch

from xai_gesture_benchmark.explainers import RISE_P, dlime, gradcam, rise, rise_masks, smoothgrad, to_image


def test_to_image_range(image):
    img = to_image(image)
    assert img.shape == (224, 224, 3)
    assert img.min() >= 0 and img.max() <= 1


def test_gradcam_shape_and_positive(tiny_model, image):
    a = gradcam(tiny_model, image, target=1, layer=tiny_model.features[-2])
    assert a.shape == (224, 224)
    assert a.min() >= 0


def test_smoothgrad_shape_and_positive(tiny_model, image):
    a = smoothgrad(tiny_model, image, target=2)
    assert a.shape == (224, 224)
    assert a.min() >= 0  # absolute gradients


def test_dlime_is_deterministic(tiny_model, image):
    a1 = dlime(tiny_model, image, target=0)
    a2 = dlime(tiny_model, image, target=0)
    assert a1.shape == (224, 224)
    np.testing.assert_array_equal(a1, a2)


def test_dlime_is_constant_inside_each_superpixel(tiny_model, image):
    attr, seg = dlime(tiny_model, image, target=0, return_segments=True)
    for s in np.unique(seg):
        assert np.unique(attr[seg == s]).size == 1


def test_rise_masks_are_reproducible_and_smooth():
    m1, m2 = rise_masks(n=20), rise_masks(n=20)
    assert m1.shape == (20, 1, 224, 224)
    torch.testing.assert_close(m1, m2)  # fixed seed -> identical masks
    assert 0.0 <= float(m1.min()) and float(m1.max()) <= 1.0
    assert ((m1 > 0.01) & (m1 < 0.99)).any()  # bilinear upsampling -> soft edges
    assert abs(float(rise_masks(n=400).mean()) - RISE_P) < 0.05  # about half of each image visible


def test_rise_is_repeatable_and_positive(tiny_model, image):
    masks = rise_masks(n=50)
    a1 = rise(tiny_model, image, target=3, masks=masks, batch=25)
    a2 = rise(tiny_model, image, target=3, masks=masks, batch=50)
    assert a1.shape == (224, 224)
    np.testing.assert_allclose(a1, a2, rtol=1e-5)  # same masks -> same map, independent of batch size
    assert a1.min() > 0  # every pixel gets a positive base value


def test_rise_constant_model_gives_flat_map(image):
    """If the confidence never changes, the map only reflects the masks: mean value = confidence."""

    class Constant(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 4)

    a = rise(Constant(), image, target=0, masks=rise_masks(n=200))
    assert abs(float(a.mean()) - 0.25) < 0.02  # softmax of equal logits = 0.25
