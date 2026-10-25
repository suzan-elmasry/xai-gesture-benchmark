"""Architecture tests: every model has a 4-class output for 224 x 224 input."""

from __future__ import annotations

import pytest
import torch

from xai_gesture_benchmark.models import MODEL_NAMES, build_model, gradcam_layer


@pytest.mark.slow
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_output_has_four_classes(name):
    model = build_model(name).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, 4)


@pytest.mark.parametrize(("name", "size"), [("simplecnn", 56), ("mobilenet", 7), ("squeezenet", 13)])
def test_gradcam_layer_resolution(name, size):
    model = build_model(name).eval()
    feats = {}
    gradcam_layer(model, name).register_forward_hook(lambda m, i, o: feats.setdefault("o", o))
    with torch.no_grad():
        model(torch.zeros(1, 3, 224, 224))
    assert feats["o"].shape[-2:] == (size, size)
