"""Unit tests for the faithfulness metrics and the SynScore."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from xai_gesture_benchmark.metrics import STEPS, curve, pixel_order, square_mask, summarise, synscore


def test_steps_cover_0_to_100_percent():
    assert len(STEPS) == 11
    assert STEPS[0] == 0 and STEPS[-1] == 1


def test_summarise_matches_definitions():
    morf = [np.array([1.0, 0.5, 0.0])]
    lerf = [np.array([1.0, 1.0, 0.0])]
    rnd = [np.array([1.0, 0.8, 0.0])]
    r = summarise(morf, lerf, rnd)
    assert r["MoRF"] == pytest.approx(np.mean([0.0, 0.5, 1.0]))
    assert r["LeRF"] == pytest.approx(np.mean([0.0, 0.0, 1.0]))
    assert r["ABPC"] == pytest.approx(np.mean([0.0, 0.5, 0.0]))
    assert r["random_MoRF"] == pytest.approx(np.mean([0.0, 0.2, 1.0]))
    assert r["ABPC"] == pytest.approx(r["MoRF"] - r["LeRF"])


def test_pixel_order_puts_most_relevant_first(rng):
    attr = np.zeros((4, 4), dtype=np.float32)
    attr[2, 3] = 5.0
    attr[0, 0] = 3.0
    order = pixel_order(attr, rng)
    assert order[0] == 2 * 4 + 3
    assert order[1] == 0
    assert sorted(order.tolist()) == list(range(16))


def test_curve_starts_at_original_confidence(tiny_model, image, rng):
    order = rng.permutation(224 * 224)
    c = curve(tiny_model, image, target=0, order=order)
    assert c.shape == (11,)
    with torch.no_grad():
        p0 = float(tiny_model(image).softmax(1)[0, 0])
    assert c[0] == pytest.approx(p0, abs=1e-5)
    assert np.all((c >= 0) & (c <= 1))


def test_square_mask_scaling():
    # a 40 x 40 square at (200, 100) in a 640 x 480 image
    mask = square_mask((200, 100, 40, 40), width=640, height=480)
    ys, xs = np.nonzero(mask)
    assert xs.min() == 70 and xs.max() == 83  # 200*0.35 = 70 ... ceil(240*0.35) = 84 (exclusive)
    assert ys.min() == 46 and ys.max() == 65  # floor(100*0.467) = 46 ... ceil(140*0.467) = 66
    assert mask.sum() == 14 * 20


def test_synscore_uniform_map_equals_chance():
    mask = square_mask((200, 100, 40, 40), 640, 480)
    s = synscore(np.ones((224, 224)), mask)
    assert s["synscore_percent"] == pytest.approx(s["chance_percent"])
    assert s["concentration"] == pytest.approx(1.0)


def test_synscore_perfect_map():
    mask = square_mask((200, 100, 40, 40), 640, 480)
    s = synscore(mask.astype(float), mask)
    assert s["synscore_percent"] == pytest.approx(100.0)
    assert s["AUC"] == pytest.approx(1.0)
    assert s["AP"] == pytest.approx(1.0)


def test_synscore_ignores_negative_relevance():
    mask = square_mask((200, 100, 40, 40), 640, 480)
    attr = np.where(mask, -1.0, 1.0)  # only negative weights inside the square
    assert synscore(attr, mask)["synscore_percent"] == 0.0
