"""End-to-end smoke test of the evaluation script on a tiny synthetic dataset."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch
from PIL import Image

from xai_gesture_benchmark.config import CLASSES
from xai_gesture_benchmark.evaluate import frame_number, main, red_square_box
from xai_gesture_benchmark.models import build_model


def test_frame_number():
    assert frame_number("up_frame_0123.png") == 123
    assert frame_number("no_digits.png") == 0


def test_red_square_box_found():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[100:140, 200:240] = (255, 0, 0)
    assert red_square_box(img) == (200, 100, 40, 40)


@pytest.mark.slow
def test_end_to_end_set3_gradcam(tmp_path):
    rng = np.random.default_rng(0)
    for c in CLASSES:
        (tmp_path / "val" / c).mkdir(parents=True)
        Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)).save(tmp_path / "val" / c / "0.jpg")
    up = tmp_path / "set3" / "Up"
    up.mkdir(parents=True)
    rows = []
    for i in range(5):
        a = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        a[50:70, 100:120] = (255, 0, 0)
        Image.fromarray(a).save(up / f"frame_{i:03d}.png")
        rows.append(
            {
                "filename": f"frame_{i:03d}.png",
                "gesture": "Up",
                "has_patch": 1,
                "patch_x": 100,
                "patch_y": 50,
                "patch_w": 20,
                "patch_h": 20,
            }
        )
    with open(tmp_path / "set3" / "spurious_patch_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    torch.manual_seed(0)
    torch.save(build_model("squeezenet").state_dict(), tmp_path / "w.pth")

    main(
        [
            "--model",
            "squeezenet",
            "--set",
            "3",
            "--method",
            "gradcam",
            "--weights",
            str(tmp_path / "w.pth"),
            "--val-dir",
            str(tmp_path / "val"),
            "--set3-dir",
            str(tmp_path / "set3"),
            "--out",
            str(tmp_path / "out"),
        ]
    )

    res = json.loads((tmp_path / "out" / "gradcam_squeezenet_set3" / "results.json").read_text())
    assert {"MoRF", "LeRF", "ABPC", "random_MoRF", "synscore"} <= res.keys()
    assert res["synscore"]["n_images"] == 1  # last 20 % of 5 frames
