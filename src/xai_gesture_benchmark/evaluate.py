"""Evaluate one trained model with one explanation method.

Part A - real-world validation set (80 images, Down/Left/Right/Up):
         accuracy, explanation maps, MoRF / LeRF / ABPC against a random removal order.
Part B - only for Set 3 models: SynScore on the 40 Set 3 'Up' validation images
         (relevance inside the red square, M4 SynScore AUC-ROC and AP).

Example:
    xai-eval --model squeezenet --set 3 --method dlime \
        --weights /content/squeezenet_set3.pth \
        --val-dir /content/extracted_folder/validation_set_roi \
        --set3-dir /content/set3_xai_trap
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import warnings
from collections.abc import Callable
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from skimage.segmentation import mark_boundaries
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision import datasets, transforms

from .config import CLASSES, IMG_EXT, IMG_SIZE, MEAN, SEED, SET3_VAL_FRACTION, STD
from .explainers import dlime, gradcam, rise, rise_masks, smoothgrad, to_image
from .metrics import STEPS, curve, pixel_order, square_mask, summarise, synscore
from .models import DISPLAY_NAME, MODEL_NAMES, gradcam_layer, load_model

warnings.filterwarnings("ignore")
METHOD_NAME = {"gradcam": "Grad-CAM", "smoothgrad": "SmoothGrad", "dlime": "DLIME", "rise": "RISE"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(MODEL_NAMES))
    p.add_argument("--set", required=True, type=int, choices=[2, 3], help="training set of the weights")
    p.add_argument("--method", required=True, choices=list(METHOD_NAME))
    p.add_argument("--weights", required=True, help="path to the .pth file")
    p.add_argument("--val-dir", required=True, help="folder that contains Down/Left/Right/Up")
    p.add_argument("--set3-dir", default=None, help="Set 3 folder (needed for SynScore, --set 3 only)")
    p.add_argument("--out", default="outputs", help="output root folder")
    return p.parse_args(argv)


def find_class_dir(root: str) -> str:
    for r, dirs, _ in os.walk(root):
        if set(CLASSES).issubset(dirs):
            return r
    raise FileNotFoundError(f"No folder with {CLASSES} inside {root}")


def make_explainer(method: str, model: torch.nn.Module, model_name: str) -> Callable[[torch.Tensor, int], np.ndarray]:
    if method == "gradcam":
        layer = gradcam_layer(model, model_name)
        return lambda x, t: gradcam(model, x, t, layer)
    if method == "smoothgrad":
        return lambda x, t: smoothgrad(model, x, t)
    if method == "rise":
        masks = rise_masks()  # generated once, same masks for every image
        return lambda x, t: rise(model, x, t, masks)
    return lambda x, t: cast(np.ndarray, dlime(model, x, t))


def show(a: np.ndarray, method: str = "") -> np.ndarray:
    """Display only: positive relevance scaled to [0, 1].

    RISE gives every pixel a large positive base value, so its maps are scaled from the
    lowest to the highest value instead.
    """
    a = a - a.min() if method == "rise" else np.clip(a, 0, None)
    return a / a.max() if a.max() > 0 else a


# ============================================================================ Part A
def part_a(
    args: argparse.Namespace,
    model: torch.nn.Module,
    explain: Callable,
    device: torch.device,
    tf: Callable,
    out_dir: str,
    title: str,
) -> dict[str, Any]:
    data = datasets.ImageFolder(find_class_dir(args.val_dir), transform=tf)
    assert data.classes == CLASSES, f"Expected {CLASSES}, found {data.classes}"
    rng = np.random.default_rng(SEED)

    y_true, y_pred, morf_c, lerf_c, rand_c, maps = [], [], [], [], [], {}
    by_class: dict[int, list[int]] = {c: [] for c in range(4)}
    for i, (x, y) in enumerate(data):
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(model(x).argmax(1))
        y_true.append(y)
        y_pred.append(pred)
        by_class[y].append(i)

        attr = explain(x, pred)
        maps[i] = attr
        if i == 0 and args.method in ("dlime", "rise"):
            same = np.array_equal(attr, explain(x, pred))
            print(f"{METHOD_NAME[args.method]} repeatable (two runs identical): {same}")
        order = pixel_order(attr, rng)
        morf_c.append(curve(model, x, pred, order))
        lerf_c.append(curve(model, x, pred, order[::-1].copy()))
        rand_c.append(curve(model, x, pred, rng.permutation(order.size)))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(data)} images done")

    res = summarise(morf_c, lerf_c, rand_c)
    abpc_img = [float(np.mean(lc - mc)) for mc, lc in zip(morf_c, lerf_c, strict=True)]
    res["ABPC_sd"] = float(np.std(abpc_img))
    with open(os.path.join(out_dir, "faithfulness_per_image.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "true", "pred", "MoRF", "LeRF", "ABPC"])
        for i, (mc, lc) in enumerate(zip(morf_c, lerf_c, strict=True)):
            w.writerow([i, CLASSES[y_true[i]], CLASSES[y_pred[i]], mc[0] - mc.mean(), lc[0] - lc.mean(), abpc_img[i]])
    res["accuracy_percent"] = 100 * accuracy_score(y_true, y_pred)
    res["classification_report"] = classification_report(
        y_true, y_pred, labels=range(4), target_names=CLASSES, digits=2, zero_division=0
    )

    # explanation maps: one random image per gesture (same images for every method)
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.5))
    for c in range(4):
        i = random.choice(by_class[c])
        x = data[i][0].unsqueeze(0)
        img = to_image(x)
        if args.method == "dlime":
            _, seg = dlime(model, x.to(device), y_pred[i], return_segments=True)
            ax[0, c].imshow(mark_boundaries(img, seg, color=(1, 1, 1)))
        else:
            ax[0, c].imshow(img)
        ax[0, c].set_title(f"True: {CLASSES[c]}", fontsize=12)
        ax[1, c].imshow(img)
        hm = ax[1, c].imshow(show(maps[i], args.method), cmap="jet", alpha=0.55, vmin=0, vmax=1)
        ax[1, c].set_title(f"Pred: {CLASSES[y_pred[i]]}", fontsize=12, color="green" if y_pred[i] == c else "red")
        ax[0, c].axis("off")
        ax[1, c].axis("off")
    fig.colorbar(hm, ax=ax[1, :], fraction=0.015, pad=0.01, label="relevance (scaled)")
    fig.suptitle(f"{METHOD_NAME[args.method]} of {title} on the real-world validation set", fontsize=14)
    fig.savefig(os.path.join(out_dir, "maps.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # perturbation curves
    x_pct = STEPS * 100
    plt.figure(figsize=(7, 4.2))
    plt.plot(x_pct, res["LeRF_curve"], "o-", color="#1baf7a", lw=2, label=f"LeRF (score {res['LeRF']:.3f})")
    plt.plot(
        x_pct,
        res["random_curve"],
        "s--",
        color="#52514e",
        lw=1.5,
        label=f"Random order (MoRF {res['random_MoRF']:.3f})",
    )
    plt.plot(x_pct, res["MoRF_curve"], "o-", color="#2a78d6", lw=2, label=f"MoRF (score {res['MoRF']:.3f})")
    plt.fill_between(
        x_pct, res["MoRF_curve"], res["LeRF_curve"], color="#2a78d6", alpha=0.12, label=f"ABPC = {res['ABPC']:.3f}"
    )
    plt.xlabel("Pixels removed (%)")
    plt.ylabel("Confidence of predicted class")
    plt.ylim(0, 1.02)
    plt.xticks(x_pct)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.title(f"Perturbation curves: {title}, {METHOD_NAME[args.method]} (n={len(data)})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "perturbation_curves.png"), dpi=300)
    plt.close()

    # confusion matrix (identical for every method - for checking only)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        confusion_matrix(y_true, y_pred, labels=range(4)),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cbar=False,
    )
    plt.xlabel("Predicted gesture")
    plt.ylabel("True gesture")
    plt.title(f"Confusion matrix: {title} (n={len(data)})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=300)
    plt.close()
    return res


# ============================================================================ Part B
def frame_number(name: str) -> int:
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else 0


def red_square_box(img_np: np.ndarray) -> tuple[int, int, int, int] | None:
    """Fallback if the manifest has no entry: find the pure-red square by colour."""
    r, g, b = (img_np[..., i].astype(int) for i in range(3))
    m = (r > 200) & (g < 60) & (b < 60)
    if m.sum() < 100:
        return None
    ys, xs = np.nonzero(m)
    x_min, x_max, y_min, y_max = int(np.min(xs)), int(np.max(xs)), int(np.min(ys)), int(np.max(ys))
    return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1


def part_b(
    args: argparse.Namespace,
    model: torch.nn.Module,
    explain: Callable,
    device: torch.device,
    tf: Callable,
    out_dir: str,
    title: str,
) -> dict[str, Any]:
    manifest_path, up_dirs = None, []
    for root, _, files in os.walk(args.set3_dir):
        low = root.lower()
        if "spurious_patch_manifest.csv" in files and manifest_path is None:
            manifest_path = os.path.join(root, "spurious_patch_manifest.csv")
        if (
            os.path.basename(root).lower() == "up"
            and "swap" not in low
            and "preview" not in low
            and any(f.lower().endswith(IMG_EXT) for f in files)
        ):
            up_dirs.append(root)
    assert up_dirs, f"No 'Up' image folder inside {args.set3_dir}"
    up_dir = sorted(up_dirs, key=len)[0]

    manifest: dict[str, tuple[int, int, int, int]] = {}
    if manifest_path:
        with open(manifest_path, newline="") as f:
            for r in csv.DictReader(f):
                if r["gesture"].lower() == "up" and str(r["has_patch"]).strip().lower() in ("1", "true", "yes"):
                    px, py, pw, ph = (int(float(r[k])) for k in ("patch_x", "patch_y", "patch_w", "patch_h"))
                    manifest[os.path.basename(r["filename"])] = (px, py, pw, ph)

    # validation split = last frames (same split as in training)
    files = sorted((f for f in os.listdir(up_dir) if f.lower().endswith(IMG_EXT)), key=frame_number)
    files = files[-int(round(len(files) * SET3_VAL_FRACTION)) :]

    rows: list[dict[str, Any]] = []
    example: tuple[Any, ...] | None = None
    for name in files:
        img = Image.open(os.path.join(up_dir, name)).convert("RGB")
        W, H = img.size
        box = manifest.get(name) or red_square_box(np.array(img))
        if box is None:
            continue
        x = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(model(x).argmax(1))
        attr = explain(x, pred)

        mask = square_mask(box, W, H, IMG_SIZE)
        ys, xs = np.nonzero(mask)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1

        s = synscore(attr, mask)
        rows.append({"file": name, "pred": CLASSES[pred], **s})
        if example is None or s["synscore_percent"] > example[0]:
            example = (s["synscore_percent"], x, attr, (x0, y0, x1, y1), CLASSES[pred])

    assert rows, "No Set 3 'Up' image with a red square found."
    with open(os.path.join(out_dir, "synscore_per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    S = {
        k: np.array([r[k] for r in rows]) for k in ("synscore_percent", "chance_percent", "concentration", "AUC", "AP")
    }
    out: dict[str, Any] = {
        k: {"mean": float(v.mean()), "sd": float(v.std()), "max": float(v.max())} for k, v in S.items()
    }
    out["n_images"] = len(rows)
    out["up_predicted_correctly_percent"] = float(100 * np.mean([r["pred"] == "Up" for r in rows]))

    assert example is not None
    syn, x, attr, (x0, y0, x1, y1), pred_name = example
    img_np = to_image(x)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    ax[0].imshow(img_np)
    ax[0].set_title("Set 3 'Up' image")
    ax[1].imshow(img_np)
    hm = ax[1].imshow(show(attr, args.method), cmap="jet", alpha=0.55, vmin=0, vmax=1)
    for a in ax[:2]:
        a.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec="yellow", lw=1.5, ls="--"))
        a.axis("off")
    ax[1].set_title(f"{METHOD_NAME[args.method]} (Pred: {pred_name}), SynScore = {syn:.2f} %")
    fig.colorbar(hm, ax=ax[1], fraction=0.046, pad=0.02)
    sns.stripplot(y=S["synscore_percent"], color="black", alpha=0.6, jitter=True, ax=ax[2])
    sns.boxplot(y=S["synscore_percent"], color="#eb6834", width=0.35, fliersize=0, boxprops={"alpha": 0.5}, ax=ax[2])
    ax[2].axhline(
        S["chance_percent"].mean(),
        color="#52514e",
        ls="--",
        lw=1.2,
        label=f"chance level ({S['chance_percent'].mean():.2f} %)",
    )
    ax[2].set_ylim(bottom=0)
    ax[2].set_ylabel("Relevance inside the red square (%)")
    ax[2].set_title(
        f"SynScore, n = {len(rows)}\nmean {S['synscore_percent'].mean():.2f} % "
        f"({S['concentration'].mean():.1f}x chance)"
    )
    ax[2].legend(fontsize=9, loc="lower right")
    ax[2].grid(axis="y", alpha=0.3)
    fig.suptitle(f"SynScore of {METHOD_NAME[args.method]} for {title}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "synscore.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================================ main
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    random.seed(SEED)
    np.random.seed(SEED)  # noqa: NPY002  (seeds libraries that use the global NumPy RNG)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN.tolist(), STD.tolist()),
        ]
    )
    title = f"{DISPLAY_NAME[args.model]} (Set {args.set})"
    out_dir = os.path.join(args.out, f"{args.method}_{args.model}_set{args.set}")
    os.makedirs(out_dir, exist_ok=True)

    model = load_model(args.model, args.weights, device)
    explain = make_explainer(args.method, model, args.model)

    res = {"model": args.model, "set": args.set, "method": args.method}
    res.update(part_a(args, model, explain, device, tf, out_dir, title))
    if args.set == 3:
        if args.set3_dir:
            res["synscore"] = part_b(args, model, explain, device, tf, out_dir, title)
        else:
            print("No --set3-dir given: SynScore skipped.")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)

    print("=" * 60)
    print(f"{METHOD_NAME[args.method]} | {title}")
    print(f"Accuracy    : {res['accuracy_percent']:.2f} %")
    for k in ("MoRF", "LeRF", "ABPC", "ABPC_sd", "random_MoRF"):
        print(f"{k:<12}: {res[k]:.4f}")
    if "synscore" in res:
        s = res["synscore"]
        print(
            f"SynScore    : {s['synscore_percent']['mean']:.2f} % +- {s['synscore_percent']['sd']:.2f} % "
            f"({s['concentration']['mean']:.2f} x chance)"
        )
        print(f"AUC-ROC / AP: {s['AUC']['mean']:.3f} / {s['AP']['mean']:.3f}")
    print("Saved in", out_dir)


if __name__ == "__main__":
    main()
