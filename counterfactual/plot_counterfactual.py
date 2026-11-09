"""Plot the counterfactual test of the red square (thesis Section 3.10, Figure 91).

Reads ``results/counterfactual/counterfactual_extra.csv`` and writes
``results/figures/counterfactual_square_test.png``:

* left:  logit margin for 'Up' in Test A (Set 3 'Up' images without / with the square),
* right: mean P(Up) in Test D (noise backgrounds without a hand, without / with the square).

The models trained on Set 2 never saw the square and serve as the control group.

Usage::

    python counterfactual/plot_counterfactual.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "results" / "counterfactual" / "counterfactual_extra.csv"
DEFAULT_OUT = ROOT / "results" / "figures" / "counterfactual_square_test.png"

SHORT = {"SimpleCNN": "SimpleCNN", "MobileNet-V3-Small": "MobileNet-V3", "SqueezeNet 1.1": "SqueezeNet"}


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot(rows: list[dict[str, str]], out: Path) -> None:
    labels = [f"{SHORT.get(r['model'], r['model'])}\nSet {r['set']}" for r in rows]
    x = np.arange(len(rows))
    w = 0.38

    a_without = np.array([float(r["A_margin_without"]) for r in rows])
    a_with = np.array([float(r["A_margin_with"]) for r in rows])
    d_without = np.array([float(r["D_P_up_without"]) for r in rows])
    d_with = np.array([float(r["D_P_up_with"]) for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    ax1.bar(x - w / 2, a_without, w, label="without square", color="#9bbbd9")
    ax1.bar(x + w / 2, a_with, w, label="with square", color="#c0392b")
    for i, r in enumerate(rows):
        change = float(r["A_margin_change"])
        ax1.annotate(f"{change:+.2f}", (x[i], max(a_without[i], a_with[i]) + 0.6), ha="center", fontsize=9)
    ax1.set_xticks(x, labels, fontsize=8.5)
    ax1.set_ylabel("Logit margin for 'Up'")
    ax1.set_title("Test A: Set 3 'Up' images (n = 40)")
    ax1.set_ylim(0, max(a_without.max(), a_with.max()) + 3)
    ax1.legend(frameon=False, fontsize=9)

    ax2.bar(x - w / 2, d_without, w, label="without square", color="#9bbbd9")
    ax2.bar(x + w / 2, d_with, w, label="with square", color="#c0392b")
    ax2.set_xticks(x, labels, fontsize=8.5)
    ax2.set_ylabel("Mean P(Up)")
    ax2.set_ylim(0, 1.08)
    ax2.set_title("Test D: noise backgrounds without a hand (n = 100)")
    ax2.legend(frameon=False, fontsize=9, loc="upper right")

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot(load(args.csv), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
