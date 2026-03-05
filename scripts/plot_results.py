from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    results_csv = Path("runs/detect/train/results.csv")
    if not results_csv.exists():
        raise SystemExit(f"results.csv not found: {results_csv}")

    rows: list[dict[str, str]] = []
    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise SystemExit("results.csv is empty")

    def col(name: str) -> list[float]:
        return [float(r[name]) for r in rows if r.get(name) not in (None, "", "nan")]

    epochs = col("epoch")
    box_loss = col("train/box_loss")
    cls_loss = col("train/cls_loss")
    dfl_loss = col("train/dfl_loss")
    map50 = col("metrics/mAP50(B)")
    map5095 = col("metrics/mAP50-95(B)")

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), dpi=150)
    ax1, ax2 = axes

    ax1.plot(epochs[: len(box_loss)], box_loss, label="train/box_loss")
    ax1.plot(epochs[: len(cls_loss)], cls_loss, label="train/cls_loss")
    ax1.plot(epochs[: len(dfl_loss)], dfl_loss, label="train/dfl_loss")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs[: len(map50)], map50, label="metrics/mAP50(B)")
    ax2.plot(epochs[: len(map5095)], map5095, label="metrics/mAP50-95(B)")
    ax2.set_title("Validation mAP")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("mAP")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    out_path = results_csv.with_name("results.png")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
