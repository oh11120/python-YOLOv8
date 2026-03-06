"""Ablation study: train all variants and print a comparison table.

Variants
--------
baseline : Standard YOLOv8n — no custom modules
+EMA     : YOLOv8n + EMA attention (Conv downsampling, standard FPN)
+DCNv3   : YOLOv8n + DCNv3 downsampling (no EMA, standard FPN)
+BiFPN   : YOLOv8n + BiFPN neck (Conv downsampling, no EMA)
Ours     : YOLOv8n + EMA + DCNv3 + BiFPN (full model)

Usage
-----
    python ablation.py                        # train all with default settings
    python ablation.py --epochs 100           # specify epoch count
    python ablation.py --skip-train           # skip training, just show results
    python ablation.py --variants baseline +EMA   # train specific variants
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import inspect
import re
import ultralytics.nn.tasks as _tasks
from ultralytics import YOLO

from models.custom_modules import EMA, DCNv3, WConcat, WeightedSum, BiFPN

# Register custom modules into Ultralytics' parse_model namespace
_tasks.EMA = EMA
_tasks.DCNv3 = DCNv3
_tasks.WConcat = WConcat
_tasks.WeightedSum = WeightedSum
_tasks.BiFPN = BiFPN


def _patch_parse_model() -> None:
    """Inject DCNv3 handling into parse_model's if/elif/else chain.

    Without this patch parse_model's else-branch calls DCNv3(*yaml_args)
    directly, which maps YAML args[0]=c2 into c1, corrupting the channel
    count.  The patch mirrors the Conv-like handling so c1 is prepended.
    """
    src = inspect.getsource(_tasks.parse_model)
    match = re.search(r"\n([ \t]+)(else:)[ \t]*\n([ \t]+)(c2 = ch\[f\])", src)
    if not match:
        import warnings as _w
        _w.warn("parse_model DCNv3 patch: pattern not found — channel tracking may be incorrect")
        return
    outer_indent = match.group(1)
    inner_indent = match.group(3)
    elif_block = (
        f"\n{outer_indent}elif m is DCNv3:\n"
        f"{inner_indent}c1, c2 = ch[f], args[0]\n"
        f"{inner_indent}try:\n"
        f"{inner_indent}    if c2 != nc:\n"
        f"{inner_indent}        c2 = make_divisible(c2 * width, ch_mul)\n"
        f"{inner_indent}except Exception:\n"
        f"{inner_indent}    pass\n"
        f"{inner_indent}args = [c1, c2, *args[1:]]"
    )
    old_str = match.group(0)
    patched = src.replace(old_str, elif_block + old_str, 1)
    exec(compile(patched, inspect.getfile(_tasks), "exec"), _tasks.__dict__)


_patch_parse_model()


# ---------------------------------------------------------------------------
# Ablation variant definitions
# ---------------------------------------------------------------------------
VARIANTS: list[dict] = [
    {
        "name": "baseline",
        "label": "Baseline (YOLOv8n)",
        "model": "models/yolov8_ablation_baseline.yaml",
        "uses_custom": False,
    },
    {
        "name": "ema",
        "label": "+EMA",
        "model": "models/yolov8_ablation_ema.yaml",
        "uses_custom": True,
    },
    {
        "name": "dcnv3",
        "label": "+DCNv3",
        "model": "models/yolov8_ablation_dcnv3.yaml",
        "uses_custom": True,
    },
    {
        "name": "bifpn",
        "label": "+BiFPN",
        "model": "models/yolov8_ablation_bifpn.yaml",
        "uses_custom": True,
    },
    {
        "name": "ours",
        "label": "Ours (EMA+DCNv3+BiFPN)",
        "model": "models/yolov8_flower.yaml",
        "uses_custom": True,
    },
]

VARIANT_NAMES = [v["name"] for v in VARIANTS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run YOLOv8 flower ablation study")
    p.add_argument("--data", default="data/flower.yaml", help="dataset YAML")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=100,
                   help="training epochs per variant (default: 100)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", default="runs/ablation",
                   help="root directory for all ablation runs")
    p.add_argument("--skip-train", action="store_true",
                   help="skip training and only print the results table")
    p.add_argument("--variants", nargs="+", choices=VARIANT_NAMES,
                   default=None,
                   help="subset of variants to train (default: all)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_variant(variant: dict, args: argparse.Namespace) -> None:
    print(f"\n{'='*60}")
    print(f"  Training: {variant['label']}")
    print(f"  Model:    {variant['model']}")
    print(f"  Epochs:   {args.epochs}")
    print(f"{'='*60}\n")

    model = YOLO(variant["model"])
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=variant["name"],
        exist_ok=True,
        optimizer="SGD",
        cos_lr=True,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        erasing=0.2,
        close_mosaic=10,
    )


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------
def _read_best_metrics(run_dir: Path) -> dict[str, float | None]:
    """Return best mAP50, mAP50-95, precision, recall from results.csv."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {"mAP50": None, "mAP50-95": None, "P": None, "R": None}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {"mAP50": None, "mAP50-95": None, "P": None, "R": None}

    # Column names in Ultralytics results.csv (strip whitespace)
    def _get(row: dict, *keys: str) -> float | None:
        for k in keys:
            for col in row:
                if col.strip().lower() == k.lower():
                    try:
                        return float(row[col])
                    except (ValueError, TypeError):
                        pass
        return None

    # Find row with best mAP50
    best_row = max(rows, key=lambda r: _get(r, "metrics/mAP50(B)") or 0.0)
    return {
        "mAP50":    _get(best_row, "metrics/mAP50(B)", "mAP50"),
        "mAP50-95": _get(best_row, "metrics/mAP50-95(B)", "mAP50-95"),
        "P":        _get(best_row, "metrics/precision(B)", "precision"),
        "R":        _get(best_row, "metrics/recall(B)", "recall"),
    }


def _count_params(weight_path: Path) -> str:
    """Return parameter count as a human-readable string."""
    try:
        import torch
        ckpt = torch.load(weight_path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        total = sum(p.numel() for p in state.values() if hasattr(p, "numel"))
        return f"{total / 1e6:.2f}M"
    except Exception:
        return "N/A"


def print_results_table(project: str, variants_to_show: list[dict]) -> None:
    project_dir = Path(project)

    col_w = [26, 8, 10, 8, 8, 9]
    header = ["Variant", "P", "R", "mAP50", "mAP50-95", "Params"]
    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"

    print(f"\n{'='*70}")
    print("  Ablation Study Results")
    print(f"{'='*70}")
    print(sep)
    print(fmt.format(*header))
    print(sep)

    for v in variants_to_show:
        run_dir = project_dir / v["name"]
        metrics = _read_best_metrics(run_dir)

        weight_path = run_dir / "weights" / "best.pt"
        params_str = _count_params(weight_path) if weight_path.exists() else "N/A"

        def fmt_val(x: float | None) -> str:
            return f"{x:.4f}" if x is not None else "—"

        row = [
            v["label"],
            fmt_val(metrics["P"]),
            fmt_val(metrics["R"]),
            fmt_val(metrics["mAP50"]),
            fmt_val(metrics["mAP50-95"]),
            params_str,
        ]
        print(fmt.format(*row))

    print(sep)
    print()

    # Improvement over baseline
    baseline_dir = project_dir / "baseline"
    baseline_metrics = _read_best_metrics(baseline_dir)
    base_map50 = baseline_metrics["mAP50"]
    if base_map50 is not None:
        print("  mAP50 gain vs Baseline:")
        for v in variants_to_show[1:]:
            run_dir = project_dir / v["name"]
            m = _read_best_metrics(run_dir)
            if m["mAP50"] is not None:
                delta = m["mAP50"] - base_map50
                sign = "+" if delta >= 0 else ""
                print(f"    {v['label']:<26} {sign}{delta:.4f}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Filter requested variants
    if args.variants:
        selected = [v for v in VARIANTS if v["name"] in args.variants]
    else:
        selected = VARIANTS

    if not args.skip_train:
        for v in selected:
            if not Path(v["model"]).exists():
                print(f"[ERROR] Model YAML not found: {v['model']}")
                sys.exit(1)
            train_variant(v, args)

    # Always print summary
    print_results_table(args.project, VARIANTS)


if __name__ == "__main__":
    main()
