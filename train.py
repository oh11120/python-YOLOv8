from __future__ import annotations

import argparse
import inspect
import re
import ultralytics.nn.tasks as _tasks
from ultralytics import YOLO

from models.custom_modules import EMA, DCNv3, WConcat, WeightedSum

# Register custom modules into Ultralytics' parse_model namespace
_tasks.EMA = EMA
_tasks.DCNv3 = DCNv3
_tasks.WConcat = WConcat
_tasks.WeightedSum = WeightedSum


def _patch_parse_model() -> None:
    """Inject DCNv3 handling into parse_model's if/elif/else chain.

    parse_model's else-branch records c2 = ch[f] (= input channels) for
    unknown custom modules, but DCNv3 is a strided conv that changes channel
    count.  We inject an elif that mirrors the Conv-like handling so that:
      - c1 = ch[f]                          (actual input channels)
      - c2 = make_divisible(args[0]*width)  (scaled output channels)
      - args = [c1, c2, *args[1:]]          (prepend c1 for correct init)
    """
    src = inspect.getsource(_tasks.parse_model)

    # Locate the catch-all else branch: `else:\n<indent>c2 = ch[f]`
    match = re.search(r"\n([ \t]+)(else:)[ \t]*\n([ \t]+)(c2 = ch\[f\])", src)
    if not match:
        import warnings as _w
        _w.warn(
            "parse_model DCNv3 patch: pattern not found — "
            "channel tracking for DCNv3 will be incorrect"
        )
        return

    outer_indent = match.group(1)
    inner_indent = match.group(3)

    elif_block = (
        f"\n{outer_indent}elif m is EMA:\n"
        f"{inner_indent}c2 = ch[f[0] if isinstance(f, list) else f]\n"
        f"{inner_indent}args = [c2]\n"
        f"{outer_indent}elif m is DCNv3:\n"
        f"{inner_indent}c1, c2 = ch[f], args[0]\n"
        f"{inner_indent}try:\n"
        f"{inner_indent}    if c2 != nc:\n"
        f"{inner_indent}        c2 = make_divisible(c2 * width, ch_mul)\n"
        f"{inner_indent}except Exception:\n"
        f"{inner_indent}    pass\n"
        f"{inner_indent}args = [c1, c2, *args[1:]]\n"
        f"{outer_indent}elif m is WeightedSum:\n"
        f"{inner_indent}c2 = ch[f[0] if isinstance(f, list) else f]"
    )

    old_str = match.group(0)
    patched = src.replace(old_str, elif_block + old_str, 1)
    exec(compile(patched, inspect.getfile(_tasks), "exec"), _tasks.__dict__)


_patch_parse_model()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 flower detector with custom modules")
    parser.add_argument("--model", default="models/yolov8_flower.yaml", help="model YAML path")
    parser.add_argument("--data", default="data/flower.yaml", help="dataset YAML path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/flower")
    parser.add_argument("--name", default="yolov8n_ema_dcn_wconcat")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
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
        close_mosaic=20,
    )


if __name__ == "__main__":
    main()
