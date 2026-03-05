from __future__ import annotations

import argparse
import ultralytics.nn.tasks as _tasks
from ultralytics import YOLO

from models.custom_modules import EMA, DCNv3, WConcat, BiFPN

_tasks.EMA = EMA
_tasks.DCNv3 = DCNv3
_tasks.WConcat = WConcat
_tasks.BiFPN = BiFPN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate model and export metrics/plots")
    p.add_argument("--weights", required=True, help="path to weights")
    p.add_argument("--data", default="data/flower.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--save-json", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        save_json=args.save_json,
        plots=True,
    )


if __name__ == "__main__":
    main()
