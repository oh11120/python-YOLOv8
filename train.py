from __future__ import annotations

import argparse
from ultralytics import YOLO
from ultralytics.nn import modules as ulm

from models.custom_modules import EMA, DCNv3, WConcat, BiFPN


# Register custom modules so Ultralytics can resolve them by name in YAML
ulm.EMA = EMA
ulm.DCNv3 = DCNv3
ulm.WConcat = WConcat
ulm.BiFPN = BiFPN


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
        close_mosaic=10,
    )


if __name__ == "__main__":
    main()
