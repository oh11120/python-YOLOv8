from __future__ import annotations

import argparse
from ultralytics import YOLO
from ultralytics.nn import modules as ulm

from models.custom_modules import EMA, DCNv3, WConcat, BiFPN

ulm.EMA = EMA
ulm.DCNv3 = DCNv3
ulm.WConcat = WConcat
ulm.BiFPN = BiFPN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出训练好的 YOLOv8 模型")
    parser.add_argument("--weights", required=True, help=".pt 权重路径")
    parser.add_argument("--format", default="onnx", choices=["onnx", "torchscript"], help="导出格式")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    model.export(format=args.format, imgsz=args.imgsz, dynamic=False, simplify=True)


if __name__ == "__main__":
    main()
