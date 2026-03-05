from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.nn import modules as ulm

from models.custom_modules import EMA, DCNv3, WConcat, BiFPN

ulm.EMA = EMA
ulm.DCNv3 = DCNv3
ulm.WConcat = WConcat
ulm.BiFPN = BiFPN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="使用 PyTorch/ONNX/TensorRT 后端进行推理")
    p.add_argument("--weights", required=True, help="模型权重文件：.pt/.onnx/.engine")
    p.add_argument("--source", required=True, help="输入图片路径")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--device", default="0")
    p.add_argument("--out", default="result.jpg")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )
    res = results[0]
    plotted = res.plot()
    cv2.imwrite(args.out, plotted)
    print(f"已保存: {args.out}")


if __name__ == "__main__":
    main()
