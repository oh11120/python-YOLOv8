from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from ultralytics import YOLO
from ultralytics.nn import modules as ulm

from models.custom_modules import EMA, DCNv3, WConcat, BiFPN

ulm.EMA = EMA
ulm.DCNv3 = DCNv3
ulm.WConcat = WConcat
ulm.BiFPN = BiFPN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv8 训练后 INT8 量化导出")
    p.add_argument("--weights", required=True, help=".pt 权重路径")
    p.add_argument("--data", required=True, help="用于校准的数据集 YAML")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--calib", type=int, default=200, help="校准图片数量")
    p.add_argument("--format", choices=["engine", "onnx-int8"], default="onnx-int8")
    p.add_argument("--out", default="", help="onnx-int8 输出路径")
    return p.parse_args()


class YoloCalibReader(CalibrationDataReader):
    def __init__(self, image_paths: list[Path], imgsz: int):
        self.image_paths = image_paths
        self.imgsz = imgsz
        self._iter = iter(self.image_paths)

    def get_next(self):
        try:
            path = next(self._iter)
        except StopIteration:
            return None
        img = cv2.imread(str(path))
        if img is None:
            return self.get_next()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        return {"images": img}


def collect_images(data_yaml: str, limit: int) -> list[Path]:
    from ultralytics.data.utils import yaml_load

    data = yaml_load(data_yaml)
    root = Path(data.get("path", "."))
    train = data.get("train", "")
    train_path = (root / train).resolve() if not os.path.isabs(train) else Path(train)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in train_path.rglob("*") if p.suffix.lower() in exts]
    return images[:limit]


def export_engine(weights: str, data: str, imgsz: int) -> None:
    model = YOLO(weights)
    model.export(format="engine", int8=True, data=data, imgsz=imgsz, dynamic=False, simplify=True)


def export_onnx_int8(weights: str, data: str, imgsz: int, calib: int, out: str) -> None:
    model = YOLO(weights)
    onnx_path = Path(weights).with_suffix(".onnx")
    model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True)

    images = collect_images(data, calib)
    if not images:
        raise RuntimeError("未找到用于校准的图片")

    reader = YoloCalibReader(images, imgsz)
    out_path = Path(out) if out else onnx_path.with_name(onnx_path.stem + "_int8.onnx")

    quantize_static(
        model_input=str(onnx_path),
        model_output=str(out_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        optimize_model=True,
    )

    # basic check
    onnx.load(str(out_path))


def main() -> None:
    args = parse_args()
    if args.format == "engine":
        export_engine(args.weights, args.data, args.imgsz)
    else:
        export_onnx_int8(args.weights, args.data, args.imgsz, args.calib, args.out)


if __name__ == "__main__":
    main()
