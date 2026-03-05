# 训练文档

本文档说明如何训练本项目的鲜花检测模型。

## 1. 环境准备

建议 Python 版本：3.9+（3.10/3.11 均可）

安装依赖：
```bash
pip install -r requirements.txt
```

## 2. 数据集检查

数据集目录结构应为：
- `data/yolo-flower/images/{train,val,test}`
- `data/yolo-flower/labels/{train,val,test}`

验证 YOLO 标签格式：
```bash
python3 scripts/validate_labels.py --labels data/yolo-flower/labels --nc 102
```

## 3. 可选：重新按 7:2:1 划分

如需重新划分数据集（带类分布均衡）：
```bash
python3 scripts/split_dataset.py \
  --images-root data/yolo-flower/images \
  --labels-root data/yolo-flower/labels \
  --out-root data/yolo-flower-split
```

会生成新的配置文件：`data/yolo-flower-split/data.yaml`
如果存在 `classes.txt`，会自动写入 `nc` 与 `names`。
也可以手动指定：`--classes path/to/classes.txt`

查看类别分布：
```bash
python3 scripts/class_balance.py --root data/yolo-flower-split --out balance.csv
```

## 4. 选择数据配置

默认使用：
- `data/flower.yaml`

如果你执行了重新划分，则使用：
- `data/yolo-flower-split/data.yaml`

## 5. 开始训练

默认训练命令：
```bash
python3 train.py \
  --model models/yolov8_flower.yaml \
  --data data/flower.yaml \
  --imgsz 640 \
  --epochs 200 \
  --batch 16 \
  --device 0
```

说明：
- `--device 0` 使用第一块 GPU，无 GPU 时可用 `--device cpu`
- 显存不足时可调小 `--batch` 或 `--imgsz`

继续训练：
```bash
python3 train.py --resume
```

## 6. 评估与 PR 曲线

```bash
python3 eval.py \
  --weights runs/flower/yolov8n_ema_dcn_wconcat/weights/best.pt \
  --data data/flower.yaml
```

会输出评估指标与 PR 曲线。

## 7. 模型导出

导出 ONNX：
```bash
python3 export.py --weights runs/flower/yolov8n_ema_dcn_wconcat/weights/best.pt --format onnx
```

INT8 量化（ONNX Runtime）：
```bash
python3 export_int8.py \
  --weights runs/flower/yolov8n_ema_dcn_wconcat/weights/best.pt \
  --data data/flower.yaml \
  --format onnx-int8
```

TensorRT INT8（需安装 TensorRT）：
```bash
python3 export_int8.py \
  --weights runs/flower/yolov8n_ema_dcn_wconcat/weights/best.pt \
  --data data/flower.yaml \
  --format engine
```

## 8. 推理

单张图片推理（支持 .pt/.onnx/.engine）：
```bash
python3 infer_backend.py --weights runs/flower/yolov8n_ema_dcn_wconcat/weights/best.pt --source path/to/image.jpg
```

## 9. GUI 推理

```bash
python3 app.py --weights runs/flower/yolov8n_ema_dcn_wconcat/weights/best.pt --device 0
```

GUI 功能：
- 单图推理
- 批量文件夹推理（递归）
- 摄像头实时推理（可调 FPS/分辨率）
- CSV/Excel 结果导出（含统计汇总）
- 模型选择、后端选择与历史记录浏览

## 常见问题

- CUDA 显存不足：
  - 降低 `--batch`（如 8、4）
  - 降低 `--imgsz`（如 512）
- `DCNv3` 不可用时会自动回退为普通卷积
- TensorRT 导出失败可改用 ONNX INT8
- ONNX Runtime GPU 需要安装 `onnxruntime-gpu` 并确保 CUDA 可用
