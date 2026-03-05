# 鲜花检测系统（YOLOv8 + EMA + DCNv3 + 加权融合）

本项目实现开题报告中的鲜花识别/检测系统，包含改进模型、训练评估、量化导出与桌面端推理。

## 项目结构
- `models/yolov8_flower.yaml`：模型结构定义
- `models/custom_modules.py`：EMA、DCNv3、BiFPN、加权融合模块
- `data/flower.yaml`：数据集配置（本地相对路径）
- `train.py`：训练入口
- `eval.py`：评估与 PR 曲线输出
- `export.py`：导出 ONNX/TorchScript
- `export_int8.py`：INT8 量化导出（ONNX QDQ / TensorRT engine）
- `infer_backend.py`：统一推理入口（.pt/.onnx/.engine）
- `scripts/validate_labels.py`：标签校验
- `scripts/split_dataset.py`：7:2:1 数据集划分（类别均衡）
- `scripts/class_balance.py`：类别分布统计
- `app.py`：PyQt5 桌面端推理 GUI

## 快速开始

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 标签校验
```bash
python3 scripts/validate_labels.py --labels data/yolo-flower/labels --nc 102
```

3. 训练
```bash
python3 train.py --epochs 200 --batch 16 --imgsz 640 --device 0
```

4. 评估（mAP + PR 曲线）
```bash
python3 eval.py --weights runs/flower/yolov8n_ema_dcn_wconcat/weights/best.pt --data data/flower.yaml
```

5. 导出 ONNX
```bash
python3 export.py --weights runs/flower/yolov8n_ema_dcn_wconcat/weights/best.pt --format onnx
```

6. INT8 量化
```bash
python3 export_int8.py --weights runs/flower/yolov8n_ema_dcn_wconcat/weights/best.pt --data data/flower.yaml --format onnx-int8
```

7. GUI 推理
```bash
python3 app.py --weights runs/flower/yolov8n_ema_dcn_wconcat/weights/best.pt --device 0
```

## 数据集划分与类别分布

7:2:1 划分（类别均衡）：
```bash
python3 scripts/split_dataset.py --images-root data/yolo-flower/images --labels-root data/yolo-flower/labels --out-root data/yolo-flower-split
```

类别分布统计：
```bash
python3 scripts/class_balance.py --root data/yolo-flower-split --out balance.csv
```

## GUI 功能
- 单图推理
- 批量文件夹推理（递归）
- 摄像头实时推理（可调 FPS/分辨率）
- CSV/Excel 结果导出（含统计汇总）
- 模型选择、后端选择、历史记录浏览
- 显式 NMS（可开关）

## 说明
- `DCNv3` 若不可用会自动回退为普通卷积
- GUI 支持后端选择（PyTorch / ONNX Runtime CPU/GPU / TensorRT engine）
- ONNX Runtime GPU 需要安装 `onnxruntime-gpu` 并确保 CUDA 可用
