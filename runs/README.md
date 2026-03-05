# Runs 目录说明

`runs/` 是 Ultralytics/YOLO 的默认输出目录。每次训练/验证/推理都会在这里生成新的子目录，保存权重、指标、图表和日志。

## 常见结构

- `runs/detect/` 检测任务输出
- `runs/segment/` 分割任务输出
- `runs/classify/` 分类任务输出
- `runs/pose/` 姿态估计输出
- `runs/obb/` 旋转框检测输出

## 一次训练的典型目录（如 `runs/detect/train/`）

- `weights/best.pt` 验证指标最好的权重
- `weights/last.pt` 最后一个 epoch 的权重
- `results.csv` 每个 epoch 的指标（loss、mAP 等）
- `results.png` 训练曲线图
- `labels.jpg` 训练集标注可视化
- `args.yaml` 本次训练的全部参数
- `metrics_legend.txt` 指标解释（自定义添加）
- `args_legend.txt` 参数列表（自定义添加）

## train 目录完整文件说明（以 `runs/detect/train/` 为例）

以下文件通常在训练结束后出现（不同版本可能略有差异）：

- `args.yaml`：训练参数配置文件，完整记录本次训练的参数。
- `args_legend.txt`：参数列表（我们自定义追加的纯文本版，便于快速查看）。
- `metrics_legend.txt`：指标解释（我们自定义追加的纯文本版）。
- `results.csv`：每个 epoch 的训练/验证指标数据（loss、mAP 等）。
- `results.png`：由 `results.csv` 生成的训练曲线图（loss/mAP 趋势）。
- `labels.jpg`：标注分布图，用于检查类别分布与框位置/尺寸。
- `train_batch0.jpg` / `train_batch1.jpg` / `train_batch2.jpg`：训练初期的 batch 可视化（含增强与标注框）。
- `weights/best.pt`：验证集指标最好的权重。
- `weights/last.pt`：最后一个 epoch 的权重。

如果你用 `train_with_log.bat` 启动训练，还会在 `runs/detect/` 下看到：

- `train_YYYYMMDD_HHMMSS.log`：完整控制台输出日志。

## train 目录“全部文件”详细说明（与当前目录一致）

**权重文件**
- `weights/best.pt`：验证集指标最好的模型权重。
- `weights/last.pt`：最后一个 epoch 的模型权重。

**参数与指标**
- `args.yaml`：训练参数配置（完整参数）。
- `args_legend.txt`：参数列表（纯文本版）。
- `metrics_legend.txt`：指标解释（纯文本版）。
- `results.csv`：每个 epoch 的训练/验证指标数据。
- `results.png`：训练曲线图（loss 与 mAP 曲线）。

**数据分布与诊断图**
- `labels.jpg`：标注分布统计图（类别分布、框位置、框大小）。
- `confusion_matrix.png`：混淆矩阵（未归一化）。
- `confusion_matrix_normalized.png`：混淆矩阵（归一化）。

**PR / P / R / F1 曲线**
- `BoxPR_curve.png`：Precision-Recall 曲线。
- `BoxP_curve.png`：Precision 曲线。
- `BoxR_curve.png`：Recall 曲线。
- `BoxF1_curve.png`：F1 曲线。

**训练样本可视化**
- `train_batch0.jpg` / `train_batch1.jpg` / `train_batch2.jpg`：训练初期保存的样本（含增强与标注框）。
- `train_batchXXXXXX.jpg`：训练过程中的随机 batch 可视化样本（数字为 batch 索引）。

**验证集可视化**
- `val_batch0_labels.jpg` / `val_batch1_labels.jpg` / `val_batch2_labels.jpg`：验证集标注可视化（真值框）。
- `val_batch0_pred.jpg` / `val_batch1_pred.jpg` / `val_batch2_pred.jpg`：验证集预测可视化（模型预测框）。

## 日志文件

如果用 `train_with_log.bat` 启动训练，完整控制台输出会保存为：

- `runs/detect/train_YYYYMMDD_HHMMSS.log`

## results.csv 常见列含义（检测任务）

不同版本/任务可能列名略有差异，常见字段如下：

- `epoch`：当前 epoch 编号（从 1 开始）
- `time`：训练累计时间（秒）
- `train/box_loss`：训练集边框回归损失（越小越好）
- `train/cls_loss`：训练集分类损失（越小越好）
- `train/dfl_loss`：训练集分布式回归损失（越小越好）
- `metrics/precision(B)`：验证集精确率（越大越好）
- `metrics/recall(B)`：验证集召回率（越大越好）
- `metrics/mAP50(B)`：验证集 mAP@0.50（越大越好）
- `metrics/mAP50-95(B)`：验证集 mAP@0.50:0.95（越大越好）

### 更详细解释（常见指标）

- `epoch`：训练轮次。每个 epoch 训练一次完整的训练集。
- `time`：从训练开始到当前 epoch 结束的累计秒数。
- `train/box_loss`：预测框与标注框的几何误差。一般包含 L1/GIoU/CIoU 等成分，越小越好。
- `train/cls_loss`：类别预测的误差（多类别交叉熵或等价形式），越小越好。
- `train/dfl_loss`：YOLOv8 的分布式回归损失（Distribution Focal Loss），用于更精细的边框回归，越小越好。
- `metrics/precision(B)`：精确率 = TP / (TP + FP)。预测为正的框中有多少是真正的目标。
- `metrics/recall(B)`：召回率 = TP / (TP + FN)。真实目标中有多少被预测出来。
- `metrics/mAP50(B)`：在 IoU=0.50 时的平均精度，衡量检测质量。
- `metrics/mAP50-95(B)`：在 IoU=0.50 到 0.95（步长 0.05）上的平均 mAP，更严格、也更重要。

## 训练/验证/推理各自会生成的文件

- 训练（`yolo train`）：
  - `weights/best.pt`、`weights/last.pt`
  - `results.csv`、`results.png`
  - `labels.jpg`、`args.yaml`
  - `metrics_legend.txt`、`args_legend.txt`（自定义）
- 验证（`yolo val`）：
  - 通常会生成 `results.csv` 或在控制台输出评估结果
  - 可能生成 `confusion_matrix.png`、`pr_curve.png` 等图表（视配置）
- 推理（`yolo predict`）：
  - 默认保存预测图片/视频到 `runs/detect/predict*`
  - 如果开启 `save_txt` 会保存预测结果文本

## 常见参数详细解释（你日志里出现的所有参数）

以下按类别归纳，并解释作用与常见影响：

**训练流程与设备**
- `device`：使用的设备，如 `0` 表示第一块 GPU，`cpu` 表示使用 CPU。
- `workers`：DataLoader 线程数。Windows 下过高可能导致不稳定。
- `amp`：自动混合精度，通常提高速度、降低显存占用。
- `deterministic`：是否强制确定性，开启会更可复现但可能更慢且有警告。
- `seed`：随机种子，用于可复现训练。
- `resume`：是否从上次中断继续训练。
- `time`：按小时限制训练时间，达到后自动停止。
- `patience`：早停耐心，验证指标不提升的容忍轮数。

**训练轮次与批量**
- `epochs`：训练总轮次。
- `batch`：批大小。显存不足时需减小。
- `nbs`：基准批大小，用于自动调整学习率与权重衰减。

**优化器与学习率**
- `optimizer`：优化器类型（如 `auto`、`SGD`、`AdamW`）。
- `lr0`：初始学习率。
- `lrf`：最终学习率系数（学习率衰减到 `lr0 * lrf`）。
- `momentum`：动量参数（SGD 类优化器）。
- `weight_decay`：权重衰减，防止过拟合。
- `warmup_epochs`：预热轮次，使学习率从低逐渐升高。
- `warmup_momentum`：预热阶段动量。
- `warmup_bias_lr`：预热阶段 bias 学习率。
- `cos_lr`：是否使用余弦学习率。

**输入与数据**
- `imgsz`：输入图片尺寸。
- `data`：数据集配置文件路径。
- `cache`：是否缓存数据，提高读取速度。
- `fraction`：使用训练集的比例（1.0 表示全部）。
- `rect`：矩形训练，减少填充但可能影响数据增强。

**增强相关**
- `mosaic`：马赛克增强强度（1.0 表示启用）。
- `close_mosaic`：在最后多少轮关闭马赛克。
- `mixup`：MixUp 增强强度。
- `copy_paste` / `copy_paste_mode`：拷贝粘贴增强及其模式。
- `auto_augment`：自动增强策略（如 `randaugment`）。
- `erasing`：随机擦除强度。
- `degrees` / `translate` / `scale` / `shear` / `perspective`：几何增强参数。
- `flipud` / `fliplr`：上下/左右翻转概率。
- `hsv_h` / `hsv_s` / `hsv_v`：颜色增强幅度。
- `bgr`：颜色通道扰动。

**检测任务相关**
- `box`：边框损失权重。
- `cls`：分类损失权重。
- `dfl`：DFL 损失权重。
- `iou`：NMS 或损失相关阈值（不同上下文略有差异）。
- `max_det`：每张图最大检测数。
- `conf`：置信度阈值（未指定通常自动）。
- `nms`：是否启用 NMS（某些模型可能用端到端）。
- `agnostic_nms`：是否类别无关 NMS。

**验证/可视化**
- `val`：是否在训练过程中进行验证。
- `plots`：是否保存训练曲线/可视化图。
- `save`：是否保存权重。
- `save_period`：每隔多少轮保存一次权重。
- `save_txt` / `save_conf` / `save_crop`：保存预测结果、置信度、裁剪图。
- `show` / `show_boxes` / `show_labels` / `show_conf`：显示相关开关。
- `line_width`：绘图线宽。

**输出与工程化**
- `project` / `name`：输出目录与实验名。
- `exist_ok`：同名目录是否允许覆盖。
- `format` / `opset` / `simplify`：导出相关参数。
- `compile`：是否使用 Torch 编译加速。
- `dynamic` / `int8` / `half`：导出或推理精度相关开关。
- `tracker`：跟踪器配置文件（推理/跟踪时用）。

## 自定义输出目录与名称

你可以通过 `project=` 与 `name=` 指定输出目录，例如：

```
yolo train model=models/yolov8_flower.yaml data=data/flower.yaml project=runs/custom name=exp01
```

日志保存也可用 `train_with_log.bat`，日志会在 `runs/detect/train_YYYYMMDD_HHMMSS.log`。

## 常见问题排查

- **找不到权重文件**  
  看 `runs/detect/train*/weights/` 目录，`best.pt` 和 `last.pt` 都在这里。

- **找不到日志**  
  如果用 `train_with_log.bat` 启动，日志在 `runs/detect/train_YYYYMMDD_HHMMSS.log`。

- **训练中断如何继续**  
  使用：
  ```
  yolo train resume model=runs/detect/trainX/weights/last.pt
  ```

- **输出目录太乱**  
  用 `project=` 和 `name=` 统一命名；或定期把旧实验移动到归档目录。

## 备注

- 每次运行会自动递增：`train`、`train2`、`train3`…
- 可以通过 `project=` 和 `name=` 参数自定义输出目录。
