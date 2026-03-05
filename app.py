from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
import ultralytics.nn.tasks as _tasks
from ultralytics import YOLO

from models.custom_modules import EMA, DCNv3, WConcat, BiFPN

_tasks.EMA = EMA
_tasks.DCNv3 = DCNv3
_tasks.WConcat = WConcat
_tasks.BiFPN = BiFPN

try:
    import onnxruntime as ort
except Exception:
    ort = None


class FlowerApp(QtWidgets.QMainWindow):
    def __init__(self, weights: str, device: str = "0") -> None:
        super().__init__()
        self.setWindowTitle("Flower Detection - YOLOv8")
        self.resize(1200, 720)

        self.model = YOLO(weights)
        self.device = device

        self.image_label = QtWidgets.QLabel("Drop image here or click Open")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #888;")
        self.image_label.setMinimumSize(640, 640)
        self.image_label.setAcceptDrops(True)

        self.open_btn = QtWidgets.QPushButton("Open Image")
        self.open_dir_btn = QtWidgets.QPushButton("Batch Folder")
        self.camera_btn = QtWidgets.QPushButton("Start Camera")
        self.pause_btn = QtWidgets.QPushButton("Pause Camera")
        self.pause_btn.setEnabled(False)
        self.run_btn = QtWidgets.QPushButton("Run Detection")
        self.save_btn = QtWidgets.QPushButton("Save Result")
        self.export_btn = QtWidgets.QPushButton("Export Report")
        self.save_btn.setEnabled(False)
        self.export_btn.setEnabled(False)

        self.conf_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(25)
        self.conf_value = QtWidgets.QLabel("0.25")

        self.iou_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.iou_slider.setRange(1, 100)
        self.iou_slider.setValue(45)
        self.iou_value = QtWidgets.QLabel("0.45")

        self.fps_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.fps_slider.setRange(1, 30)
        self.fps_slider.setValue(10)
        self.fps_value = QtWidgets.QLabel("10 FPS")

        self.resolution_box = QtWidgets.QComboBox()
        self.resolution_box.addItems(["640x480", "1280x720", "1920x1080"])

        self.result_list = QtWidgets.QTextEdit()
        self.result_list.setReadOnly(True)

        left_panel = QtWidgets.QVBoxLayout()
        left_panel.addWidget(self.open_btn)
        left_panel.addWidget(self.open_dir_btn)
        left_panel.addWidget(self.camera_btn)
        left_panel.addWidget(self.pause_btn)
        left_panel.addWidget(self.run_btn)
        left_panel.addWidget(self.save_btn)
        left_panel.addWidget(self.export_btn)
        left_panel.addSpacing(20)
        left_panel.addWidget(QtWidgets.QLabel("Confidence"))
        left_panel.addWidget(self.conf_slider)
        left_panel.addWidget(self.conf_value)
        left_panel.addWidget(QtWidgets.QLabel("IoU"))
        left_panel.addWidget(self.iou_slider)
        left_panel.addWidget(self.iou_value)
        self.nms_checkbox = QtWidgets.QCheckBox("Explicit NMS")
        self.nms_checkbox.setChecked(True)
        left_panel.addWidget(self.nms_checkbox)
        left_panel.addSpacing(10)
        left_panel.addWidget(QtWidgets.QLabel("Camera FPS"))
        left_panel.addWidget(self.fps_slider)
        left_panel.addWidget(self.fps_value)
        left_panel.addWidget(QtWidgets.QLabel("Camera Resolution"))
        left_panel.addWidget(self.resolution_box)
        left_panel.addWidget(QtWidgets.QLabel("Batch Progress"))
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        left_panel.addWidget(self.progress_bar)
        left_panel.addSpacing(10)
        left_panel.addWidget(QtWidgets.QLabel("Model Select"))
        self.model_box = QtWidgets.QComboBox()
        self.model_box.setEditable(True)
        self.model_box.addItem(weights)
        left_panel.addWidget(self.model_box)
        left_panel.addWidget(QtWidgets.QLabel("Backend"))
        self.backend_box = QtWidgets.QComboBox()
        self.backend_box.addItems(
            ["PyTorch", "ONNX Runtime (CPU)", "ONNX Runtime (GPU)", "TensorRT Engine"]
        )
        left_panel.addWidget(self.backend_box)
        self.browse_model_btn = QtWidgets.QPushButton("Browse Model")
        left_panel.addWidget(self.browse_model_btn)
        self.load_model_btn = QtWidgets.QPushButton("Load Model")
        left_panel.addWidget(self.load_model_btn)
        left_panel.addStretch(1)

        right_panel = QtWidgets.QVBoxLayout()
        right_panel.addWidget(QtWidgets.QLabel("Results"))
        right_panel.addWidget(self.result_list)
        right_panel.addWidget(QtWidgets.QLabel("History"))
        self.history_list = QtWidgets.QListWidget()
        right_panel.addWidget(self.history_list)

        center_panel = QtWidgets.QVBoxLayout()
        center_panel.addWidget(self.image_label)

        root = QtWidgets.QHBoxLayout()
        root.addLayout(left_panel, 1)
        root.addLayout(center_panel, 4)
        root.addLayout(right_panel, 2)

        container = QtWidgets.QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        self.open_btn.clicked.connect(self.open_image)
        self.open_dir_btn.clicked.connect(self.run_batch)
        self.camera_btn.clicked.connect(self.toggle_camera)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.run_btn.clicked.connect(self.run_inference)
        self.save_btn.clicked.connect(self.save_result)
        self.export_btn.clicked.connect(self.export_report)
        self.conf_slider.valueChanged.connect(self.update_conf)
        self.iou_slider.valueChanged.connect(self.update_iou)
        self.fps_slider.valueChanged.connect(self.update_fps)
        self.resolution_box.currentIndexChanged.connect(self.update_resolution)
        self.browse_model_btn.clicked.connect(self.browse_model)
        self.load_model_btn.clicked.connect(self.load_model)
        self.history_list.itemClicked.connect(self.load_history_item)

        self.current_image_path: Path | None = None
        self.current_image: np.ndarray | None = None
        self.current_result_image: np.ndarray | None = None
        self.last_results: list[dict] = []
        self.history: list[dict] = []
        self.backend_type = "PyTorch"
        self.backend_device = self.device

        self.cap: cv2.VideoCapture | None = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_camera_frame)
        self.camera_paused = False

    def update_conf(self) -> None:
        value = self.conf_slider.value() / 100.0
        self.conf_value.setText(f"{value:.2f}")

    def update_iou(self) -> None:
        value = self.iou_slider.value() / 100.0
        self.iou_value.setText(f"{value:.2f}")

    def update_fps(self) -> None:
        fps = self.fps_slider.value()
        self.fps_value.setText(f"{fps} FPS")
        if self.timer.isActive():
            self.timer.setInterval(max(1, int(1000 / fps)))

    def update_resolution(self) -> None:
        if self.cap is None:
            return
        w, h = self.parse_resolution()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    def open_image(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return
        self.load_image(Path(file_path))

    def load_image(self, path: Path) -> None:
        self.stop_camera()
        img = cv2.imread(str(path))
        if img is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Failed to load image")
            return
        self.current_image_path = path
        self.current_image = img
        self.current_result_image = None
        self.save_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.result_list.clear()
        self.show_image(img)

    def show_image(self, img: np.ndarray) -> None:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio))

    def run_inference(self) -> None:
        if self.current_image is None:
            QtWidgets.QMessageBox.information(self, "Info", "Please open an image first")
            return
        conf = self.conf_slider.value() / 100.0
        iou = self.iou_slider.value() / 100.0

        plotted, lines = self.infer_and_render(self.current_image, conf, iou)
        self.current_result_image = plotted
        self.show_image(plotted)
        self.save_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.result_list.setText("\n".join(lines) if lines else "No detections")
        if self.current_image_path is not None:
            self.add_history_item(self.current_image_path, self.current_image, plotted, lines)

    def save_result(self) -> None:
        if self.current_result_image is None:
            return
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Result", "result.jpg", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return
        cv2.imwrite(file_path, self.current_result_image)

    def export_report(self) -> None:
        if not self.last_results:
            QtWidgets.QMessageBox.information(self, "Info", "No results to export")
            return
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Report", "report.csv", "CSV (*.csv);;Excel (*.xlsx)"
        )
        if not file_path:
            return
        self.write_report(Path(file_path), self.last_results)

    def run_batch(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not out_dir:
            return
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        paths = [p for p in Path(folder).rglob("*") if p.suffix.lower() in exts]
        if not paths:
            QtWidgets.QMessageBox.information(self, "Info", "No images found in folder")
            return
        self.progress_bar.setValue(0)
        conf = self.conf_slider.value() / 100.0
        iou = self.iou_slider.value() / 100.0
        report_rows = []
        total = len(paths)
        for idx, path in enumerate(paths, 1):
            img = cv2.imread(str(path))
            if img is None:
                continue
            results = self.model.predict(source=img, conf=conf, iou=iou, device=self.device, verbose=False)
            res = results[0]
            plotted = res.plot()
            out_path = Path(out_dir) / f"{path.stem}_det{path.suffix}"
            cv2.imwrite(str(out_path), plotted)
            for box in res.boxes:
                cls = int(box.cls[0])
                name = res.names.get(cls, str(cls))
                score = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                report_rows.append(
                    {
                        "image": str(path),
                        "class": name,
                        "score": score,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )
            progress = int(idx / total * 100)
            self.progress_bar.setValue(progress)
            QtWidgets.QApplication.processEvents()
        report_path = Path(out_dir) / "batch_report.csv"
        self.write_report(report_path, report_rows)
        QtWidgets.QMessageBox.information(self, "Done", f"Saved results to {out_dir}")

    def toggle_camera(self) -> None:
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap = None
                QtWidgets.QMessageBox.warning(self, "Error", "Failed to open camera")
                return
            self.update_resolution()
            self.camera_btn.setText("Stop Camera")
            self.pause_btn.setEnabled(True)
            self.camera_paused = False
            self.pause_btn.setText("Pause Camera")
            fps = self.fps_slider.value()
            self.timer.start(max(1, int(1000 / fps)))
        else:
            self.stop_camera()

    def stop_camera(self) -> None:
        if self.cap is not None:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.camera_btn.setText("Start Camera")
            self.pause_btn.setEnabled(False)
            self.camera_paused = False
            self.pause_btn.setText("Pause Camera")

    def update_camera_frame(self) -> None:
        if self.camera_paused:
            return
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        self.current_image = frame
        self.current_image_path = None
        conf = self.conf_slider.value() / 100.0
        iou = self.iou_slider.value() / 100.0
        plotted, lines = self.infer_and_render(frame, conf, iou)
        self.current_result_image = plotted
        self.show_image(plotted)
        self.save_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.result_list.setText("\n".join(lines) if lines else "No detections")

    def add_history_item(self, path: Path, original: np.ndarray, result: np.ndarray, lines: list[str]) -> None:
        item_index = len(self.history)
        self.history.append(
            {
                "path": path,
                "original": original.copy(),
                "result": result.copy(),
                "lines": lines,
            }
        )
        item = QtWidgets.QListWidgetItem(f"{path.name}")
        item.setData(QtCore.Qt.UserRole, item_index)
        self.history_list.addItem(item)

    def load_history_item(self, item: QtWidgets.QListWidgetItem) -> None:
        idx = item.data(QtCore.Qt.UserRole)
        if idx is None:
            return
        data = self.history[int(idx)]
        self.current_image_path = data["path"]
        self.current_image = data["original"]
        self.current_result_image = data["result"]
        self.show_image(self.current_result_image)
        self.save_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.result_list.setText("\n".join(data["lines"]) if data["lines"] else "No detections")

    def parse_resolution(self) -> tuple[int, int]:
        text = self.resolution_box.currentText()
        w, h = text.split("x")
        return int(w), int(h)

    def toggle_pause(self) -> None:
        if self.cap is None:
            return
        self.camera_paused = not self.camera_paused
        self.pause_btn.setText("Resume Camera" if self.camera_paused else "Pause Camera")

    def browse_model(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Model", "", "Models (*.pt *.onnx *.engine)"
        )
        if not file_path:
            return
        if self.model_box.findText(file_path) == -1:
            self.model_box.addItem(file_path)
        self.model_box.setCurrentText(file_path)
        self.load_model()

    def load_model(self) -> None:
        model_path = self.model_box.currentText().strip()
        if not model_path:
            return
        if not Path(model_path).exists():
            QtWidgets.QMessageBox.warning(self, "Error", f"Model not found: {model_path}")
            return
        backend = self.backend_box.currentText()
        self.model, self.backend_device, self.backend_type = self.build_backend(model_path, backend)
        QtWidgets.QMessageBox.information(
            self, "Model", f"Loaded: {model_path}\nBackend: {self.backend_type}"
        )

    def build_backend(self, model_path: str, backend: str) -> tuple[YOLO, str, str]:
        path = Path(model_path)
        backend_device = self.device
        backend_type = backend

        if backend.startswith("PyTorch"):
            if path.suffix != ".pt":
                QtWidgets.QMessageBox.warning(self, "Warning", "PyTorch backend expects .pt weights")
        elif backend.startswith("ONNX Runtime"):
            if path.suffix != ".onnx":
                QtWidgets.QMessageBox.warning(self, "Warning", "ONNX Runtime backend expects .onnx weights")
            if ort is None:
                QtWidgets.QMessageBox.warning(self, "Warning", "onnxruntime not installed")
            if backend.endswith("(CPU)"):
                backend_device = "cpu"
            else:
                backend_device = "0"
                if ort is not None and "CUDAExecutionProvider" not in ort.get_available_providers():
                    QtWidgets.QMessageBox.warning(
                        self, "Warning", "ONNX Runtime GPU provider not available, falling back to CPU"
                    )
                    backend_device = "cpu"
                    backend_type = "ONNX Runtime (CPU)"
        else:
            if path.suffix != ".engine":
                QtWidgets.QMessageBox.warning(self, "Warning", "TensorRT backend expects .engine weights")
            backend_device = "0"

        model = YOLO(model_path)
        return model, backend_device, backend_type

    def infer_and_render(self, image: np.ndarray, conf: float, iou: float) -> tuple[np.ndarray, list[str]]:
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            device=self.backend_device,
            verbose=False,
        )
        res = results[0]
        names = res.names

        if res.boxes is None or len(res.boxes) == 0:
            self.last_results = []
            return image.copy(), []

        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)

        if self.nms_checkbox.isChecked():
            keep = self.nms_boxes(boxes, scores, iou)
            boxes = boxes[keep]
            scores = scores[keep]
            classes = classes[keep]

        plotted = self.draw_boxes(image.copy(), boxes, scores, classes, names)

        lines = []
        self.last_results = []
        img_name = str(self.current_image_path) if self.current_image_path else "camera_frame"
        for i in range(len(boxes)):
            name = names.get(int(classes[i]), str(int(classes[i])))
            score = float(scores[i])
            x1, y1, x2, y2 = map(float, boxes[i])
            lines.append(f"{name}\t{score:.3f}")
            self.last_results.append(
                {
                    "image": img_name,
                    "class": name,
                    "score": score,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
        return plotted, lines

    def nms_boxes(self, boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
        if len(boxes) == 0:
            return []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thres)[0]
            order = order[inds + 1]
        return keep

    def draw_boxes(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        names: dict,
    ) -> np.ndarray:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (0, 200, 0)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            label = f"{names.get(int(classes[i]), classes[i])} {scores[i]:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Measure text size to determine a safe label position
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw label above the box when there is enough room, otherwise inside
            if y1 - th - 8 >= 0:
                label_y = y1 - 5
                bg_y1, bg_y2 = y1 - th - 8, y1
            else:
                label_y = y1 + th + 3
                bg_y1, bg_y2 = y1, y1 + th + 6

            # Filled background rectangle keeps text readable on any image
            cv2.rectangle(image, (x1, bg_y1), (x1 + tw + 4, bg_y2), color, cv2.FILLED)
            cv2.putText(
                image,
                label,
                (x1 + 2, label_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )
        return image

    def write_report(self, path: Path, rows: list[dict]) -> None:
        df = pd.DataFrame(rows)
        summary = self.build_summary(df)
        if path.suffix.lower() == ".xlsx":
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="detections")
                summary.to_excel(writer, index=False, sheet_name="summary")
        else:
            df.to_csv(path, index=False)
            with path.open("a", encoding="utf-8") as f:
                f.write("\nsummary\n")
                summary.to_csv(f, index=False)

    def build_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["class", "count", "avg_score"])
        grouped = df.groupby("class", as_index=False).agg(
            count=("score", "size"),
            avg_score=("score", "mean"),
        )
        grouped["avg_score"] = grouped["avg_score"].round(6)
        return grouped

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        urls = event.mimeData().urls()
        if not urls:
            return
        path = Path(urls[0].toLocalFile())
        self.load_image(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower Detection GUI")
    parser.add_argument("--weights", required=True, help="path to trained weights (.pt or .onnx)")
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = QtWidgets.QApplication([])
    win = FlowerApp(args.weights, device=args.device)
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
