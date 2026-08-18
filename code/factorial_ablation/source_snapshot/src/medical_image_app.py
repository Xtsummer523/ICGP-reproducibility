"""PyQt application for the medical image processing course project."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

try:
    from .medical_processing import apply_method, list_images, method_names, normalize_to_bgr, read_image, save_image
except ImportError:
    from medical_processing import apply_method, list_images, method_names, normalize_to_bgr, read_image, save_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "medical_images"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "processed"


class ImagePanel(QGroupBox):
    """A fixed image display panel with a title and status text."""

    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.setObjectName("imagePanel")
        self.image_label = QLabel("暂无图像")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(360, 280)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background:#111827;color:#cbd5e1;border:1px solid #334155;")
        self.info_label = QLabel("等待加载")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("color:#475569;")
        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addWidget(self.info_label)
        self._pixmap: QPixmap | None = None

    def set_image(self, image: np.ndarray, note: str) -> None:
        bgr = normalize_to_bgr(image)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb.shape
        qimage = QImage(rgb.data, width, height, channel * width, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimage)
        self.info_label.setText(f"{note}    {width} x {height}")
        self._refresh_pixmap()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)


class MedicalImageMainWindow(QMainWindow):
    """Main window following the reference sketch: original, processed, buttons."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("医学图像处理综合应用系统")
        self.resize(1120, 760)
        self.current_path: Path | None = None
        self.current_image: np.ndarray | None = None
        self.result_image: np.ndarray | None = None
        self.result_name = ""
        self.image_files: list[Path] = []

        self.original_panel = ImagePanel("原始图像")
        self.result_panel = ImagePanel("处理结果")
        self.image_list = QListWidget()
        self.algorithm_box = QComboBox()
        self.algorithm_box.addItems(method_names())

        self._build_ui()
        self._apply_style()
        self.load_folder(DEFAULT_IMAGE_DIR)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 14, 16, 12)
        root.setSpacing(12)

        title = QLabel("医学图像处理综合应用系统")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        image_area = QGridLayout()
        image_area.setColumnStretch(0, 2)
        image_area.setColumnStretch(1, 3)
        image_area.setColumnStretch(2, 3)
        image_area.setSpacing(12)

        list_box = QGroupBox("图像文件")
        list_layout = QVBoxLayout(list_box)
        self.image_list.itemSelectionChanged.connect(self.on_image_selected)
        list_layout.addWidget(self.image_list)
        image_area.addWidget(list_box, 0, 0)
        image_area.addWidget(self.original_panel, 0, 1)
        image_area.addWidget(self.result_panel, 0, 2)
        root.addLayout(image_area, stretch=1)

        control_bar = QHBoxLayout()
        self.open_button = QPushButton("打开文件夹")
        self.open_button.clicked.connect(self.choose_folder)
        self.prev_button = QPushButton("上一张")
        self.prev_button.clicked.connect(self.show_previous)
        self.next_button = QPushButton("下一张")
        self.next_button.clicked.connect(self.show_next)
        self.run_button = QPushButton("运行所选算法")
        self.run_button.clicked.connect(self.run_selected_algorithm)
        self.save_button = QPushButton("保存结果")
        self.save_button.clicked.connect(self.save_result)
        self.exit_button = QPushButton("退出")
        self.exit_button.clicked.connect(self.close)

        control_bar.addWidget(self.open_button)
        control_bar.addWidget(self.prev_button)
        control_bar.addWidget(self.next_button)
        control_bar.addWidget(QLabel("算法"))
        control_bar.addWidget(self.algorithm_box)
        control_bar.addWidget(self.run_button)
        control_bar.addWidget(self.save_button)
        control_bar.addStretch(1)
        control_bar.addWidget(self.exit_button)
        root.addLayout(control_bar)

        quick_box = QGroupBox("基本功能与创新功能")
        quick_layout = QHBoxLayout(quick_box)
        for text in ["灰度化", "时域滤波", "直方图均衡化", "图像分割", "边缘定位", "创新-分水岭"]:
            button = QPushButton(text)
            button.clicked.connect(lambda _checked=False, name=text: self.run_algorithm(name))
            quick_layout.addWidget(button)
        root.addWidget(quick_box)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #eef2f7; }
            #titleLabel {
                font-size: 24px;
                font-weight: 700;
                color: #0f172a;
                padding: 6px;
            }
            QGroupBox {
                font-size: 15px;
                font-weight: 600;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                margin-top: 10px;
                background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #0f172a;
            }
            QListWidget {
                border: 1px solid #cbd5e1;
                background: #f8fafc;
                font-size: 13px;
            }
            QPushButton {
                min-height: 34px;
                padding: 0 14px;
                border: 1px solid #2563eb;
                border-radius: 5px;
                color: #0f172a;
                background: #dbeafe;
                font-size: 14px;
            }
            QPushButton:hover { background: #bfdbfe; }
            QPushButton:pressed { background: #93c5fd; }
            QComboBox {
                min-height: 34px;
                border: 1px solid #94a3b8;
                border-radius: 5px;
                padding-left: 8px;
                background: #ffffff;
                font-size: 14px;
            }
            """
        )

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹", str(DEFAULT_IMAGE_DIR))
        if folder:
            self.load_folder(Path(folder))

    def load_folder(self, folder: Path) -> None:
        self.image_files = list_images(folder)
        self.image_list.clear()
        for path in self.image_files:
            self.image_list.addItem(path.name)
        if not self.image_files:
            self.statusBar().showMessage(f"文件夹没有找到图像: {folder}")
            return
        self.image_list.setCurrentRow(0)
        self.statusBar().showMessage(f"已载入 {len(self.image_files)} 张图像: {folder}")

    def on_image_selected(self) -> None:
        row = self.image_list.currentRow()
        if row < 0 or row >= len(self.image_files):
            return
        self.load_image(self.image_files[row])

    def load_image(self, path: Path) -> None:
        try:
            self.current_image = read_image(path)
        except Exception as exc:
            QMessageBox.warning(self, "读取失败", str(exc))
            return
        self.current_path = path
        self.result_image = None
        self.result_name = ""
        self.original_panel.set_image(self.current_image, path.name)
        self.result_panel.image_label.setText("请选择算法")
        self.result_panel.image_label.setPixmap(QPixmap())
        self.result_panel.info_label.setText("等待处理")
        self.statusBar().showMessage(f"当前图像: {path.name}")

    def show_previous(self) -> None:
        row = self.image_list.currentRow()
        if row > 0:
            self.image_list.setCurrentRow(row - 1)

    def show_next(self) -> None:
        row = self.image_list.currentRow()
        if row < self.image_list.count() - 1:
            self.image_list.setCurrentRow(row + 1)

    def run_selected_algorithm(self) -> None:
        self.run_algorithm(self.algorithm_box.currentText())

    def run_algorithm(self, name: str) -> None:
        if self.current_image is None:
            QMessageBox.information(self, "提示", "请先打开或选择一张图像。")
            return
        try:
            result = apply_method(name, self.current_image)
        except Exception as exc:
            QMessageBox.warning(self, "处理失败", str(exc))
            return
        self.result_image = result.image
        self.result_name = result.name
        self.result_panel.set_image(result.image, result.name)
        self.statusBar().showMessage(result.description)

    def save_result(self) -> None:
        if self.result_image is None or self.current_path is None:
            QMessageBox.information(self, "提示", "还没有可保存的处理结果。")
            return
        safe_name = self.result_name.replace("：", "_").replace(":", "_").replace("-", "_")
        target = DEFAULT_OUTPUT_DIR / f"{self.current_path.stem}_{safe_name}.png"
        try:
            save_image(target, self.result_image)
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        self.statusBar().showMessage(f"结果已保存: {target}")
        QMessageBox.information(self, "保存成功", f"已保存到:\n{target}")


def main() -> int:
    app = QApplication(sys.argv)
    window = MedicalImageMainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
