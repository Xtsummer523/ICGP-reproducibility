"""Core image processing methods for the medical image demo system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ProcessResult:
    """Result returned by a processing method."""

    name: str
    image: np.ndarray
    description: str


def list_images(folder: str | Path) -> list[Path]:
    """Return image files in a folder, sorted by name."""

    root = Path(folder)
    if not root.exists():
        return []
    return sorted(
        [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
        key=lambda p: p.name.lower(),
    )


def read_image(path: str | Path) -> np.ndarray:
    """Read an image with support for Chinese paths on Windows."""

    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像: {path}")
    return image


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Save an image with support for Chinese paths on Windows."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix or ".png"
    ok, buffer = cv2.imencode(suffix, normalize_to_bgr(image))
    if not ok:
        raise ValueError(f"无法保存图像: {path}")
    buffer.tofile(str(target))


def normalize_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert gray, binary, or RGB-like arrays into a BGR uint8 image."""

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def grayscale(image: np.ndarray) -> ProcessResult:
    gray = to_gray(image)
    return ProcessResult(
        name="灰度化",
        image=gray,
        description="将彩色图像按亮度信息转换为单通道图像，便于后续阈值、滤波和分割。",
    )


def denoise_filter(image: np.ndarray) -> ProcessResult:
    gray = to_gray(image)
    median = cv2.medianBlur(gray, 5)
    gaussian = cv2.GaussianBlur(median, (5, 5), 0)
    return ProcessResult(
        name="时域滤波",
        image=gaussian,
        description="先用中值滤波抑制椒盐噪声，再用高斯滤波平滑局部亮度波动。",
    )


def histogram_equalization(image: np.ndarray) -> ProcessResult:
    gray = to_gray(image)
    equalized = cv2.equalizeHist(gray)
    return ProcessResult(
        name="直方图均衡化",
        image=equalized,
        description="重新分配灰度直方图，增强低对比度医学图像中的组织边界。",
    )


def otsu_segmentation(image: np.ndarray) -> ProcessResult:
    gray = to_gray(image)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground_ratio = float(np.count_nonzero(mask)) / mask.size
    if foreground_ratio > 0.65:
        mask = cv2.bitwise_not(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    return ProcessResult(
        name="Otsu图像分割",
        image=clean,
        description="依据类间方差自动选择阈值，并用形态学开闭运算去掉小噪点和孔洞。",
    )


def edge_location(image: np.ndarray) -> ProcessResult:
    gray = to_gray(image)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(smooth, 40, 120)
    overlay = normalize_to_bgr(image).copy()
    overlay[edges > 0] = (0, 0, 255)
    return ProcessResult(
        name="边缘定位",
        image=overlay,
        description="使用Canny算子定位高梯度边界，并将疑似组织或病灶轮廓标为红色。",
    )


def watershed_segmentation(image: np.ndarray) -> ProcessResult:
    """Innovation method: watershed segmentation with marker control."""

    bgr = normalize_to_bgr(image)
    gray = to_gray(bgr)
    equalized = cv2.equalizeHist(gray)
    _, binary = cv2.threshold(
        cv2.GaussianBlur(equalized, (5, 5), 0),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    if np.count_nonzero(binary) / binary.size > 0.6:
        binary = cv2.bitwise_not(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    threshold = 0.36 * dist.max() if dist.max() > 0 else 0
    _, sure_fg = cv2.threshold(dist, threshold, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(bgr.copy(), markers)

    result = bgr.copy()
    result[markers == -1] = (0, 0, 255)
    positive = markers > 1
    result[positive] = cv2.addWeighted(result, 0.55, np.full_like(result, (60, 180, 80)), 0.45, 0)[positive]
    return ProcessResult(
        name="创新功能：分水岭分割",
        image=result,
        description="利用距离变换构造前景标记，再用分水岭算法分离粘连区域，适合展示疑似病灶边界。",
    )


def run_pipeline(image: np.ndarray) -> list[ProcessResult]:
    """Run all project functions in the order used in the report."""

    return [
        grayscale(image),
        denoise_filter(image),
        histogram_equalization(image),
        otsu_segmentation(image),
        watershed_segmentation(image),
        edge_location(image),
    ]


METHODS: dict[str, Callable[[np.ndarray], ProcessResult]] = {
    "灰度化": grayscale,
    "时域滤波": denoise_filter,
    "直方图均衡化": histogram_equalization,
    "图像分割": otsu_segmentation,
    "边缘定位": edge_location,
    "创新-分水岭": watershed_segmentation,
}


def method_names() -> list[str]:
    return list(METHODS.keys())


def apply_method(name: str, image: np.ndarray) -> ProcessResult:
    if name not in METHODS:
        raise KeyError(f"未知算法: {name}")
    return METHODS[name](image)
