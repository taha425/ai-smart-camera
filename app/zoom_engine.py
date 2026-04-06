from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


@dataclass
class ZoomConfig:
    # Add padding around detected people (fraction of union box size).
    padding: float = 0.18
    # Clamp zoom; 1.0 means no zoom (full frame). Higher means tighter crop.
    min_zoom: float = 1.0
    max_zoom: float = 2.8
    # Smooth camera motion (0..1). Higher = smoother/slower.
    ema_alpha: float = 0.80
    # If no people detected, slowly ease back to full frame.
    ease_out_alpha: float = 0.90


@dataclass
class CameraState:
    # Normalized crop center (0..1)
    cx: float = 0.5
    cy: float = 0.5
    # Normalized crop size (0..1) where 1x1 is full frame.
    crop_w: float = 1.0
    crop_h: float = 1.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _union_box_xyxy(boxes_xyxy: np.ndarray) -> Tuple[float, float, float, float]:
    x1 = float(np.min(boxes_xyxy[:, 0]))
    y1 = float(np.min(boxes_xyxy[:, 1]))
    x2 = float(np.max(boxes_xyxy[:, 2]))
    y2 = float(np.max(boxes_xyxy[:, 3]))
    return x1, y1, x2, y2


def _expand_box(
    x1: float, y1: float, x2: float, y2: float, pad: float, w: int, h: int
) -> Tuple[float, float, float, float]:
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    x1 -= bw * pad
    x2 += bw * pad
    y1 -= bh * pad
    y2 += bh * pad
    x1 = _clamp(x1, 0.0, float(w - 1))
    y1 = _clamp(y1, 0.0, float(h - 1))
    x2 = _clamp(x2, 0.0, float(w - 1))
    y2 = _clamp(y2, 0.0, float(h - 1))
    if x2 <= x1:
        x2 = min(float(w - 1), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(h - 1), y1 + 1.0)
    return x1, y1, x2, y2


def _fit_aspect(
    x1: float, y1: float, x2: float, y2: float, aspect: float, w: int, h: int
) -> Tuple[float, float, float, float]:
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    if bh <= 0 or bw <= 0:
        return 0.0, 0.0, float(w - 1), float(h - 1)

    current = bw / bh
    if current > aspect:
        # too wide; expand height
        new_h = bw / aspect
        bh = new_h
    else:
        # too tall; expand width
        new_w = bh * aspect
        bw = new_w

    x1 = cx - bw * 0.5
    x2 = cx + bw * 0.5
    y1 = cy - bh * 0.5
    y2 = cy + bh * 0.5

    # clamp while preserving size as much as possible
    if x1 < 0:
        x2 -= x1
        x1 = 0.0
    if y1 < 0:
        y2 -= y1
        y1 = 0.0
    if x2 > w - 1:
        shift = x2 - (w - 1)
        x1 -= shift
        x2 = float(w - 1)
    if y2 > h - 1:
        shift = y2 - (h - 1)
        y1 -= shift
        y2 = float(h - 1)

    x1 = _clamp(x1, 0.0, float(w - 2))
    y1 = _clamp(y1, 0.0, float(h - 2))
    x2 = _clamp(x2, 1.0, float(w - 1))
    y2 = _clamp(y2, 1.0, float(h - 1))
    return x1, y1, x2, y2


def _ema(prev: float, target: float, alpha: float) -> float:
    return alpha * prev + (1.0 - alpha) * target


def update_camera_state(
    state: CameraState,
    frame_w: int,
    frame_h: int,
    person_boxes_xyxy: Optional[np.ndarray],
    cfg: ZoomConfig,
) -> CameraState:
    """
    person_boxes_xyxy: ndarray [N,4] in pixel coords (x1,y1,x2,y2) OR None/empty.
    Returns a new state with smoothed crop window.
    """
    aspect = frame_w / frame_h
    if person_boxes_xyxy is None or len(person_boxes_xyxy) == 0:
        # Ease back to full frame.
        target = CameraState(cx=0.5, cy=0.5, crop_w=1.0, crop_h=1.0)
        alpha = cfg.ease_out_alpha
        return CameraState(
            cx=_ema(state.cx, target.cx, alpha),
            cy=_ema(state.cy, target.cy, alpha),
            crop_w=_ema(state.crop_w, target.crop_w, alpha),
            crop_h=_ema(state.crop_h, target.crop_h, alpha),
        )

    x1, y1, x2, y2 = _union_box_xyxy(person_boxes_xyxy)
    x1, y1, x2, y2 = _expand_box(x1, y1, x2, y2, cfg.padding, frame_w, frame_h)
    x1, y1, x2, y2 = _fit_aspect(x1, y1, x2, y2, aspect, frame_w, frame_h)

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # Convert to normalized crop size.
    crop_w = bw / frame_w
    crop_h = bh / frame_h

    # Convert crop size to zoom and clamp.
    zoom_w = 1.0 / max(1e-6, crop_w)
    zoom_h = 1.0 / max(1e-6, crop_h)
    zoom = min(zoom_w, zoom_h)
    zoom = _clamp(zoom, cfg.min_zoom, cfg.max_zoom)
    crop_w = 1.0 / zoom
    crop_h = 1.0 / zoom

    # Normalize center.
    cx_n = cx / frame_w
    cy_n = cy / frame_h
    cx_n = _clamp(cx_n, 0.0, 1.0)
    cy_n = _clamp(cy_n, 0.0, 1.0)

    # Smooth.
    a = cfg.ema_alpha
    return CameraState(
        cx=_ema(state.cx, cx_n, a),
        cy=_ema(state.cy, cy_n, a),
        crop_w=_ema(state.crop_w, crop_w, a),
        crop_h=_ema(state.crop_h, crop_h, a),
    )


def crop_and_resize(frame_bgr: np.ndarray, state: CameraState) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Returns (zoomed_frame_bgr, crop_rect_xyxy_px) where crop rect is in original pixels.
    """
    h, w = frame_bgr.shape[:2]
    crop_w_px = int(max(2, round(state.crop_w * w)))
    crop_h_px = int(max(2, round(state.crop_h * h)))
    cx = int(round(state.cx * w))
    cy = int(round(state.cy * h))

    x1 = cx - crop_w_px // 2
    y1 = cy - crop_h_px // 2
    x2 = x1 + crop_w_px
    y2 = y1 + crop_h_px

    # Clamp to frame.
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        shift = x2 - w
        x1 -= shift
        x2 = w
    if y2 > h:
        shift = y2 - h
        y1 -= shift
        y2 = h
    x1 = int(_clamp(x1, 0, w - 2))
    y1 = int(_clamp(y1, 0, h - 2))
    x2 = int(_clamp(x2, x1 + 2, w))
    y2 = int(_clamp(y2, y1 + 2, h))

    cropped = frame_bgr[y1:y2, x1:x2]
    zoomed = cv2_resize(cropped, (w, h))
    return zoomed, (x1, y1, x2, y2)


def cv2_resize(img: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    import cv2

    return cv2.resize(img, size_wh, interpolation=cv2.INTER_LINEAR)

