from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class DetectionConfig:
    model_name: str = "yolov8n.pt"
    conf: float = 0.35
    iou: float = 0.45
    max_det: int = 50


class PersonDetector:
    def __init__(self, cfg: DetectionConfig):
        self.cfg = cfg
        self._model = None

    def _lazy_load(self):
        if self._model is not None:
            return
        from ultralytics import YOLO

        self._model = YOLO(self.cfg.model_name)

    def detect_people_xyxy(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Returns (boxes_xyxy, inference_ms). boxes are pixels [N,4] for class 'person'.
        """
        self._lazy_load()

        import time

        t0 = time.perf_counter()
        results = self._model.predict(
            source=frame_bgr,
            verbose=False,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            max_det=self.cfg.max_det,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), dt_ms

        cls = r0.boxes.cls.detach().cpu().numpy().astype(np.int32)
        xyxy = r0.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        # COCO class 0 is 'person'
        mask = cls == 0
        people = xyxy[mask]
        return people, dt_ms

