from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector import DetectionConfig, PersonDetector
from zoom_engine import CameraState, ZoomConfig, crop_and_resize, update_camera_state


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def draw_overlays(
    frame_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    crop_rect_xyxy: Tuple[int, int, int, int],
    people_count: int,
    infer_ms: float,
    zoom: float,
) -> np.ndarray:
    out = frame_bgr.copy()

    # person boxes (from detection on original frame)
    for (x1, y1, x2, y2) in boxes_xyxy.astype(int):
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 210, 70), 2)

    # crop rectangle used for zoom
    (cx1, cy1, cx2, cy2) = crop_rect_xyxy
    cv2.rectangle(out, (cx1, cy1), (cx2, cy2), (255, 170, 30), 2)

    label = f"people={people_count}  zoom={zoom:.2f}x  infer={infer_ms:.0f}ms"
    cv2.putText(out, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (10, 10, 10), 4, cv2.LINE_AA)
    cv2.putText(out, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (245, 245, 245), 2, cv2.LINE_AA)
    return out


def compute_zoom_from_state(state: CameraState) -> float:
    # crop_w ~ 1/zoom (clamped)
    if state.crop_w <= 1e-6:
        return 1.0
    return 1.0 / state.crop_w


def max_zoom_for_people_count(base_max_zoom: float, people_count: int) -> float:
    """
    Count-based zoom-out rule: more people => lower allowed max zoom (wider view).
    This helps when two people are detected but stand close together (union box is small),
    where a pure "fit union box" approach may not zoom out enough.
    """
    if people_count <= 0:
        return base_max_zoom
    if people_count == 1:
        return base_max_zoom
    if people_count == 2:
        return min(base_max_zoom, 2.0)
    if 3 <= people_count <= 4:
        return min(base_max_zoom, 1.6)
    if 5 <= people_count <= 7:
        return min(base_max_zoom, 1.35)
    return min(base_max_zoom, 1.2)


def synthetic_people_boxes(
    frame_w: int,
    frame_h: int,
    count: int,
    spread: float,
    min_size: float,
    seed: int = 7,
) -> np.ndarray:
    """
    Generates N fake person boxes (xyxy pixels) for testing zoom logic without real people.
    spread: 0..1 (how spread out across the frame)
    min_size: 0..1 (relative box size baseline)
    """
    if count <= 0:
        return np.zeros((0, 4), dtype=np.float32)

    rng = np.random.default_rng(seed)
    # Centers distributed around the frame center; higher spread => wider distribution.
    cx0, cy0 = 0.5, 0.5
    sigma = 0.04 + 0.40 * float(spread)
    cxs = np.clip(rng.normal(cx0, sigma, size=count), 0.05, 0.95)
    cys = np.clip(rng.normal(cy0, sigma, size=count), 0.08, 0.92)

    # Person-like aspect ratio ~ 0.45 (w/h). Vary sizes a bit.
    h_rel = np.clip(min_size * (0.9 + 0.6 * rng.random(size=count)), 0.05, 0.60)
    w_rel = np.clip(h_rel * (0.35 + 0.25 * rng.random(size=count)), 0.03, 0.45)

    x1 = (cxs - w_rel / 2.0) * frame_w
    x2 = (cxs + w_rel / 2.0) * frame_w
    y1 = (cys - h_rel / 2.0) * frame_h
    y2 = (cys + h_rel / 2.0) * frame_h

    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    boxes[:, 0] = np.clip(boxes[:, 0], 0, frame_w - 2)
    boxes[:, 2] = np.clip(boxes[:, 2], 1, frame_w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, frame_h - 2)
    boxes[:, 3] = np.clip(boxes[:, 3], 1, frame_h - 1)
    return boxes


st.set_page_config(page_title="Auto Camera Zoom (People)", layout="wide")
st.title("Automatic Camera Zoom based on People Count")

with st.sidebar:
    st.subheader("Detector (YOLO)")
    model_name = st.selectbox("Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
    conf = st.slider("Confidence", 0.10, 0.80, 0.35, 0.01)
    iou = st.slider("IOU", 0.10, 0.90, 0.45, 0.01)
    st.subheader("Zoom behavior")
    padding = st.slider("Padding", 0.00, 0.50, 0.18, 0.01)
    max_zoom = st.slider("Max zoom", 1.0, 6.0, 2.8, 0.1)
    ema_alpha = st.slider("Smoothing (EMA alpha)", 0.0, 0.95, 0.80, 0.01)
    ease_out_alpha = st.slider("Ease-out alpha (no people)", 0.0, 0.99, 0.90, 0.01)
    st.subheader("Testing (no real people)")
    use_synth = st.checkbox("Use synthetic 'people' boxes", value=False)
    synth_count = st.slider("Synthetic people count", 0, 20, 2, 1, disabled=not use_synth)
    synth_spread = st.slider("Synthetic spread", 0.0, 1.0, 0.8, 0.05, disabled=not use_synth)
    synth_size = st.slider("Synthetic size", 0.05, 0.60, 0.22, 0.01, disabled=not use_synth)
    st.caption("Tip: For better accuracy, choose `yolov8s.pt` (slower but stronger).")

tab1, tab2 = st.tabs(["Image / Video file", "Webcam (OpenCV)"])


def run_on_video_path(path: str, detector: PersonDetector, zcfg: ZoomConfig, draw_debug: bool):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        st.error("Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    st.write(f"FPS (reported): {fps:.1f}")

    state = CameraState()

    colL, colR = st.columns(2, gap="large")
    out_placeholder = colL.empty()
    dbg_placeholder = colR.empty()

    stop = st.button("Stop")

    while cap.isOpened():
        if stop:
            break
        ok, frame = cap.read()
        if not ok:
            break

        if use_synth:
            boxes = synthetic_people_boxes(frame.shape[1], frame.shape[0], synth_count, synth_spread, synth_size)
            infer_ms = 0.0
        else:
            boxes, infer_ms = detector.detect_people_xyxy(frame)

        zcfg_frame = ZoomConfig(
            padding=zcfg.padding,
            min_zoom=zcfg.min_zoom,
            max_zoom=max_zoom_for_people_count(zcfg.max_zoom, int(len(boxes))),
            ema_alpha=zcfg.ema_alpha,
            ease_out_alpha=zcfg.ease_out_alpha,
        )
        state = update_camera_state(state, frame.shape[1], frame.shape[0], boxes, zcfg_frame)
        zoomed, crop_rect = crop_and_resize(frame, state)
        zoom = compute_zoom_from_state(state)

        if draw_debug:
            dbg = draw_overlays(frame, boxes, crop_rect, int(len(boxes)), infer_ms, zoom)
            dbg_placeholder.image(bgr_to_rgb(dbg), caption="Original + boxes + crop", use_container_width=True)

        out_placeholder.image(bgr_to_rgb(zoomed), caption="Auto-zoom output", use_container_width=True)

    cap.release()


with tab1:
    uploaded = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"])
    draw_debug = st.checkbox("Show debug overlays", value=True)

    if uploaded is not None:
        suffix = Path(uploaded.name).suffix.lower()
        cfg = DetectionConfig(model_name=model_name, conf=conf, iou=iou)
        detector = PersonDetector(cfg)
        zcfg = ZoomConfig(padding=padding, max_zoom=max_zoom, ema_alpha=ema_alpha, ease_out_alpha=ease_out_alpha)

        if suffix in [".jpg", ".jpeg", ".png"]:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if use_synth:
                boxes = synthetic_people_boxes(frame.shape[1], frame.shape[0], synth_count, synth_spread, synth_size)
                infer_ms = 0.0
            else:
                boxes, infer_ms = detector.detect_people_xyxy(frame)

            zcfg_frame = ZoomConfig(
                padding=zcfg.padding,
                min_zoom=zcfg.min_zoom,
                max_zoom=max_zoom_for_people_count(zcfg.max_zoom, int(len(boxes))),
                ema_alpha=zcfg.ema_alpha,
                ease_out_alpha=zcfg.ease_out_alpha,
            )
            state = update_camera_state(CameraState(), frame.shape[1], frame.shape[0], boxes, zcfg_frame)
            zoomed, crop_rect = crop_and_resize(frame, state)
            zoom = compute_zoom_from_state(state)

            col1, col2 = st.columns(2, gap="large")
            if draw_debug:
                dbg = draw_overlays(frame, boxes, crop_rect, int(len(boxes)), infer_ms, zoom)
                col1.image(bgr_to_rgb(dbg), caption="Original + boxes + crop", use_container_width=True)
            else:
                col1.image(bgr_to_rgb(frame), caption="Original", use_container_width=True)
            col2.image(bgr_to_rgb(zoomed), caption="Auto-zoom output", use_container_width=True)

        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(uploaded.read())
                tmp_path = f.name
            run_on_video_path(tmp_path, detector, zcfg, draw_debug)


with tab2:
    st.write("This uses OpenCV's webcam capture (runs on the machine running Streamlit).")
    cam_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
    start = st.button("Start webcam")
    draw_debug_cam = st.checkbox("Show debug overlays (webcam)", value=False)

    if start:
        cfg = DetectionConfig(model_name=model_name, conf=conf, iou=iou)
        detector = PersonDetector(cfg)
        zcfg = ZoomConfig(padding=padding, max_zoom=max_zoom, ema_alpha=ema_alpha, ease_out_alpha=ease_out_alpha)
        cap = cv2.VideoCapture(int(cam_index))
        if not cap.isOpened():
            st.error("Could not open webcam.")
        else:
            state = CameraState()
            colL, colR = st.columns(2, gap="large")
            out_placeholder = colL.empty()
            dbg_placeholder = colR.empty()
            stop = st.button("Stop webcam")

            while cap.isOpened():
                if stop:
                    break
                ok, frame = cap.read()
                if not ok:
                    break

                if use_synth:
                    boxes = synthetic_people_boxes(frame.shape[1], frame.shape[0], synth_count, synth_spread, synth_size)
                    infer_ms = 0.0
                else:
                    boxes, infer_ms = detector.detect_people_xyxy(frame)

                zcfg_frame = ZoomConfig(
                    padding=zcfg.padding,
                    min_zoom=zcfg.min_zoom,
                    max_zoom=max_zoom_for_people_count(zcfg.max_zoom, int(len(boxes))),
                    ema_alpha=zcfg.ema_alpha,
                    ease_out_alpha=zcfg.ease_out_alpha,
                )
                state = update_camera_state(state, frame.shape[1], frame.shape[0], boxes, zcfg_frame)
                zoomed, crop_rect = crop_and_resize(frame, state)
                zoom = compute_zoom_from_state(state)

                out_placeholder.image(bgr_to_rgb(zoomed), caption="Auto-zoom output", use_container_width=True)
                if draw_debug_cam:
                    dbg = draw_overlays(frame, boxes, crop_rect, int(len(boxes)), infer_ms, zoom)
                    dbg_placeholder.image(bgr_to_rgb(dbg), caption="Original + boxes + crop", use_container_width=True)

            cap.release()

