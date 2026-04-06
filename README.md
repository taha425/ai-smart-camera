# Automatic Camera Zoom (People in Frame)

This is a **basic working model** of an automatic camera zoom system:

- Detects **people** in each frame using **YOLOv8** (COCO class `person`)
- Computes a stable crop that keeps detected people in view
- Applies a “virtual zoom” (crop + resize) with smoothing to reduce jitter
- Includes a simple **frontend UI** via Streamlit (web app)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app/streamlit_app.py
```

## Notes on accuracy

- For better accuracy (especially small/far people), pick `yolov8s.pt` in the sidebar (slower but stronger).
- If zoom jitters, increase **Smoothing (EMA alpha)** and/or **Padding**.

