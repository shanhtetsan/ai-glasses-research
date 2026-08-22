FROM python:3.11-slim

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    YOLO_MODEL_PATH=/app/yolov8n.pt \
    YOLO_DEVICE=cpu \
    YOLO_CONFIDENCE=0.25 \
    YOLO_IMAGE_SIZE=640 \
    YOLO_TORCH_THREADS=1 \
    HAND_MODEL_PATH=/app/hand_landmarker.task

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY yolo_service/requirements.txt ./requirements.txt
RUN python3 -m pip install --upgrade pip \
    && pip install --no-cache-dir \
      torch==2.0.1 torchvision==0.15.2 --index-url ${PYTORCH_INDEX_URL} \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps ultralytics==8.3.200 \
    && python3 -c "import importlib.metadata as m, cv2, mediapipe, numpy, torch, torchvision; from ultralytics import YOLO; assert numpy.__version__ == '1.24.3'; assert m.version('opencv-contrib-python') == '4.11.0.86'; assert all((d.metadata.get('Name') or '').lower() not in {'opencv-python', 'opencv-python-headless'} for d in m.distributions())"

COPY yolov8n.pt /app/yolov8n.pt
COPY hand_landmarker.task /app/hand_landmarker.task
COPY yolo_service/core.py yolo_service/app.py /app/

# Validate the exact creator checkpoint during build without saving output or
# permitting an implicit model download.
RUN python3 -c "import numpy as np; from ultralytics import YOLO; YOLO('/app/yolov8n.pt', task='detect').predict(source=np.zeros((240, 320, 3), dtype=np.uint8), device='cpu', imgsz=640, batch=1, save=False, verbose=False)"
RUN python3 -c "from mediapipe.tasks import python; from mediapipe.tasks.python import vision; model=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='/app/hand_landmarker.task', delegate=python.BaseOptions.Delegate.CPU), running_mode=vision.RunningMode.IMAGE, num_hands=2)); model.close()"

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://localhost:8080/healthz || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--no-access-log"]
