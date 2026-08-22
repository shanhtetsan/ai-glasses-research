# AI Glass System - Dockerfile (cloud / gemini_live-only deploy)
# Lightweight image: no CUDA, no torch/ultralytics/mediapipe — navigation
# features degrade gracefully at startup when those packages are absent
# (see app_main.py's try/except import guards). For the full local/GPU
# stack with blind-path + cross-street navigation, install requirements.txt
# instead.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies: curl for the healthcheck; libgl1/libglib for cv2
# even in the headless build (some of its codecs still dlopen these).
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# 复制 requirements-cloud.txt
COPY requirements-cloud.txt .

# 安装 Python 依赖 (cloud subset only — see requirements-cloud.txt)
RUN pip install --no-cache-dir -r requirements-cloud.txt

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p recordings model music voice static templates

# 暴露端口
EXPOSE 8081 12345/udp

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8081/api/health || exit 1

# 启动命令
# CMD ["python3", "app_main.py"]
CMD ["sh", "-c", "python3 app_main.py 2>&1 | tee -a /data/backend.log"]
