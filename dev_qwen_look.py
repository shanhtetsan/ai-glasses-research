#!/usr/bin/env python3
"""Ask local Qwen Omni about one webcam frame or image file."""

import argparse
import asyncio
import base64
import os
import platform
from pathlib import Path
from typing import Union

import cv2


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))


def parse_source(value: str) -> Union[int, str]:
    try:
        return int(value)
    except ValueError:
        return value


def capture_jpeg(source: Union[int, str], width: int, height: int, quality: int) -> bytes:
    if isinstance(source, str) and Path(source).is_file():
        frame = cv2.imread(source)
        if frame is None:
            raise RuntimeError(f"Could not read image: {source}")
    else:
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" and isinstance(source, int) else 0
        cap = cv2.VideoCapture(source, backend) if backend else cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera/video source: {source}")
        try:
            if width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            # Warm up auto-exposure before grabbing the frame to analyze.
            frame = None
            for _ in range(8):
                ok, frame = cap.read()
                if not ok:
                    frame = None
            if frame is None:
                raise RuntimeError("Could not capture a frame")
        finally:
            cap.release()

    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Could not encode frame as JPEG")
    return encoded.tobytes()


async def ask_qwen(jpeg_bytes: bytes, prompt: str) -> str:
    # Import after cache environment variables are set; omni_client loads the model at import time.
    from omni_client import stream_chat

    img_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        },
        {"type": "text", "text": prompt},
    ]

    parts = []
    async for piece in stream_chat(content):
        if piece.text_delta:
            parts.append(piece.text_delta)
    return "".join(parts).strip()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one frame and ask local Qwen what it sees.")
    parser.add_argument("--source", default="0", help="Camera index, image path, or video path.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument(
        "--prompt",
        default="Describe the visible objects in this image. Be specific and concise.",
    )
    args = parser.parse_args()

    jpeg = capture_jpeg(parse_source(args.source), args.width, args.height, args.quality)
    response = await ask_qwen(jpeg, args.prompt)
    print(response or "(empty response)")


if __name__ == "__main__":
    asyncio.run(main())
