#!/usr/bin/env python3
"""Ask Gemini about one webcam frame or image file. Laptop-only test, no ESP32 needed."""

import argparse
import os
import platform
import sys
from pathlib import Path
from typing import Union

import cv2


DEFAULT_PROMPT = (
    "Read all visible text on the product label in this image. "
    "List the product name, brand, and any key information (size, ingredients, warnings). "
    "If no label is visible, say so."
)


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


def ask_gemini(jpeg_bytes: bytes, prompt: str, model_name: str) -> str:
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY is not set. Run: export GEMINI_API_KEY=<your-key>")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        [prompt, {"mime_type": "image/jpeg", "data": jpeg_bytes}]
    )
    return (response.text or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one frame and ask Gemini what's on the label.")
    parser.add_argument("--source", default="0", help="Camera index, image path, or video path.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default="gemini-2.5-flash-lite",
                        help="Gemini model id. Default is the cheapest vision-capable Flash-Lite.")
    args = parser.parse_args()

    jpeg = capture_jpeg(parse_source(args.source), args.width, args.height, args.quality)
    try:
        response = ask_gemini(jpeg, args.prompt, args.model)
    except Exception as e:
        sys.exit(f"Gemini API error: {e}")

    print(response or "(empty response)")


if __name__ == "__main__":
    main()
