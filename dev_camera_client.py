#!/usr/bin/env python3
"""Development ESP32 camera/IMU simulator.

Streams a laptop webcam or video file to the same endpoints used by the ESP32:
- JPEG frames over WebSocket /ws/camera
- optional fake IMU readings over UDP 12345
"""

import argparse
import asyncio
import json
import math
import platform
import socket
import time
from typing import Union

import cv2
import websockets


def parse_source(value: str) -> Union[int, str]:
    try:
        return int(value)
    except ValueError:
        return value


async def send_fake_imu(host: str, port: int, hz: float) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1.0 / hz
    start = time.monotonic()

    while True:
        elapsed = time.monotonic() - start
        # Keep gravity mostly on +Y because app_main.py computes roll/pitch from
        # accel and expects near-flat values around ax=0, ay=1G, az=0.
        msg = {
            "ts": int(time.time() * 1000),
            "accel": {
                "x": 0.05 * math.sin(elapsed * 0.8),
                "y": 1.0,
                "z": 0.05 * math.cos(elapsed * 0.6),
            },
            "gyro": {
                "x": 0.0,
                "y": 0.0,
                "z": 8.0 * math.sin(elapsed * 0.5),
            },
        }
        sock.sendto(json.dumps(msg).encode("utf-8"), (host, port))
        await asyncio.sleep(period)


async def stream_camera(args: argparse.Namespace) -> None:
    source = parse_source(args.source)

    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" and isinstance(source, int) else 0
    cap = cv2.VideoCapture(source, backend) if backend else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera/video source: {args.source}")

    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    period = 1.0 / args.fps
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]
    sent_frames = 0
    last_report = time.monotonic()

    try:
        while True:
            print(f"[dev-camera] connecting to {args.url}")
            try:
                async with websockets.connect(
                    args.url,
                    max_size=None,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    print("[dev-camera] connected; streaming frames")
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            if args.loop and not isinstance(source, int):
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                continue
                            print("[dev-camera] frame capture failed; reopening source")
                            cap.release()
                            await asyncio.sleep(0.5)
                            cap = cv2.VideoCapture(source, backend) if backend else cv2.VideoCapture(source)
                            if args.width:
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                            if args.height:
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                            continue

                        ok, encoded = cv2.imencode(".jpg", frame, encode_params)
                        if ok:
                            await ws.send(encoded.tobytes())
                            sent_frames += 1

                        now = time.monotonic()
                        if now - last_report >= 2.0:
                            print(f"[dev-camera] sent {sent_frames} frames")
                            last_report = now

                        await asyncio.sleep(period)
            except (OSError, websockets.WebSocketException) as exc:
                print(f"[dev-camera] disconnected: {exc}; retrying in 2s")
                await asyncio.sleep(2.0)
    finally:
        cap.release()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream a local webcam/video to the AI glasses server."
    )
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8081/ws/camera",
        help="Camera WebSocket URL.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, e.g. 0, or path to a video file.",
    )
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--loop", action="store_true", help="Loop video-file sources.")
    parser.add_argument("--imu", action="store_true", help="Also send fake IMU UDP data.")
    parser.add_argument("--imu-host", default="127.0.0.1")
    parser.add_argument("--imu-port", type=int, default=12345)
    parser.add_argument("--imu-hz", type=float, default=50.0)
    args = parser.parse_args()

    tasks = [asyncio.create_task(stream_camera(args))]
    if args.imu:
        tasks.append(asyncio.create_task(send_fake_imu(args.imu_host, args.imu_port, args.imu_hz)))

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
