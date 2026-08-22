#!/usr/bin/env python3
"""Simulate the multiplexed ESP32 camera/thermal/IMU stability socket."""
import argparse
import asyncio
import math
import ssl
import struct
import time

import websockets

from stability_runtime import MSG_TYPE_CAM, MSG_TYPE_IMU, MSG_TYPE_THERMAL


IMU = struct.Struct("<IIffffff")


async def run(args):
    ssl_context = ssl.create_default_context() if args.url.startswith("wss://") else None
    sequence = 0
    started = time.monotonic()
    while time.monotonic() - started < args.seconds:
        try:
            async with websockets.connect(args.url, ssl=ssl_context, max_size=2**20) as ws:
                connected = time.monotonic()
                while time.monotonic() - started < args.seconds:
                    sequence += 1
                    now = time.monotonic()
                    if args.malformed and sequence % 17 == 0:
                        await ws.send(b"\x03bad")
                    else:
                        # Minimal JPEG markers are sufficient for dispatch testing.
                        await ws.send(bytes([MSG_TYPE_CAM]) + b"\xff\xd8\xff\xd9")
                        thermal = struct.pack("<768f", *[
                            22.0 + 4.0 * math.sin((i + sequence) / 40.0)
                            for i in range(768)
                        ])
                        await ws.send(bytes([MSG_TYPE_THERMAL]) + thermal)
                        imu = IMU.pack(
                            sequence, int((now - started) * 1000),
                            0.1, 0.2, 9.80, 1.0, 2.0, 3.0,
                        )
                        await ws.send(bytes([MSG_TYPE_IMU]) + imu)
                    if args.disconnect_every and now - connected >= args.disconnect_every:
                        await ws.close()
                        break
                    await asyncio.sleep(args.delay)
        except Exception as exc:
            print(f"reconnecting after {type(exc).__name__}: {exc}")
            await asyncio.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8081/ws/camera_thermal")
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--delay", type=float, default=0.25, help="slow transmission interval")
    parser.add_argument("--disconnect-every", type=float, default=0)
    parser.add_argument("--malformed", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
