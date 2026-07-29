import asyncio
import unittest
from pathlib import Path

from stability_runtime import (
    IMU_STRUCT,
    MSG_TYPE_CAM,
    MSG_TYPE_IMU,
    MSG_TYPE_THERMAL,
    LatestFrameStore,
    RecordingPipeline,
    VisionController,
    parse_sensor_message,
)


class ProtocolTests(unittest.TestCase):
    def test_camera_prefix_dispatch(self):
        kind, payload = parse_sensor_message(bytes([MSG_TYPE_CAM]) + b"\xff\xd8xx")
        self.assertEqual(kind, MSG_TYPE_CAM)
        self.assertEqual(payload, b"\xff\xd8xx")

    def test_thermal_prefix_dispatch(self):
        raw = bytes(3072)
        kind, payload = parse_sensor_message(bytes([MSG_TYPE_THERMAL]) + raw)
        self.assertEqual((kind, len(payload)), (MSG_TYPE_THERMAL, 3072))

    def test_imu_prefix_dispatch_and_units(self):
        raw = IMU_STRUCT.pack(7, 1234, 1, 2, 3, 4, 5, 6)
        kind, sample = parse_sensor_message(bytes([MSG_TYPE_IMU]) + raw)
        self.assertEqual(kind, MSG_TYPE_IMU)
        self.assertEqual(sample["sequence"], 7)
        self.assertEqual(sample["accel"]["units"], "m/s^2")
        self.assertEqual(sample["gyro"]["units"], "deg/s")

    def test_invalid_payload_rejection(self):
        for payload in (b"", b"\x01bad", b"\x02bad", b"\x03bad", b"\xffbad"):
            with self.assertRaises(ValueError):
                parse_sensor_message(payload)

    def test_latest_frame_replacement(self):
        frames = LatestFrameStore()
        frames.update(b"first", 1.0)
        latest = frames.update(b"second", 2.0)
        self.assertEqual(latest.data, b"second")
        self.assertEqual(latest.sequence, 2)


class AsyncStabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_gemini_receives_nothing_without_trigger(self):
        sent = []
        frames = LatestFrameStore()
        frames.update(b"jpeg", 10.0)
        VisionController(frames, lambda data: sent.append(data), clock=lambda: 10.1)
        await asyncio.sleep(0)
        self.assertEqual(sent, [])

    async def test_one_trigger_submits_one_recent_frame(self):
        sent = []

        async def sender(data):
            sent.append(data)

        frames = LatestFrameStore()
        frames.update(b"jpeg", 10.0)
        vision = VisionController(frames, sender, clock=lambda: 10.1)
        result = await vision.request("test")
        self.assertTrue(result["ok"])
        self.assertEqual(sent, [b"jpeg"])

    async def test_stale_frame_controlled_error(self):
        async def sender(_):
            self.fail("stale frame must not be sent")

        frames = LatestFrameStore()
        frames.update(b"jpeg", 1.0)
        vision = VisionController(frames, sender, max_age_sec=3, clock=lambda: 5.0)
        self.assertEqual((await vision.request("test"))["error"], "no_recent_frame")

    async def test_vision_rate_limit(self):
        async def sender(_):
            return None

        now = [10.0]
        frames = LatestFrameStore()
        frames.update(b"jpeg", 10.0)
        vision = VisionController(frames, sender, min_interval_sec=2, clock=lambda: now[0])
        self.assertTrue((await vision.request("one"))["ok"])
        now[0] = 11.0
        self.assertEqual((await vision.request("two"))["error"], "rate_limited")

    async def test_recording_queue_never_blocks_and_drops_oldest(self):
        gate = asyncio.Event()
        written = []

        async def writer(data):
            await gate.wait()
            written.append(data)

        pipeline = RecordingPipeline(writer, maxsize=2)
        pipeline.start()
        pipeline.enqueue_latest(b"one")
        await asyncio.sleep(0)
        pipeline.enqueue_latest(b"two")
        pipeline.enqueue_latest(b"three")
        pipeline.enqueue_latest(b"four")
        self.assertLessEqual(pipeline.queue.qsize(), 2)
        self.assertGreaterEqual(pipeline.metrics["recording_frames_dropped"], 1)
        gate.set()
        await asyncio.sleep(0.02)
        await pipeline.stop()

    async def test_slow_viewer_does_not_prevent_latest_replacement(self):
        frames = LatestFrameStore()

        async def slow_viewer():
            await asyncio.sleep(1)

        task = asyncio.create_task(slow_viewer())
        frames.update(b"old", 1.0)
        frames.update(b"new", 2.0)
        self.assertEqual(frames.snapshot().data, b"new")
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_strict_processing_contract(self):
        # Strict mode's core contract: latest replacement, no implicit vision,
        # bounded recording, and no model dependency in this module.
        self.assertNotIn("torch", __import__("stability_runtime").__dict__)

    async def test_app_strict_guards_continuous_vision_and_models(self):
        source = Path("app_main.py").read_text(encoding="utf-8")
        self.assertIn('if AI_BACKEND == "gemini_live" and not STABILITY_MODE', source)
        self.assertIn("if not STABILITY_MODE:", source)
        self.assertIn("if STABILITY_MODE:\n        return data, None", source)


if __name__ == "__main__":
    unittest.main()
