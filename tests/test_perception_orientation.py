import asyncio
import io
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np

from perception_orientation import (
    RgbCanonicalizerTelemetry,
    canonicalize_rgb_jpeg,
    canonicalize_thermal_payload,
    map_thermal_to_rgb_normalized,
    queue_latest_raw_rgb,
    run_latest_rgb_canonicalizer,
    summarize_thermal_grid,
)


class RgbOrientationTests(unittest.TestCase):
    def test_rgb_rotates_counterclockwise_swaps_shape_and_does_not_mirror(self):
        source = np.zeros((60, 80, 3), dtype=np.uint8)
        source[:30, :40] = (0, 0, 255)       # red, top-left
        source[:30, 40:] = (0, 255, 0)       # green, top-right
        source[30:, :40] = (255, 0, 0)       # blue, bottom-left
        source[30:, 40:] = (0, 255, 255)     # yellow, bottom-right
        ok, encoded = cv2.imencode(".jpg", source, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        self.assertTrue(ok)

        canonical_bytes = canonicalize_rgb_jpeg(encoded.tobytes())
        self.assertIsNotNone(canonical_bytes)
        canonical = cv2.imdecode(np.frombuffer(canonical_bytes, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(canonical.shape[:2], (80, 60))

        # A CCW rotation maps right-side source corners to the top; retaining
        # their order distinguishes rotation from an accidental mirror.
        samples = [canonical[10, 10], canonical[10, 50], canonical[70, 10], canonical[70, 50]]
        expected = [(0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 0, 0)]
        for actual, wanted in zip(samples, expected):
            self.assertLess(np.abs(actual.astype(int) - np.asarray(wanted)).max(), 25)

    def test_backend_fans_out_only_after_canonicalization_without_gemini_rotation(self):
        source = Path("app_main.py").read_text(encoding="utf-8")
        fanout = source.split("async def _handle_camera_frame", 1)[1].split(
            "def _retain_latest_thermal", 1
        )[0]
        self.assertIn("run_latest_rgb_canonicalizer", source)
        for consumer in ("latest_rgb.update", "recording_pipeline.enqueue_latest", "last_frames.append"):
            self.assertIn(consumer, fanout)
        self.assertNotIn("rotate_jpeg", source)
        self.assertIn("rotation_provider=lambda: 0", source)
        self.assertIn("display_rotation_deg=0", source)
        health_endpoint = source.split('def health():', 1)[1].split(
            '@app.get("/api/yolo/detections")', 1
        )[0]
        self.assertIn('"rgb_canonicalizer": rgb_canonicalizer_telemetry.health()', health_endpoint)


class RgbLatestOnlyTests(unittest.IsolatedAsyncioTestCase):
    async def _wait_for(self, predicate):
        for _ in range(100):
            if predicate():
                return
            await asyncio.sleep(0.01)
        self.fail("timed out waiting for canonicalizer state")

    async def test_canonicalizer_keeps_one_in_flight_and_only_newest_pending(self):
        holder = {"data": None}
        event = asyncio.Event()
        telemetry = RgbCanonicalizerTelemetry(history_size=2)
        started = threading.Event()
        release = threading.Event()
        published = []

        def transform(data):
            if data == b"frame-1":
                started.set()
                release.wait(timeout=1)
            return b"canonical-" + data

        async def publish(data, received_at):
            published.append((data, received_at))

        task = asyncio.create_task(
            run_latest_rgb_canonicalizer(
                holder, event, publish, telemetry, transform=transform
            )
        )
        queue_latest_raw_rgb(b"frame-1", 1.0, holder, event, telemetry)
        self.assertTrue(await asyncio.to_thread(started.wait, 1))
        queue_latest_raw_rgb(b"frame-2", 2.0, holder, event, telemetry)
        queue_latest_raw_rgb(b"frame-3", 3.0, holder, event, telemetry)
        release.set()
        await self._wait_for(lambda: len(published) == 2)
        running = telemetry.health()
        self.assertTrue(running["worker_running"])
        self.assertEqual(running["raw_frames_received"], 3)
        self.assertEqual(running["canonical_frames_completed"], 2)
        self.assertEqual(running["canonical_pending_replaced"], 1)
        self.assertEqual(running["canonical_failures"], 0)
        self.assertEqual(running["timing_history_size"], 2)
        self.assertEqual(running["timing_history_limit"], 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        stopped = telemetry.health()
        self.assertFalse(stopped["worker_running"])
        self.assertFalse(stopped["worker_alive"])
        self.assertEqual(stopped["worker_state"], "stopped")
        self.assertEqual(published, [
            (b"canonical-frame-1", 1.0),
            (b"canonical-frame-3", 3.0),
        ])

    async def test_malformed_jpeg_does_not_block_later_valid_frame(self):
        holder = {"data": None}
        event = asyncio.Event()
        telemetry = RgbCanonicalizerTelemetry()
        published = []
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)

        async def publish(data, received_at):
            published.append((data, received_at))

        task = asyncio.create_task(
            run_latest_rgb_canonicalizer(holder, event, publish, telemetry)
        )
        with redirect_stdout(io.StringIO()):
            queue_latest_raw_rgb(b"malformed", 1.0, holder, event, telemetry)
            await self._wait_for(lambda: telemetry.health()["canonical_failures"] == 1)
            queue_latest_raw_rgb(encoded.tobytes(), 2.0, holder, event, telemetry)
            await self._wait_for(lambda: len(published) == 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        health = telemetry.health()
        self.assertEqual(health["raw_frames_received"], 2)
        self.assertEqual(health["canonical_frames_completed"], 1)
        self.assertEqual(health["canonical_failures"], 1)

    async def test_unexpected_transform_and_publish_exceptions_are_contained(self):
        holder = {"data": None}
        event = asyncio.Event()
        telemetry = RgbCanonicalizerTelemetry()
        published = []

        def transform(data):
            if data == b"transform-error":
                raise RuntimeError("synthetic transform failure")
            return b"canonical-" + data

        async def publish(data, received_at):
            if data == b"canonical-publish-error":
                raise RuntimeError("synthetic publish failure")
            published.append((data, received_at))

        task = asyncio.create_task(run_latest_rgb_canonicalizer(
            holder, event, publish, telemetry, transform=transform
        ))
        with redirect_stdout(io.StringIO()):
            queue_latest_raw_rgb(b"transform-error", 1.0, holder, event, telemetry)
            await self._wait_for(lambda: telemetry.health()["canonical_failures"] == 1)
            queue_latest_raw_rgb(b"publish-error", 2.0, holder, event, telemetry)
            await self._wait_for(lambda: telemetry.health()["canonical_failures"] == 2)
            queue_latest_raw_rgb(b"valid", 3.0, holder, event, telemetry)
            await self._wait_for(lambda: len(published) == 1)
        self.assertFalse(task.done())
        health = telemetry.health()
        self.assertEqual(health["canonical_frames_completed"], 1)
        self.assertEqual(health["canonical_failures"], 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    def test_timing_history_is_bounded(self):
        telemetry = RgbCanonicalizerTelemetry(history_size=2)
        telemetry.record_completed(1.0)
        telemetry.record_completed(2.0)
        telemetry.record_completed(3.0)
        health = telemetry.health()
        self.assertEqual(health["timing_history_size"], 2)
        self.assertEqual(health["timing_history_limit"], 2)
        self.assertEqual(health["canonicalization_ms"]["latest"], 3.0)
        self.assertEqual(health["canonicalization_ms"]["average"], 2.5)

    def test_health_distinguishes_unexpected_exit(self):
        telemetry = RgbCanonicalizerTelemetry()
        telemetry.mark_running()
        telemetry.mark_unexpected_exit("SyntheticWorkerExit")
        health = telemetry.health()
        self.assertFalse(health["worker_running"])
        self.assertFalse(health["worker_alive"])
        self.assertEqual(health["worker_state"], "unexpected_exit")
        self.assertEqual(health["last_failure_stage"], "worker")


class ThermalOrientationTests(unittest.TestCase):
    def test_neutral_coarse_calibration_preserves_normalized_coordinates(self):
        self.assertEqual(map_thermal_to_rgb_normalized(0.5, 0.5), (0.5, 0.5))
        mapped = map_thermal_to_rgb_normalized(0.1, 0.9)
        self.assertAlmostEqual(mapped[0], 0.1)
        self.assertAlmostEqual(mapped[1], 0.9)

    def test_calibration_offsets_and_scales_are_axis_specific(self):
        self.assertEqual(
            map_thermal_to_rgb_normalized(0.5, 0.5, offset_x=0.1),
            (0.6, 0.5),
        )
        self.assertEqual(
            map_thermal_to_rgb_normalized(0.5, 0.5, offset_y=-0.2),
            (0.5, 0.3),
        )
        self.assertEqual(
            map_thermal_to_rgb_normalized(0.25, 0.75, scale_x=2, scale_y=0.5),
            (0.0, 0.625),
        )

    def test_calibration_clamps_only_when_explicitly_requested_for_display(self):
        raw = map_thermal_to_rgb_normalized(1.0, 0.0, offset_x=0.25, offset_y=-0.25)
        display = map_thermal_to_rgb_normalized(
            1.0, 0.0, offset_x=0.25, offset_y=-0.25, clamp_for_display=True
        )
        self.assertEqual(raw, (1.25, -0.25))
        self.assertEqual(display, (1.0, 0.0))

    def test_physical_corners_map_to_same_canonical_corners(self):
        raw = np.zeros((24, 32), dtype=np.float32)
        # Physical testing establishes this raw-corner correspondence after
        # accounting for the sensor's rotated mounting.
        raw[-1, -1] = 11  # physical upper-left
        raw[0, -1] = 12   # physical upper-right
        raw[-1, 0] = 21   # physical lower-left
        raw[0, 0] = 22    # physical lower-right
        canonical = canonicalize_thermal_payload(raw.astype("<f4").tobytes())
        self.assertEqual(canonical.shape, (32, 24))
        self.assertEqual(canonical[0, 0], 11)
        self.assertEqual(canonical[0, -1], 12)
        self.assertEqual(canonical[-1, 0], 21)
        self.assertEqual(canonical[-1, -1], 22)

    def test_canonicalization_preserves_temperature_scalars(self):
        raw = np.arange(24 * 32, dtype=np.float32).reshape(24, 32) / 10
        canonical = canonicalize_thermal_payload(raw.astype("<f4").tobytes())
        np.testing.assert_array_equal(np.sort(canonical, axis=None), np.sort(raw, axis=None))

    def test_regions_and_hotspots_use_canonical_spatial_semantics(self):
        canonical = np.zeros((32, 24), dtype=np.float32)
        canonical[:10, :8] = 11
        canonical[:10, 8:16] = 12
        canonical[:10, 16:] = 13
        canonical[10:21, :8] = 21
        canonical[10:21, 8:16] = 22
        canonical[10:21, 16:] = 23
        canonical[21:, :8] = 31
        canonical[21:, 8:16] = 32
        canonical[21:, 16:] = 33
        canonical[0, 0] = 99
        raw = np.ascontiguousarray(
            np.rot90(np.fliplr(canonical), k=-1), dtype="<f4"
        )
        oriented = canonicalize_thermal_payload(raw.tobytes())
        facts = summarize_thermal_grid(oriented, age_sec=0.25)
        self.assertEqual(facts["hotspot"]["position"], "upper left")
        self.assertGreater(facts["region_mean_c"]["lower_right"], facts["region_mean_c"]["upper_right"])
        self.assertEqual(facts["measurement_age_sec"], 0.25)

    def test_heatmap_has_no_second_rotation(self):
        source = Path("app_main.py").read_text(encoding="utf-8")
        colorizer = source.split("def _colorize_thermal", 1)[1].split(
            '@app.websocket("/ws/thermal")', 1
        )[0]
        self.assertNotIn("rot90", colorizer)
        self.assertNotIn("rotation_deg", colorizer)
        browser = Path("templates/index.html").read_text(encoding="utf-8")
        self.assertNotIn("scaleX(-1)", browser)

    def test_calibration_settings_are_neutral_and_do_not_enter_colorizer(self):
        source = Path("app_main.py").read_text(encoding="utf-8")
        config = source.split("thermal_display_config", 1)[1].split(
            "VISION_FRAME_MAX_AGE_SEC", 1
        )[0]
        self.assertIn('"calibration_offset_x": 0.0', config)
        self.assertIn('"calibration_offset_y": 0.0', config)
        self.assertIn('"calibration_scale_x": 1.0', config)
        self.assertIn('"calibration_scale_y": 1.0', config)
        colorizer = source.split("def _colorize_thermal", 1)[1].split(
            '@app.websocket("/ws/thermal")', 1
        )[0]
        self.assertNotIn("map_thermal_to_rgb_normalized", colorizer)
        self.assertNotIn("calibration_offset", colorizer)

        browser = Path("templates/index.html").read_text(encoding="utf-8")
        self.assertIn("Session uses normal RGB with thermal as a small inset", browser)
        self.assertIn('id="thermalAlignRgb"', browser)

    def test_shared_canonical_matrix_feeds_retention_facts_and_heatmap(self):
        source = Path("app_main.py").read_text(encoding="utf-8")
        retain = source.split("def _retain_latest_thermal", 1)[1].split(
            "async def _handle_thermal_frame", 1
        )[0]
        self.assertIn("latest_thermal_matrix = canonical", retain)
        self.assertIn("latest_thermal.update(canonical.astype", retain)
        self.assertIn("facts = summarize_thermal_grid(matrix", source)
        processor = source.split("async def _thermal_processor", 1)[1].split(
            "thermal_processor_task", 1
        )[0]
        self.assertIn("_prepare_thermal_frame_blocking, frame", processor)


if __name__ == "__main__":
    unittest.main()
