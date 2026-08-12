import unittest
from pathlib import Path

from stability_runtime import (
    AudioFreshnessTracker,
    LatencyTracker,
    append_transcription_delta,
    parse_mic_loss_message,
)


def source_with_feature_disabled(source: str, feature: str) -> str:
    """Return the source selected by exact #if FEATURE / #else blocks."""
    output = []
    selecting = []
    enabled = True
    for line in source.splitlines():
        stripped = line.strip()
        if stripped == f"#if {feature}":
            selecting.append(enabled)
            enabled = False
            continue
        if selecting and stripped == "#else":
            enabled = selecting[-1]
            continue
        if selecting and stripped.startswith("#endif"):
            enabled = selecting.pop()
            continue
        if enabled:
            output.append(line)
    return "\n".join(output)


class AudioFreshnessTests(unittest.TestCase):
    def setUp(self):
        self.now = [10.0]
        self.tracker = AudioFreshnessTracker(
            stale_after_ms=500,
            recent_window_sec=2,
            clock=lambda: self.now[0],
        )

    def test_disconnected_connect_suppressed_fresh_stale_and_recovery(self):
        self.assertEqual(self.tracker.snapshot()["state"], "disconnected")
        generation = self.tracker.connect()
        self.assertEqual(generation, 1)
        self.assertEqual(self.tracker.snapshot()["state"], "suppressed")

        self.tracker.set_gate(expected_streaming=True, reason="streaming")
        self.assertEqual(self.tracker.snapshot()["state"], "stale")
        self.tracker.record_pcm(1280)
        fresh = self.tracker.snapshot()
        self.assertEqual(fresh["state"], "fresh")
        self.assertEqual(fresh["audio_recent_chunk_count"], 2)

        self.now[0] += 0.501
        self.assertEqual(self.tracker.snapshot()["state"], "stale")
        self.tracker.set_gate(expected_streaming=False, reason="tts")
        suppressed = self.tracker.snapshot()
        self.assertEqual(suppressed["state"], "suppressed")
        self.assertEqual(suppressed["audio_gate_reason"], "tts")
        self.tracker.set_gate(expected_streaming=True, reason="streaming")
        self.tracker.record_pcm(640)
        self.assertEqual(self.tracker.snapshot()["state"], "fresh")

    def test_generation_increments_and_disconnect_is_red_state(self):
        self.tracker.connect()
        self.tracker.disconnect()
        self.assertEqual(self.tracker.snapshot()["state"], "disconnected")
        self.assertEqual(self.tracker.connect(), 2)

    def test_freshness_snapshot_has_no_credentials(self):
        snapshot = self.tracker.snapshot()
        sensitive = {"api_key", "password", "token", "authorization", "wifi_pass"}
        self.assertTrue(sensitive.isdisjoint(key.lower() for key in snapshot))
        backend = Path("app_main.py").read_text(encoding="utf-8")
        self.assertNotIn("_gk[:6]", backend)
        self.assertNotIn("_gk[-4:]", backend)

    def test_loss_epoch_is_exposed_and_marks_active_turn_degraded(self):
        self.tracker.connect()
        loss = {"epoch": 17, "chunks": 83, "duration_ms": 1660, "reason": "queue_full"}
        self.tracker.record_loss(loss)
        self.assertEqual(self.tracker.snapshot()["last_mic_loss"]["epoch"], 17)
        turns = LatencyTracker()
        turns.start_turn(3)
        self.assertTrue(turns.mark_audio_integrity_degraded(loss, 3))
        active = turns.snapshot()["current_active_turn"]
        self.assertTrue(active["audio_integrity_degraded"])
        self.assertEqual(active["mic_loss"]["chunks"], 83)


class AudioProtocolTests(unittest.TestCase):
    def test_mic_loss_parser_accepts_protocol_and_rejects_secrets_or_bad_ranges(self):
        parsed = parse_mic_loss_message(
            '{"type":"MIC_LOSS","epoch":17,"chunks":83,'
            '"duration_ms":1660,"reason":"queue_full"}'
        )
        self.assertEqual(parsed, {
            "epoch": 17,
            "chunks": 83,
            "duration_ms": 1660,
            "reason": "queue_full",
        })
        self.assertIsNone(parse_mic_loss_message('{"type":"MIC_LOSS","epoch":0,"chunks":1,"duration_ms":20}'))
        self.assertIsNone(parse_mic_loss_message('{"type":"MIC_LOSS","epoch":1,"chunks":0,"duration_ms":0}'))
        self.assertNotIn("token", repr(parsed))

    def test_cumulative_transcription_does_not_duplicate_the_utterance(self):
        parts = []
        append_transcription_delta(parts, "Can you navigate me to the stairs?")
        result = append_transcription_delta(parts, "Can you navigate me to the stairs?")
        self.assertEqual(result, "Can you navigate me to the stairs?")
        self.assertEqual("".join(parts), result)

    def test_firmware_contract_keeps_two_sockets_and_safe_feature_flags(self):
        source = Path("compile/compile.ino").read_text(encoding="utf-8")
        self.assertEqual(source.count("WebsocketsClient ws"), 2)
        self.assertIn("#define ENABLE_AUDIO_40MS_AGGREGATION 0", source)
        self.assertIn("#define ENABLE_LOCAL_BARGE_IN 0", source)
        self.assertIn("AUDIO_MAX_RECONNECT_BACKOFF_MS = 5000", source)
        self.assertIn("recordMicLoss(1, \"queue_full\")", source)
        self.assertNotIn("sendPendingMicLossReport", source)
        self.assertNotIn('"MIC_LOSS"', source)
        failed_start = source.index("if (!audioOk)")
        failed_send = source[
            failed_start:source.index("recordMicPcmSuccessfulSend", failed_start)
        ]
        self.assertIn("aud_ws_ready = false", failed_send)
        self.assertIn("wsAud.close()", failed_send)
        self.assertIn('resetMicQueue("send_failed", true)', failed_send)
        self.assertNotIn("if (audioElapsed >= CAMERA_SEND_UNHEALTHY_MS) {\n        aud_ws_ready = false", source)

    def test_default_firmware_is_one_20ms_packet_without_aggregation_wait(self):
        firmware = Path("compile/compile.ino").read_text(encoding="utf-8")
        default_firmware = source_with_feature_disabled(
            firmware, "ENABLE_AUDIO_40MS_AGGREGATION"
        )
        self.assertIn("const int CHUNK_MS        = 20", default_firmware)
        self.assertIn("const int BYTES_PER_CHUNK = SAMPLE_RATE * CHUNK_MS / 1000 * 2", default_firmware)
        self.assertIn("const uint8_t* sendData = chunk.data", default_firmware)
        self.assertIn("size_t sendBytes = chunk.n", default_firmware)
        self.assertNotIn("secondChunk", default_firmware)
        self.assertNotIn("static uint8_t aggregate", default_firmware)
        self.assertNotIn("pdMS_TO_TICKS(CHUNK_MS)", default_firmware)

    def test_disabled_barge_in_restores_head_tts_capture_suspension(self):
        firmware = Path("compile/compile.ino").read_text(encoding="utf-8")
        default_firmware = source_with_feature_disabled(
            firmware, "ENABLE_LOCAL_BARGE_IN"
        )
        self.assertNotIn("centeredPcmRms", default_firmware)
        self.assertNotIn("observeTtsMicForBargeIn", default_firmware)
        self.assertNotIn("preRoll", default_firmware)
        self.assertNotIn("bargeInControlPending", default_firmware)
        self.assertNotIn('sendAudioTextTimed("BARGE_IN"', default_firmware)
        self.assertIn(
            "if (!run_audio_stream || !aud_ws_ready || tts_playing)",
            default_firmware,
        )

        tts_start = default_firmware.index('s == "TTS:START"')
        tts_end = default_firmware.index('s == "TTS:END"', tts_start)
        start_block = default_firmware[tts_start:tts_end]
        self.assertLess(
            start_block.index("run_audio_stream = false"),
            start_block.index('resetMicQueue("tts_gate", false)'),
        )
        self.assertLess(
            start_block.index('resetMicQueue("tts_gate", false)'),
            start_block.index("tts_playing = true"),
        )
        self.assertIn("run_audio_stream = true", default_firmware)

    def test_generation_local_loss_and_safe_age_telemetry_remain(self):
        firmware = Path("compile/compile.ino").read_text(encoding="utf-8")

        aud_events = firmware.index("wsAud.onEvent")
        opened = firmware.index("WebsocketsEvent::ConnectionOpened", aud_events)
        closed = firmware.index("WebsocketsEvent::ConnectionClosed", opened)
        generation_block = firmware[opened:closed]
        self.assertIn("audioConnectionGeneration++", generation_block)
        self.assertIn('resetMicQueue("connection_generation", false)', generation_block)
        self.assertIn("static void recordMicLoss", firmware)
        self.assertIn("lossEpoch=%lu lossChunks=%lu lossDurationMs=%lu", firmware)
        self.assertIn("static uint32_t safeElapsedMs", firmware)
        self.assertIn("return (int32_t)elapsed < 0 ? 0 : elapsed", firmware)
        self.assertIn("safeElapsedMs(now, lastMicSendMs)", firmware)
        self.assertIn("safeElapsedMs(now, lastAudPollMs)", firmware)

    def test_backend_normalizes_pcm_and_owns_tts_freshness_gate(self):
        backend = Path("app_main.py").read_text(encoding="utf-8")
        self.assertIn("range(0, len(chunk), PCM_20MS_BYTES_16K_MONO)", backend)
        self.assertIn("received_messages=", backend)
        self.assertIn("received_pcm_chunks=", backend)
        self.assertIn("reopen_delay", backend)
        self.assertIn('reason="tts"', backend)
        self.assertIn('reason="streaming"', backend)


if __name__ == "__main__":
    unittest.main()
