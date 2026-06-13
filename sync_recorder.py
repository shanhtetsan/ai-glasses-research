# sync_recorder.py
# Synchronously records ESP32 video stream and audio output.
# Automatically aligns video and audio timelines.

import os
import cv2
import wave
import numpy as np
import threading
import time
from datetime import datetime
from collections import deque
import struct

class SyncRecorder:
    """Synchronous recorder — video + audio timeline alignment"""

    def __init__(self, output_dir="recordings", fps=15.0):
        """
        Initialize the recorder.
        :param output_dir: output directory
        :param fps: video frame rate (default 15 fps)
        """
        self.output_dir = output_dir
        self.fps = fps
        self.frame_duration = 1.0 / fps  # Duration per frame (seconds)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Recording state
        self.is_recording = False
        self.start_time = None

        # Video writer
        self.video_writer = None
        self.video_path = None
        self.last_frame = None
        self.frame_count = 0

        # Audio writer
        self.audio_writer = None
        self.audio_path = None
        self.audio_buffer = bytearray()
        self.last_audio_time = 0.0

        # Audio parameters (ESP32 standard: 16kHz, 16-bit, Mono)
        self.sample_rate = 16000
        self.sample_width = 2  # 16-bit = 2 bytes
        self.channels = 1

        # Thread safety
        self.lock = threading.Lock()

        # Performance monitoring
        self.frames_written = 0
        self.audio_bytes_written = 0
        self.last_log_time = time.time()

        print(f"[RECORDER] Recorder initialized - FPS={fps}, output dir={output_dir}")

    def start_recording(self):
        """Start a new recording session."""
        if self.is_recording:
            print("[RECORDER] Warning: already recording")
            return False

        # Generate filename (timestamp-based)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(self.output_dir, f"video_{timestamp}.avi")
        self.audio_path = os.path.join(self.output_dir, f"audio_{timestamp}.wav")

        # Reset state
        self.start_time = time.time()
        self.last_audio_time = 0.0
        self.frame_count = 0
        self.frames_written = 0
        self.audio_bytes_written = 0
        self.audio_buffer.clear()
        self.last_frame = None

        # Initialize audio file
        try:
            self.audio_writer = wave.open(self.audio_path, 'wb')
            self.audio_writer.setnchannels(self.channels)
            self.audio_writer.setsampwidth(self.sample_width)
            self.audio_writer.setframerate(self.sample_rate)
        except Exception as e:
            print(f"[RECORDER] Audio file initialization failed: {e}")
            return False

        self.is_recording = True
        print(f"[RECORDER] Recording started")
        print(f"  Video: {self.video_path}")
        print(f"  Audio: {self.audio_path}")
        return True

    def add_frame(self, jpeg_data: bytes):
        """
        Add a video frame (raw JPEG data).
        :param jpeg_data: JPEG image data
        """
        if not self.is_recording:
            return

        try:
            with self.lock:
                # Decode JPEG
                arr = np.frombuffer(jpeg_data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                if frame is None:
                    print(f"[RECORDER] Warning: frame decode failed")
                    return

                # First frame: initialize video writer
                if self.video_writer is None:
                    height, width = frame.shape[:2]
                    # Use XVID encoder (good Windows compatibility)
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    self.video_writer = cv2.VideoWriter(
                        self.video_path,
                        fourcc,
                        self.fps,
                        (width, height)
                    )

                    if not self.video_writer.isOpened():
                        print(f"[RECORDER] Error: video writer initialization failed")
                        self.is_recording = False
                        return

                    print(f"[RECORDER] Video writer initialized: {width}x{height} @ {self.fps}fps")

                # Write frame
                self.video_writer.write(frame)
                self.frame_count += 1
                self.frames_written += 1
                self.last_frame = frame

                # Calculate current video duration (seconds)
                current_video_time = self.frame_count * self.frame_duration

                # Audio sync: pad silence to match video duration
                self._sync_audio_to_video(current_video_time)

                # Performance log (every 10s)
                now = time.time()
                if now - self.last_log_time > 10.0:
                    elapsed = now - self.start_time
                    avg_fps = self.frames_written / elapsed if elapsed > 0 else 0
                    audio_duration = self.audio_bytes_written / (self.sample_rate * self.sample_width)
                    print(f"[RECORDER] Recording - frames={self.frames_written}, "
                          f"actual FPS={avg_fps:.1f}, "
                          f"video duration={current_video_time:.1f}s, "
                          f"audio duration={audio_duration:.1f}s")
                    self.last_log_time = now

        except Exception as e:
            print(f"[RECORDER] Failed to add frame: {e}")
            import traceback
            traceback.print_exc()

    def add_audio(self, pcm_data: bytes, text: str = ""):
        """
        Add audio data (PCM 16-bit).
        :param pcm_data: PCM audio data
        :param text: voice text (for logging)
        """
        if not self.is_recording:
            return

        try:
            with self.lock:
                # Current video duration
                current_video_time = self.frame_count * self.frame_duration

                # Pad silence to video duration before adding audio
                self._sync_audio_to_video(current_video_time)

                # Write actual audio
                self.audio_writer.writeframes(pcm_data)
                audio_duration = len(pcm_data) / (self.sample_rate * self.sample_width)
                self.last_audio_time = current_video_time + audio_duration
                self.audio_bytes_written += len(pcm_data)

                if text:
                    print(f"[RECORDER] Recording audio: {text[:30]}... (time={current_video_time:.2f}s, duration={audio_duration:.2f}s)")

        except Exception as e:
            print(f"[RECORDER] Failed to add audio: {e}")

    def _sync_audio_to_video(self, video_time: float):
        """
        Sync audio to video duration (pad with silence).
        :param video_time: current video duration (seconds)
        """
        # Compute silence duration to pad
        silence_duration = video_time - self.last_audio_time

        if silence_duration > 0.01:  # Only pad if gap > 10ms
            # Generate silence data
            silence_samples = int(silence_duration * self.sample_rate)
            silence_bytes = silence_samples * self.sample_width
            silence_data = b'\x00' * silence_bytes

            # Write silence
            self.audio_writer.writeframes(silence_data)
            self.audio_bytes_written += len(silence_data)
            self.last_audio_time = video_time

    def stop_recording(self):
        """Stop recording and save files."""
        if not self.is_recording:
            return

        print("[RECORDER] Saving recording files...")
        self.is_recording = False

        with self.lock:
            # Final audio sync
            try:
                if self.frame_count > 0:
                    final_video_time = self.frame_count * self.frame_duration
                    self._sync_audio_to_video(final_video_time)
            except Exception as e:
                print(f"[RECORDER] Final audio sync failed: {e}")

            # Close video writer (critical step)
            if self.video_writer is not None:
                try:
                    print("[RECORDER] Closing video writer...")
                    self.video_writer.release()
                    print("[RECORDER] Video writer closed")
                except Exception as e:
                    print(f"[RECORDER] Failed to close video writer: {e}")
                finally:
                    self.video_writer = None

            # Close audio writer
            if self.audio_writer is not None:
                try:
                    print("[RECORDER] Closing audio writer...")
                    self.audio_writer.close()
                    print("[RECORDER] Audio writer closed")
                except Exception as e:
                    print(f"[RECORDER] Failed to close audio writer: {e}")
                finally:
                    self.audio_writer = None

            # Summary statistics
            try:
                elapsed = time.time() - self.start_time if self.start_time else 0
                video_duration = self.frame_count * self.frame_duration
                audio_duration = self.audio_bytes_written / (self.sample_rate * self.sample_width)

                print(f"\n{'='*60}")
                print(f"[RECORDER] Recording complete")
                print(f"{'='*60}")
                print(f"  Total elapsed: {elapsed:.1f}s")
                print(f"\n  Video: {self.video_path}")
                print(f"    - Frames: {self.frames_written}")
                print(f"    - Duration: {video_duration:.2f}s")
                if elapsed > 0:
                    print(f"    - Average FPS: {self.frames_written/elapsed:.1f}")
                print(f"\n  Audio: {self.audio_path}")
                print(f"    - Data size: {self.audio_bytes_written/1024:.1f} KB")
                print(f"    - Duration: {audio_duration:.2f}s")
                print(f"\n  Time difference: {abs(video_duration - audio_duration):.3f}s")

                # Verify files
                if os.path.exists(self.video_path):
                    video_size = os.path.getsize(self.video_path) / 1024 / 1024
                    print(f"  Video file size: {video_size:.2f} MB ✓")
                else:
                    print(f"  ⚠ Warning: video file not generated")

                if os.path.exists(self.audio_path):
                    audio_size = os.path.getsize(self.audio_path) / 1024
                    print(f"  Audio file size: {audio_size:.2f} KB ✓")
                else:
                    print(f"  ⚠ Warning: audio file not generated")

                print(f"{'='*60}\n")
            except Exception as e:
                print(f"[RECORDER] Failed to display statistics: {e}")


# Global recorder instance
_global_recorder = None
_recorder_lock = threading.Lock()

def get_recorder():
    """Get the global recorder instance."""
    global _global_recorder
    with _recorder_lock:
        if _global_recorder is None:
            _global_recorder = SyncRecorder()
        return _global_recorder

def start_recording():
    """Start recording."""
    recorder = get_recorder()
    return recorder.start_recording()

def stop_recording():
    """Stop recording."""
    recorder = get_recorder()
    recorder.stop_recording()

def record_frame(jpeg_data: bytes):
    """Record a frame (called externally)."""
    recorder = get_recorder()
    if recorder.is_recording:
        recorder.add_frame(jpeg_data)

def record_audio(pcm_data: bytes, text: str = ""):
    """Record audio (called externally)."""
    recorder = get_recorder()
    if recorder.is_recording:
        recorder.add_audio(pcm_data, text)
