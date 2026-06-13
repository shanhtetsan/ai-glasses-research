# audio_compressor.py
# -*- coding: utf-8 -*-
"""
Audio compression utility - reduces network bandwidth usage.
Supports converting 16kHz 16-bit PCM to smaller formats.
"""
import os
import wave
import struct
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class AudioCompressor:
    """Audio compressor - supports multiple compression algorithms"""

    @staticmethod
    def pcm16_to_ulaw(pcm_data: bytes) -> bytes:
        """
        Convert 16-bit PCM to 8-bit μ-law.
        Compression ratio: 50% (16bit → 8bit)
        """
        # Parse 16-bit PCM
        samples = np.frombuffer(pcm_data, dtype=np.int16)

        # μ-law compression
        ulaw_data = bytearray()
        for sample in samples:
            ulaw_byte = AudioCompressor._linear_to_ulaw(sample)
            ulaw_data.append(ulaw_byte)

        return bytes(ulaw_data)

    @staticmethod
    def ulaw_to_pcm16(ulaw_data: bytes) -> bytes:
        """Convert 8-bit μ-law back to 16-bit PCM"""
        pcm_samples = []
        for ulaw_byte in ulaw_data:
            pcm_sample = AudioCompressor._ulaw_to_linear(ulaw_byte)
            pcm_samples.append(pcm_sample)

        return np.array(pcm_samples, dtype=np.int16).tobytes()

    @staticmethod
    def _linear_to_ulaw(sample: int) -> int:
        """Convert 16-bit linear PCM to μ-law"""
        # μ-law encoding table
        ULAW_MAX = 0x1FFF
        ULAW_BIAS = 0x84

        # Clamp to valid range
        sample = max(-32768, min(32767, sample))

        # Get the sign bit
        sign = 0
        if sample < 0:
            sign = 0x80
            sample = -sample

        # Add bias
        sample = sample + ULAW_BIAS

        # Clamp to maximum value
        if sample > ULAW_MAX:
            sample = ULAW_MAX

        # Find the exponent and mantissa
        exponent = 7
        for exp in range(7, -1, -1):
            if sample & (0x4000 >> exp):
                exponent = exp
                break

        mantissa = (sample >> (exponent + 3)) & 0x0F
        ulawbyte = ~(sign | (exponent << 4) | mantissa) & 0xFF

        return ulawbyte

    @staticmethod
    def _ulaw_to_linear(ulawbyte: int) -> int:
        """Convert μ-law to 16-bit linear PCM"""
        ULAW_BIAS = 0x84

        ulawbyte = ~ulawbyte & 0xFF
        sign = ulawbyte & 0x80
        exponent = (ulawbyte >> 4) & 0x07
        mantissa = ulawbyte & 0x0F

        sample = ((mantissa << 3) + ULAW_BIAS) << exponent

        if sign:
            sample = -sample

        return sample

    @staticmethod
    def pcm16_to_adpcm(pcm_data: bytes) -> bytes:
        """
        Convert 16-bit PCM to 4-bit ADPCM.
        Compression ratio: 75% (16bit → 4bit)
        Maintains good speech quality.
        """
        samples = np.frombuffer(pcm_data, dtype=np.int16)

        # IMA ADPCM step-size table
        step_table = [
            7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
            19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
            50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
            130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
            337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
            876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
            2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
            5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
            15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
        ]

        # Index adjustment table
        index_table = [-1, -1, -1, -1, 2, 4, 6, 8]

        # Initialization
        adpcm_data = bytearray()
        predicted = 0
        step_index = 0

        # Pack every two samples into one byte
        for i in range(0, len(samples), 2):
            byte = 0

            for j in range(2):
                if i + j < len(samples):
                    sample = samples[i + j]

                    # Calculate the difference
                    diff = sample - predicted

                    # Quantize
                    step = step_table[step_index]
                    adpcm_sample = 0

                    if diff < 0:
                        adpcm_sample = 8
                        diff = -diff

                    if diff >= step:
                        adpcm_sample |= 4
                        diff -= step

                    step >>= 1
                    if diff >= step:
                        adpcm_sample |= 2
                        diff -= step

                    step >>= 1
                    if diff >= step:
                        adpcm_sample |= 1

                    # Update the predicted value
                    step = step_table[step_index]
                    diff = 0
                    if adpcm_sample & 4:
                        diff += step
                    step >>= 1
                    if adpcm_sample & 2:
                        diff += step
                    step >>= 1
                    if adpcm_sample & 1:
                        diff += step
                    step >>= 1
                    diff += step

                    if adpcm_sample & 8:
                        predicted -= diff
                    else:
                        predicted += diff

                    # Clamp the predicted value to valid range
                    if predicted > 32767:
                        predicted = 32767
                    elif predicted < -32768:
                        predicted = -32768

                    # Update the step-size index
                    step_index += index_table[adpcm_sample & 7]
                    if step_index < 0:
                        step_index = 0
                    elif step_index > 88:
                        step_index = 88

                    # Pack into the output byte
                    if j == 0:
                        byte = adpcm_sample
                    else:
                        byte |= (adpcm_sample << 4)

            adpcm_data.append(byte)

        # Prepend header: initial predicted value and step-size index
        header = struct.pack('<hB', predicted, step_index)
        return header + bytes(adpcm_data)

    @staticmethod
    def adpcm_to_pcm16(adpcm_data: bytes) -> bytes:
        """Convert 4-bit ADPCM back to 16-bit PCM"""
        if len(adpcm_data) < 3:
            return b''

        # Read the header
        predicted, step_index = struct.unpack('<hB', adpcm_data[:3])
        adpcm_bytes = adpcm_data[3:]

        # IMA ADPCM step-size table
        step_table = [
            7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
            19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
            50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
            130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
            337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
            876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
            2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
            5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
            15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
        ]

        # Index adjustment table
        index_table = [-1, -1, -1, -1, 2, 4, 6, 8]

        pcm_samples = []

        for byte in adpcm_bytes:
            # Decode two 4-bit samples from this byte
            for shift in [0, 4]:
                adpcm_sample = (byte >> shift) & 0x0F

                # Calculate the difference
                step = step_table[step_index]
                diff = 0

                if adpcm_sample & 4:
                    diff += step
                step >>= 1
                if adpcm_sample & 2:
                    diff += step
                step >>= 1
                if adpcm_sample & 1:
                    diff += step
                step >>= 1
                diff += step

                if adpcm_sample & 8:
                    predicted -= diff
                else:
                    predicted += diff

                # Clamp to valid range
                if predicted > 32767:
                    predicted = 32767
                elif predicted < -32768:
                    predicted = -32768

                pcm_samples.append(predicted)

                # Update the step-size index
                step_index += index_table[adpcm_sample & 7]
                if step_index < 0:
                    step_index = 0
                elif step_index > 88:
                    step_index = 88

        return np.array(pcm_samples, dtype=np.int16).tobytes()

    @staticmethod
    def downsample_pcm16(pcm_data: bytes, from_rate: int = 16000, to_rate: int = 8000) -> bytes:
        """
        Downsample PCM audio (optional).
        16kHz → 8kHz reduces data by another 50%.
        """
        if from_rate == to_rate:
            return pcm_data

        # Parse PCM data
        samples = np.frombuffer(pcm_data, dtype=np.int16)

        # Simple downsampling: take every other sample
        if from_rate == 16000 and to_rate == 8000:
            downsampled = samples[::2]
        else:
            # More complex resampling requires scipy
            ratio = to_rate / from_rate
            new_length = int(len(samples) * ratio)
            downsampled = np.interp(
                np.linspace(0, len(samples) - 1, new_length),
                np.arange(len(samples)),
                samples
            ).astype(np.int16)

        return downsampled.tobytes()


class CompressedAudioCache:
    """Compressed audio cache"""

    def __init__(self, compression_type: str = "adpcm", use_downsample: bool = False):
        """
        compression_type: "none", "ulaw", "adpcm"
        """
        self.compression_type = compression_type
        self.use_downsample = use_downsample
        self._cache = {}  # {filepath: compressed_data}
        self._original_sizes = {}  # {filepath: original_size}

    def load_and_compress(self, filepath: str) -> Optional[bytes]:
        """Load and compress an audio file (always converts to 8 kHz)"""
        if filepath in self._cache:
            return self._cache[filepath]

        try:
            with wave.open(filepath, 'rb') as wav:
                # Check audio format
                channels = wav.getnchannels()
                sampwidth = wav.getsampwidth()
                framerate = wav.getframerate()

                if channels != 1:
                    logger.warning(f"{filepath} is not mono")
                if sampwidth != 2:
                    logger.warning(f"{filepath} is not 16-bit audio")

                # Read all audio frames
                frames = wav.readframes(wav.getnframes())

                # If stereo, convert to mono
                if channels == 2:
                    import audioop
                    frames = audioop.tomono(frames, sampwidth, 1, 0)

                # Always convert to 8 kHz (using ratecv to preserve pitch and speed)
                if framerate != 8000:
                    import audioop
                    frames, _ = audioop.ratecv(frames, sampwidth, 1, framerate, 8000, None)
                    framerate = 8000

                # Record original size (after conversion)
                self._original_sizes[filepath] = len(frames)

                # Compress
                if self.compression_type == "ulaw":
                    compressed = AudioCompressor.pcm16_to_ulaw(frames)
                    # Prepend a simple header (1-byte type tag + 4-byte original length)
                    header = struct.pack('!BI', 0x01, len(frames))  # 0x01 = μ-law
                    compressed = header + compressed
                elif self.compression_type == "adpcm":
                    compressed = AudioCompressor.pcm16_to_adpcm(frames)
                    # Prepend a simple header (1-byte type tag + 4-byte original length)
                    header = struct.pack('!BI', 0x02, len(frames))  # 0x02 = ADPCM
                    compressed = header + compressed
                else:
                    compressed = frames

                self._cache[filepath] = compressed

                # Log the compression ratio
                compression_ratio = len(compressed) / self._original_sizes[filepath]
                logger.info(f"[COMPRESS] {os.path.basename(filepath)}: "
                          f"{self._original_sizes[filepath]} -> {len(compressed)} bytes "
                          f"({compression_ratio:.1%})")

                return compressed

        except Exception as e:
            logger.error(f"Failed to compress audio {filepath}: {e}")
            return None

    def decompress(self, compressed_data: bytes) -> Optional[bytes]:
        """Decompress audio data"""
        if not compressed_data or len(compressed_data) < 5:
            return compressed_data

        try:
            # Check the header
            compression_type = compressed_data[0]
            if compression_type == 0x01:  # μ-law marker
                header_size = 5
                original_length = struct.unpack('!I', compressed_data[1:5])[0]
                ulaw_data = compressed_data[header_size:]

                # μ-law decompression
                pcm_data = AudioCompressor.ulaw_to_pcm16(ulaw_data)

                return pcm_data
            elif compression_type == 0x02:  # ADPCM marker
                header_size = 5
                original_length = struct.unpack('!I', compressed_data[1:5])[0]
                adpcm_data = compressed_data[header_size:]

                # ADPCM decompression
                pcm_data = AudioCompressor.adpcm_to_pcm16(adpcm_data)

                return pcm_data
            else:
                # Uncompressed data — return as-is
                return compressed_data

        except Exception as e:
            logger.error(f"Failed to decompress audio: {e}")
            return compressed_data

    def get_compression_stats(self) -> dict:
        """Get compression statistics"""
        total_original = sum(self._original_sizes.values())
        total_compressed = sum(len(data) for data in self._cache.values())

        return {
            "files_cached": len(self._cache),
            "total_original_size": total_original,
            "total_compressed_size": total_compressed,
            "compression_ratio": total_compressed / total_original if total_original > 0 else 0,
            "bytes_saved": total_original - total_compressed
        }


# Global compressed-audio cache instance
# Default: ADPCM compression — better audio quality, decent compression ratio (75%)
# Configurable via environment variable AIGLASS_COMPRESS_TYPE: none, ulaw, adpcm
import os
compression_type = os.getenv("AIGLASS_COMPRESS_TYPE", "adpcm").lower()
if compression_type not in ["none", "ulaw", "adpcm"]:
    compression_type = "adpcm"
compressed_audio_cache = CompressedAudioCache(compression_type=compression_type, use_downsample=False)
