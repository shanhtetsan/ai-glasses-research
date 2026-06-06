# audio_player.py
# 处理预录音频文件的播放，通过ESP32扬声器输出
# Handles playback of pre-recorded audio files through the ESP32 speaker

import os
import wave
import json
import asyncio
import threading
import queue
import time
from audio_stream import broadcast_pcm16_realtime
from audio_compressor import compressed_audio_cache, AudioCompressor

# 导入录制器（避免循环导入，在需要时动态导入）
# Import recorder (avoid circular import, import dynamically when needed)
_recorder_imported = False
_sync_recorder = None

def _get_recorder():
    """延迟导入录制器"""
    # Lazy import of recorder
    global _recorder_imported, _sync_recorder
    if not _recorder_imported:
        try:
            import sync_recorder as sr
            _sync_recorder = sr
            _recorder_imported = True
        except Exception as e:
            print(f"[AUDIO] Unable to import recorder: {e}")
            # 无法导入录制器
            _recorder_imported = True  # 标记已尝试，避免重复 / mark as attempted to avoid retry
    return _sync_recorder

# 兼容旧工程中的示例音频（保留）
# Kept for compatibility with sample audio from the original project
AUDIO_BASE_DIR = r"C:\Users\Administrator\Desktop\rebuild1002\music"

# 新增：voice 目录与映射表
# New: voice directory and mapping table
# 使用脚本所在目录的 voice 文件夹，避免工作目录问题
# Use the voice folder in the script's directory to avoid working directory issues
VOICE_DIR = os.getenv("VOICE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice"))
VOICE_MAP_FILE = os.path.join(VOICE_DIR, "map.zh-CN.json")

# 音频文件映射（将合并 voice 映射）
# Audio file mapping (will be merged with voice mapping)
AUDIO_MAP = {
    "检测到物体": os.path.join(AUDIO_BASE_DIR, "音频1.wav"),    # Object detected
    "向上": os.path.join(AUDIO_BASE_DIR, "音频2.wav"),          # Up
    "向下": os.path.join(AUDIO_BASE_DIR, "音频3.wav"),          # Down
    "向左": os.path.join(AUDIO_BASE_DIR, "音频4.wav"),          # Left
    "向右": os.path.join(AUDIO_BASE_DIR, "音频5.wav"),          # Right
    "OK": os.path.join(AUDIO_BASE_DIR, "音频6.wav"),            # OK
    "向前": os.path.join(AUDIO_BASE_DIR, "音频7.wav"),          # Forward
    "后退": os.path.join(AUDIO_BASE_DIR, "音频8.wav"),          # Back
    "拿到物体": os.path.join(AUDIO_BASE_DIR, "音频9.wav"),      # Item grabbed
}

# 音频缓存，避免重复读取
# Audio cache to avoid repeated file reads
_audio_cache = {}

# 音频播放队列和工作线程 - 使用优先级队列
# Audio playback queue and worker thread - uses priority queue
_audio_queue = queue.PriorityQueue(maxsize=10)
_audio_priority = 0  # 递增的优先级计数器 / incrementing priority counter
_worker_thread = None
_worker_loop = None
_is_playing = False  # 标记是否正在播放音频 / flag indicating whether audio is currently playing
_playing_lock = threading.Lock()  # 播放锁 / playback lock
_initialized = False
_last_play_ts = 0.0  # 记录上次播放结束时间，用于决定预热静音长度 / records last playback end time, used to determine lead silence length

def load_wav_file(filepath):
    """加载WAV文件并返回PCM数据（自动转换为8kHz）"""
    # Load WAV file and return PCM data (auto-converts to 8kHz)
    if filepath in _audio_cache:
        return _audio_cache[filepath]
    
    # 使用压缩缓存
    # Use compressed cache
    if os.getenv("AIGLASS_COMPRESS_AUDIO", "1") == "1":
        compressed_data = compressed_audio_cache.load_and_compress(filepath)
        if compressed_data:
            # 存储压缩后的数据 / Store compressed data
            _audio_cache[filepath] = compressed_data
            return compressed_data
    
    # 原始加载方式（不压缩）
    # Original load method (no compression)
    try:
        with wave.open(filepath, 'rb') as wav:
            # 检查音频格式 / Check audio format
            channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            framerate = wav.getframerate()
            
            if channels != 1:
                print(f"[AUDIO] Warning: {filepath} is not mono, will use first channel only")
                # 警告：不是单声道，将只使用第一个声道
            if sampwidth != 2:
                print(f"[AUDIO] Warning: {filepath} is not 16-bit audio")
                # 警告：不是16位音频
            
            # 读取所有帧 / Read all frames
            frames = wav.readframes(wav.getnframes())
            
            # 如果是立体声，只取左声道
            # If stereo, take left channel only
            if channels == 2:
                import audioop
                frames = audioop.tomono(frames, sampwidth, 1, 0)
            
            # 统一转换为8kHz（使用ratecv保证音调和速度不变）
            # Convert to 8kHz (use ratecv to preserve pitch and speed)
            if framerate != 8000:
                import audioop
                frames, _ = audioop.ratecv(frames, sampwidth, 1, framerate, 8000, None)
                print(f"[AUDIO] Resampled: {filepath} {framerate}Hz -> 8000Hz")
                # 重采样
            
            _audio_cache[filepath] = frames
            return frames
            
    except Exception as e:
        print(f"[AUDIO] Failed to load audio file {filepath}: {e}")
        # 加载音频文件失败
        return None

def _merge_voice_map():
    """读取 voice/map.zh-CN.json 并合并到 AUDIO_MAP"""
    # Read voice/map.zh-CN.json and merge into AUDIO_MAP
    try:
        if not os.path.exists(VOICE_MAP_FILE):
            print(f"[AUDIO] Mapping file not found: {VOICE_MAP_FILE}")
            # 未找到映射文件
            return
        with open(VOICE_MAP_FILE, "r", encoding="utf-8") as f:
            m = json.load(f)
        added = 0
        for text, info in (m or {}).items():
            files = (info or {}).get("files") or []
            if not files:
                continue
            fname = files[0]
            fpath = os.path.join(VOICE_DIR, fname)
            if os.path.exists(fpath):
                AUDIO_MAP[text] = fpath
                added += 1
            else:
                print(f"[AUDIO] Mapped file missing: {fpath}")
                # 映射文件缺失
        print(f"[AUDIO] Merged {added} voice map entries")
        # 已合并 voice 映射
    except Exception as e:
        print(f"[AUDIO] Failed to read voice mapping: {e}")
        # 读取 voice 映射失败

def preload_all_audio():
    """预加载所有音频文件到内存"""
    # Preload all audio files into memory
    print("[AUDIO] Starting audio file preload...")
    # 开始预加载音频文件
    loaded_count = 0
    
    # 【暂时禁用变速】因为需要修改缓存机制
    # [Temporarily disabled speed change] requires modifying cache mechanism
    # 需要加速的音频列表（斑马线相关）/ audio keys that need speed-up (crosswalk-related)
    # speedup_keywords = ["斑马线", "画面"]
    # speedup_factor = 1.3  # 加速30% / speed up 30%
    
    for audio_key, filepath in AUDIO_MAP.items():
        if os.path.exists(filepath):
            data = load_wav_file(filepath)
            if data:
                loaded_count += 1
        else:
            pass
    print(f"[AUDIO] Preload complete, loaded {loaded_count} audio files")
    # 预加载完成，共加载...个音频文件

def _audio_worker():
    """音频播放工作线程"""
    # Audio playback worker thread
    global _worker_loop
    
    # 尝试设置线程优先级（Windows特定）
    # Try to set thread priority (Windows-specific)
    try:
        import ctypes
        import sys
        if sys.platform == "win32":
            # 设置线程为高优先级 / Set thread to high priority
            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(),
                1  # THREAD_PRIORITY_ABOVE_NORMAL
            )
            print("[AUDIO] Audio thread set to high priority")
            # 设置音频线程为高优先级
    except Exception as e:
        print(f"[AUDIO] Failed to set thread priority: {e}")
        # 设置线程优先级失败
    
    _worker_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_worker_loop)
    
    async def process_queue():
        while True:
            try:
                # 从优先级队列获取数据 / Get data from priority queue
                priority_data = await asyncio.get_event_loop().run_in_executor(None, _audio_queue.get, True)
                if priority_data is None:
                    break
                # 解包优先级和实际音频数据 / Unpack priority and actual audio data
                if isinstance(priority_data, tuple) and len(priority_data) == 2:
                    _, audio_data = priority_data
                else:
                    audio_data = priority_data
                await _broadcast_audio_optimized(audio_data)
            except Exception as e:
                print(f"[AUDIO] Worker thread error: {e}")
                # 工作线程错误
    
    _worker_loop.run_until_complete(process_queue())

async def _broadcast_audio_optimized(pcm_data: bytes):
    """优化的音频广播：单次调用由底层按20ms节拍发送，移除重复节拍和Python层sleep"""
    # Optimized audio broadcast: single call handled by underlying 20ms pacing, removes redundant ticks and Python-level sleep
    global _last_play_ts, _is_playing
    try:
        # 设置播放标志 / Set playback flag
        with _playing_lock:
            _is_playing = True
        # 此时 pcm_data 应该已经是解压后的16位PCM数据了（8kHz）
        # At this point pcm_data should already be decompressed 16-bit PCM (8kHz)
        now = time.monotonic()
        idle_sec = now - (_last_play_ts or now)
        # 首次或长时间空闲后，预热更长静音；否则小静音
        # After first use or long idle, use longer lead silence; otherwise short silence
        lead_ms = 160 if idle_sec > 3.0 else 60
        tail_ms = 40

        lead_silence = b'\x00' * (lead_ms * 8000 * 2 // 1000)  # 8k * 2B
        tail_silence = b'\x00' * (tail_ms * 8000 * 2 // 1000)

        # 完整音频数据（包含静音）/ Full audio data (including silence)
        full_audio = lead_silence + pcm_data + tail_silence
        
        # 注意：录制在 broadcast_pcm16_realtime 中统一完成，避免重复
        # Note: recording is handled uniformly in broadcast_pcm16_realtime to avoid duplication

        # 单次调用交给底层 pacing（20ms节拍在 broadcast_pcm16_realtime 内部实现）
        # Single call handed to underlying pacing (20ms ticking implemented inside broadcast_pcm16_realtime)
        await broadcast_pcm16_realtime(full_audio)

        _last_play_ts = time.monotonic()
    except Exception as e:
        print(f"[AUDIO] Audio broadcast failed: {e}")
        # 广播音频失败
    finally:
        # 清除播放标志 / Clear playback flag
        with _playing_lock:
            _is_playing = False

def initialize_audio_system():
    """初始化音频系统"""
    # Initialize audio system
    global _initialized, _worker_thread, _last_play_ts
    
    if _initialized:
        return
    
    # 先合并 voice 映射，再预加载
    # Merge voice mapping first, then preload
    _merge_voice_map()
    preload_all_audio()
    
    _worker_thread = threading.Thread(target=_audio_worker, daemon=True)
    _worker_thread.start()
    _initialized = True
    _last_play_ts = 0.0
    
    # 显示压缩统计 / Show compression stats
    if os.getenv("AIGLASS_COMPRESS_AUDIO", "1") == "1":
        stats = compressed_audio_cache.get_compression_stats()
        print(f"[AUDIO] Audio compression stats:")
        # 音频压缩统计
        print(f"  - Files cached: {stats['files_cached']}")
        print(f"  - Original size: {stats['total_original_size'] / 1024:.1f} KB")
        print(f"  - Compressed: {stats['total_compressed_size'] / 1024:.1f} KB")
        print(f"  - Compression ratio: {stats['compression_ratio']:.1%}")
        print(f"  - Bytes saved: {stats['bytes_saved'] / 1024:.1f} KB")
    
    print("[AUDIO] Audio system initialized (preloaded + worker thread)")
    # 音频系统初始化完成（预加载+工作线程）

def play_audio_threadsafe(audio_key):
    """线程安全的音频播放函数"""
    # Thread-safe audio playback function
    global _audio_queue, _audio_priority
    
    if not _initialized:
        initialize_audio_system()
    
    if audio_key not in AUDIO_MAP:
        print(f"[AUDIO] Unknown audio key: {audio_key}")
        # 未知的音频键
        return
    
    filepath = AUDIO_MAP[audio_key]
    pcm_data = _audio_cache.get(filepath)
    if pcm_data is None:
        print(f"[AUDIO] Audio not in cache: {audio_key}")
        # 音频未在缓存中
        return
    
    # 如果是压缩的数据，先解压
    # If data is compressed, decompress first
    if pcm_data and len(pcm_data) > 5 and pcm_data[0] in [0x01, 0x02]:
        pcm_data = compressed_audio_cache.decompress(pcm_data)
        if not pcm_data:
            print(f"[AUDIO] Decompression failed: {audio_key}")
            # 解压失败
            return
    
    # 【优化】实时播报策略：保持队列最小化，避免积压延迟
    # [Optimization] Real-time broadcast strategy: keep queue minimal to avoid backlog delay
    queue_size = _audio_queue.qsize()
    
    # 检查是否正在播放 / Check if currently playing
    with _playing_lock:
        currently_playing = _is_playing
    
    # 实时策略：只允许1个积压，超过立即清空
    # Real-time strategy: allow only 1 backlog item; clear immediately if exceeded
    if queue_size > 0 and not currently_playing:
        # 未播放时立即清空，播放最新语音
        # Not playing: clear queue immediately, play latest voice
        print(f"[AUDIO] Clearing queue ({queue_size} items), playing latest")
        # 清空队列，播放最新语音
        _audio_queue = queue.PriorityQueue(maxsize=10)
    elif queue_size > 1 and currently_playing:
        # 正在播放时，如果积压>1个则清空（保持实时性）
        # While playing: if backlog > 1, clear to maintain real-time performance
        print(f"[AUDIO] Queue backlog ({queue_size} items), clearing for real-time")
        # 队列积压，清空以保持实时
        _audio_queue = queue.PriorityQueue(maxsize=10)
    try:
        # 使用优先级队列，确保音频按顺序播放
        # Use priority queue to ensure audio plays in order
        _audio_priority += 1
        _audio_queue.put_nowait((_audio_priority, pcm_data))
        if queue_size >= 1:
            print(f"[AUDIO] Playback queue size: {queue_size + 1}")
            # 播放队列当前大小
    except queue.Full:
        # 播放队列满则丢弃，保持实时性
        # Queue full, discard to maintain real-time performance
        print(f"[AUDIO] Queue full, discarding: {audio_key}")
        # 队列满，丢弃
        pass

# 全局语音节流
# Global voice throttle
_last_voice_time = 0
_last_voice_text = ""
_voice_cooldown = 1.0  # 相同语音至少间隔1秒 / same voice must wait at least 1 second

# 语音优先级定义
# Voice priority definitions
VOICE_PRIORITY = {
    'obstacle': 100,     # 障碍物 - 最高优先级 / Obstacle - highest priority
    'direction': 50,     # 转向/平移 - 中等优先级 / Turn/shift - medium priority
    'straight': 10,      # 保持直行 - 最低优先级 / Keep straight - lowest priority
    'other': 30          # 其他 - 默认优先级 / Other - default priority
}

# 新增：根据中文提示文案直接播放（会做轻度规范化与降级）
# New: play directly based on text prompt (with light normalization and fallback)
def play_voice_text(text: str):
    """
    传入中文提示，自动匹配 voice 映射并播放。
    Pass in text prompt, auto-match voice mapping and play.
    - 尝试原文 / Try original text
    - 尝试补全/去除句末标点（。.!！?？）/ Try adding/removing sentence-end punctuation
    - 若包含"前方有…注意避让"但未命中，降级到"前方有障碍物，注意避让。"
    - If contains obstacle warning but no match, fall back to default obstacle warning
    """
    global _last_voice_time, _last_voice_text
    
    if not text:
        return
    if not _initialized:
        initialize_audio_system()
    
    # 全局节流：相同文本短时间内不重复播放
    # Global throttle: do not repeat same text within cooldown period
    current_time = time.time()
    if text == _last_voice_text and current_time - _last_voice_time < _voice_cooldown:
        return  # 静默跳过 / silently skip

    candidates = []
    t = text.strip()
    candidates.append(t)
    # 尝试补全句号 / Try appending period
    if t[-1:] not in ("。", "！", "!", "？", "?", "."):
        candidates.append(t + "。")
    else:
        # 尝试去掉标点 / Try removing punctuation
        t2 = t.rstrip("。.!！?？")
        if t2 and t2 != t:
            candidates.append(t2)

    # 逐一尝试匹配 / Try each candidate
    for ck in candidates:
        if ck in AUDIO_MAP:
            play_audio_threadsafe(ck)
            _last_voice_text = text
            _last_voice_time = current_time
            return

    # 针对"前方有…注意避让"降级
    # Fallback for obstacle warning pattern
    if ("前方有" in t) and ("注意避让" in t):
        fallback = "前方有障碍物，注意避让。"
        if fallback in AUDIO_MAP:
            play_audio_threadsafe(fallback)
            _last_voice_text = text
            _last_voice_time = current_time
            return

    # 针对"请向…平移/微调/转动"类词条，常见变体尝试
    # Try common variants for direction/adjustment prompts
    base = t.rstrip("。.!！?？")
    if base in AUDIO_MAP:
        play_audio_threadsafe(base)
        _last_voice_text = text
        _last_voice_time = current_time
        return
    if base + "。" in AUDIO_MAP:
        play_audio_threadsafe(base + "。")
        _last_voice_text = text
        _last_voice_time = current_time
        return

    # 未匹配则输出日志（便于调试）
    # No match found, log for debugging
    print(f"[AUDIO] No matching voice found: {text}")
    # 未找到匹配语音

# 兼容旧接口 / Backward compatibility alias
play_audio_on_esp32 = play_audio_threadsafe