# Architecture Notes: Audio Pipeline

This document traces how audio moves through the system, why each piece is designed the way it is, and what patterns are worth reusing.

---

## 1. Audio Flow: End to End

```
ESP32 microphone
  │  raw 16kHz PCM chunks (20ms, 640 bytes each)
  ▼
WebSocket /ws_audio  (app_main.py:818)
  │  recognition.send_audio_frame(bytes)
  ▼
DashScope ASR SDK  — model: paraformer-realtime-v2
  │  fires ASRCallback.on_result() / on_event()  (asr_core.py:137)
  ▼
ASRCallback._handle()  (asr_core.py:148)
  │  partial → UI only
  │  is_end=True AND not is_playing_now() →
  ▼
start_ai_with_text_custom()  (app_main.py:453)
  │  command routing (navigation / item-search / cross-street)
  │  falls through to:
  ▼
start_ai_with_text()  (app_main.py:687)
  │  1. hard_reset_audio()          — kill any previous output
  │  2. attach last camera frame as image_url
  │  3. loop.create_task(_runner())
  ▼
omni_client.stream_chat()  (omni_client.py:29)
  │  POST to DashScope qwen-omni-turbo (OpenAI-compat)
  │  stream=True, modalities=["text","audio"]
  │  yields OmniStreamPiece(text_delta, audio_b64)
  ▼
_runner() in start_ai_with_text  (app_main.py:689)
  │  audio_b64  →  base64.b64decode  →  24kHz PCM16
  │  audioop.ratecv(pcm24, 2, 1, 24000, 8000, rate_state)  →  8kHz PCM16
  │  audioop.mul(pcm8k, 2, 0.60)  →  volume at 60%
  ▼
audio_stream.broadcast_pcm16_realtime()  (audio_stream.py:85)
  │  slices into 20ms frames, paces with asyncio.sleep
  │  puts each frame into every StreamClient.q
  ▼
FastAPI GET /stream.wav  →  gen()  (audio_stream.py:143)
  │  HTTP chunked WAV stream (infinite Content-Length)
  ▼
ESP32 speaker
```

Pre-recorded navigation prompts take a separate shorter path:

```
play_voice_text(text)  (audio_player.py:373)
  │  normalise text → lookup AUDIO_MAP → _audio_cache[path]
  │  decompress if ADPCM/ulaw header detected
  │  _audio_queue.put_nowait((priority, pcm_data))
  ▼
_audio_worker thread  (audio_player.py:184)
  │  asyncio event loop on private thread
  │  dequeues → _broadcast_audio_optimized()
  ▼
broadcast_pcm16_realtime()  →  /stream.wav  →  ESP32 speaker
```

---

## 2. audio_stream.py

### `BYTES_PER_20MS_16K`

```python
BYTES_PER_20MS_16K = STREAM_SR * STREAM_SW * 20 // 1000   # = 8000 * 2 * 20 / 1000 = 320 bytes
```

Despite the variable name saying "16K", `STREAM_SR` was changed to 8000 Hz for ESP32 compatibility, so the constant evaluates to **320 bytes** — one 20 ms frame at 8kHz 16-bit mono.

### Why 20ms pacing?

`broadcast_pcm16_realtime` uses a wall-clock tick (`next_tick += 0.020`) to pace output:

```python
next_tick = loop.time()
while off < len(pcm16):
    piece = pcm16[off : off + BYTES_PER_20MS_16K]
    # … push to all queues …
    next_tick += 0.020
    now = loop.time()
    if now < next_tick:
        await asyncio.sleep(next_tick - now)
    off += take
```

This matches the real-time **consumption rate** of the speaker. Without pacing, the entire audio blob would be pushed into queues at wire speed, instantly filling the 96-item buffer and triggering drops. Pacing ensures the producer never gets more than one frame ahead of real time.

The "catch-up" branch (`next_tick = now`) handles the case where the event loop stalls — it resets the clock rather than accumulating debt, which would cause doubled-speed playback after a lag.

### The streaming queue

Each connected HTTP client is represented by a `StreamClient` dataclass:

```python
@dataclass(frozen=True)
class StreamClient:
    q: asyncio.Queue       # maxsize=96
    abort_event: asyncio.Event
```

`broadcast_pcm16_realtime` iterates `stream_clients` and does a **drop-oldest** strategy on full queues:

```python
if sc.q.full():
    try: sc.q.get_nowait()   # evict the oldest frame
    except: pass
sc.q.put_nowait(piece)
```

This keeps the queue at the boundary rather than blocking the producer or silently skipping new audio. The `gen()` coroutine on the HTTP side does a `wait_for(q.get(), timeout=0.5)` loop, breaking when `abort_event` is set or a `None` sentinel arrives.

---

## 3. audio_player.py

### Prerecorded file playback

1. At startup, `initialize_audio_system()` calls `_merge_voice_map()` (reads `voice/map.zh-CN.json`) then `preload_all_audio()`.
2. `load_wav_file()` opens each WAV, converts stereo→mono (`audioop.tomono`), resamples to 8kHz (`audioop.ratecv`), and passes the PCM through `CompressedAudioCache.load_and_compress()`.
3. The result is stored in `_audio_cache[filepath]` — ADPCM-compressed bytes with a 5-byte header.
4. At play time, `play_audio_threadsafe()` reads from `_audio_cache`, strips the compression header, decompresses, and enqueues the raw PCM.

### Priority queue

`_audio_queue = queue.PriorityQueue(maxsize=10)` with a monotonically incrementing `_audio_priority` counter. Each enqueue is `(priority_counter, pcm_bytes)`. The worker dequeues and unpacks the tuple. In practice, since the counter always increases, the priority queue acts like a FIFO — but it gives you the option to insert urgent audio with a lower priority number if needed.

### Drop-stale strategy

`play_audio_threadsafe()` enforces a real-time policy before enqueuing:

```python
if queue_size > 0 and not currently_playing:
    _audio_queue = queue.PriorityQueue(maxsize=10)  # flush everything
elif queue_size > 1 and currently_playing:
    _audio_queue = queue.PriorityQueue(maxsize=10)  # flush backlog
```

The rule: never let more than one item wait. If guidance piles up faster than it can be spoken, throw away the backlog and speak the latest instruction. For navigation ("turn left", "obstacle ahead"), the newest command is always the most relevant.

The worker thread runs its own `asyncio.new_event_loop()` so it can await `broadcast_pcm16_realtime` without touching the FastAPI event loop, avoiding thread-safety issues on the shared `stream_clients` set.

---

## 4. audio_compressor.py

Two algorithms are implemented:

| Algorithm | Header byte | Bit depth | Compression ratio |
|-----------|-------------|-----------|-------------------|
| μ-law     | `0x01`      | 16→8 bit  | ~50%              |
| IMA-ADPCM | `0x02`      | 16→4 bit  | ~75%              |

The default is **ADPCM** (`AIGLASS_COMPRESS_TYPE=adpcm`). The compressed blob always carries a 5-byte header: `struct.pack('!BI', type_tag, original_length)` so `decompress()` can identify and undo it without side-channel information.

**The problem being solved:** preloaded navigation audio (dozens of short phrases) sits in memory between uses. At 8kHz 16-bit, even a 2-second clip is 32 KB. ADPCM shrinks that to ~8 KB, cutting total RAM footprint by 75%. This is the *server* memory problem, not the ESP32 bandwidth problem — the decompression happens on the server just before `broadcast_pcm16_realtime`, so the stream is always plain PCM16.

μ-law uses logarithmic companding (ITU-T G.711), good for telephony SNR. IMA-ADPCM stores inter-sample *differences* using adaptive step sizes (the 89-entry `step_table`), giving better quality at higher compression for typical speech signals.

---

## 5. omni_client.py

### Streaming structure

```python
completion = oai_client.chat.completions.create(
    model="qwen-omni-turbo",
    messages=[{"role": "user", "content": content_list}],
    modalities=["text", "audio"],
    audio={"voice": "Cherry", "format": "wav"},
    stream=True,
    stream_options={"include_usage": True},
)
for chunk in completion:   # synchronous iterator
    ...
    yield OmniStreamPiece(text_delta=..., audio_b64=...)
```

The function is an `async def` generator but the inner loop is a **synchronous** `for chunk in completion` — the OpenAI Python SDK's streaming object is a sync iterator. Each `yield` hands a piece back to the async caller. This works but blocks the event loop on each SDK `.next()` call (see §8).

### `OmniStreamPiece`

A minimal data carrier:

```python
class OmniStreamPiece:
    def __init__(self, text_delta: Optional[str], audio_b64: Optional[str]):
        self.text_delta = text_delta
        self.audio_b64  = audio_b64
```

Either field can be `None` on a given chunk. The model interleaves text and audio deltas in the stream — a single chunk may have both, only text, or only audio. The consumer in `_runner()` handles both fields independently: text goes to the UI, audio goes through resample→broadcast.

---

## 6. asr_core.py

### Dependency injection in `ASRCallback`

`ASRCallback` has **no imports of app-level modules**. Every external behavior is passed in through `__init__`:

```python
class ASRCallback:
    def __init__(
        self,
        on_sdk_error:           Callable[[str], None],
        post:                   Callable[[asyncio.Future], None],  # thread→loop bridge
        ui_broadcast_partial,
        ui_broadcast_final,
        is_playing_now_fn:      Callable[[], bool],
        start_ai_with_text_fn,  # async (text)
        full_system_reset_fn,   # async (reason)
        interrupt_lock:         asyncio.Lock,
    ): ...
```

The `post` callable is the key bridge: `ASRCallback` runs on a DashScope SDK background thread, but all side effects must execute on the asyncio event loop. In `ws_audio`:

```python
def post(coro):
    asyncio.run_coroutine_threadsafe(coro, loop)
```

`ASRCallback._handle()` calls `self._post(some_coroutine())` instead of `await`-ing directly — it submits the coroutine to the correct loop from the wrong thread.

### Partial vs final transcripts

The DashScope ASR SDK fires `on_result` continuously. `_extract_sentence()` reads `sentence.sentence_end` from the event payload:

- **Partial** (`is_end is None` or `False`): stored in `_last_partial_for_ui`, broadcast to UI as `PARTIAL:...`. Nothing else.
- **Final** (`is_end is True`): broadcast as `FINAL:...` and, *only if `is_playing_now()` returns False*, submitted to `start_ai_with_text_fn`. After a final, all partial state resets.

This prevents the classic real-time ASR bug of triggering multiple LLM calls mid-sentence as partial results stream in.

---

## 7. The Hard Reset

There are two levels:

### `hard_reset_audio(reason)` — audio_stream.py:62

Scope: audio output only.

1. Sets `abort_event` on every `StreamClient` → `gen()` exits its while-loop and the HTTP connection drains naturally.
2. Clears `stream_clients`.
3. Calls `cancel_current_ai()` → cancels the asyncio task running `_runner()`, which stops new audio being produced.

### `full_system_reset(reason)` — app_main.py:354

Scope: the entire pipeline back to idle state.

1. `hard_reset_audio()` — kill output.
2. `stop_current_recognition()` — call `recognition.stop()` on the DashScope SDK object and clear `_current_recognition`.
3. Clear `current_partial` and `recent_finals` — reset UI state.
4. `last_frames.clear()` — discard camera frames so the next LLM call doesn't receive a stale image.
5. Send `"RESET"` text over the ESP32 WebSocket — the device can clear its own playback buffer.

### Why does a real-time system need this?

Without a hard reset, the system has no safe way to respond to user interruptions ("stop", hotword). Consider what happens without it:

- The asyncio task is still producing audio chunks and pushing them to queues.
- The old HTTP connection is still draining the queue to the speaker.
- The ASR stream, if left open, may deliver late partial results and re-trigger the LLM.
- The camera frame buffer still holds the image that prompted the old response.

These are all *concurrent*, asynchronous state sources. There is no "pause" for any of them individually. The only correct response is to atomically tear everything down and return to the known-good initial state. `full_system_reset` is that operation. It is triggered by hotwords in `ASRCallback._handle()` and also called proactively at the start of every new LLM response (`start_ai_with_text` calls `hard_reset_audio` before launching the task).

---

## 8. Patterns Worth Copying (and Not)

### Worth copying

**1. Dependency injection into SDK callbacks**
`ASRCallback.__init__` receives every external dependency as a typed callable. The callback itself never imports from `app_main`. This makes it testable in isolation and reusable with a different transport. Apply this any time you adapt a vendor SDK callback to your app logic.

**2. 20ms real-time pacing with wall-clock drift correction**
The `next_tick += 0.020; if now < next_tick: await sleep(next_tick - now)` pattern in `broadcast_pcm16_realtime` is a clean scheduler. It keeps the producer at real-time rate and self-corrects after event-loop stalls. Use this wherever you need to stream data at a fixed real-time rate (audio, video, sensor telemetry).

**3. Partial/final transcript discipline**
Only commit to side effects (LLM calls) on final ASR results. Use partial results only for user-facing display. This single rule eliminates an entire class of duplicate-trigger bugs in voice interfaces.

**4. Hard reset as a named, atomic operation**
Give the "kill everything and return to zero" operation a name (`full_system_reset`) and a single code path. Every interrupt path calls it. This is much safer than having several places each trying to cancel only the subset of state they know about.

**5. Drop-oldest queue policy for real-time media**
When a streaming queue is full, evict the oldest frame rather than blocking the producer or discarding new data. For audio/video where recency matters more than completeness, this keeps latency bounded at the cost of a small gap — which is the right trade.

---

### Not worth copying

**Synchronous SDK iterator inside an async generator** (`omni_client.py:51`)
`for chunk in completion:` blocks the asyncio event loop on every SDK `.next()` call. During a multi-second LLM response, this means no other coroutines can run between chunks. Fix: wrap in `asyncio.to_thread` or switch to an async-native SDK.

**Module `__dict__` mutation for global task tracking** (`app_main.py:783`)
```python
_as_dict["current_ai_task"] = task
```
This sets a variable in another module by reaching into its `__dict__`. It works but is opaque to static analysis tools and type checkers. A dedicated `set_current_ai_task(task)` function in `audio_stream.py` would be three lines and much clearer.

**Hardcoded Windows absolute path as default** (`audio_player.py:35`)
`AUDIO_BASE_DIR = r"C:\Users\Administrator\Desktop\rebuild1002\music"` is a legacy artifact. Any deployment on a different machine silently has no audio for the legacy map keys. Move all paths to environment variables or a config file.

**Decompress on every play, not on first access**
`play_audio_threadsafe` detects the compression header and decompresses the full clip before every enqueue. If the same phrase is played dozens of times per session (e.g., "obstacle ahead"), this wastes CPU on repeated decompression of the same bytes. Cache the decompressed PCM on first play: `_pcm_cache[filepath] = decompressed`.
