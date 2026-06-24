# omni_client.py
# -*- coding: utf-8 -*-
#
# Changes vs original:
#   - DashScope / OpenAI-compatible remote client commented out (kept as fallback)
#   - stream_chat() now runs Qwen2.5-Omni-3B locally on MPS, TEXT ONLY
#   - Audio output (return_audio) is disabled here; TTS handled via macOS 'say'
#     in app_main.py._say_to_pcm8k → broadcast_pcm16_realtime
#   - OmniStreamPiece interface unchanged so app_main.py needs no adjustments
#   - device_map="mps" (NOT "auto") avoids meta-device disk offload on Apple Silicon
#
import os, base64, asyncio, io
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, List, Optional

# ===== [DASHSCOPE FALLBACK] Remote Qwen-Omni via OpenAI-compatible API =====
# Uncomment this block and comment out the local section below to switch back.
# from openai import OpenAI
# _API_KEY  = os.getenv("DASHSCOPE_API_KEY", "YOUR_DASHSCOPE_API_KEY")
# _QWEN_MODEL = "qwen-omni-turbo"
# _oai_client = OpenAI(
#     api_key=_API_KEY,
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
# ===========================================================================

import torch
from PIL import Image
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

_MODEL_ID = os.getenv("QWEN_OMNI_MODEL", "Qwen/Qwen2.5-Omni-3B")


def _resolve_model_path(model_id: str) -> str:
    """Prefer an already-downloaded Hugging Face snapshot to avoid startup HEAD checks."""
    if os.path.exists(model_id):
        return model_id

    repo_cache_name = f"models--{model_id.replace('/', '--')}"
    candidates = []
    for cache_root in (
        os.getenv("HF_HOME"),
        os.path.join(os.getenv("XDG_CACHE_HOME", ""), "huggingface"),
        os.path.join(os.getcwd(), ".cache", "huggingface"),
    ):
        if cache_root:
            candidates.append(Path(cache_root) / "hub" / repo_cache_name)

    for model_cache in candidates:
        ref_file = model_cache / "refs" / "main"
        if not ref_file.exists():
            continue
        revision = ref_file.read_text(encoding="utf-8").strip()
        snapshot = model_cache / "snapshots" / revision
        if (snapshot / "config.json").exists():
            return str(snapshot)

    return model_id

_MODEL_PATH = _resolve_model_path(_MODEL_ID)
_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
_DTYPE = torch.float16 if _DEVICE == "mps" else torch.float32

print(f"[OMNI] Loading local model {_MODEL_PATH!r} → {_DEVICE} {_DTYPE} …")
_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    _MODEL_PATH,
    torch_dtype=_DTYPE,
    device_map=_DEVICE,
)
_processor = Qwen2_5OmniProcessor.from_pretrained(_MODEL_PATH)
print("[OMNI] Local model ready.")


class OmniStreamPiece:
    """Unified incremental payload: text_delta and/or audio_b64."""
    def __init__(self, text_delta: Optional[str] = None, audio_b64: Optional[str] = None):
        self.text_delta = text_delta
        self.audio_b64  = audio_b64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_data_url(data_url: str) -> Optional[Image.Image]:
    """Parse a base64 data: URL into a PIL RGB Image."""
    try:
        if data_url.startswith("data:"):
            _, b64 = data_url.split(",", 1)
        else:
            b64 = data_url
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception:
        return None


def _build_qwen_messages(
    content_list: List[Dict[str, Any]],
) -> tuple:
    """
    Convert OpenAI-style content_list to (qwen_messages, pil_images).
    Image entries become {"type": "image"} placeholders; PIL objects go
    in the separate list that the processor maps to those placeholders.
    """
    qwen_content: List[Dict[str, Any]] = []
    pil_images: List[Image.Image] = []

    for item in content_list:
        t = item.get("type")
        if t == "image_url":
            url = (item.get("image_url") or {}).get("url", "")
            img = _decode_data_url(url)
            if img is not None:
                pil_images.append(img)
                qwen_content.append({"type": "image"})   # placeholder; processor fills in
        elif t == "text":
            qwen_content.append({"type": "text", "text": item.get("text", "")})

    messages = [{"role": "user", "content": qwen_content}]
    return messages, pil_images


# ---------------------------------------------------------------------------
# Main streaming interface (unchanged signature for app_main.py)
# ---------------------------------------------------------------------------

async def stream_chat(
    content_list: List[Dict[str, Any]],
    voice: str = "Cherry",       # retained for interface compat — unused here (TTS via 'say')
    audio_format: str = "wav",   # same
) -> AsyncGenerator[OmniStreamPiece, None]:
    """
    Local Qwen2.5-Omni-3B text-only inference.

    generation_mode="text" → talker/token2wav are skipped entirely.
    Audio is produced separately by _say_to_pcm8k in app_main.py.

    Yields exactly one OmniStreamPiece(text_delta=<response>, audio_b64=None).
    The rest of the pipeline (ui_broadcast, broadcast_pcm16_realtime) is unchanged.

    # ---- DASHSCOPE FALLBACK (stream from remote API) ----
    # completion = _oai_client.chat.completions.create(
    #     model=_QWEN_MODEL,
    #     messages=[{"role": "user", "content": content_list}],
    #     modalities=["text", "audio"],
    #     audio={"voice": voice, "format": audio_format},
    #     stream=True,
    #     stream_options={"include_usage": True},
    # )
    # for chunk in completion:
    #     text_delta = audio_b64 = None
    #     if getattr(chunk, "choices", None):
    #         c0 = chunk.choices[0]
    #         delta = getattr(c0, "delta", None)
    #         if delta and getattr(delta, "content", None):
    #             text_delta = delta.content
    #         if delta and getattr(delta, "audio", None):
    #             aud = delta.audio
    #             audio_b64 = aud.get("data") if isinstance(aud, dict) else getattr(aud, "data", None)
    #     if (text_delta is not None) or (audio_b64 is not None):
    #         yield OmniStreamPiece(text_delta=text_delta, audio_b64=audio_b64)
    # ---- END DASHSCOPE FALLBACK ----
    """
    messages, pil_images = _build_qwen_messages(content_list)

    text_prompt = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    proc_inputs = _processor(
        text=text_prompt,
        images=pil_images if pil_images else None,
        padding=True,
        return_tensors="pt",
    ).to(_DEVICE)

    prompt_len = proc_inputs.input_ids.shape[1]
    loop = asyncio.get_event_loop()

    def _generate() -> str:
        with torch.no_grad():
            output_ids = _model.generate(
                **proc_inputs,
                thinker_max_new_tokens=256,
                return_audio=False,        # skips talker + token2wav entirely
            )
        new_ids = output_ids[0][prompt_len:]
        return _processor.decode(new_ids, skip_special_tokens=True).strip()

    response_text = await loop.run_in_executor(None, _generate)

    if response_text:
        yield OmniStreamPiece(text_delta=response_text, audio_b64=None)
