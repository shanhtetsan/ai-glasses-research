# omni_client.py
# -*- coding: utf-8 -*-
#
# Backends:
#   MODEL_BACKEND=qwen   (default)  — local Qwen2.5-Omni-3B on MPS
#   MODEL_BACKEND=gemini             — Google Gemini via API (GEMINI_API_KEY required)
#
# Either path yields OmniStreamPiece(text_delta=..., audio_b64=None) so
# app_main.py is identical regardless of backend. TTS still goes through
# macOS 'say' in _say_to_pcm16k.
#
import os, base64, asyncio, io
from typing import AsyncGenerator, Dict, Any, List, Optional


_BACKEND = (os.getenv("MODEL_BACKEND") or "qwen").strip().lower()


class OmniStreamPiece:
    """Unified incremental payload: text_delta and/or audio_b64."""
    def __init__(self, text_delta: Optional[str] = None, audio_b64: Optional[str] = None):
        self.text_delta = text_delta
        self.audio_b64  = audio_b64


# ===========================================================================
# Qwen backend (local, MPS)
# ===========================================================================
if _BACKEND == "qwen":
    import torch
    from PIL import Image
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    _MODEL_ID = os.getenv("QWEN_OMNI_MODEL", "Qwen/Qwen2.5-Omni-3B")
    _MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "96"))

    print(f"[OMNI] Loading local model {_MODEL_ID!r} → MPS float16 …")
    _model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        _MODEL_ID,
        torch_dtype=torch.float16,
        device_map="mps",
    )
    _processor = Qwen2_5OmniProcessor.from_pretrained(_MODEL_ID)
    print("[OMNI] Local model ready.")

    def _decode_data_url(data_url: str) -> Optional["Image.Image"]:
        try:
            if data_url.startswith("data:"):
                _, b64 = data_url.split(",", 1)
            else:
                b64 = data_url
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception:
            return None

    def _build_qwen_messages(content_list: List[Dict[str, Any]]) -> tuple:
        qwen_content: List[Dict[str, Any]] = []
        pil_images: List["Image.Image"] = []
        for item in content_list:
            t = item.get("type")
            if t == "image_url":
                url = (item.get("image_url") or {}).get("url", "")
                img = _decode_data_url(url)
                if img is not None:
                    pil_images.append(img)
                    qwen_content.append({"type": "image"})
            elif t == "text":
                qwen_content.append({"type": "text", "text": item.get("text", "")})
        return [{"role": "user", "content": qwen_content}], pil_images

    async def _stream_qwen(
        content_list: List[Dict[str, Any]],
    ) -> AsyncGenerator[OmniStreamPiece, None]:
        messages, pil_images = _build_qwen_messages(content_list)
        text_prompt = _processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        proc_inputs = _processor(
            text=text_prompt,
            images=pil_images if pil_images else None,
            padding=True,
            return_tensors="pt",
        ).to("mps")
        prompt_len = proc_inputs.input_ids.shape[1]
        loop = asyncio.get_event_loop()

        def _generate() -> str:
            print(f"[OMNI] generating text, max_new_tokens={_MAX_NEW_TOKENS}", flush=True)
            with torch.no_grad():
                output_ids = _model.generate(
                    **proc_inputs,
                    thinker_max_new_tokens=_MAX_NEW_TOKENS,
                    return_audio=False,
                )
            if isinstance(output_ids, tuple):
                output_ids = output_ids[0]
            new_ids = output_ids[0][prompt_len:]
            text = _processor.decode(new_ids, skip_special_tokens=True).strip()
            print(f"[OMNI] generated {len(new_ids)} tokens, text_len={len(text)}", flush=True)
            return text

        response_text = await loop.run_in_executor(None, _generate)
        if response_text:
            yield OmniStreamPiece(text_delta=response_text, audio_b64=None)


# ===========================================================================
# Gemini backend (Google API, streaming)
# ===========================================================================
elif _BACKEND == "gemini":
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise RuntimeError(
            "MODEL_BACKEND=gemini requires google-generativeai. "
            "Install with: pip install 'google-generativeai>=0.8.0'"
        ) from e

    _GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not _GEMINI_API_KEY:
        raise RuntimeError("MODEL_BACKEND=gemini but GEMINI_API_KEY is not set.")

    _GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash-lite")

    print(f"[OMNI] Using Gemini backend, model={_GEMINI_MODEL_ID!r}")
    genai.configure(api_key=_GEMINI_API_KEY)
    _gemini_model = genai.GenerativeModel(_GEMINI_MODEL_ID)

    def _build_gemini_parts(content_list: List[Dict[str, Any]]) -> List[Any]:
        """OpenAI-style content_list → Gemini parts list."""
        parts: List[Any] = []
        for item in content_list:
            t = item.get("type")
            if t == "image_url":
                url = (item.get("image_url") or {}).get("url", "")
                try:
                    _, b64 = url.split(",", 1) if url.startswith("data:") else ("", url)
                    parts.append({"mime_type": "image/jpeg", "data": base64.b64decode(b64)})
                except Exception:
                    continue
            elif t == "text":
                parts.append(item.get("text", ""))
        return parts

    async def _stream_gemini(
        content_list: List[Dict[str, Any]],
    ) -> AsyncGenerator[OmniStreamPiece, None]:
        parts = _build_gemini_parts(content_list)
        loop = asyncio.get_event_loop()

        def _start_stream():
            # generate_content(stream=True) returns a GenerateContentResponse that
            # is iterable but not itself an iterator — wrap with iter() so next() works.
            return iter(_gemini_model.generate_content(parts, stream=True))

        stream_iter = await loop.run_in_executor(None, _start_stream)
        print(f"[OMNI] Gemini stream started", flush=True)

        def _next_chunk(it):
            return next(it, None)

        while True:
            chunk = await loop.run_in_executor(None, _next_chunk, stream_iter)
            if chunk is None:
                break
            delta = getattr(chunk, "text", None)
            if delta:
                yield OmniStreamPiece(text_delta=delta, audio_b64=None)


else:
    raise RuntimeError(
        f"Unknown MODEL_BACKEND={_BACKEND!r}. Expected 'qwen' or 'gemini'."
    )


# ===========================================================================
# Unified entry point — same signature regardless of backend
# ===========================================================================
async def stream_chat(
    content_list: List[Dict[str, Any]],
    voice: str = "Cherry",       # retained for interface compat; TTS uses macOS 'say'
    audio_format: str = "wav",
) -> AsyncGenerator[OmniStreamPiece, None]:
    if _BACKEND == "gemini":
        async for piece in _stream_gemini(content_list):
            yield piece
    else:
        async for piece in _stream_qwen(content_list):
            yield piece
