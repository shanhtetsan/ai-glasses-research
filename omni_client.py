# omni_client.py
# -*- coding: utf-8 -*-
import os
import base64
import asyncio
import soundfile as sf
import io
import numpy as np
import torch
from typing import AsyncGenerator, Dict, Any, List, Optional
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

# ===== Original: Alibaba DashScope compatible mode =====
# API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-a9440db694924559ae4ebdc2023d2b9a")
# if not API_KEY:
#     raise RuntimeError("未设置 DASHSCOPE_API_KEY")
# QWEN_MODEL = "qwen-omni-turbo"
# from openai import OpenAI
# oai_client = OpenAI(
#     api_key=API_KEY,
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )

# ===== Fallback: OpenAI API (comment in if local model quality is insufficient) =====
# API_KEY = os.getenv("OPENAI_API_KEY")
# if not API_KEY:
#     raise RuntimeError("OPENAI_API_KEY is not set")
# MODEL = "gpt-4o-audio-preview"
# from openai import OpenAI
# oai_client = OpenAI(api_key=API_KEY)

# ===== Current: Local Qwen2.5-Omni-7B (free, no API key needed) =====
# Why: No DashScope access from US, OpenAI costs money.
# Qwen2.5-Omni-7B runs locally on MacBook M5 24GB via MPS.
# Same architecture as original — text + audio output in one model call.

MODEL_NAME = os.getenv("QWEN_OMNI_MODEL", "Qwen/Qwen2.5-Omni-7B")
SPEAKER = os.getenv("QWEN_OMNI_SPEAKER", "Chelsie")  # options: Chelsie, Ethan
SAMPLE_RATE = 24000

print(f"[OMNI] Loading local model: {MODEL_NAME}")
_processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_NAME)
_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
print(f"[OMNI] Model loaded successfully")


class OmniStreamPiece:
    """Unified incremental output: text delta and/or audio chunk (base64 encoded WAV)."""
    # 对外的统一增量数据：text/audio 二选一或同时。
    def __init__(self, text_delta: Optional[str] = None, audio_b64: Optional[str] = None):
        self.text_delta = text_delta
        self.audio_b64  = audio_b64


async def stream_chat(
    content_list: List[Dict[str, Any]],
    voice: str = SPEAKER,
    audio_format: str = "wav",
) -> AsyncGenerator[OmniStreamPiece, None]:
    """
    Run one round of multimodal inference using local Qwen2.5-Omni.
    发起一轮本地 Qwen2.5-Omni 多模态推理：
    - content_list: list of text/image_url dicts (OpenAI chat format)
    - Yields OmniStreamPiece with text_delta and/or audio_b64

    Note: Qwen2.5-Omni does not support true streaming locally.
    We generate the full response then yield it as a single piece.
    This keeps the interface identical to the original streaming design.
    """

    # Build conversation in Qwen2.5-Omni format
    # 构建 Qwen2.5-Omni 格式的对话
    conversation = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful AI assistant for visually impaired users. Give short, clear responses."
            }]
        }
    ]

    # Convert content_list to Qwen format
    # 将 content_list 转换为 Qwen 格式
    user_content = []
    for item in content_list:
        if item.get("type") == "text":
            user_content.append({"type": "text", "text": item["text"]})
        elif item.get("type") == "image_url":
            # Extract base64 image / 提取 base64 图像
            url = item.get("image_url", {}).get("url", "")
            if url.startswith("data:image"):
                user_content.append({"type": "image", "image": url})

    conversation.append({"role": "user", "content": user_content})

    # Run inference in executor to avoid blocking the event loop
    # 在 executor 中运行推理，避免阻塞事件循环
    loop = asyncio.get_event_loop()

    def _infer():
        text = _processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        inputs = _processor(text=text, return_tensors="pt")

        # Move inputs to same device as model
        # 将输入移到与模型相同的设备
        device = next(_model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        text_ids, audio = _model.generate(
            **inputs,
            speaker=voice,
            return_audio=True,
            max_new_tokens=200,
        )

        # Decode text / 解码文本
        full_text = _processor.batch_decode(text_ids, skip_special_tokens=True)
        response_text = full_text[0] if full_text else ""

        # Extract just the assistant response
        # 仅提取 assistant 的回复
        if "assistant\n" in response_text:
            response_text = response_text.split("assistant\n")[-1].strip()

        # Convert audio to base64 WAV
        # 将音频转换为 base64 WAV
        audio_b64 = None
        if audio is not None:
            audio_np = audio.reshape(-1).detach().cpu().numpy()
            buf = io.BytesIO()
            sf.write(buf, audio_np, samplerate=SAMPLE_RATE, format="WAV", subtype="PCM_16")
            buf.seek(0)
            audio_b64 = base64.b64encode(buf.read()).decode("ascii")

        return response_text, audio_b64

    response_text, audio_b64 = await loop.run_in_executor(None, _infer)

    # Yield as single piece (text + audio together)
    # 作为单个片段产出（文本 + 音频一起）
    yield OmniStreamPiece(text_delta=response_text, audio_b64=audio_b64)