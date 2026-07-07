import os
import io
import base64
import asyncio
from typing import Optional, AsyncGenerator, List, Dict, Any

from PIL import Image
from google import genai
from google.genai import types

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


class OmniStreamPiece:
    def __init__(
        self,
        text_delta: Optional[str] = None,
        audio_b64: Optional[str] = None,
    ):
        self.text_delta = text_delta
        self.audio_b64 = audio_b64


def _decode_data_url(data_url: str):
    try:
        if data_url.startswith("data:"):
            _, b64 = data_url.split(",", 1)
        else:
            b64 = data_url

        return Image.open(
            io.BytesIO(base64.b64decode(b64))
        ).convert("RGB")

    except Exception:
        return None


async def stream_chat(
    content_list: List[Dict[str, Any]],
    voice: str = "Cherry",
    audio_format: str = "wav",
) -> AsyncGenerator[OmniStreamPiece, None]:

    parts = []

    for item in content_list:

        if item["type"] == "text":
            parts.append(item["text"])

        elif item["type"] == "image_url":
            img = _decode_data_url(item["image_url"]["url"])
            if img is not None:
                parts.append(img)

    response = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=parts,
    )

    for chunk in response:

        if chunk.text:

            yield OmniStreamPiece(
                text_delta=chunk.text,
                audio_b64=None,
            )

        await asyncio.sleep(0)