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

    # client.models.generate_content_stream(...) and iterating its response
    # are both blocking/synchronous under the hood — running them directly
    # inside this async function would stall the whole event loop (other
    # websockets, camera frames, etc.) for the duration of generation. Run
    # it in a worker thread instead and relay chunks back through a queue,
    # so this stays non-blocking and still streams incrementally.
    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue" = asyncio.Queue()
    _DONE = object()

    def _run_stream():
        try:
            response = client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=parts,
            )
            for chunk in response:
                if chunk.text:
                    loop.call_soon_threadsafe(queue.put_nowait, chunk.text)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)

    loop.run_in_executor(None, _run_stream)

    while True:
        item = await queue.get()
        if item is _DONE:
            break
        if isinstance(item, Exception):
            raise item
        yield OmniStreamPiece(text_delta=item, audio_b64=None)