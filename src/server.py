import asyncio
import base64
import json
import logging
import os
import shutil
import sys
import tempfile
import uuid
import pathlib
from typing import AsyncGenerator

_src_dir = str(pathlib.Path(__file__).resolve().parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ingestor.ingestor import ChunkingIngestor
from ingestor.vap import VAP
from narrator.narrator import StreamingAINarrator
from pipeline import DescribedVideoPipeline
from synchronizer import Synchronizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SWM Gemini Video Streaming API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def sse_event(event_type: str, data: dict) -> str:
    """Format data as Server-Sent Event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/process")
async def process_video(file: UploadFile = File(...)):
    """Accept an MP4 file and stream described fragments back via SSE."""
    ext = os.path.splitext(file.filename or "")[1] or ".mp4"
    temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{ext}")
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    async def event_stream() -> AsyncGenerator[str, None]:
        yield await sse_event("status", {"message": "Initializing pipeline..."})
        try:
            ingestor = ChunkingIngestor(
                video_path=temp_path,
                chunk_duration=10.0,
                simulate_realtime=False,
            )
            narrator = StreamingAINarrator(project_id="swmxgemini")
            await narrator.warmup()

            pipeline = DescribedVideoPipeline(
                ingestor=ingestor,
                narrator=narrator,
                vap=VAP(),
                synchronizer=Synchronizer(),
                output_mode="segmented_with_text",
            )

            chunk_idx = 0
            async for payload in pipeline.run():
                chunk_idx += 1
                segment_bytes, text = payload
                b64_data = base64.b64encode(segment_bytes).decode("utf-8")

                yield await sse_event(
                    "segment",
                    {
                        "index": chunk_idx,
                        "text": text,
                        "data": b64_data,
                    },
                )

            yield await sse_event("done", {"total_chunks": chunk_idx})

        except asyncio.CancelledError:
            logger.info("Client disconnected, pipeline cancelled.")
        except Exception as e:
            logger.error("Pipeline error", exc_info=True)
            yield await sse_event("error", {"message": str(e)})
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
