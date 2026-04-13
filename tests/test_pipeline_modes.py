from __future__ import annotations

import asyncio
import pathlib
import queue
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ingestor.models import MediaChunk
from narrator.narrator import NarratorResult
from pipeline import DescribedVideoPipeline


class FakeIngestor:
    def __init__(self, chunks: list[MediaChunk | None]):
        self.processing_queue = queue.Queue()
        for chunk in chunks:
            self.processing_queue.put(chunk)
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakeNarrator:
    def __init__(self, fail: bool = False):
        self.fail = fail

    async def process_media_chunk(self, _chunk: MediaChunk) -> NarratorResult:
        if self.fail:
            raise RuntimeError("narrator failed")
        return NarratorResult(audio_bytes=b"narration", text="ok")

    def _generate_silent_audio(self) -> bytes:
        return b"silent"


class FakeVAP:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def process_chunk(self, chunk: MediaChunk) -> list[float]:
        if self.fail:
            raise RuntimeError("vap failed")
        return [0.5] * len(chunk.raw_video_frames)


class FakeSynchronizer:
    def __init__(self):
        self.process_calls = 0

    def process(
        self,
        chunk: MediaChunk,
        narrator_result: NarratorResult,
        vap_masks: list[float],
    ) -> bytes:
        self.process_calls += 1
        assert len(vap_masks) == len(chunk.raw_video_frames)
        if narrator_result.text == "":
            assert narrator_result.audio_bytes
            assert all(mask == 1.0 for mask in vap_masks)
        return f"segment-{self.process_calls}".encode("utf-8")

    def concat_segments(self, segments):
        return b"|".join(segments)


def _chunk(start: float, end: float, frames: int = 3) -> MediaChunk:
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    return MediaChunk(
        start_time=start,
        end_time=end,
        raw_video_frames=[frame.copy() for _ in range(frames)],
        compressed_frames=[b"jpeg"] * frames,
        raw_audio_bytes=None,
        audio_sample_rate=48000,
        audio_channels=2,
    )


def test_segmented_mode_yields_per_chunk() -> None:
    ingestor = FakeIngestor([_chunk(0.0, 5.0), _chunk(5.0, 10.0), None])
    pipeline = DescribedVideoPipeline(
        ingestor=ingestor,
        narrator=FakeNarrator(),
        vap=FakeVAP(),
        synchronizer=FakeSynchronizer(),
        output_mode="segmented",
    )

    async def _run() -> list[bytes]:
        output = []
        async for payload in pipeline.run():
            output.append(payload)
        return output

    result = asyncio.run(_run())
    assert result == [b"segment-1", b"segment-2"]
    assert ingestor.started is True
    assert ingestor.stopped is True


def test_single_file_mode_yields_one_payload() -> None:
    ingestor = FakeIngestor([_chunk(0.0, 5.0), _chunk(5.0, 10.0), None])
    pipeline = DescribedVideoPipeline(
        ingestor=ingestor,
        narrator=FakeNarrator(),
        vap=FakeVAP(),
        synchronizer=FakeSynchronizer(),
        output_mode="single_file",
    )

    async def _run() -> list[bytes]:
        output = []
        async for payload in pipeline.run():
            output.append(payload)
        return output

    result = asyncio.run(_run())
    assert result == [b"segment-1|segment-2"]


def test_fallbacks_keep_pipeline_running() -> None:
    ingestor = FakeIngestor([_chunk(0.0, 5.0, frames=2), None])
    pipeline = DescribedVideoPipeline(
        ingestor=ingestor,
        narrator=FakeNarrator(fail=True),
        vap=FakeVAP(fail=True),
        synchronizer=FakeSynchronizer(),
        output_mode="segmented",
    )

    async def _run() -> list[bytes]:
        output = []
        async for payload in pipeline.run():
            output.append(payload)
        return output

    result = asyncio.run(_run())
    assert result == [b"segment-1"]
