"""Pipeline orchestration for described video output.

Supports two output modes controlled by a flag:
- segmented: yields one self-contained fMP4 segment per chunk
- single_file: yields one final MP4 after end-of-stream
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal, cast

# Ensure sibling packages are importable
_src_dir = str(pathlib.Path(__file__).resolve().parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from ingestor.ingestor import ChunkingIngestor
from narrator.narrator import NarratorResult, StreamingAINarrator
from synchronizer import Synchronizer

logger = logging.getLogger(__name__)

OutputMode = Literal["segmented", "single_file", "segmented_with_text"]


@dataclass(slots=True)
class DescribedVideoPipeline:
    """Runs the full ingestor + narrator + VAP + synchronizer flow."""

    ingestor: Any
    narrator: Any
    vap: Any
    synchronizer: Any
    output_mode: OutputMode = "segmented"
    queue_timeout_s: float = 30.0

    async def run(self) -> AsyncIterator[bytes | tuple[bytes, str]]:
        """Yield output MP4 bytes according to the configured output mode."""
        if self.output_mode not in ("segmented", "single_file", "segmented_with_text"):
            raise ValueError(f"Unsupported output_mode: {self.output_mode}")

        segments: list[bytes] = []
        self.ingestor.start()

        sync_task = None
        pending_text = ""

        try:
            while True:
                chunk = await asyncio.to_thread(
                    self.ingestor.processing_queue.get,
                    timeout=self.queue_timeout_s,
                )

                if chunk is None:
                    logger.info("End-of-stream sentinel received.")
                    # Flush the final sync task
                    if sync_task:
                        segment = await sync_task
                        if self.output_mode == "segmented":
                            yield segment
                        elif self.output_mode == "segmented_with_text":
                            yield segment, pending_text
                        else:
                            segments.append(segment)
                    break

                # Start Gemini for CURRENT chunk immediately
                model_task = asyncio.create_task(self._process_chunk_models(chunk))

                # While Gemini runs, wait for PREVIOUS chunk's synchronizer to finish and yield it
                if sync_task:
                    segment = await sync_task
                    if self.output_mode == "segmented":
                        yield segment
                    elif self.output_mode == "segmented_with_text":
                        yield segment, pending_text
                    else:
                        segments.append(segment)

                # Now wait for CURRENT chunk's Gemini to finish
                narrator_result, vap_masks = await model_task

                # Start the synchronizer for CURRENT chunk in background
                sync_task = asyncio.create_task(
                    asyncio.to_thread(
                        self.synchronizer.process,
                        chunk,
                        narrator_result,
                        vap_masks,
                    )
                )
                pending_text = narrator_result.text

            if self.output_mode == "single_file":
                final_mp4 = await asyncio.to_thread(self.synchronizer.concat_segments, segments)
                if final_mp4:
                    yield final_mp4
        finally:
            self.ingestor.stop()

    async def _process_chunk_models(self, chunk) -> tuple[NarratorResult, list[float]]:
        narrator_task = self.narrator.process_media_chunk(chunk)
        vap_task = asyncio.to_thread(self.vap.process_chunk, chunk)

        narrator_res, vap_res = await asyncio.gather(
            narrator_task,
            vap_task,
            return_exceptions=True,
        )

        if isinstance(narrator_res, Exception):
            logger.warning("Narrator failed for chunk; using silent fallback", exc_info=True)
            narrator_res = self._silent_result()

        if isinstance(vap_res, Exception):
            logger.warning("VAP failed for chunk; using no-duck masks", exc_info=True)
            vap_res = [1.0] * len(chunk.raw_video_frames)

        if not isinstance(vap_res, list):
            vap_res = [1.0] * len(chunk.raw_video_frames)

        if len(vap_res) < len(chunk.raw_video_frames):
            vap_res = vap_res + [1.0] * (len(chunk.raw_video_frames) - len(vap_res))
        elif len(vap_res) > len(chunk.raw_video_frames):
            vap_res = vap_res[: len(chunk.raw_video_frames)]

        return cast(NarratorResult, narrator_res), [float(v) for v in vap_res]

    def _silent_result(self) -> NarratorResult:
        if hasattr(self.narrator, "_generate_silent_audio"):
            try:
                return NarratorResult(
                    audio_bytes=self.narrator._generate_silent_audio(),
                    text="",
                )
            except Exception:
                logger.warning("Failed to generate narrator silence", exc_info=True)
        return NarratorResult(audio_bytes=b"", text="")


async def run_pipeline(
    video_path: str,
    project_id: str,
    output_dir: str,
    chunk_duration: float = 5.0,
    simulate_realtime: bool = False,
    output_mode: OutputMode = "segmented",
) -> list[str]:
    """Run the full merged-media pipeline and write outputs to disk."""
    from ingestor.vap import VAP

    os.makedirs(output_dir, exist_ok=True)

    ingestor = ChunkingIngestor(
        video_path=video_path,
        chunk_duration=chunk_duration,
        simulate_realtime=simulate_realtime,
    )
    narrator = StreamingAINarrator(project_id=project_id)
    await narrator.warmup()

    pipeline = DescribedVideoPipeline(
        ingestor=ingestor,
        narrator=narrator,
        vap=VAP(),
        synchronizer=Synchronizer(),
        output_mode=output_mode,
    )

    saved_paths: list[str] = []
    chunk_num = 0
    async for payload in pipeline.run():
        if output_mode == "segmented":
            chunk_num += 1
            out_path = os.path.join(output_dir, f"segment_{chunk_num:03d}.mp4")
        else:
            out_path = os.path.join(output_dir, "final.mp4")

        with open(out_path, "wb") as f:
            f.write(payload)

        saved_paths.append(out_path)
        logger.info("Wrote output: %s", out_path)

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Described video pipeline")
    parser.add_argument("video_file", help="Path to the input video file")
    parser.add_argument("--project-id", default="swmxgemini", help="GCP project ID")
    parser.add_argument("--output-dir", default="pipeline_output", help="Output directory")
    parser.add_argument(
        "--output-mode",
        choices=["segmented", "single_file"],
        default="segmented",
        help="Output mode: chunk segments or one final file",
    )
    parser.add_argument("--chunk-duration", type=float, default=10.0, help="Chunk duration in seconds")
    parser.add_argument("--realtime", action="store_true", help="Simulate real-time ingestion")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    outputs = asyncio.run(
        run_pipeline(
            video_path=args.video_file,
            project_id=args.project_id,
            output_dir=args.output_dir,
            chunk_duration=args.chunk_duration,
            simulate_realtime=args.realtime,
            output_mode=args.output_mode,
        )
    )

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
