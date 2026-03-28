"""Pipeline — Wires the ChunkingIngestor to the StreamingAINarrator.

Usage:
    uv run src/pipeline.py <video_file> [--project-id <id>] [--output-dir <dir>]

Reads a video file, chunks it into 5-second blocks via the ingestor,
generates audio descriptions for each chunk via the narrator, and saves
the resulting MP3 files sequentially.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import pathlib

# Ensure both sibling packages are importable
_src_dir = str(pathlib.Path(__file__).resolve().parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from ingestor.ingestor import ChunkingIngestor
from narrator.narrator import StreamingAINarrator, NarratorResult

logger = logging.getLogger(__name__)


async def run_pipeline(
    video_path: str,
    project_id: str,
    output_dir: str,
    chunk_duration: float = 10.0,
    simulate_realtime: bool = False,
) -> list[NarratorResult]:
    """Run the full ingestor → narrator pipeline.

    Args:
        video_path: Path to the input video file.
        project_id: GCP project ID for Gemini and TTS.
        output_dir: Directory to write output MP3 files.
        chunk_duration: Duration of each chunk in seconds.
        simulate_realtime: If True, the ingestor waits in real-time between chunks.

    Returns:
        List of NarratorResult objects, one per chunk.
    """
    os.makedirs(output_dir, exist_ok=True)

    ingestor = ChunkingIngestor(
        video_path=video_path,
        chunk_duration=chunk_duration,
        simulate_realtime=simulate_realtime,
    )
    narrator = StreamingAINarrator(project_id=project_id)
    await narrator.warmup()

    ingestor.start()
    results: list[NarratorResult] = []
    chunk_num = 0

    try:
        while True:
            # Bridge: blocking queue.get() → async via asyncio.to_thread
            chunk = await asyncio.to_thread(
                ingestor.processing_queue.get, timeout=30.0
            )

            # None sentinel = end of stream
            if chunk is None:
                logger.info("End-of-stream sentinel received.")
                break

            chunk_num += 1
            logger.info(
                "Chunk %d received [%.1fs–%.1fs] (%d frames)",
                chunk_num,
                chunk.start_time,
                chunk.end_time,
                len(chunk.compressed_frames),
            )

            # Run narrator
            result = await narrator.process_media_chunk(chunk)
            results.append(result)

            # Save MP3
            mp3_path = os.path.join(output_dir, f"chunk_{chunk_num:03d}.mp3")
            with open(mp3_path, "wb") as f:
                f.write(result.audio_bytes)

            status = f"'{result.text}'" if result.text else "(silent fallback)"
            print(f"  [{chunk.start_time:.1f}s–{chunk.end_time:.1f}s] {status} → {mp3_path}")

    except Exception as e:
        logger.error("Pipeline error: %s", e, exc_info=True)
    finally:
        ingestor.stop()

    print(f"\nDone — {chunk_num} chunks processed, {len(results)} audio files saved to {output_dir}/")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Video → Audio Description Pipeline")
    parser.add_argument("video_file", help="Path to the input video file")
    parser.add_argument("--project-id", default="swmxgemini", help="GCP project ID")
    parser.add_argument("--output-dir", default="pipeline_output", help="Output directory for MP3 files")
    parser.add_argument("--realtime", action="store_true", help="Simulate real-time playback speed")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    asyncio.run(
        run_pipeline(
            video_path=args.video_file,
            project_id=args.project_id,
            output_dir=args.output_dir,
            simulate_realtime=args.realtime,
        )
    )


if __name__ == "__main__":
    main()
