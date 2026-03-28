"""
Combined test: verifies both per-frame VAP mask creation and chunk JSON serialization.
Processes a few blocks from input.mp4, checks mask integrity, and saves
each chunk as a JSON file with embedded masks alongside the frame/audio bytestreams.
"""
import os
import json
import base64
from ingestor import ChunkingIngestor
from vap import VAP

OUTPUT_DIR = "chunk_outputs"
VIDEO_FILE = "/home/mateusz/Downloads/594249580_25246747415017303_8710433012125806941_n.mp4"
MAX_BLOCKS = 2  # Process first 2 blocks for fast testing


def test_mask_and_save():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ingestor = ChunkingIngestor(video_path=VIDEO_FILE, chunk_duration=5.0, simulate_realtime=False)
    vap = VAP(strategy="rms")

    ingestor.start()
    block_num = 0
    all_passed = True

    try:
        while block_num < MAX_BLOCKS:
            chunk = ingestor.processing_queue.get(timeout=15.0)
            if chunk is None:
                break

            block_num += 1
            num_frames = len(chunk.raw_video_frames)
            num_jpegs = len(chunk.compressed_frames)

            # ── Mask creation tests ──────────────────────────────────
            masks = vap.process_chunk(chunk)

            # 1. One mask per frame
            assert len(masks) == num_frames, (
                f"Block {block_num}: mask count {len(masks)} != frame count {num_frames}"
            )

            # 2. Every mask value is a valid float in [0, 1]
            for i, m in enumerate(masks):
                assert isinstance(m, float), f"Block {block_num}, frame {i}: mask is not float"
                assert 0.0 <= m <= 1.0, f"Block {block_num}, frame {i}: mask {m} out of range"

            # 3. JPEG count matches frame count
            assert num_jpegs == num_frames, (
                f"Block {block_num}: jpeg count {num_jpegs} != frame count {num_frames}"
            )

            ducking = sum(1 for m in masks if m < 1.0)
            print(f"✓ Block {block_num} masks: {num_frames} frames, {ducking} ducking")

            # ── Chunk saving test ────────────────────────────────────
            payload = {
                "block_id": block_num,
                "start_time": chunk.start_time,
                "end_time": chunk.end_time,
                "frame_count": num_frames,
                "masks": masks,
                "frames": [
                    base64.b64encode(jpg).decode("utf-8")
                    for jpg in chunk.compressed_frames
                ],
                "audio": (
                    base64.b64encode(chunk.raw_audio_bytes).decode("utf-8")
                    if chunk.raw_audio_bytes
                    else None
                ),
            }

            json_path = os.path.join(OUTPUT_DIR, f"block_{block_num}.json")
            with open(json_path, "w") as f:
                json.dump(payload, f)

            # 4. File was written and is valid JSON
            assert os.path.exists(json_path), f"Block {block_num}: JSON file missing"
            with open(json_path, "r") as f:
                loaded = json.load(f)

            # 5. Round-trip integrity: counts match
            assert loaded["frame_count"] == num_frames
            assert len(loaded["frames"]) == num_frames
            assert len(loaded["masks"]) == num_frames

            # 6. Verify a JPEG can be decoded back from base64
            first_jpg = base64.b64decode(loaded["frames"][0])
            assert first_jpg[:2] == b"\xff\xd8", "First frame is not a valid JPEG"

            # 7. Verify audio round-trip (if present)
            if loaded["audio"]:
                audio_bytes = base64.b64decode(loaded["audio"])
                assert len(audio_bytes) == len(chunk.raw_audio_bytes)

            file_kb = os.path.getsize(json_path) / 1024
            print(f"✓ Block {block_num} saved: {json_path} ({file_kb:.0f} KB)")

    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        all_passed = False
    finally:
        ingestor.stop()

    print()
    if all_passed:
        print(f"═══ ALL TESTS PASSED ({block_num} blocks) ═══")
    else:
        print("═══ SOME TESTS FAILED ═══")


if __name__ == "__main__":
    test_mask_and_save()
