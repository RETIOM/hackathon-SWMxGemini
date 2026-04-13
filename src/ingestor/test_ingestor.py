import time
from ingestor import ChunkingIngestor
import numpy as np


def run_test():
    """Testing routine that acts as the Consumer (Queue_A watcher) reading slices dynamically natively."""
    video_file = "/home/mateusz/hackathon-SWMxGemini/input.mp4"
    print(f"Testing the Chunking Ingestor natively using {video_file}...")

    block_duration = 5.0
    ingestor = ChunkingIngestor(video_path=video_file, chunk_duration=block_duration)

    ingestor.start()

    chunk_counter = 1

    try:
        while True:
            chunk = ingestor.processing_queue.get(timeout=10.0)

            if chunk is None:
                print(
                    "--- Sentinel End-of-Stream received. Consumer exiting smoothly natively."
                )
                break

            print(
                f"--- Received Block {chunk_counter}: [{chunk.start_time:.1f}s - {chunk.end_time:.1f}s] ---"
            )
            print(f"    Raw Video Frames Count:  {len(chunk.raw_video_frames)}")
            if chunk.raw_video_frames:
                example_frame = chunk.raw_video_frames[0]
                print(
                    f"    Raw Numpy Shape/Type:    {example_frame.shape} | {example_frame.dtype}"
                )
            print(
                f"    Raw Audio Length:        {len(chunk.raw_audio_bytes) if chunk.raw_audio_bytes else 'None'} bytes natively"
            )
            print(
                f"    Extracted JPEG Count:    {len(chunk.compressed_frames)} locally"
            )
            print("-" * 65)

            import os
            import json
            import base64

            os.makedirs("chunk_outputs", exist_ok=True)

            payload = {
                "block_id": chunk_counter,
                "start_time": chunk.start_time,
                "end_time": chunk.end_time,
                "frames": [
                    base64.b64encode(img).decode("utf-8")
                    for img in chunk.compressed_frames
                ],
                "audio": base64.b64encode(chunk.raw_audio_bytes).decode("utf-8")
                if chunk.raw_audio_bytes
                else None,
            }

            json_path = f"chunk_outputs/block_{chunk_counter}.json"
            with open(json_path, "w") as f:
                json.dump(payload, f)

            print(f"    -> Serialized Block {chunk_counter} gracefully to {json_path}")

            chunk_counter += 1

    except KeyboardInterrupt:
        print("Aborting gracefully locally.")
    except Exception as e:
        print(f"Test crashed naturally: {e}")
    finally:
        ingestor.stop()
        print("Consumer test completed natively.")


if __name__ == "__main__":
    run_test()
