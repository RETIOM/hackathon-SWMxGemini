import asyncio
import logging
import io
from PIL import Image
from narrator import StreamingAINarrator

logging.basicConfig(level=logging.INFO)


async def main():
    img = Image.new("RGB", (640, 480), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    dummy_frame = buf.getvalue()

    narrator = StreamingAINarrator(project_id="swmxgemini")

    print("--- Starting Smoke Test ---")
    print("Requesting narration for 5 dummy frames...")

    try:
        result = await narrator.process_chunk([dummy_frame] * 5)

        print(f"\nSUCCESS!")
        print(f"Description: '{result.text}'")
        print(f"Audio Length: {len(result.audio_bytes)} bytes")

        with open("smoke_test_output.mp3", "wb") as f:
            f.write(result.audio_bytes)

        print("\nTest audio saved to: smoke_test_output.mp3")
        print("You can play this file to verify the sync and voice.")

    except Exception as e:
        print(f"\nFAILED: {e}")


if __name__ == "__main__":
    asyncio.run(main())
