# Implementation Plan: Merged Video + Audio Description Pipeline

## Overview

Wire `ChunkingIngestor`, `StreamingAINarrator`, and `VAP` together into a unified pipeline that outputs a merged MP4 bytestream with ducked original audio and AI-generated narration overlay.

---

## Architecture

```
ChunkingIngestor
      │
   Queue_A (MediaChunk)
      │
      ├──────────────────────────────┐
      │                              │
      ▼                              ▼
StreamingAINarrator (async)     VAP (sync → asyncio.to_thread)
      │                              │
  NarratorResult                List[float] ducking masks
  (MP3 audio desc.)                  │
      │                              │
      └──────────────┬───────────────┘
                     │
              + original frames/audio (passthrough)
                     │
                     ▼
              Synchronizer
         (duck + mix audio, mux via PyAV)
                     │
                     ▼
           output MP4 bytestream
```

`asyncio.gather` runs narrator and VAP concurrently per chunk.
On narrator failure → silence. On VAP failure → mask=1.0 (no ducking).

---

## New Components

### 1. `synchronizer.py` — `Synchronizer` class

Core new piece. Accepts results from both `StreamingAINarrator` and `VAP` for a single `MediaChunk`, then muxes everything into an MP4 bytestream.

**Responsibilities:**
- Apply VAP ducking mask to original audio (multiply sample amplitudes per frame-aligned window)
- Mix ducked original audio with narration MP3 from `NarratorResult`
- Re-encode combined audio + original video frames into in-memory MP4 using PyAV
- Return `bytes` (the MP4 segment)

**Key method signature:**
```python
def process(
    self,
    chunk: MediaChunk,
    narrator_result: NarratorResult,
    vap_masks: list[float],
) -> bytes:  # raw MP4 segment bytes
```

**Ducking implementation detail:**

The VAP returns one `float` per video frame. Map those to audio samples:

```python
samples_per_frame = chunk.audio_sample_rate / fps
for frame_idx, mask in enumerate(vap_masks):
    start = int(frame_idx * samples_per_frame)
    end = int((frame_idx + 1) * samples_per_frame)
    audio_array[start:end] *= mask
```

Use numpy on the raw float32 audio from `chunk.raw_audio_bytes`. After ducking, re-encode to pydub `AudioSegment` for mixing via `overlay()`.

**Output format:**

Write a **fragmented MP4** (fMP4) bytestream — each segment is self-contained and streamable (DASH/HLS compatible). In PyAV:

```python
container = av.open(output_buffer, mode='w', format='mp4')
# set movflags: frag_keyframe+empty_moov+default_base_moof
```

If streaming is not a requirement, plain MP4 segments concatenated at the end also work, but fMP4 is the safer default for a real-time pipeline.

---

### 2. `pipeline.py` — `DescribedVideoPipeline` orchestrator

Top-level class tying all components together.

```python
class DescribedVideoPipeline:
    def __init__(self, ingestor, narrator, vap, synchronizer): ...
    async def run(self) -> AsyncIterator[bytes]: ...
```

**Loop logic per chunk:**
```python
chunk = ingestor.processing_queue.get()
narrator_result, vap_masks = await asyncio.gather(
    narrator.process_media_chunk(chunk),
    asyncio.to_thread(vap.process_chunk, chunk),  # VAP is sync
)
segment_bytes = synchronizer.process(chunk, narrator_result, vap_masks)
yield segment_bytes
```

Note: `VAP.process_chunk` is synchronous (uses `torch`), so wrap it in `asyncio.to_thread` to avoid blocking the event loop.

---

## Changes to Existing Files

### `narrator.py`
- No functional changes needed.
- **Cleanup:** `VISION_PROMPT_TEMPLATE` is defined twice (the second definition silently overrides the first). Remove the first definition.

### `vap.py`
- No changes needed. Interface is already pipeline-compatible.

### `ingestor.py`
- `raw_audio_bytes` is built from `frame.to_ndarray().tobytes()` for `AudioFrame`. PyAV's `to_ndarray()` on a stereo `fltp` frame yields shape `(channels, samples)`. `vap.py` handles this correctly already, but add a comment in the ingestor clarifying the byte layout so `VAP._prepare_audio` and `Synchronizer` can rely on the same documented contract.

---

## Suggested File Structure

```
src/
  ingestor/
    models.py          # MediaChunk (existing)
  narrator.py          # existing
  vap.py               # existing
  synchronizer.py      # NEW — Synchronizer class
  pipeline.py          # NEW — DescribedVideoPipeline orchestrator
```

---

## Failure Mode Handling

| Failure | Behaviour |
|---|---|
| Narrator timeout/error | Already returns silent `NarratorResult(audio_bytes=silence, text="")` — Synchronizer mixes silence |
| VAP model load failure | Already falls back to `strategy="rms"`; if that also fails, `process_chunk` returns `[1.0]*n` (no ducking) |
| Empty audio chunk | `chunk.raw_audio_bytes is None` — Synchronizer skips audio mixing, writes video-only segment |
| Queue sentinel `None` | Pipeline loop exits cleanly on `None` chunk |

---

## Open Question

Before implementation, decide whether the output needs to be:

- **A continuous single file** — requires a final mux pass concatenating all segments. Affects how `Pipeline.run()` terminates and whether a finalizer step is needed.
- **A segmented stream** — fMP4 segments work directly, no finalizer needed.

This choice drives the termination logic in `DescribedVideoPipeline.run()`.
