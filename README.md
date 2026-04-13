# 🎧 Third Ear
**Real-Time AI Audio Description for Live Video**

*A project built for SWMxGemini Hackathon - Track 2: Real-Time Multimodal AI*

## 👁️ The Problem: Left in the Dark
The digital world has shifted from static, pre-recorded media to live broadcasting, but accessibility has completely failed to keep up. 

* **2.2+ Billion** people globally live with near or distance vision impairment.
* **Traditional Audio Description (AD)** relies entirely on human writers and voice actors. It is slow, highly expensive, and restricted to pre-recorded media (e.g., Netflix).
* **The Gap:** Live streams, breaking news, user-generated content, and interactive broadcasts currently offer **zero** accessibility. The visually impaired community is locked out of the modern streaming era.

## 💡 The Solution
**Third Ear** is a fully automated, real-time audio description engine that requires zero human input. It watches live streams and instantly generates spoken descriptions of on-screen action.

**Killer Feature: "Smart Injection"**
If an AI talks constantly, it ruins the experience by talking over the native dialogue. Third Ear doesn't just watch; it listens. Our custom pipeline continuously monitors the stream's native audio track. Using neural Voice Activity Detection (VAD), the system waits for natural silences and pauses in the dialogue, seamlessly injecting the AI-generated spoken description *only* during these gaps. 

## ⚙️ Under the Hood: The Tech Stack

Our zero-latency pipeline is built using industry-standard real-time media processing tools and state-of-the-art multimodal AI.

* **[Fishjam](https://fishjam.io/):** Handles the robust, low-latency real-time video and audio streaming infrastructure.
* **[Gemini Live API](https://ai.google.dev/):** Acts as our "visual brain," continuously analyzing incoming video frames to generate context-aware descriptions of the on-screen action.
* **[PyAV](https://pyav.org/) / FFmpeg (via PyAV):** Intercepts the live stream to demux, transcode, and mix media packets directly in Python memory.
* **[PyTorch](https://pytorch.org/) & [Torchaudio](https://pytorch.org/audio/):** The backend framework powering high-speed audio data transformation.
* **VAD:** Analyzes the local stream's audio track in milliseconds to identify guaranteed silence gaps for the description overlay.

### 🔄 Architecture Flow
1. **Ingest:** Fishjam receives the live broadcast stream.
2. **Fork:** The stream is split. Video frames are sent to the Gemini Live API; audio is routed to our local PyAV processor.
3. **Analyze:** Gemini generates text descriptions of visual events. Simultaneously, Torchaudio/VAD analyzes the audio for human speech.
4. **Inject:** When Gemini outputs a description AND the VAD detects a pause in dialogue, the system triggers Text-to-Speech (TTS).
5. **Broadcast:** PyAV mixes the generated AI voice back into the audio track and sends it to Fishjam to stream to the end user.


## 📦 Usage

### Prerequisites

- **Python 3.11** (exact version — see `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** package manager (recommended) or pip
- **FFmpeg** installed and available on `PATH` (required by PyAV / pydub)
- **Node.js ≥ 18** (for the frontend)
- **Google Cloud credentials** with access to:
  - Vertex AI (Gemini API)
  - Cloud Text-to-Speech API

### Installation

```bash
# Clone the repo
git clone https://github.com/RETIOM/hackathon-SWMxGemini.git
cd hackathon-SWMxGemini

# Create venv & install Python deps
uv sync

# Install frontend deps
cd frontend && npm install && cd ..
```

### Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to your GCP service account JSON key |
| `GOOGLE_CLOUD_PROJECT` | Your GCP project ID (defaults to `swmxgemini`) |

### Running the Backend (API Server)

The FastAPI server accepts video uploads via SSE and streams back described segments:

```bash
# Start the dev server on port 8000 (with hot reload)
uv run uvicorn src.server:app --reload --port 8000
```

The server exposes:
- **`POST /api/process`** — upload an MP4 file; returns an SSE stream of base64-encoded described video segments with AI-generated narration text.

### Running the Frontend

```bash
cd frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173), drag & drop an MP4 file, and watch the AI-described version stream back in real time.

### CLI Pipeline (Offline Processing)

Process a video file directly from the command line without the web UI:

```bash
# Basic usage — outputs segments to ./pipeline_output/
uv run python src/pipeline.py input_video.mp4

# Custom options
uv run python src/pipeline.py input_video.mp4 \
  --project-id my-gcp-project \
  --output-dir ./output \
  --output-mode single_file \
  --chunk-duration 10.0 \
  --realtime
```

| Flag | Default | Description |
|---|---|---|
| `--project-id` | `swmxgemini` | GCP project ID for Vertex AI and TTS |
| `--output-dir` | `pipeline_output` | Directory for output MP4 files |
| `--output-mode` | `segmented` | `segmented` (one file per chunk) or `single_file` (one concatenated output) |
| `--chunk-duration` | `10.0` | Duration of each processing chunk in seconds |
| `--realtime` | off | Simulate real-time ingestion speed |

### Running Tests

```bash
# Run all tests (42 tests, no API keys needed)
uv run pytest tests/ -v

# Run only the sanity/unit tests
uv run pytest tests/test_sanity.py -v

# Run narrator unit tests (mocked API clients)
uv run pytest src/narrator/test_narrator.py -v
```

## 🚀 Future Roadmap & Impact
This project was built with Track 2's "Multi-user scenarios" in mind. Our next steps include:
* **Multi-User Personalization:** Utilizing Fishjam's routing to send personalized audio streams to different users in the same room (e.g., basic action cues vs. highly detailed cinematic descriptions).
* **Global Reach:** Real-time translation of generated audio descriptions into multiple languages on the fly.
* **Broad Application:** Expanding beyond entertainment to live educational webinars, e-sports, and breaking news broadcasts.

