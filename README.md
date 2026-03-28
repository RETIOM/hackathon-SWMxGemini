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


## 🚀 Future Roadmap & Impact
This project was built with Track 2's "Multi-user scenarios" in mind. Our next steps include:
* **Multi-User Personalization:** Utilizing Fishjam's routing to send personalized audio streams to different users in the same room (e.g., basic action cues vs. highly detailed cinematic descriptions).
* **Global Reach:** Real-time translation of generated audio descriptions into multiple languages on the fly.
* **Broad Application:** Expanding beyond entertainment to live educational webinars, e-sports, and breaking news broadcasts.

