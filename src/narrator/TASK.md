### **Prerequisites & Environment Setup**
Before writing the class, ensure your environment is prepared for asynchronous API calls and audio manipulation.
1.  **System Packages:** Install `ffmpeg` on the host server (required by Python audio libraries to process and stretch media).
2.  **Python Libraries:** Install `google-cloud-aiplatform` (for Vertex AI), `google-cloud-texttospeech`, and `pydub` (for audio manipulation).
3.  **Authentication:** Set up Google Cloud Application Default Credentials (ADC) on the host machine with a service account that has Vertex AI and Cloud TTS permissions.

---

### **Step 1: Module Initialization & State Management**
Create a standalone Python class (e.g., `StreamingAINarrator`) that holds the API clients and tracks the narrative state.

* **Initialization Variables:**
    * `target_duration_ms`: Set to `5000` (representing the strict 5.0-second output requirement).
    * `previous_context`: A string initialized to `""`.
    * `consecutive_failures`: An integer initialized to `0` to track when to reset the context.
* **Client Setup:**
    * Initialize the Vertex AI `GenerativeModel` explicitly targeting `"gemini-2.5-flash-lite"`.
    * Initialize the Google Cloud `TextToSpeechAsyncClient`.

---

### **Step 2: Phase 1 - Vision AI Processing**
Implement a private asynchronous method (e.g., `_generate_description`) to handle the Gemini API interaction.

* **Input:** A list of 5 compressed JPEG byte strings.
* **Payload Construction:** Map the raw bytes into Vertex AI `Part` objects (`mime_type="image/jpeg"`).
* **Prompting:** Combine the `previous_context` with the strict directive: *"Describe the continuous action in these sequential frames. CRITICAL: Write ONE sentence. Maximum 10 words. Focus only on physical movement."*
* **Execution:** Call `generate_content_async`.
* **Cleanup:** As soon as the text is returned, explicitly `del` the input list of JPEG bytes to trigger Python's garbage collection and prevent memory leaks.
* **Output:** Return the generated text string.

---

### **Step 3: Phase 2 - Speech Synthesis**
Implement a private asynchronous method (e.g., `_generate_tts`) to handle the text-to-audio conversion.

* **Input:** The text string generated in Step 2.
* **Voice Configuration:** Create a `VoiceSelectionParams` object. Select a high-quality voice model like `"en-US-Journey-D"` or `"en-US-Neural2-J"`.
* **Audio Configuration:** Create an `AudioConfig` object. Set the format to `MP3`. Crucially, set the `speaking_rate` to `1.15`. (This slight acceleration makes it highly likely the resulting audio will naturally fall under 5.0 seconds).
* **Execution:** Call `synthesize_speech` asynchronously.
* **Output:** Return the raw MP3 audio bytes.

---

### **Step 4: Phase 3 - The Synchronization Engine**
Implement a synchronous private method (e.g., `_sync_audio_duration`) to mechanically fit the audio to the 5.0-second window.

* **Input:** The raw MP3 audio bytes from Step 3.
* **Measurement:** Use `pydub.AudioSegment.from_file()` to load the bytes into memory. Measure its length in milliseconds (`len(audio)`).
* **Branching Logic:**
    * **If Length == 5000ms:** Do nothing.
    * **If Length < 5000ms (Underflow):** Generate a silent `AudioSegment` where `duration = 5000 - len(audio)`. Append this silence to the end of the TTS audio track.
    * **If Length > 5000ms (Overflow):** Calculate the stretch ratio: `len(audio) / 5000`. Use Pydub's `speedup(playback_speed=ratio)` to compress the audio footprint without shifting the pitch.
* **Output:** Export the fitted `AudioSegment` back into raw MP3 bytes and return them.

---

### **Step 5: The "Silent Fallback" Generator**
Implement a simple helper method (e.g., `_generate_silent_audio`) for emergency use.
* **Action:** Use Pydub to generate exactly 5000ms of pure silence.
* **Output:** Return the exported MP3 bytes.

---

### **Step 6: The Main Orchestration Method**
This is the public method (e.g., `process_chunk`) that your existing ingestor will call. It acts as the traffic controller and enforces the strict streaming constraints.

* **Input:** The list of JPEG bytes.
* **Execution Flow:**
    1.  **Try Block Initiation:** Begin a `try/except` block to catch API timeouts or safety filters.
    2.  **Await Phase 1 (Vision):** Wrap the call to `_generate_description` in `asyncio.wait_for` with a hard **2.0-second timeout**. 
    3.  **State Update:** If successful, set `previous_context` to the new text and reset `consecutive_failures` to `0`.
    4.  **Await Phase 2 (TTS):** Wrap the call to `_generate_tts` in `asyncio.wait_for` with a hard **1.5-second timeout**.
    5.  **Execute Phase 3 (Sync):** Pass the audio to `_sync_audio_duration`.
    6.  **Return:** Return a tuple or dictionary containing the fitted audio bytes and the generated text string.
* **Exception Handling:**
    1.  **Catch Exceptions:** Catch `asyncio.TimeoutError` or Google API exceptions.
    2.  **State Management:** Increment `consecutive_failures`. If `consecutive_failures >= 3`, set `previous_context = ""` so the AI doesn't get stuck in the past when the connection recovers.
    3.  **Fallback:** Call `_generate_silent_audio()` and return the silent audio bytes alongside an empty text string.
