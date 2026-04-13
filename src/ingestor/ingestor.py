import av
import queue
import threading
import io
import time
from ingestor.models import MediaChunk


class ChunkingIngestor:
    """
    Component A: The Chunking Ingestor (Producer).
    Spawns a pure python background thread that decodes a video file natively into memory
    without any filesystem IO latency using PyAV streams. Packages frames and natively routes
    them into a sequential processing queue block by block.
    """

    def __init__(
        self,
        video_path: str,
        chunk_duration: float = 5.0,
        simulate_realtime: bool = True,
    ):
        self.video_path = video_path
        self.chunk_duration = chunk_duration
        self.simulate_realtime = simulate_realtime
        self.processing_queue = queue.Queue(maxsize=10)
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        """Spins up the isolated AV producer thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._ingestion_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Sends the kill signal to gracefully halt production."""
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def _ingestion_loop(self):
        try:
            container = av.open(self.video_path)

            v_stream = next((s for s in container.streams if s.type == "video"), None)
            a_stream = next((s for s in container.streams if s.type == "audio"), None)

            if not v_stream:
                print("Error: Ingestor could not find a video stream.")
                self.processing_queue.put(None)
                return

            streams = [v_stream]
            audio_rate = 48000
            audio_channels = 2
            resampler = None

            if a_stream:
                streams.append(a_stream)
                audio_rate = a_stream.rate if a_stream.rate else 48000
                audio_channels = 2
                resampler = av.AudioResampler(
                    format="flt", layout="stereo", rate=audio_rate
                )

            start_wall_time = time.time()
            current_chunk_start = 0.0
            accumulated_video_frames = []
            accumulated_audio_frames = []
            accumulated_jpegs = []
            last_jpeg_pts_time = -1.0

            for packet in container.demux(streams):
                if self._stop_event.is_set():
                    break

                if packet.dts is None:
                    continue

                for frame in packet.decode():
                    if isinstance(frame, av.video.frame.VideoFrame):
                        f_time = float(frame.pts * v_stream.time_base)

                        if f_time >= current_chunk_start + self.chunk_duration:
                            chunk = MediaChunk(
                                start_time=round(current_chunk_start, 3),
                                end_time=round(
                                    current_chunk_start + self.chunk_duration, 3
                                ),
                                raw_video_frames=accumulated_video_frames,
                                compressed_frames=accumulated_jpegs,
                                raw_audio_bytes=b"".join(accumulated_audio_frames)
                                if accumulated_audio_frames
                                else None,
                                audio_sample_rate=audio_rate,
                                audio_channels=audio_channels,
                            )

                            if self.simulate_realtime:
                                expected_yield_time = (
                                    start_wall_time
                                    + current_chunk_start
                                    + self.chunk_duration
                                )
                                now = time.time()
                                if expected_yield_time > now:
                                    time.sleep(expected_yield_time - now)

                            self.processing_queue.put(chunk)

                            current_chunk_start += self.chunk_duration
                            accumulated_video_frames = []
                            accumulated_jpegs = []
                            accumulated_audio_frames = []

                        ndarray = frame.to_ndarray(format="rgb24")
                        accumulated_video_frames.append(ndarray)

                        img = frame.to_image()
                        mem_file = io.BytesIO()
                        img.save(mem_file, format="JPEG", quality=85)
                        accumulated_jpegs.append(mem_file.getvalue())

                    elif isinstance(frame, av.audio.frame.AudioFrame):
                        if resampler:
                            resampled_frames = resampler.resample(frame)
                            for r_frame in resampled_frames:
                                accumulated_audio_frames.append(
                                    r_frame.to_ndarray().tobytes()
                                )
                        else:
                            accumulated_audio_frames.append(
                                frame.to_ndarray().tobytes()
                            )

            if accumulated_video_frames or accumulated_audio_frames:
                chunk = MediaChunk(
                    start_time=round(current_chunk_start, 3),
                    end_time=round(current_chunk_start + self.chunk_duration, 3),
                    raw_video_frames=accumulated_video_frames,
                    compressed_frames=accumulated_jpegs,
                    raw_audio_bytes=b"".join(accumulated_audio_frames)
                    if accumulated_audio_frames
                    else None,
                    audio_sample_rate=audio_rate,
                    audio_channels=audio_channels,
                )

                if self.simulate_realtime:
                    expected_yield_time = start_wall_time + (
                        f_time if "f_time" in locals() else current_chunk_start
                    )
                    now = time.time()
                    if expected_yield_time > now:
                        time.sleep(expected_yield_time - now)

                self.processing_queue.put(chunk)

        except Exception as e:
            print(f"Ingestor pipeline crashed cleanly: {e}")
        finally:
            self.processing_queue.put(None)
