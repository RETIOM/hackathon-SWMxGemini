import av
import queue
import threading
import io
import time
from models import MediaChunk

class ChunkingIngestor:
    """
    Component A: The Chunking Ingestor (Producer).
    Spawns a pure python background thread that decodes a video file natively into memory
    without any filesystem IO latency using PyAV streams. Packages frames and natively routes
    them into a sequential processing queue block by block.
    """
    def __init__(self, video_path: str, chunk_duration: float = 5.0, simulate_realtime: bool = True):
        self.video_path = video_path
        self.chunk_duration = chunk_duration
        self.simulate_realtime = simulate_realtime
        self.processing_queue = queue.Queue(maxsize=10) # Avoid unbound memory explosions
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
            
            # Identify valid streams
            v_stream = next((s for s in container.streams if s.type == "video"), None)
            a_stream = next((s for s in container.streams if s.type == "audio"), None)
            
            if not v_stream:
                print("Error: Ingestor could not find a video stream.")
                self.processing_queue.put(None)
                return

            streams = [v_stream]
            if a_stream:
                streams.append(a_stream)

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
                    # Evaluate time via the video tracker
                    if isinstance(frame, av.video.frame.VideoFrame):
                        # Calculate exact presentation timestamp mathematically
                        f_time = float(frame.pts * v_stream.time_base)
                        
                        # Has the window crossed out of the 5-second block boundary?
                        if f_time >= current_chunk_start + self.chunk_duration:
                            # Package the Raw Memory Chunk Sequence dynamically!
                            chunk = MediaChunk(
                                start_time=round(current_chunk_start, 3),
                                end_time=round(current_chunk_start + self.chunk_duration, 3),
                                raw_video_frames=accumulated_video_frames,
                                compressed_frames=accumulated_jpegs,
                                raw_audio_bytes=b"".join(accumulated_audio_frames) if accumulated_audio_frames else None
                            )
                            
                            # Simulate a live In-Real-Time (IRT) WebRTC/RTSP stream feed
                            if self.simulate_realtime:
                                expected_yield_time = start_wall_time + current_chunk_start + self.chunk_duration
                                now = time.time()
                                if expected_yield_time > now:
                                    time.sleep(expected_yield_time - now)

                            # Push it directly onto Queue_A (Blocks if consumer hasn't pulled)
                            self.processing_queue.put(chunk)
                            
                            # Shift the boundary window dynamically
                            current_chunk_start += self.chunk_duration
                            accumulated_video_frames = []
                            accumulated_jpegs = []
                            accumulated_audio_frames = []

                        # 1. Store uncompressed target frame into memory explicitly (per User Request)
                        ndarray = frame.to_ndarray(format='rgb24')
                        accumulated_video_frames.append(ndarray)

                        # 2. Capture all frames as explicitly requested
                        img = frame.to_image()
                        mem_file = io.BytesIO()
                        img.save(mem_file, format='JPEG', quality=85)
                        accumulated_jpegs.append(mem_file.getvalue())

                    elif isinstance(frame, av.audio.frame.AudioFrame):
                        # Rapidly cache audio payload slices as completely uncompressed raw bytes natively
                        accumulated_audio_frames.append(frame.to_ndarray().tobytes())

            # Yield whatever uncompleted trailing chunk remains at the very end of the file safely
            if accumulated_video_frames or accumulated_audio_frames:
                 chunk = MediaChunk(
                     start_time=round(current_chunk_start, 3),
                     end_time=round(current_chunk_start + self.chunk_duration, 3),
                     raw_video_frames=accumulated_video_frames,
                     compressed_frames=accumulated_jpegs,
                     raw_audio_bytes=b"".join(accumulated_audio_frames) if accumulated_audio_frames else None
                 )
                 
                 if self.simulate_realtime:
                     expected_yield_time = start_wall_time + (f_time if 'f_time' in locals() else current_chunk_start)
                     now = time.time()
                     if expected_yield_time > now:
                         time.sleep(expected_yield_time - now)

                 self.processing_queue.put(chunk)

        except Exception as e:
            print(f"Ingestor pipeline crashed cleanly: {e}")
        finally:
            # Drop a None termination sentinel to accurately cascade teardown downstream
            self.processing_queue.put(None)
