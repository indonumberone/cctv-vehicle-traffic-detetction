import cv2
import subprocess
import threading
import queue
import os
import time
from pathlib import Path


class HLSStreamer:
    def __init__(self, output_dir='stream', segment_time=2, playlist_size=5, 
                 width=1280, height=720, fps=25, bitrate='2000k'):

        self.output_dir = output_dir
        self.segment_time = segment_time
        self.playlist_size = playlist_size
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        
        # Increase queue size untuk buffer lebih besar
        self.frame_queue = queue.Queue(maxsize=60)  # Increased from 30
        self.stop_event = threading.Event()
        self.ffmpeg_process = None
        self.stderr_file = None
        self.streaming_thread = None
        self.is_running = False
        
        # Statistics
        self.frames_written = 0
        self.frames_dropped = 0
        self.last_log_time = time.time()
        
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
    def start(self):
        if self.is_running:
            print("⚠️  HLS streaming already running")
            return False
            
        # Clean old segments
        self._clean_output_dir()
        
        # FFmpeg command for HLS streaming
        playlist_file = os.path.join(self.output_dir, 'playlist.m3u8')
        segment_pattern = os.path.join(self.output_dir, 'segment_%03d.ts')
        
        ffmpeg_cmd = [
        'ffmpeg',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{self.width}x{self.height}',
        '-r', 25,
        '-i', '-',

        # Video options
        '-c:v', 'libx264',
        '-b:v', '500k',
        '-preset', 'veryfast',
        '-tune', 'zerolatency',
        '-g', str(self.fps * 2),  # Keyframe interval
        '-sc_threshold', '0',
        '-an',

        # HLS options
        '-f', 'hls',
        '-hls_time', str(self.segment_time),
        '-hls_list_size', str(self.playlist_size),
        '-hls_flags', 'delete_segments+append_list',
        '-hls_segment_filename', segment_pattern,

        # Force overwrite
        '-y',
        playlist_file
        
    ]
        

        
        try:
            # Untuk debugging, simpan stderr ke file
            stderr_log = os.path.join(self.output_dir, 'ffmpeg_error.log')
            self.stderr_file = open(stderr_log, 'w')
            
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self.stderr_file,  # Log errors untuk debugging
                bufsize=10**8
            )
            
            self.is_running = True
            
            # Start streaming thread
            self.streaming_thread = threading.Thread(
                target=self._streaming_worker, 
                daemon=True
            )
            self.streaming_thread.start()
            
            print(f"✅ HLS streaming started: {playlist_file}")
            print(f"   Resolution: {self.width}x{self.height} @ {self.fps}fps")
            print(f"   Bitrate: {self.bitrate}")
            return True
            
        except FileNotFoundError:
            print("❌ FFmpeg not found. Please install FFmpeg:")
            print("   Ubuntu/Debian: sudo apt-get install ffmpeg")
            return False
        except Exception as e:
            print(f"❌ Failed to start HLS streaming: {e}")
            return False
    
    def _streaming_worker(self):
        """Worker thread to write frames to FFmpeg - OPTIMIZED"""
        local_frames_written = 0
        
        while not self.stop_event.is_set():
            try:
                # Get frame dengan timeout
                frame = self.frame_queue.get(timeout=1.0)
                
                if self.ffmpeg_process and self.ffmpeg_process.stdin:
                    try:
                        # Resize frame if needed
                        if frame.shape[1] != self.width or frame.shape[0] != self.height:
                            frame = cv2.resize(frame, (self.width, self.height), 
                                             interpolation=cv2.INTER_LINEAR)  # Faster interpolation
                        
                        # Write frame to FFmpeg stdin
                        self.ffmpeg_process.stdin.write(frame.tobytes())
                        local_frames_written += 1
                        self.frames_written = local_frames_written
                        
                        # Log progress setiap 250 frames (reduced logging)
                        current_time = time.time()
                        if current_time - self.last_log_time >= 10.0:  # Every 10 seconds
                            queue_size = self.frame_queue.qsize()
                            print(f"📹 HLS: {local_frames_written} frames | "
                                  f"Queue: {queue_size}/60 | "
                                  f"Dropped: {self.frames_dropped}")
                            self.last_log_time = current_time
                        
                    except BrokenPipeError:
                        print("HLS FFmpeg pipe broken")
                        break
                    except Exception as e:
                        print(f"HLS Error writing to FFmpeg: {e}")
                        break
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"HLS Streaming worker error: {e}")
                break
        
    
    def write_frame(self, frame):
        """
        Add frame to streaming queue - NON-BLOCKING & OPTIMIZED
        
        Args:
            frame: OpenCV frame (numpy array)
        
        Returns:
            bool: True if frame was queued, False if dropped
        """
        if not self.is_running:
            return False
        
        try:
            # NON-BLOCKING put - drop frame jika queue penuh
            # IMPORTANT: Tidak pakai .copy() di sini untuk speed!
            # Frame akan di-copy di dalam thread worker jika perlu resize
            self.frame_queue.put_nowait(frame)
            return True
        except queue.Full:
            # Skip frame jika queue penuh
            self.frames_dropped += 1
            return False
    
    def _clean_output_dir(self):
        """Clean old segments and playlist"""
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                if file.endswith('.ts') or file.endswith('.m3u8'):
                    try:
                        os.remove(os.path.join(self.output_dir, file))
                    except Exception as e:
                        pass  # Silent fail untuk cleanup
    
    def stop(self):
        """Stop HLS streaming"""
        if not self.is_running:
            return
        
        self.stop_event.set()
        self.is_running = False
        
        # Close FFmpeg stdin
        if self.ffmpeg_process and self.ffmpeg_process.stdin:
            try:
                self.ffmpeg_process.stdin.close()
            except:
                pass
        
        # Wait for thread
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=3)
        
        # Terminate FFmpeg
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ffmpeg_process.kill()
            except:
                pass
        
        # Close stderr file
        if self.stderr_file:
            try:
                self.stderr_file.close()
            except:
                pass
        
        print(f"✅ HLS stopped. Total frames: {self.frames_written}, Dropped: {self.frames_dropped}")
    
    def get_stats(self):
        """Get streaming statistics"""
        return {
            'is_running': self.is_running,
            'queue_size': self.frame_queue.qsize(),
            'frames_written': self.frames_written,
            'frames_dropped': self.frames_dropped,
            'output_dir': self.output_dir,
            'resolution': f"{self.width}x{self.height}",
            'fps': self.fps,
            'bitrate': self.bitrate,
            'segment_time': self.segment_time,
            'playlist_size': self.playlist_size
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()