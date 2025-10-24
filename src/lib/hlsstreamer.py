import cv2
import subprocess
import threading
import queue
import os
import time
import json
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
        self.current_fps = 0
        self.fps_update_time = time.time()
        self.fps_frame_count = 0
        
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
    def start(self):
        if self.is_running:
            print("HLS streaming already running")
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
        '-r', str(self.fps),
        '-i', '-',

        # Video options
        '-c:v', 'libx264',
        '-b:v', self.bitrate,
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
            stderr_log = os.path.join(self.output_dir, 'ffmpeg_error.log')
            self.stderr_file = open(stderr_log, 'w', buffering=1) 
            
            print(f"tarting FFmpeg with command:")
            print(f"   {' '.join(ffmpeg_cmd)}")
            
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, 
                bufsize=10**8
            )
            
            time.sleep(0.5)
            poll_result = self.ffmpeg_process.poll()
            
            if poll_result is not None:
                # FFmpeg sudah mati
                stderr_output = self.ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                self.stderr_file.write(stderr_output)
                self.stderr_file.flush()
                print(f"FFmpeg failed to start (exit code: {poll_result})")
                print(f"   Check {stderr_log} for details")
                if stderr_output:
                    print(f"   Error: {stderr_output[:200]}")
                return False
            
            self.is_running = True
            
            # Start stderr monitoring thread
            stderr_thread = threading.Thread(
                target=self._monitor_stderr,
                daemon=True
            )
            stderr_thread.start()
            
            # Start streaming thread
            self.streaming_thread = threading.Thread(
                target=self._streaming_worker, 
                daemon=True
            )
            self.streaming_thread.start()
            
            print(f"✅ HLS streaming started: {playlist_file}")
            print(f"   Resolution: {self.width}x{self.height} @ {self.fps}fps")
            print(f"   Bitrate: {self.bitrate}")
            print(f"   Segment time: {self.segment_time}s")
            print(f"   FFmpeg PID: {self.ffmpeg_process.pid}")
            return True
            
        except FileNotFoundError:
            print("❌ FFmpeg not found. Please install FFmpeg:")
            print("   Ubuntu/Debian: sudo apt-get install ffmpeg")
            return False
        except Exception as e:
            print(f"❌ Failed to start HLS streaming: {e}")
            return False
    
    def _monitor_stderr(self):
        """Monitor FFmpeg stderr output"""
        if not self.ffmpeg_process or not self.ffmpeg_process.stderr:
            return
        
        try:
            for line in iter(self.ffmpeg_process.stderr.readline, b''):
                if not line:
                    break
                decoded_line = line.decode('utf-8', errors='ignore').strip()
                if self.stderr_file:
                    self.stderr_file.write(decoded_line + '\n')
                    self.stderr_file.flush()
                
                # Print critical errors
                if 'error' in decoded_line.lower() or 'failed' in decoded_line.lower():
                    print(f"⚠️  FFmpeg: {decoded_line}")
        except Exception as e:
            print(f"Error monitoring FFmpeg stderr: {e}")
    
    def _streaming_worker(self):
        """Worker thread to write frames to FFmpeg - OPTIMIZED"""
        local_frames_written = 0
        
        print("🎬 Streaming worker started, waiting for frames...")
        
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
                        self.fps_frame_count += 1
                        
                        # Log first frame
                        if local_frames_written == 1:
                            print(f"✅ First frame written to FFmpeg!")
                        
                        # Calculate FPS and update stats file
                        current_time = time.time()
                        time_diff = current_time - self.fps_update_time
                        if time_diff >= 1.0:  # Update FPS every second
                            self.current_fps = self.fps_frame_count / time_diff
                            self.fps_frame_count = 0
                            self.fps_update_time = current_time
                            self._save_stats_to_file()
                        
                        # Log progress setiap 10 detik
                        if current_time - self.last_log_time >= 10.0:
                            queue_size = self.frame_queue.qsize()
                            print(f"📹 HLS: {local_frames_written} frames | "
                                  f"FPS: {self.current_fps:.1f} | "
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
            # Log only first time
            if not hasattr(self, '_not_running_logged'):
                print("⚠️  HLS write_frame called but streaming not running!")
                self._not_running_logged = True
            return False
        
        try:
            self.frame_queue.put_nowait(frame)
            
            # Log first successful queue
            if self.frame_queue.qsize() == 1 and self.frames_written == 0:
                print(f"✅ First frame queued! Shape: {frame.shape}")
            
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
        
        # # Terminate FFmpeg
        # if self.ffmpeg_process:
        #     try:
        #         self.ffmpeg_process.terminate()
        #         self.ffmpeg_process.wait(timeout=5)
        #     except subprocess.TimeoutExpired:
        #         self.ffmpeg_process.kill()
        #     except:
        #         pass
        
        # # Close stderr file
        # if self.stderr_file:
        #     try:
        #         self.stderr_file.close()
        #     except:
        #         pass
        
        # print(f"✅ HLS stopped. Total frames: {self.frames_written}, Dropped: {self.frames_dropped}")
    
    def get_stats(self):
        """Get streaming statistics"""
        return {
            'is_running': self.is_running,
            'queue_size': self.frame_queue.qsize(),
            'frames_written': self.frames_written,
            'frames_dropped': self.frames_dropped,
            'output_dir': self.output_dir,
            'resolution': f"{self.width}x{self.height}",
            'fps': round(self.current_fps, 1),
            'target_fps': self.fps,
            'bitrate': self.bitrate,
            'segment_time': self.segment_time,
            'playlist_size': self.playlist_size
        }
    
    def _save_stats_to_file(self):
        """Save current stats to JSON file for web API"""
        try:
            stats_file = os.path.join(self.output_dir, 'stream_stats.json')
            stats = self.get_stats()
            with open(stats_file, 'w') as f:
                json.dump(stats, f)
        except Exception as e:
            # Silent fail - tidak kritis
            pass
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()