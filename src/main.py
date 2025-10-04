from .lib import FPSMeter, FrameProcessor, LineCrossingCounter, RTSPReconnector
import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
import supervision as sv
import time
import threading
import os
import queue
from dotenv import load_dotenv
load_dotenv()

rtsp_url = os.getenv("RTSP_URL")
class Main:
    def __init__(self, model_path, video_output, video_input, global_cleanup_timeout=3600.0, target_fps=25,retry=5):
        self.model_path = model_path
        self.video_output = video_output
        self.video_input = video_input
        self.global_cleanup_timeout = global_cleanup_timeout
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.retry  = retry
        
        self.LINES = [
            ((312, 359), (150, 448)), 
            ((132, 448), (949, 463)),
            ((719, 347), (852, 396))
        ]
        self.CLASSES_TO_TRACK = [0, 1, 2]
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.reconnect_event = threading.Event()
        self.last_display_time = None
        
    def read_frames(self, rtsp_conn):
        """Thread untuk membaca frame dengan auto-reconnect"""
        consecutive_failures = 0
        max_failures = 10 
        last_success_time = time.time()
        
        while not self.stop_event.is_set():
            ret, frame = rtsp_conn.read()
            
            if not ret:
                consecutive_failures += 1
                
                if time.time() - last_success_time > 25:
                    print(f"Stream timeout detected ({time.time() - last_success_time:.1f}s)")
                    consecutive_failures = max_failures  # Force reconnect
                
                if consecutive_failures >= max_failures:
                    print(f"Stream disconnected, attempting reconnect...")
                    self.reconnect_event.set()
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            pass
                    
                    if rtsp_conn.reconnect():
                        consecutive_failures = 0
                        last_success_time = time.time()
                        self.reconnect_event.clear()
                        print("✅ Reconnected successfully!")
                    else:
                        print("❌ Failed to reconnect, stopping...")
                        self.stop_event.set()
                        break
                else:
                    time.sleep(0.1)
                continue
            
            # Success
            consecutive_failures = 0
            last_success_time = time.time()
            timestamp = time.time()
            
            # Skip frame jika queue penuh
            try:
                self.frame_queue.put_nowait((frame, timestamp))
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait((frame, timestamp))
                except:
                    pass
                
    def process_frames(self, processor):
        """Thread untuk memproses frame"""
        while not self.stop_event.is_set():
            # Pause processing saat reconnecting
            if self.reconnect_event.is_set():
                time.sleep(0.5)
                continue
                
            try:
                frame, timestamp = self.frame_queue.get(timeout=0.5)
                start_proc = time.time()
                result_frame = processor.process(frame)
                end_proc = time.time()
                
                try:
                    self.result_queue.put_nowait((result_frame, timestamp, start_proc, end_proc))
                except queue.Full:
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait((result_frame, timestamp, start_proc, end_proc))
                    except:
                        pass
                    
            except queue.Empty:
                continue

    def detect(self):
        model = YOLO(self.model_path)
        
        rtsp_conn = RTSPReconnector(self.video_input, max_retry=self.retry, retry_delay=3)
        
        if not rtsp_conn.connect():
            print("Failed to connect to RTSP stream")
            return
        
        counter = LineCrossingCounter(self.LINES, global_cleanup_timeout=self.global_cleanup_timeout)
        processor = FrameProcessor(model, counter, model.names, self.CLASSES_TO_TRACK)
        fps_meter = FPSMeter(buffer_size=60)
        
        width = int(rtsp_conn.get_property(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(rtsp_conn.get_property(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(self.video_output, cv2.VideoWriter_fourcc(*'mp4v'), 
                              self.target_fps, (width, height))

        # Start threads
        read_thread = threading.Thread(target=self.read_frames, args=(rtsp_conn,), daemon=True)
        process_thread = threading.Thread(target=self.process_frames, args=(processor,), daemon=True)
        read_thread.start()
        process_thread.start()

  
        self.last_display_time = time.time()
        frame_count = 0
        last_frame_time = time.time()
        no_frame_warning_shown = False

        while not self.stop_event.is_set():
            try:
                frame, capture_time, start_proc, end_proc = self.result_queue.get(timeout=1.0)
                no_frame_warning_shown = False
            except queue.Empty:
                if self.reconnect_event.is_set():
                    if not no_frame_warning_shown:
                        print("Waiting for reconnection...")
                        no_frame_warning_shown = True
                    continue
                if time.time() - last_frame_time > 5.0:
                    if not no_frame_warning_shown:
                        print("No frames received for 5 seconds")
                        no_frame_warning_shown = True
                continue

            last_frame_time = time.time()
            current_time = time.time()
            
            # Frame pacing
            elapsed = current_time - self.last_display_time
            if elapsed < self.frame_time:
                sleep_time = self.frame_time - elapsed
                time.sleep(sleep_time)
                current_time = time.time()
            
            self.last_display_time = current_time

            # Draw lines
            for i, (start, end) in enumerate(self.LINES):
                cv2.line(frame, start, end, (0, 0, 255), 2)

            fps_meter.update_processing_time(start_proc, end_proc)
            fps_meter.update_frame_interval(capture_time)

            avg_proc_fps = fps_meter.get_avg_processing_fps()

            # Draw counts
            y_pos = 85
            for class_name, count in counter.get_counts().items():
                cv2.putText(frame, f"{class_name}: {count}", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_pos += 30

            # Draw FPS info
            cv2.putText(frame, f"Proc: {avg_proc_fps:.1f} fps", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Display: {self.target_fps} fps", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            

            cv2.imshow("Tracking", frame)
            out.write(frame)
            
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                print("ESC pressed, stopping...")
                self.stop_event.set()
                break

        self.stop_event.set()
        read_thread.join(timeout=2)
        process_thread.join(timeout=2)
        rtsp_conn.release()
        out.release()
        cv2.destroyAllWindows()


def main():
    """Entry point for the application"""
    app = Main(
        'model/best.pt', 
        'video/output/result.mp4', 
        rtsp_url, 
        global_cleanup_timeout=3600.0,
        target_fps=25,
        retry=10
    )
    app.detect()


if __name__ == "__main__":
    main()