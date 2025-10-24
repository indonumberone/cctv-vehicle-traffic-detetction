from .lib import FPSMeter, FrameProcessor, LineCrossingCounter, RTSPReconnector, InfluxDBLogger,HLSStreamer
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

# RTSP and InfluxDB environment variables
rtsp_url = os.getenv("RTSP_URL")
influxdb_url = os.getenv("INFLUXDB_URL", "http://localhost:8086")
influxdb_token = os.getenv("INFLUXDB_INIT_ADMIN_TOKEN")
influxdb_org = os.getenv("INFLUXDB_INIT_ORG")
influxdb_bucket = os.getenv("INFLUXDB_INIT_BUCKET", "vehicle_counting")
location_name = os.getenv("LOCATION_NAME", "camera1")

class Main:
    def __init__(self, model_path, video_output, video_input, global_cleanup_timeout=3600.0, 
                 target_fps=25, retry=5, use_influxdb=True, outputdir='./output'):
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
        self.use_influxdb = use_influxdb
        self.influxdb_logger = None
        self.hls = HLSStreamer(
            output_dir=outputdir,  
            width=1920,
            height=1080,
            fps=25,
            bitrate='5000k',
            segment_time=2,
        playlist_size=10
    )
        
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
        if self.hls.start():
            print("HLS streaming started")
        counter = LineCrossingCounter(self.LINES, global_cleanup_timeout=self.global_cleanup_timeout)
        
        # Initialize InfluxDB logger if needed
        if self.use_influxdb and influxdb_token:
            try:
                self.influxdb_logger = InfluxDBLogger(
                    url=influxdb_url,
                    token=influxdb_token,
                    org=influxdb_org,
                    bucket=influxdb_bucket,
                    location_name=location_name
                )
                
                # Set up callback for crossing events
                counter.set_crossing_callback(self.influxdb_logger.log_crossing_event)
                
                # Start periodic logging (every 60 seconds)
                self.influxdb_logger.start_periodic_logging(counter, interval=60)
                print("Connected to InfluxDB")
            except Exception as e:
                print(f"Failed to connect to InfluxDB: {e}")
                self.influxdb_logger = None
        
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
            self.hls.write_frame(frame)
            out.write(frame)
            
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                print("ESC pressed, stopping...")
                self.stop_event.set()
                self.hls.stop()
                break

            # Log counts to InfluxDB when they change
            if self.influxdb_logger:
                current_counts = counter.get_counts()
                self.influxdb_logger.log_counts(current_counts)

        # Cleanup at the end
        self.stop_event.set()
        read_thread.join(timeout=2)
        process_thread.join(timeout=2)
        if self.influxdb_logger:
            self.influxdb_logger.close()
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
        retry=10,
        use_influxdb=True  # Set to False if you don't want to use InfluxDB
    )
    app.detect()


if __name__ == "__main__":
    main()