import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
import supervision as sv
import time
import threading
import queue
from datetime import datetime


class FPSMeter:
    def __init__(self, buffer_size=60):
        self.processing_times = deque(maxlen=buffer_size)
        self.frame_intervals = deque(maxlen=buffer_size)
        self.last_capture_time = None

    def update_processing_time(self, start_time, end_time):
        elapsed = end_time - start_time
        if elapsed > 0:
            fps = 1.0 / elapsed
            self.processing_times.append(fps)

    def update_frame_interval(self, current_time):
        if self.last_capture_time is not None:
            interval = current_time - self.last_capture_time
            if interval > 0:
                real_fps = 1.0 / interval
                self.frame_intervals.append(real_fps)
        self.last_capture_time = current_time

    def get_avg_processing_fps(self):
        return sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0.0

    def get_avg_real_fps(self):
        return sum(self.frame_intervals) / len(self.frame_intervals) if self.frame_intervals else 0.0


class LineCrossingCounter:
    def __init__(self, lines, global_cleanup_timeout=3600.0):
        self.lines = lines
        self.track_history = {}
        self.class_counts = defaultdict(int)
        self.crossed_ids_per_line = [set() for _ in lines]  
        self.global_crossed_ids = set() 
        self.last_reset_day = datetime.now().day
        self.global_cleanup_timeout = global_cleanup_timeout  
        self.global_crossed_timestamps = {}  

    def update(self, track_id, class_name, center):
        current_time = time.time()
        current_day = datetime.now().day
        if current_day != self.last_reset_day:
            print(f"\n{'='*50}")
            print(f"RESET HARIAN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Count sebelumnya: {dict(self.class_counts)}")
            
            self.class_counts = defaultdict(int)
            self.crossed_ids_per_line = [set() for _ in self.lines]
            self.global_crossed_ids = set()
            self.global_crossed_timestamps = {}
            self.track_history = {}
            self.last_reset_day = current_day
            
            print(f"Count direset ke 0")
            print(f"{'='*50}\n")
        
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=30)
        self.track_history[track_id].append(center)

        self._cleanup_global_ids(current_time)

        if len(self.track_history[track_id]) >= 2:
            prev, curr = self.track_history[track_id][-2], self.track_history[track_id][-1]
            for i, (start, end) in enumerate(self.lines):
                if self._is_crossing_line(prev, curr, start, end):
                    if track_id not in self.crossed_ids_per_line[i]:
                        self.crossed_ids_per_line[i].add(track_id)
                        print(f"{class_name} ID {track_id} crossed line {i+1}")
                        
                        if track_id not in self.global_crossed_ids:
                            self.global_crossed_ids.add(track_id)
                            self.global_crossed_timestamps[track_id] = current_time
                            self.class_counts[class_name] += 1
                            print(f"COUNTED: {class_name} ID {track_id}")
                        else:
                            print(f"Already counted: {class_name} ID {track_id}")

    def _cleanup_global_ids(self, current_time):
        old_global_ids = []
        
        for track_id, crossed_time in self.global_crossed_timestamps.items():
            if current_time - crossed_time > self.global_cleanup_timeout:
                old_global_ids.append(track_id)
        
        for track_id in old_global_ids:
            print(f"CLEANUP GLOBAL: Removing ID {track_id} (counted {self.global_cleanup_timeout/3600:.1f}h ago)")
            self.global_crossed_ids.discard(track_id)
            if track_id in self.global_crossed_timestamps:
                del self.global_crossed_timestamps[track_id]

    def _is_crossing_line(self, p1, p2, line_start, line_end):
        def ccw(a, b, c):
            return (b[1]-a[1])*(c[0]-a[0]) - (b[0]-a[0])*(c[1]-a[1])
        return (ccw(p1, line_start, line_end) * ccw(p2, line_start, line_end) < 0) and \
               (ccw(p1, p2, line_start) * ccw(p1, p2, line_end) < 0)

    def get_counts(self):
        return self.class_counts
    
    def get_line_crossings(self):
        return {f"Line {i+1}": len(crossed_ids) 
                for i, crossed_ids in enumerate(self.crossed_ids_per_line)}


class FrameProcessor:
    def __init__(self, model, counter, class_names, classes_to_track):
        self.model = model
        self.counter = counter
        self.class_names = class_names
        self.classes_to_track = classes_to_track
        self.tracker = sv.ByteTrack()

    def process(self, frame):
        results = self.model(frame, classes=self.classes_to_track, conf=0.4, iou=0.2)
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return frame
        
        detections = sv.Detections.from_ultralytics(results[0])
        detections = self.tracker.update_with_detections(detections)
        
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            track_id = detections.tracker_id[i] if detections.tracker_id is not None else None
            class_id = int(detections.class_id[i])
            class_name = self.class_names[class_id]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            if track_id is not None:
                self.counter.update(track_id, class_name, center)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, center, 4, (255, 0, 0), -1)
                cv2.putText(frame, f"{class_name} {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame


class RTSPReconnector:
    """Handles RTSP stream with auto-reconnect"""
    def __init__(self, rtsp_url, max_retry=5, retry_delay=3):
        self.rtsp_url = rtsp_url
        self.max_retry = max_retry
        self.retry_delay = retry_delay
        self.cap = None
        self.reconnect_count = 0
        self.lock = threading.Lock()
        
    def connect(self):
        """Open RTSP connection"""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self.cap.isOpened():
                print(f"✅ Connected to RTSP stream (attempt {self.reconnect_count + 1})")
                return True
            return False
    
    def reconnect(self):
        """Attempt to reconnect"""
        print(f"Reconnecting... (attempt {self.reconnect_count + 1}/{self.max_retry})")
        
        for attempt in range(self.max_retry):
            time.sleep(self.retry_delay)
            if self.connect():
                self.reconnect_count += 1
                return True
            print(f"reconnect failed ({attempt + 1}/{self.max_retry})")
        
        print(f"Max reconnection attempts reached")
        return False
    
    def read(self):
        """Read frame with auto-reconnect on failure"""
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                return False, None
            
            ret, frame = self.cap.read()
            return ret, frame
    
    def release(self):
        """Release the capture"""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
    
    def get_property(self, prop):
        """Get video property"""
        with self.lock:
            if self.cap is not None:
                return self.cap.get(prop)
            return 0


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
            ((312, 359), (150, 418)), 
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
                        print("⏸Waiting for reconnection...")
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


if __name__ == "__main__":
    app = Main(
        'model/best.pt', 
        'video/output/result.mp4', 
        'rtsp://admin:tri@2024@41.216.190.93:554/Streaming/Channels/101', 
        global_cleanup_timeout=3600.0,
        target_fps=25,
        retry=10
    )
    app.detect()