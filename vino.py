import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
import time
import threading
import queue


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
    def __init__(self, lines):
        self.lines = lines
        self.track_history = {}
        self.class_counts = defaultdict(int)
        self.crossed_ids_per_line = [set() for _ in lines]

    def update(self, track_id, class_name, center):
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=30)
        self.track_history[track_id].append(center)

        if len(self.track_history[track_id]) >= 2:
            prev, curr = self.track_history[track_id][-2], self.track_history[track_id][-1]
            for i, (start, end) in enumerate(self.lines):
                if self._is_crossing_line(prev, curr, start, end):
                    if track_id not in self.crossed_ids_per_line[i]:
                        self.crossed_ids_per_line[i].add(track_id)
                        self.class_counts[class_name] += 1
                        print(f"{class_name} ID {track_id} crossed line {i+1}")

    def _is_crossing_line(self, p1, p2, line_start, line_end):
        def ccw(a, b, c):
            return (b[1]-a[1])*(c[0]-a[0]) - (b[0]-a[0])*(c[1]-a[1])
        return (ccw(p1, line_start, line_end) * ccw(p2, line_start, line_end) < 0) and \
               (ccw(p1, p2, line_start) * ccw(p1, p2, line_end) < 0)

    def get_counts(self):
        return self.class_counts

class FrameProcessor:
    def __init__(self, model, counter, class_names, classes_to_track):
        self.model = model
        self.counter = counter
        self.class_names = class_names
        self.classes_to_track = classes_to_track

    def process(self, frame):
        results = self.model.track(frame, persist=True, classes=self.classes_to_track, device='CPU')
        if results[0].boxes.id is None:
            return frame
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()
        cls_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, ids, cls_ids):
            x1, y1, x2, y2 = map(int, box)
            class_name = self.class_names[class_id]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            self.counter.update(track_id, class_name, center)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, center, 4, (255, 0, 0), -1)
            cv2.putText(frame, f"{class_name} {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

class Main:
    def __init__(self,model_path,video_output,video_input):
        self.model_path = model_path
        self.video_output = video_output
        self.video_input = video_input
        self.LINES = [
            ((312, 359), (150, 418)),
            ((132, 448), (492, 463)),
            ((521, 462), (949, 448)),
            ((719, 347), (852, 396))
        ]
        self.CLASSES_TO_TRACK = [1, 2, 3, 4, 5, 7],
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue(maxsize=30)
        self.stop_event = threading.Event()
        
    def read_frames(self, cap):
        while not self.stop_event.is_set():
            if self.frame_queue.qsize() < 30:
                ret, frame = cap.read()
                if not ret:
                    self.stop_event.set()
                    break
                timestamp = time.time()
                self.frame_queue.put((frame, timestamp))
            else:
                time.sleep(0.001)
                
    def process_frames(self, processor):
                while not self.stop_event.is_set():
                    if not self.frame_queue.empty():
                        try:
                            frame, timestamp = self.frame_queue.get(timeout=0.001)
                            start_proc = time.time()
                            result_frame = processor.process(frame)
                            end_proc = time.time()
                            self.result_queue.put((result_frame, timestamp, start_proc, end_proc))
                        except queue.Empty:
                            continue
                    else:
                        time.sleep(0.001)

    def detect(self):
        model = YOLO(self.model_path)
        cap = cv2.VideoCapture(self.video_input)
        counter = LineCrossingCounter(self.LINES)
        processor = FrameProcessor(model, counter, model.names, self.CLASSES_TO_TRACK)
        fps_meter = FPSMeter(buffer_size=60)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(self.video_output, cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))

        read_thread = threading.Thread(target=self.read_frames, args=(cap,))
        process_thread = threading.Thread(target=self.process_frames, args=(processor,))
        read_thread.start()
        process_thread.start()

        while not self.stop_event.is_set() or not self.result_queue.empty():
            if self.result_queue.empty():
                time.sleep(0.001)
                continue

            frame, capture_time, start_proc, end_proc = self.result_queue.get()

            for i, (start, end) in enumerate(self.LINES):
                cv2.line(frame, start, end, (0, 0, 255), 2)

            fps_meter.update_processing_time(start_proc, end_proc)
            fps_meter.update_frame_interval(capture_time)

            avg_proc_fps = fps_meter.get_avg_processing_fps()
            avg_real_fps = fps_meter.get_avg_real_fps()

            y_pos = 85
            for class_name, count in counter.get_counts().items():
                cv2.putText(frame, f"{class_name}: {count}", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_pos += 30

            cv2.putText(frame, f"Processing FPS: {avg_proc_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Real Video FPS: {avg_real_fps:.1f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Tracking", frame)
            out.write(frame)

            if cv2.waitKey(1) == 27:
                self.stop_event.set()
                break

        read_thread.join()
        process_thread.join()
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        
if __name__ == "__main__":
    app = Main('yolov8n_openvino_model', 'output_yolone.mp4', 'out_pens0.mp4')
    app.detect()


