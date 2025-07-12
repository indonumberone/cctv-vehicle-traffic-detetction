import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
import time
import threading
import queue
import numpy as np

class FPSMeter:
    def __init__(self, buffer_size=30):
        self.processing_times = deque(maxlen=buffer_size)
        self.frame_intervals = deque(maxlen=buffer_size)
        self.last_capture_time = None

    def update_processing_time(self, start, end):
        elapsed = end - start
        if elapsed > 0:
            self.processing_times.append(1.0 / elapsed)

    def update_frame_interval(self, current_time):
        if self.last_capture_time is not None:
            interval = current_time - self.last_capture_time
            if interval > 0:
                self.frame_intervals.append(1.0 / interval)
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
            self.track_history[track_id] = deque(maxlen=10)
        self.track_history[track_id].append(center)

        if len(self.track_history[track_id]) >= 2:
            prev, curr = self.track_history[track_id][-2], self.track_history[track_id][-1]
            for i, (start, end) in enumerate(self.lines):
                if self._is_crossing_line(prev, curr, start, end):
                    if track_id not in self.crossed_ids_per_line[i]:
                        self.crossed_ids_per_line[i].add(track_id)
                        self.class_counts[class_name] += 1

    def _is_crossing_line(self, p1, p2, a, b):
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return ccw(p1, a, b) != ccw(p2, a, b) and ccw(p1, p2, a) != ccw(p1, p2, b)

    def get_counts(self):
        return self.class_counts

class FrameProcessor:
    def __init__(self, model, counter, class_names, classes_to_track):
        self.model = model
        self.counter = counter
        self.class_names = class_names
        self.classes_to_track = classes_to_track

    def process(self, frame):
        results = self.model.track(frame, persist=True, classes=self.classes_to_track, device='cpu')
        if results[0].boxes.id is None:
            return frame

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()
        cls_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, ids, cls_ids):
            x1, y1, x2, y2 = map(int, box)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            class_name = self.class_names[class_id]
            self.counter.update(track_id, class_name, center)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, center, 4, (255, 0, 0), -1)
            cv2.putText(frame, f"{class_name} {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

class Main:
    def __init__(self, model_path, video_output, video_input):
        self.model_path = model_path
        self.video_output = video_output
        self.video_input = video_input
        self.lines = [
            ((312, 359), (150, 418)),
            ((132, 448), (492, 463)),
            ((521, 462), (949, 448)),
            ((719, 347), (852, 396))
        ]
        self.classes_to_track = [1, 2, 3, 4, 5, 7]
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()

    def read_frames(self, cap):
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                self.stop_event.set()
                break
            timestamp = time.time()
            try:
                self.frame_queue.put((frame, timestamp), timeout=0.05)
            except queue.Full:
                continue

    def process_frames(self, processor):
        while not self.stop_event.is_set():
            try:
                frame, timestamp = self.frame_queue.get(timeout=0.05)
                start_proc = time.time()
                result_frame = processor.process(frame)
                end_proc = time.time()
                self.result_queue.put((result_frame, timestamp, start_proc, end_proc), timeout=0.05)
            except queue.Empty:
                continue
            except queue.Full:
                continue

    def detect(self):
        cap = cv2.VideoCapture(self.video_input)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        model = YOLO(self.model_path)
        counter = LineCrossingCounter(self.lines)
        processor = FrameProcessor(model, counter, model.names, self.classes_to_track)
        fps_meter = FPSMeter()
        out = cv2.VideoWriter(self.video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        thread1 = threading.Thread(target=self.read_frames, args=(cap,), daemon=True)
        thread2 = threading.Thread(target=self.process_frames, args=(processor,), daemon=True)
        thread1.start()
        thread2.start()

        while not self.stop_event.is_set() or not self.result_queue.empty():
            try:
                frame, cap_time, start_proc, end_proc = self.result_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            for (start, end) in self.lines:
                cv2.line(frame, start, end, (0, 0, 255), 2)

            fps_meter.update_processing_time(start_proc, end_proc)
            fps_meter.update_frame_interval(cap_time)

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

            if cv2.waitKey(1) & 0xFF == 27:
                self.stop_event.set()
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        thread1.join(timeout=1)
        thread2.join(timeout=1)
        print("Final Counts:", dict(counter.get_counts()))

if __name__ == "__main__":
    app = Main('yolov8m_openvino_model', 'output_yolone.mp4', 'out_pens0.mp4')
    app.detect()
