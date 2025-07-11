import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
import time
import numpy as np

LINES = [
    ((312, 359), (150, 418)),
    ((132, 448), (492, 463)),
    ((521, 462), (949, 448)),
    ((719, 347), (852, 396))
]
OUTPUT_FILE = 'output_yolone.mp4'
MODEL_PATH = 'yolov8n_openvino_model'  
VIDEO_PATH = 'out_pens0.mp4'
CLASSES_TO_TRACK = [1, 2, 3, 4, 5, 7]  

model = YOLO(MODEL_PATH) 
cap = cv2.VideoCapture(VIDEO_PATH)
class_list = model.names
print("Classes in model:", class_list)

class_counts = defaultdict(int)
track_history = {}
track_object_ids_per_line = [set() for _ in LINES]

fps_buffer = deque(maxlen=30)
fps_output = 25
warmup_frames = 10

out = None
frame_count = 0
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        height, width = frame.shape[:2]
        out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), fps_output, (width, height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

def is_crossing_line(p1, p2, line_start, line_end):
    """Check if line segment p1-p2 crosses the reference line."""
    def ccw(a, b, c):
        return (b[1]-a[1])*(c[0]-a[0]) - (b[0]-a[0])*(c[1]-a[1])
    return (ccw(p1, line_start, line_end) * ccw(p2, line_start, line_end) < 0) and \
           (ccw(p1, p2, line_start) * ccw(p1, p2, line_end) < 0)

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    results =  model.track(frame, persist=True, classes=CLASSES_TO_TRACK,device='CPU')
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_list[class_id]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if track_id not in track_history:
                track_history[track_id] = deque(maxlen=30)
            track_history[track_id].append(center)

            if len(track_history[track_id]) >= 2:
                prev, curr = track_history[track_id][-2], track_history[track_id][-1]
                for i, (line_start, line_end) in enumerate(LINES):
                    if is_crossing_line(prev, curr, line_start, line_end):
                        if track_id not in track_object_ids_per_line[i]:
                            track_object_ids_per_line[i].add(track_id)
                            class_counts[class_name] += 1
                            print(f"{class_name} ID {track_id} crossed line {i+1}")

            # Visualization
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (255, 0, 0), -1)
            cv2.putText(frame, f"{class_name} {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for i, (start, end) in enumerate(LINES):
        cv2.line(frame, start, end, (0, 0, 255), 2)
        cv2.putText(frame, f"Line {i+1}", (start[0], start[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    current_fps = 1 / (time.time() - start_time + 1e-6)
    fps_buffer.append(current_fps)
    avg_fps = sum(fps_buffer) / len(fps_buffer)

    if frame_count > warmup_frames:
        new_fps = max(1, min(60, int(avg_fps)))
        if abs(new_fps - fps_output) > 1:
            fps_output = new_fps
            out.release()
            out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), fps_output, (width, height))

    y_pos = 30
    for class_name, count in class_counts.items():
        cv2.putText(frame, f"{class_name}: {count}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 25

    cv2.putText(frame, f"FPS: {avg_fps:.1f} (Output: {fps_output})", (10, y_pos + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('Tracking', frame)
    out.write(frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Final stats - Avg FPS: {avg_fps:.1f}, Counts: {dict(class_counts)}")