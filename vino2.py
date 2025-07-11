import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
import time
import numpy as np
import threading
import queue

# Configuration
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
BUFFER_SIZE = 30  
MAX_WAIT_TIME = 0.001 

def is_crossing_line(p1, p2, line_start, line_end):
    """Check if line segment p1-p2 crosses the reference line."""
    def ccw(a, b, c):
        return (b[1]-a[1])*(c[0]-a[0]) - (b[0]-a[0])*(c[1]-a[1])
    return (ccw(p1, line_start, line_end) * ccw(p2, line_start, line_end) < 0) and \
           (ccw(p1, p2, line_start) * ccw(p1, p2, line_end) < 0)

def read_frames(cap, frame_queue, stop_event):
    """Read frames from source and add timestamps"""
    frame_count = 0
    while not stop_event.is_set():
        if frame_queue.qsize() < BUFFER_SIZE:
            ret, frame = cap.read()
            if not ret:
                stop_event.set()
                break
            timestamp = time.time()
            frame_count += 1
            frame_queue.put((frame, timestamp, frame_count))
        else:
            time.sleep(MAX_WAIT_TIME)

def process_frames(model, frame_queue, result_queue, stop_event):
    """Process frames with model and track processing time"""
    while not stop_event.is_set():
        if not frame_queue.empty():
            try:
                frame, capture_time, frame_count = frame_queue.get(timeout=MAX_WAIT_TIME)
                inference_start = time.time()
                results = model.track(frame, persist=True, classes=CLASSES_TO_TRACK, device='CPU')
                inference_time = time.time() - inference_start
                result_queue.put((frame, results, capture_time, inference_time, frame_count))
            except queue.Empty:
                pass
        else:
            time.sleep(MAX_WAIT_TIME)

def main():
    model = YOLO(MODEL_PATH) 
    cap = cv2.VideoCapture(VIDEO_PATH)
    class_list = model.names
    print("Classes in model:", class_list)

    class_counts = defaultdict(int)
    track_history = {}
    track_object_ids_per_line = [set() for _ in LINES]
    
    fps_buffer = deque(maxlen=60) 
    real_fps_buffer = deque(maxlen=60)
    fps_output = 25
    last_frame_time = None
    start_time = time.time()
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video dimensions: {width}x{height}, Total frames: {total_frames}")
    
    out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), fps_output, (width, height))
    
    frame_queue = queue.Queue(maxsize=BUFFER_SIZE)
    result_queue = queue.Queue(maxsize=BUFFER_SIZE)
    stop_event = threading.Event()
    
    read_thread = threading.Thread(target=read_frames, args=(cap, frame_queue, stop_event), name="FrameReader")
    process_thread = threading.Thread(target=process_frames, args=(model, frame_queue, result_queue, stop_event), name="ModelInference")
    
    read_thread.daemon = True
    process_thread.daemon = True
    
    print("Starting threads...")
    read_thread.start()
    process_thread.start()
    
    time.sleep(0.5)
    
    try:
        frame_count = 0
        last_display_update = time.time()
        display_interval = 1.0  
        
        while not stop_event.is_set() or not result_queue.empty():
            loop_start = time.time()
            
            if result_queue.empty():
                if frame_queue.empty() and stop_event.is_set():
                    print("Processing complete, no more frames")
                    break
                time.sleep(MAX_WAIT_TIME)
                continue
                
            frame, results, capture_time, inference_time, original_frame_count = result_queue.get()
            frame_count += 1
            
            if last_frame_time:
                real_frame_interval = capture_time - last_frame_time
                if real_frame_interval > 0:
                    real_fps = 1.0 / real_frame_interval
                    real_fps_buffer.append(real_fps)
            
            last_frame_time = capture_time
            
            vis_start = time.time()
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
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, center, 5, (255, 0, 0), -1)
                    cv2.putText(frame, f"{class_name} {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            for i, (start, end) in enumerate(LINES):
                cv2.line(frame, start, end, (0, 0, 255), 2)
                cv2.putText(frame, f"Line {i+1}", (start[0], start[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            loop_time = time.time() - loop_start
            current_fps = 1.0 / (loop_time + 1e-6)
            fps_buffer.append(current_fps)
            
            avg_fps = sum(fps_buffer) / len(fps_buffer)
            real_avg_fps = sum(real_fps_buffer) / max(1, len(real_fps_buffer))
            
            vis_time = time.time() - vis_start
            total_process_time = vis_time + inference_time
            
            if time.time() - last_display_update > display_interval:
                print(f"Frame: {frame_count}/{total_frames} | "
                      f"Processing FPS: {avg_fps:.1f} | "
                      f"Video FPS: {real_avg_fps:.1f} | "
                      f"Queue sizes: {frame_queue.qsize()}/{result_queue.qsize()}")
                last_display_update = time.time()
            
            new_fps = max(10, min(60, int(avg_fps)))
            if abs(new_fps - fps_output) > 5: 
                print(f"Adjusting output FPS: {fps_output} -> {new_fps}")
                fps_output = new_fps
                out.release()
                out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), fps_output, (width, height))
            
            y_pos = 30
            for class_name, count in class_counts.items():
                cv2.putText(frame, f"{class_name}: {count}", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_pos += 25
            
            cv2.putText(frame, f"Processing FPS: {avg_fps:.1f}", (10, y_pos + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Video FPS: {real_avg_fps:.1f}", (10, y_pos + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Output FPS: {fps_output}", (10, y_pos + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, y_pos + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.putText(frame, f"Queues: {frame_queue.qsize()}/{result_queue.qsize()}", (10, y_pos + 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Tracking', frame)
            out.write(frame)
            
            if cv2.waitKey(1) == 27:  # ESC to exit
                print("User interrupted")
                stop_event.set()
                break
    
    finally:
        print("Cleaning up resources...")
        stop_event.set()
        read_thread.join(timeout=2)
        process_thread.join(timeout=2)
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print(f"Processing complete after {elapsed:.1f} seconds")
        print(f"Final stats - Avg FPS: {avg_fps:.1f}, Real FPS: {real_avg_fps:.1f}")
        print(f"Counts: {dict(class_counts)}")

if __name__ == "__main__":
    main()