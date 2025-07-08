
import cv2
from ultralytics import YOLO
from collections import defaultdict
import time
import subprocess
import numpy as np



        #   ((724, 343), (805, 379))

lines = [ ((312, 359), (150, 418)),
          ((132, 448), (492, 463)),
          ((521, 462), (949, 448)),
          ((719, 347), (852, 396))
]

url = "https://surakarta.atcsindonesia.info:8086/camera/Tipes.flv"
width, height = 704, 576
frame_size = width * height * 3

output_file = 'output_yolone.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps_output = 25  # Default


def read_exactly(pipe, size):
    """Baca tepat sejumlah byte dari pipe."""
    buffer = b''
    while len(buffer) < size:
        chunk = pipe.read(size - len(buffer))
        if not chunk:
            return None
        buffer += chunk
    return buffer

ffmpeg_cmd = [
    'ffmpeg',
    '-fflags', 'nobuffer',
    '-flags', 'low_delay',
    '-i', url,
    '-vf', 'fps=25,scale=704:576',
    '-f', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-loglevel', 'quiet',
    '-an', '-sn',
    '-'
]

process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=0)


model = YOLO('yolov8n.pt')
model.to('cuda')  
video_path = r'out_pens0.mp4'
# video_path = r'../video_train/simpanglima229.ts'
cap = cv2.VideoCapture(video_path)

class_list = model.names
print("Classes in the model:", class_list)
class_counts = defaultdict(int)
track_object_ids = set()
track_history = {}

# line_start = (29, 253)
# line_end = (655, 216)

track_object_ids_per_line = [set() for _ in lines]


def is_crossing_line(p1, p2, a, b):
    def ccw(p, q, r):
        return (r[1]-p[1]) * (q[0]-p[0]) > (q[1]-p[1]) * (r[0]-p[0])
    return ccw(p1, a, b) != ccw(p2, a, b) and ccw(p1, p2, a) != ccw(p1, p2, b)


if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        height, width = frame.shape[:2]
        out = cv2.VideoWriter(output_file, fourcc, fps_output, (width, height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# while True:
while cap.isOpened():
    start_time = time.time()
    
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # raw_frame = read_exactly(process.stdout, frame_size)
    # if raw_frame is None:
    #     print("Stream berhenti.")
    #     break

    # frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3)).copy()
    # frame.setflags(write=1)
    resuls = model.track(frame, persist=True,classes=[1,2,3,4,5,7])
    print("Results:", resuls[0].boxes.data )
    if resuls[0].boxes.data is not None and resuls[0].boxes.id is not None:
        boxes = resuls[0].boxes.xyxy
        track_ids = resuls[0].boxes.id.int().tolist()
        class_ids = resuls[0].boxes.cls.int().tolist()
        confidence = resuls[0].boxes.conf
        
        for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confidence):
            x1, y1, x2, y2 = map(int, box.tolist())
            class_name = class_list[class_id]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            center_object = (cx, cy)
            
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append(center_object)
            
            if len(track_history[track_id]) >= 2:
                prev, curr = track_history[track_id][-2], track_history[track_id][-1]
                
                for i, (line_start, line_end) in enumerate(lines):
                    if is_crossing_line(prev, curr, line_start, line_end):
                        if track_id not in track_object_ids_per_line[i]:
                            track_object_ids_per_line[i].add(track_id)
                            class_counts[class_name] += 1
                            print(f"Track ID {track_id} crossed line {i+1}")
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, center_object, 5, (255, 0, 0), -1)
            cv2.putText(frame, f"class {class_name} id{track_id}  conf{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for i, (line_start, line_end) in enumerate(lines):
        cv2.line(frame, line_start, line_end, (0, 0, 255), 2)
        cv2.putText(frame, f"Line {i+1}", (line_start[0], line_start[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, f"Total Count: {sum(class_counts.values())}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for class_name, count in class_counts.items():
        cv2.putText(frame, f"{class_name}: {count}", (10, 100 + 20 * list(class_counts.keys()).index(class_name)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    end_time = time.time()
    fps = 1 / (end_time - start_time + 1e-6)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow('Frame', frame)
    out.write(frame)
    key = cv2.waitKey(1)
    if key == 27:  
        break
cv2.destroyAllWindows()
cap.release()
out.release()