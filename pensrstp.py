import cv2
from ultralytics import YOLO
from collections import defaultdict
import time
import subprocess
import numpy as np
import threading
import queue

# Line definitions
lines = [ ((312, 359), (150, 418)),
          ((132, 448), (492, 463)),
          ((521, 462), (949, 448)),
          ((719, 347), (852, 396))
]

# RTSP server configuration
# RTSP_URL = "rtmp://localhost:1935/live/stream"  # Atau gunakan RTSP URL sesuai server Anda
RTSP_URL = "rtsp://localhost:8554/stream"

url = "https://surakarta.atcsindonesia.info:8086/camera/Tipes.flv"
width, height = 704, 576
frame_size = width * height * 3

output_file = 'output_yolone.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps_output = 25

class RTSPStreamer:
    def __init__(self, rtsp_url, width, height, fps=25):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_queue = queue.Queue(maxsize=10)
        self.streaming = False
        self.process = None
        
    def start_streaming(self):
        """Mulai streaming RTSP menggunakan FFmpeg"""
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',  # Input dari stdin
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-f', 'rtsp',  # Atau 'flv' untuk RTMP
            self.rtsp_url
        ]
        
        try:
            self.process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            self.streaming = True
            print(f"RTSP streaming dimulai ke: {self.rtsp_url}")
        except Exception as e:
            print(f"Error memulai streaming: {e}")
            
    def stream_frame(self, frame):
        """Kirim frame ke stream RTSP"""
        if self.streaming and self.process and self.process.poll() is None:
            try:
                # Resize frame jika diperlukan
                if frame.shape[:2] != (self.height, self.width):
                    frame = cv2.resize(frame, (self.width, self.height))
                
                # Kirim frame ke FFmpeg
                self.process.stdin.write(frame.tobytes())
                self.process.stdin.flush()
            except Exception as e:
                print(f"Error streaming frame: {e}")
                self.streaming = False
                
    def stop_streaming(self):
        """Hentikan streaming"""
        self.streaming = False
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except:
                self.process.kill()
            self.process = None
        print("RTSP streaming dihentikan")

def read_exactly(pipe, size):
    """Baca tepat sejumlah byte dari pipe."""
    buffer = b''
    while len(buffer) < size:
        chunk = pipe.read(size - len(buffer))
        if not chunk:
            return None
        buffer += chunk
    return buffer

def is_crossing_line(p1, p2, a, b):
    def ccw(p, q, r):
        return (r[1]-p[1]) * (q[0]-p[0]) > (q[1]-p[1]) * (r[0]-p[0])
    return ccw(p1, a, b) != ccw(p2, a, b) and ccw(p1, p2, a) != ccw(p1, p2, b)

# Setup FFmpeg untuk membaca stream input
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

# Inisialisasi model YOLO
model = YOLO('yolov8n.pt')
model.to('cuda')

# Alternatif: gunakan video file untuk testing
video_path = r'out_pens0.mp4'
cap = cv2.VideoCapture(video_path)

class_list = model.names
print("Classes in the model:", class_list)
class_counts = defaultdict(int)
track_object_ids = set()
track_history = {}
track_object_ids_per_line = [set() for _ in lines]

# Inisialisasi RTSP streamer
rtsp_streamer = RTSPStreamer(RTSP_URL, width, height, fps_output)

# Setup video writer untuk file output
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        height, width = frame.shape[:2]
        out = cv2.VideoWriter(output_file, fourcc, fps_output, (width, height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Mulai RTSP streaming
        rtsp_streamer.start_streaming()

try:
    while cap.isOpened():
        start_time = time.time()
        
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # YOLO detection dan tracking
        results = model.track(frame, persist=True, classes=[1,2,3,4,5,7])
        
        if results[0].boxes.data is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy
            track_ids = results[0].boxes.id.int().tolist()
            class_ids = results[0].boxes.cls.int().tolist()
            confidence = results[0].boxes.conf
            
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
                
                # Gambar bounding box dan info
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, center_object, 5, (255, 0, 0), -1)
                cv2.putText(frame, f"class {class_name} id{track_id} conf{conf:.2f}", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Gambar garis deteksi
        for i, (line_start, line_end) in enumerate(lines):
            cv2.line(frame, line_start, line_end, (0, 0, 255), 2)
            cv2.putText(frame, f"Line {i+1}", (line_start[0], line_start[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Tambahkan informasi statistik
        cv2.putText(frame, f"Total Count: {sum(class_counts.values())}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for class_name, count in class_counts.items():
            cv2.putText(frame, f"{class_name}: {count}", 
                       (10, 100 + 20 * list(class_counts.keys()).index(class_name)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Hitung dan tampilkan FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time + 1e-6)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Output ke file video
        out.write(frame)
        
        # Stream ke RTSP
        rtsp_streamer.stream_frame(frame)
        
        # Tampilkan frame lokal
        cv2.imshow('Frame', frame)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break

except KeyboardInterrupt:
    print("Streaming dihentikan oleh user")

finally:
    # Cleanup
    rtsp_streamer.stop_streaming()
    cv2.destroyAllWindows()
    cap.release()
    out.release()
    print("Cleanup selesai")