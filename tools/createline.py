import cv2

# Simpan koordinat titik klik
points = []
line_segments = []  # To store all line segments
frame_copy = None

# Mouse callback
def draw_line(event, x, y, flags, param):
    global points, frame_copy, line_segments

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Klik di: ({x}, {y})")
        points.append((x, y))

        # Draw a line after every two points
        if len(points) >= 2 and len(points) % 2 == 0:
            # Get the last two points
            pt1 = points[-2]
            pt2 = points[-1]
            line_segments.append((pt1, pt2))  # Store the line segment
            
            # Redraw the frame_copy with all lines
            frame_copy = frame.copy()
            for segment in line_segments:
                cv2.line(frame_copy, segment[0], segment[1], (0, 255, 0), 2)
            
            cv2.imshow("Video", frame_copy)
            print(f"Garis dari {pt1} ke {pt2}")

# Load video (bisa ganti dengan stream URL)
video_path = "/home/muqsith/pengmas/testing/video/input/input0.mp4"  # ganti path kamu di sini
cap = cv2.VideoCapture(video_path)

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", draw_line)

paused = False
points = []  # Make sure points is initialized

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Create a copy of the frame and draw all lines on it
        frame_copy = frame.copy()
        for segment in line_segments:
            cv2.line(frame_copy, segment[0], segment[1], (0, 255, 0), 2)
            
        cv2.imshow("Video", frame_copy)

    key = cv2.waitKey(30) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord('c'):
        # Clear garis
        points = []
        frame_copy = frame.copy()
        print("Reset garis")
    elif key == ord('p'):
        paused = not paused
        print("Paused" if paused else "Play")

cap.release()
cv2.destroyAllWindows()
