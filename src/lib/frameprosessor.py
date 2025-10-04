import cv2
import supervision as sv



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
