from collections import  deque



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

