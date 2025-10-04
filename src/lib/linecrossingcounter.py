from collections import defaultdict, deque
import time
from datetime import datetime



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
