import cv2
import time
import threading



class RTSPReconnector:
    """Handles RTSP stream with auto-reconnect"""
    def __init__(self, rtsp_url, max_retry=5, retry_delay=3):
        self.rtsp_url = rtsp_url
        self.max_retry = max_retry
        self.retry_delay = retry_delay
        self.cap = None
        self.reconnect_count = 0
        self.lock = threading.Lock()
        
    def connect(self):
        """Open RTSP connection"""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self.cap.isOpened():
                print(f"✅ Connected to RTSP stream (attempt {self.reconnect_count + 1})")
                return True
            return False
    
    def reconnect(self):
        """Attempt to reconnect"""
        print(f"Reconnecting... (attempt {self.reconnect_count + 1}/{self.max_retry})")
        
        attempt = 0
        
        while True:
            attempt += 1
            time.sleep(self.retry_delay)
            if self.connect():
                self.reconnect_count += 1
                print(f"Reconnected to RTSP stream (attempt {self.reconnect_count})")
                return True
            print(f"reconnect failed ({attempt}")
            
        
        
        # for attempt in range(self.max_retry):
        #     time.sleep(self.retry_delay)
        #     if self.connect():
        #         self.reconnect_count += 1
        #         return True
        #     print(f"reconnect failed ({attempt + 1}/{self.max_retry})")
        
        # print(f"Max reconnection attempts reached")
        # return False
    
    def read(self):
        """Read frame with auto-reconnect on failure"""
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                return False, None
            
            ret, frame = self.cap.read()
            return ret, frame
    
    def release(self):
        """Release the capture"""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
    
    def get_property(self, prop):
        """Get video property"""
        with self.lock:
            if self.cap is not None:
                return self.cap.get(prop)
            return 0
