from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import time
from datetime import datetime
import threading


class InfluxDBLogger:
    def __init__(self, url, token, org, bucket, location_name="camera1", measurement="vehicle_counts"):
        """
        Initialize the InfluxDB logger
        
        Args:
            url: InfluxDB server URL
            token: Authentication token
            org: Organization name
            bucket: Bucket name to write to
            location_name: Identifier for this camera/counter location
            measurement: Measurement name in InfluxDB
        """
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.bucket = bucket
        self.org = org
        self.location_name = location_name
        self.measurement = measurement
        self.last_counts = {}
        self.lock = threading.Lock()
        self.running = True
        
    def log_count(self, class_name, count):
        """Log a single vehicle class count"""
        point = Point(self.measurement) \
            .tag("location", self.location_name) \
            .tag("class", class_name) \
            .field("count", count) \
            .time(time.time_ns())
        
        try:
            self.write_api.write(bucket=self.bucket, record=point)
            return True
        except Exception as e:
            print(f"Error writing to InfluxDB: {e}")
            return False
            
    def log_counts(self, counts_dict):
        """Log all vehicle class counts at once"""
        with self.lock:
            # Only log if counts have changed
            if counts_dict == self.last_counts:
                return True
                
            points = []
            for class_name, count in counts_dict.items():
                point = Point(self.measurement) \
                    .tag("location", self.location_name) \
                    .tag("class", class_name) \
                    .field("count", count) \
                    .time(time.time_ns())
                points.append(point)
                
            try:
                if points:
                    self.write_api.write(bucket=self.bucket, record=points)
                    self.last_counts = counts_dict.copy()
                return True
            except Exception as e:
                print(f"Error writing to InfluxDB: {e}")
                return False

    def log_crossing_event(self, class_name, track_id, line_id):
        """Log individual crossing events"""
        point = Point("crossing_events") \
            .tag("location", self.location_name) \
            .tag("class", class_name) \
            .tag("line_id", f"line_{line_id}") \
            .field("track_id", track_id) \
            .field("event", 1) \
            .time(time.time_ns())
        
        try:
            self.write_api.write(bucket=self.bucket, record=point)
            return True
        except Exception as e:
            print(f"Error writing crossing event to InfluxDB: {e}")
            return False
            
    def start_periodic_logging(self, counter, interval=60):
        """Start a background thread to periodically log counts"""
        def _log_periodically():
            while self.running:
                counts = counter.get_counts()
                self.log_counts(counts)
                time.sleep(interval)
                
        thread = threading.Thread(target=_log_periodically, daemon=True)
        thread.start()
        return thread
        
    def close(self):
        """Close the InfluxDB client connection"""
        self.running = False
        self.client.close()