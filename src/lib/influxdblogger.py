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
        """Log vehicle class counts incrementally (+1 per event)
        Use sum() in InfluxDB queries to get totals"""
        with self.lock:
            # Skip if no new events (all values are 0)
            if not counts_dict or all(v == 0 for v in counts_dict.values()):
                return True
                
            points = []
            for class_name, count in counts_dict.items():
                if count > 0:  # Only log if there's an actual event
                    point = Point(self.measurement) \
                        .tag("location", self.location_name) \
                        .tag("class", class_name) \
                        .field("count", count) \
                        .time(time.time_ns())
                    points.append(point)
                
            try:
                if points:
                    self.write_api.write(bucket=self.bucket, record=points)
                    print(f"✅ Logged {len(points)} incremental event(s) to InfluxDB")
                return True
            except Exception as e:
                print(f"Error writing to InfluxDB: {e}")
                return False

    def log_crossing_event(self, class_name, track_id, line_id, count=1):
        """Log individual crossing events with incremental count"""
        point = Point("crossing_events") \
            .tag("location", self.location_name) \
            .tag("class", class_name) \
            .tag("line_id", f"line_{line_id}") \
            .field("track_id", track_id) \
            .field("event", count) \
            .time(time.time_ns())
        
        try:
            self.write_api.write(bucket=self.bucket, record=point)
            return True
        except Exception as e:
            print(f"Error writing crossing event to InfluxDB: {e}")
            return False
            
    def start_periodic_logging(self, counter, interval=60):
        """Start a background thread to periodically log incremental counts"""
        def _log_periodically():
            while self.running:
                # Get incremental counts (resets after reading)
                incremental_counts = counter.get_incremental_counts()
                if incremental_counts:
                    self.log_counts(incremental_counts)
                time.sleep(interval)
                
        thread = threading.Thread(target=_log_periodically, daemon=True)
        thread.start()
        return thread
        
    def close(self):
        """Close the InfluxDB client connection"""
        self.running = False
        self.client.close()