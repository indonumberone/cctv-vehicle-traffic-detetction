from flask import Flask, render_template, send_from_directory, Response, jsonify, redirect, url_for, request
import os
import json
import time
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')

# InfluxDB Configuration
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_INIT_ADMIN_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_INIT_ORG")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_INIT_BUCKET", "vehicle_counting")
LOCATION_NAME = os.getenv("LOCATION_NAME", "camera1")

fps_data = {
    'current_fps': 0,
    'target_fps': 25,
    'last_update': time.time()
}

def get_influxdb_client():
    """Create and return InfluxDB client"""
    if INFLUXDB_TOKEN:
        return InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    return None

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/cctv')
def cctv():
    return render_template('cctv.html')

@app.route('/output/<path:filename>')
def serve_hls(filename):
    """Serve HLS files (playlist.m3u8 dan segment .ts files)"""
    try:
        if filename.endswith('.m3u8'):
            mimetype = 'application/vnd.apple.mpegurl'
        elif filename.endswith('.ts'):
            mimetype = 'video/mp2t'
        else:
            mimetype = None
        
        return send_from_directory(OUTPUT_DIR, filename, mimetype=mimetype)
    except Exception as e:
        return Response(f"Error: {str(e)}", status=404)

@app.route('/api/stats')
def get_stats():
    """API endpoint untuk mendapatkan statistik real-time"""
    try:
        # Baca stats dari file jika ada (HLSStreamer menyimpan stats)
        stats_file = os.path.join(OUTPUT_DIR, 'stream_stats.json')
        
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                fps_data['current_fps'] = stats.get('fps', 0)
                fps_data['target_fps'] = stats.get('target_fps', 25)
                fps_data['last_update'] = time.time()
                return jsonify(stats)
        else:
            return jsonify({
                'fps': fps_data['current_fps'],
                'target_fps': fps_data['target_fps'],
                'status': 'no_data',
                'last_update': fps_data['last_update']
            })
    except Exception as e:
        return jsonify({'error': str(e), 'fps': 0}), 500

@app.route('/api/vehicle-counts')
def get_vehicle_counts():
    """API endpoint untuk mendapatkan total kendaraan per kelas"""
    try:
        client = get_influxdb_client()
        if not client:
            return jsonify({'error': 'InfluxDB not configured'}), 500
        
        query_api = client.query_api()
        
        # Query untuk mendapatkan total kendaraan per kelas (sum dari incremental counts)
        # Menggunakan range 24 jam terakhir
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: -24h)
          |> filter(fn: (r) => r._measurement == "vehicle_counts")
          |> filter(fn: (r) => r.location == "{LOCATION_NAME}")
          |> filter(fn: (r) => r._field == "count")
          |> group(columns: ["class"])
          |> sum()
        '''
        
        result = query_api.query(query=query)
        
        counts = {
            'car': 0,
            'motorcycle': 0,
            'truck': 0
        }
        
        for table in result:
            for record in table.records:
                vehicle_class = record.values.get('class', '').lower()
                count_value = record.get_value()
                if vehicle_class in counts:
                    counts[vehicle_class] = int(count_value)
        
        client.close()
        return jsonify(counts)
        
    except Exception as e:
        print(f"Error fetching vehicle counts: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/vehicle-chart')
def get_vehicle_chart():
    """API endpoint untuk mendapatkan data grafik kendaraan per interval waktu"""
    try:
        # Get time range from query parameters (default: last 6 hours)
        hours_str = request.args.get('hours', '6')
        hours = float(hours_str)  # Support decimal hours (e.g., 0.17 for 10 minutes)
        interval = request.args.get('interval', '10m')  # default 10 menit
        
        client = get_influxdb_client()
        if not client:
            return jsonify({'error': 'InfluxDB not configured'}), 500
        
        query_api = client.query_api()
        
        # Convert hours to minutes for better precision on short ranges
        if hours < 1:
            minutes = int(hours * 60)
            time_range = f'{minutes}m'
        else:
            time_range = f'{int(hours)}h'
        
        # Query untuk mendapatkan data per interval waktu
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: -{time_range})
          |> filter(fn: (r) => r._measurement == "vehicle_counts")
          |> filter(fn: (r) => r.location == "{LOCATION_NAME}")
          |> filter(fn: (r) => r._field == "count")
          |> aggregateWindow(every: {interval}, fn: sum, createEmpty: false)
          |> group(columns: ["class", "_time"])
          |> sort(columns: ["_time"])
        '''
        
        result = query_api.query(query=query)
        
        # Organize data by class
        data_by_class = {
            'car': [],
            'motorcycle': [],
            'truck': []
        }
        timestamps = set()
        
        for table in result:
            for record in table.records:
                vehicle_class = record.values.get('class', '').lower()
                timestamp = record.get_time()
                count_value = record.get_value()
                
                if vehicle_class in data_by_class:
                    timestamps.add(timestamp)
                    data_by_class[vehicle_class].append({
                        'time': timestamp.isoformat(),
                        'count': int(count_value) if count_value else 0
                    })
        
        client.close()
        
        return jsonify({
            'car': data_by_class['car'],
            'motorcycle': data_by_class['motorcycle'],
            'truck': data_by_class['truck']
        })
        
    except Exception as e:
        print(f"Error fetching vehicle chart data: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        print(f"Warning: Output directory tidak ditemukan: {OUTPUT_DIR}")
    else:
        print(f"Serving HLS files from: {OUTPUT_DIR}")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
