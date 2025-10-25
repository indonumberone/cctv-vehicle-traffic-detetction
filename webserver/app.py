from flask import Flask, render_template, send_from_directory, Response, jsonify,redirect, url_for
import os
import json
import time

app = Flask(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')

fps_data = {
    'current_fps': 0,
    'target_fps': 25,
    'last_update': time.time()
}

@app.route('/')
@app.route('/cctv')
def index():
    return render_template('index.html')

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

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        print(f"Warning: Output directory tidak ditemukan: {OUTPUT_DIR}")
    else:
        print(f"Serving HLS files from: {OUTPUT_DIR}")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
