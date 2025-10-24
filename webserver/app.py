from flask import Flask, render_template, send_from_directory, Response
import os

app = Flask(__name__)

# Path ke folder output yang berisi HLS files
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')

@app.route('/')
def index():
    """Halaman utama dengan video player"""
    return render_template('index.html')

@app.route('/output/<path:filename>')
def serve_hls(filename):
    """Serve HLS files (playlist.m3u8 dan segment .ts files)"""
    try:
        # Set correct MIME types untuk HLS
        if filename.endswith('.m3u8'):
            mimetype = 'application/vnd.apple.mpegurl'
        elif filename.endswith('.ts'):
            mimetype = 'video/mp2t'
        else:
            mimetype = None
        
        return send_from_directory(OUTPUT_DIR, filename, mimetype=mimetype)
    except Exception as e:
        return Response(f"Error: {str(e)}", status=404)

@app.route('/about')
def about():
    """Halaman about"""
    return render_template('about.html')

@app.route('/status')
def status():
    """Halaman status"""
    return render_template('status.html')

if __name__ == '__main__':
    # Pastikan folder output ada
    if not os.path.exists(OUTPUT_DIR):
        print(f"Warning: Output directory tidak ditemukan: {OUTPUT_DIR}")
    else:
        print(f"Serving HLS files from: {OUTPUT_DIR}")
    
    # Jalankan Flask app
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
