#!/usr/bin/env python3
"""
Development server for YOLO frontend.
Serves the source files directly (not the dist build) for visual testing.
"""

import http.server
import socketserver
import os
import urllib.parse

PORT = 8081
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class DevHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Add CORS headers for development
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def do_GET(self):
        # Handle SPA routing - serve index.html for all non-file paths
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # Check if it's a file request
        if path.startswith('/src/') or path.startswith('/public/'):
            return super().do_GET()

        # For API routes, return mock data
        if path.startswith('/api/'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            if path == '/api/health':
                self.wfile.write(b'{"status":"ok"}')
            elif path == '/api/cameras':
                self.wfile.write(b'[{"id":0,"name":"默认摄像头"}]')
            elif path == '/api/detection/status':
                self.wfile.write(b'{"running":false,"fps":0}')
            elif path == '/api/events/stats':
                self.wfile.write(b'{"total":0,"by_type":{},"by_hour":[]}')
            elif path == '/api/events':
                self.wfile.write(b'{"items":[],"total":0}')
            elif path == '/api/alarms':
                self.wfile.write(b'{"items":[],"total":0}')
            elif path == '/api/alarms/stats':
                self.wfile.write(b'{"total":0,"by_level":{},"by_status":{}}')
            elif path == '/api/config':
                self.wfile.write(b'{"confidence":0.5,"iou":0.45}')
            elif path == '/api/mllm/config':
                self.wfile.write(b'{"enabled":false}')
            else:
                self.wfile.write(b'{}')
            return

        # Serve index.html for SPA routes
        if path == '/' or not os.path.exists(os.path.join(DIRECTORY, path.lstrip('/'))):
            self.path = '/index.html'

        return super().do_GET()

if __name__ == '__main__':
    with socketserver.TCPServer(("", PORT), DevHandler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        print(f"Directory: {DIRECTORY}")
        httpd.serve_forever()
