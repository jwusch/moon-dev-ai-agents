#!/usr/bin/env python3
"""
🌊 Fractal Alpha Dashboard Server
Simple web server to view the dashboard in your browser
"""

import http.server
import socketserver
import webbrowser
import os
import json
from datetime import datetime

PORT = 8888

class DashboardServer(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            # Serve the HTML viewer
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            with open('fractal_dashboard_viewer.html', 'rb') as f:
                self.wfile.write(f.read())
                
        elif self.path == '/latest':
            # Find and serve the latest dashboard JSON
            json_files = [f for f in os.listdir('.') if f.startswith('fractal_alpha_dashboard_') and f.endswith('.json')]
            
            if json_files:
                # Sort by timestamp in filename
                latest_file = sorted(json_files)[-1]
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                with open(latest_file, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'No dashboard files found')
                
        else:
            # Default file serving
            super().do_GET()

def main():
    print("🌊💎 Fractal Alpha Dashboard Server 💎🌊")
    print("=" * 50)
    print(f"🚀 Starting server on http://localhost:{PORT}")
    print("📊 Opening dashboard in your browser...")
    print("\n✋ Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start server
    with socketserver.TCPServer(("", PORT), DashboardServer) as httpd:
        # Open browser
        webbrowser.open(f'http://localhost:{PORT}')
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n🛑 Server stopped")
            return

if __name__ == "__main__":
    main()