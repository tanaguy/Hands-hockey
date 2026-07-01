#!/usr/bin/env python3
"""Minimal static file server for the Hand Hockey app.

Serves the project folder over http://localhost:5173 with correct JavaScript
MIME types so native ES modules load in the browser. localhost counts as a
secure context, so getUserMedia() (webcam) works without HTTPS.
"""
import functools
import http.server
import os
import socketserver

PORT = 5175
ROOT = os.path.dirname(os.path.abspath(__file__))

Handler = http.server.SimpleHTTPRequestHandler
# Guarantee correct MIME types regardless of the host OS registry.
Handler.extensions_map = {
    **Handler.extensions_map,
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".html": "text/html",
}


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    handler = functools.partial(Handler, directory=ROOT)
    with Server(("", PORT), handler) as httpd:
        print(f"Hand Hockey serving at http://localhost:{PORT}")
        httpd.serve_forever()
