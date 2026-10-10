#!/usr/bin/env python3
"""Serve the controller demo. One Swift process stays loaded and answers prompts."""

import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = Path(__file__).resolve().parent / "index.html"
PORT = 8787

SCHEMA = {
    "set_timer": ["minutes", "label"],
    "set_alarm": ["time", "label"],
    "add_reminder": ["text", "when"],
    "send_message": ["contact", "text"],
    "create_event": ["title", "date", "time"],
    "get_weather": ["location", "when"],
    "play_music": ["title", "artist"],
    "navigate": ["destination"],
    "add_note": ["text"],
    "open_app": ["app"],
}


def parse_call(text):
    raw = text.strip()
    if "<refuse>" in raw:
        request = raw.split("<refuse>", 1)[0].replace("<request>", "").strip()
        return {"kind": "refuse", "request": request, "tool": None, "arguments": {}, "raw": raw}
    if "<call>" not in raw:
        request = raw.replace("<request>", "").strip()
        return {"kind": "none", "request": request, "tool": None, "arguments": {}, "raw": raw}
    head, tail = raw.split("<call>", 1)
    request = head.replace("<request>", "").strip()
    toks = tail.split()
    if not toks or toks[0] not in SCHEMA:
        return {"kind": "none", "request": request, "tool": None, "arguments": {}, "raw": raw}
    tool = toks[0]
    keys = set(SCHEMA[tool])
    args = {}
    i = 1
    while i < len(toks):
        key = toks[i]
        if key not in keys:
            i += 1
            continue
        i += 1
        value = []
        while i < len(toks) and toks[i] not in keys:
            value.append(toks[i])
            i += 1
        args[key] = " ".join(value)
    return {"kind": "call", "request": request, "tool": tool, "arguments": args, "raw": raw}


class Model:
    def __init__(self):
        binary = ROOT / "build" / "gpt"
        if not binary.exists():
            raise SystemExit("build/gpt is missing. Run make demo from the repo.")
        ckpt = ROOT / "build" / "tools.ckpt"
        if not ckpt.exists():
            raise SystemExit("build/tools.ckpt is missing.")
        self.lock = threading.Lock()
        self.proc = subprocess.Popen(
            [str(binary), "tools-repl", "--ckpt", str(ckpt), "--vocab", str(ROOT / "vocab" / "tools.txt"), "--temp", "0"],
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def ask(self, prompt):
        line_in = " ".join(prompt.split())
        if not line_in:
            return {"error": "empty"}
        with self.lock:
            if self.proc.poll() is not None:
                raise RuntimeError("the model process stopped")
            self.proc.stdin.write(line_in + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("the model process stopped")
        data = json.loads(line)
        if data.get("error") == "unknown":
            return {"error": "unknown", "unknown": data.get("unknown") or []}
        return parse_call(data.get("text") or "")


MODEL = Model()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def send_json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?", 1)[0] not in ("/", "/index.html"):
            self.send_error(404)
            return
        body = PAGE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/ask":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode() or "{}")
        except json.JSONDecodeError:
            self.send_json(400, {"error": "bad json"})
            return
        try:
            result = MODEL.ask(str(payload.get("prompt") or ""))
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})
            return
        self.send_json(200, result)


def main():
    # First call pays for the Metal compile. Later taps stay on the loaded weights.
    MODEL.ask("Could you set a 20 minute timer for pasta")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"demo http://127.0.0.1:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
