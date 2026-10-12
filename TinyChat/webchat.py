"""
webchat.py - puts the bot on a web page and lets other programs talk to it.

    python3 webchat.py          then open http://localhost:8000 in a browser

While this is running it offers two things:

    the chat page   http://localhost:8000         (the file chat_page.html)
    the API         http://localhost:8000/chat    for your own app or website

The API is how anything you build talks to the bot. Send a message as JSON
and the answer comes back as JSON:

    send     POST /chat   {"message": "weather in tokyo"}
    receive               {"reply": "in Tokyo, Japan it is ...", "tool": "weather"}

Normally only this computer can reach it. To let a phone or another computer
on the same wifi reach it as well:

    python3 webchat.py --network

On a hosting service (see "Put it online" in README.md) no options are needed:
the host sets PORT and this program opens itself to the internet on that port.
"""
import argparse
import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tinychat import Bot

PAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_page.html")
LONGEST_MESSAGE = 300      # characters; longer messages are cut off

bot = Bot()
one_at_a_time = threading.Lock()   # the brain answers one message at a time


class Handler(BaseHTTPRequestHandler):
    def send(self, status, body, kind):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", kind + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        # These three lines let a web page hosted somewhere else call the API.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, status, data):
        self.send(status, json.dumps(data), "application/json")

    def do_OPTIONS(self):
        # Browsers ask permission this way before a page calls the API.
        self.send(204, "", "text/plain")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            with open(PAGE_FILE, encoding="utf-8") as f:
                self.send(200, f.read(), "text/html")
        elif path == "/info":
            self.send_json(200, {"numbers": bot.size})
        else:
            self.send_json(404, {"error": "there is nothing at this address"})

    def do_POST(self):
        if self.path.split("?")[0] != "/chat":
            return self.send_json(404, {"error": "send messages to /chat"})
        try:
            length = min(int(self.headers.get("Content-Length", 0)), 10_000)
            message = json.loads(self.rfile.read(length))["message"]
            message = str(message).strip()[:LONGEST_MESSAGE]
        except (ValueError, KeyError, TypeError):
            return self.send_json(400, {"error": 'send JSON like {"message": "hello"}'})
        if not message:
            return self.send_json(400, {"error": "the message was empty"})
        with one_at_a_time:
            reply, tool = bot.respond(message)
        print(f"you: {message}\nbot: {reply}\n")
        self.send_json(200, {"reply": reply, "tool": tool})

    def log_message(self, *args):
        pass    # keep the terminal quiet; do_POST prints each exchange instead


def wifi_address():
    """This computer's address on the local network."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))    # nothing is actually sent
        return probe.getsockname()[0]
    except OSError:
        return "this-computer"
    finally:
        probe.close()


if __name__ == "__main__":
    options = argparse.ArgumentParser(description="Serve the chatbot as a web page and an API.")
    options.add_argument("--network", action="store_true", help="let other devices on your wifi connect")
    options.add_argument("--port", type=int, default=8000)
    chosen = options.parse_args()

    # A hosting company tells the program which port to use through a setting
    # called PORT. If it is there, we are on a host: accept visitors from
    # anywhere on that port.
    hosted_port = os.environ.get("PORT")
    if hosted_port:
        server = ThreadingHTTPServer(("0.0.0.0", int(hosted_port)), Handler)
        print(f"The bot is running on a host, port {hosted_port}.\n", flush=True)
    else:
        server = ThreadingHTTPServer(("0.0.0.0" if chosen.network else "127.0.0.1", chosen.port), Handler)
        print(f"The bot is running. Open this in a browser:  http://localhost:{chosen.port}")
        if chosen.network:
            print(f"From a phone on the same wifi:              http://{wifi_address()}:{chosen.port}")
        print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
