import json

import numpy as np
from flask import Flask, render_template
from flask_sock import Sock

from detector import PedalDetector


app = Flask(__name__)
sock = Sock(app)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@sock.route("/ws/audio")
def audio_socket(ws):
    detector = None

    while True:
        message = ws.receive()
        if message is None:
            break

        if isinstance(message, str):
            config = json.loads(message)
            if config.get("type") == "config":
                detector = PedalDetector(sample_rate=int(config["sampleRate"]))
                ws.send(json.dumps({"type": "ready", "sample_rate": detector.sample_rate}))
            continue

        if detector is None:
            ws.send(json.dumps({"type": "error", "message": "Missing audio config"}))
            continue

        samples = np.frombuffer(message, dtype=np.float32)
        result = detector.process_samples(samples)
        result["type"] = "analysis"
        ws.send(json.dumps(result))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=9000, debug=False, use_reloader=False)
