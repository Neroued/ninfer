"""Stop a real server with a live client still waiting for generation."""

import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request


def main() -> int:
    artifact = os.environ.get("NINFER_TEST_ARTIFACT")
    if not artifact:
        print("SKIP: set NINFER_TEST_ARTIFACT to an explicit .ninfer artifact")
        return 77
    root = Path(__file__).resolve().parents[1] / ".agent-tmp/serve-shutdown-test"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        log_path = Path(temporary) / "server.log"
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [sys.argv[1], artifact, "--host", "127.0.0.1", "--port", str(port),
                 "--max-context", "8192", "--max-concurrency", "1", "--spec", "none",
                 "--no-cuda-graph", "--host-cache-capacity", "0B", "--stats-interval", "1s"],
                stdout=log, stderr=subprocess.STDOUT,
            )
            client = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
            try:
                endpoint = f"http://127.0.0.1:{port}"
                deadline = time.monotonic() + 120
                while True:
                    if process.poll() is not None:
                        raise AssertionError("server exited during startup")
                    try:
                        with urllib.request.urlopen(endpoint + "/v1/models", timeout=1) as r:
                            model = json.load(r)["data"][0]["id"]
                        break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise AssertionError("server startup timed out")
                        time.sleep(0.05)
                client.request("POST", "/v1/chat/completions", json.dumps({
                    "model": model, "stream": False, "max_tokens": 4096,
                    "messages": [{"role": "user", "content":
                        "Read this sequence: " + " item" * 3500 +
                        " Then write 2000 numbered sentences. Continue without summarizing."}],
                }), {"Content-Type": "application/json"})
                deadline = time.monotonic() + 15
                while True:
                    with urllib.request.urlopen(endpoint + "/monitor/snapshot", timeout=2) as r:
                        snapshot = json.load(r)
                    residents = [q for q in snapshot["engine"]["requests"] if q["resident"]]
                    if residents:
                        break
                    if time.monotonic() > deadline:
                        raise AssertionError("shutdown fixture never became resident")
                    time.sleep(0.01)
                started = time.monotonic()
                # Keep the socket open: client disconnection must not rescue shutdown.
                process.send_signal(signal.SIGTERM)
                assert process.wait(timeout=10) == 0, "server did not exit successfully"
                response = client.getresponse()
                response.read()
                assert response.status == 503, "in-flight request did not receive shutdown error"
                print(f"PASS SIGTERM drained a resident request with client connected in "
                      f"{time.monotonic() - started:.3f}s")
            except BaseException:
                print(log_path.read_text(), file=sys.stderr)
                raise
            finally:
                client.close()
                if process.poll() is None:
                    process.kill()
                    process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
