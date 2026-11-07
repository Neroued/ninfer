"""long_context.py: each request of the server's log pairs with the decode probe that sent it."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

BENCH = Path(__file__).resolve().parents[1] / "bench" / "capy_flash"
sys.path.insert(0, str(BENCH))
import long_context  # noqa: E402


def request_done(client, prompt_tokens):
    return {
        "event": "request_done",
        "request": {"client": client},
        "result": {"prompt_tokens": prompt_tokens, "completion_tokens": 100},
        "timings_seconds": {"decode": 1.0},
    }


def elastic(slots, kv_mib):
    return f"elastic KV (2 GPUs) ({slots} slots/layer/GPU), MTP KV 4 pages, {kv_mib << 20} KV bytes"


class Tokenizer:
    def encode(self, text):
        return SimpleNamespace(offsets=[(i, i + 1) for i in range(len(text))])


class Response:
    def __enter__(self):
        return iter([b'data: {"choices": [{"delta": {"content": "x"}}]}', b"data: [DONE]"])

    def __exit__(self, *exc):
        return False


class LongContextTests(unittest.TestCase):
    def test_response_timing_excludes_log_wait_and_ignores_wall_clock_changes(self):
        for first_at, fail in [(0.0, False), (0.25, False), (None, True), (0.25, True)]:
            with self.subTest(first_at=first_at, fail=fail), tempfile.TemporaryDirectory() as tmp:
                clock = [0.0]

                class TimedResponse(Response):
                    def __enter__(self):
                        return iter(self)

                    def __iter__(self):
                        if first_at is not None:
                            clock[0] = first_at
                            yield b'data: {"choices": [{"delta": {"content": "x"}}]}'
                        clock[0] = 0.75
                        if fail:
                            raise OSError("connection lost")
                        yield b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}'
                        yield b"data: [DONE]"

                def sleep(seconds):
                    clock[0] += seconds

                out = Path(tmp) / "lc.jsonl"
                args = SimpleNamespace(
                    url="http://x", model="m", out=str(out), sizes=[3], max_tokens=4, server_log=None
                )
                with (
                    patch.object(long_context, "post", return_value=TimedResponse()),
                    patch.object(long_context.time, "perf_counter", side_effect=lambda: clock[0]),
                    patch.object(long_context.time, "time", side_effect=lambda: 100 - clock[0] * 1000),
                    patch.object(long_context.time, "sleep", side_effect=sleep),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    long_context.decode(args, Tokenizer(), "a" * 64)
                row = json.loads(out.read_text())
                self.assertEqual(row["wall_s"], 0.75)
                self.assertEqual(row["ttft_s"], first_at)
                self.assertEqual(clock[0], 1.75)
                self.assertEqual(row["error"] is not None, fail)

    def test_decode_records_the_label_it_sends(self):
        sent = []

        def post(url, body, timeout=3600, client=None):
            sent.append(client)
            return Response()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "lc.jsonl"
            args = SimpleNamespace(url="http://x", model="m", out=str(out), sizes=[3, 5], max_tokens=4, server_log=None)
            with patch.object(long_context, "post", post), patch.object(long_context.time, "sleep"):
                with contextlib.redirect_stdout(io.StringIO()):
                    long_context.decode(args, Tokenizer(), "a" * 64)
            probes = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual([probe["client"] for probe in probes], sent)
        self.assertEqual(len(set(sent)), 2)

    def test_summarize_pairs_by_label_across_runs(self):
        # Another client's request first, then two runs of the same sizes into one appended OUT.
        log = [request_done(None, 50)]
        probes = []
        for run, slots in (("r1", 10), ("r2", 20)):
            for n in (1000, 2000):
                client = f"long_context/{run}/{n}"
                log.append(request_done(client, n + slots))
                probes.append({"n": n, "client": client, "elastic": elastic(slots + n // 1000, n // 100)})
        with tempfile.TemporaryDirectory() as tmp:
            log_path, out_path = Path(tmp) / "reqlog.jsonl", Path(tmp) / "lc.jsonl"
            log_path.write_text("".join(json.dumps(r) + "\n" for r in log))
            out_path.write_text("".join(json.dumps(p) + "\n" for p in probes))
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                long_context.summarize(SimpleNamespace(request_log=str(log_path), out=str(out_path)))
        lines = printed.getvalue().splitlines()
        self.assertEqual(len(lines), 5)
        self.assertNotIn("slots", lines[0])
        expected = {1010: (11, 10), 2010: (12, 20), 1020: (21, 10), 2020: (22, 20)}
        for line in lines[1:]:
            prompt = int(line.split(":")[0].split()[1])
            slots, kv = expected[prompt]
            self.assertIn(f"slots/layer/GPU {slots}, KV {kv} MiB/GPU", line)


if __name__ == "__main__":
    unittest.main()
