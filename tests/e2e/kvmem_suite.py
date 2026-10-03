#!/usr/bin/env python3
"""Real-model, fail-closed sparse KV regression runner (Python 3.11+).

Owns only its child server. Every request is attempted once. JSON and JUnit reports
include failures, incomplete SSE, startup errors, and engine errors after HTTP 200.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import os
import re
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET


def parse_sse(lines):
    chunks, done, finished, usage, tool_calls = [], False, False, None, False
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        event = json.loads(data)
        if "error" in event:
            raise AssertionError(f"SSE error: {event['error']}")
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            tool_calls |= bool(delta.get("tool_calls"))
            chunks.append(delta.get("content") or delta.get("reasoning_content") or "")
            finished |= choice.get("finish_reason") is not None
    if not done or not finished:
        raise AssertionError(f"incomplete SSE: done={done}, finish_reason={finished}")
    if not any(chunks) and not tool_calls:
        raise AssertionError("empty SSE generation")
    if not usage or usage.get("completion_tokens", 0) <= 0:
        raise AssertionError("missing SSE usage/completion tokens")
    return {"text": "".join(chunks), "usage": usage}


def validate_json(data):
    if "error" in data:
        raise AssertionError(f"JSON error: {data['error']}")
    choices = data.get("choices", [])
    if not choices or choices[0].get("finish_reason") is None:
        raise AssertionError("missing choices/finish_reason")
    msg = choices[0].get("message", {})
    text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
    if not text and not msg.get("tool_calls"):
        raise AssertionError("empty generation")
    usage = data.get("usage", {})
    if usage.get("prompt_tokens", 0) <= 0 or usage.get("completion_tokens", 0) <= 0:
        raise AssertionError(f"invalid usage: {usage}")
    return {"text": text, "usage": usage}


def validate_answer(result, expected):
    text = result["text"].strip()
    last_line = text.splitlines()[-1].strip().strip('`*"') if text else ""
    if last_line != expected:
        raise AssertionError(f"incorrect final answer, expected {expected}: {text[:500]}")
    result["exact_answer_format"] = text == expected
    return result


def validate_dual_lane_trace(text, backend):
    batches = re.findall(rf"KVMEM decode backend={re.escape(backend)} lanes=2\b", text)
    retrieved = {int(lane) for scored, promoted, lane in re.findall(
        r"KVMEM retrieval scored=(\d+) selected=\d+ promoted=(\d+) demoted=\d+ lane=(\d+)", text)
        if int(scored) > 0 and int(promoted) > 0}
    if not batches or retrieved != {0, 1}:
        raise AssertionError(f"no verified sparse dual-lane execution: batches={len(batches)}, "
                             f"retrieved_lanes={sorted(retrieved)}")
    return {"dual_lane_rounds": len(batches), "retrieved_lanes": sorted(retrieved)}


class Suite:
    def __init__(self, args):
        self.args = args
        # Quality runners share this server owner and retain their single-lane default.
        self.args.concurrency = getattr(args, "concurrency", 1)
        self.url = f"http://127.0.0.1:{args.port}"
        self.results = []
        self.proc = None
        self.log = args.output / "server.log"
        self.log_file = None
        self.monitor = None
        self.monitor_file = None
        self.host_monitor_stop = threading.Event()
        self.host_monitor = None

    def check_memory_headroom(self, available_kib, swap_free_kib):
        # Abort the owned test process before repeating the observed WSL-wide
        # OOM. This is a failed run, never a passing or silently skipped case.
        reserve_kib = 3 * 1024**2
        if available_kib + swap_free_kib >= reserve_kib or self.proc.poll() is not None:
            return
        (self.args.output / 'memory-abort.json').write_text(json.dumps({
            'unix_seconds': time.time(), 'pid': self.proc.pid,
            'available_kib': available_kib, 'swap_free_kib': swap_free_kib,
            'required_combined_headroom_kib': reserve_kib}, indent=2), encoding='utf-8')
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.host_monitor_stop.set()

    def monitor_host_memory(self):
        # Linux process RSS/swap plus VM headroom distinguish Host OOM from GPU
        # allocation failure. Sample only the server owned by this runner.
        fields = ('VmRSS', 'RssAnon', 'RssFile', 'VmSwap')
        with (self.args.output / 'host-memory.csv').open('w', encoding='utf-8') as output:
            print('unix_seconds,pid,rss_kib,anon_kib,file_kib,swap_kib,available_kib,swap_free_kib',
                  file=output, flush=True)
            while not self.host_monitor_stop.is_set():
                try:
                    status = dict(line.split(':', 1) for line in
                                  Path(f'/proc/{self.proc.pid}/status').read_text().splitlines())
                    memory = dict(line.split(':', 1) for line in
                                  Path('/proc/meminfo').read_text().splitlines())
                    values = [int(status.get(key, '0 kB').split()[0]) for key in fields]
                    values += [int(memory[key].split()[0]) for key in ('MemAvailable', 'SwapFree')]
                    print(','.join(map(str, [time.time(), self.proc.pid, *values])), file=output, flush=True)
                    self.check_memory_headroom(*values[-2:])
                except FileNotFoundError:
                    break
                self.host_monitor_stop.wait(2)

    def request(self, messages, tokens=64, stream=False, **extra):
        body = {"model": "kvmem-test", "messages": messages, "max_tokens": tokens,
                "temperature": 0, "stream": stream,
                "chat_template_kwargs": {"enable_thinking": False}, **extra}
        if stream:
            body["stream_options"] = {"include_usage": True}
        req = urllib.request.Request(self.url + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.args.request_timeout) as response:
                result = parse_sse(response) if stream else validate_json(json.load(response))
        except urllib.error.HTTPError as exc:
            raise AssertionError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
        result["seconds"] = time.monotonic() - start
        return result

    def case(self, name, fn):
        start = time.monotonic()
        try:
            data = fn()
            result = {"name": name, "status": "passed", "details": data}
        except Exception:
            result = {"name": name, "status": "failed", "error": traceback.format_exc()}
        result["seconds"] = time.monotonic() - start
        self.results.append(result)
        self.save()
        print(f"{result['status'].upper()}: {name} ({result['seconds']:.1f}s)", flush=True)
        if result["status"] == "failed":
            print(result["error"], flush=True)

    def save(self):
        report = {"configuration": {k: str(v) if isinstance(v, Path) else v
                                    for k, v in vars(self.args).items()},
                  "cases": self.results}
        (self.args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        root = ET.Element("testsuite", name="kvmem-e2e", tests=str(len(self.results)),
                          failures=str(sum(r["status"] == "failed" for r in self.results)))
        for result in self.results:
            node = ET.SubElement(root, "testcase", name=result["name"], time=str(result["seconds"]))
            if result["status"] == "failed":
                ET.SubElement(node, "failure").text = result["error"]
        ET.ElementTree(root).write(self.args.output / "junit.xml", encoding="utf-8", xml_declaration=True)

    def server_command(self):
        command = [str(self.args.binary.resolve()), str(self.args.model.resolve()),
                   "--host", "127.0.0.1", "--port", str(self.args.port),
                   "--model-id", "kvmem-test", "--max-context", str(self.args.context),
                   "--kv-dtype", self.args.dtype, "--kv-capacity", "auto",
                   "--kvmem-window-pages", str(self.args.window),
                   "--max-concurrency", str(self.args.concurrency),
                   "--prefill-chunk", str(self.args.chunk), "--host-kv-mib", str(self.args.host_mib),
                   "--pending-timeout-ms", "1800000"]
        if self.args.spec == "mtp":
            command += ["--spec", "mtp", "--draft-tokens", "3", "--lm-head-draft"]
        if self.args.spec == "dflash2":
            command += ["--spec", "dflash2", "--draft-tokens", "7"]
        return command

    def start(self):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", self.args.port))
        command = self.server_command()
        with self.args.binary.open("rb") as binary:
            binary_sha256 = hashlib.file_digest(binary, "sha256").hexdigest()
        model_stat = self.args.model.stat()
        self.log_file = self.log.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(command, stdout=self.log_file, stderr=subprocess.STDOUT,
                                     start_new_session=True, env={**os.environ, "NINFER_KVMEM_TRACE": "1"})
        self.host_monitor = threading.Thread(target=self.monitor_host_memory, daemon=True)
        self.host_monitor.start()
        self.monitor_file = (self.args.output / "gpu.csv").open("w", encoding="utf-8")
        self.monitor = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu", "--format=csv", "-l", "2"],
            stdout=self.monitor_file, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + self.args.startup_timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise AssertionError(f"server exited during startup ({self.proc.returncode}); see {self.log}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as response:
                    if response.status == 200:
                        return {"pid": self.proc.pid, "command": command,
                                "binary_sha256": binary_sha256,
                                "execution_environment": {key: os.environ[key] for key in (
                                    'GGML_CUDA_DISABLE_GRAPHS', 'MALLOC_ARENA_MAX',
                                    'MALLOC_TRIM_THRESHOLD_', 'MALLOC_MMAP_THRESHOLD_')
                                    if key in os.environ},
                                "model_bytes": model_stat.st_size,
                                "model_mtime_ns": model_stat.st_mtime_ns}
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(1)
        raise TimeoutError("server readiness timeout")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=15)
        self.host_monitor_stop.set()
        if self.host_monitor:
            self.host_monitor.join(timeout=5)
        if self.log_file:
            self.log_file.close()
        if self.monitor:
            self.monitor.terminate()
            self.monitor.wait(timeout=10)
        if self.monitor_file:
            self.monitor_file.close()

    def health(self):
        if self.proc.poll() is not None:
            raise AssertionError(f"server exited ({self.proc.returncode})")
        result = self.request([{"role": "user", "content": "Reply with exactly READY."}])
        return validate_answer(result, "READY")

    def lane_messages(self, marker, extra=0):
        messages = self.long_messages(self.args.window * 64 + 1024 + extra)
        text = messages[-1]["content"]
        pivot = len(text) // 4
        # The answer is only in old history, outside the retained recency window.
        # Distinct codes with the same query also expose cross-lane page selection.
        messages[-1]["content"] = (text[:pivot] +
            f"\nThe special recovery code for the Aurora gateway is {marker}.\n" + text[pivot:]).replace(
            "Reply with exactly GATEWAY.",
            "First print the special recovery code for the Aurora gateway on its own line. "
            "Then print every integer from 1 to 2000 "
            "in order, separated by commas. Do not abbreviate or add explanations.")
        return messages

    def parallel_pair(self, round_index):
        # Compare the same greedy prompts alone and batched. Reuse the two slots
        # with different queries next round to expose stale lane-local retrieval.
        markers = [f"AMBER-{round_index}-7193", f"COBALT-{round_index}-8426"]
        prompts = [self.lane_messages(marker, i * 512) for i, marker in enumerate(markers)]
        baseline = [self.request(prompt, tokens=512, stream=True) for prompt in prompts]
        begin = self.log.stat().st_size
        barrier = threading.Barrier(2)
        def generate(index):
            barrier.wait(timeout=10)
            return self.request(prompts[index], tokens=512, stream=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(generate, i) for i in range(2)]
            paired = [future.result() for future in futures]
        trace = self.log.read_text(encoding="utf-8", errors="replace")[begin:]
        evidence = validate_dual_lane_trace(trace, self.args.spec)
        for i, result in enumerate(paired):
            if result["usage"]["prompt_tokens"] <= self.args.window * 64:
                raise AssertionError("paired request did not cross the sparse window")
            if result["usage"]["completion_tokens"] < 256:
                raise AssertionError("paired decode did not exercise repeated page growth")
            if markers[i] not in result["text"] or markers[1 - i] in result["text"]:
                raise AssertionError(f"lane answer contaminated or incorrect: {result['text'][:200]}")
            if result["text"] != baseline[i]["text"]:
                raise AssertionError(f"greedy batch output differs from isolated output: "
                                     f"lane={i}, isolated={baseline[i]['text'][:800]!r}, "
                                     f"batched={result['text'][:800]!r}")
        return {**evidence, "isolated": baseline, "concurrent": paired,
                "exact_greedy_matches": 2}

    def cancel_one_lane(self):
        begin = self.log.stat().st_size
        barrier = threading.Barrier(2)
        def survivor():
            barrier.wait(timeout=10)
            return self.request(self.lane_messages("SURVIVOR-9281", 512), tokens=768, stream=True)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(survivor)
            connection = http.client.HTTPConnection("127.0.0.1", self.args.port,
                                                    timeout=self.args.request_timeout)
            response = None
            try:
                barrier.wait(timeout=10)
                body = {"model": "kvmem-test", "messages": self.lane_messages("CANCELLED-1467"),
                        "temperature": 0, "stream": True, "max_tokens": 2048,
                        "chat_template_kwargs": {"enable_thinking": False}}
                connection.request("POST", "/v1/chat/completions", json.dumps(body),
                                   {"Content-Type": "application/json"})
                response = connection.getresponse()
                if response.status != 200:
                    raise AssertionError(f"cancellation request HTTP {response.status}")
                deadline = time.monotonic() + self.args.request_timeout
                while True:
                    trace = self.log.read_text(encoding="utf-8", errors="replace")[begin:]
                    if f"KVMEM decode backend={self.args.spec} lanes=2" in trace:
                        break
                    if future.done() or time.monotonic() >= deadline:
                        raise AssertionError("requests did not overlap before cancellation")
                    time.sleep(0.05)
            finally:
                if response is not None:
                    response.close()
                connection.close()
            recovered = self.health()
            if future.done():
                raise AssertionError("cancelled slot was not reused while the other lane was active")
            survivor_result = future.result()
        if "SURVIVOR-9281" not in survivor_result["text"] or "CANCELLED-1467" in survivor_result["text"]:
            raise AssertionError("surviving lane returned incorrect content")
        if survivor_result["usage"]["completion_tokens"] < 512:
            raise AssertionError("surviving lane stopped prematurely")
        trace = self.log.read_text(encoding="utf-8", errors="replace")[begin:]
        if not re.search(r"cancelled during transport.*HTTP 499", trace):
            raise AssertionError("engine did not confirm cancellation of the disconnected request")
        return {"survivor": survivor_result, "after_cancel": recovered}

    def disconnect(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.args.port, timeout=60)
        body = {"model": "kvmem-test", "messages": [{"role": "user", "content": "Count from 1 to 1000."}],
                "stream": True, "max_tokens": 1024}
        try:
            connection.request("POST", "/v1/chat/completions", json.dumps(body),
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                raise AssertionError(f"disconnect test HTTP {response.status}")
            # Read actual generated data, then disconnect while work remains.
            for line in response:
                if b'"content"' in line or b'"reasoning_content"' in line:
                    break
            else:
                raise AssertionError("stream ended before disconnect")
        finally:
            connection.close()
        return self.health()

    def cancel_query_replay(self):
        if not self.args.window or self.args.context < 65536:
            raise ValueError("replay cancellation needs sparse KV and at least 64K context")
        offset = self.log.stat().st_size
        messages = self.long_messages(4096)
        messages += [{"role": "assistant", "content": "History received."},
                     {"role": "user", "content": "Check the tool output, then reply READY."},
                     {"role": "assistant", "content": None, "tool_calls": [
                         {"id": "cancel_probe", "type": "function", "function": {
                             "name": "read_status", "arguments": "{}"}}]},
                     {"role": "tool", "tool_call_id": "cancel_probe", "content":
                         self.long_messages(int(self.args.context * .70))[1]["content"]}]
        body = {"model": "kvmem-test", "messages": messages, "stream": True,
                "max_tokens": 128, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
        connection = http.client.HTTPConnection("127.0.0.1", self.args.port, timeout=60)
        try:
            connection.request("POST", "/v1/chat/completions", json.dumps(body),
                               {"Content-Type": "application/json"})
            deadline = time.monotonic() + self.args.request_timeout
            while time.monotonic() < deadline:
                with self.log.open("rb") as log:
                    log.seek(offset)
                    trace = log.read().decode(errors="replace")
                if ("KVMEM retrieval scored=" in trace and
                        "KVMEM probe sample_history=" in trace):
                    match = re.search(r"query source=last_user begin=(\d+) end=\d+ prompt=(\d+)", trace)
                    if not match or int(match[2]) - int(match[1]) < 32768:
                        raise AssertionError("fixture did not create a long replay suffix")
                    break
                if self.proc.poll() is not None:
                    raise AssertionError("server died before query replay")
                time.sleep(.02)
            else:
                raise AssertionError("query replay was never reached")
            # The audit is emitted inside the first replay step, so cancellation
            # must release partially replayed state rather than only the probe.
            connection.sock.shutdown(socket.SHUT_RDWR)
        finally:
            connection.close()
        started = time.monotonic()
        result = self.health()
        seconds = time.monotonic() - started
        if seconds > 5:
            raise AssertionError(f"cancelled replay blocked the next request for {seconds:.2f}s")
        return {"next_request_seconds": seconds, "replay_suffix_tokens": int(match[2]) - int(match[1]),
                "response": result}

    def long_messages(self, target):
        # Calibrated on Qwen's tokenizer; actual depth is asserted from response usage.
        filler = ("The industrial gateway aggregates sensor telemetry across LoRaWAN and Modbus "
                  "links, normalizes the payloads, and forwards them to the cloud platform with "
                  "retries and local buffering. ")
        return [{"role": "system", "content": "Follow the user's instructions precisely."},
                {"role": "user", "content": filler * int(target / 33.59) +
                 "\nReply with exactly GATEWAY."}]

    def depth(self, target):
        result = self.request(self.long_messages(target), tokens=128)
        actual = result["usage"]["prompt_tokens"]
        if not target * .85 <= actual <= target * 1.15:
            raise AssertionError(f"wrong depth: target={target}, actual={actual}")
        return validate_answer(result, "GATEWAY")

    def replay_sampling_history(self):
        offset = self.log.stat().st_size
        # Tokenizer calibration is approximate: +1024 alone can still fall below
        # a 96K window. Verify actual usage before interpreting the probe audit.
        target = min(self.args.window * 64 * 11 // 10 + 1024, self.args.context - 2048)
        result = self.request(self.long_messages(target),
                              tokens=128, presence_penalty=2, frequency_penalty=2)
        if result["usage"]["prompt_tokens"] <= self.args.window * 64:
            raise AssertionError("sampling-history fixture did not cross the configured KV window")
        with self.log.open("rb") as log:
            log.seek(offset)
            new_log = log.read().decode(errors="replace")
        if re.findall(r"KVMEM probe sample_history=(\d+)", new_log) != ["0"]:
            raise AssertionError("unpublished probe modified sampling history or audit missing: " +
                                 "\n".join(line for line in new_log.splitlines() if "sample_history" in line))
        return validate_answer(result, "GATEWAY")

    def replay_service_budget(self):
        if not self.args.window:
            raise ValueError("query replay budget needs sparse KV")
        offset = self.log.stat().st_size
        messages = self.long_messages(self.args.window * 64 + 1024)
        messages += [{"role": "assistant", "content": "History received."},
                     {"role": "user", "content": "Read the tool output and reply READY."},
                     {"role": "assistant", "content": None, "tool_calls": [
                         {"id": "budget_probe", "type": "function", "function": {
                             "name": "read_status", "arguments": "{}"}}]},
                     {"role": "tool", "tool_call_id": "budget_probe", "content":
                         self.long_messages(8192)[1]["content"]}]
        # A one-token limit leaves no unused decode budget to hide omitted replay
        # steps. The real tool suffix must span multiple Scheduler boundaries.
        result = self.request(messages, tokens=1)
        with self.log.open("rb") as log:
            log.seek(offset)
            trace = log.read().decode(errors="replace")
        match = re.search(r"query source=last_user begin=(\d+) end=\d+ prompt=(\d+)", trace)
        if not match or int(match[2]) - int(match[1]) <= 2 * self.args.chunk:
            raise AssertionError("fixture did not exercise multiple replay steps")
        if "KVMEM probe sample_history=" not in trace:
            raise AssertionError("query probe/replay was not reached")
        if result["usage"]["completion_tokens"] != 1:
            raise AssertionError("one-token service budget was not exercised")
        return {"replay_suffix_tokens": int(match[2]) - int(match[1]), "response": result}

    def needle(self):
        target = min(self.args.window * 128 + 1024, self.args.context - 1024)
        messages = self.long_messages(target)
        text = messages[-1]["content"]
        messages[-1]["content"] = (
            text[:len(text) // 4] +
            "\nThe special recovery code for the Aurora gateway is ORCHID-7392.\n" +
            text[len(text) // 4:].replace("Reply with exactly GATEWAY.",
                "What is the special recovery code for the Aurora gateway? Reply only with the code."))
        result = self.request(messages, tokens=64)
        return validate_answer(result, "ORCHID-7392")

    def long_decode(self):
        # Leave enough margin for tokenizer calibration error even with a 96K window.
        target = min(self.args.window * 64 * 11 // 10 + 1024, self.args.context - 2048)
        messages = self.long_messages(target)
        messages[-1]["content"] = messages[-1]["content"].replace(
            "Reply with exactly GATEWAY.",
            "Print every integer from 1 to 1000 in order, separated by commas. Do not abbreviate.")
        result = self.request(messages, tokens=768, stream=True)
        if result["usage"]["prompt_tokens"] <= self.args.window * 64:
            raise AssertionError("long decode did not actually cross the configured KV window")
        if result["usage"]["completion_tokens"] < 512:
            raise AssertionError("decode stopped before exercising repeated page growth")
        return result

    def host_capacity_rejection(self):
        body = {"model": "kvmem-test", "messages": self.long_messages(self.args.context - 1024),
                "max_tokens": 128, "temperature": 0, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(self.url + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                raise AssertionError(f"insufficient Host capacity was admitted: HTTP {response.status}")
        except urllib.error.HTTPError as exc:
            payload = json.loads(exc.read())
            if exc.code != 400 or "request reservation exceeds Engine shared KV capacity" not in str(payload):
                raise AssertionError(f"unexpected capacity rejection: HTTP {exc.code}: {payload}") from exc
            return {"status": exc.code, "payload": payload, "seconds": time.monotonic() - start}

    def check_retrieval(self):
        text = self.log.read_text(encoding="utf-8", errors="replace")
        captures = re.findall(r"KVMEM capture .*query_norm=([0-9.e+\-]+)", text)
        placements = re.findall(r"KVMEM retrieval scored=(\d+) selected=(\d+) promoted=(\d+)", text)
        if not captures or not all(float(norm) > 0 for norm in captures):
            raise AssertionError("missing or empty real query captures")
        if self.args.profile != "smoke" and not any(int(s) > 0 and int(p) > 0 for s, _, p in placements):
            raise AssertionError("no scored historical page was actually restored from Host KV")
        if any(int(selected) > self.args.window for _, selected, _ in placements):
            raise AssertionError("retrieval exceeds configured window")
        return {"captures": len(captures), "retrievals": len(placements)}

    def dialogue(self):
        messages = [{"role": "system", "content": "You are a concise assistant. Use the supplied tool results."},
                    {"role": "user", "content": "Remember the project code RAVEN-5839."}]
        details = []
        tools = [{"type": "function", "function": {"name": "read_status", "description": "Read device status",
                 "parameters": {"type": "object", "properties": {"device": {"type": "string"}}}}}]
        for turn in range(4):
            messages += [{"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call_{turn}", "type": "function", "function": {
                    "name": "read_status", "arguments": '{"device":"gateway"}'}}]},
                {"role": "tool", "tool_call_id": f"call_{turn}", "content": "Gateway online. " * (64 + turn * 31)},
                {"role": "user", "content": "What is the project code? Reply only with the code."}]
            result = self.request(messages, tokens=max(128, self.args.context - 2048), stream=True, tools=tools)
            validate_answer(result, "RAVEN-5839")
            details.append(result)
            messages.append({"role": "assistant", "content": result["text"]})
        return details

    def concurrent_host_claims(self):
        """Two future contexts fit Device windows but not the shared Host budget.

        Cancel well before materializing the requested output. This exercises
        admission claims rather than risking an allocation failure at full depth.
        """
        def open_stream(index):
            body = {"model": "kvmem-test", "messages": [
                {"role": "system", "content": "Treat this archive as irrelevant context:\n" +
                    (" alpha beta gamma delta epsilon." * 800)},
                {"role": "user", "content": f"Request {index}: Count every integer from 1 to 100000 "
                    "in order, separated by commas. Do not skip or abbreviate. No commentary or code blocks."}],
                "max_tokens": self.args.context, "temperature": 0, "stream": True,
                "chat_template_kwargs": {"enable_thinking": False}}
            request = urllib.request.Request(self.url + "/v1/chat/completions",
                data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            return urllib.request.urlopen(request, timeout=30)

        def first_content(response):
            for raw in response:
                if not raw.startswith(b"data:"): continue
                payload = raw[5:].strip()
                if payload == b"[DONE]": raise AssertionError("claim probe finished without output")
                event = json.loads(payload)
                if "error" in event: raise AssertionError(f"SSE error: {event['error']}")
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content") or delta.get("reasoning_content"): return
            raise AssertionError("claim probe stream ended without content")

        first, second = open_stream(0), None
        ready, errors, stamps = threading.Event(), [], {}
        try:
            first_content(first)
            stamps["first_ready"] = time.monotonic()
            second = open_stream(1)
            def receive():
                try:
                    first_content(second)
                    stamps["second_ready"] = time.monotonic()
                    ready.set()
                except Exception as error:
                    errors.append(repr(error))
            worker = threading.Thread(target=receive, daemon=True)
            worker.start()
            admitted_early = ready.wait(4)
            stamps["first_cancelled"] = time.monotonic()
            first.close()
            assert not admitted_early, "second full-future Host claim admitted before first release"
            assert ready.wait(20), f"queued Host claim did not resume after release: {errors}"
            worker.join(timeout=2)
            assert not errors, errors
            return {"second_admitted_before_first_cancel": admitted_early,
                    "resume_after_cancel_seconds": stamps["second_ready"] - stamps["first_cancelled"],
                    "logical_context": self.args.context, "shared_host_mib": self.args.host_mib,
                    "requested_output_tokens_each": self.args.context,
                    "bounded_probe": "both streams cancelled before large page growth"}
        finally:
            first.close()
            if second is not None: second.close()

    def check_log(self):
        for name in ('memory-abort.json', 'windows-memory-abort.json'):
            if (self.args.output / name).exists():
                raise AssertionError(f'memory guard aborted server: see {name}')
        text = self.log.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if self.args.profile == "capacity":
            lines = [line for line in lines if "request reservation exceeds Engine shared KV capacity" not in line]
        bad = [line for line in lines if any(term in line.lower() for term in
               ("error", "bad_alloc", "inconsistent", "terminate called", "illegal memory", "assertion"))]
        if bad:
            raise AssertionError("engine diagnostics:\n" + "\n".join(bad[-30:]))
        return {"bytes": len(text)}

    def run(self):
        self.case("startup", self.start)
        if self.results[-1]["status"] == "failed":
            return
        self.case("json-generation", self.health)
        if self.args.profile == "concurrent-host-claims":
            self.case("two-full-future-host-claims", self.concurrent_host_claims)
            self.case("generation-after-released-host-claims", self.health)
            return
        if self.args.profile == "concurrency":
            for repeat in range(self.args.repeats):
                self.case(f"parallel-sparse-pair-{repeat}", lambda r=repeat: self.parallel_pair(r))
            self.case("cancel-one-lane-and-reuse", self.cancel_one_lane)
            self.case("final-health", self.health)
            return
        if self.args.profile == "replay-budget":
            self.case("query-replay-service-budget", self.replay_service_budget)
            self.case("generation-after-replay-budget", self.health)
            return
        if self.args.profile == "replay-cancel":
            self.case("cancel-long-query-replay", self.cancel_query_replay)
            self.case("generation-after-replay-cancel", self.health)
            return
        if self.args.profile == "penalty":
            self.case("query-probe-does-not-change-sampling-history", self.replay_sampling_history)
            self.case("generation-after-penalty-query", self.health)
            return
        if self.args.profile == "capacity":
            self.case("host-capacity-rejected-before-execution", self.host_capacity_rejection)
            self.case("generation-after-rejection", self.health)
            return
        self.case("sse-generation", lambda: self.request(
            [{"role": "user", "content": "Reply with exactly STREAM."}], stream=True))
        self.case("tool-history-large-max-tokens", self.dialogue)
        self.case("disconnect-and-recover", self.disconnect)
        if self.args.profile != "smoke":
            if self.args.profile == "long":
                self.case("cancel-long-query-replay", self.cancel_query_replay)
            if self.args.window:
                self.case("query-probe-does-not-change-sampling-history", self.replay_sampling_history)
                self.case("query-replay-service-budget", self.replay_service_budget)
            window = self.args.window * 64 or self.args.context // 2
            depths = ([8192, 32768, 65536, 98304, 131072, 196608, 250000]
                      if self.args.profile == "long" else
                      [max(512, window // 2), window + 512, min(window * 2 + 1024, self.args.context - 1024)])
            for repeat in range(self.args.repeats):
                for depth in depths:
                    if depth + 128 < self.args.context:
                        self.case(f"depth-{depth}-repeat-{repeat}", lambda d=depth: self.depth(d))
                self.case(f"small-after-large-{repeat}", self.health)
            self.case("historical-needle-recall", self.needle)
            self.case("long-decode-page-growth", self.long_decode)
        if self.args.window:
            self.case("retrieval-is-live", self.check_retrieval)
        self.case("final-health", self.health)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, default=Path("build/apps/ninfer-serve"))
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=8095)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--context", type=int, default=16384)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--host-mib", type=int, default=12288)
    p.add_argument("--dtype", choices=["bf16", "int8", "fp8", "nvfp4", "k8v4"], default="int8")
    p.add_argument("--spec", choices=["none", "mtp", "dflash2"], default="mtp")
    p.add_argument("--concurrency", type=int, choices=[1, 2], default=1)
    p.add_argument("--profile", choices=["smoke", "regression", "long", "capacity", "penalty", "replay-cancel", "replay-budget", "concurrency", "concurrent-host-claims"], default="regression")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--startup-timeout", type=float, default=600)
    p.add_argument("--request-timeout", type=float, default=1800)
    args = p.parse_args()
    if args.profile == "concurrent-host-claims" and (
            args.concurrency != 2 or args.spec != "none" or args.window == 0 or args.context < 65536):
        p.error("concurrent-host-claims requires two lanes, ordinary decoding, sparse KV and >=64K context")
    if args.profile == "concurrency" and (args.concurrency != 2 or args.window == 0 or
                                           args.context < args.window * 64 + 4096):
        p.error("concurrency profile requires two lanes and context >= window * 64 + 4096")
    args.output.mkdir(parents=True, exist_ok=False)
    suite = Suite(args)
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        suite.run()
    except KeyboardInterrupt as exc:
        suite.results.append({"name": "interrupted", "status": "failed", "seconds": 0, "error": str(exc)})
    finally:
        suite.case("shutdown", suite.stop)
        if suite.log.exists():
            suite.case("no-engine-errors", suite.check_log)
        suite.save()
    return int(any(r["status"] == "failed" for r in suite.results))


if __name__ == "__main__":
    raise SystemExit(main())
