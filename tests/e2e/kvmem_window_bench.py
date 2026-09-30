#!/usr/bin/env python3
"""Paired request benchmark; preserve timings, actual copies and failures."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time
import urllib.request

from kvmem_suite import Suite, parse_sse, validate_dual_lane_trace


def read_since(path, offset):
    with path.open('rb') as source:
        source.seek(offset)
        return source.read().decode('utf-8', errors='replace')


def summarize_placements(text):
    groups = {}
    for line in text.splitlines():
        if not line.startswith('KVPLACEMENT '):
            continue
        values = dict(item.split('=', 1) for item in line.split()[1:])
        key = f"{values['phase']}/planes-{values['planes']}"
        group = groups.setdefault(key, {'calls': 0, 'demoted': 0, 'promoted': 0,
            'd2h_pages': 0, 'd2h_bytes': 0, 'h2d_bytes': 0,
            'd2h_submit_wait_ms': 0.0, 'h2d_submit_wait_ms': 0.0, 'total_ms': 0.0})
        group['calls'] += 1
        for field in group:
            if field != 'calls':
                group[field] += float(values[field]) if field.endswith('_ms') else int(values[field])
    if not groups:
        raise AssertionError('missing actual KV placement measurements')
    return groups


def validate_execution_trace(trace, backend, concurrency, records=()):
    if concurrency == 2:
        return validate_dual_lane_trace(trace, backend)
    if concurrency != 1:
        raise AssertionError('window benchmark supports one or two lanes')
    rounds = [line for line in trace.splitlines()
              if line.startswith(f'KVMEM decode backend={backend} lanes=')]
    # The engine only emits the KVMEM decode diagnostic for batches > 1.
    # Positive single-lane evidence comes from scheduler/batch telemetry.
    if any(line.rsplit('=', 1)[1] != '1' for line in rounds):
        raise AssertionError('single-lane comparison requires exclusively one-lane decode')
    decoded = [item for item in records if item.get('event') == 'throughput'
               and item.get('decode_batch', {}).get('rounds', 0) > 0]
    if not decoded or any(item['decode_batch']['row_rounds'] != item['decode_batch']['rounds']
                          or item['scheduler']['running'] > 1 for item in decoded):
        raise AssertionError('missing exclusively single-lane scheduler evidence')
    retrieved = [dict(item.split('=', 1) for item in line.split()[2:])
                 for line in trace.splitlines() if line.startswith('KVMEM retrieval ')]
    if len(retrieved) != 2 or any(int(item.get('promoted', 0)) == 0 or
                                  item.get('lane') != '0' for item in retrieved):
        raise AssertionError('single-lane comparison requires two separate sparse retrievals')
    return {'single_lane_telemetry_intervals':len(decoded), 'dual_lane_rounds':0,
            'retrieved_lanes':[0]}


class WindowBench(Suite):
    def server_command(self):
        return super().server_command() + ['--request-log-jsonl', str(self.args.output / 'requests.jsonl'),
                                           '--log-level', 'debug']

    def monitor_host_memory(self):
        active = {'pid': self.proc.pid, 'port': self.args.port, 'window': self.args.window,
                  'output': str(self.args.output.resolve()),
                  'start_ticks': Path(f'/proc/{self.proc.pid}/stat').read_text().split()[21]}
        self.args.active_file.write_text(json.dumps(active))
        super().monitor_host_memory()

    def messages(self, depth, marker):
        content = self.long_messages(depth)[1]['content'].replace('Reply with exactly GATEWAY.', '')
        pivot = len(content) // 4
        content = content[:pivot] + f'\nThe special Aurora recovery code is {marker}.\n' + content[pivot:]
        return [{'role': 'system', 'content': "Follow the user's instructions precisely."},
                {'role': 'user', 'content': content},
                {'role': 'assistant', 'content': 'History received.'},
                {'role': 'user', 'content': 'First print the special Aurora recovery code on its own line. '
                 'Then print every integer from 1 to 100000 in order, separated by commas. '
                 'Do not abbreviate or add explanations. Continue until the output limit.'}]

    def measured_request(self, messages, barrier):
        body = {'model': 'kvmem-test', 'messages': messages, 'max_tokens': self.args.output_tokens,
                'temperature': 0, 'stream': True, 'stream_options': {'include_usage': True},
                'chat_template_kwargs': {'enable_thinking': False}}
        request = urllib.request.Request(self.url + '/v1/chat/completions',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        barrier.wait(timeout=30)
        begin = time.monotonic()
        first = last = None
        with urllib.request.urlopen(request, timeout=self.args.request_timeout) as response:
            def lines():
                nonlocal first, last
                for line in response:
                    if line.startswith(b'data:') and line[5:].strip() != b'[DONE]':
                        event = json.loads(line[5:])
                        if any(c.get('delta', {}).get('content') or c.get('delta', {}).get('reasoning_content')
                               for c in event.get('choices', [])):
                            last = time.monotonic()
                            if first is None:
                                first = last
                    yield line
            result = parse_sse(lines())
        if first is None or last is None:
            raise AssertionError('stream contains no text timestamps')
        result.update(client_ttft_seconds=first-begin, client_total_seconds=time.monotonic()-begin,
                      client_text_stream_seconds=last-first)
        return result

    def pair(self, depth):
        markers = ['BENCH-ALPHA-7193', 'BENCH-BETA-8426']
        prompts = [self.messages(depth, marker) for marker in markers]
        fixture = json.dumps(prompts, ensure_ascii=False).encode()
        fixture_path = self.args.output / f'fixture-{depth}.json'
        fixture_path.write_bytes(fixture)
        log_offset = self.log.stat().st_size
        request_log = self.args.output / 'requests.jsonl'
        request_offset = request_log.stat().st_size
        barrier = threading.Barrier(2)
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [pool.submit(self.measured_request, prompt, barrier) for prompt in prompts]
            results = [future.result() for future in pending]
        elapsed = time.monotonic() - started
        trace = read_since(self.log, log_offset)
        records = [json.loads(line) for line in read_since(request_log, request_offset).splitlines()]
        # Preserve client measurements even when a subsequent assertion fails.
        (self.args.output / f'pair-{depth}-raw.json').write_text(json.dumps({
            'pair_wall_seconds': elapsed, 'responses': results, 'records': records,
            'fixture_sha256': hashlib.sha256(fixture).hexdigest()}, indent=2))
        evidence = validate_execution_trace(trace, self.args.spec, self.args.concurrency, records)
        placements = summarize_placements(trace)
        done = [record for record in records if record.get('event') == 'request_done']
        if len(done) != 2:
            raise AssertionError(f'expected two complete engine timing records, found {len(done)}')
        for index, result in enumerate(results):
            if not depth * .95 <= result['usage']['prompt_tokens'] < self.args.context:
                raise AssertionError(f'fixture depth outside calibrated range: {result["usage"]}')
            if result['usage']['completion_tokens'] != self.args.output_tokens:
                raise AssertionError('generation ended before the benchmark output limit')
            result['needle_first_line_correct'] = result['text'].splitlines()[0].strip() == markers[index]
            result['contains_other_lane_marker'] = markers[1-index] in result['text']
        for record in done:
            count, timing = record['result'], record['timings_seconds']
            if count['prefix_cache_hit_tokens'] != 0:
                raise AssertionError('cold-prompt comparison unexpectedly reused a prefix')
            record['measured_rates'] = {
                'prefill_tokens_per_second': count['computed_prefill_tokens'] / timing['prefill'],
                'decode_tokens_per_second': (count['completion_tokens']-1) / timing['decode']}
        return {'depth_target': depth, 'fixture_sha256': hashlib.sha256(fixture).hexdigest(),
                'pair_wall_seconds': elapsed, **evidence, 'responses': results,
                'engine_requests': done, 'placement_totals': placements,
                'timing_scope': 'copy = CPU submission plus existing transfer-stream wait; '
                                'placement total includes mapping/bookkeeping; not GPU-only PCIe duration'}

    def run(self):
        self.case('startup', self.start)
        if self.results[-1]['status'] != 'passed':
            return
        self.case('warmup-generation', self.health)
        for depth in self.args.depths:
            self.case(f'paired-depth-{depth}', lambda d=depth: self.pair(d))
            self.case(f'health-after-{depth}', self.health)

    def stop(self):
        super().stop()
        if self.args.active_file.exists():
            active = json.loads(self.args.active_file.read_text())
            if self.proc and active['pid'] == self.proc.pid:
                self.args.active_file.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=Path('build/apps/ninfer-serve'))
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window', type=int, choices=[576, 800, 1152], required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--active-file', type=Path, required=True)
    parser.add_argument('--depths', nargs='+', type=int, default=[131072, 262144])
    parser.add_argument('--output-tokens', type=int, default=2048)
    parser.add_argument('--concurrency', type=int, choices=[1, 2], default=2,
                        help='one queues the same two clients serially; two batches active lanes')
    parser.set_defaults(context=262144, chunk=1024, host_mib=18432, dtype='int8', spec='mtp',
                        startup_timeout=600, request_timeout=1800, profile='window-bench')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['NINFER_KVMEM_TRANSFER_TRACE'] = '1'
    suite = WindowBench(args)
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        suite.run()
    except KeyboardInterrupt as exc:
        suite.results.append({'name':'interrupted', 'status':'failed', 'seconds':0, 'error':str(exc)})
    finally:
        suite.case('shutdown', suite.stop)
        if suite.log.exists():
            suite.case('no-engine-errors', suite.check_log)
        suite.save()
    return int(any(result['status'] != 'passed' for result in suite.results))


if __name__ == '__main__':
    raise SystemExit(main())
