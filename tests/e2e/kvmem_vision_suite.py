#!/usr/bin/env python3
"""Real-model media replay and dual-lane regression, using the KVMem server owner."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import re
import signal
import socket
import struct
import threading
import time
import zlib

from kvmem_suite import Suite, validate_answer, validate_dual_lane_trace


def color_image(color, size=800):
    rgb = {'RED': (255, 0, 0), 'BLUE': (0, 0, 255)}[color]
    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data))
    rows = (b'\0' + bytes(rgb) * size) * size
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', size, size, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b'')
    return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png).decode()}}


def text(value):
    return {'type': 'text', 'text': value}


class VisionSuite(Suite):
    def monitor_host_memory(self):
        self.active_file = self.args.output.parent / 'active.json'
        self.active_file.write_text(json.dumps({'pid': self.proc.pid, 'window': self.args.window,
            'output': str(self.args.output.resolve()),
            'start_ticks': Path(f'/proc/{self.proc.pid}/stat').read_text().split()[21]}))
        super().monitor_host_memory()

    def stop(self):
        super().stop()
        active = getattr(self, 'active_file', None)
        if active and active.exists() and json.loads(active.read_text())['pid'] == self.proc.pid:
            active.unlink()

    def server_command(self):
        return super().server_command() + ['--vision', '--request-log-jsonl',
            str(self.args.output / 'requests.jsonl'), '--log-level', 'debug']

    def request(self, *args, **kwargs):
        result = super().request(*args, **kwargs)
        # Preserve each completed response before assertions, including failed answers.
        with self.response_lock:
            with (self.args.output / 'responses.jsonl').open('a', encoding='utf-8') as output:
                output.write(json.dumps(result) + '\n')
        return result

    def trace(self, offset):
        with self.log.open('rb') as source:
            source.seek(offset)
            return source.read().decode(errors='replace')

    def filler(self, tokens):
        return self.long_messages(tokens)[1]['content'].replace('Reply with exactly GATEWAY.', '')

    def prompt(self, color, mode='short', count=False, size=800):
        question = ('What is the color of the supplied image? First print its color in uppercase. '
                    'Then print all integers from 1 to 100000, comma separated, without abbreviation.'
                    if count else 'What is the color of the supplied image? Reply with exactly one uppercase word.')
        image = color_image(color, size)
        depth = self.args.window * 64 + 2048 if self.args.window else 6144
        system = {'role': 'system', 'content': 'Follow the user instructions precisely. Do not explain.'}
        if mode == 'query-image':
            return [system, {'role': 'user', 'content': self.filler(depth)},
                    {'role': 'assistant', 'content': 'History received.'},
                    {'role': 'user', 'content': [image, text(question)]}]
        if mode == 'history-image':
            return [system, {'role': 'user', 'content': [image, text('Remember this image.')]},
                    {'role': 'assistant', 'content': 'Image received.'},
                    {'role': 'user', 'content': self.filler(depth)},
                    {'role': 'assistant', 'content': 'History received.'},
                    {'role': 'user', 'content': question}]
        if mode == 'tool-tail':
            return [system, {'role': 'user', 'content': [image, text(question)]},
                    {'role': 'assistant', 'content': None, 'tool_calls': [
                        {'id': 'call_notes', 'type': 'function', 'function': {'name': 'read_notes', 'arguments': '{}'}}]},
                    {'role': 'tool', 'tool_call_id': 'call_notes',
                     'content': self.filler(depth) + '\nNow answer the user image question.'}]
        return [system, {'role': 'user', 'content': [image, text(question)]}]

    def color_case(self, mode, color):
        offset = self.log.stat().st_size
        result = self.request(self.prompt(color, mode), tokens=16)
        validate_answer(result, color)
        trace = self.trace(offset)
        if self.args.window and mode != 'short':
            assert result['usage']['prompt_tokens'] > self.args.window * 64
            spans = re.findall(r'KVMEM replay begin=(\d+) end=(\d+) tokens=(\d+)', trace)
            assert spans, 'No completed query replay'
            result['replay_spans'] = spans
            if mode == 'query-image':
                query = re.findall(r'KVMEM query source=last_user begin=(\d+) end=(\d+)', trace)
                assert query and int(query[-1][1]) - int(query[-1][0]) > 512, 'Did not test query clipping inside image'
            if mode == 'tool-tail':
                assert int(spans[-1][2]) > self.args.window * 64, 'Did not replay a tail beyond the window'
        return result

    def cache_suffix(self):
        prompt = self.prompt('RED')
        first = self.request(prompt, tokens=16)
        validate_answer(first, 'RED')
        prompt += [{'role': 'assistant', 'content': first['text']},
                   {'role': 'user', 'content': 'Repeat the image color, exactly one uppercase word.'}]
        second = self.request(prompt, tokens=16)
        validate_answer(second, 'RED')
        assert second['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0) > 0, 'No media prefix reuse observed'
        return [first, second]

    def multiple_images(self):
        prompt = [{'role': 'user', 'content': [color_image('BLUE'), color_image('RED'),
                  text('Return the colors of the two images in order, uppercase, separated by a space. Nothing else.')]}]
        result = self.request(prompt, tokens=16)
        return validate_answer(result, 'BLUE RED')

    def multiple_images_replay(self):
        prompt = self.prompt('BLUE', 'query-image')
        prompt[-1]['content'] = [color_image('BLUE'), color_image('RED'),
            text('Return the colors of the two images in order, uppercase, separated by a space. Nothing else.')]
        offset = self.log.stat().st_size
        result = self.request(prompt, tokens=16)
        validate_answer(result, 'BLUE RED')
        if self.args.window:
            assert 'KVMEM vision replay ' in self.trace(offset)
        return result

    def video_route(self):
        # A still PNG is decoded by the native video route into one temporal item.
        # This qualifies video coordinate plumbing, not motion understanding/codecs.
        prompt = self.prompt('BLUE', 'query-image')
        url = color_image('BLUE')['image_url']['url']
        prompt[-1]['content'] = [{'type': 'video_url', 'video_url': url},
            text('What is the color of this video? Reply with exactly one uppercase word.')]
        offset = self.log.stat().st_size
        result = self.request(prompt, tokens=16)
        validate_answer(result, 'BLUE')
        if self.args.window:
            assert 'KVMEM vision replay ' in self.trace(offset)
        return result

    def parallel_images(self):
        prompts = [self.prompt(color, 'query-image', count=True) for color in ('RED', 'BLUE')]
        baseline = [self.request(prompt, tokens=1024, stream=True) for prompt in prompts]
        offset = self.log.stat().st_size
        barrier = threading.Barrier(2)
        def one(i):
            barrier.wait(timeout=30)
            return self.request(prompts[i], tokens=1024, stream=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            paired = list(pool.map(one, range(2)))
        for i, color in enumerate(('RED', 'BLUE')):
            assert paired[i]['text'].strip().startswith(color), paired[i]['text'][:80]
            assert paired[i]['text'] == baseline[i]['text'], 'Serial/parallel greedy output changed'
            assert paired[i]['usage']['completion_tokens'] == 1024, 'Early output termination'
        trace = self.trace(offset)
        if self.args.window:
            evidence = validate_dual_lane_trace(trace, self.args.spec)
        else:
            evidence = {'dense_control': True}
        return {'baseline': baseline, 'parallel': paired, 'trace': evidence}

    def oversized_media(self):
        try:
            self.request(self.prompt('BLUE', size=2560), tokens=16)
        except AssertionError as exc:
            assert str(exc).startswith('HTTP 400:'), str(exc)
            assert 'media_budget_exceeded' in str(exc), str(exc)
            assert 'KVMem window cannot hold a complete media group' in str(exc), str(exc)
            return {'expected_capacity_rejection': str(exc)}
        raise AssertionError('Oversized image was not rejected')

    def cancel_and_reuse(self):
        offset = self.log.stat().st_size
        body = {'model': 'kvmem-test', 'messages': self.prompt('BLUE', 'tool-tail'),
                'max_tokens': 1024, 'stream': True, 'temperature': 0,
                'chat_template_kwargs': {'enable_thinking': False}}
        connection = http.client.HTTPConnection('127.0.0.1', self.args.port, timeout=120)
        try:
            connection.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
            # Observe the actual replay entry before disconnecting, not a fixed sleep.
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if 'KVMEM vision replay ' in self.trace(offset): break
                time.sleep(.01)
            else: raise AssertionError('No media replay entry before cancellation')
            connection.sock.shutdown(socket.SHUT_RDWR)
        finally:
            connection.close()
        result = self.request(self.prompt('RED'), tokens=16)
        validate_answer(result, 'RED')
        return result

    def check_log(self):
        # Capacity rejection is intentional; every other engine failure remains fatal.
        data = self.log.read_text(encoding='utf-8', errors='replace')
        bad = [line for line in data.splitlines() if any(word in line.lower() for word in
            ('error', 'bad_alloc', 'inconsistent', 'terminate called', 'illegal memory', 'assertion'))
            and 'KVMem window cannot hold a complete media group' not in line]
        assert not bad, '\n'.join(bad[-30:])
        for name in ('memory-abort.json', 'windows-memory-abort.json'):
            assert not (self.args.output / name).exists(), name
        return {'bytes': len(data)}

    def draft_ring_rollover(self):
        result = self.request(self.prompt('RED', 'query-image', count=True), tokens=2304, stream=True)
        assert result['text'].strip().startswith('RED'), result['text'][:100]
        assert result['usage']['completion_tokens'] == 2304, 'Did not wrap the 2048-token draft ring'
        return result

    def draft_rejection_workload(self):
        # Counting is unusually easy for this companion. Exercise real rejection
        # without changing the acceptance rule or manufacturing draft metadata.
        prompts = [
            'First print the image color in uppercase. Then explain in Chinese how to implement '
            'an LRU cache with a hash map and a doubly linked list, including three tricky test cases.',
            'First print the image color in uppercase. Then write a Python parser for nested '
            'brackets that ignores brackets inside quoted strings and handles escaped quotes. '
            'Include three runnable edge case tests.',
            'First print the image color in uppercase. Then compute the prime factorizations '
            'of 1001, 1729 and 41041, showing and checking each multiplication.']
        results = []
        for question in prompts:
            prompt = self.prompt('BLUE', 'query-image')
            prompt[-1]['content'][-1] = text(question)
            result = self.request(prompt, tokens=384, stream=True)
            assert result['text'].strip().startswith('BLUE'), result['text'][:100]
            assert result['usage']['completion_tokens'] >= 64, 'Insufficient rejection workload'
            results.append(result)
            if re.search(r'KVMEM draft lane=\d+ extent=7 accepted=0', self.trace(0)):
                return results
        raise AssertionError('No actual zero-acceptance seven-draft round observed')

    def speculative_evidence(self):
        records = [json.loads(line) for line in (self.args.output / 'requests.jsonl').read_text().splitlines()]
        runs = [r['speculative'] for r in records if r.get('event') == 'request_done']
        assert runs, 'No completed request telemetry'
        assert all(r['backend'] == 'dflash2' and r['draft_window'] == 7 for r in runs)
        drafted = sum(r['drafted_tokens'] for r in runs)
        accepted = sum(r['accepted_tokens'] for r in runs)
        assert drafted > 0 and accepted > 0, 'No actual DFlash2 drafting/acceptance'
        evidence = {'draft_window': 7, 'drafted': drafted, 'accepted': accepted,
                    'acceptance_rate': accepted / drafted, 'completed_requests': len(runs)}
        if self.args.window:
            rows = [tuple(map(int, x)) for x in re.findall(
                r'KVMEM draft lane=(\d+) extent=(\d+) accepted=(\d+)', self.trace(0))]
            categories = {name: sum(condition(e, a) for _, e, a in rows) for name, condition in (
                ('zero', lambda e, a: e > 0 and a == 0),
                ('partial', lambda e, a: 0 < a < e),
                ('full', lambda e, a: e > 0 and a == e))}
            evidence['round_acceptance'] = categories
            assert all(categories.values()), f'Missing rejection/partial/full coverage: {categories}'
        return evidence

    def run(self):
        self.response_lock = threading.Lock()
        self.case('startup', self.start)
        if self.results[-1]['status'] != 'passed': return
        self.case('single-image', lambda: self.color_case('short', 'RED'))
        self.case('cached-media-text-suffix', self.cache_suffix)
        self.case('two-images', self.multiple_images)
        for mode in ('query-image', 'history-image', 'tool-tail'):
            self.case(mode, lambda m=mode: self.color_case(m, 'BLUE'))
        self.case('two-images-query-replay', self.multiple_images_replay)
        self.case('single-frame-video-route', self.video_route)
        self.case('two-lane-visual-isolation', self.parallel_images)
        if self.args.window:
            self.case('cancel-media-replay-and-reuse', self.cancel_and_reuse)
        if 0 < self.args.window <= 64:
            self.case('oversized-media-rejection', self.oversized_media)
        self.case('text-after-media', self.health)
        if self.args.spec == 'dflash2':
            if self.args.window:
                self.case('draft-ring-rollover', self.draft_ring_rollover)
            if 0 < self.args.window <= 64:
                self.case('text-two-lane-isolation', lambda: self.parallel_pair(0))
                self.case('cancel-decode-lane-and-reuse', self.cancel_one_lane)
            if self.args.window:
                self.case('draft-rejection-workload', self.draft_rejection_workload)
            self.case('dflash2-seven-draft-evidence', self.speculative_evidence)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, default=Path('build/apps/ninfer-serve'))
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--window', type=int, default=64)
    p.add_argument('--context', type=int, default=16384)
    p.add_argument('--host-mib', type=int, default=2048)
    p.add_argument('--spec', choices=['none', 'mtp', 'dflash2'], default='mtp')
    p.add_argument('--port', type=int, default=8099)
    args = p.parse_args()
    args.dtype, args.chunk, args.concurrency = 'int8', 1024, 2
    args.startup_timeout, args.request_timeout, args.profile = 600, 1200, 'vision'
    args.output.mkdir(parents=True, exist_ok=False)
    suite = VisionSuite(args)
    os.environ['NINFER_KVMEM_TRANSFER_TRACE'] = '1'
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        suite.run()
    except KeyboardInterrupt as exc:
        suite.results.append({'name': 'interrupted', 'status': 'failed', 'seconds': 0, 'error': str(exc)})
    finally:
        suite.case('shutdown', suite.stop)
        if suite.log.exists(): suite.case('no-engine-errors', suite.check_log)
        suite.save()
    return int(any(item['status'] == 'failed' for item in suite.results))


if __name__ == '__main__':
    raise SystemExit(main())
