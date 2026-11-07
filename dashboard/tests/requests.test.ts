import assert from 'node:assert/strict';
import test from 'node:test';
import { Requests } from '../server/requests.js';
import type { History } from '../server/history.js';
import type { NinferSnapshot } from '../server/ninfer.js';
import type { MonitorEvent } from '../shared/types.js';

function fixture() {
  const history = { request: () => null, saveRequest: () => {} } as unknown as History;
  const requests = new Requests(history, 1000);
  requests.switchInstance('engine');
  const snapshot = {
    engine: {
      sampled_ns: 1_000_000_000,
      sample_age_ns: 0,
      progress_owner: 19,
      requests: [
        {
          id: 19,
          phase: 'replay',
          placement: 0,
          submitted_ns: 0,
          admitted_ns: 1,
          first_token_ns: null,
          prompt_tokens: 1024,
          reused_prompt_tokens: 0,
          processed_prompt_tokens: 512,
          generated_tokens: 0,
          logical_context_tokens: 512,
          preemptions: 1,
          replayed_tokens: 128,
          recovery_pending: true,
          last_progress_ns: 900_000_000,
          recovery: { route: 'replay', begin_frontier: 64, frontier: 192, target_frontier: 512 },
        },
      ],
    },
    service: { requests: [{ id: 7, engine_request_id: 19, received_ns: 0, preparing: false }] },
  } as NinferSnapshot;
  return { requests, snapshot };
}

test('replay progress and speed use recomputation without changing initial prompt/output', () => {
  const { requests, snapshot } = fixture();
  requests.observe(snapshot, 1000, 1000);
  const value = snapshot.engine.requests[0]!;
  value.replayed_tokens = 256;
  value.recovery!.frontier = 320;
  snapshot.engine.sampled_ns += 1_000_000_000;
  requests.observe(snapshot, 2000, 2000);
  const row = requests.current()[0]!;
  assert.equal(row.replaySpeed, 128);
  assert.equal(row.speed, null);
  assert.equal(row.processedTokens, 512);
  assert.equal(row.generatedTokens, 0);
  assert.equal(row.recovery!.target_frontier - row.recovery!.frontier, 192);
  assert.equal(row.progressOwner, true);
  assert.equal(row.lastProgressAt, 900);
  value.preemptions = 2;
  value.recovery = { route: 'replay', begin_frontier: 0, frontier: 64, target_frontier: 512 };
  value.replayed_tokens += 64;
  requests.observe(snapshot, 3000, 3000);
  assert.equal(row.replaySpeed, null, 'a new restoration must not inherit the previous rate');
  assert.equal(row.replayedTokens, 320);
  requests.observe(snapshot, 4000, 4000);
  assert.equal(row.replaySpeed, 0, 'stalled replay must show zero completed work');
});

test('restoration events retain the producer phase across snapshot races', () => {
  const { requests, snapshot } = fixture();
  requests.observe(snapshot, 1000, 1000);
  for (const kind of ['restored', 'replay_complete', 'recovery_complete']) {
    requests.event({
      instanceId: 'engine',
      requestId: 7,
      at: 2500,
      kind,
      details: { phase: 'prefill', route: 'direct' },
    } as MonitorEvent);
    assert.equal(requests.current()[0]!.phase, 'prefill');
    requests.observe(snapshot, 2600, 2400);
    assert.equal(requests.current()[0]!.phase, 'prefill', 'an older poll overwrote the event');
  }
  snapshot.engine.requests[0]!.phase = 'control';
  requests.observe(snapshot, 3000, 3000);
  assert.equal(requests.current()[0]!.phase, 'control');
});
