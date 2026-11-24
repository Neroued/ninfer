import assert from 'node:assert/strict';
import test from 'node:test';
import { emptyBucket, summarize } from '../server/statistics.js';
import type { Numbers } from '../shared/types.js';

function speculation(counters: Numbers) {
  return summarize({ ...emptyBucket(0, 1000), coveredMs: 1000, counters }, {});
}

test('an all-bypass interval has zero verification depth and no acceptance sample', () => {
  const summary = speculation({ speculative_bypass_steps: 4 });
  assert.equal(summary.specPercent, null);
  assert.equal(summary.bypassPercent, 100);
  assert.equal(summary.fallbackPercent, 0);
  assert.equal(summary.avgDraftDepth, 0);
  assert.equal(summary.bypassSteps, 4);
  assert.equal(summary.fallbackSteps, 0);
});

test('rounds, bypass and fallback all contribute to the decode step denominator', () => {
  const summary = speculation({
    speculative_rounds: 2,
    speculative_bypass_steps: 1,
    speculative_fallback_steps: 1,
    speculative_proposed_tokens: 6,
    speculative_draft_tokens: 6,
    speculative_accepted_tokens: 3,
  });
  assert.equal(summary.specRounds, 2);
  assert.equal(summary.bypassSteps, 1);
  assert.equal(summary.fallbackSteps, 1);
  assert.equal(summary.bypassPercent, 25);
  assert.equal(summary.fallbackPercent, 25);
  assert.equal(summary.avgDraftDepth, 1.5);
  assert.equal(summary.specPercent, 50);
});

test('split proposals preserve candidate counts and use only verified drafts for acceptance', () => {
  const summary = speculation({
    speculative_rounds: 1,
    speculative_bypass_steps: 1,
    speculative_proposed_tokens: 10,
    speculative_draft_tokens: 2,
    speculative_accepted_tokens: 1,
  });
  assert.equal(summary.proposedDrafts, 10);
  assert.equal(summary.verifiedDrafts, 2);
  assert.equal(summary.avgDraftDepth, 1);
  assert.equal(summary.specPercent, 50);
});

test('legacy buckets retain their existing counts and default missing counters to zero', () => {
  const summary = speculation({
    speculative_rounds: 3,
    speculative_fallback_steps: 1,
    speculative_draft_tokens: 8,
  });
  assert.equal(summary.proposedDrafts, 0);
  assert.equal(summary.verifiedDrafts, 8);
  assert.equal(summary.bypassSteps, 0);
  assert.equal(summary.bypassPercent, 0);
  assert.equal(summary.fallbackPercent, 25);
  assert.equal(summary.avgDraftDepth, 2);
  const empty = speculation({});
  assert.equal(empty.bypassPercent, null);
  assert.equal(empty.fallbackPercent, null);
  assert.equal(empty.avgDraftDepth, null);
});
