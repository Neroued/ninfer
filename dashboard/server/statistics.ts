import type {
  Bucket,
  ChartPoint,
  Distribution,
  ExecutionSummary,
  Gauge,
  Summary,
} from '../shared/types.js';
import type { Frame } from './ninfer.js';

export function emptyBucket(start = 0, end = start): Bucket {
  return { start, end, coveredMs: 0, counters: {}, gauges: {}, bins: {} };
}

export function interval(before: Frame, after: Frame): Bucket | null {
  const elapsed = after.monotonicMs - before.monotonicMs;
  if (before.instanceId !== after.instanceId || elapsed <= 0 || after.at <= before.at) return null;
  const result = emptyBucket(before.at, after.at);
  result.coveredMs = elapsed;
  for (const [key, value] of Object.entries(after.counters)) {
    if (!(key in before.counters)) continue;
    const delta = value - before.counters[key];
    if (delta < 0) return null;
    if (delta) result.counters[key] = delta;
  }
  for (const [key, values] of Object.entries(after.bins)) {
    const previous = before.bins[key];
    if (!previous || previous.length !== values.length) continue;
    const delta = values.map((value, i) => value - previous[i]);
    if (delta.some((value) => value < 0)) return null;
    if (delta.some(Boolean)) result.bins[key] = delta;
  }
  for (const [key, value] of Object.entries(after.gauges)) {
    result.gauges[key] = [value * elapsed, elapsed, value, value, value];
  }
  const c = result.counters;
  for (const [key, value] of Object.entries({
    decode_tps: ((c.committed_decode_tokens ?? 0) * 1000) / elapsed,
    prefill_tps: ((c.computed_prefill_tokens ?? 0) * 1000) / elapsed,
    step_ms: divide(c['execution.decode.total_ns'], c['execution.decode.total_count'], 1e-6),
  })) {
    if (value !== null) result.gauges[key] = [value * elapsed, elapsed, value, value, value];
  }
  return result;
}

export function merge(target: Bucket, source: Bucket): void {
  if (!target.coveredMs) target.start = source.start;
  target.end = Math.max(target.end, source.end);
  target.coveredMs += source.coveredMs;
  for (const [key, value] of Object.entries(source.counters)) {
    target.counters[key] = (target.counters[key] ?? 0) + value;
  }
  for (const [key, value] of Object.entries(source.gauges)) {
    const prior = target.gauges[key];
    target.gauges[key] = prior
      ? [
          prior[0] + value[0],
          prior[1] + value[1],
          Math.min(prior[2], value[2]),
          Math.max(prior[3], value[3]),
          value[4],
        ]
      : [...value];
  }
  for (const [key, value] of Object.entries(source.bins)) {
    const prior = target.bins[key];
    target.bins[key] = prior ? value.map((n, i) => n + (prior[i] ?? 0)) : [...value];
  }
}

function divide(numerator = 0, denominator = 0, scale = 1): number | null {
  return denominator > 0 ? (numerator / denominator) * scale : null;
}

function quantile(bounds: number[], bins: number[], fraction: number): number | null {
  const count = bins.reduce((a, b) => a + b, 0);
  if (!count) return null;
  const target = count * fraction;
  let accumulated = 0;
  for (let i = 0; i < bins.length; ++i) {
    if (accumulated + bins[i] >= target && bins[i] > 0) {
      if (i >= bounds.length) return null;
      const low = i ? bounds[i - 1] : 0;
      return low + ((bounds[i] - low) * (target - accumulated)) / bins[i];
    }
    accumulated += bins[i];
  }
  return null;
}

export function summarize(bucket: Bucket, bounds: Record<string, number[]>): Summary {
  const c = bucket.counters;
  const distribution = (key: string, sum: number): Distribution => {
    const edges = bounds[key] ?? [];
    const bins = bucket.bins[key] ?? edges.map(() => 0).concat(0);
    const count = bins.reduce((a, b) => a + b, 0);
    return {
      count,
      mean: divide(sum, count),
      p50: quantile(edges, bins, 0.5),
      p90: quantile(edges, bins, 0.9),
      p99: quantile(edges, bins, 0.99),
      bounds: edges,
      bins,
    };
  };
  const execution = (prefix: string, label: string): ExecutionSummary => ({
    label,
    count: c[`${prefix}.count`] ?? 0,
    totalMs: divide(c[`${prefix}.total_ns`], c[`${prefix}.total_count`], 1e-6),
    resultMs: divide(c[`${prefix}.result_ns`], c[`${prefix}.result_count`], 1e-6),
    resolutionMs: divide(c[`${prefix}.resolution_ns`], c[`${prefix}.resolution_count`], 1e-6),
    retirementMs: divide(c[`${prefix}.retirement_ns`], c[`${prefix}.retirement_count`], 1e-6),
  });
  const batches = Object.keys(bounds)
    .filter((k) => k.startsWith('batch.'))
    .sort((a, b) => Number(a.split('.')[1]) - Number(b.split('.')[1]));
  const positions = Object.keys(c)
    .filter((k) => /^spec\.\d+\.attempted$/.test(k))
    .map((k) => Number(k.split('.')[1]))
    .sort((a, b) => a - b);
  return {
    coveredMs: bucket.coveredMs,
    decodeTps: divide(c.committed_decode_tokens, bucket.coveredMs, 1000),
    prefillTps: divide(c.computed_prefill_tokens, bucket.coveredMs, 1000),
    prefillEfficiency: divide(
      c['execution.prefill.input_tokens'],
      c['execution.prefill.total_ns'],
      1e9,
    ),
    stepMs: divide(c['execution.decode.total_ns'], c['execution.decode.total_count'], 1e-6),
    specPercent: divide(c.speculative_accepted_tokens, c.speculative_draft_tokens, 100),
    prefixPercent: divide(c.reused_prompt_tokens, c.prompt_tokens, 100),
    fallbackPercent: divide(
      c.speculative_fallback_steps,
      (c.speculative_rounds ?? 0) + (c.speculative_fallback_steps ?? 0),
      100,
    ),
    committedDrafts: c.speculative_committed_draft_tokens ?? 0,
    reusedTokens: c.reused_prompt_tokens ?? 0,
    specPositions: positions.map((position) => ({
      position,
      attempted: c[`spec.${position}.attempted`],
      accepted: c[`spec.${position}.accepted`] ?? 0,
      percent: divide(c[`spec.${position}.accepted`], c[`spec.${position}.attempted`], 100),
    })),
    ttft: distribution('service.ttft', c['service.ttft.sum_ms'] ?? 0),
    duration: distribution('service.duration', c['service.duration.sum_ms'] ?? 0),
    queue: distribution('service.queue', c['service.queue.sum_ms'] ?? 0),
    decodeLatency: distribution('execution.decode', (c['execution.decode.total_ns'] ?? 0) / 1e6),
    batches: batches.map((key) => execution(key, `B${key.split('.')[1]}`)),
    executions: ['prefill', 'decode', 'replay', 'control', 'context', 'score'].map((kind) =>
      execution(`execution.${kind}`, kind),
    ),
    gpuIntervals: Object.keys(c)
      .filter((key) => /^rank\.\d+\.gpu\.[^.]+\.count$/.test(key))
      .map((key) => {
        const fields = key.split('.');
        return {
          rank: Number(fields[1]),
          name: fields[3],
          count: c[key],
          meanMs: divide(c[key.replace(/\.count$/, '.ns')], c[key], 1e-6),
        };
      }),
    communication: Object.keys(c)
      .filter((key) => /^rank\.\d+\.communication\.[^.]+\.[^.]+\.calls$/.test(key))
      .map((key) => {
        const fields = key.split('.');
        return {
          rank: Number(fields[1]),
          path: fields[3],
          kind: fields[4],
          calls: c[key],
          payloadBytes: c[key.replace(/\.calls$/, '.payload_bytes')] ?? 0,
        };
      }),
    counters: c,
    gauges: bucket.gauges,
  };
}

export function gaugeMean(gauge: Gauge | undefined): number | null {
  return gauge ? divide(gauge[0], gauge[1]) : null;
}

export function chartPoint(bucket: Bucket): ChartPoint {
  const c = bucket.counters;
  const g = bucket.gauges;
  return {
    at: bucket.end,
    start: bucket.start,
    coveredMs: bucket.coveredMs,
    decode: divide(c.committed_decode_tokens, bucket.coveredMs, 1000),
    prefill: divide(c.computed_prefill_tokens, bucket.coveredMs, 1000),
    step: divide(c['execution.decode.total_ns'], c['execution.decode.total_count'], 1e-6),
    stepCount: c['execution.decode.total_count'] ?? 0,
    running: gaugeMean(g.running_requests),
    waiting: gaugeMean(g.waiting_requests),
    prefilling: gaugeMean(g.prefilling_requests),
    context: gaugeMean(g.context_percent),
    decodeMin: g.decode_tps?.[2] ?? null,
    decodeMax: g.decode_tps?.[3] ?? null,
    prefillMin: g.prefill_tps?.[2] ?? null,
    prefillMax: g.prefill_tps?.[3] ?? null,
    stepMin: g.step_ms?.[2] ?? null,
    stepMax: g.step_ms?.[3] ?? null,
    runningMin: g.running_requests?.[2] ?? null,
    runningMax: g.running_requests?.[3] ?? null,
    contextMax: g.context_percent?.[3] ?? null,
  };
}
