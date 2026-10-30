import type { ChartPoint } from './types.js';

export function chartInterval(rangeMs: number, sourceMs: number, maxPoints: number): number {
  const intervals = [
    [60_000, 1000],
    [300_000, 2000],
    [900_000, 5000],
    [3_600_000, 15_000],
    [21_600_000, 60_000],
    [86_400_000, 300_000],
    [604_800_000, 3_600_000],
    [Infinity, 7_200_000],
  ];
  const preferred = intervals.find(([range]) => rangeMs <= range)![1];
  return Math.ceil(Math.max(preferred, rangeMs / maxPoints, sourceMs) / sourceMs) * sourceMs;
}

function chartGap(start: number, at: number): ChartPoint {
  return {
    start,
    at,
    coveredMs: 0,
    decode: null,
    prefill: null,
    step: null,
    stepCount: 0,
    running: null,
    waiting: null,
    prefilling: null,
    context: null,
    decodeMin: null,
    decodeMax: null,
    prefillMin: null,
    prefillMax: null,
    stepMin: null,
    stepMax: null,
    runningMin: null,
    runningMax: null,
    contextMax: null,
  };
}

function mean(a: number | null, aWeight: number, b: number | null, bWeight: number): number | null {
  if (a === null) return b;
  if (b === null) return a;
  return (a * aWeight + b * bWeight) / (aWeight + bWeight);
}

function extreme(a: number | null, b: number | null, minimum: boolean): number | null {
  if (a === null) return b;
  if (b === null) return a;
  return minimum ? Math.min(a, b) : Math.max(a, b);
}

// Complete observations belong to the wall-clock bucket containing their end time.
// Keep separate segments across sampling gaps, even within the same display bucket.
export function appendChartPoints(
  points: ChartPoint[],
  updates: ChartPoint[],
  aggregationMs: number,
): void {
  for (const point of updates) {
    const previous = points.at(-1);
    if (previous && point.at <= previous.at) continue;
    if (previous && point.start > previous.at) {
      points.push(chartGap(previous.at, (previous.at + point.start) / 2));
    } else if (
      previous?.coveredMs &&
      point.coveredMs &&
      Math.ceil(previous.at / aggregationMs) === Math.ceil(point.at / aggregationMs)
    ) {
      const merged = { ...previous };
      for (const key of [
        'decode',
        'prefill',
        'running',
        'waiting',
        'prefilling',
        'context',
      ] as const) {
        merged[key] = mean(previous[key], previous.coveredMs, point[key], point.coveredMs);
      }
      merged.step = mean(previous.step, previous.stepCount, point.step, point.stepCount);
      merged.stepCount += point.stepCount;
      for (const key of ['decodeMin', 'prefillMin', 'stepMin', 'runningMin'] as const)
        merged[key] = extreme(previous[key], point[key], true);
      for (const key of ['decodeMax', 'prefillMax', 'stepMax', 'runningMax', 'contextMax'] as const)
        merged[key] = extreme(previous[key], point[key], false);
      merged.at = point.at;
      merged.coveredMs += point.coveredMs;
      points[points.length - 1] = merged;
      continue;
    }
    points.push({ ...point });
  }
}
