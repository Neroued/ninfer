export type Numbers = Record<string, number>;

export interface ModelInfo {
  id: string;
  name: string;
  architecture: string;
  max_context: number;
  max_concurrency: number;
  prefill_chunk: number;
  kv_cache: string;
  context_token_capacity_estimate: number;
}

export interface RankInfo {
  rank: number;
  node: number;
  device: number;
  hardware: string;
  shared_physical_memory: boolean;
  tp: { rank: number; size: number };
  ep: { rank: number; size: number };
  dp: { rank: number; size: number };
  pp: { rank: number; size: number };
  adp: { rank: number; size: number };
}

export interface EngineInfo {
  server_instance_id: string;
  start_time_seconds: number;
  timestamp_unix_ms: number;
  model: ModelInfo;
  ranks: RankInfo[];
  placements: { id: number; attention_ranks: number[] }[];
  configuration: Record<string, string | number | boolean>;
  resources: {
    ranks: { rank: number; budget: Record<string, number | Record<string, number> | null> }[];
    total_allocations: Numbers;
  };
}

export interface Instance {
  id: string;
  engineUrl: string;
  firstSeen: number;
  lastSeen: number;
  info: EngineInfo;
  bounds: Record<string, number[]>;
  samplingIntervalMs: number;
}

export interface InstanceListEntry {
  id: string;
  model: string;
  architecture: string;
  ranks: number;
  firstSeen: number;
  lastSeen: number;
}

// Integral and duration support time-weighted means; extrema are observed samples.
export type Gauge = [integral: number, durationMs: number, min: number, max: number, last: number];

export interface Bucket {
  start: number;
  end: number;
  coveredMs: number;
  counters: Numbers;
  gauges: Record<string, Gauge>;
  bins: Record<string, number[]>;
}

export interface Distribution {
  count: number;
  mean: number | null;
  p50: number | null;
  p90: number | null;
  p99: number | null;
  bounds: number[];
  bins: number[];
}

export interface ExecutionSummary {
  label: string;
  count: number;
  totalMs: number | null;
  resultMs: number | null;
  resolutionMs: number | null;
  retirementMs: number | null;
}

export interface Summary {
  coveredMs: number;
  decodeTps: number | null;
  prefillTps: number | null;
  prefillEfficiency: number | null;
  stepMs: number | null;
  specPercent: number | null;
  prefixPercent: number | null;
  fallbackPercent: number | null;
  committedDrafts: number;
  reusedTokens: number;
  specPositions: {
    position: number;
    attempted: number;
    accepted: number;
    percent: number | null;
  }[];
  ttft: Distribution;
  duration: Distribution;
  queue: Distribution;
  decodeLatency: Distribution;
  batches: ExecutionSummary[];
  executions: ExecutionSummary[];
  gpuIntervals: { rank: number; name: string; count: number; meanMs: number | null }[];
  communication: {
    rank: number;
    path: string;
    kind: string;
    calls: number;
    payloadBytes: number;
  }[];
  counters: Numbers;
  gauges: Record<string, Gauge>;
}

export interface ChartPoint {
  at: number;
  start: number;
  coveredMs: number;
  decode: number | null;
  prefill: number | null;
  step: number | null;
  stepCount: number;
  running: number | null;
  waiting: number | null;
  prefilling: number | null;
  context: number | null;
  decodeMin: number | null;
  decodeMax: number | null;
  prefillMin: number | null;
  prefillMax: number | null;
  stepMin: number | null;
  stepMax: number | null;
  runningMin: number | null;
  runningMax: number | null;
  contextMax: number | null;
}

export interface HistoryResult {
  instanceId: string;
  from: number;
  to: number;
  resolutionMs: number;
  aggregationMs: number;
  points: ChartPoint[];
  summary: Summary;
}

export interface RequestRow {
  instanceId: string;
  id: number;
  engineId: number | null;
  httpId: string | null;
  protocol: string | null;
  phase: string;
  firstSeen: number;
  lastSeen: number;
  receivedAt: number | null;
  endedAt: number | null;
  terminal: boolean;
  responseFailed: boolean;
  promptTokens: number | null;
  reusedTokens: number | null;
  processedTokens: number | null;
  generatedTokens: number | null;
  contextTokens: number | null;
  ttftMs: number | null;
  speed: number | null;
  placement: number | null;
  preemptions: number;
  trace: { at: number; speed: number }[];
}

export interface MonitorEvent {
  key: string;
  instanceId: string;
  sequence: number | null;
  at: number;
  sourceAt: number | null;
  kind: string;
  requestId: number | null;
  details: Record<string, unknown>;
}

export interface Page<T> {
  items: T[];
  next: string | null;
}

export interface LiveResult {
  now: number;
  instanceId: string | null;
  connected: boolean;
  eventsConnected: boolean;
  engineStatus: string | null;
  lastSampleAt: number | null;
  error: string | null;
  storageError: string | null;
  defaultTheme: string;
  samplingIntervalMs: number;
  historyDays: number;
  fineHistoryHours: number;
  summary: Summary;
  gauges: Numbers;
  requests: RequestRow[];
  recentRequests: RequestRow[];
  events: MonitorEvent[];
  points: ChartPoint[];
}
