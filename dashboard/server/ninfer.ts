import type { EngineInfo, Numbers } from '../shared/types.js';

export interface EngineRequest {
  id: number;
  phase: string;
  placement: number | null;
  submitted_ns: number;
  admitted_ns: number | null;
  first_token_ns: number | null;
  prompt_tokens: number;
  reused_prompt_tokens: number;
  processed_prompt_tokens: number;
  generated_tokens: number;
  logical_context_tokens: number;
  preemptions: number;
}

export interface ServiceRequest {
  id: number;
  engine_request_id: number;
  http_request_id: string;
  protocol: string;
  received_ns: number;
  preparing: boolean;
}

interface Duration {
  count: number;
  total_ns: number;
}

interface Execution {
  units: number;
  rows: number;
  input_tokens: number;
  scheduled_positions: number;
  result: Duration;
  resolution: Duration;
  retirement: Duration;
  total: Duration;
  latency_bins: number[];
}

interface RankSnapshot {
  rank: number;
  sample_age_ns: number | null;
  physical_memory_age_ns: number | null;
  allocations: Numbers;
  workspace: Numbers;
  physical_memory: Record<string, number | boolean> | null;
  gpu_intervals: Record<string, Duration>;
  host: Numbers;
  attention_work: Record<string, Numbers>;
  communication_eager: Record<string, Numbers>;
  communication_graph: Record<string, Numbers>;
  vision: Numbers | null;
  experts: {
    [key: string]: number | number[];
    uploaded_bytes: number;
    prefetched_bytes: number;
  } | null;
  ple: Numbers | null;
}

interface Histogram {
  bins: number[];
  sum_seconds: number;
}

interface ContextPool {
  rank: number;
  domain: string;
  capacity_bytes: number;
  in_use_bytes?: number;
  cached_bytes?: number;
  assigned_free_bytes: number;
  unassigned_free_bytes: number;
  [key: string]: number | string | undefined;
}

interface ContextResource {
  rank: number;
  domain: string;
  history_pool: number | null;
  live_bytes: number;
  in_use_bytes?: number;
  cached_bytes?: number;
  pending_retirement_bytes: number;
}

export interface NinferSnapshot {
  server_instance_id: string;
  timestamp_unix_ms: number;
  engine: {
    sequence: number;
    sampled_ns: number;
    sample_age_ns: number;
    status: string;
    runtime: {
      [key: string]: unknown;
      requests: Numbers;
      host_work: Numbers;
      speculative_attempted_per_position: number[];
      speculative_accepted_per_position: number[];
      prefix_cache: Record<string, Numbers>;
    };
    execution_latency_bounds_ns: number[];
    executions: Record<string, Execution>;
    decode_by_batch: (Execution & { batch: number })[];
    context: {
      pools: ContextPool[];
      resources: ContextResource[];
      metadata: Numbers;
    };
    placements: (Numbers & { id: number })[];
    ranks: RankSnapshot[];
    media: Numbers;
    requests: EngineRequest[];
  };
  service: {
    [key: string]: unknown;
    requests: ServiceRequest[];
    duration_bounds_seconds: number[];
    ttft: Histogram;
    duration: Histogram;
    queue: Histogram;
  };
}

export interface Frame {
  instanceId: string;
  at: number;
  monotonicMs: number;
  counters: Numbers;
  gauges: Numbers;
  bins: Record<string, number[]>;
  bounds: Record<string, number[]>;
}

const runtimeGauges = new Set([
  'running_requests',
  'prefilling_requests',
  'decode_ready_requests',
  'waiting_requests',
  'paused_requests',
  'replaying_requests',
  'materializing_requests',
  'capture_pending_requests',
  'terminal_pending_requests',
  'last_selected_frontier_tokens',
]);

function numbers(target: Numbers, prefix: string, source: object) {
  for (const [key, value] of Object.entries(source)) {
    if (typeof value === 'number' && Number.isFinite(value)) target[`${prefix}${key}`] = value;
  }
}

export function frameOf(snapshot: NinferSnapshot, at: number, monotonicMs: number): Frame {
  const counters: Numbers = {};
  const gauges: Numbers = {};
  const bins: Record<string, number[]> = {};
  const bounds: Record<string, number[]> = {};
  const engine = snapshot.engine;
  const runtime = engine.runtime;
  for (const [key, value] of Object.entries(runtime)) {
    if (typeof value === 'number') (runtimeGauges.has(key) ? gauges : counters)[key] = value;
  }
  numbers(counters, 'request.', runtime.requests);
  numbers(counters, 'host.', runtime.host_work);
  runtime.speculative_attempted_per_position.forEach((value, i) => {
    counters[`spec.${i + 1}.attempted`] = value;
    counters[`spec.${i + 1}.accepted`] = runtime.speculative_accepted_per_position[i] ?? 0;
  });
  for (const [kind, values] of Object.entries(runtime.prefix_cache)) {
    gauges[`prefix.${kind}.retained`] = values.retained;
    for (const key of ['published', 'released', 'reclaimed', 'adopted', 'reused_tokens']) {
      const value = values[key];
      if (typeof value === 'number') counters[`prefix.${kind}.${key}`] = value;
    }
  }

  const execution = (prefix: string, value: Execution) => {
    counters[`${prefix}.count`] = value.units;
    for (const key of ['rows', 'input_tokens', 'scheduled_positions'] as const) {
      counters[`${prefix}.${key}`] = value[key];
    }
    for (const stage of ['total', 'result', 'resolution', 'retirement'] as const) {
      counters[`${prefix}.${stage}_ns`] = value[stage].total_ns;
      counters[`${prefix}.${stage}_count`] = value[stage].count;
    }
    bins[prefix] = value.latency_bins;
    bounds[prefix] = engine.execution_latency_bounds_ns.map((ns) => ns / 1e6);
  };
  for (const [kind, value] of Object.entries(engine.executions))
    execution(`execution.${kind}`, value);
  for (const value of engine.decode_by_batch) execution(`batch.${value.batch}`, value);
  for (const [key, value] of Object.entries(snapshot.service)) {
    if (typeof value === 'number') {
      (key === 'preparing' ? gauges : counters)[`service.${key}`] = value;
    }
  }
  for (const kind of ['ttft', 'duration', 'queue'] as const) {
    bins[`service.${kind}`] = snapshot.service[kind].bins;
    bounds[`service.${kind}`] = snapshot.service.duration_bounds_seconds.map((s) => s * 1000);
    counters[`service.${kind}.sum_ms`] = snapshot.service[kind].sum_seconds * 1000;
  }

  gauges.context_capacity_bytes = 0;
  gauges.context_occupied_bytes = 0;
  for (const pool of engine.context.pools) {
    const prefix = `pool.${pool.domain}.${pool.rank}.`;
    numbers(gauges, prefix, pool);
    const occupied = pool.capacity_bytes - pool.assigned_free_bytes - pool.unassigned_free_bytes;
    gauges[`${prefix}occupied_bytes`] = occupied;
    if (pool.domain === 'device') {
      gauges.context_capacity_bytes += pool.capacity_bytes;
      gauges.context_occupied_bytes += occupied;
    }
  }
  if (gauges.context_capacity_bytes > 0) {
    gauges.context_percent = (100 * gauges.context_occupied_bytes) / gauges.context_capacity_bytes;
    const pools = engine.context.pools.filter((pool) => pool.domain === 'device');
    if (pools.every((pool) => pool.in_use_bytes !== undefined && pool.cached_bytes !== undefined)) {
      gauges.context_in_use_bytes = pools.reduce((sum, pool) => sum + pool.in_use_bytes!, 0);
      gauges.context_cached_bytes = pools.reduce((sum, pool) => sum + pool.cached_bytes!, 0);
      gauges.context_in_use_percent =
        (100 * gauges.context_in_use_bytes) / gauges.context_capacity_bytes;
    }
  }
  for (const resource of engine.context.resources) {
    const prefix = `pool.${resource.domain}.${resource.rank}.${resource.history_pool === null ? 'state' : 'kv'}`;
    for (const field of [
      'live_bytes',
      'in_use_bytes',
      'cached_bytes',
      'pending_retirement_bytes',
    ] as const) {
      if (resource[field] !== undefined)
        gauges[`${prefix}_${field}`] = (gauges[`${prefix}_${field}`] ?? 0) + resource[field];
    }
  }
  numbers(gauges, 'metadata.', engine.context.metadata);
  for (const placement of engine.placements)
    numbers(gauges, `placement.${placement.id}.`, placement);

  for (const rank of engine.ranks) {
    const prefix = `rank.${rank.rank}.`;
    if (rank.sample_age_ns !== null) gauges[`${prefix}sample_age_ms`] = rank.sample_age_ns / 1e6;
    if (rank.physical_memory_age_ns !== null) {
      gauges[`${prefix}physical_age_ms`] = rank.physical_memory_age_ns / 1e6;
    }
    numbers(gauges, `${prefix}allocation.`, rank.allocations);
    numbers(gauges, `${prefix}workspace.`, rank.workspace);
    if (rank.physical_memory) numbers(gauges, `${prefix}physical.`, rank.physical_memory);
    if (rank.vision) numbers(gauges, `${prefix}vision.`, rank.vision);
    numbers(counters, `${prefix}host.`, rank.host);
    for (const [name, value] of Object.entries(rank.gpu_intervals)) {
      counters[`${prefix}gpu.${name}.ns`] = value.total_ns;
      counters[`${prefix}gpu.${name}.count`] = value.count;
    }
    for (const [kind, value] of Object.entries(rank.attention_work)) {
      numbers(counters, `${prefix}attention.${kind}.`, value);
    }
    for (const path of ['eager', 'graph'] as const) {
      for (const [kind, value] of Object.entries(rank[`communication_${path}`])) {
        numbers(counters, `${prefix}communication.${path}.${kind}.`, value);
      }
    }
    if (rank.experts) {
      numbers(gauges, `${prefix}experts.`, rank.experts);
      for (const key of ['uploaded_bytes', 'prefetched_bytes'] as const) {
        counters[`${prefix}experts.${key}`] = rank.experts[key];
        delete gauges[`${prefix}experts.${key}`];
      }
    }
    if (rank.ple) numbers(counters, `${prefix}ple.`, rank.ple);
  }
  const mediaCounters = new Set([
    'hits',
    'misses',
    'singleflight_waits',
    'evictions',
    'oversize_bypasses',
  ]);
  for (const [key, value] of Object.entries(engine.media)) {
    (mediaCounters.has(key) ? counters : gauges)[`media.${key}`] = value;
  }
  gauges.engine_sample_age_ms = engine.sample_age_ns / 1e6;
  return {
    instanceId: snapshot.server_instance_id,
    at,
    monotonicMs,
    counters,
    gauges,
    bins,
    bounds,
  };
}

export async function getInfo(url: string, signal: AbortSignal): Promise<EngineInfo> {
  return getJson<EngineInfo>(url, '/monitor/info', signal);
}

export async function getSnapshot(url: string, signal: AbortSignal): Promise<NinferSnapshot> {
  return getJson<NinferSnapshot>(url, '/monitor/snapshot', signal);
}

async function getJson<T>(base: string, path: string, signal: AbortSignal): Promise<T> {
  const response = await fetch(`${base}${path}`, { signal });
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json() as Promise<T>;
}
