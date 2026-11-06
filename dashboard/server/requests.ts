import type { MonitorEvent, RequestRow } from '../shared/types.js';
import type { NinferSnapshot } from './ninfer.js';
import type { History } from './history.js';

const terminalKinds: Record<string, string> = {
  request_completed: 'completed',
  request_cancelled: 'cancelled',
  request_failed: 'failed',
  request_rejected: 'rejected',
};

export class Requests {
  private readonly active = new Map<number, RequestRow>();
  private recent: RequestRow[] = [];
  private instanceId = '';
  private readonly samples = new Map<number, { at: number; tokens: number }>();
  private readonly eventTimes = new Map<number, number>();

  constructor(
    private readonly history: History,
    private readonly intervalMs: number,
  ) {}

  switchInstance(id: string): void {
    if (id === this.instanceId) return;
    this.finishObservation();
    this.instanceId = id;
    this.recent = [];
    this.samples.clear();
    this.eventTimes.clear();
  }

  private find(id: number, at: number): RequestRow {
    return (
      this.active.get(id) ??
      this.history.request(this.instanceId, id) ?? {
        instanceId: this.instanceId,
        id,
        engineId: null,
        httpId: null,
        protocol: null,
        phase: 'waiting',
        firstSeen: at,
        lastSeen: at,
        receivedAt: null,
        endedAt: null,
        terminal: false,
        responseFailed: false,
        promptTokens: null,
        reusedTokens: null,
        processedTokens: null,
        generatedTokens: null,
        contextTokens: null,
        ttftMs: null,
        speed: null,
        placement: null,
        preemptions: 0,
        trace: [],
      }
    );
  }

  private save(row: RequestRow): void {
    this.history.saveRequest(row);
    if (row.terminal) {
      this.active.delete(row.id);
      this.samples.delete(row.id);
      this.eventTimes.delete(row.id);
      this.recent = [row, ...this.recent.filter((r) => r.id !== row.id)].slice(0, 20);
    } else {
      this.active.set(row.id, row);
      this.recent = this.recent.filter((r) => r.id !== row.id);
    }
  }

  observe(snapshot: NinferSnapshot, at: number, pollStarted: number): void {
    const engine = new Map(snapshot.engine.requests.map((r) => [r.id, r]));
    const present = new Set<number>();
    const steadyNow = snapshot.engine.sampled_ns + snapshot.engine.sample_age_ns;
    for (const request of snapshot.service.requests) {
      present.add(request.id);
      const row = this.find(request.id, at);
      if (row.terminal && row.phase !== 'unknown') continue;
      const previous = this.samples.get(row.id);
      const newerEvent = (this.eventTimes.get(request.id) ?? 0) > pollStarted;
      row.terminal = false;
      row.lastSeen = Math.max(row.lastSeen, at);
      row.engineId = request.engine_request_id || null;
      row.httpId = request.http_request_id || null;
      row.protocol = request.protocol;
      row.receivedAt = at - Math.max(0, steadyNow - request.received_ns) / 1e6;
      const value = engine.get(request.engine_request_id);
      if (!newerEvent) row.phase = request.preparing ? 'preparing' : (value?.phase ?? 'submitted');
      row.speed = null;
      if (value) {
        row.promptTokens = value.prompt_tokens;
        row.reusedTokens = value.reused_prompt_tokens;
        row.processedTokens = value.processed_prompt_tokens;
        row.generatedTokens = value.generated_tokens;
        row.contextTokens = value.logical_context_tokens;
        row.placement = value.placement;
        row.preemptions = value.preemptions;
        if (row.ttftMs === null && value.first_token_ns !== null) {
          row.ttftMs = Math.max(0, value.first_token_ns - request.received_ns) / 1e6;
        }
        if (value.phase === 'decode' && previous && at > previous.at) {
          row.speed =
            (Math.max(0, value.generated_tokens - previous.tokens) * 1000) / (at - previous.at);
          row.trace = [...row.trace, { at, speed: row.speed }].slice(-60);
        }
        this.samples.set(row.id, { at, tokens: value.generated_tokens });
      }
      this.save(row);
    }
    for (const row of [...this.active.values()]) {
      // A poll may have started before the SSE event that created this row.
      if (!present.has(row.id) && row.lastSeen < pollStarted - this.intervalMs * 2) {
        row.phase = 'unknown';
        row.terminal = true;
        row.speed = null;
        this.save(row);
      }
    }
  }

  event(event: MonitorEvent): void {
    if (event.requestId === null || event.instanceId !== this.instanceId) return;
    const row = this.find(event.requestId, event.at);
    this.eventTimes.set(row.id, event.at);
    const details = event.details;
    row.lastSeen = Math.max(row.lastSeen, event.at);
    if (typeof details.engine_request_id === 'number')
      row.engineId = details.engine_request_id || null;
    if (event.kind === 'request_preparing') {
      row.receivedAt ??= event.at;
      if (!row.terminal) row.phase = 'preparing';
    } else if (event.kind === 'request_submitted' && !row.terminal) row.phase = 'waiting';
    else if (event.kind === 'first_token') {
      if (typeof details.ttft_seconds === 'number') row.ttftMs = details.ttft_seconds * 1000;
      if (!row.terminal) row.phase = 'decode';
    } else if (event.kind === 'response_failed') row.responseFailed = true;
    else if (terminalKinds[event.kind]) {
      row.terminal = true;
      row.phase = terminalKinds[event.kind];
      row.endedAt = event.at;
      row.speed = null;
      if (typeof details.prompt_tokens === 'number') row.promptTokens = details.prompt_tokens;
      if (typeof details.completion_tokens === 'number')
        row.generatedTokens = details.completion_tokens;
    } else if (!row.terminal) {
      const phases: Record<string, string> = {
        pause_started: 'pausing',
        paused: 'paused',
        restore_started: 'materializing',
        restored: 'replay',
        replay_complete: 'decode',
        recovery_complete: 'decode',
        terminal: 'finishing',
      };
      if (phases[event.kind]) row.phase = phases[event.kind];
    }
    this.save(row);
  }

  finishObservation(): void {
    for (const row of [...this.active.values()]) {
      row.phase = 'unknown';
      row.terminal = true;
      row.speed = null;
      this.save(row);
    }
  }

  samplingGap(): void {
    this.samples.clear();
    for (const row of this.active.values()) {
      row.speed = null;
      row.trace = [];
    }
  }

  current(): RequestRow[] {
    return [...this.active.values()].sort((a, b) => a.firstSeen - b.firstSeen);
  }
  completed(): RequestRow[] {
    return this.recent;
  }
}
