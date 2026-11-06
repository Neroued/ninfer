import { randomUUID } from 'node:crypto';
import { setTimeout as sleep } from 'node:timers/promises';
import { EventSourceParserStream } from 'eventsource-parser/stream';
import type { Bucket, Instance, LiveResult, MonitorEvent } from '../shared/types.js';
import { frameOf, getInfo, getSnapshot } from './ninfer.js';
import type { Frame } from './ninfer.js';
import type { History } from './history.js';
import { Requests } from './requests.js';
import { chartPoint, emptyBucket, interval, merge, summarize } from './statistics.js';

export interface CollectorOptions {
  engineUrl: string;
  intervalMs: number;
  defaultTheme: string;
}

export class Collector {
  private readonly stopSignal = new AbortController();
  private eventSignal: AbortController | null = null;
  private pollTask: Promise<void> | null = null;
  private eventTask: Promise<void> | null = null;
  private instance: Instance | null = null;
  private previous: Frame | null = null;
  private latest: Frame | null = null;
  private buckets: Bucket[] = [];
  private events: MonitorEvent[] = [];
  private readonly requests: Requests;
  private connected = false;
  private eventsConnected = false;
  private engineStatus: string | null = null;
  private error: string | null = null;
  private storageError: string | null = null;
  private lastSequence: number | null = null;
  private lastPruned = 0;

  constructor(
    private readonly options: CollectorOptions,
    private readonly history: History,
    private readonly log: (message: string) => void,
  ) {
    this.requests = new Requests(history, options.intervalMs);
  }

  start(): void {
    this.pollTask = this.poll();
  }

  private persist(action: () => void): void {
    try {
      action();
      this.storageError = null;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (message !== this.storageError) this.log(`History: ${message}`);
      this.storageError = message;
    }
  }

  private localEvent(kind: string, details: Record<string, unknown> = {}): void {
    if (!this.instance) return;
    this.recordEvent({
      key: randomUUID(),
      instanceId: this.instance.id,
      sequence: null,
      at: Date.now(),
      sourceAt: null,
      kind,
      requestId: null,
      details,
    });
  }

  private recordEvent(event: MonitorEvent): void {
    this.persist(() => {
      if (this.history.saveEvent(event)) this.requests.event(event);
    });
    this.events = [event, ...this.events.filter((e) => e.key !== event.key)].slice(0, 50);
  }

  private async selectInstance(id: string): Promise<boolean> {
    const info = await getInfo(
      this.options.engineUrl,
      AbortSignal.any([this.stopSignal.signal, AbortSignal.timeout(5000)]),
    );
    if (info.server_instance_id !== id) return false;
    this.eventSignal?.abort();
    await this.eventTask;
    const now = Date.now();
    const prior = this.history.instance(id);
    this.instance = {
      id,
      engineUrl: this.options.engineUrl,
      info,
      firstSeen: prior?.firstSeen ?? now,
      lastSeen: now,
      bounds: {},
      samplingIntervalMs: this.options.intervalMs,
    };
    this.persist(() => {
      this.requests.switchInstance(id);
      this.history.saveInstance(this.instance!);
    });
    this.previous = null;
    this.latest = null;
    this.buckets = [];
    this.events = [];
    this.lastSequence = null;
    this.localEvent('instance_observed');
    this.eventSignal = new AbortController();
    this.eventTask = this.readEvents(id, this.eventSignal.signal);
    this.log(`Observing ${info.model.name} · ${info.ranks.length} rank(s)`);
    return true;
  }

  private async poll(): Promise<void> {
    while (!this.stopSignal.signal.aborted) {
      const started = performance.now();
      const pollStarted = Date.now();
      try {
        const snapshot = await getSnapshot(
          this.options.engineUrl,
          AbortSignal.any([
            this.stopSignal.signal,
            AbortSignal.timeout(Math.max(5000, this.options.intervalMs * 2)),
          ]),
        );
        const frame = frameOf(snapshot, Date.now(), performance.now());
        if (
          this.instance?.id !== frame.instanceId &&
          !(await this.selectInstance(frame.instanceId))
        )
          continue;
        const instance = this.instance!;
        if (!Object.keys(instance.bounds).length) {
          instance.bounds = frame.bounds;
          this.persist(() => this.history.saveInstance(instance));
        }
        if (!this.connected && this.latest) this.localEvent('sampling_resumed');
        this.connected = true;
        this.error = null;
        if (this.previous) {
          const bucket = interval(this.previous, frame);
          if (bucket && bucket.coveredMs <= Math.max(5000, this.options.intervalMs * 3)) {
            this.buckets.push(bucket);
            this.buckets = this.buckets.slice(-600);
            this.persist(() => this.history.append(instance.id, bucket));
          } else {
            this.requests.samplingGap();
            this.localEvent('sampling_gap');
          }
        }
        this.previous = frame;
        this.latest = frame;
        this.engineStatus = snapshot.engine.status;
        instance.lastSeen = frame.at;
        this.persist(() => this.requests.observe(snapshot, frame.at, pollStarted));
        if (frame.at - this.lastPruned > 600_000) {
          this.persist(() => this.history.prune(frame.at));
          this.lastPruned = frame.at;
        }
      } catch (error) {
        if (this.stopSignal.signal.aborted) break;
        const message = error instanceof Error ? error.message : String(error);
        if (message !== this.error) this.log(`Upstream: ${message}`);
        if (this.connected) this.localEvent('sampling_gap', { reason: message });
        this.connected = false;
        this.error = message;
        this.previous = null;
        this.requests.samplingGap();
      }
      const remaining = this.options.intervalMs - (performance.now() - started);
      try {
        await sleep(Math.max(50, remaining), undefined, { signal: this.stopSignal.signal });
      } catch {
        break;
      }
    }
  }

  private async readEvents(instanceId: string, connectionSignal: AbortSignal): Promise<void> {
    const signal = AbortSignal.any([this.stopSignal.signal, connectionSignal]);
    let retry = 1000;
    let gap = false;
    while (!signal.aborted) {
      const opening = new AbortController();
      const timeout = setTimeout(() => opening.abort(), 5000);
      try {
        const response = await fetch(`${this.options.engineUrl}/monitor/events`, {
          signal: AbortSignal.any([signal, opening.signal]),
        });
        clearTimeout(timeout);
        if (!response.ok || !response.body) {
          await response.body?.cancel();
          throw new Error(`Events: HTTP ${response.status}`);
        }
        const reader = response.body
          .pipeThrough(new TextDecoderStream())
          .pipeThrough(new EventSourceParserStream())
          .getReader();
        try {
          while (!signal.aborted) {
            const { done, value } = await reader.read();
            if (done) break;
            const data = JSON.parse(value.data) as {
              server_instance_id: string;
              sequence?: number;
              timestamp_unix_ms: number;
              kind?: string;
              request_id?: number;
              details?: Record<string, unknown>;
            };
            if (data.server_instance_id !== instanceId) break;
            if (!data.kind) {
              this.eventsConnected = true;
              if (gap) this.localEvent('events_resumed');
              gap = false;
              retry = 1000;
              continue;
            }
            const sequence = data.sequence!;
            if (this.lastSequence !== null && sequence <= this.lastSequence) continue;
            if (this.lastSequence !== null && sequence !== this.lastSequence + 1) {
              this.localEvent('event_gap', { after: this.lastSequence, before: sequence });
            }
            this.lastSequence = sequence;
            this.recordEvent({
              key: `${instanceId}:${sequence}`,
              instanceId,
              sequence,
              at: Date.now(),
              sourceAt: data.timestamp_unix_ms,
              kind: data.kind,
              requestId: data.request_id ?? null,
              details: data.details ?? {},
            });
          }
        } finally {
          await reader.cancel().catch(() => {});
        }
      } catch (error) {
        if (!signal.aborted && !gap) {
          this.localEvent('event_gap', {
            reason: error instanceof Error ? error.message : String(error),
          });
          gap = true;
        }
      } finally {
        clearTimeout(timeout);
        this.eventsConnected = false;
      }
      if (signal.aborted) break;
      if (!gap) {
        this.localEvent('event_gap');
        gap = true;
      }
      try {
        await sleep(retry, undefined, { signal });
      } catch {
        break;
      }
      retry = Math.min(10_000, retry * 2);
    }
  }

  live(since = 0): LiveResult {
    const now = Date.now();
    const window = emptyBucket(now - 5000);
    for (const bucket of this.buckets) if (bucket.end > now - 5000) merge(window, bucket);
    return {
      now,
      instanceId: this.instance?.id ?? null,
      connected: this.connected,
      eventsConnected: this.eventsConnected,
      engineStatus: this.engineStatus,
      lastSampleAt: this.latest?.at ?? null,
      error: this.error,
      storageError: this.storageError,
      defaultTheme: this.options.defaultTheme,
      samplingIntervalMs: this.options.intervalMs,
      historyDays: this.history.days,
      fineHistoryHours: this.history.fineHours,
      summary: summarize(window, this.instance?.bounds ?? {}),
      gauges: this.latest?.gauges ?? {},
      requests: this.requests.current(),
      recentRequests: this.requests.completed(),
      events: this.events,
      points: this.buckets
        .filter((b) => b.end > since)
        .slice(-120)
        .map(chartPoint),
    };
  }

  async stop(): Promise<void> {
    this.stopSignal.abort();
    this.eventSignal?.abort();
    await Promise.allSettled([this.pollTask, this.eventTask]);
    this.persist(() => this.requests.finishObservation());
  }
}
