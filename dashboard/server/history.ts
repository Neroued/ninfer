import Database from 'better-sqlite3';
import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { setImmediate } from 'node:timers/promises';
import { deflateRawSync, inflateRawSync } from 'node:zlib';
import type {
  Bucket,
  ChartPoint,
  HistoryResult,
  Instance,
  InstanceListEntry,
  MonitorEvent,
  Page,
  RequestRow,
} from '../shared/types.js';
import { appendChartPoints, chartInterval } from '../shared/timeline.js';
import { chartPoint, emptyBucket, merge, summarize } from './statistics.js';

interface MetricRow {
  end_ms: number;
  data: Buffer;
}
interface DataRow {
  data: string;
}

export class History {
  private readonly db: Database.Database;
  private readonly aggregates = new Map<
    number,
    { instanceId: string; key: number; bucket: Bucket }
  >();

  constructor(
    path: string,
    readonly fineHours: number,
    readonly days: number,
  ) {
    mkdirSync(dirname(path), { recursive: true });
    this.db = new Database(path);
    this.db.pragma('journal_mode = WAL');
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS instances (
        id TEXT PRIMARY KEY, first_seen INTEGER NOT NULL, last_seen INTEGER NOT NULL,
        info TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS metrics (
        instance_id TEXT NOT NULL, resolution INTEGER NOT NULL, end_ms INTEGER NOT NULL,
        data BLOB NOT NULL, PRIMARY KEY(instance_id, resolution, end_ms)
      ) WITHOUT ROWID;
      CREATE INDEX IF NOT EXISTS metrics_expiry ON metrics(resolution, end_ms);
      CREATE TABLE IF NOT EXISTS requests (
        instance_id TEXT NOT NULL, id INTEGER NOT NULL, first_seen INTEGER NOT NULL,
        last_seen INTEGER NOT NULL, ended_at INTEGER, terminal INTEGER NOT NULL,
        phase TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY(instance_id, id)
      );
      CREATE INDEX IF NOT EXISTS request_time ON requests(instance_id, first_seen DESC, id DESC);
      CREATE INDEX IF NOT EXISTS request_expiry ON requests(terminal, last_seen DESC);
      CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY, event_key TEXT NOT NULL UNIQUE, instance_id TEXT NOT NULL,
        at INTEGER NOT NULL, request_id INTEGER, data TEXT NOT NULL
      );
      CREATE INDEX IF NOT EXISTS event_time ON events(instance_id, at DESC, id DESC);
      CREATE INDEX IF NOT EXISTS event_request ON events(instance_id, request_id, id DESC);
      CREATE INDEX IF NOT EXISTS event_expiry ON events(at DESC, id DESC);
    `);
    const unfinished = this.db
      .prepare('SELECT data FROM requests WHERE terminal = 0')
      .all() as DataRow[];
    this.db.transaction(() => {
      for (const row of unfinished) {
        const request = JSON.parse(row.data) as RequestRow;
        request.phase = 'unknown';
        request.terminal = true;
        request.speed = null;
        this.saveRequest(request);
      }
    })();
  }

  instance(id: string): Instance | null {
    const row = this.db
      .prepare('SELECT info, first_seen, last_seen FROM instances WHERE id = ?')
      .get(id) as { info: string; first_seen: number; last_seen: number } | undefined;
    return row
      ? { ...JSON.parse(row.info), firstSeen: row.first_seen, lastSeen: row.last_seen }
      : null;
  }

  saveInstance(instance: Instance): void {
    this.db
      .prepare(
        `INSERT INTO instances VALUES (?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET
      last_seen = excluded.last_seen, info = excluded.info`,
      )
      .run(instance.id, instance.firstSeen, instance.lastSeen, JSON.stringify(instance));
  }

  instances(): InstanceListEntry[] {
    const rows = this.db
      .prepare('SELECT id, info, first_seen, last_seen FROM instances ORDER BY last_seen DESC')
      .all() as { id: string; info: string; first_seen: number; last_seen: number }[];
    return rows.map((row) => {
      const instance = JSON.parse(row.info) as Instance;
      return {
        id: row.id,
        model: instance.info.model.name,
        architecture: instance.info.model.architecture,
        ranks: instance.info.ranks.length,
        firstSeen: row.first_seen,
        lastSeen: row.last_seen,
      };
    });
  }

  append(instanceId: string, bucket: Bucket): void {
    // Assign complete observations by end time. Hour summaries accelerate wide queries;
    // minute data remains available throughout the retention window.
    const updates = [60_000, 3_600_000].map((resolution) => {
      const key = Math.ceil(bucket.end / resolution) * resolution;
      let cached = this.aggregates.get(resolution);
      if (!cached || cached.instanceId !== instanceId || cached.key !== key) {
        const row = this.db
          .prepare(
            'SELECT data FROM metrics WHERE instance_id = ? AND resolution = ? AND end_ms = ?',
          )
          .get(instanceId, resolution, key) as Pick<MetricRow, 'data'> | undefined;
        cached = {
          instanceId,
          key,
          bucket: row ? decodeBucket(row.data) : emptyBucket(bucket.start),
        };
      }
      const next = structuredClone(cached.bucket);
      merge(next, bucket);
      return { resolution, instanceId, key, bucket: next };
    });
    this.db.transaction(() => {
      const insert = this.db.prepare('INSERT OR REPLACE INTO metrics VALUES (?, ?, ?, ?)');
      insert.run(instanceId, 0, bucket.end, encodeBucket(bucket));
      for (const update of updates)
        insert.run(instanceId, update.resolution, update.key, encodeBucket(update.bucket));
      this.db
        .prepare('UPDATE instances SET last_seen = ? WHERE id = ?')
        .run(bucket.end, instanceId);
    })();
    for (const update of updates) this.aggregates.set(update.resolution, update);
  }

  async query(
    instance: Instance,
    from: number,
    to: number,
    maxPoints: number,
    signal?: AbortSignal,
  ): Promise<HistoryResult> {
    const raw = from >= Date.now() - this.fineHours * 3_600_000 && to - from <= maxPoints * 10_000;
    const hourly = to - from >= 2 * 86_400_000;
    const resolutionMs = raw ? instance.samplingIntervalMs : hourly ? 3_600_000 : 60_000;
    const aggregationMs = chartInterval(to - from, resolutionMs, maxPoints);
    const summary = emptyBucket(from);
    const points: ChartPoint[] = [];
    const select = this.db.prepare(`SELECT end_ms, data FROM metrics
      WHERE instance_id = ? AND resolution = ? AND end_ms > ? AND end_ms <= ?
      ORDER BY end_ms LIMIT 256`);
    const upper = raw ? to : Math.ceil(to / 60_000) * 60_000;
    const firstHour = Math.ceil(from / 3_600_000) * 3_600_000;
    const lastHour = Math.floor(to / 3_600_000) * 3_600_000;
    const segments = hourly
      ? [
          { resolution: 60_000, from, to: firstHour },
          { resolution: 3_600_000, from: firstHour, to: lastHour },
          { resolution: 60_000, from: lastHour, to: upper },
        ]
      : [{ resolution: raw ? 0 : 60_000, from, to: upper }];
    for (const segment of segments) {
      let cursor = segment.from;
      while (cursor < segment.to) {
        signal?.throwIfAborted();
        const rows = select.all(instance.id, segment.resolution, cursor, segment.to) as MetricRow[];
        if (!rows.length) break;
        for (const row of rows) {
          const bucket = decodeBucket(row.data);
          if (bucket.end > to || bucket.end <= from) continue;
          merge(summary, bucket);
          appendChartPoints(points, [chartPoint(bucket)], aggregationMs);
        }
        cursor = rows[rows.length - 1].end_ms;
        await setImmediate();
      }
    }
    return {
      instanceId: instance.id,
      from,
      to,
      resolutionMs,
      aggregationMs,
      points,
      summary: summarize(summary, instance.bounds),
    };
  }

  request(instanceId: string, id: number): RequestRow | null {
    const row = this.db
      .prepare('SELECT data FROM requests WHERE instance_id = ? AND id = ?')
      .get(instanceId, id) as DataRow | undefined;
    return row ? (JSON.parse(row.data) as RequestRow) : null;
  }

  saveRequest(request: RequestRow): void {
    this.db
      .prepare(
        `INSERT INTO requests VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(instance_id, id) DO UPDATE SET last_seen = excluded.last_seen,
      ended_at = excluded.ended_at, terminal = excluded.terminal,
      phase = excluded.phase, data = excluded.data`,
      )
      .run(
        request.instanceId,
        request.id,
        request.firstSeen,
        request.lastSeen,
        request.endedAt,
        Number(request.terminal),
        request.phase,
        JSON.stringify(request),
      );
  }

  requestPage(
    instanceId: string,
    from: number,
    to: number,
    limit: number,
    cursor?: string,
    phase?: string,
  ): Page<RequestRow> {
    const [time, id] = cursor
      ? decodeCursor(cursor)
      : [Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER];
    const rows = this.db
      .prepare(
        `SELECT data, first_seen, id FROM requests WHERE instance_id = ?
      AND first_seen <= ? AND COALESCE(ended_at, last_seen) >= ?
      AND (first_seen < ? OR (first_seen = ? AND id < ?)) ${phase ? 'AND phase = ?' : ''}
      ORDER BY first_seen DESC, id DESC LIMIT ?`,
      )
      .all(instanceId, to, from, time, time, id, ...(phase ? [phase] : []), limit + 1) as {
      data: string;
      first_seen: number;
      id: number;
    }[];
    const items = rows.slice(0, limit);
    const last = items.at(-1);
    return {
      items: items.map((r) => JSON.parse(r.data) as RequestRow),
      next: rows.length > limit && last ? encodeCursor([last.first_seen, last.id]) : null,
    };
  }

  saveEvent(event: MonitorEvent): boolean {
    return (
      this.db
        .prepare(
          'INSERT OR IGNORE INTO events(event_key, instance_id, at, request_id, data) VALUES (?, ?, ?, ?, ?)',
        )
        .run(event.key, event.instanceId, event.at, event.requestId, JSON.stringify(event))
        .changes > 0
    );
  }

  eventPage(
    instanceId: string,
    from: number,
    to: number,
    limit: number,
    cursor?: string,
    requestId?: number,
  ): Page<MonitorEvent> {
    const [time, id] = cursor
      ? decodeCursor(cursor)
      : [Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER];
    const rows = this.db
      .prepare(
        `SELECT id, at, data FROM events WHERE instance_id = ? AND at >= ? AND at <= ?
      AND (at < ? OR (at = ? AND id < ?)) ${requestId !== undefined ? 'AND request_id = ?' : ''}
      ORDER BY at DESC, id DESC LIMIT ?`,
      )
      .all(
        instanceId,
        from,
        to,
        time,
        time,
        id,
        ...(requestId !== undefined ? [requestId] : []),
        limit + 1,
      ) as { data: string; at: number; id: number }[];
    const items = rows.slice(0, limit);
    const last = items.at(-1);
    return {
      items: items.map((r) => JSON.parse(r.data) as MonitorEvent),
      next: rows.length > limit && last ? encodeCursor([last.at, last.id]) : null,
    };
  }

  prune(now: number): void {
    const cutoff = now - this.days * 86_400_000;
    this.db.transaction(() => {
      this.db
        .prepare('DELETE FROM metrics WHERE resolution = 0 AND end_ms < ?')
        .run(now - this.fineHours * 3_600_000);
      this.db.prepare('DELETE FROM metrics WHERE resolution > 0 AND end_ms < ?').run(cutoff);
      this.db.prepare('DELETE FROM requests WHERE terminal = 1 AND last_seen < ?').run(cutoff);
      this.db.exec(`DELETE FROM requests WHERE rowid IN (
        SELECT rowid FROM requests WHERE terminal = 1 ORDER BY last_seen DESC LIMIT -1 OFFSET 100000)`);
      this.db.prepare('DELETE FROM events WHERE at < ?').run(cutoff);
      this.db.exec(
        'DELETE FROM events WHERE id IN (SELECT id FROM events ORDER BY at DESC LIMIT -1 OFFSET 500000)',
      );
      this.db
        .prepare(
          `DELETE FROM instances WHERE last_seen < ?
        AND NOT EXISTS (SELECT 1 FROM metrics WHERE instance_id = instances.id)
        AND NOT EXISTS (SELECT 1 FROM requests WHERE instance_id = instances.id)
        AND NOT EXISTS (SELECT 1 FROM events WHERE instance_id = instances.id)`,
        )
        .run(cutoff);
    })();
  }

  close(): void {
    this.db.close();
  }
}

function encodeCursor(value: number[]): string {
  return Buffer.from(JSON.stringify(value)).toString('base64url');
}

function encodeBucket(value: Bucket): Buffer {
  return deflateRawSync(JSON.stringify(value), { level: 1 });
}

function decodeBucket(value: Buffer): Bucket {
  return JSON.parse(inflateRawSync(value).toString('utf8')) as Bucket;
}
function decodeCursor(value: string): [number, number] {
  const pair: unknown = JSON.parse(Buffer.from(value, 'base64url').toString());
  if (!Array.isArray(pair) || pair.length !== 2 || pair.some((v) => !Number.isFinite(v))) {
    throw new Error('Invalid pagination cursor');
  }
  return pair as [number, number];
}
