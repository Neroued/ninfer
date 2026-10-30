import Fastify from 'fastify';
import staticFiles from '@fastify/static';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { Collector } from './collector.js';
import type { History } from './history.js';

type Query = {
  instance?: string;
  from?: string;
  to?: string;
  points?: string;
  limit?: string;
  cursor?: string;
  phase?: string;
  request_id?: string;
  since?: string;
};

function number(value: string | undefined, fallback: number): number {
  if (value === undefined) return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed))
    throw Object.assign(new Error('Expected a finite number'), { statusCode: 400 });
  return parsed;
}

export function createApi(collector: Collector, history: History) {
  const app = Fastify({ logger: false });
  app.addHook('onSend', async (_req, reply) => {
    reply.header('Cache-Control', 'no-store');
  });
  const range = (q: Query) => {
    const to = number(q.to, Date.now());
    const from = number(q.from, to - 300_000);
    if (from >= to || to - from > history.days * 86_400_000 + 60_000) {
      throw Object.assign(new Error(`Choose a time range of up to ${history.days} days`), {
        statusCode: 400,
      });
    }
    return { from, to };
  };
  const instance = (id?: string) => {
    const value = id ? history.instance(id) : null;
    if (!value) throw Object.assign(new Error('Instance not found'), { statusCode: 404 });
    return value;
  };
  app.get('/api/instances', async () => history.instances());
  app.get<{ Params: { id: string } }>('/api/instances/:id', async (req) => instance(req.params.id));
  app.get<{ Querystring: Query }>('/api/live', async (req) =>
    collector.live(number(req.query.since, 0)),
  );
  app.get<{ Querystring: Query }>('/api/history', async (req, reply) => {
    const q = req.query;
    const { from, to } = range(q);
    const cancelled = new AbortController();
    const close = () => {
      if (!reply.raw.writableFinished) cancelled.abort();
    };
    reply.raw.once('close', close);
    try {
      return await history.query(
        instance(q.instance),
        from,
        to,
        Math.max(100, Math.min(1500, number(q.points, 1000))),
        cancelled.signal,
      );
    } finally {
      reply.raw.removeListener('close', close);
    }
  });
  app.get<{ Querystring: Query }>('/api/requests', async (req) => {
    const q = req.query;
    const { from, to } = range(q);
    return history.requestPage(
      instance(q.instance).id,
      from,
      to,
      Math.max(1, Math.min(200, Math.floor(number(q.limit, 50)))),
      q.cursor,
      q.phase,
    );
  });
  app.get<{ Querystring: Query }>('/api/events', async (req) => {
    const q = req.query;
    const { from, to } = range(q);
    return history.eventPage(
      instance(q.instance).id,
      from,
      to,
      Math.max(1, Math.min(200, Math.floor(number(q.limit, 50)))),
      q.cursor,
      q.request_id === undefined ? undefined : number(q.request_id, 0),
    );
  });
  const root = fileURLToPath(new URL('../web', import.meta.url));
  if (existsSync(`${root}/index.html`)) {
    app.register(staticFiles, { root });
  } else {
    app.get('/', async (_req, reply) =>
      reply
        .type('text/plain')
        .send('Build the dashboard with npm run build, or use the Vite development server.'),
    );
  }
  return app;
}
