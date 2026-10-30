import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { Collector } from './collector.js';
import { History } from './history.js';
import { createApi } from './api.js';

interface Config {
  engine_url: string;
  host: string;
  port: number;
  database: string;
  sampling_interval_ms: number;
  fine_history_hours: number;
  history_days: number;
  default_theme: string;
}

function readConfig(): Config {
  const { values } = parseArgs({
    options: {
      config: { type: 'string', default: 'config.example.json' },
      help: { type: 'boolean', short: 'h' },
    },
  });
  if (values.help) {
    console.log(
      'NInfer Monitor\n  npm start -- --config config.local.json\n\nPaths are relative to the configuration file.',
    );
    process.exit(0);
  }
  const path = resolve(values.config);
  const config = JSON.parse(readFileSync(path, 'utf8')) as Config;
  const url = new URL(config.engine_url);
  if (!['http:', 'https:'].includes(url.protocol))
    throw new Error('engine_url must use HTTP or HTTPS');
  config.engine_url = config.engine_url.replace(/\/+$/, '');
  for (const field of [
    'port',
    'sampling_interval_ms',
    'fine_history_hours',
    'history_days',
  ] as const) {
    if (!Number.isFinite(config[field]) || config[field] <= 0)
      throw new Error(`${field} must be positive`);
  }
  if (!Number.isInteger(config.port) || config.port > 65535)
    throw new Error('port must be an integer from 1 to 65535');
  if (config.sampling_interval_ms < 1000)
    throw new Error('sampling_interval_ms must be at least 1000');
  if (config.fine_history_hours > config.history_days * 24)
    throw new Error('fine_history_hours exceeds history_days');
  config.database = resolve(dirname(path), config.database);
  return config;
}

async function main() {
  const config = readConfig();
  const history = new History(config.database, config.fine_history_hours, config.history_days);
  const log = (message: string) => console.log(`${new Date().toISOString()} ${message}`);
  const collector = new Collector(
    {
      engineUrl: config.engine_url,
      intervalMs: config.sampling_interval_ms,
      defaultTheme: config.default_theme,
    },
    history,
    log,
  );
  const app = createApi(collector, history);
  try {
    await app.listen({ host: config.host, port: config.port });
  } catch (error) {
    await app.close();
    history.close();
    throw error;
  }
  log(`NInfer Monitor · http://${config.host}:${config.port} · ${config.engine_url}`);
  log(
    `History · ${config.database} · ${config.fine_history_hours}h fine / ${config.history_days}d total`,
  );
  collector.start();
  let closing = false;
  const close = async () => {
    if (closing) return;
    closing = true;
    await collector.stop();
    await app.close();
    history.close();
  };
  for (const signal of ['SIGINT', 'SIGTERM'] as const) {
    process.on(signal, () => {
      close().catch((error) => {
        console.error(error);
        process.exitCode = 1;
      });
    });
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
