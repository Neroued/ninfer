# NInfer Monitor

A standalone dashboard for NInfer's monitoring API. The collector runs continuously,
including when no browser is open. Browsers share the same history.

## Run

Requires Node.js 22.12 or newer. From this directory:

```bash
npm ci
npm run build
cp config.example.json config.local.json
# Set engine_url to the ninfer-serve coordinator URL.
npm start -- --config config.local.json
```

Open `http://localhost:8899`. A distributed engine needs only its coordinator URL.
The dashboard can run on a different machine from the engine.

The dashboard reads `/monitor/info`, `/monitor/snapshot`, and `/monitor/events`.
It has no authentication or engine-control actions. GPU timing details are available
when GPU timing is enabled in the engine; missing measurements appear as `—`.

Configuration paths are relative to the configuration file. Defaults in the example:

| Setting                | Meaning                                                       |
| ---------------------- | ------------------------------------------------------------- |
| `engine_url`           | NInfer Serve HTTP endpoint                                    |
| `host`, `port`         | Dashboard listening address, `0.0.0.0:8899`                   |
| `database`             | Local SQLite file, `data/monitor.sqlite`                      |
| `sampling_interval_ms` | Snapshot polling interval, 1000 ms; at least 1000 ms          |
| `fine_history_hours`   | Retention of individual samples, 24 hours                     |
| `history_days`         | Retention of minute aggregates, 30 days                       |
| `default_theme`        | Initial theme, `ember`; a browser preference takes precedence |

`config.local.json`, `data/`, `dist/`, and `node_modules/` are ignored by Git. Keep the
database and its SQLite sidecar files together on the collector's local filesystem.
Metric payloads use lossless compression; time and instance keys remain indexed in SQLite.
Hour summaries accelerate wide queries while minute observations remain available for zooming.
Completed request summaries and events expire after `history_days`, with additional
limits of 100,000 completed requests and 500,000 events. These limits do not shorten
the metric history. Retention cleanup runs every ten minutes.

## Use

- **Live:** throughput numbers show the latest five seconds. Resource and request
  observations update each second. Window distributions refresh every ten seconds,
  or every minute for ranges longer than an hour.
- **History:** choose a range or drag across a chart. Curves and summaries describe
  the selected interval. The footer identifies the stored resolution and observed time.
- **Pause:** freezes the page; collection continues.
- **Requests:** click a row for its observed lifecycle and summary.
- **Details:** actual-batch decode timings, execution phases, per-rank GPU intervals,
  resources, and effective configuration.
- **Instance selector:** inspect history from earlier engine runs.

Rates use counter deltas and the actual observed interval. Decode throughput counts
tokens committed by decode rounds; the first token produced by prefill is excluded.
Prefill throughput in the main chart is work per wall second; execution efficiency is
shown separately in Details. Work is recorded at its engine publication boundary.

Latency percentiles are estimates from the engine's histogram buckets. A percentile
falling in the unbounded overflow bucket is shown as `—`; bucket counts remain visible.
Charts use wall-clock-aligned averages, from 1s at a 1m range to 5s at 15m and 5m at 24h.
Throughput is weighted by observed time; decode step latency is weighted by step count.
Faint bands and hover readouts show sampled minima and maxima. Live updates use the same
aggregation as history, and zooming restores detail within the stored resolution.
Window summaries preserve counts, elapsed totals, histogram bins and time-weighted gauges.
Queries select complete stored observations by their end time.

Context usage represents physical backing and includes reservations, retirement and
layout space. KV and State are subsets of the pool. Token capacity is the engine's
estimate. GPU intervals are per measured interval and can overlap. Communication
payload is the quantity passed to collectives, not measured network traffic.

Engine restarts create separate histories. Connection gaps stay unobserved, and live
events have no upstream replay. Request fields that were not observed remain unknown.
Request traces retain the latest 60 uninterrupted sampled rates; histories do not store prompts or
generated text.

## Development

```bash
npm run dev:server -- --config config.local.json
# In another terminal; proxies /api to port 8899:
npm run dev
```

`npm run check` checks TypeScript and Vue types. `npm run format` formats source files.
There is no test framework in this application.

`server/` owns collection, statistics and SQLite. `web/` owns presentation and
interaction. `shared/types.ts` is the dashboard's browser/server contract. The engine's
JSON representation is adapted in `server/ninfer.ts`.

## Themes

Copy `web/themes/paper.json` or `ember.json`, choose a unique `id` and `name`, and change
the semantic color values. Rebuild; the new theme appears in the selector. Components
and charts use those same colors. Layout geometry lives separately in
`web/styles/split.css` and `web/layouts/SplitLayout.vue`.

Instrument Sans and IBM Plex Mono are bundled locally under their accompanying SIL
Open Font Licenses in `public/fonts/`.
