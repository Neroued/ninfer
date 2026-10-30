import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import type { ChartPoint } from '../../shared/types.js';

export interface Line {
  key: keyof ChartPoint;
  label: string;
  color: string;
  dash?: number[];
  width?: number;
  range?: [min: keyof ChartPoint, max: keyof ChartPoint];
}

export class TimeSeries {
  private readonly plot: uPlot;
  private readonly resize: ResizeObserver;
  private readonly keys: (keyof ChartPoint)[] = [];

  constructor(
    element: HTMLElement,
    lines: Line[],
    height: number,
    sync: string,
    select: (from: number, to: number) => void,
    hover: (at: number | null) => void,
    showTime = true,
  ) {
    const css = getComputedStyle(document.documentElement);
    const color = (key: string) => css.getPropertyValue(`--${key}`).trim();
    const series: uPlot.Series[] = [{}];
    const bands: uPlot.Band[] = [];
    // Draw unoutlined ranges first so the mean lines remain in the foreground.
    for (const line of lines) {
      if (!line.range) continue;
      const upper = series.length;
      for (const key of [line.range[1], line.range[0]]) {
        this.keys.push(key);
        series.push({
          label: key,
          stroke: color(line.color),
          width: 0,
          spanGaps: false,
          points: { show: false },
        });
      }
      bands.push({ series: [upper, upper + 1], fill: color(`${line.color}-fill`) });
    }
    const rangeCount = this.keys.length;
    for (const line of lines) {
      this.keys.push(line.key);
      series.push({
        label: line.label,
        stroke: color(line.color),
        width: line.width ?? 1.5,
        dash: line.dash,
        spanGaps: false,
        points: { show: false },
      });
    }
    this.plot = new uPlot(
      {
        width: Math.max(200, element.clientWidth),
        height,
        padding: [10, 28, 2, 0],
        legend: { show: false },
        cursor: {
          drag: { x: true, y: false, setScale: false },
          sync: { key: sync },
          points: { size: (_plot, seriesIndex) => (seriesIndex <= rangeCount ? 0 : 5) },
        },
        scales: {
          x: { time: true },
          y: { range: (_u, min, max) => [Math.min(0, min), Math.max(1, max * 1.12)] },
        },
        axes: [
          {
            show: showTime,
            stroke: color('muted'),
            size: 28,
            space: 100,
            grid: { show: false },
            ticks: { show: false },
            font: '11px "IBM Plex Mono", monospace',
            gap: 6,
            values: (plot, values) => {
              const span = (plot.scales.x.max ?? 0) - (plot.scales.x.min ?? 0);
              return values.map((value) =>
                new Date(value * 1000).toLocaleString(
                  'en-GB',
                  span > 86_400
                    ? { day: '2-digit', month: 'short' }
                    : {
                        hour: '2-digit',
                        minute: '2-digit',
                        ...(span <= 600 ? { second: '2-digit' as const } : {}),
                      },
                ),
              );
            },
          },
          {
            stroke: color('muted'),
            size: 48,
            font: '11px "IBM Plex Mono", monospace',
            grid: { stroke: color('grid'), width: 1 },
            ticks: { show: false },
            values: (_u, values) =>
              values.map((v) =>
                v >= 1000 ? `${(v / 1000).toFixed(1)}k` : `${Math.round(v * 10) / 10}`,
              ),
          },
        ],
        series,
        bands,
        hooks: {
          setSelect: [
            (plot) => {
              if (plot.select.width < 8) return;
              const from = plot.posToVal(plot.select.left, 'x') * 1000;
              const to = plot.posToVal(plot.select.left + plot.select.width, 'x') * 1000;
              plot.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false);
              select(from, to);
            },
          ],
          setCursor: [
            (plot) => hover(plot.cursor.idx == null ? null : plot.data[0][plot.cursor.idx] * 1000),
          ],
        },
      },
      [[], ...this.keys.map(() => [])] as uPlot.AlignedData,
      element,
    );
    this.resize = new ResizeObserver(() =>
      this.plot.setSize({ width: Math.max(200, element.clientWidth), height }),
    );
    this.resize.observe(element);
  }

  update(points: ChartPoint[], from: number, to: number): void {
    const data = [
      points.map((p) => p.at / 1000),
      ...this.keys.map((key) => points.map((p) => p[key] as number | null)),
    ] as uPlot.AlignedData;
    this.plot.batch(() => {
      this.plot.setData(data);
      this.plot.setScale('x', { min: from / 1000, max: to / 1000 });
    });
  }

  destroy(): void {
    this.resize.disconnect();
    this.plot.destroy();
  }
}
