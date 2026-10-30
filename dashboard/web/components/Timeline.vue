<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import type { ChartPoint } from '../../shared/types.js';
import { TimeSeries } from '../charts/timeseries.js';
import { themeId } from '../theme.js';
import { clock, ms, number } from '../format.js';

const props = defineProps<{
  points: ChartPoint[];
  from: number;
  to: number;
  aggregationMs: number;
}>();
const emit = defineEmits<{ select: [from: number, to: number] }>();
const metric = ref('throughput');
const primary = ref<HTMLElement>();
const secondary = ref<HTMLElement>();
const hovered = ref<number | null>(null);
let charts: TimeSeries[] = [];
const aggregation = computed(() => {
  const seconds = props.aggregationMs / 1000;
  if (seconds >= 3600) return `${number(seconds / 3600, 0)}h`;
  if (seconds >= 60) return `${number(seconds / 60, 0)}m`;
  return `${number(seconds, Number.isInteger(seconds) ? 0 : 1)}s`;
});
const point = computed(() => {
  if (hovered.value === null) return props.points.at(-1);
  return props.points.reduce<ChartPoint | undefined>(
    (best, p) =>
      !best || Math.abs(p.at - hovered.value!) < Math.abs(best.at - hovered.value!) ? p : best,
    undefined,
  );
});

async function create() {
  charts.forEach((chart) => chart.destroy());
  charts = [];
  await nextTick();
  if (!primary.value) return;
  const select = (from: number, to: number) => emit('select', from, to);
  const hover = (at: number | null) => {
    hovered.value = at;
  };
  if (metric.value === 'throughput') {
    if (secondary.value)
      charts.push(
        new TimeSeries(
          secondary.value,
          [
            {
              key: 'prefill',
              label: 'Prefill',
              color: 'prefill',
              range: ['prefillMin', 'prefillMax'],
            },
          ],
          73,
          'main',
          select,
          hover,
          false,
        ),
      );
    charts.push(
      new TimeSeries(
        primary.value,
        [
          {
            key: 'decode',
            label: 'Decode',
            color: 'decode',
            width: 1.8,
            range: ['decodeMin', 'decodeMax'],
          },
        ],
        159,
        'main',
        select,
        hover,
      ),
    );
  } else {
    charts.push(
      new TimeSeries(
        primary.value,
        metric.value === 'step'
          ? [
              {
                key: 'step',
                label: 'Decode step · ms',
                color: 'decode',
                range: ['stepMin', 'stepMax'],
              },
            ]
          : [
              {
                key: 'running',
                label: 'Running',
                color: 'decode',
                range: ['runningMin', 'runningMax'],
              },
              { key: 'waiting', label: 'Waiting', color: 'muted', dash: [3, 3] },
              { key: 'prefilling', label: 'Prefilling', color: 'prefill' },
            ],
        232,
        'main',
        select,
        hover,
      ),
    );
  }
  update();
}

function update() {
  charts.forEach((chart) => chart.update(props.points, props.from, props.to));
}
watch([metric, themeId], create);
watch(() => [props.points, props.from, props.to], update);
onMounted(create);
onBeforeUnmount(() => charts.forEach((chart) => chart.destroy()));
</script>

<template>
  <section class="timeline">
    <div class="section-heading chart-heading">
      <div class="tabs" aria-label="Chart metric">
        <button
          v-for="item in ['throughput', 'step', 'load']"
          :key="item"
          :class="{ selected: metric === item }"
          @click="metric = item"
        >
          {{ item === 'step' ? 'decode step' : item }}
        </button>
      </div>
      <div class="chart-readout mono">
        <div>
          <span v-if="hovered !== null && point">{{ clock(point.at) }} · </span>
          <template v-if="metric === 'throughput'"
            ><b class="decode">{{ number(point?.decode) }}</b> decode <span class="dot-sep">·</span>
            <b class="prefill">{{ number(point?.prefill, 0) }}</b> prefill
            <span class="muted">tok/s</span></template
          >
          <template v-else-if="metric === 'step'"
            >{{ ms(point?.step) }} <span class="muted">per step</span></template
          >
          <template v-else
            >{{ number(point?.running, 1) }} running ·
            {{ number(point?.waiting, 1) }} waiting</template
          >
          <span class="muted"> avg</span>
        </div>
        <div class="chart-extrema" :class="{ visible: hovered !== null }">
          <template v-if="metric === 'throughput'">
            min–max {{ number(point?.decodeMin, 0) }}–{{ number(point?.decodeMax, 0) }} decode ·
            {{ number(point?.prefillMin, 0) }}–{{ number(point?.prefillMax, 0) }} prefill
          </template>
          <template v-else-if="metric === 'step'">
            min–max {{ ms(point?.stepMin) }}–{{ ms(point?.stepMax) }}
          </template>
          <template v-else>
            min–max {{ number(point?.runningMin, 0) }}–{{ number(point?.runningMax, 0) }} running
          </template>
        </div>
      </div>
    </div>
    <div class="plot-stack">
      <div v-if="metric === 'throughput'" ref="secondary" class="plot plot-top" />
      <div ref="primary" class="plot" />
      <div v-if="!points.some((p) => p.coveredMs > 0)" class="plot-empty">
        Waiting for observations
      </div>
    </div>
    <div class="chart-caption mono">
      <span>{{ clock(from, true) }}</span
      ><span>{{ aggregation }} averages · band: sampled min–max · drag to explore</span
      ><span>{{ clock(to, true) }}</span>
    </div>
  </section>
</template>
