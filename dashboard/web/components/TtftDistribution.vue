<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import type { Distribution } from '../../shared/types.js';
import { ms } from '../format.js';

const props = defineProps<{ ttft: Distribution | null | undefined }>();

const plot = ref<HTMLDivElement | null>(null);
const width = ref(400);
let observer: ResizeObserver | null = null;

onMounted(() => {
  if (!plot.value) return;
  width.value = Math.max(280, plot.value.clientWidth);
  observer = new ResizeObserver(([entry]) => {
    width.value = Math.max(280, entry.contentRect.width);
  });
  observer.observe(plot.value);
});

onBeforeUnmount(() => observer?.disconnect());

const geometry = computed(() => {
  const ttft = props.ttft;
  if (!ttft || ttft.count <= 0 || !ttft.bins.length) return null;
  const populated = ttft.bins
    .map((count, index) => ({ count, index }))
    .filter((bin) => bin.count > 0);
  if (!populated.length) return null;

  const bounds = ttft.bounds;
  const first = populated[0].index;
  const last = populated.at(-1)!.index;
  const lower = first === 0 ? 0 : (bounds[first - 1] ?? 0);
  const upper = bounds[last] ?? bounds.at(-1) ?? 0;
  const minMs = Math.max(1, lower);
  const maxMs = Math.max(
    1000,
    upper,
    (ttft.p90 ?? 0) * 1.5,
    (ttft.p99 ?? 0) * 1.2,
  );

  const left = 0;
  const right = Math.max(280, width.value) - 8;
  const top = 10;
  const bottom = 48;
  const logMin = Math.log10(minMs);
  const logMax = Math.log10(maxMs);
  const x = (value: number) =>
    left +
    ((Math.log10(Math.max(minMs, Math.min(maxMs, value))) - logMin) / (logMax - logMin)) *
      (right - left);

  const counts = ttft.bins.slice(first, last + 1);
  const maxCount = Math.max(1, ...counts);
  const bars = counts.map((count, offset) => {
    const index = first + offset;
    const binLower = Math.max(minMs, index === 0 ? 0 : (bounds[index - 1] ?? minMs));
    const binUpper = Math.min(maxMs, bounds[index] ?? maxMs);
    return {
      key: index,
      count,
      x: x(binLower),
      width: Math.max(1, x(binUpper) - x(binLower) - 1),
      y: bottom - (count / maxCount) * (bottom - top),
      height: (count / maxCount) * (bottom - top),
      title: `${ms(binLower)} – ${binUpper >= maxMs ? '∞' : ms(binUpper)}: ${count} request${count === 1 ? '' : 's'}`,
    };
  });

  const markers = [
    { kind: 'p50', value: ttft.p50, percent: 0.5 },
    { kind: 'p90', value: ttft.p90, percent: 0.9 },
    { kind: 'p99', value: ttft.p99, percent: 0.99 },
  ].filter(
    (item): item is { kind: 'p50' | 'p90' | 'p99'; value: number; percent: number } =>
      item.value != null,
  );

  const tickValues = [...new Set([lower, 100, 1000, 5000, upper])]
    .filter((value) => value >= minMs && value <= maxMs)
    .sort((a, b) => a - b);
  const ticks = tickValues.map((value) => ({ value, text: ms(value) }));

  const aria = `TTFT distribution, ${ttft.count} samples, p50 ${ms(ttft.p50)}, p90 ${ms(ttft.p90)}, p99 ${ms(ttft.p99)}`;
  return { bars, markers, ticks, x, aria, minMs, maxMs, width: right + 8 };
});
</script>

<template>
  <div ref="plot" class="ttft-pdf" role="img" :aria-label="geometry?.aria ?? 'No first-token samples'">
    <svg
      v-if="geometry"
      :viewBox="`0 0 ${geometry.width} 64`"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <g class="ttft-pdf-bars">
        <rect
          v-for="bar in geometry.bars"
          :key="bar.key"
          class="ttft-pdf-bar"
          :x="bar.x"
          :y="bar.y"
          :width="bar.width"
          :height="bar.height"
        >
          <title>{{ bar.title }}</title>
        </rect>
      </g>
      <line
        v-for="marker in geometry.markers"
        :key="marker.kind"
        :class="['ttft-marker-line', marker.kind]"
        :x1="geometry.x(marker.value)"
        :x2="geometry.x(marker.value)"
        :y1="10"
        :y2="48"
      />
      <g class="ttft-axis">
        <text
          v-for="tick in geometry.ticks"
          :key="tick.value"
          :x="geometry.x(tick.value)"
          y="60"
          :text-anchor="tick.value === geometry.minMs ? 'start' : tick.value === geometry.maxMs ? 'end' : 'middle'"
        >
          {{ tick.text }}
        </text>
      </g>
    </svg>
    <span v-else class="ttft-pdf-empty mono">No first-token samples</span>
  </div>
</template>
