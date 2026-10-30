<script setup lang="ts">
import type { RequestRow } from '../../shared/types.js';
import { age, ms, number, tokens } from '../format.js';

defineProps<{
  rows: RequestRow[];
  completed: RequestRow[];
  maxContext: number;
  now: number;
  historical: boolean;
  more: boolean;
}>();
const emit = defineEmits<{ select: [row: RequestRow]; loadMore: [] }>();

function trace(row: RequestRow): string {
  if (row.trace.length < 2) return '';
  const maximum = Math.max(1, ...row.trace.map((point) => point.speed));
  const start = row.trace[0].at;
  const span = Math.max(1, row.trace.at(-1)!.at - start);
  return row.trace
    .map(
      (point, i) =>
        `${i ? 'L' : 'M'}${(((point.at - start) * 96) / span).toFixed(1)},${(20 - (point.speed / maximum) * 16).toFixed(1)}`,
    )
    .join(' ');
}
</script>

<template>
  <section class="requests-panel">
    <div class="section-heading">
      <span class="label">{{ historical ? 'Request history' : 'Requests' }}</span
      ><span class="mono muted">{{ rows.length }} {{ historical ? 'shown' : 'active' }}</span>
    </div>
    <div class="request-scroll">
      <div class="request-grid request-head label">
        <span>Phase</span><span>ID</span><span>Age</span><span>Context</span><span>Speed</span
        ><span>TTFT</span><span>Gen</span><span class="trace-column">Trace</span>
      </div>
      <template v-for="(group, index) in [rows, completed]" :key="index">
        <div v-if="index && group.length" class="request-divider label">Recently completed</div>
        <button
          v-for="row in group"
          :key="`${row.instanceId}:${row.id}`"
          class="request-grid request-row"
          :class="[`phase-${row.phase}`, { finished: row.terminal }]"
          @click="emit('select', row)"
        >
          <span class="request-phase mono">{{ row.phase }}</span>
          <span class="request-id mono">#{{ row.id }}</span>
          <span class="request-age mono">{{
            age(
              ((row.endedAt ?? (row.terminal ? row.lastSeen : now)) -
                (row.receivedAt ?? row.firstSeen)) /
                1000,
            )
          }}</span>
          <span class="request-context"
            ><span class="context-track"
              ><span
                :style="{
                  width: `${Math.min(100, ((row.contextTokens ?? 0) / Math.max(1, maxContext)) * 100)}%`,
                }" /></span
            ><span class="mono"
              >{{ tokens(row.contextTokens)
              }}<span class="muted"> / {{ tokens(maxContext) }}</span></span
            ></span
          >
          <span class="request-speed mono"
            >{{ number(row.speed) }}<small v-if="row.speed !== null"> tok/s</small></span
          >
          <span class="request-ttft mono">{{ ms(row.ttftMs) }}</span>
          <span class="request-gen mono">{{ tokens(row.generatedTokens) }}</span>
          <svg class="trace-column request-trace" viewBox="0 0 100 24" aria-hidden="true">
            <path :d="trace(row)" />
          </svg>
        </button>
      </template>
      <div v-if="!rows.length && !completed.length" class="empty-state">
        {{ historical ? 'No observed requests in this range' : 'No active requests' }}
      </div>
      <button v-if="more" class="text-button load-more" @click="emit('loadMore')">
        Load more requests
      </button>
    </div>
  </section>
</template>
