<script setup lang="ts">
import { computed } from 'vue';
import type { Instance, MonitorEvent, Numbers, Summary } from '../../shared/types.js';
import { age, bytes, label, ms, number, sampleAge, tokens } from '../format.js';
import TtftDistribution from './TtftDistribution.vue';

const props = defineProps<{
  instance: Instance | null;
  summary: Summary | null;
  gauges: Numbers;
  events: MonitorEvent[];
  now: number;
  rangeLabel: string;
  mode: 'live' | 'paused' | 'history';
}>();
const emit = defineEmits<{ details: []; events: [] }>();
const devicePools = computed(
  () =>
    props.instance?.info.ranks.map((rank) => {
      const g = props.gauges;
      const prefix = `pool.device.${rank.rank}.`;
      const capacity = g[`${prefix}capacity_bytes`] ?? 0;
      return {
        rank: rank.rank,
        capacity,
        occupied: g[`${prefix}occupied_bytes`] ?? 0,
        live: g[`${prefix}live_bytes`] ?? 0,
        reserved: g[`${prefix}reserved_bytes`] ?? 0,
        pending: g[`${prefix}pending_retirement_bytes`] ?? 0,
        padding: (g[`${prefix}padding_bytes`] ?? 0) + (g[`${prefix}tail_bytes`] ?? 0),
        kv: g[`${prefix}kv_live_bytes`] ?? 0,
        state: g[`${prefix}state_live_bytes`] ?? 0,
      };
    }) ?? [],
);
const total = computed(() =>
  devicePools.value.reduce(
    (result, pool) => ({
      capacity: result.capacity + pool.capacity,
      live: result.live + pool.live,
      reserved: result.reserved + pool.reserved,
      pending: result.pending + pool.pending,
    }),
    { capacity: 0, live: 0, reserved: 0, pending: 0 },
  ),
);
const ttft = computed(() => props.summary?.ttft);
const nowScope = computed(() =>
  props.mode === 'paused' ? 'paused' : props.mode === 'history' ? 'at end' : 'now',
);
const eventScope = computed(() =>
  props.mode === 'paused' ? 'paused' : props.mode === 'history' ? 'in range' : 'live',
);
</script>

<template>
  <aside class="telemetry-rail">
    <section class="instrument instrument--context">
      <div class="instrument-heading">
        <span class="label">Context pool <span class="scope">{{ nowScope }}</span></span
        ><span class="instrument-value">{{ number(gauges.context_percent) }}<small>%</small></span>
      </div>
      <p class="instrument-note mono">
        {{ bytes(total.live) }} live · {{ bytes(total.reserved) }} reserved
      </p>
      <div
        v-for="pool in devicePools"
        :key="pool.rank"
        class="pool-row"
        :title="`R${pool.rank}: KV ${bytes(pool.kv)}, State ${bytes(pool.state)}, pending ${bytes(pool.pending)}, layout ${bytes(pool.padding)}`"
      >
        <span class="mono muted">R{{ pool.rank }}</span>
        <div class="pool-meter">
          <span
            class="pool-live"
            :style="{ width: `${pool.capacity ? (pool.live / pool.capacity) * 100 : 0}%` }"
          /><span
            class="pool-reserved"
            :style="{ width: `${pool.capacity ? (pool.reserved / pool.capacity) * 100 : 0}%` }"
          /><span
            class="pool-pending"
            :style="{
              width: `${pool.capacity ? ((pool.pending + pool.padding) / pool.capacity) * 100 : 0}%`,
            }"
          />
        </div>
        <span class="mono"
          >{{ bytes(pool.occupied) }}<span class="muted"> / {{ bytes(pool.capacity) }}</span></span
        >
      </div>
      <p class="instrument-note mono">
        ≈ {{ tokens(instance?.info.model.context_token_capacity_estimate) }} token capacity
      </p>
    </section>

    <section class="instrument instrument--ttft">
      <div class="instrument-heading">
        <span class="label">TTFT <span class="scope">{{ rangeLabel }}</span></span
        ><span class="instrument-value">{{ number(ttft?.p50, 0) }}<small>ms</small></span>
      </div>
      <TtftDistribution :ttft="ttft" />
      <div class="instrument-note mono">
        ≈ p50 {{ ms(ttft?.p50) }} · p90 {{ ms(ttft?.p90) }} · p99 {{ ms(ttft?.p99) }}
      </div>
      <div class="instrument-note mono muted">
        {{ number(ttft?.count ?? 0, 0) }} samples
      </div>
    </section>

    <section class="instrument instrument--load">
      <div class="instrument-heading">
        <span class="label">Load <span class="scope">{{ nowScope }}</span></span
        ><span class="instrument-value"
          >{{ number(gauges.running_requests, 0) }}<small>running</small></span
        >
      </div>
      <div class="load-grid">
        <div>
          <span class="label">Prefill</span
          ><strong>{{ number(gauges.prefilling_requests, 0) }}</strong>
        </div>
        <div>
          <span class="label">Waiting</span
          ><strong>{{ number(gauges.waiting_requests, 0) }}</strong>
        </div>
        <div>
          <span class="label">Preparing</span
          ><strong>{{ number(gauges['service.preparing'], 0) }}</strong>
        </div>
        <div>
          <span class="label">Queue p50 <span class="scope">{{ rangeLabel }}</span></span
          ><strong>{{ ms(summary?.queue.p50) }}</strong>
        </div>
      </div>
      <p class="instrument-note mono">
        {{ number(gauges.paused_requests, 0) }} paused ·
        {{ number(gauges.replaying_requests, 0) }} replay · max
        {{ instance?.info.model.max_concurrency ?? '—' }}
      </p>
    </section>

    <section class="instrument instrument--spec">
      <div class="instrument-heading">
        <span class="label">Speculation <span class="scope">{{ rangeLabel }}</span></span
        ><span class="instrument-value">{{ number(summary?.specPercent) }}<small>%</small></span>
      </div>
      <p class="instrument-note mono">
        accepted / drafted · fallback {{ number(summary?.fallbackPercent) }}%
      </p>
      <div class="thin-meter"><span :style="{ width: `${summary?.specPercent ?? 0}%` }" /></div>
    </section>

    <section class="instrument instrument--prefix">
      <div class="instrument-heading">
        <span class="label">Prefix cache <span class="scope">{{ rangeLabel }}</span></span
        ><span class="instrument-value">{{ number(summary?.prefixPercent) }}<small>%</small></span>
      </div>
      <p class="instrument-note mono">
        prompt tokens reused · {{ tokens(summary?.reusedTokens) }} tokens
      </p>
      <div class="thin-meter prefix-meter">
        <span :style="{ width: `${summary?.prefixPercent ?? 0}%` }" />
      </div>
    </section>

    <section class="instrument instrument--ranks">
      <div class="instrument-heading">
        <button class="label" @click="emit('details')">Ranks ↗ <span class="scope">{{ nowScope }}</span></button
        ><span class="instrument-value"
          >{{ instance?.info.ranks.length ?? '—' }}<small>ranks</small></span
        >
      </div>
      <div v-for="rank in instance?.info.ranks ?? []" :key="rank.rank" class="rank-row mono">
        <span class="rank-marker" /> <span>R{{ rank.rank }}</span>
        <span class="muted">{{ sampleAge(gauges[`rank.${rank.rank}.sample_age_ms`]) }}</span>
        <span class="rank-memory"
          >{{ bytes(gauges[`rank.${rank.rank}.allocation.device_bytes`]) }}
          <span class="muted">allocated</span></span
        >
      </div>
      <p v-if="(instance?.info.placements.length ?? 0) > 1" class="instrument-note mono">
        <span v-for="placement in instance?.info.placements" :key="placement.id"
          >ADP{{ placement.id }}
          {{ number(gauges[`placement.${placement.id}.resident`], 0) }} resident &nbsp;
        </span>
      </p>
    </section>

    <section class="instrument instrument--recent">
      <div class="instrument-heading">
        <button class="label" @click="emit('events')">Recent events ↗ <span class="scope">{{ eventScope }}</span></button>
      </div>
      <div
        v-for="event in events.slice(0, 3)"
        :key="event.key"
        class="event-line mono"
        :class="{ alert: event.kind.includes('fail') || event.kind.includes('gap') }"
      >
        <span
          >{{ label(event.kind)
          }}<template v-if="event.requestId !== null"> · #{{ event.requestId }}</template></span
        ><span class="muted">{{ age(Math.max(0, now - event.at) / 1000) }} ago</span>
      </div>
      <p v-if="!events.length" class="instrument-note muted">No observed events</p>
    </section>
  </aside>
</template>
