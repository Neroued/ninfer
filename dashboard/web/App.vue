<script setup lang="ts">
import { computed, ref } from 'vue';
import { useDashboard } from './state.js';
import { applyTheme, themeId, themes } from './theme.js';
import { age, clock, dateInput, ms, number, tokens } from './format.js';
import SplitLayout from './layouts/SplitLayout.vue';
import Timeline from './components/Timeline.vue';
import RequestTable from './components/RequestTable.vue';
import TelemetryRail from './components/TelemetryRail.vue';
import DetailsDrawer from './components/DetailsDrawer.vue';

const {
  live,
  instance,
  instances,
  history,
  summary,
  gauges,
  points,
  requests,
  completed,
  events,
  mode,
  rangeMs,
  from,
  to,
  error,
  loading,
  drawer,
  selectedRequest,
  requestEvents,
  eventLog,
  historyRequests,
  selectInstance,
  setRange,
  explore,
  resume,
  pause,
  openRequest,
  moreRequests,
  moreEvents,
  openEvents,
} = useDashboard();
const ranges = [
  [60_000, '1m'],
  [300_000, '5m'],
  [900_000, '15m'],
  [3_600_000, '1h'],
  [21_600_000, '6h'],
  [86_400_000, '24h'],
  [604_800_000, '7d'],
  [2_592_000_000, '30d'],
] as const;
const dateFrom = ref('');
const dateTo = ref('');
const rangePicker = ref<HTMLDetailsElement>();
const hero = computed(() => (mode.value === 'history' ? summary.value : live.value?.summary));
const now = computed(() => (mode.value === 'history' ? to.value : (live.value?.now ?? Date.now())));
const warning = computed(() => error.value ?? live.value?.storageError ?? live.value?.error);
const status = computed(() =>
  mode.value === 'paused'
    ? 'paused'
    : mode.value === 'history'
      ? 'history'
      : live.value?.connected
        ? live.value.engineStatus && live.value.engineStatus !== 'ready'
          ? `engine ${live.value.engineStatus}`
          : 'live'
        : 'disconnected',
);
const rangeLabel = computed(() => {
  const preset = ranges.find(([value]) => value === rangeMs.value);
  return preset?.[1] ?? 'custom';
});
const ttft = computed(() => summary.value?.ttft ?? null);
const model = computed(() => instance.value?.info.model);
const topology = computed(() => {
  const rank = instance.value?.info.ranks[0];
  if (!rank) return '';
  return (
    (['tp', 'ep', 'dp', 'adp'] as const)
      .filter((key) => rank[key].size > 1)
      .map((key) => `${key.toUpperCase()}${rank[key].size}`)
      .join(' · ') || 'single rank'
  );
});

function themeChanged(event: Event) {
  applyTheme((event.target as HTMLSelectElement).value, true);
}
async function instanceChanged(event: Event) {
  try {
    await selectInstance((event.target as HTMLSelectElement).value);
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause);
  }
}
function rangeOpened() {
  if (!rangePicker.value?.open) return;
  dateFrom.value = dateInput(from.value);
  dateTo.value = dateInput(to.value);
}
async function applyDates() {
  const start = new Date(dateFrom.value).getTime();
  const end = new Date(dateTo.value).getTime();
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    error.value = 'Choose a valid start and end time';
    return;
  }
  rangePicker.value?.removeAttribute('open');
  await explore(start, end);
}
</script>

<template>
  <SplitLayout>
    <template #header>
      <div class="brand">
        <span class="brand-mark" /><span class="wordmark mono">NINFER</span>
        <h1>{{ model?.name ?? 'Monitor' }}</h1>
        <span class="brand-meta mono"
          >{{ model?.kv_cache.toUpperCase() ?? '—' }} · {{ tokens(model?.max_context) }} ·
          {{ topology }}</span
        >
      </div>
      <div class="top-actions">
        <span
          class="connection mono"
          :class="{
            stale: !live?.connected || live.engineStatus === 'failed',
            frozen: mode !== 'live',
          }"
          ><i />{{ status }}</span
        >
        <select
          class="theme-select mono"
          aria-label="Color theme"
          :value="themeId"
          @change="themeChanged"
        >
          <option v-for="theme in themes" :key="theme.id" :value="theme.id">
            {{ theme.name }}
          </option>
        </select>
        <button class="text-button" @click="mode === 'live' ? pause() : resume()">
          {{ mode === 'live' ? 'pause' : 'live ↗' }}
        </button>
        <button class="text-button" @click="drawer = 'deployment'">details</button>
      </div>
    </template>
    <template #notice
      ><div v-if="warning" class="notice mono">{{ warning }}</div></template
    >
    <template #hero>
      <section class="hero">
        <div class="hero-main">
          <div class="label">
            Decode
            <span class="hero-window">{{ mode === 'history' ? 'window mean' : 'last 5s' }}</span>
          </div>
          <div class="hero-number mono">{{ number(hero?.decodeTps) }}<span>tok/s</span></div>
        </div>
        <div class="hero-secondary">
          <div>
            <span class="label">Prefill</span
            ><strong>{{ number(hero?.prefillTps, 0) }}<small>tok/s</small></strong>
          </div>
          <div>
            <span class="label">TTFT p50 <span class="scope">{{ rangeLabel }}</span></span
            ><strong>{{ ms(ttft?.p50) }}</strong>
          </div>
          <div>
            <span class="label">Running</span
            ><strong
              >{{ number(gauges.running_requests, 0)
              }}<small>/{{ model?.max_concurrency ?? '—' }}</small></strong
            >
          </div>
        </div>
      </section>
    </template>
    <template #timeline>
      <div class="timeline-controls">
        <nav class="ranges" aria-label="Time range">
          <button
            v-for="[value, text] in ranges"
            :key="value"
            :class="{ selected: rangeMs === value }"
            @click="setRange(value)"
          >
            {{ text }}
          </button>
        </nav>
        <details ref="rangePicker" class="range-picker" @toggle="rangeOpened">
          <summary class="text-button">range</summary>
          <form @submit.prevent="applyDates">
            <label>From<input v-model="dateFrom" type="datetime-local" required /></label
            ><label>To<input v-model="dateTo" type="datetime-local" required /></label
            ><button class="apply-button" type="submit">Explore range</button>
          </form>
        </details>
        <span v-if="loading" class="mono muted loading-note">loading</span>
      </div>
      <Timeline
        :points="points"
        :from="from"
        :to="to"
        :aggregation-ms="history?.aggregationMs ?? 1000"
        @select="explore"
      />
    </template>
    <template #requests
      ><RequestTable
        :rows="requests"
        :completed="completed"
        :max-context="model?.max_context ?? 0"
        :now="now"
        :historical="mode === 'history'"
        :more="Boolean(historyRequests.next) && mode === 'history'"
        @select="openRequest"
        @load-more="moreRequests"
    /></template>
    <template #rail
      ><TelemetryRail
        :instance="instance"
        :summary="summary"
        :gauges="gauges"
        :events="events"
        :now="now"
        :range-label="rangeLabel"
        :mode="mode"
        @details="drawer = 'deployment'"
        @events="openEvents"
    /></template>
    <template #footer>
      <div class="footer-left">
        <span>{{
          live
            ? `${live.samplingIntervalMs / 1000}s sampling · ${live.fineHistoryHours}h fine · ${live.historyDays}d history`
            : 'Connecting to collector'
        }}</span
        ><span v-if="instance"
          >up {{ age((now - instance.info.start_time_seconds * 1000) / 1000) }}</span
        >
      </div>
      <div class="footer-right">
        <span v-if="mode === 'history'"
          >{{
            history?.resolutionMs === 3600000
              ? '1h'
              : history?.resolutionMs === 60000
                ? '1m'
                : 'fine'
          }}
          resolution · {{ number((summary?.coveredMs ?? 0) / 1000, 0) }}s observed</span
        ><span v-else-if="live?.lastSampleAt" :title="clock(live.lastSampleAt, true)">{{
          live.eventsConnected ? 'events connected' : 'events reconnecting'
        }}</span
        ><select
          v-if="instances.length"
          class="instance-select"
          aria-label="Service instance"
          :value="instance?.id"
          @change="instanceChanged"
        >
          <option v-for="item in instances" :key="item.id" :value="item.id">
            {{ item.model }} · {{ clock(item.firstSeen, true) }}
          </option>
        </select>
      </div>
    </template>
    <template #overlay
      ><DetailsDrawer
        :kind="drawer"
        :instance="instance"
        :summary="summary"
        :gauges="gauges"
        :request="selectedRequest"
        :events="eventLog.items"
        :request-events="requestEvents"
        :now="now"
        :more-events="Boolean(eventLog.next)"
        @close="drawer = null"
        @load-events="moreEvents"
    /></template>
  </SplitLayout>
</template>
