import { computed, onMounted, onUnmounted, ref, shallowRef } from 'vue';
import type {
  ChartPoint,
  HistoryResult,
  Instance,
  InstanceListEntry,
  LiveResult,
  MonitorEvent,
  Page,
  RequestRow,
} from '../shared/types.js';
import { appendChartPoints } from '../shared/timeline.js';
import { initializeTheme } from './theme.js';

export type ViewMode = 'live' | 'paused' | 'history';

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, {
    signal: signal
      ? AbortSignal.any([signal, AbortSignal.timeout(15_000)])
      : AbortSignal.timeout(15_000),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: `HTTP ${response.status}` }));
    throw new Error(error.message ?? error.error ?? `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function useDashboard() {
  const live = shallowRef<LiveResult | null>(null);
  const instance = shallowRef<Instance | null>(null);
  const instances = shallowRef<InstanceListEntry[]>([]);
  const history = shallowRef<HistoryResult | null>(null);
  const historyRequests = shallowRef<Page<RequestRow>>({ items: [], next: null });
  const historyEvents = shallowRef<Page<MonitorEvent>>({ items: [], next: null });
  const mode = ref<ViewMode>('live');
  const rangeMs = ref(300_000);
  const from = ref(Date.now() - rangeMs.value);
  const to = ref(Date.now());
  const error = ref<string | null>(null);
  const loading = ref(false);
  const drawer = ref<'deployment' | 'request' | 'events' | null>(null);
  const selectedRequest = shallowRef<RequestRow | null>(null);
  const requestEvents = shallowRef<MonitorEvent[]>([]);
  const eventLog = shallowRef<Page<MonitorEvent>>({ items: [], next: null });
  let eventRange: { instance: string; from: number; to: number } | null = null;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let controller: AbortController | null = null;
  let selectionController: AbortController | null = null;
  let stopped = false;
  let lastHistory = 0;
  let lastInstances = 0;
  let lastPointAt = 0;
  let livePoints: ChartPoint[] = [];

  const summary = computed(() => history.value?.summary ?? live.value?.summary ?? null);
  const gauges = computed(() =>
    mode.value === 'live' || mode.value === 'paused'
      ? (live.value?.gauges ?? {})
      : Object.fromEntries(
          Object.entries(summary.value?.gauges ?? {}).map(([key, value]) => [key, value[4]]),
        ),
  );
  const points = computed(() => history.value?.points ?? []);
  const requests = computed(() =>
    mode.value === 'history' ? historyRequests.value.items : (live.value?.requests ?? []),
  );
  const completed = computed(() =>
    mode.value === 'history' ? [] : (live.value?.recentRequests ?? []),
  );
  const events = computed(() =>
    mode.value === 'history' ? historyEvents.value.items : (live.value?.events ?? []),
  );

  async function loadHistory() {
    const id = instance.value?.id;
    if (!id) return;
    controller?.abort();
    controller = new AbortController();
    const signal = controller.signal;
    loading.value = true;
    if (mode.value === 'live') {
      to.value = Date.now();
      from.value = to.value - rangeMs.value;
    }
    const params = new URLSearchParams({
      instance: id,
      from: String(from.value),
      to: String(to.value),
      points: '1100',
    });
    try {
      const result = await get<HistoryResult>(`/api/history?${params}`, signal);
      if (signal.aborted) return;
      if (mode.value === 'live' && live.value?.instanceId === id) {
        appendChartPoints(
          result.points,
          livePoints.filter((p) => p.at > result.from),
          result.aggregationMs,
        );
      }
      history.value = result;
      lastHistory = Date.now();
      lastPointAt = Math.max(lastPointAt, result.points.at(-1)?.at ?? 0);
      if (mode.value === 'history') {
        const [rows, eventRows] = await Promise.all([
          get<Page<RequestRow>>(`/api/requests?${params}`, signal),
          get<Page<MonitorEvent>>(`/api/events?${params}`, signal),
        ]);
        if (signal.aborted) return;
        historyRequests.value = rows;
        historyEvents.value = eventRows;
      }
      error.value = null;
    } catch (cause) {
      if (!signal.aborted) error.value = cause instanceof Error ? cause.message : String(cause);
    } finally {
      if (!signal.aborted) loading.value = false;
    }
  }

  async function selectInstance(id: string) {
    controller?.abort();
    selectionController?.abort();
    selectionController = new AbortController();
    const signal = selectionController.signal;
    const historical = id !== live.value?.instanceId;
    if (historical) mode.value = 'history';
    try {
      const selected = await get<Instance>(`/api/instances/${encodeURIComponent(id)}`, signal);
      if (signal.aborted) return;
      instance.value = selected;
      history.value = null;
      historyRequests.value = { items: [], next: null };
      historyEvents.value = { items: [], next: null };
      lastPointAt = 0;
      livePoints = [];
      if (historical) {
        to.value = selected.lastSeen;
        from.value = Math.max(selected.firstSeen, to.value - rangeMs.value);
        if (from.value >= to.value) from.value = to.value - rangeMs.value;
      }
      await loadHistory();
    } catch (cause) {
      if (!signal.aborted) error.value = cause instanceof Error ? cause.message : String(cause);
    }
  }

  function appendPoints(updates: ChartPoint[]) {
    if (!history.value || live.value?.instanceId !== history.value.instanceId) return;
    const newPoints = updates.filter((p) => p.at > lastPointAt);
    livePoints = [...livePoints, ...newPoints].slice(-120);
    const cutoff = Date.now() - rangeMs.value;
    const points = history.value.points.filter((p) => p.at >= cutoff);
    appendChartPoints(points, newPoints, history.value.aggregationMs);
    history.value = {
      ...history.value,
      points,
    };
    lastPointAt = newPoints.at(-1)?.at ?? lastPointAt;
  }

  async function poll() {
    if (stopped) return;
    try {
      if (mode.value !== 'paused') {
        const value = await get<LiveResult>(`/api/live?since=${lastPointAt}`);
        // The user can pause while this request is in flight.
        if (stopped || (mode.value as ViewMode) === 'paused') return;
        live.value = value;
        initializeTheme(value.defaultTheme);
        if (Date.now() - lastInstances > 15_000) {
          instances.value = await get<InstanceListEntry[]>('/api/instances');
          lastInstances = Date.now();
        }
        if (mode.value === 'live' && value.instanceId && instance.value?.id !== value.instanceId) {
          await selectInstance(value.instanceId);
        } else if (!instance.value && instances.value.length) {
          await selectInstance(instances.value[0].id);
        }
        if (mode.value === 'live') {
          to.value = value.now;
          from.value = value.now - rangeMs.value;
          appendPoints(value.points);
          const refreshMs = rangeMs.value > 3_600_000 ? 60_000 : 10_000;
          if (Date.now() - lastHistory > refreshMs && !loading.value) void loadHistory();
        }
        if (drawer.value === 'request' && selectedRequest.value && mode.value === 'live') {
          const row = [...value.requests, ...value.recentRequests].find(
            (r) =>
              r.id === selectedRequest.value?.id &&
              r.instanceId === selectedRequest.value.instanceId,
          );
          if (row) {
            const finished = row.terminal && !selectedRequest.value.terminal;
            selectedRequest.value = row;
            const merged = new Map(requestEvents.value.map((event) => [event.key, event]));
            for (const event of value.events) {
              if (event.instanceId === row.instanceId && event.requestId === row.id)
                merged.set(event.key, event);
            }
            requestEvents.value = [...merged.values()].sort((a, b) => b.at - a.at).slice(0, 200);
            if (finished) void openRequest(row);
          }
        }
      }
    } catch (cause) {
      error.value = cause instanceof Error ? cause.message : String(cause);
    } finally {
      if (!stopped) timer = setTimeout(poll, 1000);
    }
  }

  async function setRange(milliseconds: number) {
    rangeMs.value = milliseconds;
    if (mode.value === 'live') to.value = Date.now();
    from.value = to.value - milliseconds;
    await loadHistory();
  }

  async function explore(start: number, end: number) {
    if (end <= start) return;
    mode.value = 'history';
    from.value = start;
    to.value = end;
    rangeMs.value = end - start;
    await loadHistory();
  }

  async function resume() {
    mode.value = 'live';
    if (live.value?.instanceId && instance.value?.id !== live.value.instanceId) {
      await selectInstance(live.value.instanceId);
    } else await loadHistory();
  }

  function pause() {
    if (mode.value === 'paused') {
      void resume();
      return;
    }
    controller?.abort();
    loading.value = false;
    mode.value = 'paused';
  }

  async function openRequest(row: RequestRow) {
    selectedRequest.value = row;
    requestEvents.value = [];
    drawer.value = 'request';
    try {
      const query = new URLSearchParams({
        instance: row.instanceId,
        from: String(Math.max(0, row.firstSeen - 1000)),
        to: String(Math.max(row.lastSeen + 1000, Date.now())),
        request_id: String(row.id),
        limit: '200',
      });
      const result = await get<Page<MonitorEvent>>(`/api/events?${query}`);
      if (
        selectedRequest.value?.id === row.id &&
        selectedRequest.value.instanceId === row.instanceId
      )
        requestEvents.value = result.items;
    } catch (cause) {
      error.value = cause instanceof Error ? cause.message : String(cause);
    }
  }

  async function moreRequests() {
    if (!historyRequests.value.next || !instance.value) return;
    const query = new URLSearchParams({
      instance: instance.value.id,
      from: String(from.value),
      to: String(to.value),
      cursor: historyRequests.value.next,
    });
    const cursor = historyRequests.value.next;
    try {
      const page = await get<Page<RequestRow>>(`/api/requests?${query}`);
      if (cursor === historyRequests.value.next && query.get('instance') === instance.value?.id) {
        historyRequests.value = {
          items: [...historyRequests.value.items, ...page.items],
          next: page.next,
        };
      }
    } catch (cause) {
      error.value = cause instanceof Error ? cause.message : String(cause);
    }
  }

  async function openEvents() {
    if (!instance.value) return;
    drawer.value = 'events';
    eventLog.value = { items: [], next: null };
    eventRange = { instance: instance.value.id, from: from.value, to: to.value };
    await loadEvents(false);
  }

  async function loadEvents(more: boolean) {
    const range = eventRange;
    if (!range || (more && !eventLog.value.next)) return;
    const cursor = eventLog.value.next;
    const query = new URLSearchParams({
      instance: range.instance,
      from: String(range.from),
      to: String(range.to),
    });
    if (more && cursor) query.set('cursor', cursor);
    try {
      const page = await get<Page<MonitorEvent>>(`/api/events?${query}`);
      if (eventRange === range && (!more || cursor === eventLog.value.next)) {
        eventLog.value = {
          items: more ? [...eventLog.value.items, ...page.items] : page.items,
          next: page.next,
        };
      }
    } catch (cause) {
      error.value = cause instanceof Error ? cause.message : String(cause);
    }
  }

  onMounted(poll);
  onUnmounted(() => {
    stopped = true;
    clearTimeout(timer);
    controller?.abort();
    selectionController?.abort();
  });
  return {
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
    historyEvents,
    selectInstance,
    setRange,
    explore,
    resume,
    pause,
    openRequest,
    moreRequests,
    openEvents,
    moreEvents: () => loadEvents(true),
  };
}
