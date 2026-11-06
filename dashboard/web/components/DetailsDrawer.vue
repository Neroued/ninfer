<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue';
import type { Instance, MonitorEvent, Numbers, RequestRow, Summary } from '../../shared/types.js';
import { bytes, clock, label, ms, number, sampleAge, tokens } from '../format.js';

const props = defineProps<{
  kind: 'deployment' | 'request' | 'events' | null;
  instance: Instance | null;
  summary: Summary | null;
  gauges: Numbers;
  request: RequestRow | null;
  events: MonitorEvent[];
  requestEvents: MonitorEvent[];
  now: number;
  moreEvents: boolean;
}>();
const emit = defineEmits<{ close: []; loadEvents: [] }>();
const dialog = ref<HTMLDialogElement>();
const tab = ref('execution');
const tabs = ['execution', 'resources', 'configuration'];
watch(
  () => props.kind,
  async (value) => {
    await nextTick();
    if (value && !dialog.value?.open) dialog.value?.showModal();
    else if (!value && dialog.value?.open) dialog.value?.close();
  },
);
const eventRows = computed(() => (props.kind === 'request' ? props.requestEvents : props.events));
const counters = computed(() => props.summary?.counters ?? {});
const gpuStages = [
  'draft_stream',
  'target_stream',
  'commit_stream',
  'vision_stream',
  'prefill_stream',
];

function fields(source: Numbers, prefix: string, onlyBytes = false) {
  return Object.entries(source)
    .filter(([key]) => key.startsWith(prefix) && (!onlyBytes || key.endsWith('_bytes')))
    .map(([key, value]) => ({
      key,
      label: label(key.slice(prefix.length)),
      value: key.endsWith('_bytes') ? bytes(value, 2) : number(value, 0),
    }));
}

const rankResources = computed(
  () =>
    props.instance?.info.ranks.map((rank) => {
      const prefix = `rank.${rank.rank}.`;
      const budget =
        props.instance?.info.resources.ranks.find((r) => r.rank === rank.rank)?.budget ?? {};
      const budgetRows = Object.entries(budget).filter(
        (entry): entry is [string, number] => typeof entry[1] === 'number',
      );
      const graph = budget.graph_calibration;
      return {
        rank,
        allocations: fields(props.gauges, `${prefix}allocation.`, true),
        device: fields(props.gauges, `pool.device.${rank.rank}.`, true),
        host: fields(props.gauges, `pool.host.${rank.rank}.`, true),
        runtime: fields(props.gauges, prefix).filter((row) =>
          /\.(vision|experts|workspace)\./.test(row.key),
        ),
        ple: fields(counters.value, `${prefix}ple.`),
        budget: fields(Object.fromEntries(budgetRows), ''),
        graph: graph && typeof graph === 'object' ? fields(graph, '') : [],
      };
    }) ?? [],
);
const mediaRows = computed(() => fields(props.gauges, 'media.'));

function average(prefix: string): number | null {
  const fields = prefix.split('.');
  return (
    props.summary?.gpuIntervals.find(
      (row) => row.rank === Number(fields[1]) && row.name === fields[3],
    )?.meanMs ?? null
  );
}

function closeOnBackdrop(event: MouseEvent) {
  if (event.target === dialog.value && dialog.value) {
    const bounds = dialog.value.getBoundingClientRect();
    if (
      event.clientX < bounds.left ||
      event.clientX > bounds.right ||
      event.clientY < bounds.top ||
      event.clientY > bounds.bottom
    )
      emit('close');
  }
}
</script>

<template>
  <dialog
    ref="dialog"
    class="drawer"
    @cancel.prevent="emit('close')"
    @click="closeOnBackdrop"
    @close="emit('close')"
  >
    <div class="drawer-heading">
      <div>
        <span class="label"
          >NInfer /
          {{ kind === 'request' ? 'request' : kind === 'events' ? 'events' : 'engine' }}</span
        >
        <h2>
          {{
            kind === 'request'
              ? `Request #${request?.id}`
              : kind === 'events'
                ? 'Event log'
                : (instance?.info.model.name ?? 'Engine details')
          }}
        </h2>
      </div>
      <button class="close-button" aria-label="Close details" @click="emit('close')">×</button>
    </div>

    <template v-if="kind === 'deployment' && instance">
      <div class="tabs drawer-tabs">
        <button
          v-for="name in tabs"
          :key="name"
          :class="{ selected: tab === name }"
          @click="tab = name"
        >
          {{ name }}
        </button>
      </div>
      <template v-if="tab === 'execution'">
        <section class="drawer-section">
          <h3 class="label">Decode by actual batch</h3>
          <p class="muted">
            Retired executions in the selected window ·
            {{ number((summary?.coveredMs ?? 0) / 1000, 1) }} s observed
          </p>
          <div class="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Batch</th>
                  <th>Steps</th>
                  <th>Total</th>
                  <th>Result</th>
                  <th>Resolution</th>
                  <th>Retirement</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in summary?.batches" :key="row.label">
                  <td>{{ row.label }}</td>
                  <td>{{ number(row.count, 0) }}</td>
                  <td>{{ ms(row.totalMs) }}</td>
                  <td>{{ ms(row.resultMs) }}</td>
                  <td>{{ ms(row.resolutionMs) }}</td>
                  <td>{{ ms(row.retirementMs) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
        <section class="drawer-section">
          <h3 class="label">Execution phases</h3>
          <table>
            <thead>
              <tr>
                <th>Phase</th>
                <th>Executions</th>
                <th>Mean</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in summary?.executions.filter((r) => r.count)" :key="row.label">
                <td>{{ row.label }}</td>
                <td>{{ number(row.count, 0) }}</td>
                <td>{{ ms(row.totalMs) }}</td>
              </tr>
            </tbody>
          </table>
          <p class="instrument-note mono">
            Prefill execution efficiency · {{ number(summary?.prefillEfficiency, 0) }} tok/s
          </p>
        </section>
        <section class="drawer-section">
          <h3 class="label">GPU intervals by rank</h3>
          <p class="muted">
            Mean per measured interval; overlapping streams retain their own timing.
          </p>
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Interval</th>
                <th>Samples</th>
                <th>Mean</th>
              </tr>
            </thead>
            <tbody>
              <template v-for="rank in instance.info.ranks" :key="rank.rank"
                ><tr v-for="stage in gpuStages" :key="stage">
                  <td>R{{ rank.rank }}</td>
                  <td>{{ label(stage) }}</td>
                  <td>{{ number(counters[`rank.${rank.rank}.gpu.${stage}.count`] ?? 0, 0) }}</td>
                  <td>{{ ms(average(`rank.${rank.rank}.gpu.${stage}`)) }}</td>
                </tr></template
              >
            </tbody>
          </table>
        </section>
        <section class="drawer-section">
          <h3 class="label">Draft positions</h3>
          <table>
            <thead>
              <tr>
                <th>Position</th>
                <th>Attempted</th>
                <th>Accepted</th>
                <th>Acceptance</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="position in summary?.specPositions" :key="position.position">
                <td>{{ position.position }}</td>
                <td>{{ number(position.attempted, 0) }}</td>
                <td>{{ number(position.accepted, 0) }}</td>
                <td>{{ number(position.percent) }}%</td>
              </tr>
            </tbody>
          </table>
        </section>
        <section class="drawer-section">
          <h3 class="label">Attention work / decode</h3>
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Forwards</th>
                <th>Rows</th>
                <th>Valid positions</th>
                <th>Scheduled</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="rank in instance.info.ranks" :key="rank.rank">
                <td>R{{ rank.rank }}</td>
                <td
                  v-for="key in ['forwards', 'rows', 'valid_positions', 'scheduled_positions']"
                  :key="key"
                >
                  {{ number(counters[`rank.${rank.rank}.attention.decode.${key}`] ?? 0, 0) }}
                </td>
              </tr>
            </tbody>
          </table>
        </section>
        <section class="drawer-section">
          <h3 class="label">Communication payload</h3>
          <p class="muted">Payload passed to collectives, grouped by rank and execution path.</p>
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Path</th>
                <th>Collective</th>
                <th>Calls</th>
                <th>Payload</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="row in summary?.communication"
                :key="`${row.rank}:${row.path}:${row.kind}`"
              >
                <td>R{{ row.rank }}</td>
                <td>{{ row.path }}</td>
                <td>{{ label(row.kind) }}</td>
                <td>{{ number(row.calls, 0) }}</td>
                <td>{{ bytes(row.payloadBytes) }}</td>
              </tr>
            </tbody>
          </table>
        </section>
        <section class="drawer-section">
          <h3 class="label">Prefix checkpoints</h3>
          <table>
            <thead>
              <tr>
                <th>Role</th>
                <th>Retained</th>
                <th>Published</th>
                <th>Reclaimed</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="role in ['input_replay', 'long_anchor', 'shared_prefix', 'continuation']"
                :key="role"
              >
                <td>{{ label(role) }}</td>
                <td>{{ number(gauges[`prefix.${role}.retained`], 0) }}</td>
                <td>{{ number(counters[`prefix.${role}.published`] ?? 0, 0) }}</td>
                <td>{{ number(counters[`prefix.${role}.reclaimed`] ?? 0, 0) }}</td>
              </tr>
            </tbody>
          </table>
        </section>
      </template>
      <template v-else-if="tab === 'resources'">
        <section v-for="resource in rankResources" :key="resource.rank.rank" class="drawer-section">
          <h3 class="label">R{{ resource.rank.rank }} · {{ resource.rank.hardware }}</h3>
          <p class="mono muted">
            node {{ resource.rank.node }} / device {{ resource.rank.device }} · TP{{
              resource.rank.tp.rank
            }}/{{ resource.rank.tp.size }} EP{{ resource.rank.ep.rank }}/{{
              resource.rank.ep.size
            }}
            DP{{ resource.rank.dp.rank }}/{{ resource.rank.dp.size }} ADP{{
              resource.rank.adp.rank
            }}/{{ resource.rank.adp.size }}
          </p>
          <dl class="detail-list">
            <template v-for="row in resource.allocations" :key="row.key"
              ><dt>{{ row.label }}</dt>
              <dd>{{ row.value }}</dd></template
            >
          </dl>
          <p class="instrument-note mono">
            Physical sample ·
            {{ sampleAge(gauges[`rank.${resource.rank.rank}.physical_age_ms`]) }} old
          </p>
          <dl class="detail-list">
            <dt>{{ resource.rank.shared_physical_memory ? 'System available' : 'CUDA free' }}</dt>
            <dd>
              {{
                bytes(
                  gauges[
                    `rank.${resource.rank.rank}.physical.${resource.rank.shared_physical_memory ? 'system_available_bytes' : 'cuda_free_bytes'}`
                  ],
                )
              }}
            </dd>
          </dl>
          <details
            v-for="section in [
              { name: 'Device Context pool', rows: resource.device },
              { name: 'Host Context pool', rows: resource.host },
              { name: 'Vision, expert and workspace', rows: resource.runtime },
              { name: 'PLE activity', rows: resource.ple },
              { name: 'Startup budget', rows: resource.budget },
              { name: 'Graph calibration', rows: resource.graph },
            ].filter((section) => section.rows.length)"
            :key="section.name"
          >
            <summary>{{ section.name }}</summary>
            <dl class="detail-list">
              <template v-for="row in section.rows" :key="row.key"
                ><dt>{{ row.label }}</dt>
                <dd>{{ row.value }}</dd></template
              >
            </dl>
          </details>
        </section>
        <section class="drawer-section">
          <h3 class="label">Media</h3>
          <dl class="detail-list">
            <template v-for="row in mediaRows" :key="row.key"
              ><dt>{{ row.label }}</dt>
              <dd>{{ row.value }}</dd></template
            >
          </dl>
        </section>
      </template>
      <template v-else>
        <section class="drawer-section">
          <h3 class="label">Instance</h3>
          <dl class="detail-list">
            <dt>Identity</dt>
            <dd>{{ instance.id }}</dd>
            <dt>Engine</dt>
            <dd>{{ instance.engineUrl }}</dd>
            <dt>Started</dt>
            <dd>{{ clock(instance.info.start_time_seconds * 1000, true) }}</dd>
            <dt>Architecture</dt>
            <dd>{{ instance.info.model.architecture }}</dd>
            <dt>KV format</dt>
            <dd>{{ instance.info.model.kv_cache }}</dd>
            <dt>Context limit</dt>
            <dd>{{ number(instance.info.model.max_context, 0) }}</dd>
            <dt>Token capacity</dt>
            <dd>≈ {{ number(instance.info.model.context_token_capacity_estimate, 0) }}</dd>
          </dl>
        </section>
        <section class="drawer-section">
          <h3 class="label">Effective configuration</h3>
          <dl class="detail-list">
            <template v-for="(value, key) in instance.info.configuration" :key="key"
              ><dt>{{ key }}</dt>
              <dd>{{ value }}</dd></template
            >
          </dl>
        </section>
      </template>
    </template>

    <template v-if="kind === 'request' && request">
      <section class="drawer-section">
        <dl class="detail-list">
          <dt>Phase</dt>
          <dd>{{ label(request.phase) }}</dd>
          <dt>Engine request</dt>
          <dd>{{ request.engineId ?? '—' }}</dd>
          <dt>HTTP request</dt>
          <dd>{{ request.httpId ?? '—' }}</dd>
          <dt>Protocol</dt>
          <dd>{{ request.protocol ?? '—' }}</dd>
          <dt>Received</dt>
          <dd>{{ request.receivedAt === null ? '—' : clock(request.receivedAt, true) }}</dd>
          <dt>First observed</dt>
          <dd>{{ clock(request.firstSeen, true) }}</dd>
          <dt>Ended</dt>
          <dd>{{ request.endedAt === null ? '—' : clock(request.endedAt, true) }}</dd>
          <dt>TTFT</dt>
          <dd>{{ ms(request.ttftMs) }}</dd>
          <dt>Prompt</dt>
          <dd>{{ tokens(request.promptTokens) }}</dd>
          <dt>Prefix reused</dt>
          <dd>{{ tokens(request.reusedTokens) }}</dd>
          <dt>Prompt processed</dt>
          <dd>{{ tokens(request.processedTokens) }}</dd>
          <dt>Generated</dt>
          <dd>{{ tokens(request.generatedTokens) }}</dd>
          <dt>Last observed context</dt>
          <dd>{{ tokens(request.contextTokens) }}</dd>
          <dt>Placement</dt>
          <dd>{{ request.placement ?? '—' }}</dd>
          <dt>Preemptions</dt>
          <dd>{{ request.preemptions }}</dd>
          <dt>Response transport</dt>
          <dd>{{ request.responseFailed ? 'failed' : '—' }}</dd>
        </dl>
      </section>
    </template>
    <section v-if="kind === 'events' || kind === 'request'" class="drawer-section">
      <h3 class="label">Observed events</h3>
      <div v-for="event in eventRows" :key="event.key" class="event-detail">
        <time class="mono muted">{{ clock(event.at, true) }}</time
        ><strong class="mono" :class="{ alert: /fail|gap/.test(event.kind) }"
          >{{ label(event.kind)
          }}<template v-if="event.requestId !== null"> · #{{ event.requestId }}</template></strong
        ><span v-if="event.details.reason" class="muted">{{ event.details.reason }}</span>
      </div>
      <p v-if="!eventRows.length" class="muted">No recorded events</p>
      <button
        v-if="kind === 'events' && moreEvents"
        class="text-button"
        @click="emit('loadEvents')"
      >
        Load more events
      </button>
    </section>
  </dialog>
</template>
