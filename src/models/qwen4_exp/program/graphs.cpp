#include "models/qwen4_exp/program/graphs.h"
#include "ninfer/ops/sequence.h"
#include "models/qwen4_exp/program/internal.h"
#include "models/qwen4_exp/program/device_graphs.h"
#include "core/startup.h"
#include "core/transfer.h"
#include "core/nvtx.h"

#include <algorithm>
#include <thread>

namespace ninfer::models::qwen4_exp::detail {

std::vector<ops::ContextBucket> decode_context_buckets(std::uint32_t capacity) {
    std::vector<ops::ContextBucket> buckets;
    ops::ContextBucketRange::through(capacity).for_each(
        [&](ops::ContextBucket bucket) { buckets.push_back(bucket); });
    return buckets;
}

namespace {
template <class T>
void write_values(const Tensor& target, std::span<const T> values, CudaExecutionView cuda) {
    copy_bytes({target.data, target.bytes(), cuda.device()},
               {values.data(), values.size_bytes(), Device::cpu()}, cuda);
}

} // namespace

void GraphInputs::prepare() {
    auto& r            = owner_.resources;
    const auto& images = r.plan.persistent.state.images;
    std::vector<RecordSpec> specs{{images.image_bytes, images.alignment}};
    for (std::uint32_t history = 0; history < r.pool_count(); ++history) {
        const auto& layout = history == 0 ? r.plan.persistent.main : *r.plan.persistent.mtp;
        specs.push_back({layout.pages.page_bytes, layout.pages.alignment});
    }
    const auto geometry = describe_context_pool(specs);
    std::vector<RecordRequest> demand;
    for (const auto id : geometry.spec_classes)
        demand.push_back({id, owner_.shape.max_concurrency});
    const auto bytes = geometry.required_bytes(demand);
    if (bytes > r.context_backing().bytes)
        throw std::logic_error("Graph records exceed the planned Context working set");
    records_      = std::make_unique<ContextPool>(0, bytes, specs);
    auto reserved = records_->reserve(demand);
    if (!reserved.reservation || !reserved.reservation->ready())
        throw std::logic_error("Graph preparation cannot reserve its complete Context records");
    const auto allocate = [&](std::size_t kind) {
        const auto allocation = reserved.reservation->materialize(records_->record_class(kind));
        allocations_.push_back(allocation);
        const auto binding = records_->resolve(allocation);
        const auto record  = r.context_backing().subspan(binding.offset, binding.bytes);
        CUDA_CHECK(cudaMemsetAsync(record.data, 0, record.bytes, owner_.cuda.stream()));
        return static_cast<std::int64_t>(binding.offset);
    };
    for (std::uint32_t b = 0; b < owner_.shape.max_concurrency; ++b)
        state_offsets_.push_back(allocate(0));
    for (std::uint32_t pool = 0; pool < r.pool_count(); ++pool) {
        auto& tables         = r.pool(pool).tables();
        const auto capacity  = tables.logical_page_capacity();
        const auto row_bytes = std::size_t(capacity) * sizeof(std::int64_t);
        mappings_[pool]      = PinnedHostBuffer(row_bytes * owner_.shape.max_concurrency);
        for (std::uint32_t b = 0; b < owner_.shape.max_concurrency; ++b) {
            const auto page_offset = allocate(pool + 1);
            const auto staging     = mappings_[pool].span().subspan(b * row_bytes, row_bytes);
            auto* offsets          = static_cast<std::int64_t*>(staging.data);
            std::fill_n(offsets, capacity, page_offset);
            tables.publish(b, 0, {offsets, capacity}, staging, owner_.cuda);
        }
    }
    for (std::uint32_t slot = 0; r.ple_upload && slot < 2; ++slot) {
        auto embedding = r.ple_upload->device_slot(slot, r.plan.ple.max_positions, 1);
        CUDA_CHECK(cudaMemsetAsync(embedding.data, 0, embedding.bytes(), owner_.cuda.stream()));
    }
    const auto& buffers = r.buffers();
    if (buffers.mtp) {
        for (const auto* region :
             {&buffers.mtp->base_conditioning, &buffers.mtp->hidden, &buffers.mtp->tail_keys,
              &buffers.mtp->tail_coordinates, &buffers.mtp->drafts}) {
            const auto value = r.tensor(*region);
            CUDA_CHECK(cudaMemsetAsync(value.data, 0, value.bytes(), owner_.cuda.stream()));
        }
    }
    std::vector<ops::SamplingConfig> configs(owner_.shape.max_concurrency);
    for (auto& config : configs) config.temperature = 0;
    const auto target = r.tensor(*buffers.sampling);
    copy_bytes({target.data, target.bytes(), owner_.cuda.device()},
               {configs.data(), configs.size() * sizeof(configs[0]), Device::cpu()}, owner_.cuda);
}

GraphInputs::~GraphInputs() {
    // prepare() reports execution failures; unwinding still drains the staging readers.
    (void)cudaStreamSynchronize(owner_.cuda.stream());
    for (const auto id : allocations_) records_->release(id);
}

void GraphInputs::stage() {
    auto& r             = owner_.resources;
    const auto& ingress = r.buffers().ingress;
    copy_bytes(r.ingress().subspan(0, ingress.bytes), ingress.bind(r.persistent()), owner_.cuda);
}

void GraphInputs::stage_proposal(std::uint32_t rows, std::uint32_t depth, std::uint32_t visible) {
    auto& r   = owner_.resources;
    auto call = r.main_batch(owner_.shape.draft_tokens + 1, rows, visible);
    const auto extent = std::min(depth, visible - 1);
    for (std::size_t s = 0; s < r.buffers().mtp->draft_steps.size(); ++s) {
        const auto bound = s < extent ? visible : 0U;
        batch(r.mtp_batch(r.buffers().mtp->draft_steps[s], 1, rows, bound), bound);
    }
    batch(call, visible);
    const auto live = r.plan.batches->minimum_rows(rows);
    std::vector<std::int32_t> extents(rows), columns(rows);
    for (std::uint32_t b = 0; b < live; ++b) {
        extents[b] = std::min(depth, visible - 1);
        columns[b] = extents[b] + 1;
    }
    write_values<std::int32_t>(r.tensor(r.buffers().mtp->extents).slice(0, 0, rows), extents,
                               owner_.cuda);
    write_values<std::int32_t>(call.valid_columns, columns, owner_.cuda);
    stage();
}

void GraphInputs::batch(execution::BatchView call, std::uint32_t visible) {
    const auto t = call.tokens, b = call.batch;
    const auto live  = visible ? owner_.resources.plan.batches->minimum_rows(b) : 0U;
    const auto valid = std::min<std::uint32_t>(t, visible);
    const auto base  = visible - valid;
    std::vector<std::int32_t> ids(t * b, owner_.parameters.profile->config.text.ple.eos);
    std::vector<std::int32_t> positions(t * b), rope(t * b * 3), rows(b, -1), bases(b);
    std::vector<std::int64_t> state(b, -1);
    std::vector<std::int32_t> live_rows(b), counts(b), offsets(b, -1);
    for (std::uint32_t row = 0; row < live; ++row) {
        rows[row]      = row;
        bases[row]     = base;
        state[row]     = state_offsets_[row];
        live_rows[row] = 1;
        counts[row]    = valid;
        offsets[row]   = 0;
        for (int j = 0; j < t; ++j) {
            const auto position    = base + std::min<std::uint32_t>(j, valid - 1);
            positions[j + t * row] = position;
            for (int axis = 0; axis < 3; ++axis) rope[j + t * (axis + 3 * row)] = position;
        }
    }
    write_values<std::int32_t>(call.ids, ids, owner_.cuda);
    write_values<std::int32_t>(call.positions, positions, owner_.cuda);
    write_values<std::int32_t>(call.rope_positions, rope, owner_.cuda);
    write_values<std::int64_t>(call.source_offsets, state, owner_.cuda);
    write_values<std::int64_t>(call.destination_offsets, state, owner_.cuda);
    write_values<std::int32_t>(call.table_rows, rows, owner_.cuda);
    write_values<std::int32_t>(call.base_positions, bases, owner_.cuda);
    write_values<std::int32_t>(call.live_rows, live_rows, owner_.cuda);
    write_values<std::int32_t>(call.valid_columns, counts, owner_.cuda);
    write_values<std::int32_t>(call.record_offsets, offsets, owner_.cuda);
    const auto& buffers = owner_.resources.buffers();
    std::vector<ops::SamplingConfig> configs(b);
    for (auto& config : configs) config.temperature = 0;
    const auto target = owner_.resources.tensor(*buffers.sampling);
    copy_bytes({target.data, target.bytes(), owner_.cuda.device()},
               {configs.data(), configs.size() * sizeof(configs[0]), Device::cpu()}, owner_.cuda);
    const auto& io = buffers.batch;
    std::vector<std::int32_t> slots(b, -1), sampling_positions(b), extents(b);
    for (std::uint32_t row = 0; row < live; ++row) {
        slots[row]              = row;
        sampling_positions[row] = base;
        extents[row]            = std::min(owner_.shape.draft_tokens, visible - 1);
    }
    write_values<std::int32_t>(owner_.resources.tensor(io.count_slots).slice(0, 0, b), slots,
                               owner_.cuda);
    write_values<std::int32_t>(owner_.resources.tensor(io.sampling_positions).slice(0, 0, b),
                               sampling_positions, owner_.cuda);
    write_values<std::int32_t>(
        owner_.resources.tensor(buffers.anchors).slice(0, 0, b),
        std::vector<std::int32_t>(b, owner_.parameters.profile->config.text.ple.eos), owner_.cuda);
    if (buffers.mtp)
        write_values<std::int32_t>(owner_.resources.tensor(buffers.mtp->extents).slice(0, 0, b),
                                   extents, owner_.cuda);
    write_values<std::int32_t>(
        owner_.resources.packed(io.raw_tail, {2, b}),
        std::vector<std::int32_t>(2 * b, owner_.parameters.profile->config.text.ple.eos),
        owner_.cuda);
    std::vector<std::int64_t> tail_offsets(b, -1);
    for (std::uint32_t row = 0; row < live; ++row) tail_offsets[row] = row * 2 * sizeof(TokenId);
    write_values<std::int64_t>(owner_.resources.tensor(io.raw_tail_offsets).slice(0, 0, b),
                               tail_offsets, owner_.cuda);
}

Graphs::Graphs(RankProgramImpl& owner)
    : owner_(owner), buckets_(decode_context_buckets(owner.shape.max_context)) {}

void Graphs::prepare_main(Batch& graph, std::uint32_t batch, bool verify,
                          GraphPreparation& preparation) {
    auto& p             = owner_;
    auto& r             = p.resources;
    const auto& buffers = r.buffers();
    const auto tokens   = verify ? p.shape.draft_tokens + 1 : 1;
    auto call           = r.main_batch(tokens, batch, p.shape.max_context);
    call.active_tokens  = std::min(tokens, p.shape.max_context) * r.plan.batches->minimum_rows(batch);
    const auto& c       = p.parameters.profile->config.text;
    const auto t = static_cast<std::int32_t>(tokens), b = static_cast<std::int32_t>(batch);
    execution::TextOutputs output{r.packed(buffers.pending_sample, {dimension(c.hidden), t, b}),
                                  {}};
    if (buffers.pending_conditioning)
        output.conditioning =
            r.packed(*buffers.pending_conditioning, {dimension(c.residual_width()), t, b});
    const auto logits =
        r.packed(*buffers.logits, {dimension(p.parameters.profile->local.vocabulary), t, b});
    const auto records = r.main_records(t, b);
    const execution::TextExecution execution{
        r.workspace,  *r.states,           *r.main,     r.experts.get(),
        records.keys, records.coordinates, records.ple, p.parallel ? &*p.parallel : nullptr};
    execution::StateUpdate update = execution::StepUpdate{};
    if (verify) update.emplace<execution::VerifyRecords>(r.replay(t).records);
    const std::array ple{r.ple_upload->binding(0, tokens, batch),
                         r.ple_upload->binding(1, tokens, batch)};
    auto& selected = verify ? graph.verify : graph.main;
    {
        execution::TextRun run(p.parameters, execution, call, update, p.cuda);
        if (!preparation.capturing()) {
            run.stem();
            for (std::uint32_t layer = 0; layer < c.ple.layer; ++layer)
                while (!run.advance_layer(layer)) std::this_thread::yield();
            run.consume_ple(ple[0].values);
            for (std::uint32_t layer = c.ple.layer; layer < p.parameters.text.layers.size();
                 ++layer)
                while (!run.advance_layer(layer)) std::this_thread::yield();
            run.finish(output, logits);
            preparation.checkpoint();
            return;
        }
        selected.prepare(run, p.parameters, r.experts.get(), buckets_, ple, output, logits,
                         moe::decode_geometry(*r.plan.batches, batch, tokens), r.workspace,
                         p.device, [&] {
                             if (!verify) p.install_decode(batch);
                         });
    }
    selected.select(buckets_.back());
    execution::TextRun run(p.parameters, execution, call, update, p.cuda);
    if (selected.resident()) {
        selected.replay(1, p.cuda);
        selected.replay(0, p.cuda);
    } else {
        selected.begin_prefix(p.cuda);
        while (!selected.advance_prefix(p.cuda)) std::this_thread::yield();
        selected.consume_ple(0, p.cuda);
        while (!selected.advance_suffix(p.cuda)) std::this_thread::yield();
    }
    preparation.checkpoint();
}

void Graphs::prepare_drafts(Batch& graph, std::uint32_t batch, GraphInputs& inputs,
                            GraphPreparation& preparation) {
    auto& p = owner_;
    graph.drafts.resize(p.shape.draft_tokens);
    for (std::uint32_t step = 0; step < p.shape.draft_tokens; ++step) {
        auto& call = graph.drafts[step];
        if (!preparation.capturing()) {
            for (const auto bucket : buckets_) {
                const auto bound = std::min(p.shape.max_context, ops::context_bucket_tokens(bucket));
                inputs.batch(p.resources.mtp_batch(p.resources.buffers().mtp->draft_steps[step], 1,
                                                   batch, bound),
                             bound);
                p.draft_forward(step, batch, bound);
            }
            preparation.checkpoint();
            continue;
        }
        p.resources.workspace.reset_peak();
        {
            auto run = p.draft_run(step, batch, p.shape.max_context);
            call.readout.prepare(run.local_qsa(), buckets_, p.cuda);
        }
        call.workspace_bytes = p.resources.workspace.peak_used();
        const auto ids = p.draft_output_ids(step, batch);
        for (std::uint32_t slot = 0; slot < 2; ++slot)
            call.handoffs[slot].capture(
                p.cuda.stream(), [&] { p.host_ple->capture_append(slot, ids, step, p.cuda); });
    }
    if (!preparation.capturing()) return;
    CudaCompletionEvent fork(p.device), join(p.device);
    const auto transfer = p.resources.ple_transfer();
    graph.chains.resize(p.shape.draft_tokens);
    for (const auto& route : p.resources.plan.decode_routes->at({batch, batch})) {
        const auto depth = route.mtp_steps;
        if (!depth) continue;
        inputs.stage_proposal(batch, depth, p.shape.max_context);
        auto& chain = graph.chains[depth - 1];
        chain.installed.resize(depth);
        chain.graph.prepare(p.cuda, [&] {
            p.install_decode(batch);
            for (std::uint32_t step = 0; step < depth; ++step) {
                auto& call = graph.drafts[step];
                p.resources.workspace.reset_peak();
                p.draft_forward(step, batch, p.shape.max_context, [&](const execution::QsaReadout& r) {
                    chain.steps.push_back(call.readout.append(r, p.cuda.stream()));
                });
                call.workspace_bytes =
                    std::max(call.workspace_bytes, p.resources.workspace.peak_used());
                fork.record(p.cuda.stream());
                fork.wait(transfer.stream());
                chain.handoffs.push_back(call.handoffs[0].append_to(transfer.stream()));
            }
            join.record(transfer.stream());
            join.wait(p.cuda.stream());
            const auto& io = p.resources.buffers();
            auto ids       = p.resources.main_batch(route.width, batch, 1).ids;
            ops::prepare_verify_ids(
                p.device_tensor(io.anchors).slice(0, 0, batch), p.draft_proposals(batch),
                p.device_tensor(io.mtp->extents).slice(0, 0, batch), ids, p.cuda);
        });
        for (std::size_t context = 0; context < buckets_.size(); ++context) {
            for (std::uint32_t step = 0; step < depth; ++step)
                chain.graph.executable.update_child(chain.steps[step],
                                                    graph.drafts[step].readout.at(context));
            chain.graph.executable.upload(p.cuda.stream());
            p.device.synchronize();
        }
        std::fill(chain.installed.begin(), chain.installed.end(), buckets_.size() - 1);
        for (std::uint32_t slot : {1U, 0U}) {
            for (std::uint32_t step = 0; step < depth; ++step)
                chain.graph.executable.update_child(chain.handoffs[step],
                                                    graph.drafts[step].handoffs[slot]);
            chain.graph.executable.launch(p.cuda.stream());
            preparation.checkpoint();
        }
    }
}

void Graphs::prepare(GraphInputs& inputs, GraphPreparation& observation) {
    auto& p = owner_;
    auto& r = p.resources;
    std::optional<moe::Schedule::Preparation> preparation;
    if (r.experts) preparation.emplace(*r.experts);
    for (auto batch : r.plan.batches->graph_rows()) {
        r.select_batch(batch);
        auto& graph = batches_[batch];
        inputs.batch(r.main_batch(1, batch, p.shape.max_context), p.shape.max_context);
        inputs.stage();
        prepare_main(graph, batch, false, observation);
        r.workspace.reset_peak();
        if (observation.capturing()) {
            graph.sample.prepare(p.cuda, [&] { p.sample_forward(batch, false); });
            graph.sample.launch(p.cuda);
        } else {
            p.sample_forward(batch, false);
        }
        observation.checkpoint();
        graph.sample_workspace = r.workspace.peak_used();
        if (p.parameters.mtp) {
            const auto& frame = *r.buffers().mtp;
            for (std::uint32_t step = 0; step < p.shape.draft_tokens; ++step)
                inputs.batch(r.mtp_batch(frame.draft_steps[step], 1, batch, p.shape.max_context),
                             p.shape.max_context);
            inputs.stage();
            prepare_drafts(graph, batch, inputs, observation);
            inputs.batch(r.main_batch(p.shape.draft_tokens + 1, batch, p.shape.max_context),
                         p.shape.max_context);
            prepare_main(graph, batch, true, observation);
            r.workspace.reset_peak();
            if (observation.capturing()) {
                graph.accept.prepare(p.cuda,
                                     [&] { p.accept_forward(batch, p.shape.draft_tokens + 1); });
                graph.accept.launch(p.cuda);
            } else {
                p.accept_forward(batch, p.shape.draft_tokens + 1);
            }
            observation.checkpoint();
            graph.accept_workspace = r.workspace.peak_used();
        }
    }
    if (r.experts) r.experts->restore_cache_after_preparation();
    p.device.synchronize();
    r.workspace.reset_peak();
}

TextGraph& Graphs::main(std::uint32_t batch, bool verify, std::uint32_t visible, bool head) {
    auto& group = batches_.at(batch);
    auto& graph = verify ? group.verify : group.main;
    graph.set_head(head);
    graph.select(ops::select_context_bucket(std::max(1U, visible)));
    workspace_peak_ = std::max(workspace_peak_, graph.workspace_bytes());
    return graph;
}

void Graphs::draft(std::uint32_t batch, std::span<const execution::BatchView> calls,
                   ple::Ticket ticket, std::span<const std::uint32_t> extents) {
    auto& group = batches_.at(batch);
    auto& chain = group.chains.at(calls.size() - 1);
    for (std::size_t step = 0; step < calls.size(); ++step) {
        const auto& call = group.drafts[step];
        const auto bucket =
            ops::select_context_bucket(std::max(1U, calls[step].max_visible_tokens));
        const auto found = std::lower_bound(buckets_.begin(), buckets_.end(), bucket);
        if (found == buckets_.end()) throw std::logic_error("MTP graph context exceeds capacity");
        const auto selected = static_cast<std::size_t>(found - buckets_.begin());
        workspace_peak_     = std::max(workspace_peak_, call.workspace_bytes);
        if (chain.installed[step] != selected) {
            chain.graph.executable.update_child(chain.steps[step], call.readout.at(selected));
            chain.installed[step] = selected;
        }
        if (chain.slot != ticket.slot)
            chain.graph.executable.update_child(chain.handoffs[step],
                                                call.handoffs.at(ticket.slot));
    }
    chain.slot = ticket.slot;
    chain.graph.executable.launch(owner_.cuda.stream());
    for (std::uint32_t step = 0; step < calls.size(); ++step)
        owner_.host_ple->append_submitted(ticket, extents, step);
}

void Graphs::sample(std::uint32_t batch) {
    workspace_peak_ = std::max(workspace_peak_, batches_.at(batch).sample_workspace);
    batches_.at(batch).sample.executable.launch(owner_.cuda.stream());
}

void Graphs::accept(std::uint32_t batch) {
    workspace_peak_ = std::max(workspace_peak_, batches_.at(batch).accept_workspace);
    batches_.at(batch).accept.executable.launch(owner_.cuda.stream());
}

void MtpCommitGraphs::prepare(GraphInputs& inputs, GraphPreparation& preparation) {
    auto& p          = owner_;
    auto& r          = p.resources;
    const auto& io   = r.buffers();
    for (auto batch : r.plan.batches->graph_rows()) {
        for (const auto width : {1U, p.shape.draft_tokens + 1}) {
            r.select_batch(batch);
            inputs.batch(r.main_batch(width, batch, p.shape.max_context), p.shape.max_context);
            inputs.batch(r.mtp_batch(io.mtp->draft_steps[0], 1, batch, p.shape.max_context),
                         p.shape.max_context);
            const auto live  = r.plan.batches->minimum_rows(batch);
            const auto valid = std::min(width, p.shape.max_context);
            std::vector<std::int32_t> bases(batch), previous_rope(3 * batch), accepted(batch),
                skip(batch), last(batch, -1), active(batch, -1);
            for (std::uint32_t b = 0; b < live; ++b) {
                accepted[b]         = valid;
                const auto frontier = p.shape.max_context - valid;
                skip[b]             = !frontier || width > 1;
                bases[b]            = frontier ? frontier - 1 + (width > 1) : 0;
                std::fill_n(previous_rope.begin() + 3 * b, 3, frontier ? frontier - 1 : 0);
                last[b]   = valid - 1;
                active[b] = 0;
            }
            write_values<std::int32_t>(r.tensor(io.batch.accepted_columns).slice(0, 0, batch),
                                       accepted, p.cuda);
            write_values<std::int32_t>(r.tensor(io.mtp->skip_first).slice(0, 0, batch), skip,
                                       p.cuda);
            write_values<std::int32_t>(r.tensor(io.mtp->last_columns).slice(0, 0, batch), last,
                                       p.cuda);
            write_values<std::int32_t>(r.tensor(io.mtp->active_columns).slice(0, 0, batch), active,
                                       p.cuda);
            write_values<std::int32_t>(r.tensor(io.batch.count_columns).slice(0, 0, batch),
                                       std::vector<std::int32_t>(batch, 0), p.cuda);
            write_values<std::int32_t>(r.tensor(io.mtp->history_base_positions).slice(0, 0, batch),
                                       bases, p.cuda);
            write_values<std::int32_t>(r.tensor(io.mtp->previous_rope), previous_rope, p.cuda);
            const auto destination = r.tensor(io.commit_destination_offsets).slice(0, 0, batch);
            const auto source      = r.tensor(io.batch.destination_offsets).slice(0, 0, batch);
            copy_bytes({destination.data, destination.bytes(), p.cuda.device()},
                       {source.data, source.bytes(), p.cuda.device()}, p.cuda);
            r.workspace.reset_peak();
            if (!preparation.capturing()) {
                p.execute_mtp_commit(batch, width, true, true);
                preparation.checkpoint();
                continue;
            }
            auto& target = batches_[{batch, width}];
            target.graph.prepare(p.cuda, [&] {
                p.execute_mtp_commit(batch, width, true, true, &target.repair, &target.counts);
            });
            target.workspace_bytes = r.workspace.peak_used();
            target.graph.launch(p.cuda);
            preparation.checkpoint();
        }
    }
    r.workspace.reset_peak();
}

void MtpCommitGraphs::replay(std::uint32_t batch, std::uint32_t width, bool rebuild_history,
                             bool count_tokens) {
    auto& target = batches_.at({batch, width});
    target.repair.set_enabled(target.graph.executable, rebuild_history);
    target.counts.set_enabled(target.graph.executable, count_tokens);
    workspace_peak_ = std::max(workspace_peak_, target.workspace_bytes);
    target.graph.launch(owner_.cuda);
}

GraphPreparationSample RankProgramImpl::initialize_graphs(GraphPreparationMode mode) {
    if (!shape.cuda_graph) return {};
    if (graphs || device_graphs) throw std::logic_error("Qwen4Exp Graphs are already prepared");
    if (mode == GraphPreparationMode::Complete) {
        initialize_graphs(GraphPreparationMode::Warmup);
        return initialize_graphs(GraphPreparationMode::Capture);
    }
    nvtx::ScopedRange range(nvtx::Name::CudaGraphPrepare, nvtx::Category::Graph);
    GraphPreparation observation(device, mode);
    graph_inputs = std::make_unique<GraphInputs>(*this);
    graph_inputs->prepare();
    if (host_ple) {
        graphs = std::make_unique<Graphs>(*this);
        graphs->prepare(*graph_inputs, observation);
    } else {
        device_graphs = std::make_unique<DeviceGraphs>(*this);
        device_graphs->prepare(*graph_inputs, observation);
    }
    if (parameters.mtp && shape.draft_tokens) {
        commit_graphs = std::make_unique<MtpCommitGraphs>(*this);
        commit_graphs->prepare(*graph_inputs, observation);
    }
    observation.checkpoint();
    graph_inputs.reset();
    const auto result = observation.sample();
    if (!graph_memory) graph_memory.emplace();
    if (observation.capturing()) {
        graph_memory->preparation_increment_bytes = result.increment_bytes;
        graph_memory->preparation_peak_bytes      = result.peak_bytes;
    } else {
        release_graphs();
        graph_memory->warmup_increment_bytes = result.increment_bytes;
    }
    return result;
}

void RankProgramImpl::release_graphs() {
    resources.synchronize();
    graphs.reset();
    device_graphs.reset();
    commit_graphs.reset();
    graph_inputs.reset();
}

} // namespace ninfer::models::qwen4_exp::detail
