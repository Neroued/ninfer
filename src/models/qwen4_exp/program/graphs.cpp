#include "models/qwen4_exp/program/graphs.h"
#include "models/qwen4_exp/program/internal.h"
#include "models/qwen4_exp/program/device_graphs.h"
#include "core/startup.h"
#include "core/transfer.h"
#include "core/nvtx.h"

#include <algorithm>
#include <thread>

namespace ninfer::models::qwen4_exp::detail {

std::vector<std::uint32_t> decode_context_bounds(std::uint32_t capacity) {
    std::vector<std::uint32_t> out;
    for (const auto bound : {128U, 512U, 2048U, 8192U, 32768U, 131072U}) {
        if (bound >= capacity) break;
        out.push_back(bound);
    }
    out.push_back(capacity);
    return out;
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
    const auto& buffers = r.plan.persistent.buffers;
    if (buffers.mtp) {
        for (const auto* region : {&buffers.mtp->base_conditioning, &buffers.mtp->hidden,
                                   &buffers.mtp->tail_keys, &buffers.mtp->tail_coordinates,
                                   &buffers.mtp->drafts, &buffers.mtp->extents, &buffers.anchors}) {
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
    const auto& ingress = r.plan.persistent.buffers.ingress;
    copy_bytes(r.ingress().subspan(0, ingress.bytes), ingress.bind(r.persistent()), owner_.cuda);
}

void GraphInputs::batch(execution::BatchView call, std::uint32_t visible) {
    const auto t = call.tokens, b = call.batch;
    const auto valid = std::min<std::uint32_t>(t, visible);
    const auto base  = visible - valid;
    std::vector<std::int32_t> ids(t * b, owner_.parameters.profile->config.text.ple.eos);
    std::vector<std::int32_t> positions(t * b), rope(t * b * 3), rows(b), bases(b, base);
    for (int row = 0; row < b; ++row) {
        rows[row] = row;
        for (int j = 0; j < t; ++j) {
            const auto position    = base + std::min<std::uint32_t>(j, valid - 1);
            positions[j + t * row] = position;
            for (int axis = 0; axis < 3; ++axis) rope[j + t * (axis + 3 * row)] = position;
        }
    }
    write_values<std::int32_t>(call.ids, ids, owner_.cuda);
    write_values<std::int32_t>(call.positions, positions, owner_.cuda);
    write_values<std::int32_t>(call.rope_positions, rope, owner_.cuda);
    write_values<std::int64_t>(call.source_offsets, std::span(state_offsets_).first(b),
                               owner_.cuda);
    write_values<std::int64_t>(call.destination_offsets, std::span(state_offsets_).first(b),
                               owner_.cuda);
    write_values<std::int32_t>(call.table_rows, rows, owner_.cuda);
    write_values<std::int32_t>(call.base_positions, bases, owner_.cuda);
    std::vector<std::int32_t> counts(b, valid), offsets(b, 0);
    write_values<std::int32_t>(call.valid_columns, counts, owner_.cuda);
    write_values<std::int32_t>(call.record_offsets, offsets, owner_.cuda);
    const auto& io = owner_.resources.plan.persistent.buffers.batch;
    write_values<std::int32_t>(
        owner_.resources.packed(io.raw_tail, {2, b}),
        std::vector<std::int32_t>(2 * b, owner_.parameters.profile->config.text.ple.eos),
        owner_.cuda);
    std::vector<std::int64_t> tail_offsets(b);
    for (int row = 0; row < b; ++row) tail_offsets[row] = row * 2 * sizeof(TokenId);
    write_values<std::int64_t>(owner_.resources.tensor(io.raw_tail_offsets).slice(0, 0, b),
                               tail_offsets, owner_.cuda);
}

Graphs::Graphs(RankProgramImpl& owner)
    : owner_(owner), bounds_(decode_context_bounds(owner.shape.max_context)) {}

void Graphs::prepare_main(Batch& graph, std::uint32_t batch, bool verify) {
    auto& p             = owner_;
    auto& r             = p.resources;
    const auto& buffers = r.plan.persistent.buffers;
    const auto tokens   = verify ? p.shape.draft_tokens + 1 : 1;
    const auto call     = r.main_batch(tokens, batch, bounds_.back());
    const auto& c       = p.parameters.profile->config.text;
    const auto t = static_cast<std::int32_t>(tokens), b = static_cast<std::int32_t>(batch);
    execution::TextOutputs output{r.packed(buffers.pending_sample, {dimension(c.hidden), t, b}),
                                  {}};
    if (buffers.pending_conditioning)
        output.conditioning =
            r.packed(*buffers.pending_conditioning, {dimension(c.residual_width()), t, b});
    const auto logits =
        r.packed(*buffers.logits, {dimension(p.parameters.profile->local.vocabulary), t, b});
    const execution::TextExecution execution{r.workspace,
                                             *r.states,
                                             *r.main,
                                             r.experts.get(),
                                             r.tensor(buffers.main_qsa.keys),
                                             r.tensor(buffers.main_qsa.coordinates),
                                             r.tensor(buffers.ple_records),
                                             p.parallel ? &*p.parallel : nullptr};
    execution::StateUpdate update = execution::StepUpdate{};
    if (verify) update.emplace<execution::VerifyRecords>(*r.gdn_records);
    const std::array ple{r.ple_upload->binding(0, tokens, batch),
                         r.ple_upload->binding(1, tokens, batch)};
    auto& selected = verify ? graph.verify : graph.main;
    {
        // Warm the actual bodies before capture. No disk request is needed for these bindings.
        execution::TextRun run(p.parameters, execution, call, update, p.cuda);
        run.stem();
        for (std::uint32_t layer = 0; layer < c.ple.layer; ++layer)
            while (!run.advance_layer(layer)) std::this_thread::yield();
        run.consume_ple(ple[0].values);
        for (std::uint32_t layer = c.ple.layer; layer < p.parameters.text.layers.size(); ++layer)
            while (!run.advance_layer(layer)) std::this_thread::yield();
        run.finish(output, logits);
        p.device.synchronize();
        selected.prepare(run, p.parameters, r.experts.get(), bounds_, ple, output, logits,
                         tokens * batch, r.workspace, p.device, [&] {
                             if (!verify) p.install_decode(batch);
                         });
    }
    selected.select(bounds_.back());
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
    p.device.synchronize();
}

void Graphs::prepare_drafts(Batch& graph, std::uint32_t batch) {
    auto& p = owner_;
    graph.drafts.resize(p.shape.draft_tokens);
    for (std::uint32_t step = 0; step < p.shape.draft_tokens; ++step) {
        auto& call = graph.drafts[step];
        p.draft_forward(step, batch, bounds_.back());
        p.device.synchronize();
        for (const auto bound : bounds_) {
            p.resources.workspace.reset_peak();
            call.definitions.emplace_back();
            call.definitions.back().capture(p.cuda.stream(),
                                            [&] { p.draft_forward(step, batch, bound); });
            call.workspace_bytes.push_back(p.resources.workspace.peak_used());
        }
        const auto ids = p.draft_output_ids(step, batch);
        for (std::uint32_t slot = 0; slot < 2; ++slot)
            call.handoffs[slot].capture(
                p.cuda.stream(), [&] { p.host_ple->capture_append(slot, ids, step, p.cuda); });
    }
    CudaCompletionEvent fork(p.device), join(p.device);
    const auto transfer = p.resources.ple_transfer();
    {
        const auto depth = p.shape.draft_tokens;
        auto& chain      = graph.chain;
        chain.installed.resize(depth);
        chain.graph.prepare(p.cuda, [&] {
            p.install_decode(batch);
            for (std::uint32_t step = 0; step < depth; ++step) {
                const auto& call = graph.drafts[step];
                chain.steps.push_back(call.definitions.front().append_to(p.cuda.stream()));
                fork.record(p.cuda.stream());
                fork.wait(transfer.stream());
                chain.handoffs.push_back(call.handoffs[0].append_to(transfer.stream()));
            }
            join.record(transfer.stream());
            join.wait(p.cuda.stream());
        });
        for (std::size_t context = 0; context < bounds_.size(); ++context) {
            for (std::uint32_t step = 0; step < depth; ++step)
                chain.graph.executable.update_child(chain.steps[step],
                                                    graph.drafts[step].definitions[context]);
            chain.graph.executable.upload(p.cuda.stream());
            p.device.synchronize();
        }
        std::fill(chain.installed.begin(), chain.installed.end(), bounds_.size() - 1);
        for (std::uint32_t slot : {1U, 0U}) {
            for (std::uint32_t step = 0; step < depth; ++step)
                chain.graph.executable.update_child(chain.handoffs[step],
                                                    graph.drafts[step].handoffs[slot]);
            chain.graph.executable.launch(p.cuda.stream());
            p.device.synchronize();
        }
    }
}

void Graphs::prepare(GraphInputs& inputs) {
    auto& p = owner_;
    auto& r = p.resources;
    std::optional<moe::Schedule::Preparation> preparation;
    if (r.experts) preparation.emplace(*r.experts);
    batches_.resize(p.shape.max_concurrency);
    for (std::uint32_t batch = 1; batch <= p.shape.max_concurrency; ++batch) {
        auto& graph = batches_[batch - 1];
        inputs.batch(r.main_batch(1, batch, bounds_.back()), bounds_.back());
        inputs.stage();
        prepare_main(graph, batch, false);
        p.sample_forward(batch, false);
        p.device.synchronize();
        r.workspace.reset_peak();
        graph.sample.prepare(p.cuda, [&] { p.sample_forward(batch, false); });
        graph.sample_workspace = r.workspace.peak_used();
        if (p.parameters.mtp) {
            const auto& frame = *r.plan.persistent.buffers.mtp;
            for (std::uint32_t step = 0; step < p.shape.draft_tokens; ++step)
                inputs.batch(r.mtp_batch(frame.draft_steps[step], 1, batch, bounds_.back()),
                             bounds_.back());
            inputs.stage();
            prepare_drafts(graph, batch);
            inputs.batch(r.main_batch(p.shape.draft_tokens + 1, batch, bounds_.back()),
                         bounds_.back());
            prepare_main(graph, batch, true);
            p.accept_forward(batch);
            p.device.synchronize();
            r.workspace.reset_peak();
            graph.accept.prepare(p.cuda, [&] { p.accept_forward(batch); });
            graph.accept_workspace = r.workspace.peak_used();
        }
    }
    if (r.experts) r.experts->restore_cache_after_preparation();
    p.device.synchronize();
    r.workspace.reset_peak();
}

TextGraph& Graphs::main(std::uint32_t batch, bool verify, std::uint32_t visible, bool head) {
    auto& group = batches_.at(batch - 1);
    auto& graph = verify ? group.verify : group.main;
    graph.set_head(head);
    graph.select(visible);
    workspace_peak_ = std::max(workspace_peak_, graph.workspace_bytes());
    return graph;
}

void Graphs::draft(std::uint32_t batch, std::span<const execution::BatchView> calls,
                   ple::Ticket ticket, std::span<const std::uint32_t> extents) {
    auto& group = batches_.at(batch - 1);
    auto& chain = group.chain;
    for (std::size_t step = 0; step < calls.size(); ++step) {
        const auto& call = group.drafts[step];
        const auto found =
            std::lower_bound(bounds_.begin(), bounds_.end(), calls[step].max_visible_tokens);
        if (found == bounds_.end()) throw std::logic_error("MTP graph context exceeds capacity");
        const auto selected = static_cast<std::size_t>(found - bounds_.begin());
        workspace_peak_     = std::max(workspace_peak_, call.workspace_bytes[selected]);
        if (chain.installed[step] != selected) {
            chain.graph.executable.update_child(chain.steps[step], call.definitions[selected]);
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
    workspace_peak_ = std::max(workspace_peak_, batches_.at(batch - 1).sample_workspace);
    batches_.at(batch - 1).sample.executable.launch(owner_.cuda.stream());
}

void Graphs::accept(std::uint32_t batch) {
    workspace_peak_ = std::max(workspace_peak_, batches_.at(batch - 1).accept_workspace);
    batches_.at(batch - 1).accept.executable.launch(owner_.cuda.stream());
}

void MtpCommitGraphs::prepare(GraphInputs& inputs) {
    auto& p          = owner_;
    auto& r          = p.resources;
    const auto& io   = r.plan.persistent.buffers;
    const auto width = p.shape.draft_tokens + 1;
    batches_.resize(p.shape.max_concurrency);
    for (std::uint32_t batch = 1; batch <= p.shape.max_concurrency; ++batch) {
        inputs.batch(r.main_batch(width, batch, p.shape.max_context), p.shape.max_context);
        inputs.batch(r.mtp_batch(io.mtp->draft_steps[0], 1, batch, p.shape.max_context),
                     p.shape.max_context);
        std::vector<std::int32_t> slots(batch);
        for (std::uint32_t b = 0; b < batch; ++b) slots[b] = b;
        write_values<std::int32_t>(r.tensor(io.batch.count_slots).slice(0, 0, batch), slots,
                                   p.cuda);
        write_values<std::int32_t>(r.tensor(io.batch.accepted_columns).slice(0, 0, batch),
                                   std::vector<std::int32_t>(batch, width), p.cuda);
        write_values<std::int32_t>(r.tensor(io.mtp->skip_first).slice(0, 0, batch),
                                   std::vector<std::int32_t>(batch, 1), p.cuda);
        write_values<std::int32_t>(r.tensor(io.mtp->last_columns).slice(0, 0, batch),
                                   std::vector<std::int32_t>(batch, width - 1), p.cuda);
        write_values<std::int32_t>(r.tensor(io.mtp->active_columns).slice(0, 0, batch),
                                   std::vector<std::int32_t>(batch, 0), p.cuda);
        write_values<std::int32_t>(r.tensor(io.batch.count_columns).slice(0, 0, batch),
                                   std::vector<std::int32_t>(batch, 0), p.cuda);
        const auto destination = r.tensor(io.commit_destination_offsets).slice(0, 0, batch);
        const auto source      = r.tensor(io.batch.destination_offsets).slice(0, 0, batch);
        copy_bytes({destination.data, destination.bytes(), p.cuda.device()},
                   {source.data, source.bytes(), p.cuda.device()}, p.cuda);
        r.workspace.reset_peak();
        auto& target = batches_[batch - 1];
        target.graph.prepare(p.cuda, [&] {
            p.execute_mtp_commit(batch, true, true, &target.repair, &target.counts);
        });
        target.workspace_bytes = r.workspace.peak_used();
    }
    r.workspace.reset_peak();
}

void MtpCommitGraphs::replay(std::uint32_t batch, bool rebuild_history, bool count_tokens) {
    auto& target = batches_.at(batch - 1);
    target.repair.set_enabled(target.graph.executable, rebuild_history);
    target.counts.set_enabled(target.graph.executable, count_tokens);
    workspace_peak_ = std::max(workspace_peak_, target.workspace_bytes);
    target.graph.launch(owner_.cuda);
}

void RankProgramImpl::initialize_graphs() {
    if (!shape.cuda_graph) return;
    nvtx::ScopedRange range(nvtx::Name::CudaGraphPrepare, nvtx::Category::Graph);
    const auto before = device_memory_snapshot(cuda.device());
    GraphInputs inputs(*this);
    inputs.prepare();
    if (host_ple) {
        graphs = std::make_unique<Graphs>(*this);
        graphs->prepare(inputs);
    } else {
        device_graphs = std::make_unique<DeviceGraphs>(*this);
        device_graphs->prepare(inputs);
    }
    if (parameters.mtp && shape.draft_tokens) {
        commit_graphs = std::make_unique<MtpCommitGraphs>(*this);
        commit_graphs->prepare(inputs);
    }
    graph_memory.emplace();
    graph_memory->preparation_increment_bytes =
        device_memory_growth(before, device_memory_snapshot(cuda.device()));
    device.synchronize();
}

void RankProgramImpl::release_graphs() {
    resources.synchronize();
    graphs.reset();
    device_graphs.reset();
    commit_graphs.reset();
}

} // namespace ninfer::models::qwen4_exp::detail
