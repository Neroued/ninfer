#include "models/qwen4_exp/program/graphs.h"
#include "models/qwen4_exp/program/internal.h"
#include "core/startup.h"
#include "core/transfer.h"
#include "core/nvtx.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {
namespace {

std::vector<std::uint32_t> context_bounds(std::uint32_t capacity) {
    std::vector<std::uint32_t> out;
    for (const auto bound : {128U, 512U, 2048U, 8192U, 32768U, 131072U}) {
        if (bound >= capacity) break;
        out.push_back(bound);
    }
    out.push_back(capacity);
    return out;
}

void write_i32(const Tensor& target, std::span<const std::int32_t> values, CudaExecutionView cuda) {
    copy_bytes({target.data, target.bytes(), cuda.device()},
               {values.data(), values.size_bytes(), Device::cpu()}, cuda);
}

// Private histories keep warmup reads valid through every context envelope. Only their own
// pages are repeated; no live request can observe these temporary tables or state images.
class CaptureInputs {
public:
    explicit CaptureInputs(ProgramImpl& owner) : p_(owner) {}

    void prepare() {
        auto& r = p_.resources;
        for (std::uint32_t b = 0; b < p_.shape.max_concurrency; ++b) {
            const auto state = r.state_store->reserve_destination();
            if (!state) throw std::bad_alloc();
            states_.push_back(*state);
            slots_.push_back(r.state_store->physical_slot(*state));
            r.states->storage().clear(slots_.back(), p_.cuda);
            r.state_store->complete_initialization(*state);
        }
        for (std::uint32_t pool = 0; pool < r.pool_count(); ++pool) {
            auto& storage = r.pool(pool);
            for (std::uint32_t b = 0; b < p_.shape.max_concurrency; ++b) {
                auto binding = storage.histories.prepare_binding(nullptr, 0, 1, b);
                binding.publish_restores();
                binding.prepare_tail_copy();
                binding.enqueue_mapping(p_.cuda.stream());
                histories_[pool].push_back(binding.commit());
                auto& history = histories_[pool].back();
                storage.histories.prepare_write(history, 0, 1, p_.cuda.stream());
                const auto page = storage.histories.physical_page(history.handle(), 0);
                storage.cache.pages().zero_pages({&page, 1}, p_.cuda.stream());
                storage.cache.tables().publish_repeated(
                    storage.histories.execution_row(history.handle()).handle(), page,
                    storage.cache.tables().logical_page_capacity(), p_.cuda.stream());
            }
        }
        for (std::uint32_t slot = 0; slot < 2; ++slot) {
            auto embedding = r.ple_upload->device_slot(slot, r.plan.ple.max_positions, 1);
            CUDA_CHECK(cudaMemsetAsync(embedding.data, 0, embedding.bytes(), p_.cuda.stream()));
        }
        const auto& buffers = r.plan.persistent.buffers;
        if (buffers.mtp) {
            for (const auto* region :
                 {&buffers.mtp->base_conditioning, &buffers.mtp->hidden, &buffers.mtp->tail_keys,
                  &buffers.mtp->tail_coordinates, &buffers.mtp->drafts, &buffers.mtp->extents,
                  &buffers.mtp->anchors, &buffers.mtp->frontiers}) {
                const auto value = r.tensor(*region);
                CUDA_CHECK(cudaMemsetAsync(value.data, 0, value.bytes(), p_.cuda.stream()));
            }
        }
        std::vector<ops::SamplingConfig> configs(p_.shape.max_concurrency);
        for (auto& config : configs) config.temperature = 0;
        const auto target = r.tensor(*buffers.sampling);
        copy_bytes({target.data, target.bytes(), p_.cuda.device()},
                   {configs.data(), configs.size() * sizeof(configs[0]), Device::cpu()}, p_.cuda);
    }

    ~CaptureInputs() {
        p_.device.synchronize();
        for (auto& histories : histories_) histories.clear();
        for (const auto state : states_)
            if (!p_.resources.state_store->release(state)) std::terminate();
    }

    void batch(execution::BatchView call, std::uint32_t pool, std::uint32_t visible) {
        const auto t = call.tokens, b = call.batch;
        const auto valid = std::min<std::uint32_t>(t, visible);
        const auto base  = visible - valid;
        std::vector<std::int32_t> ids(t * b, p_.parameters.model.config().text.ple.eos);
        std::vector<std::int32_t> positions(t * b), rope(t * b * 3), rows(b), bases(b, base);
        for (int row = 0; row < b; ++row) {
            rows[row] = p_.resources.pool(pool).histories.bound_row(histories_[pool][row].handle());
            for (int j = 0; j < t; ++j) {
                const auto position    = base + std::min<std::uint32_t>(j, valid - 1);
                positions[j + t * row] = position;
                for (int axis = 0; axis < 3; ++axis) rope[j + t * (axis + 3 * row)] = position;
            }
        }
        write_i32(call.ids, ids, p_.cuda);
        write_i32(call.positions, positions, p_.cuda);
        write_i32(call.rope_positions, rope, p_.cuda);
        write_i32(call.source_slots, std::span(slots_).first(b), p_.cuda);
        write_i32(call.destination_slots, std::span(slots_).first(b), p_.cuda);
        write_i32(call.table_rows, rows, p_.cuda);
        write_i32(call.base_positions, bases, p_.cuda);
        std::vector<std::int32_t> counts(b, valid), offsets(b, 0);
        write_i32(call.valid_columns, counts, p_.cuda);
        write_i32(call.record_offsets, offsets, p_.cuda);
    }

private:
    ProgramImpl& p_;
    std::vector<Image> states_;
    std::vector<std::int32_t> slots_;
    std::array<std::vector<context::KVHistory>, 2> histories_;
};
} // namespace

Graphs::Graphs(ProgramImpl& owner)
    : owner_(owner), bounds_(context_bounds(owner.shape.max_context)) {}

void Graphs::prepare_main(Batch& graph, std::uint32_t batch, bool verify) {
    auto& p             = owner_;
    auto& r             = p.resources;
    const auto& buffers = r.plan.persistent.buffers;
    const auto tokens   = verify ? p.shape.draft_tokens + 1 : 1;
    const auto call     = r.main_batch(tokens, batch, bounds_.back());
    const auto& c       = p.parameters.model.config().text;
    const auto t = static_cast<std::int32_t>(tokens), b = static_cast<std::int32_t>(batch);
    execution::TextOutputs output{r.packed(buffers.pending_sample, {dimension(c.hidden), t, b}),
                                  {}};
    if (buffers.pending_conditioning)
        output.conditioning =
            r.packed(*buffers.pending_conditioning, {dimension(c.residual_width()), t, b});
    const execution::TextExecution execution{r.workspace,
                                             *r.states,
                                             r.main->cache,
                                             r.experts.get(),
                                             r.tensor(buffers.main_qsa.keys),
                                             r.tensor(buffers.main_qsa.coordinates),
                                             r.tensor(buffers.ple_records)};
    execution::StateUpdate update = execution::StepUpdate{};
    if (verify) update.emplace<execution::VerifyRecords>(*r.gdn_records);
    const std::array ple{r.ple_upload->device_slot(0, tokens, batch),
                         r.ple_upload->device_slot(1, tokens, batch)};
    auto& selected = verify ? graph.verify : graph.ordinary;
    {
        // Warm the actual bodies before capture. No disk request is needed for these bindings.
        execution::TextRun run(p.parameters, execution, call, update, p.cuda);
        run.prefix();
        run.consume_ple(ple[0]);
        run.suffix(output);
        p.device.synchronize();
        selected.prepare(run, p.parameters, r.experts.get(), bounds_, ple, output, tokens * batch,
                         r.workspace, p.device);
    }
    selected.select(bounds_.back());
    execution::TextRun run(p.parameters, execution, call, update, p.cuda);
    selected.prefix(p.cuda);
    selected.consume_ple(0, p.cuda);
    selected.suffix(p.cuda);
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
        call.executable.instantiate(call.definitions.front());
        for (const auto& definition : call.definitions) {
            call.executable.update(definition);
            call.executable.upload(p.cuda.stream());
            p.device.synchronize();
        }
        call.installed = bounds_.size() - 1;
        call.executable.launch(p.cuda.stream());
        p.device.synchronize();
    }
}

void Graphs::prepare() {
    auto& p = owner_;
    auto& r = p.resources;
    std::optional<moe::Schedule::Preparation> preparation;
    if (r.experts) preparation.emplace(*r.experts);
    CaptureInputs inputs(p);
    inputs.prepare();
    batches_.resize(p.shape.max_concurrency);
    for (std::uint32_t batch = 1; batch <= p.shape.max_concurrency; ++batch) {
        auto& graph = batches_[batch - 1];
        inputs.batch(r.main_batch(1, batch, bounds_.back()), 0, bounds_.back());
        prepare_main(graph, batch, false);
        const auto hidden =
            r.packed(r.plan.persistent.buffers.pending_sample,
                     {dimension(p.parameters.model.config().text.hidden), static_cast<int>(batch)});
        p.sample_forward(batch, hidden, false);
        p.device.synchronize();
        r.workspace.reset_peak();
        graph.sample.prepare(p.cuda, [&] { p.sample_forward(batch, hidden, false); });
        graph.sample_workspace = r.workspace.peak_used();
        if (p.parameters.mtp) {
            const auto& frame = *r.plan.persistent.buffers.mtp;
            for (std::uint32_t step = 0; step < p.shape.draft_tokens; ++step)
                inputs.batch(r.mtp_batch(frame.draft_steps[step], 1, batch, bounds_.back()), 1,
                             bounds_.back());
            prepare_drafts(graph, batch);
            inputs.batch(r.main_batch(p.shape.draft_tokens + 1, batch, bounds_.back()), 0,
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

TextGraph& Graphs::main(std::uint32_t batch, bool verify, std::uint32_t visible) {
    auto& group = batches_.at(batch - 1);
    auto& graph = verify ? group.verify : group.ordinary;
    graph.select(visible);
    workspace_peak_ = std::max(workspace_peak_, graph.workspace_bytes());
    return graph;
}

void Graphs::draft(std::uint32_t batch, std::uint32_t step, std::uint32_t visible) {
    auto& call       = batches_.at(batch - 1).drafts.at(step);
    const auto found = std::lower_bound(bounds_.begin(), bounds_.end(), visible);
    if (found == bounds_.end()) throw std::logic_error("MTP graph context exceeds capacity");
    const auto selected = static_cast<std::size_t>(found - bounds_.begin());
    workspace_peak_     = std::max(workspace_peak_, call.workspace_bytes[selected]);
    if (call.installed != selected) {
        call.executable.update(call.definitions[selected]);
        call.installed = selected;
    }
    call.executable.launch(owner_.cuda.stream());
}

void Graphs::sample(std::uint32_t batch) {
    workspace_peak_ = std::max(workspace_peak_, batches_.at(batch - 1).sample_workspace);
    batches_.at(batch - 1).sample.executable.launch(owner_.cuda.stream());
}

void Graphs::accept(std::uint32_t batch) {
    workspace_peak_ = std::max(workspace_peak_, batches_.at(batch - 1).accept_workspace);
    batches_.at(batch - 1).accept.executable.launch(owner_.cuda.stream());
}

void ProgramImpl::initialize_graphs() {
    if (!shape.cuda_graph) return;
    nvtx::ScopedRange range(nvtx::Name::CudaGraphPrepare, nvtx::Category::Graph);
    const auto before = device_memory_snapshot(cuda.device());
    graphs            = std::make_unique<Graphs>(*this);
    graphs->prepare();
    graph_memory.emplace();
    graph_memory->preparation_increment_bytes =
        device_memory_growth(before, device_memory_snapshot(cuda.device()));
    resources.states->storage().clear_all(cuda);
    device.synchronize();
}

void ProgramImpl::release_graphs() {
    resources.synchronize();
    graphs.reset();
}

} // namespace ninfer::models::qwen4_exp::detail
