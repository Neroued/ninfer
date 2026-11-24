#include "models/qwen4_exp/program/graph_inputs.h"

#include "core/transfer.h"
#include "models/qwen4_exp/program/internal.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

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
    auto& r           = owner_.resources;
    auto call         = r.main_batch(owner_.shape.draft_tokens + 1, rows, visible);
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

void GraphInputs::stage_commit(std::uint32_t rows, std::uint32_t width) {
    auto& p             = owner_;
    auto& r             = p.resources;
    const auto& io      = r.buffers();
    const auto live     = r.plan.batches->minimum_rows(rows);
    const auto valid    = std::min(width, p.shape.max_context);
    const auto frontier = p.shape.max_context - valid;
    std::vector<std::int32_t> bases(rows), previous_rope(3 * rows), accepted(rows), skip(rows),
        last(rows, -1), active(rows, -1);
    for (std::uint32_t b = 0; b < live; ++b) {
        accepted[b] = valid;
        skip[b]     = !frontier || width > 1;
        bases[b]    = frontier ? frontier - 1 + (width > 1) : 0;
        std::fill_n(previous_rope.begin() + 3 * b, 3, frontier ? frontier - 1 : 0);
        last[b]   = valid - 1;
        active[b] = 0;
    }
    const auto write = [&](const TensorRegion& region, std::span<const std::int32_t> values) {
        write_values<std::int32_t>(r.tensor(region), values, p.cuda);
    };
    write(io.batch.accepted_columns, accepted);
    write(io.mtp->skip_first, skip);
    write(io.mtp->last_columns, last);
    write(io.mtp->active_columns, active);
    write(io.batch.count_columns, std::vector<std::int32_t>(rows, 0));
    write(io.mtp->history_base_positions, bases);
    write(io.mtp->previous_rope, previous_rope);
    const auto destination = r.tensor(io.commit_destination_offsets).slice(0, 0, rows);
    const auto source      = r.tensor(io.batch.destination_offsets).slice(0, 0, rows);
    copy_bytes({destination.data, destination.bytes(), p.cuda.device()},
               {source.data, source.bytes(), p.cuda.device()}, p.cuda);
}

} // namespace ninfer::models::qwen4_exp::detail
