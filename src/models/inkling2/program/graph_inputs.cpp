#include "models/inkling2/program/graph_inputs.h"

#include "core/transfer.h"
#include "models/inkling2/program/internal.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::inkling2::detail {
namespace {

std::vector<RecordSpec> specs(const ExecutionPlan& p) {
    std::vector<RecordSpec> out{{p.state.image.image_bytes, p.state.image.alignment},
                                {p.main.page.page_bytes, p.main.page.alignment}};
    if (p.dspark) out.push_back({p.dspark->page.page_bytes, p.dspark->page.alignment});
    return out;
}

std::vector<RecordRequest> demand(const ContextPoolGeometry& geometry, std::uint32_t batch) {
    std::vector<RecordRequest> out;
    for (auto id : geometry.spec_classes) out.push_back({id, batch});
    return out;
}

} // namespace

std::size_t graph_context_bytes(const ExecutionPlan& p) {
    const auto geometry = describe_context_pool(specs(p));
    return geometry.required_bytes(demand(geometry, p.shape.max_concurrency));
}

GraphInputs::GraphInputs(RankProgramImpl& p) : owner_(p) {}

void GraphInputs::prepare() {
    auto& p             = owner_;
    auto& r             = p.resources;
    const auto records  = specs(r.plan);
    const auto geometry = describe_context_pool(records);
    const auto requests = demand(geometry, p.shape.max_concurrency);
    const auto bytes    = geometry.required_bytes(requests);
    if (bytes > r.context_backing().bytes)
        throw std::logic_error("Inkling2 Graph inputs exceed Context capacity");
    records_         = std::make_unique<ContextPool>(0, bytes, records);
    auto reservation = records_->reserve(requests);
    if (!reservation.reservation || !reservation.reservation->ready())
        throw std::logic_error("Inkling2 Graph Context reservation failed");

    const auto allocate = [&](int kind) {
        const auto id = reservation.reservation->materialize(records_->record_class(kind));
        allocations_.push_back(id);
        const auto binding = records_->resolve(id);
        const auto span    = r.context_backing().subspan(binding.offset, binding.bytes);
        CUDA_CHECK(cudaMemsetAsync(span.data, 0, span.bytes, p.cuda.stream()));
        return std::int64_t(binding.offset);
    };

    for (std::uint32_t row = 0; row < p.shape.max_concurrency; ++row)
        states_.push_back(allocate(0));

    const auto mapping_bytes = [&](const CacheViews& cache) {
        return cache.tables().logical_page_capacity() * p.shape.max_concurrency *
               sizeof(std::int64_t);
    };
    mapping_ = PinnedHostBuffer(mapping_bytes(r.main) + (r.dspark ? mapping_bytes(*r.dspark) : 0));
    const auto map = [&](CacheViews& cache, int kind, std::size_t offset) {
        auto& tables         = cache.tables();
        const auto capacity  = tables.logical_page_capacity();
        const auto row_bytes = std::size_t(capacity) * sizeof(std::int64_t);
        for (std::uint32_t row = 0; row < p.shape.max_concurrency; ++row) {
            const auto page_offset = allocate(kind);
            const auto staging     = mapping_.span().subspan(offset + row * row_bytes, row_bytes);
            auto* offsets          = static_cast<std::int64_t*>(staging.data);
            std::fill_n(offsets, capacity, page_offset);
            tables.publish(row, 0, {offsets, capacity}, staging, p.cuda);
        }
    };
    map(r.main, 1, 0);
    if (r.dspark) map(*r.dspark, 2, mapping_bytes(r.main));
    p.device.synchronize();
}

GraphInputs::~GraphInputs() {
    (void)cudaStreamSynchronize(owner_.cuda.stream());
    for (auto id : allocations_) records_->release(id);
}

void GraphInputs::set(std::int32_t rows, std::int32_t width, std::int32_t steps) {
    auto& p = owner_;
    p.device.synchronize();
    p.select_frame({std::uint32_t(rows), std::uint32_t(rows)});
    const auto& io = p.frame;
    const auto& g  = io.rows;
    const int base = std::clamp(dimension(p.shape.max_context) - std::max(width, steps + 1), 0, 4);
    const auto i32 = [&](const TensorRegion& r) {
        return static_cast<std::int32_t*>(p.host(r).data);
    };
    const auto i64 = [&](const TensorRegion& r) {
        return static_cast<std::int64_t*>(p.host(r).data);
    };

    for (int j = 0; j < rows; ++j) {
        i32(g.anchors)[j]             = 0;
        i32(g.live_rows)[j]           = 1;
        i32(g.valid_columns)[j]       = width;
        i64(g.source_offsets)[j]      = states_[j];
        i64(g.destination_offsets)[j] = states_[j];
        i32(g.main_tables)[j]         = j;
        i32(g.count_slots)[j]         = j;
        i32(g.sampling_positions)[j]  = base;
        if (io.dspark) {
            i32(io.decode.ids)[j]               = 0;
            i32(g.dspark_tables)[j]             = j;
            i32(g.base_positions)[j]            = base;
            i32(g.proposal_rows)[j]             = steps > 0;
            i32(g.proposal_columns)[j]          = steps ? steps + 1 : 0;
            i32(g.draft_extents)[j]             = width - 1;
            i32(g.snapshot_slots)[j]            = j * width;
            i32(io.dspark->accepted_columns)[j] = width;
            i32(io.dspark->last_columns)[j]     = width - 1;
            i32(g.count_columns)[j]             = 0;
        }
        for (int k = 0; k < width; ++k) i32(io.decode.positions)[j * width + k] = base + k;
    }
    auto* configs = static_cast<ops::SamplingConfig*>(p.host(*io.sampling).data);
    for (int j = 0; j < rows; ++j) configs[j] = {.temperature = 0, .top_k = 1, .top_p = 1};
    p.upload_region(io.ingress, io.ingress.bytes);
    if (io.dspark) { p.upload_region(io.dspark->commit_ingress, io.dspark->commit_ingress.bytes); }
}

} // namespace ninfer::models::inkling2::detail
