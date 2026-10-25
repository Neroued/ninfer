#include "models/glm5next/program/graph_inputs.h"

#include "core/transfer.h"
#include "models/glm5next/program/internal.h"

#include <algorithm>

namespace ninfer::models::glm5next::detail {
namespace {
std::vector<RecordSpec> specs(const ExecutionPlan& p) {
    std::vector<RecordSpec> out{{p.state.owner.image_bytes, p.state.owner.alignment},
                                {p.main.cache.page.page_bytes, p.main.cache.page.alignment}};
    if (p.mtp) out.push_back({p.mtp->cache.page.page_bytes, p.mtp->cache.page.alignment});
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
        throw std::logic_error("GLM Graph inputs exceed Context capacity");
    records_         = std::make_unique<ContextPool>(0, bytes, records);
    auto reservation = records_->reserve(requests);
    if (!reservation.reservation || !reservation.reservation->ready())
        throw std::logic_error("GLM Graph Context reservation failed");
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
    for (int pool = 0; pool < (r.mtp ? 2 : 1); ++pool) {
        auto& tables    = (pool ? *r.mtp : r.main).tables();
        mappings_[pool] = PinnedHostBuffer(p.shape.max_concurrency * sizeof(std::int64_t));
        auto* offsets   = static_cast<std::int64_t*>(mappings_[pool].data());
        for (std::uint32_t row = 0; row < p.shape.max_concurrency; ++row) {
            offsets[row] = allocate(pool + 1);
            tables.publish(
                row, 0, {offsets + row, 1},
                mappings_[pool].span().subspan(row * sizeof(std::int64_t), sizeof(std::int64_t)),
                p.cuda);
        }
    }
    p.device.synchronize();
}

GraphInputs::~GraphInputs() {
    (void)cudaStreamSynchronize(owner_.cuda.stream());
    for (auto id : allocations_) records_->release(id);
}

void GraphInputs::set(int g, int c, int depth) {
    auto& p = owner_;
    auto& r = p.resources;
    p.device.synchronize();
    const auto& io = r.plan.frame;
    const auto& a  = io.attention;
    const int rank = p.parameters.profile->local.rank, ranks = p.parameters.profile->local.size;
    std::vector<int> counts(ranks), begins(ranks);
    for (int remaining = g, i = 0; i < ranks; ++i) {
        counts[i] = std::min(c, remaining);
        remaining -= counts[i];
        if (i) begins[i] = begins[i - 1] + counts[i - 1];
    }
    const int width  = depth ? p.shape.draft_tokens + 1 : 1;
    const int base   = std::max(0, std::min(4, int(p.shape.max_context) - width));
    const int extent = std::min(depth, int(p.shape.max_context) - base - 1);
    const auto i32   = [&](const TensorRegion& region) {
        return static_cast<int*>(p.host(region).data);
    };
    const auto i64 = [&](const TensorRegion& region) {
        return static_cast<std::int64_t*>(p.host(region).data);
    };
    const auto batch = [&](const BatchLayout& b, int t, int start, int valid) {
        for (int j = 0; j < g; ++j) {
            for (int k = 0; k < t; ++k) {
                i32(b.ids)[j * t + k]       = 0;
                i32(b.positions)[j * t + k] = start + std::min(k, std::max(0, valid - 1));
            }
            i32(b.valid_columns)[j] = valid;
        }
    };
    batch(io.main, width, base, extent + 1);
    for (int j = 0; j < g; ++j) {
        i64(io.rows.source_offsets)[j]      = states_[j];
        i64(io.rows.destination_offsets)[j] = states_[j];
        i32(io.rows.base_positions)[j]      = base;
        i32(io.rows.accepted_columns)[j]    = 1;
        i32(io.rows.count_slots)[j]         = j;
        i32(io.rows.sampling_positions)[j]  = base;
    }
    for (int j = 0; j < c; ++j) {
        const bool live               = j < counts[rank];
        const int row                 = begins[rank] + j;
        i32(a.to_global)[j]           = live ? row : -1;
        i32(a.live_rows)[j]           = live;
        i64(a.source_offsets)[j]      = live ? states_[row] : -1;
        i64(a.destination_offsets)[j] = live ? states_[row] : -1;
        i32(a.main_tables)[j]         = live ? row : -1;
        i32(a.mtp_tables)[j]          = live ? row : -1;
        i32(a.base_positions)[j]      = live ? base : 0;
        i32(a.record_offsets)[j]      = 0;
        i32(a.accepted_columns)[j]    = live ? 1 : 0;
        i32(a.active_columns)[j]      = live ? 0 : -1;
    }
    for (int rank = 0; rank < ranks; ++rank)
        for (int j = 0; j < counts[rank]; ++j) i32(a.from_gather)[begins[rank] + j] = rank * c + j;
    if (io.mtp) {
        const auto& m = *io.mtp;
        batch(m.history, width, std::max(base - 1, 0), 1);
        for (int s = 0; s < depth; ++s)
            batch(m.draft_steps[s], 1, std::max(base - 1, 0) + std::min(s, extent), s < extent);
        for (int j = 0; j < g; ++j) {
            i32(m.extents)[j]    = extent;
            i32(m.anchors)[j]    = 0;
            i32(m.skip_first)[j] = depth ? extent != 0 : base == 0;
        }
    }
    p.upload_region(io.ingress, io.ingress.bytes);
    std::vector<ops::SamplingConfig> configs(g);
    for (int j = 0; j < g; ++j)
        configs[j] = {.temperature  = 0,
                      .top_k        = 1,
                      .top_p        = 1,
                      .token_counts = static_cast<int*>(
                          r.tensor(!p.parallel && depth ? *io.mtp->token_counts : *io.token_counts)
                              .slice(1, j, 1)
                              .data)};
    copy_bytes(io.sampling->region.bind(r.fixed()).subspan(0, configs.size() * sizeof(configs[0])),
               {configs.data(), configs.size() * sizeof(configs[0]), Device::cpu()}, p.cuda);
}
} // namespace ninfer::models::glm5next::detail
