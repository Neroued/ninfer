#include "models/glm5next/program/resources.h"
#include "core/transfer.h"
#include "ninfer/ops/sampling.h"

#include <algorithm>
#include <cstring>

namespace ninfer::models::glm5next {
namespace {
ExecutionPlan checked(ExecutionPlan p) {
    if (!p.profile || !p.capacity.device_bytes ||
        p.capacity.runtime_reservation_bytes != p.reservation_bytes())
        throw std::invalid_argument("GLM5Next Resources require a finalized capacity plan");
    return p;
}
} // namespace

Resources::Resources(ExecutionPlan input, DeviceContext& device, ContextCommunication peer)
    : plan(checked(std::move(input))), device_(device),
      context_(device.device(), plan.capacity.device_bytes),
      fixed_(device.device(), plan.fixed_bytes), workspace_(device.device(), plan.workspace.bytes),
      ingress_(plan.frame.ingress.bytes), sampling_(plan.frame.sampling_ingress.bytes),
      egress_(plan.pinned_output_bytes),
      context_counts_(plan.frame.token_counts ? plan.frame.token_counts->region.bytes : 0),
      workspace(workspace_.span().subspan(0, plan.workspace.general_bytes)),
      states(context_.span(), plan.state), main(context_.span(), fixed_.span(), plan.main) {
    if (plan.workspace.vision) vision.emplace(device, *plan.workspace.vision);
    if (plan.mtp) mtp = std::make_unique<CacheViews>(context_.span(), fixed_.span(), *plan.mtp);
    std::vector<context::ContextTableStorage> tables{{0, &main.tables()}};
    if (mtp) tables.push_back({1, &mtp->tables()});
    context = std::make_unique<context::ContextStorage>(context_.span(), MemorySpan{}, tables,
                                                        plan.profile->local.rank);
    const auto binding = [&](std::size_t slot) {
        return peer.group ? context::ContextPeerBinding{peer.group, peer.members,
                                                        plan.peer_staging[slot].bind(fixed_.span()),
                                                        plan.peer_staging[slot].bytes}
                          : context::ContextPeerBinding{};
    };
    compute_actions = std::make_unique<context::ContextMigration>(
        device, *context, device.execution_view(), plan.mapping_elements, 0, binding(0));
    context_actions = std::make_unique<context::ContextMigration>(
        device, *context, device.transfer_view(), plan.mapping_elements, 0, binding(1));
    if (plan.frame.kda_records) {
        kda_records.emplace(fixed_.span(), *plan.frame.kda_records);
        kda_fold.emplace(*kda_records, states.kda().all_layers_view());
    }
    std::memset(ingress_.data(), 0, ingress_.size());
    if (const auto& m = plan.frame.mtp) {
        for (std::size_t slot = 0; slot < m->tail_offsets.size(); ++slot) {
            const auto offset = m->tail_offsets[slot].region.offset - plan.frame.ingress.offset;
            auto* values =
                reinterpret_cast<std::int64_t*>(static_cast<std::byte*>(ingress_.data()) + offset);
            for (std::uint32_t row = 0; row < plan.shape.max_concurrency; ++row)
                values[row] = (slot * plan.shape.max_concurrency + row) * m->tail_keys.shape[0] *
                              m->tail_keys.shape[1] * sizeof(std::uint16_t);
        }
    }
    copy_bytes_async(plan.frame.ingress.bind(fixed_.span()), ingress_.span(),
                     device.execution_view());
    if (const auto& m = plan.frame.mtp) {
        std::array<ops::SamplingConfig, kMaximumConcurrency> proposal;
        std::array<std::int32_t, kMaximumConcurrency> columns;
        for (std::uint32_t i = 0; i < plan.shape.max_concurrency; ++i) {
            proposal[i] = {.temperature = 0, .top_k = 1, .top_p = 1};
            columns[i]  = 1;
        }
        // Every physical proposal row produces a vocabulary ID, including inert rows whose
        // dense stem still reads that ID in the next recursive step.
        copy_bytes(m->proposal_sampling.region.bind(fixed_.span()),
                   {proposal.data(), m->proposal_sampling.region.bytes, Device::cpu()},
                   device.execution_view());
        copy_bytes(m->proposal_columns.region.bind(fixed_.span()),
                   {columns.data(), m->proposal_columns.region.bytes, Device::cpu()},
                   device.execution_view());
    }
    const auto clear = [&](const TensorRegion& r, int value) {
        const auto region = r.region.bind(fixed_.span());
        CUDA_CHECK(cudaMemsetAsync(region.data, value, region.bytes, device.stream()));
    };
    clear(plan.main.tables.page_offsets, 255);
    if (plan.mtp) clear(plan.mtp->tables.page_offsets, 255);
    if (plan.frame.token_counts) clear(*plan.frame.token_counts, 0);
    if (plan.frame.mtp) clear(plan.frame.mtp->token_counts, 0);
    device.synchronize();
}

Resources::~Resources() {
    try {
        synchronize();
    } catch (const std::logic_error&) { std::terminate(); } catch (...) {
    }
}

void Resources::synchronize() const {
    std::exception_ptr error;
    const auto attempt = [&](auto&& f) {
        try {
            f();
        } catch (...) {
            if (!error) error = std::current_exception();
        }
    };
    if (compute_actions) attempt([&] { compute_actions->drain(); });
    if (context_actions) attempt([&] { context_actions->drain(); });
    attempt([&] { device_.synchronize(); });
    if (error) std::rethrow_exception(error);
}

Tensor Resources::tensor(const TensorRegion& region) { return region.bind(fixed_.span()); }

Tensor Resources::packed(const TensorRegion& region, std::initializer_list<std::int32_t> shape) {
    std::uint64_t elements = 1;
    for (const auto n : shape) elements *= dimension(n);
    auto value = tensor(region);
    return value.view({dimension(value.numel())}).slice(0, 0, dimension(elements)).view(shape);
}

ops::KpoolTailStateView Resources::mtp_tail(std::uint32_t slot, int rows) {
    const auto& m = plan.frame.mtp.value();
    if (slot >= m.tail_offsets.size() || rows < 1 ||
        rows > static_cast<int>(plan.shape.max_concurrency))
        throw std::out_of_range("GLM5Next provisional tail exceeds frame capacity");
    const auto offsets = tensor(m.tail_offsets[slot]).slice(0, 0, rows);
    const int d = m.tail_keys.shape[0], tail = m.tail_keys.shape[1];
    const TensorRegion field{
        {0, std::size_t(d) * tail * sizeof(std::uint16_t), 16}, DType::BF16, {d, tail, 1, 1}};
    return {{{m.tail_keys.region.bind(fixed_.span()), offsets}, {field, 1, 0}},
            {{m.tail_gates.region.bind(fixed_.span()), offsets}, {field, 1, 0}}};
}

RankMemoryUsage Resources::observe() const {
    RankMemoryUsage out;
    out.allocations.persistent_device_bytes = fixed_.size();
    out.allocations.context_device_bytes    = context_.size();
    out.allocations.workspace_device_bytes  = workspace_.size();
    out.allocations.io_pinned_bytes         = ingress_.size() + sampling_.size() + egress_.size() +
                                      context_counts_.size() + compute_actions->pinned_bytes() +
                                      context_actions->pinned_bytes();
    const auto peak = std::max(workspace.peak_used(), vision ? vision->workspace_peak_bytes() : 0);
    out.workspace   = {workspace.used(), peak, peak};
    if (vision) {
        out.vision = vision->observe();
        out.allocations.persistent_device_bytes += vision->capacity().output_bytes;
        out.allocations.io_pinned_bytes += vision->capacity().encoder.host_staging_bytes;
    }
    return out;
}

} // namespace ninfer::models::glm5next
