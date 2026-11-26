#include "models/inkling2/program/resources.h"

#include "core/transfer.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ninfer::models::inkling2 {
namespace {

ExecutionPlan checked(ExecutionPlan p) {
    if (!p.profile || !p.capacity.device_bytes ||
        p.capacity.runtime_reservation_bytes != p.reservation_bytes())
        throw std::invalid_argument("Inkling2 Resources require a finalized capacity plan");
    return p;
}

} // namespace

Resources::Resources(ExecutionPlan input, DeviceContext& device)
    : plan(checked(std::move(input))), device_(device),
      context_(device.device(), plan.capacity.device_bytes),
      fixed_(device.device(), plan.fixed_bytes), workspace_(device.device(), plan.workspace.bytes),
      ingress_(plan.frame.host_ingress.bytes), sampling_(plan.frame.sampling_ingress.bytes),
      egress_(plan.pinned_output_bytes), context_counts_(plan.pinned_context_counts_bytes()),
      workspace(workspace_.span().subspan(0, plan.workspace.general_bytes)),
      states(context_.span(), plan.state), main(context_.span(), fixed_.span(), plan.main) {
    if (plan.workspace.vision) vision.emplace(device, *plan.workspace.vision);
    std::vector<context::ContextTableStorage> tables{{0, &main.tables()}};
    if (plan.dspark) {
        dspark.emplace(context_.span(), fixed_.span(), *plan.dspark);
        tables.push_back({1, &dspark->tables()});
    }
    context = std::make_unique<context::ContextStorage>(context_.span(), MemorySpan{}, tables,
                                                        plan.profile->local.rank);
    compute_actions = std::make_unique<context::ContextMigration>(
        device, *context, device.execution_view(), plan.mapping_elements, 0);
    context_actions = std::make_unique<context::ContextMigration>(
        device, *context, device.transfer_view(), plan.mapping_elements, 0);

    std::memset(ingress_.data(), 0, ingress_.size());
    copy_bytes_async(plan.frame.host_ingress.bind(fixed_.span()), ingress_.span(),
                     device.execution_view());

    const auto clear = [&](const TensorRegion& r, int value) {
        const auto region = r.region.bind(fixed_.span());
        CUDA_CHECK(cudaMemsetAsync(region.data, value, region.bytes, device.stream()));
    };
    clear(plan.main.tables.page_offsets, 255);
    if (plan.dspark) clear(plan.dspark->tables.page_offsets, 255);
    if (plan.frame.token_counts) clear(*plan.frame.token_counts, 0);
    if (plan.frame.dspark) {
        const auto& d = *plan.frame.dspark;
        clear(d.initial_markov_state, 0);
        for (std::uint32_t b = 1; b <= plan.shape.max_concurrency; ++b) {
            for (std::uint32_t steps = 1; steps <= plan.shape.draft_tokens; ++steps) {
                const auto index = (b - 1) * plan.shape.draft_tokens + steps - 1;
                std::vector<int> stem(b * (steps + 1), dimension(b));
                std::vector<int> tail(b * steps);
                for (std::uint32_t i = 0; i < b; ++i) {
                    stem[i * (steps + 1)] = extent(i);
                    for (std::uint32_t j = 0; j < steps; ++j)
                        tail[j * b + i] = extent(i * steps + j);
                }
                const auto upload = [&](const TensorRegion& region,
                                        const std::vector<int>& values) {
                    const auto dest = tensor(region);
                    CUDA_CHECK(
                        cudaMemcpy(dest.data, values.data(), dest.bytes(), cudaMemcpyHostToDevice));
                };
                upload(d.stem_indices[index], stem);
                upload(d.tail_indices[index], tail);
            }
        }
    }
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

RankMemoryUsage Resources::observe() const {
    RankMemoryUsage out;
    out.allocations.persistent_device_bytes = fixed_.size();
    out.allocations.context_device_bytes    = context_.size();
    out.allocations.workspace_device_bytes  = workspace_.size();
    out.allocations.io_pinned_bytes         = ingress_.size() + sampling_.size() + egress_.size() +
                                      context_counts_.size() + compute_actions->pinned_bytes() +
                                      context_actions->pinned_bytes();
    if (vision) {
        out.vision = vision->observe();
        out.allocations.persistent_device_bytes += vision->capacity().output_bytes;
        out.allocations.io_pinned_bytes += vision->capacity().encoder.host_staging_bytes;
    }
    const auto peak = std::max(workspace.peak_used(), vision ? vision->workspace_peak_bytes() : 0);
    out.workspace   = {workspace.used(), peak, peak};
    return out;
}

} // namespace ninfer::models::inkling2
