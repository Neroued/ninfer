#include "models/qwen4_exp/program/internal.h"
#include "core/transfer.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

void RankProgramImpl::advance_context() {
    if (!context) return;
    auto& f          = *context;
    const auto& work = std::get<ContextWork>(f.request.work.operation);
    if (!f.submitted) {
        const auto transfer   = device.transfer_view();
        const auto vocabulary = parameters.profile->public_tokens;
        auto* counts          = static_cast<std::int32_t*>(resources.context_counts().data);
        for (const auto id : work.releases) {
            auto& resident = binding(id);
            if (host_ple) host_ple->cancel_sequence(resident.member.sequence.value);
            retire_vision(resident);
            bindings[id.lane.value].reset();
        }
        for (std::size_t b = 0; b < work.bindings.size(); ++b) {
            const auto& input = work.bindings[b];
            const auto lane   = input.member.residency.lane.value;
            if (bindings[lane]) retire_vision(*bindings[lane]);
            bindings[lane].emplace(ResidentBinding{input.member, input.vision, {}, 0});
            if (resources.plan.persistent.buffers.token_counts) {
                auto output = device_tensor(*resources.plan.persistent.buffers.token_counts)
                                  .slice(1, lane, 1);
                if (input.generated.empty()) {
                    CUDA_CHECK(cudaMemsetAsync(output.data, 0, output.bytes(), transfer.stream()));
                } else {
                    auto* row = counts + b * vocabulary;
                    std::fill_n(row, vocabulary, 0);
                    for (const auto id : input.generated) ++row[id];
                    copy_bytes_async({output.data, output.bytes(), transfer.device()},
                                     {row, output.bytes(), Device::cpu(), true}, transfer);
                }
            }
        }
        f.submitted = true;
        context_done.record(transfer.stream());
    }
    if (!context_done.ready()) return;
    ContextResult result;
    const auto observations = resources.context_actions->observations();
    result.transfers.assign(observations.begin(), observations.end());
    resources.context_actions->finish();
    const auto ticket  = f.request.ticket;
    const auto command = f.request.command;
    context.reset();
    events.emplace_back(runtime::ResultReady<ExecutionTypes>{ticket, Result(std::move(result))});
    events.emplace_back(runtime::DeviceRetired{ticket, command});
}

} // namespace ninfer::models::qwen4_exp::detail
