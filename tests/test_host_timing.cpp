#include "runtime/observability/execution.h"
#include "runtime/engine/request_record.h"
#include "runtime/contract/resources.h"

#include <cmath>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::runtime;
    int failures     = 0;
    const auto check = [&](bool value, const char* message) {
        if (!value) {
            ++failures;
            std::cerr << message << '\n';
        }
    };
    ExecutionClock clock;
    clock.start({ExecutionKind::Decode, 2, 2, 8}, 100);
    clock.result(160);
    clock.resolve(180);
    const auto sample = clock.finish(210);
    check(sample.latency.result_ns == 60 && sample.latency.resolution_ns == 20 &&
              sample.latency.retirement_ns == 30 && sample.latency.total_ns == 110,
          "accepted/result/resolve/retire boundaries must partition one execution");
    ExecutionStatistics global{};
    accumulate(global, sample);
    RequestExecutionStatistics a{}, b{};
    for (auto* row : {&a, &b}) (*row)[1].observe(sample.latency);
    check(global[1].units == 1 && global[1].rows == 2 && global[1].scheduled_positions == 8 &&
              global[1].exposure.total.total_ns == 110 && a[1].total.total_ns == 110 &&
              b[1].total.total_ns == 110,
          "batch work is counted once, each member observes its full latency");
    clock.start({ExecutionKind::Context}, 300);
    clock.result(320);
    const auto context = clock.finish(330);
    check(context.latency.result_ns == 20 && !context.latency.resolution_ns &&
              !context.latency.retirement_ns && context.latency.total_ns == 30,
          "automatic retirement must not fabricate a commit interval");
    clock.start({ExecutionKind::Decode}, 400);
    clock.resolve(410);
    const auto cancelled = clock.finish(450);
    check(!cancelled.latency.result_ns && !cancelled.latency.resolution_ns &&
              cancelled.latency.retirement_ns == 40 && cancelled.latency.total_ns == 50,
          "discard before result preserves actual drain time");
    RequestHostTiming request;
    request.expose_engine(RequestEngineHostPhase::Boundary, 20, true);
    request.expose_call(CoordinatorCall::Session, 10, true);
    request.expose_call(CoordinatorCall::Poll, 5, true);
    request.expose_call(CoordinatorCall::Yield, 30, true);
    check(request.decode_host_exposed_ns == 35 && request.decode_scheduler_yield_exposed_ns == 30,
          "yield/relock latency is not host-active time or GPU time");
    GpuExecutionObservation gpu;
    merge_gpu_intervals(gpu, {.result_stream_ns = 20, .prefill_stream_ns = 50});
    merge_gpu_intervals(gpu, {.result_stream_ns = 30});
    std::array<DurationCounter, 4> measured{};
    accumulate_gpu(measured, gpu);
    check(measured[0].count == 1 && measured[0].total_ns == 30 && measured[1].count == 0 &&
              measured[2].total_ns == 50,
          "rank max is per execution; unavailable GPU intervals stay unmeasured");
    GenerationTransferTiming peer_timing;
    ContextTransferObservation peer{.direction   = ContextTransferDirection::PeerDeviceToDevice,
                                    .image_count = 1,
                                    .work        = {4096, 2},
                                    .elapsed_ns  = 7000,
                                    .rank        = 0,
                                    .peer        = ContextPeerTransfer{0, 1}};
    accumulate_context_transfer(peer_timing, peer);
    peer.rank       = 1;
    peer.elapsed_ns = 9000;
    accumulate_context_transfer(peer_timing, peer);
    check(peer_timing.bytes == 4096 && peer_timing.images == 1 &&
              peer_timing.copy_operations == 2 && std::abs(peer_timing.seconds - 0.000016) < 1e-12,
          "peer payload must count once while both endpoint intervals remain observable");
    return failures ? 1 : 0;
}
