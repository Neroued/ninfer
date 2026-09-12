#include "runtime/contract/types.h"
#include "runtime/engine/request_record.h"

#include <cmath>
#include <cstdint>
#include <iostream>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

bool near(double lhs, double rhs) { return std::abs(lhs - rhs) <= 1.0e-20; }

} // namespace

int main() {
    int failures = 0;

    ninfer::runtime::ExecutionTiming sum{
        .submit_host_ns = 11, .device_wait_ns = 13, .post_host_ns = 17};
    sum += ninfer::runtime::ExecutionTiming{
        .submit_host_ns = 19,
        .device_wait_ns = 23,
        .post_host_ns   = 29,
    };
    failures += check(sum.submit_host_ns == 30 && sum.device_wait_ns == 36 &&
                          sum.post_host_ns == 46 && sum.host_ns() == 76 && sum.elapsed_ns() == 112,
                      "ExecutionTiming integer phase arithmetic is inconsistent");

    ninfer::runtime::ExecutionTimingRecorder aggregate(
        ninfer::runtime::ExecutionTimingPhase::Paused);
    aggregate.include(sum);
    const ninfer::runtime::ExecutionTiming observed = aggregate.finish();
    failures += check(observed.submit_host_ns == sum.submit_host_ns &&
                          observed.device_wait_ns == sum.device_wait_ns &&
                          observed.post_host_ns == sum.post_host_ns,
                      "paused timing composition changed an included child observation");

    ninfer::runtime::ExecutionTimingRecorder post_only(ninfer::runtime::ExecutionTimingPhase::Post);
    const ninfer::runtime::ExecutionTiming post_observed = post_only.finish();
    failures += check(post_observed.submit_host_ns == 0 && post_observed.device_wait_ns == 0,
                      "post-only recorder attributed initialization to another phase");

    ninfer::runtime::ExecutionTiming abandoned_observation;
    {
        ninfer::runtime::ExecutionTimingRecorder abandoned(
            ninfer::runtime::ExecutionTimingPhase::Paused, &abandoned_observation);
        abandoned.include(sum);
    }
    failures += check(abandoned_observation.submit_host_ns == sum.submit_host_ns &&
                          abandoned_observation.device_wait_ns == sum.device_wait_ns &&
                          abandoned_observation.post_host_ns == sum.post_host_ns,
                      "abandoned Program timing was lost during exception unwinding");

    ninfer::runtime::RequestHostTiming request{
        .queue_wait_ns                   = 1,
        .engine_boundary_exposed_ns      = 2,
        .program_submit_exposed_ns       = 3,
        .program_post_exposed_ns         = 5,
        .engine_commit_output_exposed_ns = 7,
        .engine_maintenance_exposed_ns   = 11,
        .device_wait_exposed_ns          = 13,
        .decode_host_exposed_ns          = 17,
        .decode_device_wait_exposed_ns   = 19,
        .prefill_units                   = 23,
        .decode_rounds                   = 29,
        .control_units                   = 31,
    };
    const ninfer::GenerationEngineTiming public_timing = request.public_snapshot();
    failures += check(near(public_timing.queue_wait_seconds, 1.0e-9) &&
                          near(public_timing.engine_boundary_exposed_seconds, 2.0e-9) &&
                          near(public_timing.program_submit_exposed_seconds, 3.0e-9) &&
                          near(public_timing.program_post_exposed_seconds, 5.0e-9) &&
                          near(public_timing.engine_commit_output_exposed_seconds, 7.0e-9) &&
                          near(public_timing.engine_maintenance_exposed_seconds, 11.0e-9) &&
                          near(public_timing.device_wait_exposed_seconds, 13.0e-9) &&
                          near(public_timing.decode_host_exposed_seconds, 17.0e-9) &&
                          near(public_timing.decode_device_wait_exposed_seconds, 19.0e-9) &&
                          public_timing.prefill_units == 23 && public_timing.decode_rounds == 29 &&
                          public_timing.control_units == 31,
                      "request timing publication changed integer observations");

    ninfer::runtime::RequestHostTiming batch_row_a;
    ninfer::runtime::RequestHostTiming batch_row_b;
    const ninfer::runtime::ExecutionTiming shared_program{
        .submit_host_ns = 10,
        .device_wait_ns = 20,
        .post_host_ns   = 30,
    };
    for (ninfer::runtime::RequestHostTiming* row : {&batch_row_a, &batch_row_b}) {
        row->expose_engine(ninfer::runtime::RequestEngineHostPhase::Boundary, 100, true);
        row->expose_program(shared_program, true);
    }
    failures += check(batch_row_a.engine_boundary_exposed_ns == 100 &&
                          batch_row_b.engine_boundary_exposed_ns == 100 &&
                          batch_row_a.program_submit_exposed_ns == 10 &&
                          batch_row_b.program_submit_exposed_ns == 10 &&
                          batch_row_a.decode_host_exposed_ns == 140 &&
                          batch_row_b.decode_host_exposed_ns == 140 &&
                          batch_row_a.device_wait_exposed_ns == 20 &&
                          batch_row_b.device_wait_exposed_ns == 20,
                      "compact-batch rows did not each receive full elapsed exposure");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
