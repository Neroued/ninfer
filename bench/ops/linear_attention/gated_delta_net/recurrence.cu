// Public GDN and production-stage timing. Private launchers also measure the prefill
// recurrent/chunked crossover, independently of decode/ReplaySSM.
// Graph/eager timing and cache conditioning use the shared Op measurement harness.
#include "ops/linear_attention/gated_delta_net/recurrence/plan.h"
#include "core/buffer.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/l2norm.h"
#include "op_measurement.cuh"
#include "recurrence_fixture.cuh"
#include "ops/linear_attention/gated_delta_net/recurrence/chunked/launch.h"
#include "ops/linear_attention/gated_delta_net/recurrence/launch.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <optional>

namespace ninfer::bench::gdn_recurrence {
namespace gated_delta_net_detail = ninfer::ops::detail::gated_delta_net;
namespace chunked_detail         = ninfer::ops::detail::gated_delta_net::chunked;

struct BenchRow {
    const char* normalization = "";
    std::string implementation;
    std::int32_t full_chunks    = 0;
    std::int32_t tail_tokens    = 0;
    std::int32_t value_tile     = 0;
    std::size_t workspace_bytes = 0;
    gdn::Traffic traffic;
    measurement::Result measured;
    std::size_t workspace_peak = 0;
};

template <class Launch>
measurement::Result measure_operation(
    Launch& launch, L2FlushBuffer& flush, CudaExecutionView cuda, const Options& options,
    const launch_fn& restore = [](CudaExecutionView) {}) {
    return measurement::measure(launch, restore, flush, cuda, options.timing,
                                options.timing.execution, options.timing.cache);
}

float gated_delta_net_scale() { return 1.0F / std::sqrt(128.0F); }

std::uint64_t packet_bytes(const Problem& p) {
    return std::uint64_t(chunked_detail::chunk_count(p.tokens)) *
           (p.qk_heads * sizeof(chunked_detail::QkChunk) +
            p.value_heads * sizeof(chunked_detail::ControlChunk));
}

gdn::Traffic prepare_traffic(const Problem& p) {
    gdn::Traffic result;
    result.logical                  = {2 * p.qk_bytes() + 2 * p.gate_bytes(), packet_bytes(p)};
    result.workspace_requests.write = packet_bytes(p);
    result.modeled_request_bytes    = 2 * p.qk_bytes() * (p.value_heads / p.qk_heads) +
                                   2 * p.gate_bytes() + result.workspace_requests.write;
    result.active_rows  = p.batch;
    result.valid_tokens = p.tokens * p.batch;
    return result;
}

gdn::Traffic recurrence_traffic(const Problem& p, int slices) {
    gdn::Traffic result;
    result.state = {p.state_bytes(), p.state_bytes()};
    result.workspace_requests.read =
        std::uint64_t(chunked_detail::chunk_count(p.tokens)) * p.value_heads * slices *
        (sizeof(chunked_detail::QkChunk) + sizeof(chunked_detail::ControlChunk));
    result.logical = {p.value_bytes() + result.state.read + packet_bytes(p),
                      p.value_bytes() + result.state.write};
    result.modeled_request_bytes =
        result.workspace_requests.read + 2 * p.value_bytes() + result.state.total();
    result.active_rows  = p.batch;
    result.valid_tokens = p.tokens * p.batch;
    return result;
}

std::vector<BenchRow> run_batch_update(const Options& options, std::int32_t tokens,
                                       bench::L2FlushBuffer& flush, CudaExecutionView cuda) {
    const Problem problem{options.qk_heads, options.value_heads, tokens, options.batch};
    Operands operands(problem, false);

    const std::size_t qk_elements = static_cast<std::size_t>(gated_delta_net_detail::kStateDim) *
                                    problem.qk_heads * tokens * problem.batch;
    const std::size_t state_elements = static_cast<std::size_t>(gated_delta_net_detail::kStateDim) *
                                       gated_delta_net_detail::kStateDim * problem.value_heads;
    const std::int32_t records = 2 * problem.batch;
    DeviceBuffer states        = bench::make_f32(state_elements * records, 211U, -.02F, .02F);
    SavedBuffer initial_states(states);
    const auto restore = [&](CudaExecutionView cuda) { initial_states.restore(cuda); };
    std::vector<std::int64_t> source_offsets_host(static_cast<std::size_t>(problem.batch));
    std::vector<std::int64_t> destination_offsets_host(static_cast<std::size_t>(problem.batch));
    for (std::int32_t row = 0; row < problem.batch; ++row) {
        source_offsets_host[static_cast<std::size_t>(row)] =
            2 * row * state_elements * sizeof(float);
        destination_offsets_host[static_cast<std::size_t>(row)] =
            (2 * row + 1) * state_elements * sizeof(float);
    }
    DeviceBuffer source_offsets(ninfer::bench::device_context().device(),
                                source_offsets_host.size() * sizeof(std::int64_t));
    source_offsets.copy_from_host(source_offsets_host.data(), source_offsets.size());
    DeviceBuffer destination_offsets(ninfer::bench::device_context().device(),
                                     destination_offsets_host.size() * sizeof(std::int64_t));
    destination_offsets.copy_from_host(destination_offsets_host.data(), destination_offsets.size());
    DeviceBuffer q_normalized;
    DeviceBuffer k_normalized;
    if (options.qk_norm == "composed") {
        q_normalized = make_zeros(qk_elements * sizeof(std::uint16_t));
        k_normalized = make_zeros(qk_elements * sizeof(std::uint16_t));
    }

    Tensor q    = operands.query();
    Tensor k    = operands.key();
    Tensor v    = operands.value();
    Tensor g    = operands.gate();
    Tensor beta = operands.beta_tensor();
    Tensor out  = operands.output();
    Tensor source_table(source_offsets.span(), DType::I64, {problem.batch});
    Tensor destinations = options.state_form == "alias"
                              ? source_table
                              : Tensor(destination_offsets.span(), DType::I64, {problem.batch});
    TensorRegion state_region{
        {0, state_elements * sizeof(float), 16}, DType::FP32, {128, 128, problem.value_heads, 1}};
    StateFieldView source_state{{states.span(), source_table},
                                StateFieldLayout{state_region, 1, 0}};
    StateFieldView destination_state{{states.span(), destinations},
                                     StateFieldLayout{state_region, 1, 0}};
    Tensor q_norm;
    Tensor k_norm;
    if (options.qk_norm == "composed") {
        q_norm =
            Tensor(q_normalized.span(), DType::BF16,
                   {gated_delta_net_detail::kStateDim, problem.qk_heads, tokens, problem.batch});
        k_norm =
            Tensor(k_normalized.span(), DType::BF16,
                   {gated_delta_net_detail::kStateDim, problem.qk_heads, tokens, problem.batch});
    }

    const bool composed = options.qk_norm == "composed";
    auto launch         = [&](CudaExecutionView cuda) {
        if (composed) {
            ops::l2norm(q, kQkNormEpsilon, q_norm, cuda);
            ops::l2norm(k, kQkNormEpsilon, k_norm, cuda);
        }
        const Tensor& q_input = composed ? q_norm : q;
        const Tensor& k_input = composed ? k_norm : k;
        ops::gated_delta_net_batch_update(q_input, k_input, v, g, beta, gated_delta_net_scale(),
                                                  !composed, source_state, destination_state, out, cuda);
    };
    BenchRow row;
    row.normalization  = composed ? "composed" : "fused";
    row.implementation = composed ? "l2norm_x2+public.batch_update.qk_pre_normalized"
                                  : "public.batch_update.qk_fused";
    row.traffic        = gdn::running_traffic(problem);
    // Alias shares the read-only selector tensor; distinct form uses two arrays.
    row.traffic.logical.read +=
        sizeof(std::int64_t) * problem.batch * (options.state_form == "alias" ? 1 : 2);
    if (composed) {
        row.workspace_bytes = row.workspace_peak = 2 * problem.qk_bytes();
        row.traffic.workspace_requests           = {row.workspace_bytes, row.workspace_bytes};
    }
    row.measured = measure_operation(launch, flush, cuda, options, restore);
    return {row};
}

std::vector<BenchRow> run_prefill(const Options& options, std::int32_t tokens,
                                  bench::L2FlushBuffer& flush, CudaExecutionView execution) {
    constexpr auto state_dim = gated_delta_net_detail::kStateDim;
    const bool normalize     = options.mode != Mode::ChunkedNormalized;
    const bool force_chunked =
        options.mode == Mode::ChunkedRaw || options.mode == Mode::ChunkedNormalized;
    const bool force_recurrent = options.mode == Mode::Recurrent;
    const bool chunked =
        force_chunked || (!force_recurrent && gated_delta_net_detail::select_recurrence_route(
                                                  options.qk_heads, options.value_heads, tokens) ==
                                                  gated_delta_net_detail::RecurrenceRoute::Chunked);
    const Problem problem{options.qk_heads, options.value_heads, tokens};
    Operands operands(problem, !normalize);
    const std::size_t state_elements = std::size_t(state_dim) * state_dim * problem.value_heads;
    DeviceBuffer state_in            = bench::make_f32(state_elements, 211U, -.02F, .02F);
    DeviceBuffer state_out           = make_zeros(state_elements * sizeof(float));
    Tensor q = operands.query(), k = operands.key(), v = operands.value();
    Tensor g = operands.gate(), beta = operands.beta_tensor(), out = operands.output();
    Tensor si(state_in.span(), DType::FP32, {state_dim, state_dim, problem.value_heads});
    Tensor so(options.state_form == "alias" ? state_in.span() : state_out.span(), DType::FP32,
              {state_dim, state_dim, problem.value_heads});
    std::optional<SavedBuffer> saved;
    if (options.state_form == "alias") saved.emplace(state_in);
    const launch_fn restore = [&](CudaExecutionView cuda) {
        if (saved) saved->restore(cuda);
    };
    const auto layout =
        chunked_detail::workspace_layout(problem.qk_heads, problem.value_heads, tokens);
    std::size_t bytes = 0;
    if (force_chunked) {
        bytes = layout.total_bytes;
    } else if (!force_recurrent) {
        bytes = ops::gated_delta_net_workspace_capacity_bytes(problem.qk_heads, problem.value_heads,
                                                              tokens, tokens);
    }
    ninfer::DeviceBuffer workspace_storage(ninfer::bench::device_context().device(),
                                           std::max<std::size_t>(256, bytes));
    MemoryArena workspace(workspace_storage.span());
    const auto backing = workspace.storage().subspan(0, bytes);
    auto* qk =
        chunked ? static_cast<chunked_detail::QkChunk*>(layout.qk.bind(backing).data) : nullptr;
    auto* control =
        chunked ? static_cast<chunked_detail::ControlChunk*>(layout.control.bind(backing).data)
                : nullptr;
    const gated_delta_net_detail::RecurrenceOperands args{static_cast<const __nv_bfloat16*>(q.data),
                                                          static_cast<const __nv_bfloat16*>(k.data),
                                                          static_cast<const __nv_bfloat16*>(v.data),
                                                          static_cast<const float*>(g.data),
                                                          static_cast<const float*>(beta.data),
                                                          static_cast<const float*>(si.data),
                                                          static_cast<float*>(so.data),
                                                          static_cast<__nv_bfloat16*>(out.data),
                                                          problem.qk_heads,
                                                          problem.value_heads,
                                                          tokens,
                                                          gated_delta_net_scale()};
    const int tile     = chunked ? ops::detail::gated_delta_net::select_value_tile(
                                   problem.qk_heads, problem.value_heads, tokens,
                                   execution.multiprocessor_count())
                                 : 0;
    const auto prep_io = prepare_traffic(problem);
    const auto rec_io  = recurrence_traffic(problem, tile ? state_dim / tile : 0);
    auto total_io      = gdn::running_traffic(problem);
    if (chunked) {
        total_io.workspace_requests = {rec_io.workspace_requests.read,
                                       prep_io.workspace_requests.write};
        total_io.modeled_request_bytes =
            *prep_io.modeled_request_bytes + *rec_io.modeled_request_bytes;
    }
    auto prepare = [&](CudaExecutionView cuda) {
        const auto stream = cuda.stream();
        chunked_detail::launch_prepare(args, qk, control, normalize, stream);
    };
    auto recurrence = [&](CudaExecutionView cuda) {
        const auto stream = cuda.stream();
        chunked_detail::launch_recurrence(args, qk, control, tile, stream);
    };
    auto pipeline = [&](CudaExecutionView cuda) {
        const auto stream = cuda.stream();
        if (force_chunked) {
            prepare(cuda);
            recurrence(cuda);
        } else if (force_recurrent)
            gated_delta_net_detail::launch_recurrent_inout(
                q, k, v, g, beta, gated_delta_net_scale(), normalize, si, so, out, stream);
        else
            ops::gated_delta_net(q, k, v, g, beta, gated_delta_net_scale(), normalize, workspace,
                                 si, so, out, cuda);
    };
    std::vector<BenchRow> rows;
    auto measure = [&](const char* name, gdn::Traffic io, auto& body) {
        BenchRow row;
        row.normalization   = normalize ? "fused" : "pre_normalized";
        row.implementation  = name;
        row.full_chunks     = chunked ? tokens / chunked_detail::kChunkSize : 0;
        row.tail_tokens     = chunked ? tokens % chunked_detail::kChunkSize : 0;
        row.value_tile      = tile;
        row.workspace_bytes = bytes;
        row.traffic         = io;
        row.measured        = measure_operation(body, flush, execution, options, restore);
        row.workspace_peak  = force_chunked ? bytes : workspace.peak_used();
        rows.push_back(row);
    };
    measure(force_chunked     ? "forced.chunked"
            : force_recurrent ? "forced.recurrent"
                              : "public.total",
            total_io, pipeline);
    if (options.breakdown && chunked) {
        measure("chunked.prepare", prep_io, prepare);
        measure("chunked.recurrence", rec_io, recurrence);
    }
    return rows;
}

} // namespace ninfer::bench::gdn_recurrence

int main(int argc, char** argv) {
    using namespace ninfer;
    using namespace ninfer::bench;
    using namespace ninfer::bench::gdn_recurrence;
    try {
        const auto requested = parse_options(argc, argv);
        measurement::hardware();
        auto& device = ninfer::bench::device_context();
        L2FlushBuffer flush(requested.timing.flush_bytes);
        measurement::Csv csv(
            requested.timing.csv_out,
            std::string("normalization,full_chunks,tail_tokens,value_tile,state_form,") +
                gdn::Traffic::columns);
        for (const auto& execution :
             measurement::selections(requested.timing.execution, "graph", "eager"))
            for (const auto& cache :
                 measurement::selections(requested.timing.cache, "cold", "warm")) {
                auto options             = requested;
                options.timing.execution = execution;
                options.timing.cache     = cache;
                for (const int t : options.tokens) {
                    const auto rows =
                        options.mode == Mode::BatchUpdate
                            ? run_batch_update(options, t, flush, device.execution_view())
                            : run_prefill(options, t, flush, device.execution_view());
                    for (const auto& r : rows) {
                        measurement::Row row;
                        row.op              = r.implementation;
                        row.tokens          = t;
                        row.batch           = options.batch;
                        row.qk_heads        = options.qk_heads;
                        row.value_heads     = options.value_heads;
                        row.workspace_bytes = r.workspace_bytes;
                        row.workspace_peak  = r.workspace_peak;
                        const auto extra    = std::string(r.normalization) + "," +
                                           std::to_string(r.full_chunks) + "," +
                                           std::to_string(r.tail_tokens) + "," +
                                           std::to_string(r.value_tile) + "," + options.state_form +
                                           "," + r.traffic.csv(r.measured.gpu.median_us);
                        csv.write(row, r.measured, execution, cache, extra);
                    }
                }
            }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
