#include "ops/suite.h"
#include "ops/state_fixture.h"
#include "ninfer/ops/gated_delta_net.h"
#include "cuda_fixture.h"
#include "core/buffer.h"

#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/recurrence/launch.h"

#include "ops/op_tester.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kStateDim    = 128;
constexpr std::uint16_t kBf16Poison = 0xffffU;
constexpr std::uint32_t kFp32Poison = 0xffffffffU;

std::vector<std::uint16_t> make_bf16(std::size_t count, std::uint32_t seed) {
    std::vector<float> values(count);
    fill_uniform(values, seed, -0.08F, 0.08F);
    round_to_bf16(values);
    std::vector<std::uint16_t> bits(count);
    for (std::size_t index = 0; index < count; ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

int verify_equal(const std::string& label, const std::vector<std::uint16_t>& lhs,
                 const std::vector<std::uint16_t>& rhs) {
    if (lhs == rhs) { return 0; }
    std::cerr << label << ": BF16 bits differ\n";
    return 1;
}

int run_case(std::int32_t value_heads, std::int32_t width, std::int32_t batch,
             std::vector<std::int32_t> valid_columns, std::uint32_t seed) {
    constexpr std::int32_t kQkHeads = 16;
    const bool dense                = valid_columns.empty();
    if (dense) { valid_columns.assign(static_cast<std::size_t>(batch), width); }
    const std::int32_t columns       = width * batch;
    const std::int32_t slots         = 8;
    const std::size_t qk_elements    = static_cast<std::size_t>(kStateDim) * kQkHeads * columns;
    const std::size_t value_elements = static_cast<std::size_t>(kStateDim) * value_heads * columns;
    const std::size_t gate_elements  = static_cast<std::size_t>(value_heads) * columns;
    const std::size_t state_elements =
        static_cast<std::size_t>(kStateDim) * kStateDim * value_heads * slots;

    std::vector<std::uint16_t> q_bits = make_bf16(qk_elements, seed);
    std::vector<std::uint16_t> k_bits = make_bf16(qk_elements, seed + 1);
    std::vector<std::uint16_t> v_bits = make_bf16(value_elements, seed + 2);
    std::vector<float> g(gate_elements);
    std::vector<float> beta(gate_elements);
    fill_uniform(g, seed + 3, -1.2F, -0.02F);
    fill_uniform(beta, seed + 4, 0.02F, 0.98F);
    k_bits[0] = 0x8000U;
    k_bits[1] = 0x0001U;
    v_bits[0] = 0x8000U;
    v_bits[1] = 0x0001U;
    g[0]      = std::bit_cast<float>(0x80000000U);
    beta[0]   = std::bit_cast<float>(0x00000001U);
    std::vector<float> state(state_elements);
    fill_uniform(state, seed + 5, -0.03F, 0.03F);
    state[0] = std::bit_cast<float>(0x80000000U);

    std::vector<std::int32_t> initial_slots(static_cast<std::size_t>(batch));
    for (std::int32_t row = 0; row < batch; ++row) {
        initial_slots[static_cast<std::size_t>(row)] = row == 7 ? 7 : (row * 3 + 7) % slots;
    }

    DeviceBuffer device_q        = to_device(q_bits);
    DeviceBuffer device_k        = to_device(k_bits);
    DeviceBuffer device_v        = to_device(v_bits);
    DeviceBuffer device_g        = to_device(g);
    DeviceBuffer device_beta     = to_device(beta);
    DeviceBuffer reference_state = to_device(state);
    DeviceBuffer reference_final(ninfer::test::cuda_environment().context.device(),
                                 static_cast<std::size_t>(kStateDim) * kStateDim * value_heads *
                                     batch * sizeof(float));
    DeviceBuffer record_state   = to_device(state);
    DeviceBuffer device_initial = to_device(initial_slots);
    DeviceBuffer device_valid;
    if (!dense) { device_valid = to_device(valid_columns); }

    DeviceBuffer reference_out(ninfer::test::cuda_environment().context.device(),
                               value_elements * sizeof(std::uint16_t));
    DeviceBuffer record_out(ninfer::test::cuda_environment().context.device(),
                            value_elements * sizeof(std::uint16_t));
    DeviceBuffer key_record(ninfer::test::cuda_environment().context.device(),
                            qk_elements * sizeof(std::uint16_t));
    DeviceBuffer value_record(ninfer::test::cuda_environment().context.device(),
                              value_elements * sizeof(std::uint16_t));
    DeviceBuffer gate_record(ninfer::test::cuda_environment().context.device(),
                             gate_elements * 2 * sizeof(std::uint32_t));
    reference_out.fill(0);
    record_out.fill(0xff);
    key_record.fill(0xff);
    value_record.fill(0xff);
    gate_record.fill(0xff);

    Tensor q(device_q.span(), DType::BF16, {kStateDim, kQkHeads, width, batch});
    Tensor k(device_k.span(), DType::BF16, {kStateDim, kQkHeads, width, batch});
    Tensor v(device_v.span(), DType::BF16, {kStateDim, value_heads, width, batch});
    Tensor g_tensor(device_g.span(), DType::FP32, {value_heads, width, batch});
    Tensor beta_tensor(device_beta.span(), DType::FP32, {value_heads, width, batch});
    Tensor reference_states(reference_state.span(), DType::FP32,
                            {kStateDim, kStateDim, value_heads, slots});
    Tensor record_states(record_state.span(), DType::FP32,
                         {kStateDim, kStateDim, value_heads, slots});
    SlottedStateFixture history(record_states, 3);
    record_states = history.tensor;
    Tensor reference_final_states(reference_final.span(), DType::FP32,
                                  {kStateDim, kStateDim, value_heads, batch});
    Tensor valid;
    if (!dense) { valid = Tensor(device_valid.span(), DType::I32, {batch}); }
    Tensor initial(device_initial.span(), DType::I32, {batch});
    Tensor reference_output(reference_out.span(), DType::BF16,
                            {kStateDim, value_heads, width, batch});
    Tensor record_output(record_out.span(), DType::BF16, {kStateDim, value_heads, width, batch});
    Tensor key_record_tensor(key_record.span(), DType::BF16, {kStateDim, kQkHeads, width, batch});
    Tensor value_record_tensor(value_record.span(), DType::BF16,
                               {kStateDim, value_heads, width, batch});
    Tensor gate_record_tensor(gate_record.span(), DType::FP32, {2, value_heads, width, batch});

    constexpr float kScale = 1.0F / std::sqrt(128.0F);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const auto stream_execution = ninfer::test::cuda_environment().context.execution_view(stream);
    cuda_synchronize();
    // Replay is bitwise equivalent to recurrence, independently of the prefill dispatcher.
    const auto launch_reference = [&] {
        CUDA_CHECK(cudaMemsetAsync(reference_out.data(), 0, reference_out.size(), stream));
        for (std::int32_t row = 0; row < batch; ++row) {
            const std::int32_t valid_extent = valid_columns[static_cast<std::size_t>(row)];
            Tensor q_row                    = q.slice(3, row, 1)
                               .slice(2, 0, valid_extent)
                               .view({kStateDim, kQkHeads, valid_extent});
            Tensor k_row = k.slice(3, row, 1)
                               .slice(2, 0, valid_extent)
                               .view({kStateDim, kQkHeads, valid_extent});
            Tensor v_row = v.slice(3, row, 1)
                               .slice(2, 0, valid_extent)
                               .view({kStateDim, value_heads, valid_extent});
            Tensor g_row = g_tensor.slice(2, row, 1)
                               .slice(1, 0, valid_extent)
                               .view({value_heads, valid_extent});
            Tensor beta_row = beta_tensor.slice(2, row, 1)
                                  .slice(1, 0, valid_extent)
                                  .view({value_heads, valid_extent});
            Tensor state_row =
                reference_states.slice(3, initial_slots[static_cast<std::size_t>(row)], 1)
                    .view({kStateDim, kStateDim, value_heads});
            Tensor out_row = reference_output.slice(3, row, 1)
                                 .slice(2, 0, valid_extent)
                                 .view({kStateDim, value_heads, valid_extent});
            Tensor final_row =
                reference_final_states.slice(3, row, 1).view({kStateDim, kStateDim, value_heads});
            ops::detail::gated_delta_net::launch_recurrent_inout(q_row, k_row, v_row, g_row,
                                                                 beta_row, kScale, true, state_row,
                                                                 final_row, out_row, stream);
        }
    };
    const auto launch_record = [&] {
        ops::gated_delta_net_replay_record(
            q, k, v, g_tensor, beta_tensor, kScale, record_states, valid, initial,
            {key_record_tensor, value_record_tensor, gate_record_tensor}, record_output,
            stream_execution);
    };
    launch_reference();
    launch_record();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (width == 2 || width == 9 || width == 16) {
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch_record();
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (auto& bits : q_bits) bits ^= 0x8000U;
        CUDA_CHECK(cudaMemcpyAsync(device_q.data(), q_bits.data(), device_q.size(),
                                   cudaMemcpyHostToDevice, stream));
        if (!dense) {
            for (auto& count : valid_columns) count = 1 + count % width;
            CUDA_CHECK(cudaMemcpyAsync(device_valid.data(), valid_columns.data(),
                                       device_valid.size(), cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaMemsetAsync(key_record.data(), 0xff, key_record.size(), stream));
        CUDA_CHECK(cudaMemsetAsync(value_record.data(), 0xff, value_record.size(), stream));
        CUDA_CHECK(cudaMemsetAsync(gate_record.data(), 0xff, gate_record.size(), stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        launch_reference();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));

    int failures             = 0;
    const std::string suffix = " Hv=" + std::to_string(value_heads) +
                               " T=" + std::to_string(width) + " B=" + std::to_string(batch);
    const std::vector<std::uint16_t> reference_output_bits =
        from_device<std::uint16_t>(reference_out, value_elements);
    const std::vector<std::uint16_t> record_output_bits =
        from_device<std::uint16_t>(record_out, value_elements);
    failures +=
        verify_equal("replay record output" + suffix, reference_output_bits, record_output_bits);

    const std::vector<std::uint16_t> key_bits_after =
        from_device<std::uint16_t>(key_record, qk_elements);
    const std::vector<std::uint16_t> value_bits_after =
        from_device<std::uint16_t>(value_record, value_elements);
    const std::vector<std::uint32_t> gate_bits_after =
        from_device<std::uint32_t>(gate_record, gate_elements * 2);
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t valid_extent = valid_columns[static_cast<std::size_t>(row)];
        for (std::int32_t token = 0; token < width; ++token) {
            const std::int64_t column = static_cast<std::int64_t>(row) * width + token;
            const bool active         = token < valid_extent;
            for (std::int32_t head = 0; head < kQkHeads; ++head) {
                const std::size_t base =
                    static_cast<std::size_t>((column * kQkHeads + head) * kStateDim);
                for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                    const std::uint16_t expected = active ? k_bits[base + dim] : kBf16Poison;
                    if (key_bits_after[base + dim] != expected) {
                        std::cerr << "key record mismatch" << suffix << "\n";
                        return failures + 1;
                    }
                }
            }
            for (std::int32_t head = 0; head < value_heads; ++head) {
                const std::size_t vector_base =
                    static_cast<std::size_t>((column * value_heads + head) * kStateDim);
                for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                    const std::uint16_t expected = active ? v_bits[vector_base + dim] : kBf16Poison;
                    if (value_bits_after[vector_base + dim] != expected) {
                        std::cerr << "value record mismatch" << suffix << "\n";
                        return failures + 1;
                    }
                }
                const std::size_t gate_offset =
                    static_cast<std::size_t>((column * value_heads + head) * 2);
                const std::size_t source_offset =
                    static_cast<std::size_t>(column * value_heads + head);
                const std::uint32_t expected_g =
                    active ? std::bit_cast<std::uint32_t>(g[source_offset]) : kFp32Poison;
                const std::uint32_t expected_beta =
                    active ? std::bit_cast<std::uint32_t>(beta[source_offset]) : kFp32Poison;
                if (gate_bits_after[gate_offset] != expected_g ||
                    gate_bits_after[gate_offset + 1] != expected_beta) {
                    std::cerr << "gate record mismatch" << suffix << "\n";
                    return failures + 1;
                }
            }
            if (!active) {
                const std::size_t output_base =
                    static_cast<std::size_t>(column) * value_heads * kStateDim;
                for (std::int32_t index = 0; index < value_heads * kStateDim; ++index) {
                    if (record_output_bits[output_base + index] != 0) {
                        std::cerr << "record invalid output is not zero" << suffix << "\n";
                        return failures + 1;
                    }
                }
            }
        }
    }

    history.copy_back();
    failures += history.verify_guards("record source");
    const std::vector<float> state_after = from_device<float>(record_state, state_elements);
    if (std::memcmp(state_after.data(), state.data(), state_elements * sizeof(float)) != 0) {
        std::cerr << "replay record modified source state" << suffix << "\n";
        ++failures;
    }
    if (from_device<std::uint16_t>(device_q, qk_elements) != q_bits ||
        from_device<std::uint16_t>(device_k, qk_elements) != k_bits ||
        from_device<std::uint16_t>(device_v, value_elements) != v_bits) {
        std::cerr << "replay record modified inputs" << suffix << "\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    const auto add = [&](int heads, int width, int batch, std::vector<int> valid, unsigned seed) {
        suite.add("gdn_record/h" + std::to_string(heads) + "/w" + std::to_string(width) + "/b" +
                      std::to_string(batch) + (valid.empty() ? "/dense" : "/ragged"),
                  [=] { return run_case(heads, width, batch, valid, seed); },
                  {{"execution", width == 2 || width == 9 || width == 16 ? "graph" : "eager"}});
    };
    add(32, 2, 1, {}, 1701U);
    add(32, 16, 1, {7}, 1711U);
    add(32, 6, 8, {6, 5, 4, 3, 2, 1, 6, 2}, 1721U);
    for (int width = 2; width <= 16; ++width) {
        add(48, width, 1, {}, 1730U + width);
        std::vector<std::int32_t> valid(8);
        for (int b = 0; b < 8; ++b) valid[b] = b == 0 ? width : 1 + (3 * b) % width;
        add(48, width, 8, valid, 1760U + width);
    }
    add(48, 5, 3, {5, 3, 1}, 1791U);

    return suite.main(argc, argv);
}
