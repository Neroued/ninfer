#include "gdn_ref.h"
#include "ops/suite.h"
#include "ops/state_fixture.h"
#include "ninfer/ops/gated_delta_net.h"
#include "cuda_fixture.h"
#include "core/buffer.h"

#include "core/device.h"

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

int run_case(std::int32_t value_heads, std::int32_t width, std::int32_t batch,
             std::vector<std::int32_t> valid_columns, std::uint32_t seed,
             bool minimum_alignment = false, DType state_dtype = DType::FP32) {
    const std::int32_t kQkHeads = value_heads == 24 ? 8 : 16;
    const bool dense            = valid_columns.empty();
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
    if (state_dtype == DType::BF16) round_to_bf16(state);
    state[0] = std::bit_cast<float>(0x80000000U);

    std::vector<std::int32_t> initial_slots(static_cast<std::size_t>(batch));
    for (std::int32_t row = 0; row < batch; ++row) {
        initial_slots[static_cast<std::size_t>(row)] = row == 7 ? 7 : (row * 3 + 7) % slots;
    }

    const std::size_t qk_offset = minimum_alignment ? 8 : 0;
    const std::size_t v_offset  = minimum_alignment ? 2 : 0;
    const auto upload           = [&](const std::vector<std::uint16_t>& bits, std::size_t offset) {
        DeviceBuffer buffer(ninfer::test::cuda_environment().context.device(),
                                      offset + bits.size() * sizeof(std::uint16_t));
        CUDA_CHECK(cudaMemcpy(static_cast<std::byte*>(buffer.data()) + offset, bits.data(),
                                        bits.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
        return buffer;
    };
    DeviceBuffer device_q    = upload(q_bits, qk_offset);
    DeviceBuffer device_k    = upload(k_bits, qk_offset);
    DeviceBuffer device_v    = upload(v_bits, v_offset);
    DeviceBuffer device_g    = to_device(g);
    DeviceBuffer device_beta = to_device(beta);
    std::vector<std::uint16_t> state_bf16;
    if (state_dtype == DType::BF16) {
        state_bf16.resize(state_elements);
        std::transform(state.begin(), state.end(), state_bf16.begin(), f32_to_bf16);
    }
    DeviceBuffer record_state =
        state_dtype == DType::BF16 ? to_device(state_bf16) : to_device(state);
    DeviceBuffer device_valid;
    if (!dense) { device_valid = to_device(valid_columns); }

    DeviceBuffer record_out(cuda_environment().context.device(),
                            value_elements * sizeof(std::uint16_t));
    LayoutBuilder record_builder;
    const auto record_layout = plan_gdn_replay_records(
        record_builder, {1, batch, width, (2 * kQkHeads + value_heads) * kStateDim, kQkHeads,
                         value_heads, kStateDim, kStateDim});
    DeviceBuffer storage(cuda_environment().context.device(), record_builder.finish(256));
    const GdnReplayRecords records(storage.span(), record_layout);
    const auto layer = records.layer(0, batch);
    record_out.fill(0xff);
    storage.fill(0xff);

    Tensor q(device_q.span().subspan(qk_offset, qk_elements * 2), DType::BF16,
             {kStateDim, kQkHeads, width, batch});
    Tensor k(device_k.span().subspan(qk_offset, qk_elements * 2), DType::BF16,
             {kStateDim, kQkHeads, width, batch});
    Tensor v(device_v.span().subspan(v_offset, value_elements * 2), DType::BF16,
             {kStateDim, value_heads, width, batch});
    Tensor g_tensor(device_g.span(), DType::FP32, {value_heads, width, batch});
    Tensor beta_tensor(device_beta.span(), DType::FP32, {value_heads, width, batch});
    Tensor record_states(record_state.span(), state_dtype,
                         {kStateDim, kStateDim, value_heads, slots});
    RecordStateFixture history(record_states, 3);
    Tensor valid;
    if (!dense) { valid = Tensor(device_valid.span(), DType::I32, {batch}); }
    auto initial_offsets = history.record_offsets(initial_slots);
    for (int row = 0; row < batch; ++row)
        if (!valid_columns[row]) initial_offsets[row] = -1;
    DeviceBuffer device_initial = to_device(initial_offsets);
    Tensor initial(device_initial.span(), DType::I64, {batch});
    Tensor record_output(record_out.span(), DType::BF16, {kStateDim, value_heads, width, batch});

    constexpr float kScale = 1.0F / std::sqrt(128.0F);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const auto stream_execution = ninfer::test::cuda_environment().context.execution_view(stream);
    cuda_synchronize();
    const auto launch_record = [&] {
        ops::gated_delta_net_replay_record(q, k, v, g_tensor, beta_tensor, kScale,
                                           history.view(initial), valid, layer.recurrence,
                                           record_output, stream_execution);
    };
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
        CUDA_CHECK(
            cudaMemcpyAsync(q.data, q_bits.data(), q.bytes(), cudaMemcpyHostToDevice, stream));
        for (auto& slot : initial_slots) slot = (slot + 3) % slots;
        if (!dense) {
            for (auto& count : valid_columns) count = (count + 1) % (width + 1);
            CUDA_CHECK(cudaMemcpyAsync(device_valid.data(), valid_columns.data(),
                                       device_valid.size(), cudaMemcpyHostToDevice, stream));
        }
        initial_offsets = history.record_offsets(initial_slots);
        for (int row = 0; row < batch; ++row)
            if (!valid_columns[row]) initial_offsets[row] = -1;
        CUDA_CHECK(cudaMemcpyAsync(device_initial.data(), initial_offsets.data(),
                                   device_initial.size(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(storage.data(), 0xff, storage.size(), stream));
        CUDA_CHECK(cudaMemsetAsync(record_out.data(), 0xff, record_out.size(), stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));

    int failures             = 0;
    const std::string suffix = " Hv=" + std::to_string(value_heads) +
                               " T=" + std::to_string(width) + " B=" + std::to_string(batch);
    const auto record_output_bits = from_device<std::uint16_t>(record_out, value_elements);
    const auto [key_record, delta_record, alpha_record] =
        ninfer::detail::GdnReplayRecordAccess::planes(layer.recurrence);
    const auto key_after   = from_device<std::uint16_t>(key_record.data, qk_elements);
    const auto delta_after = from_device<std::uint32_t>(delta_record.data, value_elements);
    const auto alpha_after = from_device<std::uint32_t>(alpha_record.data, gate_elements);
    const auto conv_after  = from_device<std::uint16_t>(layer.conv.data, layer.conv.numel());
    const auto all_equal = [](const auto& values, std::size_t first, std::size_t count, auto word) {
        return std::all_of(values.begin() + first, values.begin() + first + count,
                           [word](auto value) { return value == word; });
    };
    if (!all_equal(conv_after, 0, conv_after.size(), kBf16Poison)) {
        std::cerr << "recurrent record modified convolution plane" << suffix << '\n';
        ++failures;
    }
    for (int row = 0; row < batch; ++row) {
        const int tokens     = valid_columns[row];
        const auto qk_base   = std::size_t(row * width) * kQkHeads * kStateDim;
        const auto v_base    = std::size_t(row * width) * value_heads * kStateDim;
        const auto gate_base = std::size_t(row * width) * value_heads;
        const auto qk_count  = std::size_t(tokens) * kQkHeads * kStateDim;
        const auto v_count   = std::size_t(tokens) * value_heads * kStateDim;
        const auto g_count   = std::size_t(tokens) * value_heads;
        if (tokens) {
            const auto state_begin = state.begin() + initial_slots[row] * (state_elements / slots);
            const auto expected    = gdn_ref::evaluate(
                {kStateDim,
                    kQkHeads,
                    value_heads,
                    tokens,
                    gdn_ref::represented_bf16(std::span(q_bits).subspan(qk_base, qk_count)),
                    gdn_ref::represented_bf16(std::span(k_bits).subspan(qk_base, qk_count)),
                    gdn_ref::represented_bf16(std::span(v_bits).subspan(v_base, v_count)),
                    {g.begin() + gate_base, g.begin() + gate_base + g_count},
                    {beta.begin() + gate_base, beta.begin() + gate_base + g_count},
                    {state_begin, state_begin + state_elements / slots}},
                1.0 / std::sqrt(128.0), true);
            const auto output =
                gdn_ref::represented_bf16(std::span(record_output_bits).subspan(v_base, v_count));
            failures += verify_reduction("replay record output" + suffix,
                                         std::vector<double>(output.begin(), output.end()),
                                         expected.out, gdn_ref::kOutputCriterion);
            if (!std::equal(key_after.begin() + qk_base, key_after.begin() + qk_base + qk_count,
                            k_bits.begin() + qk_base)) {
                std::cerr << "represented record key differs" << suffix << '\n';
                ++failures;
            }
        }
        const auto tail = width - tokens;
        if (!all_equal(record_output_bits, v_base + v_count, tail * value_heads * kStateDim, 0) ||
            !all_equal(key_after, qk_base + qk_count, tail * kQkHeads * kStateDim, kBf16Poison) ||
            !all_equal(delta_after, v_base + v_count, tail * value_heads * kStateDim,
                       kFp32Poison) ||
            !all_equal(alpha_after, gate_base + g_count, tail * value_heads, kFp32Poison)) {
            std::cerr << "invalid output or record suffix differs" << suffix << '\n';
            ++failures;
        }
    }

    if (width == 2 && batch == 1) {
        const auto rejects = [&](const Tensor& query, const Tensor& key, const Tensor& gate,
                                 const Tensor& mask, Tensor output) {
            try {
                ops::gated_delta_net_replay_record(
                    query, key, v, gate, beta_tensor, kScale, history.view(initial), mask,
                    layer.recurrence, output, cuda_environment().context.execution_view());
            } catch (const std::invalid_argument&) { return 0; }
            std::cerr << "record accepted invalid operands" << suffix << '\n';
            return 1;
        };
        failures += rejects(q, q, g_tensor, valid, record_output);
        failures += rejects(q, k, beta_tensor, valid, record_output);
        failures += rejects(q, k, g_tensor, valid, v);
        failures +=
            rejects(q, k, g_tensor, valid,
                    Tensor(MemorySpan{delta_record.data, delta_record.bytes(), delta_record.device},
                           DType::BF16, {kStateDim, value_heads, width, batch}));
        failures += rejects(q, k, g_tensor, valid, q);
        if (state_dtype == DType::FP32) {
            auto bad_source                = history.view(initial);
            bad_source.field.tensor.dtype = DType::FP16;
            bool rejected           = false;
            try {
                ops::gated_delta_net_replay_record(
                    q, k, v, g_tensor, beta_tensor, kScale, bad_source, valid, layer.recurrence,
                    record_output, cuda_environment().context.execution_view());
            } catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) {
                std::cerr << "record accepted FP16 source state" << suffix << '\n';
                ++failures;
            }
        }
    }

    history.copy_back();
    failures += history.verify_guards("record source");
    const bool state_unchanged =
        state_dtype == DType::BF16
            ? from_device<std::uint16_t>(record_state, state_elements) == state_bf16
            : std::memcmp(from_device<float>(record_state, state_elements).data(), state.data(),
                          state_elements * sizeof(float)) == 0;
    if (!state_unchanged) {
        std::cerr << "replay record modified source state" << suffix << "\n";
        ++failures;
    }
    if (from_device<std::int64_t>(device_initial, batch) != initial_offsets ||
        (!dense && from_device<int>(device_valid, batch) != valid_columns) ||
        from_device<float>(device_g, gate_elements) != g ||
        from_device<float>(device_beta, gate_elements) != beta ||
        from_device<std::uint16_t>(q.data, qk_elements) != q_bits ||
        from_device<std::uint16_t>(k.data, qk_elements) != k_bits ||
        from_device<std::uint16_t>(v.data, value_elements) != v_bits) {
        std::cerr << "replay record modified inputs" << suffix << "\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    const auto add = [&](int heads, int width, int batch, std::vector<int> valid, unsigned seed) {
        suite.add(std::string(heads == 24 ? "gdn_record_local/h" : "gdn_record/h") +
                      std::to_string(heads) + "/w" + std::to_string(width) + "/b" +
                      std::to_string(batch) + (valid.empty() ? "/dense" : "/ragged"),
                  [=] { return run_case(heads, width, batch, valid, seed); },
                  {{"execution", width == 2 || width == 9 || width == 16 ? "graph" : "eager"}});
    };
    for (const auto [batch, width] : {std::pair{9, 4}, std::pair{17, 9}, std::pair{64, 2}}) {
        std::vector<int> valid(batch);
        for (int row = 0; row < batch; ++row) valid[row] = row % (width + 1);
        suite.add("gdn_record/expanded/b" + std::to_string(batch),
                  [=] { return run_case(24, width, batch, valid, 1900U + batch); },
                  {{"suite", "concurrency"}});
    }
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
    suite.add("gdn_record/h48/minimum-alignment/graph",
              [] { return run_case(48, 9, 2, {9, 3}, 1793U, true); }, {{"execution", "graph"}});

    for (int t : {2, 4, 8, 9, 16}) {
        add(24, t, 1, {}, 2000U + t);
        add(24, t, 4, {t, 0, 1, t - 1}, 3000U + t);
    }
    add(32, 5, 1, {0}, 1715U);

    // BF16 recurrent state storage cases
    const auto add_bf16 = [&](int heads, int width, int batch, std::vector<int> valid,
                              unsigned seed, bool min_align = false) {
        suite.add("gdn_record_bf16/h" + std::to_string(heads) + "/w" + std::to_string(width) +
                      "/b" + std::to_string(batch) + (valid.empty() ? "/dense" : "/ragged") +
                      (min_align ? "/min-align" : ""),
                  [=] {
                      return run_case(heads, width, batch, valid, seed, min_align, DType::BF16);
                  },
                  {{"execution", width == 2 || width == 9 || width == 16 ? "graph" : "eager"}});
    };
    add_bf16(24, 2, 1, {}, 4001U);
    add_bf16(24, 9, 4, {9, 0, 1, 8}, 4002U);
    add_bf16(32, 6, 8, {6, 5, 4, 3, 2, 1, 6, 2}, 4003U);
    add_bf16(32, 5, 1, {0}, 4004U);
    add_bf16(48, 2, 1, {}, 4005U);
    add_bf16(48, 16, 1, {}, 4006U);
    add_bf16(48, 16, 8, {16, 1, 4, 7, 10, 13, 0, 15}, 4007U);
    add_bf16(48, 9, 2, {9, 3}, 4008U, true);
    return suite.main(argc, argv);
}
