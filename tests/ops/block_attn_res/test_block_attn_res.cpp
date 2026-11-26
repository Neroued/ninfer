#include "oracle.h"

#include "core/decode_graph.h"

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::test;

namespace {

int run(int d, int m, int n, int q, bool norm, bool add, bool normalize, bool graph) {
    const auto cuda    = cuda_environment().context.execution_view();
    const auto count   = std::size_t(d) * m;
    const auto wcount  = std::size_t(d) * q;
    const auto p       = bf16_words(attn_res::values(count, 41));
    const auto s       = bf16_words(attn_res::values(count * n, 42));
    const auto a       = bf16_words(attn_res::values(add ? count : 0, 43, 0.007f));
    const auto w       = bf16_words(attn_res::values(norm ? wcount : 0, 44));
    const auto queries = attn_res::values(wcount, 45, normalize ? 2.0f : 5.0f);
    auto dp = to_device(p), dq = to_device(queries);
    DeviceBuffer ds, da, dw;
    const auto optional = [](DeviceBuffer& storage, const auto& words, auto shape) {
        if (words.empty()) return Tensor{};
        storage = to_device(words);
        return Tensor(storage.span(), DType::BF16, shape);
    };
    const auto stack  = optional(ds, s, std::initializer_list<int>{d, m, n});
    const auto addend = optional(da, a, std::initializer_list<int>{d, m});
    const auto weight = optional(dw, w, std::initializer_list<int>{d, q});
    const Tensor partial(dp.span(), DType::BF16, {d, m});
    const Tensor query(dq.span(), DType::FP32, {d, q});
    DeviceBuffer output(cuda.device(), count * q * (norm ? 2 : 4));
    Tensor out(output.span(), norm ? DType::BF16 : DType::FP32, {d, m, q});
    const BlockAttnResParameters params{normalize ? 1.0f / std::sqrt(float(d)) : 1.0f, 1.7e-5f,
                                        6.1e-3f, normalize};
    const auto body = [&] {
        block_attn_res(stack, partial, addend, query, weight, params, out, cuda);
    };
    auto expected = [&](const auto& partial_words) {
        return attn_res::oracle(d, m, q, n, decode_bf16_words(s), decode_bf16_words(partial_words),
                                decode_bf16_words(a), queries, decode_bf16_words(w), params);
    };
    int failed = 0;
    if (graph) {
        DecodeGraphDefinition definition;
        definition.capture(cuda.stream(), body);
        DecodeGraphExecutable replay;
        replay.instantiate(definition);
        replay.launch(cuda.stream());
        cuda_synchronize(cuda.stream());
        failed += attn_res::check("first replay", output, expected(p), norm);
        const auto next = bf16_words(attn_res::values(count, 81));
        dp.copy_from_host(next.data(), dp.size());
        replay.launch(cuda.stream());
        cuda_synchronize(cuda.stream());
        failed += attn_res::check("second replay", output, expected(next), norm);
    } else {
        body();
        cuda_synchronize(cuda.stream());
        failed += attn_res::check("attention mix", output, expected(p), norm);
    }
    return failed;
}

int extreme_norm(int d, float magnitude) {
    const auto cuda    = cuda_environment().context.execution_view();
    const auto partial = std::vector<double>(d, magnitude);
    const auto stack   = std::vector<double>(d, d == 1 ? magnitude : -magnitude);
    const auto queries = std::vector<float>(d, d == 1 ? 0.0f : 1.0f / d);
    const auto weights = std::vector<double>(d, 1.0);
    auto dp            = to_device_bf16(std::vector<float>(partial.begin(), partial.end()));
    auto ds            = to_device_bf16(std::vector<float>(stack.begin(), stack.end()));
    auto dq            = to_device(queries);
    auto dw            = to_device_bf16(std::vector<float>(d, 1.0f));
    DeviceBuffer output(cuda.device(), std::size_t(d) * 2);
    Tensor out(output.span(), DType::BF16, {d, 1, 1});
    const BlockAttnResParameters params{1.0f, 1.0e-6f, 1.0e-6f, false};
    block_attn_res(Tensor(ds.span(), DType::BF16, {d, 1, 1}),
                   Tensor(dp.span(), DType::BF16, {d, 1}), {},
                   Tensor(dq.span(), DType::FP32, {d, 1}), Tensor(dw.span(), DType::BF16, {d, 1}),
                   params, out, cuda);
    cuda_synchronize(cuda.stream());
    const auto expected =
        attn_res::oracle(d, 1, 1, 1, stack, partial, {}, queries, weights, params);
    return attn_res::check("extreme RMSNorm", output, expected, true);
}

int validation() {
    const auto cuda = cuda_environment().context.execution_view();
    DeviceBuffer input(cuda.device(), 128), queries(cuda.device(), 128), output(cuda.device(), 128);
    const Tensor p(input.span(), DType::BF16, {8, 1});
    const Tensor q(queries.span(), DType::FP32, {8, 1});
    Tensor good(output.span(), DType::FP32, {8, 1, 1});
    int failed  = 0;
    auto invoke = [&](const Tensor& source, const Tensor& query, Tensor& dst, const auto& params) {
        block_attn_res({}, source, {}, query, {}, params, dst, cuda);
    };
    const BlockAttnResParameters defaults;
    auto wrong  = good;
    wrong.dtype = DType::BF16;
    failed += attn_res::rejects("dtype", [&] { invoke(p, q, wrong, defaults); });
    wrong      = good;
    wrong.data = input.data();
    failed += attn_res::rejects("alias", [&] { invoke(p, q, wrong, defaults); });
    auto bad = p;
    bad.data = static_cast<char*>(input.data()) + 2;
    failed += attn_res::rejects("alignment", [&] { invoke(bad, q, good, defaults); });
    bad       = q;
    bad.ne[1] = 0;
    failed += attn_res::rejects("zero queries", [&] { invoke(p, bad, good, defaults); });
    bad        = p;
    bad.device = Device{};
    failed += attn_res::rejects("device", [&] { invoke(bad, q, good, defaults); });
    auto params    = defaults;
    params.rms_eps = std::numeric_limits<float>::infinity();
    failed += attn_res::rejects("epsilon", [&] { invoke(p, q, good, params); });
    params             = defaults;
    params.query_scale = 0.0f;
    failed += attn_res::rejects("scale", [&] { invoke(p, q, good, params); });
    return failed;
}

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    suite.add("single_raw_fp32_addend", [] { return run(1, 1, 0, 1, false, true, false, false); });
    suite.add("one_token_many_sources",
              [] { return run(2304, 1, 19, 4, true, true, true, false); });
    suite.add("odd_channels_graph", [] { return run(257, 3, 2, 5, true, true, false, true); });
    suite.add("seven_columns", [] { return run(256, 7, 0, 3, false, false, true, false); });
    suite.add("raw_graph", [] { return run(2304, 17, 7, 4, false, true, true, true); });
    suite.add("vision", [] { return run(1024, 63, 1, 2, true, false, false, false); });
    suite.add("large_channels", [] { return run(4096, 129, 2, 3, true, true, true, false); });
    suite.add("odd_large_channels", [] { return run(2305, 3, 3, 3, true, true, true, false); });
    suite.add("unbounded_channels", [] { return run(4104, 3, 1, 5, false, true, false, false); });
    suite.add("norm_numerator_overflow", [] { return extreme_norm(1, 0x1p63f); });
    suite.add("norm_square_sum_overflow", [] { return extreme_norm(1024, 0x1p60f); });
    suite.add("wide_scalar_4097", [] { return run(4097, 3, 7, 2, false, true, true, false); });
    suite.add("wide_vector_4608", [] { return run(4608, 3, 5, 2, true, true, false, false); });
    suite.add("wide_many_sources", [] { return run(513, 1, 256, 1, false, true, true, false); });
    suite.add("prefill_509", [] { return run(2304, 509, 3, 1, true, true, true, false); });
    suite.add("prefill_1021", [] { return run(2304, 1021, 3, 1, true, true, true, false); });
    suite.add("prefill_2053", [] { return run(2304, 2053, 5, 3, true, true, true, false); });
    suite.add("validation", validation);
    return suite.main(argc, argv);
}
