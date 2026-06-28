#include "qus/core/weight.h"

#include <cstdint>
#include <iostream>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

int check_i64(std::int64_t actual, std::int64_t expected, const char* label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int test_as_dense_bf16() {
    int failures = 0;
    int blob     = 0;
    qus::Weight w{};
    w.qtype    = qus::QType::BF16_CTRL;
    w.layout   = qus::QuantLayout::Contiguous;
    w.qdata    = &blob;
    w.ndim     = 2;
    w.shape[0] = 48;
    w.shape[1] = 5120;

    const qus::Tensor t = qus::as_dense(w);
    failures += (t.data == &blob) ? 0 : fail("as_dense bf16: data pointer changed");
    failures += (t.dtype == qus::DType::BF16) ? 0 : fail("as_dense bf16: dtype not BF16");
    failures += check_i64(t.ne[0], 48, "as_dense bf16 ne[0]");
    failures += check_i64(t.ne[1], 5120, "as_dense bf16 ne[1]");
    failures += check_i64(t.ne[2], 1, "as_dense bf16 ne[2]");
    failures += check_i64(t.ne[3], 1, "as_dense bf16 ne[3]");
    failures += t.is_contiguous() ? 0 : fail("as_dense bf16: not contiguous");
    return failures;
}

int test_as_dense_fp32() {
    int failures = 0;
    int blob     = 0;
    qus::Weight w{};
    w.qtype    = qus::QType::FP32_CTRL;
    w.layout   = qus::QuantLayout::Contiguous;
    w.qdata    = &blob;
    w.ndim     = 1;
    w.shape[0] = 64;

    const qus::Tensor t = qus::as_dense(w);
    failures += (t.dtype == qus::DType::FP32) ? 0 : fail("as_dense fp32: dtype not FP32");
    failures += check_i64(t.ne[0], 64, "as_dense fp32 ne[0]");
    return failures;
}

int test_weight_from_dense() {
    int failures = 0;
    int blob     = 0;
    const qus::Tensor t(&blob, qus::DType::BF16, {48, 5120});

    const qus::Weight w = qus::weight_from_dense(t);
    failures +=
        (w.qtype == qus::QType::BF16_CTRL) ? 0 : fail("weight_from_dense: qtype not BF16_CTRL");
    failures += (w.layout == qus::QuantLayout::Contiguous)
                    ? 0
                    : fail("weight_from_dense: layout not Contiguous");
    failures += (w.q5090_scale_dtype == qus::ScaleDType::None)
                    ? 0
                    : fail("weight_from_dense: scale dtype not None");
    failures += (w.qdata == &blob) ? 0 : fail("weight_from_dense: qdata mismatch");
    failures += check_i64(static_cast<std::int64_t>(w.payload_bytes),
                          static_cast<std::int64_t>(t.bytes()), "weight_from_dense payload_bytes");
    failures += check_i64(w.n, 48, "weight_from_dense n");
    failures += check_i64(w.k, 5120, "weight_from_dense k");
    failures += check_i64(w.group, 0, "weight_from_dense group");
    failures += check_i64(w.ndim, 2, "weight_from_dense ndim");
    failures += check_i64(w.shape[0], 48, "weight_from_dense shape[0]");
    failures += check_i64(w.shape[1], 5120, "weight_from_dense shape[1]");
    failures += check_i64(w.padded_shape[0], 48, "weight_from_dense padded_shape[0]");
    failures += check_i64(w.padded_shape[1], 5120, "weight_from_dense padded_shape[1]");
    failures += check_i64(w.padded_shape[2], 1, "weight_from_dense padded_shape[2]");
    failures += check_i64(w.padded_shape[3], 1, "weight_from_dense padded_shape[3]");
    return failures;
}

int test_round_trip() {
    int failures = 0;
    int blob     = 0;
    const qus::Tensor t(&blob, qus::DType::BF16, {48, 5120});

    const qus::Tensor back = qus::as_dense(qus::weight_from_dense(t));
    failures += (back.data == t.data) ? 0 : fail("round-trip: data changed");
    failures += (back.dtype == t.dtype) ? 0 : fail("round-trip: dtype changed");
    failures += check_i64(back.ne[0], t.ne[0], "round-trip ne[0]");
    failures += check_i64(back.ne[1], t.ne[1], "round-trip ne[1]");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_as_dense_bf16();
    failures += test_as_dense_fp32();
    failures += test_weight_from_dense();
    failures += test_round_trip();
    return failures == 0 ? 0 : fail("weight bridge test failed");
}
