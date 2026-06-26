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

} // namespace

int main() {
    int failures = 0;
    failures += test_as_dense_bf16();
    failures += test_as_dense_fp32();
    return failures == 0 ? 0 : fail("weight bridge test failed");
}
