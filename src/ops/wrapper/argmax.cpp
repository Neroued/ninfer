// ninfer::ops - argmax wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/argmax.h"

#include "ops/launcher/argmax.h" // detail::argmax_launch

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t numel_allow_zero(const Tensor& t, const char* label) {
    bool has_zero = false;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string("argmax: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { has_zero = true; }
    }
    if (has_zero) { return 0; }

    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("argmax: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

} // namespace

void argmax(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream) {
    if (logits.dtype != DType::BF16) { throw std::invalid_argument("argmax: logits must be BF16"); }
    if (out.dtype != DType::I32) { throw std::invalid_argument("argmax: out must be I32"); }

    const std::int64_t logits_n = numel_allow_zero(logits, "logits");
    (void)numel_allow_zero(out, "out");

    if (logits.ne[2] != 1 || logits.ne[3] != 1) {
        throw std::invalid_argument("argmax: logits must be rank-2 [vocab,T]");
    }
    if (out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("argmax: out must be rank-1 [T]");
    }
    if (logits.ne[0] <= 0) {
        throw std::invalid_argument("argmax: physical rows must be positive");
    }
    if (valid_rows <= 0 || valid_rows > logits.ne[0]) {
        throw std::invalid_argument("argmax: valid_rows must be in [1, logits.ne[0]]");
    }
    if (out.ne[0] != logits.ne[1]) {
        throw std::invalid_argument("argmax: out shape must be [logits.ne[1]]");
    }
    if (logits_n == 0) { return; }

    if (!logits.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("argmax: logits/out must be contiguous");
    }
    if (logits.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("argmax: logits/out data must be non-null");
    }

    detail::argmax_launch(logits, out, valid_rows, stream);
}

} // namespace ninfer::ops
