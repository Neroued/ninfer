// qus::kernels - sample_column wrapper: public api validation and dispatch.
#include "qus/kernels/sampling.h"

#include "kernels/launcher/sampling.h"  // detail::sample_column_launch

#include <stdexcept>

namespace qus::kernels {

void sample_column(const Tensor& logits, Tensor& out, const SamplingConfig* config,
                   const std::int32_t* pos_base, std::int32_t purpose, cudaStream_t stream) {
    if (logits.dtype != DType::BF16) {
        throw std::invalid_argument("sample_column: logits must be BF16");
    }
    if (out.dtype != DType::I32) {
        throw std::invalid_argument("sample_column: out must be I32");
    }
    if (logits.ne[2] != 1 || logits.ne[3] != 1) {
        throw std::invalid_argument("sample_column: logits must be rank-2 [vocab,T]");
    }
    if (out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("sample_column: out must be rank-1 [T]");
    }
    if (logits.ne[0] <= 0) {
        throw std::invalid_argument("sample_column: vocab must be positive");
    }
    if (out.ne[0] != logits.ne[1]) {
        throw std::invalid_argument("sample_column: out shape must be [logits.ne[1]]");
    }
    if (logits.ne[1] <= 0) { return; }
    if (!logits.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("sample_column: logits/out must be contiguous");
    }
    if (logits.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("sample_column: logits/out data must be non-null");
    }
    if (config == nullptr) {
        throw std::invalid_argument("sample_column: config must be non-null");
    }
    if (pos_base == nullptr) {
        throw std::invalid_argument("sample_column: pos_base must be non-null");
    }

    detail::sample_column_launch(logits, out, config, pos_base, purpose, stream);
}

} // namespace qus::kernels
