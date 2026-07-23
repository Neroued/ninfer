#include "ninfer/ops/mtp_round.h"
#include "ops/launcher/mtp_round.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_contiguous_nonnull(const Tensor& t, const char* op, const char* name) {
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

void require_vector(const Tensor& t, DType dtype, std::int32_t n, const char* op,
                    const char* name) {
    if (t.dtype != dtype || n <= 0 || t.ne[0] != n || t.ne[1] != 1 || t.ne[2] != 1 ||
        t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid vector shape for " + name);
    }
    require_contiguous_nonnull(t, op, name);
}

void require_scalar(const Tensor& t, DType dtype, const char* op, const char* name) {
    require_vector(t, dtype, 1, op, name);
}

} // namespace

void mtp_prepare_alignment_ids(const Tensor& verify_ids, const Tensor& token,
                               const Tensor& accepted, Tensor& alignment_ids, cudaStream_t stream) {
    constexpr const char* op = "mtp_prepare_alignment_ids";
    const std::int32_t T     = verify_ids.ne[0];
    if (T < 2 || T > 6) {
        throw std::invalid_argument("mtp_prepare_alignment_ids: T must be in [2,6]");
    }
    require_vector(verify_ids, DType::I32, T, op, "verify_ids");
    require_scalar(token, DType::I32, op, "token");
    require_scalar(accepted, DType::I32, op, "accepted");
    require_vector(alignment_ids, DType::I32, T, op, "alignment_ids");
    detail::mtp_prepare_alignment_ids_launch(verify_ids, token, accepted, alignment_ids, stream);
}

} // namespace ninfer::ops
