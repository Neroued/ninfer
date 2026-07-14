// ninfer::kernels - causal_conv1d wrapper: public api validation and launcher dispatch.
#include "ninfer/kernels/causal_conv1d_silu.h"

#include "kernels/launcher/causal_conv1d.h" // detail::causal_conv1d_*_launch

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::kernels {
namespace {

std::int64_t numel_allow_zero(const Tensor& t, const char* label) {
    bool has_zero = false;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string("causal_conv1d: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { has_zero = true; }
    }
    if (has_zero) { return 0; }

    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("causal_conv1d: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

void require_x_shape(const Tensor& x) {
    if (x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("causal_conv1d: x must have shape [C,T]");
    }
    if (x.ne[0] <= 0) { throw std::invalid_argument("causal_conv1d: C must be positive"); }
}

void require_weight_shape(const Tensor& weight, std::int32_t C) {
    if (weight.ne[0] != C || weight.ne[1] != 4 || weight.ne[2] != 1 || weight.ne[3] != 1) {
        throw std::invalid_argument("causal_conv1d: weight must have shape [C,4]");
    }
}

void require_state_shape(const Tensor& conv_state, std::int32_t C) {
    if (conv_state.ne[0] != C || conv_state.ne[1] != 3 || conv_state.ne[2] != 1 ||
        conv_state.ne[3] != 1) {
        throw std::invalid_argument("causal_conv1d: conv_state must have shape [C,3]");
    }
}

void require_snapshot_state_shape(const Tensor& conv_states, std::int32_t C, std::int32_t T) {
    if (conv_states.ne[0] != C || conv_states.ne[1] != 3 || conv_states.ne[2] < T ||
        conv_states.ne[3] != 1) {
        throw std::invalid_argument("causal_conv1d: conv_states must have shape [C,3,S] with S>=T");
    }
}

void require_out_shape(const Tensor& x, const Tensor& out) {
    for (int d = 0; d < 4; ++d) {
        if (out.ne[d] != x.ne[d]) {
            throw std::invalid_argument("causal_conv1d: out shape must match x");
        }
    }
}

void require_initial_slot_shape(const Tensor& initial_slot) {
    if (initial_slot.ne[0] != 1 || initial_slot.ne[1] != 1 || initial_slot.ne[2] != 1 ||
        initial_slot.ne[3] != 1) {
        throw std::invalid_argument("causal_conv1d: initial_slot must have shape [1]");
    }
}

std::int64_t validate_common(const Tensor& x, const Tensor& weight, const Tensor& conv_state,
                             const Tensor& out) {
    if (x.dtype != DType::BF16 || weight.dtype != DType::BF16 || conv_state.dtype != DType::BF16 ||
        out.dtype != DType::BF16) {
        throw std::invalid_argument("causal_conv1d: x/weight/conv_state/out must be BF16");
    }

    const std::int64_t n = numel_allow_zero(x, "x");
    (void)numel_allow_zero(weight, "weight");
    (void)numel_allow_zero(conv_state, "conv_state");
    (void)numel_allow_zero(out, "out");

    require_x_shape(x);
    require_weight_shape(weight, x.ne[0]);
    require_state_shape(conv_state, x.ne[0]);
    require_out_shape(x, out);
    return n;
}

void require_non_empty_accessible(const Tensor& x, const Tensor& weight, const Tensor& conv_state,
                                  const Tensor& out) {
    if (!x.is_contiguous() || !weight.is_contiguous() || !conv_state.is_contiguous() ||
        !out.is_contiguous()) {
        throw std::invalid_argument("causal_conv1d: all tensors must be contiguous");
    }
    if (x.data == nullptr || weight.data == nullptr || conv_state.data == nullptr ||
        out.data == nullptr) {
        throw std::invalid_argument("causal_conv1d: all tensor data pointers must be non-null");
    }
}

void require_initial_slot_accessible(const Tensor& initial_slot) {
    if (!initial_slot.is_contiguous()) {
        throw std::invalid_argument("causal_conv1d: initial_slot must be contiguous");
    }
    if (initial_slot.data == nullptr) {
        throw std::invalid_argument("causal_conv1d: initial_slot data pointer must be non-null");
    }
}

} // namespace

void causal_conv1d_silu(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                        Tensor& conv_state_out, Tensor& out, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || weight.dtype != DType::BF16 ||
        conv_state_in.dtype != DType::BF16 || conv_state_out.dtype != DType::BF16 ||
        out.dtype != DType::BF16) {
        throw std::invalid_argument("causal_conv1d: x/weight/conv_state/out must be BF16");
    }

    const std::int64_t n = numel_allow_zero(x, "x");
    (void)numel_allow_zero(weight, "weight");
    (void)numel_allow_zero(conv_state_in, "conv_state_in");
    (void)numel_allow_zero(conv_state_out, "conv_state_out");
    (void)numel_allow_zero(out, "out");

    require_x_shape(x);
    require_weight_shape(weight, x.ne[0]);
    require_state_shape(conv_state_in, x.ne[0]);
    require_state_shape(conv_state_out, x.ne[0]);
    require_out_shape(x, out);
    if (n == 0) { return; }

    require_non_empty_accessible(x, weight, conv_state_in, out);
    if (!conv_state_out.is_contiguous() || conv_state_out.data == nullptr) {
        throw std::invalid_argument(
            "causal_conv1d: conv_state_out must be contiguous and non-null");
    }
    detail::causal_conv1d_prefill_launch(x, weight, conv_state_in, conv_state_out, out, stream);
}

void causal_conv1d_silu(const Tensor& x, const Tensor& weight, Tensor& conv_state, Tensor& out,
                        cudaStream_t stream) {
    const std::int64_t n = validate_common(x, weight, conv_state, out);
    if (n == 0) { return; }

    require_non_empty_accessible(x, weight, conv_state, out);
    if (x.ne[1] == 1) {
        detail::causal_conv1d_decode_launch(x, weight, conv_state, out, stream);
    } else {
        detail::causal_conv1d_prefill_launch(x, weight, conv_state, conv_state, out, stream);
    }
}

void causal_conv1d_silu_snapshot(const Tensor& x, const Tensor& weight, Tensor& conv_states,
                                 const Tensor& initial_slot, Tensor& out, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || weight.dtype != DType::BF16 || conv_states.dtype != DType::BF16 ||
        out.dtype != DType::BF16) {
        throw std::invalid_argument("causal_conv1d: x/weight/conv_states/out must be BF16");
    }
    if (initial_slot.dtype != DType::I32) {
        throw std::invalid_argument("causal_conv1d: initial_slot must be I32");
    }

    const std::int64_t n = numel_allow_zero(x, "x");
    (void)numel_allow_zero(weight, "weight");
    (void)numel_allow_zero(conv_states, "conv_states");
    (void)numel_allow_zero(initial_slot, "initial_slot");
    (void)numel_allow_zero(out, "out");

    require_x_shape(x);
    if (x.ne[1] <= 0) { throw std::invalid_argument("causal_conv1d: snapshot T must be positive"); }
    require_weight_shape(weight, x.ne[0]);
    require_snapshot_state_shape(conv_states, x.ne[0], x.ne[1]);
    require_initial_slot_shape(initial_slot);
    require_out_shape(x, out);
    if (n == 0) { return; }

    require_non_empty_accessible(x, weight, conv_states, out);
    require_initial_slot_accessible(initial_slot);
    detail::causal_conv1d_sequence_snapshot_launch(x, weight, conv_states, initial_slot, out,
                                                   stream);
}

} // namespace ninfer::kernels
