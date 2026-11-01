#pragma once

#include "models/qwen3_5/execution/ple_gather.h"
#include "models/qwen3_5/model.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/moe_slots.h"
#include "ninfer/ops/moe_slots_gguf.h"
#include "ninfer/ops/weight_input.h"

#include <array>
#include <memory>
#include <limits>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

using LinearParameters = ops::SingleProjectionWeight;

[[nodiscard]] inline std::int32_t dimension(std::uint64_t value) {
    if (value > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("model dimension exceeds the Tensor integer domain");
    }
    return static_cast<std::int32_t>(value);
}

struct DenseParameters {
    LinearParameters gate_up;
    LinearParameters down;
};

struct AttentionParameters {
    ops::ProjectionWeights projection;
    Tensor query_norm, key_norm;
    LinearParameters output;
};

struct GdnParameters {
    ops::ProjectionWeights projection;
    ops::ProjectionWeights control;
    Tensor a_log, dt_bias, convolution, norm;
    LinearParameters output;
};

// The Text embedding and output head; the blocks and the final norm are Qwen4Parameters.
struct TextParameters {
    Weight token_embedding;
    LinearParameters output_head;
};

struct MtpProjectionParameters {
    LinearParameters packed;
    // Q, K, gate and V alone: the incremental path projects K/V and Q/gate independently.
    std::array<LinearParameters, 4> rows;
};

struct MtpParameters {
    LinearParameters input_projection;
    Tensor embedding_norm, hidden_norm, input_norm, post_attention_norm, final_norm;
    MtpProjectionParameters projection;
    Tensor query_norm, key_norm;
    LinearParameters output;
    DenseParameters ffn;
    // Empty on a tensor-parallel rank that does not hold the selected head: a PrimaryOnly
    // optimized proposal head proposes on rank 0 alone. A vocabulary-split one is this rank's
    // row block on both ranks.
    LinearParameters output_head;
};

struct NormParameters {
    Tensor weight, bias;
};

struct VisionBlockParameters {
    NormParameters norm1, norm2;
    LinearParameters qkv;
    Tensor qkv_bias;
    LinearParameters output, fc1, fc2;
    Tensor output_bias, fc1_bias, fc2_bias;
};

struct VisionParameters {
    LinearParameters patch_embedding;
    Tensor patch_embedding_bias, position_embedding;
    std::vector<VisionBlockParameters> layers;
    NormParameters merger_norm;
    LinearParameters merger_fc1, merger_fc2;
    Tensor merger_fc1_bias, merger_fc2_bias;
};

struct DynamicConvParameters {
    Tensor base_kernel;
    LinearParameters kernel_projection;
};

struct DraftBlockParameters {
    Tensor input_norm, post_attention_norm;
    LinearParameters query_key_value, context_key, context_value;
    Tensor query_norm, key_norm;
    LinearParameters output;
    DenseParameters mlp;
    std::optional<DynamicConvParameters> attention_conv, mlp_conv;
};

struct SelectorParameters {
    LinearParameters hidden_projection;
    Tensor predecessor_codebook, successor_codebook;
};

struct DraftParameters {
    LinearParameters feature_projection;
    Tensor context_norm, final_norm;
    std::vector<DraftBlockParameters> layers;
    std::optional<SelectorParameters> selector;
    LinearParameters output_head;
};

struct ProposalParameters {
    LinearParameters head; // This rank's rows: all of them, or a vocabulary block when split.
    std::optional<Tensor> token_ids; // Row -> token ID map; rank 0 only at tp 2.
    std::uint32_t rows = 0;          // Rows of the whole head.

    // At tp 2 under MTP each rank holds half of the rows (load/sharding.h).
    [[nodiscard]] bool split() const noexcept {
        return static_cast<std::uint64_t>(head.weight.n) != rows;
    }
};

// ---- Qwen4Exp (Qwen3.8-Flash-Next) -------------------------------------------------------------
// Names follow ninfer-ext's execution (giveen/ninfer-ext @9c933f01, Apache-2.0) and T12's
// bindings (weights.h `Qwen4*Weights`).

// One hyper-connection mixer: the T10 Op operands (down = [low rank (+ S inject rows), S*H]).
struct HyperConnectionParameters {
    ops::HyperConnectionWeights weights;
    bool inject = false;
};

struct QsaParameters {
    LinearParameters projection; // query, key, gate, value rows
    Tensor query_norm, key_norm;
    LinearParameters output;
    LinearParameters indexer; // indexer query rows, then the raw key row block
    Tensor indexer_query_norm, indexer_key_norm;
};

// Shape-keyed routes of a Qwen4 block (docs/maintainer/qwen4-exp-model.md): the 27B's fused Ops
// where the rank's projection geometry and format have a registered route, else the generic chain
// (ops::linear + split copies + the elementwise Ops). Resolved once when the Parameters are
// prepared (Prepare::qwen4_routes), per rank: at tp 2 they name the two-device split forms.
struct Qwen4FusedRoutes {
    bool gdn_input  = false; // ops::gdn_input_proj(+conv snapshot/record) | tp 2 *_split
    bool gdn_gating = false; // ops::gdn_gating_proj | tp 2 gdn_control_split
    bool attn_input = false; // ops::attn_input_proj | tp 2 attention_projection_split
    bool swiglu     = false; // ops::linear_swiglu | tp 2 linear_swiglu_column_parallel
};

// The routed experts of one layer as stored: the host-mapped NVFP4 banks (device-addressable
// pinned memory, ShardAxis::HostMapped) in the moe_slots plane description, expert e of a plane at
// base + e * stride; the stacked weight divisors (gate of e at 2e, up at 2e+1; down at e), also
// host-mapped; and the layer's two activation divisors.
//
// GGUF banks (gguf = true; artifact layout gguf_blocks_v1, capy-flash v0.7): the same two parents
// as ggml block rows, expert e = gate|up rows [e*1280, (e+1)*1280) and down rows [e*2560,
// (e+1)*2560); no divisors, `bank` and the divisor fields stay empty.
struct Qwen4ExpertBank {
    ops::MoeSlotPool bank; // slots = experts
    const float* gate_up_divisors       = nullptr;
    const float* down_divisors          = nullptr;
    float gate_up_input_divisor         = 0.0F;
    float down_input_divisor            = 0.0F;
    bool gguf                           = false;
    ops::MoeGgufFormat gguf_gate_up     = ops::MoeGgufFormat::IQ3_XXS;
    ops::MoeGgufFormat gguf_down        = ops::MoeGgufFormat::IQ4_NL;
    const std::byte* gguf_gate_up_rows  = nullptr;
    const std::byte* gguf_down_rows     = nullptr;
    std::int64_t gguf_gate_up_row_bytes = 0;
    std::int64_t gguf_down_row_bytes    = 0;

    // Bytes of one expert (1280 gate|up rows and 2560 down rows).
    [[nodiscard]] std::int64_t gguf_expert_bytes() const noexcept {
        return 1280 * gguf_gate_up_row_bytes + 2560 * gguf_down_row_bytes;
    }
};

// The shared expert (gate|up rows, down) of one block.
struct Qwen4SharedExpertParameters {
    LinearParameters gate_up;
    LinearParameters down;
};

struct Qwen4MoeParameters {
    Tensor router; // BF16 [H, E+1]: the routed rows then the shared-expert score row
    // Whole on every block at one device; at tp 2 held by rank (block % 2) alone (the MTP block
    // by rank 0, load/sharding.h), so empty on the other rank.
    std::optional<Qwen4SharedExpertParameters> shared;
    Qwen4ExpertBank experts;
};

struct PleParameters {
    PleTableView table;
    PleHashConfig hash;
    LinearParameters key_value; // key rows [S*H], then value rows [H]
    Tensor key_norm, query_norm, conv_norm, convolution;
};

// The FFN of a block: the MoE (Qwen3.8-Flash-Next) or a dense SwiGLU FFN, gate|up rows in one
// projection and down (at tp 2 this rank's intermediate rows and down columns).
using Qwen4FfnParameters = std::variant<Qwen4MoeParameters, DenseParameters>;

struct Qwen4BlockParameters {
    // Hyper-connection mixers; empty for a model without hyper-connections (TextConfig has no
    // hyper_connection), whose residual is one [H,T] stream normed by the two RMSNorm weights
    // below before the mixer and the FFN (the Qwen3.5 block).
    HyperConnectionParameters attention_hc, ffn_hc;
    Tensor input_norm, post_attention_norm;
    // QSA (sparse attention with its indexer), GDN or dense gated full attention. GDN and dense
    // attention are the Qwen3.5 mixers' operands: Qwen4Exp binds each projection (q|k|v|z, a|b,
    // q|k|gate|v) as one parent; a dense Qwen3.5 model keeps its native forms.
    std::variant<QsaParameters, GdnParameters, AttentionParameters> mixer;
    Qwen4FfnParameters ffn;
    std::optional<PleParameters> ple;
    Qwen4FusedRoutes fused;

    // The MoE of a routed-expert block (std::bad_variant_access for a dense FFN).
    [[nodiscard]] const Qwen4MoeParameters& moe() const {
        return std::get<Qwen4MoeParameters>(ffn);
    }

    [[nodiscard]] Qwen4MoeParameters& moe() { return std::get<Qwen4MoeParameters>(ffn); }
};

struct Qwen4Parameters {
    // The Text output mixer (hyper-connections), or, without hyper-connections, the final RMSNorm
    // weight: the carried hidden is then the final-normed residual and the output head reads it
    // directly (the Qwen3.5 convention).
    HyperConnectionParameters head;
    Tensor final_norm;
    std::vector<Qwen4BlockParameters> layers;
};

// The MTP layer (docs/maintainer/qwen4-exp-model.md#mtp): stem over the wide residual and the
// shifted token, one QSA block with its own 512 routed experts (or, for a dense model, its own
// dense FFN), and its own output mixer in front of the Text output head.
//
// Its routed experts reach the W4A4 expert Op like the Text banks, with two differences the
// preparation resolves (T24):
//   * weight divisors: one per matrix (this tree's recipes) or one for the whole bank (ninfer-ext's
//     artifact); `uniform_*_divisor` holds the latter, which the Program expands into per-expert
//     planes (moe_slots indexes divisors by expert id);
//   * activation divisors: the checkpoint calibrates none for the MTP experts (the recipes store
//     them A16Only). A calibrated pair stored with the bank (AllowA4) is used as is; otherwise the
//     pair is derived from the Text layers' calibrated ranges: the widest range of all 48 layers
//     times kQwen4MtpActivationRangeMargin. The W4A4 error is flat while the range lies between
//     the largest activation and ~64x it (E4M3 block scales keep three mantissa bits over that
//     span); on the real model the MTP inputs reach 24.5 (x) and 826 (h) against Text ranges of
//     13.9 and 57.8, and margins 16-64 give the minimum error with no saturated block, 4 saturates
//     h blocks (T24, docs/capy-flash/development-log.md). 32 centres both inputs in the flat
//     region.
inline constexpr float kQwen4MtpActivationRangeMargin = 32.0F;

struct Qwen4MtpParameters {
    Tensor embedding_norm; // [H]
    Tensor hidden_norm;    // [S*H]: one RMS over the whole wide residual
    LinearParameters embedding_projection;
    LinearParameters hidden_projection;
    HyperConnectionParameters head;       // no inject
    Qwen4BlockParameters layer;           // QSA mixer, MoE over the MTP banks or dense FFN
    float uniform_gate_up_divisor = 0.0F; // > 0: the bank keeps one weight divisor per projection
    float uniform_down_divisor    = 0.0F;
    bool derived_activation_divisors = false;
};

// Cold native preparation for the fixed model implementation. This owner is stable before
// startup sizing, execution, or Graph capture; all weight addresses borrow the source Model.
// Shape-dependent kernel selection and scratch remain with the calling implementation and Op.
//
// A tensor-parallel Model has one Parameters per rank. Its operands are that rank's shards,
// with head, intermediate and vocabulary extents divided by tp (load/sharding.h); Vision, draft
// and proposal parameters exist only on the rank that holds them.
class Parameters {
public:
    explicit Parameters(const Model& source) : Parameters(source, 0) {}

    Parameters(const Model& source, int device);
    Parameters(const Parameters&)            = delete;
    Parameters& operator=(const Parameters&) = delete;
    Parameters(Parameters&&)                 = delete;
    Parameters& operator=(Parameters&&)      = delete;

    const Model& model;
    const int device;
    TextParameters text;
    std::optional<MtpParameters> mtp;
    std::optional<VisionParameters> vision;
    std::optional<DraftParameters> draft;
    std::optional<ProposalParameters> proposal;
    // The Text forward's blocks and final norm for every architecture: Qwen4Exp binds its own, a
    // dense Qwen3.5 model its Qwen3.5 blocks without hyper-connections. Qwen4Exp at tp 2: this
    // rank's shards (QSA 12 query
    // + 1 KV heads, GDN 8 key + 24 value heads, the vocabulary half of the output head), the
    // replicated router, indexer, PLE and hyper-connection mixers, the shared expert of its blocks,
    // and the one host-mapped copy of the routed experts that both ranks read.
    std::optional<Qwen4Parameters> qwen4;
    // Qwen4Exp MTP layer, when the artifact's MTP component is loaded.
    std::optional<Qwen4MtpParameters> qwen4_mtp;
};

} // namespace ninfer::models::qwen3_5::execution
