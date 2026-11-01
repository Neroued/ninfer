#include "models/qwen3_5/execution/parameters.h"

#include "core/device.h"
#include "core/weight_view.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/moe_route.h"
#include "ninfer/ops/tp2/attn_input_proj.h"
#include "ninfer/ops/tp2/gdn_gating_proj.h"
#include "ninfer/ops/tp2/gdn_input_proj.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5::execution {
namespace {

template <class Function>
auto with_context(const std::string& context, Function&& function) {
    try {
        return function();
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(context + ": " + error.what());
    }
}

bool contiguous(std::span<const ops::WeightInput> inputs) {
    const WeightParent* parent = nullptr;
    std::uint64_t end          = 0;
    for (const auto& input : inputs) {
        for (const auto& part : input.weight.parts) {
            if (parent && (part.parent != parent || part.begin != end)) { return false; }
            parent = part.parent;
            end    = part.end;
        }
    }
    return parent != nullptr;
}

// Whether a capacity query of a fused Op accepts the profile (registered routes answer, the
// others throw). Startup only: the answer is kept in Qwen4FusedRoutes.
template <class Query>
bool registered(Query&& query) {
    try {
        (void)query();
        return true;
    } catch (const std::exception&) { return false; }
}

// NINFER_QWEN4_FUSED=0 keeps every Qwen4Exp block on the generic chain (A/B and bisection).
bool qwen4_fused_enabled() {
    static const bool on = [] {
        const char* value = std::getenv("NINFER_QWEN4_FUSED");
        return value == nullptr || std::string(value) != "0";
    }();
    return on;
}

// The largest prompt chunk a capacity query covers (the routes are T-independent past it).
constexpr std::int32_t kRouteQueryColumns = 65536;

class Prepare {
public:
    Prepare(const Model& model, int device)
        : model_(model), device_(device), sharded_(model.device_count() > 1) {}

    const BoundWeight& weight(WeightId id) const { return model_.weight(id, device_); }

    bool resident(WeightId id) const { return model_.resident(id, device_); }

    ops::WeightInput input(WeightId id) const { return model_.input(id, device_); }

    ops::WeightInput input(WeightUseId id) const { return model_.input(id, device_); }

    LinearParameters linear(WeightId id) const {
        return with_context(weight(id).name, [&] { return ops::prepare_linear_weight(input(id)); });
    }

    LinearParameters linear(WeightUseId id) const {
        return with_context(weight(id.parameter).name,
                            [&] { return ops::prepare_linear_weight(input(id)); });
    }

    // ops::prepare_{attn,gdn}_input_proj_weights and prepare_gdn_gating_proj_weights name the
    // complete tp1 geometries. A tensor-parallel rank keeps the same row order and native forms
    // over its shard parents: one contiguous parent, or a pair split after `pair` inputs. Format
    // support for the shard geometry is enforced by the split Op that consumes it.
    ops::ProjectionWeights shard_bank(std::span<const ops::WeightInput> inputs,
                                      std::size_t pair) const {
        if (contiguous(inputs)) { return ops::prepare_linear_weight(inputs); }
        return ops::PairedProjectionWeights{
            ops::prepare_linear_weight(inputs.first(pair)).weight,
            ops::prepare_linear_weight(inputs.subspan(pair)).weight};
    }

    ops::ProjectionWeights attention_projection(const AttentionWeights& a) const {
        if (sharded_) {
            const std::array inputs{input(a.query), input(a.key), input(a.gate), input(a.value)};
            return shard_bank(inputs, 2);
        }
        return ops::prepare_attn_input_proj_weights(input(a.query), input(a.key), input(a.gate),
                                                    input(a.value));
    }

    ops::ProjectionWeights gdn_projection(const GdnWeights& g) const {
        if (sharded_) {
            const std::array inputs{input(g.query), input(g.key), input(g.value), input(g.z)};
            return shard_bank(inputs, 2);
        }
        return ops::prepare_gdn_input_proj_weights(input(g.query), input(g.key), input(g.value),
                                                   input(g.z));
    }

    ops::ProjectionWeights gdn_control(const GdnWeights& g) const {
        if (sharded_) {
            const std::array inputs{input(g.a_projection), input(g.b_projection)};
            return shard_bank(inputs, 1);
        }
        return ops::prepare_gdn_gating_proj_weights(input(g.a_projection), input(g.b_projection));
    }

    Tensor tensor(WeightId id) const {
        const auto& bound = weight(id);
        return with_context(bound.name, [&] {
            const auto& view = bound.view;
            if (view.shape.size() > 4) {
                throw std::invalid_argument("direct parameter exceeds Tensor rank");
            }
            std::array<std::int32_t, 4> axes{1, 1, 1, 1};
            for (std::size_t i = 0; i < view.shape.size(); ++i) {
                const auto extent = view.shape[view.shape.size() - 1 - i];
                if (extent > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
                    throw std::invalid_argument("direct parameter exceeds Tensor extent");
                }
                axes[i] = static_cast<std::int32_t>(extent);
            }
            return weight_tensor(view, {axes[0], axes[1], axes[2], axes[3]});
        });
    }

    DenseParameters dense(const DenseWeights& w) const {
        return {with_context(
                    weight(w.gate).name,
                    [&] { return ops::prepare_linear_swiglu_weight(input(w.gate), input(w.up)); }),
                linear(w.down)};
    }

    // A Qwen3.5 block as the Text forward runs it: plain residual, and the 27B's fused Ops for
    // every projection (they dispatch on the bound forms themselves).
    Qwen4BlockParameters qwen35_block(const BlockWeights& w) const {
        Qwen4BlockParameters out;
        out.input_norm          = tensor(w.input_norm);
        out.post_attention_norm = tensor(w.post_attention_norm);
        out.ffn                 = dense(w.ffn);
        out.fused               = Qwen4FusedRoutes{
                          .gdn_input = true, .gdn_gating = true, .attn_input = true, .swiglu = true};
        if (const auto* a = std::get_if<AttentionWeights>(&w.mixer)) {
            out.mixer = AttentionParameters{attention_projection(*a), tensor(a->query_norm),
                                            tensor(a->key_norm), linear(a->output)};
        } else {
            const auto& g = std::get<GdnWeights>(w.mixer);
            out.mixer     = GdnParameters{.projection  = gdn_projection(g),
                                          .control     = gdn_control(g),
                                          .a_log       = tensor(g.a_log),
                                          .dt_bias     = tensor(g.dt_bias),
                                          .convolution = tensor(g.convolution),
                                          .norm        = tensor(g.norm),
                                          .output      = linear(g.output)};
        }
        return out;
    }

    MtpParameters mtp(const MtpWeights& w) const {
        const auto& a = std::get<AttentionWeights>(w.layer.mixer);
        const std::array inputs{input(a.query), input(a.key), input(a.gate), input(a.value)};
        MtpParameters out;
        out.input_projection    = linear(w.input_projection);
        out.embedding_norm      = tensor(w.embedding_norm);
        out.hidden_norm         = tensor(w.hidden_norm);
        out.input_norm          = tensor(w.layer.input_norm);
        out.post_attention_norm = tensor(w.layer.post_attention_norm);
        out.final_norm          = tensor(w.final_norm);
        out.projection.packed   = ops::prepare_linear_weight(inputs);
        out.projection.rows     = {linear(a.query), linear(a.key), linear(a.gate), linear(a.value)};
        out.query_norm          = tensor(a.query_norm);
        out.key_norm            = tensor(a.key_norm);
        out.output              = linear(a.output);
        out.ffn                 = dense(w.layer.ffn);
        if (resident(w.output_head)) { out.output_head = linear(w.output_head_use); }
        return out;
    }

    NormParameters norm(const NormWeights& w) const { return {tensor(w.weight), tensor(w.bias)}; }

    std::optional<Tensor> joined_bias(const std::array<WeightId, 3>& ids) const {
        WeightView view;
        std::uint64_t count = 0;
        for (const auto id : ids) {
            const auto& input = weight(id).view;
            count += weight_element_count(input.shape);
            for (const auto& part : input.parts) {
                if (!view.parts.empty() && (view.parts.back().parent != part.parent ||
                                            view.parts.back().end != part.begin)) {
                    return std::nullopt;
                }
                view.parts.push_back(part);
            }
        }
        if (count > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument("Vision bias exceeds Tensor extent");
        }
        view.shape = {count};
        return weight_tensor(view, {static_cast<std::int32_t>(count)});
    }

    VisionParameters vision(const VisionWeights& w) const {
        VisionParameters out;
        out.patch_embedding      = linear(w.patch_embedding);
        out.patch_embedding_bias = tensor(w.patch_embedding_bias);
        out.position_embedding   = tensor(w.position_embedding);
        out.layers.reserve(w.layers.size());
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(with_context("vision/layers/" + std::to_string(i), [&] {
                const auto& layer = w.layers[i];
                const std::array qkv{input(layer.query), input(layer.key), input(layer.value)};
                const std::array ids{layer.query_bias, layer.key_bias, layer.value_bias};
                const auto bias = joined_bias(ids);
                if (!bias) {
                    throw std::invalid_argument(
                        "Vision QKV bias: this fixed call requires a contiguous bias bank");
                }
                return VisionBlockParameters{norm(layer.norm1),
                                             norm(layer.norm2),
                                             ops::prepare_linear_weight(qkv),
                                             *bias,
                                             linear(layer.output),
                                             linear(layer.fc1),
                                             linear(layer.fc2),
                                             tensor(layer.output_bias),
                                             tensor(layer.fc1_bias),
                                             tensor(layer.fc2_bias)};
            }));
        }
        out.merger_norm     = norm(w.merger_norm);
        out.merger_fc1      = linear(w.merger_fc1);
        out.merger_fc2      = linear(w.merger_fc2);
        out.merger_fc1_bias = tensor(w.merger_fc1_bias);
        out.merger_fc2_bias = tensor(w.merger_fc2_bias);
        return out;
    }

    DynamicConvParameters convolution(const DynamicConvWeights& w) const {
        return {tensor(w.base_kernel), linear(w.kernel_projection)};
    }

    DraftParameters draft(const DraftWeights& w) const {
        DraftParameters out;
        out.feature_projection = linear(w.feature_projection);
        out.context_norm       = tensor(w.context_norm);
        out.final_norm         = tensor(w.final_norm);
        out.output_head        = linear(w.output_head_use);
        out.layers.reserve(w.layers.size());
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(
                with_context(std::string(model_.options().speculative_component()) + "/layers/" +
                                 std::to_string(i),
                             [&] {
                                 const auto& layer = w.layers[i];
                                 const auto& a     = layer.attention;
                                 DraftBlockParameters result;
                                 result.input_norm          = tensor(layer.input_norm);
                                 result.post_attention_norm = tensor(layer.post_attention_norm);
                                 result.query_key_value     = ops::prepare_attn_input_proj_weights(
                                     input(a.query), input(a.key), input(a.value));
                                 result.context_key   = linear(a.context_key);
                                 result.context_value = linear(a.context_value);
                                 result.query_norm    = tensor(a.query_norm);
                                 result.key_norm      = tensor(a.key_norm);
                                 result.output        = linear(a.output);
                                 result.mlp           = dense(layer.mlp);
                                 if (layer.attention_conv) {
                                     result.attention_conv = convolution(*layer.attention_conv);
                                 }
                                 if (layer.mlp_conv) {
                                     result.mlp_conv = convolution(*layer.mlp_conv);
                                 }
                                 return result;
                             }));
        }
        // At tp 2 the split drafter's selector stays on rank 0 (load/sharding.h).
        if (w.selector && resident(w.selector->hidden_projection)) {
            out.selector = SelectorParameters{linear(w.selector->hidden_projection),
                                              tensor(w.selector->predecessor_codebook),
                                              tensor(w.selector->successor_codebook)};
        }
        return out;
    }

    // ---- Qwen4Exp ---------------------------------------------------------------------------

    // Rows of several direct BF16 parameters that sit consecutively in one parent.
    Tensor joined_matrix(std::initializer_list<WeightId> ids) const {
        WeightView view;
        std::uint64_t rows = 0, columns = 0;
        for (const auto id : ids) {
            const auto& input = weight(id).view;
            if (input.shape.size() != 2 || (columns != 0 && input.shape[1] != columns)) {
                throw std::invalid_argument("joined matrix rows must share one width");
            }
            columns = input.shape[1];
            rows += input.shape[0];
            for (const auto& part : input.parts) {
                if (part.parent->geometry.format != QType::BF16 ||
                    (!view.parts.empty() && (view.parts.back().parent != part.parent ||
                                             view.parts.back().end != part.begin))) {
                    throw std::invalid_argument("joined matrix is not one contiguous BF16 region");
                }
                view.parts.push_back(part);
            }
        }
        view.shape = {rows, columns};
        return weight_tensor(view, {dimension(columns), dimension(rows)});
    }

    HyperConnectionParameters hyper_connection(const HyperConnectionWeights& w) const {
        HyperConnectionParameters out;
        out.weights.norm = tensor(w.norm);
        if (w.inject) {
            out.weights.down =
                ops::prepare_linear_weight(std::array{input(w.down), input(*w.inject)}).weight;
            out.inject = true;
        } else {
            out.weights.down = linear(w.down).weight;
        }
        out.weights.up = linear(w.up).weight;
        return out;
    }

    // `mtp`: the MTP layer's bank, which may keep one weight divisor per bank (ninfer-ext's
    // artifact; reported through `uniform`) and may have no calibrated activation divisors (left
    // 0 here; Parameters derives them from the Text layers).
    struct UniformDivisors {
        float gate_up = 0.0F;
        float down    = 0.0F;
    };

    // A layer bank of ggml blocks (gguf_blocks_v1): the expert-major row order was checked by the
    // caller; here the formats the GGUF expert Ops support.
    static Qwen4ExpertBank gguf_bank(const WeightParent& gate_up, const WeightParent& down) {
        const auto format = [](QType q) -> std::optional<ops::MoeGgufFormat> {
            switch (q) {
            case QType::GGUF_IQ3_XXS:
                return ops::MoeGgufFormat::IQ3_XXS;
            case QType::GGUF_IQ4_NL:
                return ops::MoeGgufFormat::IQ4_NL;
            case QType::GGUF_IQ3_S:
                return ops::MoeGgufFormat::IQ3_S;
            case QType::GGUF_IQ2_S:
                return ops::MoeGgufFormat::IQ2_S;
            case QType::GGUF_IQ4_XS:
                return ops::MoeGgufFormat::IQ4_XS;
            case QType::GGUF_Q2_0:
                return ops::MoeGgufFormat::Q2_0;
            default:
                return std::nullopt;
            }
        };
        const auto gu = format(gate_up.geometry.format);
        const auto dn = format(down.geometry.format);
        if (gate_up.geometry.layout != QuantLayout::GgufBlocks ||
            down.geometry.layout != QuantLayout::GgufBlocks || !gu || !dn ||
            !ops::moe_gguf_gate_up_supported(*gu) || !ops::moe_gguf_down_supported(*dn)) {
            throw std::invalid_argument(
                "GGUF routed experts: gate/up IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS or IQ4_NL and down "
                "IQ4_NL or Q2_0, both gguf_blocks_v1");
        }
        Qwen4ExpertBank out;
        out.gguf                   = true;
        out.gguf_gate_up           = *gu;
        out.gguf_down              = *dn;
        out.gguf_gate_up_rows      = gate_up.data;
        out.gguf_down_rows         = down.data;
        out.gguf_gate_up_row_bytes = static_cast<std::int64_t>(gate_up.geometry.code_bytes_per_row);
        out.gguf_down_row_bytes    = static_cast<std::int64_t>(down.geometry.code_bytes_per_row);
        if (out.gguf_gate_up_row_bytes != ops::moe_gguf_row_bytes(*gu, 2560) ||
            out.gguf_down_row_bytes != ops::moe_gguf_row_bytes(*dn, 640)) {
            throw std::invalid_argument("GGUF routed experts: row bytes differ from the format");
        }
        return out;
    }

    Qwen4ExpertBank expert_bank(const MoeWeights& moe, bool mtp = false,
                                UniformDivisors* uniform = nullptr) const {
        const auto& config = model_.config().text;
        const auto& geo    = std::get<MoeConfig>(config.ffn);
        if (geo.num_experts != 512 || geo.num_experts_per_tok != ops::kMoeSlotsTopK ||
            geo.moe_intermediate_size != ops::kMoeSlotsIntermediate ||
            config.hidden_size != ops::kMoeSlotsHidden) {
            throw std::invalid_argument(
                "routed experts implement the 2560/512/top-10/640 geometry");
        }
        const auto single = [&](WeightId id) -> const WeightRegion& {
            const auto& parts = weight(id).view.parts;
            if (parts.size() != 1) {
                throw std::invalid_argument("a routed expert matrix must be one bank region");
            }
            return parts.front();
        };
        const WeightParent* gate_up = single(moe.experts.front().gate).parent;
        const WeightParent* down    = single(moe.experts.front().down).parent;
        const std::uint64_t h = config.hidden_size, ir = geo.moe_intermediate_size;
        for (std::uint32_t e = 0; e < geo.num_experts; ++e) {
            const auto& expert = moe.experts[e];
            const auto& g      = single(expert.gate);
            const auto& u      = single(expert.up);
            const auto& d      = single(expert.down);
            if (g.parent != gate_up || u.parent != gate_up || d.parent != down ||
                g.begin != (2 * e) * ir * h || u.begin != (2 * e + 1) * ir * h ||
                d.begin != e * h * ir) {
                throw std::invalid_argument("routed experts are not stored expert-major");
            }
        }
        if (gate_up->geometry.layout == QuantLayout::GgufBlocks ||
            down->geometry.layout == QuantLayout::GgufBlocks) {
            return gguf_bank(*gate_up, *down);
        }
        for (const auto* parent : {gate_up, down}) {
            if (parent->geometry.format != QType::NVFP4 ||
                parent->geometry.layout != QuantLayout::BlockScaleK16M128x4) {
                throw std::invalid_argument("routed experts must be block-scaled NVFP4 banks");
            }
        }
        const bool stacked = gate_up->geometry.divisor_count == 2ULL * geo.num_experts &&
                             down->geometry.divisor_count == geo.num_experts;
        const bool bank_wide =
            mtp && gate_up->geometry.divisor_count == 1 && down->geometry.divisor_count == 1;
        if (!stacked && !bank_wide) {
            throw std::invalid_argument(
                "routed expert banks must keep one weight divisor per matrix (stacked planes)");
        }
        Qwen4ExpertBank out;
        out.bank.gate_up_codes  = {gate_up->data, ops::kMoeSlotsGateUpCodeBytes};
        out.bank.gate_up_scales = {gate_up->data + gate_up->geometry.scale_offset,
                                   ops::kMoeSlotsGateUpScaleBytes};
        out.bank.down_codes     = {down->data, ops::kMoeSlotsDownCodeBytes};
        out.bank.down_scales    = {down->data + down->geometry.scale_offset,
                                   ops::kMoeSlotsDownScaleBytes};
        out.bank.slots          = static_cast<std::int32_t>(geo.num_experts);
        out.gate_up_divisors =
            reinterpret_cast<const float*>(gate_up->data + gate_up->geometry.divisor_offset);
        out.down_divisors =
            reinterpret_cast<const float*>(down->data + down->geometry.divisor_offset);
        if (bank_wide) {
            // Host-mapped words, readable here; the Program expands them into per-expert planes.
            if (uniform == nullptr || !(*out.gate_up_divisors > 0.0F) ||
                !(*out.down_divisors > 0.0F)) {
                throw std::invalid_argument("MTP expert bank divisors are invalid");
            }
            uniform->gate_up     = *out.gate_up_divisors;
            uniform->down        = *out.down_divisors;
            out.gate_up_divisors = nullptr;
            out.down_divisors    = nullptr;
        }
        if (mtp) {
            // Calibrated activation divisors stored with the bank (AllowA4 uses) are used like the
            // Text layers'; the checkpoint has none for MTP, so its banks are A16Only and the
            // divisors are derived later (Qwen4MtpParameters).
            bool calibrated = true;
            for (const auto& expert : moe.experts) {
                for (const WeightId id : {expert.gate, expert.up, expert.down}) {
                    const ops::WeightInput use = input(id);
                    calibrated = calibrated && use.policy == ops::LinearPolicy::AllowA4 &&
                                 use.activation_input_divisor &&
                                 *use.activation_input_divisor > 0.0F;
                }
            }
            if (!calibrated) { return out; }
        }
        // moe_slots_nvfp4 quantizes each input once per call with one activation divisor per
        // layer and projection input. ModelOpt calibrates the same input_scale for every expert of
        // a layer in the real checkpoint (T15), so this is exact there; where experts differ (the
        // synthetic checkpoints) the smallest divisor, i.e. the largest calibrated range, is used
        // so no activation saturates its E4M3 block scale.
        const auto smallest = [&](auto projections, const char* label) {
            float smallest_divisor = std::numeric_limits<float>::infinity();
            for (const auto& expert : moe.experts) {
                for (const WeightId id : projections(expert)) {
                    const ops::WeightInput use = input(id);
                    if (use.policy != ops::LinearPolicy::AllowA4 || !use.activation_input_divisor ||
                        !(*use.activation_input_divisor > 0.0F)) {
                        throw std::invalid_argument(
                            std::string("routed expert ") + label +
                            " inputs must permit NVFP4 activations with a calibrated divisor");
                    }
                    smallest_divisor = std::min(smallest_divisor, *use.activation_input_divisor);
                }
            }
            return smallest_divisor;
        };
        out.gate_up_input_divisor =
            smallest([](const DenseWeights& expert) { return std::array{expert.gate, expert.up}; },
                     "gate/up");
        out.down_input_divisor =
            smallest([](const DenseWeights& expert) { return std::array{expert.down}; }, "down");
        return out;
    }

    Qwen4MoeParameters qwen4_moe(const MoeWeights& w, bool mtp = false,
                                 UniformDivisors* uniform = nullptr) const {
        Qwen4MoeParameters out;
        out.router = joined_matrix({w.router, w.shared_score});
        if (out.router.ne[0] != ops::kMoeRouteHidden ||
            out.router.ne[1] != ops::kMoeRouteExperts + 1) {
            throw std::invalid_argument("router must be BF16 [513, 2560]");
        }
        if (resident(w.shared.gate)) {
            out.shared = Qwen4SharedExpertParameters{
                ops::prepare_linear_weight(std::array{input(w.shared.gate), input(w.shared.up)}),
                linear(w.shared.down)};
        }
        out.experts = expert_bank(w, mtp, uniform);
        return out;
    }

    // The FP32 scalar of an NVFP4 n-gram table, read back once for the host decode.
    float global_scale(WeightId id) const {
        const auto& bound = weight(id);
        return with_context(bound.name, [&] {
            const Tensor scalar = tensor(id);
            if (scalar.dtype != DType::FP32 || scalar.numel() != 1) {
                throw std::invalid_argument("PLE table global scale must be one FP32 value");
            }
            float value = 0.0F;
            CUDA_CHECK(cudaMemcpy(&value, scalar.data, sizeof(value), cudaMemcpyDefault));
            if (!std::isfinite(value) || value <= 0.0F) {
                throw std::invalid_argument("PLE table global scale must be finite and positive");
            }
            return value;
        });
    }

    PleParameters ple(const PleWeights& w) const {
        const auto& config = *model_.config().text.ple;
        const auto& table  = model_.file_weight(w.table);
        const auto& geo    = table.mapped_geometry;
        const bool fp8 =
            geo.format == QType::FP8_E4M3FN_ROW_BF16 && geo.layout == QuantLayout::RowScale;
        const bool nvfp4 =
            geo.format == QType::NVFP4_ROW_E4M3_G16 && geo.layout == QuantLayout::RowInterleaved;
        if (!(fp8 || nvfp4) || geo.shape.size() != 2) {
            throw std::invalid_argument("PLE table must be a file-mapped row-scaled FP8 or "
                                        "row-interleaved NVFP4 matrix");
        }
        if (nvfp4 == (w.table_global_scale == WeightId{})) {
            throw std::invalid_argument(
                "PLE table global scale must be bound for an NVFP4 table and only for it");
        }
        PleParameters out;
        for (const auto& segment : table.mapped) {
            out.table.segments.push_back({.object_offset = segment.object_offset,
                                          .bytes         = segment.bytes,
                                          .data          = segment.data,
                                          .file          = segment.file,
                                          .file_offset   = segment.file_offset});
        }
        out.table.rows      = geo.shape[0];
        out.table.row_width = static_cast<std::uint32_t>(geo.shape[1]);
        if (fp8) {
            out.table.format      = PleRowFormat::Fp8RowBf16;
            out.table.row_bytes   = geo.code_bytes_per_row;
            out.table.scale_plane = geo.scale_offset;
        } else {
            // row_interleaved_v1: each row's codes then its group scales, rows back to back.
            out.table.format       = PleRowFormat::Nvfp4RowE4m3G16;
            out.table.row_bytes    = out.table.row_record_bytes();
            out.table.global_scale = global_scale(w.table_global_scale);
            if (out.table.row_width % 16 != 0 || geo.bytes < out.table.rows * out.table.row_bytes) {
                throw std::invalid_argument("PLE NVFP4 table geometry is inconsistent");
            }
        }
        out.hash.ngram_size      = config.ngram_size;
        out.hash.heads_per_ngram = config.heads_per_ngram;
        out.hash.eos_token_id    = config.eos_token_id;
        out.hash.multipliers     = config.multipliers;
        out.hash.head_moduli     = config.head_moduli;
        out.hash.head_offsets    = config.head_offsets;
        if (out.hash.heads() * out.table.row_width != config.embed_dim ||
            out.table.rows != config.table_rows) {
            throw std::invalid_argument("PLE table geometry does not match the configuration");
        }
        out.key_value   = ops::prepare_linear_weight(std::array{input(w.key), input(w.value)});
        out.key_norm    = tensor(w.key_norm);
        out.query_norm  = tensor(w.query_norm);
        out.conv_norm   = tensor(w.conv_norm);
        out.convolution = tensor(w.convolution);
        return out;
    }

    Qwen4BlockParameters qwen4_block(const Qwen4BlockWeights& w, bool mtp = false,
                                     UniformDivisors* uniform = nullptr) const {
        Qwen4BlockParameters out;
        out.attention_hc = hyper_connection(w.attention_hc);
        out.ffn_hc       = hyper_connection(w.ffn_hc);
        if (const auto* q = std::get_if<QsaWeights>(&w.mixer)) {
            const auto& a = q->attention;
            out.mixer =
                QsaParameters{ops::prepare_linear_weight(std::array{input(a.query), input(a.key),
                                                                    input(a.gate), input(a.value)}),
                              tensor(a.query_norm),
                              tensor(a.key_norm),
                              linear(a.output),
                              ops::prepare_linear_weight(
                                  std::array{input(q->indexer_query), input(q->indexer_key)}),
                              tensor(q->indexer_query_norm),
                              tensor(q->indexer_key_norm)};
        } else {
            const auto& g = std::get<GdnWeights>(w.mixer);
            out.mixer =
                GdnParameters{ops::prepare_linear_weight(std::array{input(g.query), input(g.key),
                                                                    input(g.value), input(g.z)}),
                              ops::prepare_linear_weight(
                                  std::array{input(g.a_projection), input(g.b_projection)}),
                              tensor(g.a_log),
                              tensor(g.dt_bias),
                              tensor(g.convolution),
                              tensor(g.norm),
                              linear(g.output)};
        }
        if (const auto* moe = std::get_if<MoeWeights>(&w.ffn)) {
            out.ffn = qwen4_moe(*moe, mtp, uniform);
        } else {
            out.ffn = dense(std::get<DenseWeights>(w.ffn));
        }
        if (w.ple) { out.ple = ple(*w.ple); }
        out.fused = qwen4_routes(out);
        return out;
    }

    // Shape-keyed fused routes of a Qwen4Exp block on this rank (Qwen4FusedRoutes): the 27B's
    // fused Ops answer for their registered geometries and formats (e.g. hidden 5120, 24 Q / 4 KV
    // heads of 256, GDN 16 K / 48 V heads of 128, NVFP4/FP8 parents); every other geometry,
    // Qwen3.8-Flash-Next's hidden 2560 included, keeps the generic chain. At tp 2 the routes are
    // the two-device split forms at this rank's shard shapes.
    Qwen4FusedRoutes qwen4_routes(const Qwen4BlockParameters& b) const {
        Qwen4FusedRoutes out;
        if (!qwen4_fused_enabled()) { return out; }
        const int last = kRouteQueryColumns;
        if (const auto* g = std::get_if<GdnParameters>(&b.mixer)) {
            if (const auto* p = std::get_if<LinearParameters>(&g->projection)) {
                const Weight& w = p->weight;
                out.gdn_input   = registered([&] {
                    if (sharded_) {
                        (void)ops::gdn_input_proj_column_parallel_workspace_capacity_bytes(
                            w.qtype, w.n, w.k, p->policy, 1, last);
                    } else {
                        (void)ops::gdn_input_proj_workspace_capacity_bytes(w.qtype, w.n, w.k,
                                                                             p->policy, 1, last);
                    }
                    for (const std::int32_t batch : {1, 8}) {
                        if (sharded_) {
                            (void)ops::
                                gdn_input_proj_conv_snapshot_column_parallel_workspace_capacity_bytes(
                                    w.qtype, w.n, w.k, p->policy, batch, 1, 16);
                            (void)ops::
                                gdn_input_proj_conv_record_column_parallel_workspace_capacity_bytes(
                                    w.qtype, w.n, w.k, p->policy, batch, 2, 16);
                        } else {
                            (void)ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                                w.qtype, w.n, w.k, p->policy, batch, 1, 16);
                            (void)ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                                w.qtype, w.n, w.k, p->policy, batch, 2, 16);
                        }
                    }
                });
            }
            if (const auto* c = std::get_if<LinearParameters>(&g->control);
                c != nullptr && c->weight.qtype == QType::BF16 && c->weight.n % 2 == 0) {
                const Weight& w = c->weight;
                out.gdn_gating  = registered([&] {
                    if (sharded_) {
                        (void)ops::gdn_gating_proj_column_parallel_workspace_capacity_bytes(
                            w.n / 2, w.k, 1, last);
                    } else {
                        (void)ops::gdn_gating_proj_workspace_capacity_bytes(w.n / 2, w.k, 1, last);
                    }
                });
            }
        }
        if (const auto* q = std::get_if<QsaParameters>(&b.mixer)) {
            const Weight& w     = q->projection.weight;
            const auto& a       = *model_.config().text.attention;
            const auto width    = model_.device_count() > 1 ? 2U : 1U;
            const auto expected = 2 * (a.query_width() + a.key_width()) / width;
            // The registered parents hold q|k|gate|v rows of 24 query / 4 KV heads of 256 (one
            // rank's half at tp 2) over hidden 5120.
            const bool shape = static_cast<std::uint64_t>(w.n) == expected &&
                               w.n == (sharded_ ? 7168 : 14336) && w.k == 5120 &&
                               a.head_dim == 256 && a.num_attention_heads == 24 &&
                               a.num_key_value_heads == 4;
            out.attn_input =
                shape && registered([&] {
                    if (sharded_) {
                        if (w.qtype != QType::NVFP4 && w.qtype != QType::FP8_E4M3FN_ROW_BF16) {
                            throw std::invalid_argument("no split route");
                        }
                        (void)ops::attn_input_proj_column_parallel_workspace_capacity_bytes(
                            w.qtype, q->projection.policy, 1, last);
                    } else {
                        (void)ops::attn_input_proj_workspace_capacity_bytes(
                            w.qtype, w.n, w.k, q->projection.policy, 1, last);
                    }
                });
        }
        if (const auto* d = std::get_if<DenseParameters>(&b.ffn)) {
            const Weight& w = d->gate_up.weight;
            out.swiglu      = ops::linear_swiglu_registered(w.qtype, w.n, w.k) && registered([&] {
                             (void)ops::linear_swiglu_workspace_capacity_bytes(
                                 w.qtype, w.n, w.k, d->gate_up.policy, 1, last);
                         });
        }
        return out;
    }

    Qwen4MtpParameters qwen4_mtp(const Qwen4MtpWeights& w) const {
        Qwen4MtpParameters out;
        out.embedding_norm       = tensor(w.embedding_norm);
        out.hidden_norm          = tensor(w.hidden_norm);
        out.embedding_projection = linear(w.embedding_projection);
        out.hidden_projection    = linear(w.hidden_projection);
        out.head                 = hyper_connection(w.head);
        if (out.head.inject || w.layer.ple || !std::holds_alternative<QsaWeights>(w.layer.mixer)) {
            throw std::invalid_argument("MTP layer must be one QSA block without PLE");
        }
        UniformDivisors uniform;
        out.layer                   = qwen4_block(w.layer, true, &uniform);
        out.uniform_gate_up_divisor = uniform.gate_up;
        out.uniform_down_divisor    = uniform.down;
        return out;
    }

private:
    const Model& model_;
    int device_   = 0;
    bool sharded_ = false;
};

} // namespace

Parameters::Parameters(const Model& source, int rank) : model(source), device(rank) {
    const Prepare prepare(model, device);
    const auto& w = model.weights();
    if (model.config().text.qwen4()) {
        // Qwen4Exp Text at one device or at tp 2 (this rank's shards), its MTP layer (T24) and the
        // Vision tower on the rank that holds it (the Qwen3.5 tower; its merger widens to H).
        // Draft and proposal heads have no Qwen4Exp execution; loading and the Program refuse
        // them.
        if (model.device_count() > 2 || !w.qwen4) {
            throw std::invalid_argument(
                "Qwen4ExpForCausalLM executes on one or two devices with its Text weights bound");
        }
        text.token_embedding = native_weight(prepare.weight(w.text.token_embedding).view);
        text.output_head     = prepare.linear(w.text.output_head_use);
        Qwen4Parameters q;
        q.head =
            with_context("text/hc_head", [&] { return prepare.hyper_connection(w.qwen4->head); });
        q.layers.reserve(w.qwen4->layers.size());
        for (std::size_t i = 0; i < w.qwen4->layers.size(); ++i) {
            q.layers.push_back(with_context("text/layers/" + std::to_string(i), [&] {
                return prepare.qwen4_block(w.qwen4->layers[i]);
            }));
        }
        qwen4 = std::move(q);
        if (w.qwen4_mtp) {
            qwen4_mtp = with_context("mtp", [&] { return prepare.qwen4_mtp(*w.qwen4_mtp); });
        }
        if (qwen4_mtp && model.config().text.qwen4_routed_experts()) {
            auto& bank = qwen4_mtp->layer.moe().experts;
            if (!bank.gguf &&
                (!(bank.gate_up_input_divisor > 0.0F) || !(bank.down_input_divisor > 0.0F))) {
                // No calibrated activation divisors for the MTP experts: the widest calibrated
                // range of the Text layers (smallest divisor), widened by the margin.
                float gate_up = std::numeric_limits<float>::infinity();
                float down    = std::numeric_limits<float>::infinity();
                for (const auto& layer : qwen4->layers) {
                    gate_up = std::min(gate_up, layer.moe().experts.gate_up_input_divisor);
                    down    = std::min(down, layer.moe().experts.down_input_divisor);
                }
                bank.gate_up_input_divisor             = gate_up / kQwen4MtpActivationRangeMargin;
                bank.down_input_divisor                = down / kQwen4MtpActivationRangeMargin;
                qwen4_mtp->derived_activation_divisors = true;
            }
        }
        if (w.vision && prepare.resident(w.vision->patch_embedding)) {
            vision = with_context("vision", [&] { return prepare.vision(*w.vision); });
        }
        return;
    }
    text.token_embedding = native_weight(prepare.weight(w.text.token_embedding).view);
    text.output_head     = prepare.linear(w.text.output_head_use);
    if (w.mtp) {
        mtp = with_context("mtp", [&] { return prepare.mtp(*w.mtp); });
    }
    // A dense Qwen3.5 model runs its Text layers through the Qwen4 forward with the plain
    // residual. The MTP head stays the Qwen3.5 one (`mtp`).
    Qwen4Parameters q;
    q.final_norm = prepare.tensor(w.text.final_norm);
    q.layers.reserve(w.text.layers.size());
    for (std::size_t i = 0; i < w.text.layers.size(); ++i) {
        q.layers.push_back(with_context("text/layers/" + std::to_string(i),
                                        [&] { return prepare.qwen35_block(w.text.layers[i]); }));
    }
    qwen4 = std::move(q);
    if (w.vision && prepare.resident(w.vision->patch_embedding)) {
        vision = with_context("vision", [&] { return prepare.vision(*w.vision); });
    }
    if (w.draft && prepare.resident(w.draft->feature_projection)) {
        draft = with_context(std::string(model.options().speculative_component()),
                             [&] { return prepare.draft(*w.draft); });
    }
    if (w.proposal && prepare.resident(w.proposal->head)) {
        proposal =
            ProposalParameters{prepare.linear(w.proposal->head), std::nullopt, w.proposal->rows};
        if (w.proposal->token_ids && prepare.resident(*w.proposal->token_ids)) {
            proposal->token_ids = prepare.tensor(*w.proposal->token_ids);
        }
        if (proposal->split()) {
            // A format without the half's route would fail only at the first proposal.
            const auto& head = proposal->head;
            try {
                (void)ops::linear_workspace_capacity_bytes(head.weight.qtype, head.weight.n,
                                                           head.weight.k, head.policy, 1, 1);
            } catch (const std::invalid_argument& error) {
                throw std::invalid_argument(
                    "proposal/head: the tensor-parallel half [" + std::to_string(head.weight.n) +
                    "," + std::to_string(head.weight.k) + "] has no registered linear route (" +
                    error.what() + "); NINFER_TP_DRAFT_HEAD=primary keeps it whole on rank 0");
            }
        }
    }
}

} // namespace ninfer::models::qwen3_5::execution
