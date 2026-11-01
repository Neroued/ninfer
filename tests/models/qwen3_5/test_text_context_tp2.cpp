// Two-device tensor-parallel (tp == 2) forward of the Qwen3.5 Text backbone.
//
// Default mode: parity. A synthetic two-layer model (one Gated DeltaNet and one full-attention
// block) at the Qwen3.8 27B per-layer geometry -- the geometry every registered split Op shape
// requires -- is written as a v3 artifact with random FP8 row-scaled projections, loaded once at
// tp 1 on device 0 and once at tp 2 on devices 0 and 1, and driven through the same 64-token
// prompt (two prefill chunks) and four ordinary decode steps. The rank-0 logits of the split
// forward are compared with the single-device logits after every call. Last, the rank-0 logits
// gather runs alone over three columns and is checked byte for byte against the two partials, and
// a 512-column chunk that fails inside its sliced FFN must leave no all-reduce lane open: rerun on
// intact Parameters, the chunk reproduces its logits (failed_slice_closes_lanes).
//
// Criterion: the split forward differs from the single-device one only in summation order and in
// the BF16 rounding of each rank's partial before every all-reduce (four per token here: two per
// layer). Both runs store BF16 logits. Every column must agree within two BF16 ulps of the
// column's largest single-device magnitude. That is the per-projection bound of the split Op
// suites' A16 cases only: their FP8 A8 row-split cases allow a looser relative error and the
// head-local attention suite twice its oracle criterion, and this forward runs both, so the bound
// is an end-to-end expectation for this synthetic model rather than a composition of Op criteria.
// The logit gather itself is exact.
//
// `real` mode: consistency on NINFER_TEST_ARTIFACT, which does not fit one 16 GB device. The
// model is loaded at tp 2 only; a chat-formatted "What is 17*23?" is prefilled and decoded
// greedily with thinking off, asking for the number alone; every logit must be finite and the
// answer (at most 64 tokens) must contain "391".
//
// Both modes return 77 below two CUDA devices; `real` also returns 77 without the artifact.

#include "artifact/formats.h"
#include "artifact/reader.h"
#include "artifact/schema.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "core/weight_view.h"
#include "models/qwen3_5/execution/linear.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/qwen4_runtime.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/execution/tp.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/scalar.h"

#include <cuda_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace ninfer;
namespace qwen = ninfer::models::qwen3_5;
namespace exec = ninfer::models::qwen3_5::execution;
using Json     = artifact::Json;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

// ------------------------------------------------------------------------------------------------
// Synthetic artifact
// ------------------------------------------------------------------------------------------------

constexpr std::uint64_t kHidden       = 5120;
constexpr std::uint64_t kVocab        = 248320;
constexpr std::uint64_t kHeadDim      = 256;
constexpr std::uint64_t kQueryHeads   = 24;
constexpr std::uint64_t kKvHeads      = 4;
constexpr std::uint64_t kGdnKeyHeads  = 16;
constexpr std::uint64_t kGdnValHeads  = 48;
constexpr std::uint64_t kGdnHeadDim   = 128;
constexpr std::uint64_t kConvKernel   = 4;
constexpr std::uint64_t kIntermediate = 17408;

std::uint64_t splitmix(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z               = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z               = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
}

float uniform(std::uint64_t& state, float lo, float hi) {
    const auto unit = static_cast<float>(splitmix(state) >> 40U) / static_cast<float>(1ULL << 24U);
    return lo + (hi - lo) * unit;
}

std::uint16_t bf16_bits(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

float bf16_value(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

// Streams one single-file v3 artifact. Objects are declared first (offsets and the JSON header
// depend on every size), then their payload is generated in offset order straight to disk, so the
// multi-GiB synthetic model never exists in host memory.
class SyntheticArtifact {
public:
    enum class Fill : std::uint8_t { Fp8Rows, Bf16Uniform, Fp32Values, Bytes };

    struct Object {
        std::string id;
        std::uint64_t offset = 0;
        WeightGeometry geometry;
        Fill fill          = Fill::Bytes;
        std::uint64_t seed = 0;
        float lo = 0, hi = 0; // Bf16Uniform range; Fp8Rows uses `hi` as the row scale
        std::vector<float> values;
        std::string bytes;
    };

    explicit SyntheticArtifact(std::filesystem::path directory) : directory_(std::move(directory)) {
        std::filesystem::create_directories(directory_);
        root_ = {
            {"components", {{"text", {{"config", Json::object()}, {"resources", Json::object()}}}}},
            {"objects", Json::array()},
            {"bindings", Json::object()},
            {"uses", Json::array()},
            {"metadata", {{"name", "synthetic-tensor-parallel-parity"}}}};
    }

    ~SyntheticArtifact() {
        if (keep_) { return; }
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
    }

    // Leaves the written artifact on disk (the `write DIR` mode of this executable).
    void keep() noexcept { keep_ = true; }

    SyntheticArtifact(const SyntheticArtifact&)            = delete;
    SyntheticArtifact& operator=(const SyntheticArtifact&) = delete;

    Json& config() { return root_["components"]["text"]["config"]; }

    void fp8(const std::string& id, std::uint64_t rows, std::uint64_t columns, float scale) {
        auto& object = add(id, {rows, columns}, QType::FP8_E4M3FN_ROW_BF16, QuantLayout::RowScale);
        object.fill  = Fill::Fp8Rows;
        object.hi    = scale;
    }

    void bf16(const std::string& id, std::vector<std::uint64_t> shape, float lo, float hi) {
        auto& object = add(id, std::move(shape), QType::BF16, QuantLayout::Contiguous);
        object.fill  = Fill::Bf16Uniform;
        object.lo    = lo;
        object.hi    = hi;
    }

    void fp32(const std::string& id, std::vector<float> values) {
        auto& object  = add(id, {values.size()}, QType::FP32, QuantLayout::Contiguous);
        object.fill   = Fill::Fp32Values;
        object.values = std::move(values);
    }

    void resource(const std::string& role, std::string bytes) {
        Object object;
        object.id     = role;
        object.offset = align(cursor_);
        object.fill   = Fill::Bytes;
        cursor_       = object.offset + bytes.size();
        root_["objects"].push_back({{"id", role},
                                    {"kind", "resource"},
                                    {"encoding", std::string(artifact::kRawBytesEncoding)},
                                    {"offset", object.offset},
                                    {"bytes", bytes.size()}});
        root_["components"]["text"]["resources"][role] = role;
        object.bytes                                   = std::move(bytes);
        objects_.push_back(std::move(object));
    }

    // Binds `name` to `object`, whole or to rows [first, last) of its row-major element range.
    void bind(const std::string& name, const std::string& object,
              std::optional<std::array<std::uint64_t, 2>> rows = std::nullopt) {
        if (!rows) {
            root_["bindings"][name] = {{"object", object}};
            return;
        }
        const auto row_elements = find(object).geometry.shape.back();
        root_["bindings"][name] = {
            {"parts",
             Json::array({{{"object", object},
                           {"range", {(*rows)[0] * row_elements, (*rows)[1] * row_elements}}}})}};
    }

    void use(const std::string& name, const std::string& input) {
        root_["uses"].push_back(
            {{"parameter", name}, {"input", input}, {"activation_policy", "A16Only"}});
    }

    std::filesystem::path write() {
        const std::uint64_t payload = align(cursor_);
        root_["files"]         = Json::array({{{"path", nullptr}, {"payload_bytes", payload}}});
        const std::string text = root_.dump();
        const auto entry       = directory_ / "model.ninfer";
        std::ofstream file(entry, std::ios::binary | std::ios::trunc);
        file.exceptions(std::ios::badbit | std::ios::failbit);
        std::array<unsigned char, 32> header{};
        const std::array<unsigned char, 8> magic{'N', 'I', 'N', 'F', 'E', 'R', 0, 3};
        std::copy(magic.begin(), magic.end(), header.begin());
        for (unsigned i = 0; i < 8; ++i) {
            header[8 + i] = static_cast<unsigned char>((text.size() >> (8 * i)) & 255U);
        }
        header[16] = 0x71;
        file.write(reinterpret_cast<const char*>(header.data()), header.size());
        file.write(text.data(), static_cast<std::streamsize>(text.size()));
        const auto position = 32 + text.size();
        const std::string padding((position + 4095) / 4096 * 4096 - position, '\0');
        file.write(padding.data(), static_cast<std::streamsize>(padding.size()));

        std::vector<char> buffer;
        std::uint64_t written = 0;
        const auto flush      = [&] {
            file.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            written += buffer.size();
            buffer.clear();
        };
        const auto put = [&](const void* data, std::size_t bytes) {
            const auto* begin = static_cast<const char*>(data);
            buffer.insert(buffer.end(), begin, begin + bytes);
            if (buffer.size() >= (std::size_t{1} << 24U)) { flush(); }
        };
        const auto zeros = [&](std::uint64_t until) {
            const char zero = 0;
            while (written + buffer.size() < until) { put(&zero, 1); }
        };
        for (const auto& object : objects_) {
            zeros(object.offset);
            generate(object, put, zeros);
        }
        zeros(payload);
        flush();
        return entry;
    }

private:
    static std::uint64_t align(std::uint64_t value) { return (value + 255) / 256 * 256; }

    Object& add(const std::string& id, std::vector<std::uint64_t> shape, QType format,
                QuantLayout layout) {
        Object object;
        object.id       = id;
        object.offset   = align(cursor_);
        object.geometry = weight_geometry(format, layout, shape);
        object.seed     = 0x5eedULL + objects_.size() * 0x1000193ULL;
        cursor_         = object.offset + object.geometry.bytes;
        root_["objects"].push_back({{"id", id},
                                    {"kind", "tensor"},
                                    {"shape", shape},
                                    {"format", std::string(artifact::format_name(format))},
                                    {"layout", std::string(artifact::layout_name(layout))},
                                    {"offset", object.offset},
                                    {"bytes", object.geometry.bytes}});
        objects_.push_back(std::move(object));
        return objects_.back();
    }

    const Object& find(const std::string& id) const {
        for (const auto& object : objects_) {
            if (object.id == id) { return object; }
        }
        throw std::logic_error("synthetic artifact has no object " + id);
    }

    template <class Put, class Zeros>
    static void generate(const Object& object, Put& put, Zeros& zeros) {
        std::uint64_t state = object.seed;
        const auto& g       = object.geometry;
        switch (object.fill) {
        case Fill::Fp8Rows: {
            // Finite E4M3 codes with a random sign and magnitudes in [0, 1.875]; the row scale
            // brings a K-term dot product of unit activations to about unit size.
            std::uint64_t remaining = g.code_bytes;
            while (remaining != 0) {
                std::uint64_t word = splitmix(state);
                const auto count   = static_cast<unsigned>(std::min<std::uint64_t>(8, remaining));
                std::array<unsigned char, 8> codes{};
                for (unsigned i = 0; i < count; ++i) {
                    const auto byte = static_cast<unsigned char>(word & 255U);
                    codes[i]        = static_cast<unsigned char>((byte & 0x80U) | (byte & 0x3fU));
                    word >>= 8U;
                }
                put(codes.data(), count);
                remaining -= count;
            }
            zeros(object.offset + g.scale_offset);
            const std::uint64_t rows = g.shape.front();
            for (std::uint64_t row = 0; row < rows; ++row) {
                const std::uint16_t scale = bf16_bits(object.hi * uniform(state, 0.75F, 1.25F));
                put(&scale, sizeof(scale));
            }
            zeros(object.offset + g.bytes);
            break;
        }
        case Fill::Bf16Uniform:
            for (std::uint64_t i = 0; i < g.elements; ++i) {
                const std::uint16_t value = bf16_bits(uniform(state, object.lo, object.hi));
                put(&value, sizeof(value));
            }
            break;
        case Fill::Fp32Values:
            put(object.values.data(), object.values.size() * sizeof(float));
            break;
        case Fill::Bytes:
            put(object.bytes.data(), object.bytes.size());
            break;
        }
    }

    std::filesystem::path directory_;
    bool keep_ = false;
    Json root_;
    std::vector<Object> objects_;
    std::uint64_t cursor_ = 0;
};

// As in the official tokenizer, <think> and </think> are added but not special: the frontend
// requires </think> to survive a decode that skips special tokens.
Json added_token(int id, const std::string& content) {
    const bool special = content != "<think>" && content != "</think>";
    return {{"id", id},        {"content", content}, {"special", special}, {"single_word", false},
            {"lstrip", false}, {"rstrip", false},    {"normalized", false}};
}

// A byte-level BPE tokenizer with the Qwen specials: the frontend requires a valid tokenizer, and
// the synthetic prompt only needs byte ids.
void add_tokenizer(SyntheticArtifact& file) {
    Json vocab    = Json::object();
    unsigned next = 256;
    for (unsigned byte = 0; byte < 256; ++byte) {
        const bool visible =
            (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174;
        const unsigned cp = visible ? byte : next++;
        std::string token;
        if (cp < 128) {
            token.push_back(static_cast<char>(cp));
        } else {
            token.push_back(static_cast<char>(0xc0 | (cp >> 6)));
            token.push_back(static_cast<char>(0x80 | (cp & 63)));
        }
        vocab[token] = byte;
    }
    Json added                = Json::array();
    Json decoder              = Json::object();
    const std::array specials = {"<|endoftext|>",  "<|im_start|>",  "<|im_end|>",
                                 "<think>",        "</think>",      "<|vision_start|>",
                                 "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"};
    for (std::size_t i = 0; i < specials.size(); ++i) {
        const auto token = added_token(static_cast<int>(256 + i), specials[i]);
        if (i < 8) { added.push_back(token); }
        decoder[std::to_string(256 + i)] = token;
    }
    file.resource("tokenizer.json",
                  Json{{"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", Json::array()}}},
                       {"added_tokens", added}}
                      .dump());
    // The Engine frontend (tp 1 golden gate) also validates the Qwen prefix semantics and pad
    // token.
    file.resource("tokenizer_config.json", Json{{"added_tokens_decoder", decoder},
                                                {"add_bos_token", false},
                                                {"add_prefix_space", false},
                                                {"pad_token", "<|endoftext|>"}}
                                               .dump());
    file.resource("generation_config.json", Json{{"eos_token_id", {256, 258}}}.dump());
    file.resource("chat_template.jinja", "synthetic");
}

// Layer 0 is a Gated DeltaNet block, layer 1 a full-attention block; every projection is one
// contiguous FP8 parent in the section order the split Ops shard.
std::filesystem::path write_synthetic_model(SyntheticArtifact& file) {
    const float unit = 1.0F / (0.61F * std::sqrt(static_cast<float>(kHidden)));
    auto& config     = file.config();
    config           = {{"architectures", {"Qwen3_5ForCausalLM"}},
                        {"model_type", "qwen3_5_text"},
                        {"hidden_size", kHidden},
                        {"vocab_size", kVocab},
                        {"num_hidden_layers", 2},
                        {"max_position_embeddings", 262144},
                        {"tie_word_embeddings", false},
                        {"rms_norm_eps", 1e-6},
                        {"layer_types", {"linear_attention", "full_attention"}},
                        {"num_attention_heads", kQueryHeads},
                        {"num_key_value_heads", kKvHeads},
                        {"head_dim", kHeadDim},
                        {"rope_parameters",
                         {{"rope_theta", 10000000},
                          {"partial_rotary_factor", 0.25},
                          {"mrope_section", {11, 11, 10}}}},
                        {"linear_num_key_heads", kGdnKeyHeads},
                        {"linear_key_head_dim", kGdnHeadDim},
                        {"linear_num_value_heads", kGdnValHeads},
                        {"linear_value_head_dim", kGdnHeadDim},
                        {"linear_conv_kernel_dim", kConvKernel},
                        {"intermediate_size", kIntermediate}};
    add_tokenizer(file);

    file.fp8("embedding", kVocab, kHidden, 1.0F);
    file.bind("text/token_embedding", "embedding");
    file.fp8("head", kVocab, kHidden, 4.0F * unit);
    file.bind("text/output_head", "head");
    file.use("text/output_head", "text/final_hidden");
    file.bf16("final_norm", {kHidden}, -0.1F, 0.1F);
    file.bind("text/final_norm", "final_norm");

    const std::uint64_t q  = kQueryHeads * kHeadDim;
    const std::uint64_t kv = kKvHeads * kHeadDim;
    const std::uint64_t gk = kGdnKeyHeads * kGdnHeadDim;
    const std::uint64_t gv = kGdnValHeads * kGdnHeadDim;
    for (int layer = 0; layer < 2; ++layer) {
        const std::string p = "text/layers/" + std::to_string(layer) + "/";
        const std::string o = "l" + std::to_string(layer) + ".";
        for (const auto* norm : {"input_norm", "post_attention_norm"}) {
            file.bf16(o + norm, {kHidden}, -0.1F, 0.1F);
            file.bind(p + norm, o + norm);
        }
        if (layer == 0) {
            file.fp8(o + "qkvz", 2 * gk + 2 * gv, kHidden, unit);
            const std::array<std::pair<const char*, std::array<std::uint64_t, 2>>, 4> sections{
                {{"query", {0, gk}},
                 {"key", {gk, 2 * gk}},
                 {"value", {2 * gk, 2 * gk + gv}},
                 {"z", {2 * gk + gv, 2 * gk + 2 * gv}}}};
            for (const auto& [role, rows] : sections) {
                file.bind(p + "gdn/" + role, o + "qkvz", rows);
                file.use(p + "gdn/" + role, p + "mixer_input");
            }
            const float ab = std::sqrt(3.0F / static_cast<float>(kHidden));
            file.bf16(o + "ab", {2 * kGdnValHeads, kHidden}, -ab, ab);
            file.bind(p + "gdn/a_projection", o + "ab", std::array<std::uint64_t, 2>{0, 48});
            file.bind(p + "gdn/b_projection", o + "ab", std::array<std::uint64_t, 2>{48, 96});
            file.use(p + "gdn/a_projection", p + "mixer_input");
            file.use(p + "gdn/b_projection", p + "mixer_input");
            std::vector<float> a_log;
            std::vector<float> dt_bias;
            for (std::uint64_t h = 0; h < kGdnValHeads; ++h) {
                a_log.push_back(std::log(1.0F + static_cast<float>(h % 16)));
                dt_bias.push_back(-1.0F - 3.0F * static_cast<float>(h % 7) / 7.0F);
            }
            file.fp32(o + "a_log", std::move(a_log));
            file.bind(p + "gdn/a_log", o + "a_log");
            file.fp32(o + "dt_bias", std::move(dt_bias));
            file.bind(p + "gdn/dt_bias", o + "dt_bias");
            file.bf16(o + "convolution", {kConvKernel, 2 * gk + gv}, -0.5F, 0.5F);
            file.bind(p + "gdn/convolution", o + "convolution");
            file.bf16(o + "gdn_norm", {kGdnHeadDim}, 0.9F, 1.1F);
            file.bind(p + "gdn/norm", o + "gdn_norm");
            file.fp8(o + "gdn_output", kHidden, gv,
                     0.5F / (0.61F * std::sqrt(static_cast<float>(gv))));
            file.bind(p + "gdn/output", o + "gdn_output");
            file.use(p + "gdn/output", p + "gdn/gated_output");
        } else {
            file.fp8(o + "qkgv", 2 * q + 2 * kv, kHidden, unit);
            const std::array<std::pair<const char*, std::array<std::uint64_t, 2>>, 4> sections{
                {{"query", {0, q}},
                 {"key", {q, q + kv}},
                 {"gate", {q + kv, 2 * q + kv}},
                 {"value", {2 * q + kv, 2 * q + 2 * kv}}}};
            for (const auto& [role, rows] : sections) {
                file.bind(p + "attention/" + role, o + "qkgv", rows);
                file.use(p + "attention/" + role, p + "mixer_input");
            }
            for (const auto* norm : {"query_norm", "key_norm"}) {
                file.bf16(o + norm, {kHeadDim}, -0.1F, 0.1F);
                file.bind(p + "attention/" + norm, o + norm);
            }
            file.fp8(o + "attention_output", kHidden, q,
                     0.5F / (0.61F * std::sqrt(static_cast<float>(q))));
            file.bind(p + "attention/output", o + "attention_output");
            file.use(p + "attention/output", p + "attention/gated_output");
        }
        file.fp8(o + "gate_up", 2 * kIntermediate, kHidden, unit);
        file.bind(p + "mlp/gate", o + "gate_up", std::array<std::uint64_t, 2>{0, kIntermediate});
        file.bind(p + "mlp/up", o + "gate_up",
                  std::array<std::uint64_t, 2>{kIntermediate, 2 * kIntermediate});
        file.use(p + "mlp/gate", p + "ffn_input");
        file.use(p + "mlp/up", p + "ffn_input");
        file.fp8(o + "down", kHidden, kIntermediate,
                 0.5F / (0.61F * std::sqrt(static_cast<float>(kIntermediate))));
        file.bind(p + "mlp/down", o + "down");
        file.use(p + "mlp/down", p + "mlp/product");
    }
    return file.write();
}

// ------------------------------------------------------------------------------------------------
// Per-rank Program storage: what the Program owns for one device in production
// ------------------------------------------------------------------------------------------------

constexpr std::uint32_t kCapacity = 128;

DeviceBuffer device_bytes(int device, std::size_t bytes) {
    CUDA_CHECK(cudaSetDevice(device));
    DeviceBuffer buffer(bytes);
    buffer.fill(0);
    return buffer;
}

// One device's KV cache (one bound execution row), GDN state pool (one slot), RoundState and
// transient arena. `config` is the rank's share of the Text config; `logit_rows` is the complete
// vocabulary, which every RoundState holds (rank 0 alone gathers the logits into its copy).
struct RankStorage {
    DeviceBuffer decoder_bytes;
    std::optional<qwen::DecoderState> decoder;
    std::optional<DeviceKVPageReservation> reservation;
    std::vector<DeviceKVPageLease> pages;
    KVExecutionRowLease row;
    DeviceBuffer state_bytes;
    std::optional<LinearAttentionStatePool> state;
    DeviceBuffer round_bytes;
    std::optional<qwen::RoundState> round;
    std::optional<WorkspaceArena> work;
    Tensor prefill_hidden;
    // The Text forward's Program runtime: empty for a dense model without PLE or experts.
    exec::Qwen4Runtime qwen4_runtime;

    RankStorage(const DeviceContext& device, const qwen::TextConfig& config,
                std::uint32_t logit_rows, KvCacheStorage storage, std::size_t workspace_bytes,
                std::uint32_t capacity = kCapacity) {
        CUDA_CHECK(cudaSetDevice(device.device));
        const std::uint32_t page_groups = capacity / kPagedKVPageSize;
        LayoutBuilder kv_builder;
        const auto kv = qwen::plan_decoder_state(
            kv_builder, qwen::DecoderStateSpec{
                            .full_attention_layers = config.full_attention_layers,
                            .mtp_layers            = 1,
                            .capacity              = capacity,
                            .kv_heads = exec::dimension(config.attention->num_key_value_heads),
                            .attention_head_dim = exec::dimension(config.attention->head_dim),
                            .kv_storage         = storage,
                            .enable_mtp         = false,
                            .kv_table_rows      = 1,
                            .text_physical_page_groups = page_groups,
                            .mtp_physical_page_groups  = 0,
                        });
        decoder_bytes = device_bytes(device.device, kv_builder.finish(256));
        decoder.emplace(device.device, DeviceSpan{decoder_bytes.p, decoder_bytes.bytes}, kv);
        auto& pool  = decoder->text_kv.page_pool();
        reservation = pool.reserve(page_groups);
        require(reservation.has_value(), "KV page reservation failed");
        pages.reserve(page_groups);
        pool.materialize(*reservation, page_groups, pages);
        row = decoder->text_kv.execution_tables().acquire(0);
        decoder->text_kv.execution_tables().publish(
            row.handle(), 0, std::span<const DeviceKVPageLease>(pages), device.stream);

        const auto& gdn = *config.gdn;
        LayoutBuilder state_builder;
        const auto linear = plan_linear_attention_state_pool(
            state_builder, LinearAttentionStatePoolSpec{
                               .layers         = config.linear_attention_layers,
                               .conv_channels  = exec::dimension(gdn.conv_channels()),
                               .conv_width     = exec::dimension(gdn.linear_conv_kernel_dim - 1),
                               .value_heads    = exec::dimension(gdn.linear_num_value_heads),
                               .value_head_dim = exec::dimension(gdn.linear_value_head_dim),
                               .key_head_dim   = exec::dimension(gdn.linear_key_head_dim),
                               .slot_count     = 1,
                               .conv_dtype     = DType::BF16,
                           });
        state_bytes = device_bytes(device.device, state_builder.finish(256));
        state.emplace(DeviceSpan{state_bytes.p, state_bytes.bytes}, linear);

        LayoutBuilder round_builder;
        auto round_layout = qwen::begin_round_state_layout(
            round_builder, qwen::RoundStateSpec{.hidden      = exec::dimension(config.hidden_size),
                                                .output_rows = exec::dimension(logit_rows),
                                                .batch_capacity = 1});
        qwen::complete_round_state_layout(round_builder, round_layout);
        round_bytes = device_bytes(device.device, round_builder.finish(256));
        round.emplace(DeviceSpan{round_bytes.p, round_bytes.bytes}, round_layout);
        CUDA_CHECK(cudaSetDevice(device.device));
        ops::set_i32_scalar(round->text_kv_table_row, row.row_index(), device.stream);
        work.emplace(workspace_bytes);
        device.synchronize();
    }

    RankStorage(const RankStorage&)            = delete;
    RankStorage& operator=(const RankStorage&) = delete;
};

// ------------------------------------------------------------------------------------------------
// One execution of the Text model over one or two ranks
// ------------------------------------------------------------------------------------------------

class Runner {
public:
    Runner(ExecutionContext& execution, const exec::Parameters& rank0, RankStorage& storage0,
           std::uint32_t prefill_chunk, const exec::TpExecution* tp, RankStorage* storage1)
        : execution_(execution), rank0_(rank0), storage0_(storage0), chunk_(prefill_chunk), tp_(tp),
          storage1_(storage1) {}

    // Prefills the prompt chunk by chunk, as the Program does, and returns rank 0's logits.
    // `rope_delta` is the offset a cached multimodal prefix leaves; the Program passes the
    // sequence's to every chunk, including the text-only ones that continue it.
    std::vector<float> prefill(std::span<const int> prompt, std::int32_t rope_delta = 0) {
        // Like the Program, enter with rank 0's device current: the single-device schedule
        // launches on the current device.
        CUDA_CHECK(cudaSetDevice(execution_.dev[0]->device));
        std::uint32_t base = 0;
        while (base < prompt.size()) {
            exec::TextContext card(*execution_.dev[0], rank0_, *storage0_.work,
                                   qwen::PagedKVCacheView{}, *storage0_.state, *storage0_.round,
                                   storage0_.prefill_hidden, chunk_, base, {},
                                   &storage0_.decoder->text_kv, nullptr, tp_, rope_delta);
            card.set_qwen4_runtime(&storage0_.qwen4_runtime);
            const auto result = card.prefill_chunk(
                prompt, base, static_cast<std::uint32_t>(prompt.size()) - base, true);
            require(result.processed_tokens != 0, "prefill made no progress");
            base += result.processed_tokens;
            require(result.finalized == (base == prompt.size()), "prefill finalization differs");
        }
        synchronize();
        return logits(storage0_.round->logits.slice(1, 0, 1));
    }

    [[nodiscard]] int prefill_token() const {
        int token = 0;
        CUDA_CHECK(
            cudaMemcpy(&token, storage0_.round->token.data, sizeof(token), cudaMemcpyDeviceToHost));
        return token;
    }

    // One ordinary decode step of `token` at position `position`; returns rank 0's logits.
    std::vector<float> decode(int token, std::int32_t position) {
        qwen::OrdinaryDecodeIngress ingress{};
        ingress.tokens[0]                  = token;
        ingress.cache_positions[0]         = position;
        ingress.rope_positions[0]          = position;
        ingress.text_kv_table_rows[0]      = storage0_.row.row_index();
        ingress.state_source_slots[0]      = 0;
        ingress.state_destination_slots[0] = 0;
        // The Program uploads the same record into both ranks' frames on their own streams.
        for (int rank = 0; rank < (tp_ != nullptr ? 2 : 1); ++rank) {
            RankStorage& storage = rank == 0 ? storage0_ : *storage1_;
            const auto& device   = *execution_.dev[static_cast<std::size_t>(rank)];
            CUDA_CHECK(cudaSetDevice(device.device));
            CUDA_CHECK(cudaMemcpyAsync(storage.round->ordinary->ingress.data, &ingress,
                                       sizeof(ingress), cudaMemcpyHostToDevice, device.stream));
        }
        CUDA_CHECK(cudaSetDevice(execution_.dev[0]->device));
        exec::TextContext card(*execution_.dev[0], rank0_, *storage0_.work,
                               qwen::PagedKVCacheView{}, *storage0_.state, *storage0_.round,
                               storage0_.prefill_hidden, chunk_, 0, {}, &storage0_.decoder->text_kv,
                               nullptr, tp_);
        card.set_qwen4_runtime(&storage0_.qwen4_runtime);
        auto& frame        = *storage0_.round->ordinary;
        Tensor hidden      = frame.hidden.slice(1, 0, 1);
        Tensor out         = frame.logits.slice(1, 0, 1);
        const auto visible = static_cast<std::uint32_t>(position + 1);
        card.ordinary_decode_batch(
            frame.tokens.slice(0, 0, 1), frame.cache_positions.slice(0, 0, 1),
            frame.rope_positions.slice(0, 0, 1), frame.text_kv_table_rows.slice(0, 0, 1),
            frame.state_source_slots.slice(0, 0, 1), frame.state_destination_slots.slice(0, 0, 1),
            {visible, visible}, hidden, out);
        synchronize();
        return logits(out);
    }

private:
    void synchronize() const {
        execution_.dev[0]->synchronize();
        if (tp_ != nullptr) { execution_.dev[1]->synchronize(); }
    }

    static std::vector<float> logits(const Tensor& column) {
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(column.ne[0]));
        CUDA_CHECK(cudaMemcpy(bits.data(), column.data, bits.size() * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost));
        std::vector<float> out(bits.size());
        std::transform(bits.begin(), bits.end(), out.begin(), bf16_value);
        return out;
    }

    ExecutionContext& execution_;
    const exec::Parameters& rank0_;
    RankStorage& storage0_;
    std::uint32_t chunk_;
    const exec::TpExecution* tp_;
    RankStorage* storage1_;
};

// Rank 1's half of a tp2 execution, as the Program would publish it.
struct PeerBinding {
    exec::OrdinaryPeerFrame frame;
    exec::TpExecution tp;

    PeerBinding(const ExecutionContext& execution, const ops::PeerEvents& events,
                const exec::Parameters& rank1, RankStorage& storage)
        : frame(exec::ordinary_peer_frame(*storage.round->ordinary)) {
        tp = exec::TpExecution{.execution         = &execution,
                               .events            = &events,
                               .parameters        = &rank1,
                               .work              = &*storage.work,
                               .linear_attention  = &*storage.state,
                               .text_cache        = &storage.decoder->text_kv,
                               .text_kv_table_row = storage.round->text_kv_table_row,
                               .rope_delta        = storage.round->rope_delta,
                               .ordinary          = &frame,
                               .qwen4             = &storage.qwen4_runtime};
    }

    PeerBinding(const PeerBinding&)            = delete;
    PeerBinding& operator=(const PeerBinding&) = delete;
};

std::size_t argmax(std::span<const float> values, std::size_t domain) {
    return static_cast<std::size_t>(
        std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(domain)) -
        values.begin());
}

float bf16_ulp(float magnitude) {
    if (magnitude <= 0 || !std::isfinite(magnitude)) { return 0; }
    return std::ldexp(1.0F, std::ilogb(magnitude) - 7);
}

// Compares one column of split logits with the single-device column; see the file comment.
void compare(const std::vector<float>& reference, const std::vector<float>& split,
             std::size_t domain, const std::string& label) {
    require(reference.size() == split.size(), label + ": logit extents differ");
    float largest = 0;
    float worst   = 0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        require(std::isfinite(reference[i]) && std::isfinite(split[i]),
                label + ": non-finite logit at row " + std::to_string(i));
        largest = std::max(largest, std::abs(reference[i]));
        worst   = std::max(worst, std::abs(reference[i] - split[i]));
    }
    const float tolerance = 2.0F * bf16_ulp(largest);
    std::cout << label << ": max |tp2 - tp1| " << worst << " (largest " << largest << ", tolerance "
              << tolerance << ")\n";
    require(worst <= tolerance, label + ": split logits exceed two BF16 ulps of the largest");
    const auto expected = argmax(reference, domain);
    const auto actual   = argmax(split, domain);
    require(expected == actual || reference[expected] - reference[actual] <= tolerance,
            label + ": split argmax differs beyond the tolerance");
}

std::vector<std::uint16_t> read_bf16(const Tensor& tensor) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(tensor.numel()));
    CUDA_CHECK(cudaMemcpy(bits.data(), tensor.data, bits.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));
    return bits;
}

// The rank-0 logits gather over C = 3 columns, the case ordinary decode at batch > 1 relies on:
// every gathered column must be rank 0's and then rank 1's partial column byte for byte (the gather
// only relocates storage), and must match the single-device head within the criterion above.
void gather_columns(const ExecutionContext& execution, const ops::PeerEvents& events,
                    const exec::Parameters& reference, const exec::Parameters& rank0,
                    const exec::Parameters& rank1, WorkspaceArena& reference_work,
                    WorkspaceArena& work0, WorkspaceArena& work1, std::size_t domain) {
    constexpr std::int32_t kColumns     = 3;
    const auto hidden_size              = static_cast<std::int32_t>(kHidden);
    const exec::LinearParameters& head0 = rank0.text.output_head;
    const exec::LinearParameters& head1 = rank1.text.output_head;
    const std::int32_t rows0            = head0.weight.n;
    const std::int32_t rows1            = head1.weight.n;
    const std::int32_t vocab            = rows0 + rows1;
    const DeviceContext& device0        = *execution.dev[0];
    const DeviceContext& device1        = *execution.dev[1];
    std::vector<std::uint16_t> host_hidden(static_cast<std::size_t>(hidden_size) * kColumns);
    std::uint64_t state = 29;
    for (auto& value : host_hidden) { value = bf16_bits(uniform(state, -2.0F, 2.0F)); }

    reference_work.reset();
    work0.reset();
    work1.reset();
    const Tensor hidden0  = work0.alloc(DType::BF16, {hidden_size, kColumns});
    const Tensor partial0 = work0.alloc(DType::BF16, {rows0, kColumns});
    const Tensor staging  = work0.alloc(DType::BF16, {rows1, kColumns});
    const Tensor gathered = work0.alloc(DType::BF16, {vocab, kColumns});
    const Tensor hidden1  = work1.alloc(DType::BF16, {hidden_size, kColumns});
    const Tensor partial1 = work1.alloc(DType::BF16, {rows1, kColumns});
    Tensor single         = reference_work.alloc(DType::BF16, {vocab, kColumns});
    for (const Tensor* hidden : {&hidden0, &hidden1}) {
        CUDA_CHECK(
            cudaMemcpy(hidden->data, host_hidden.data(), hidden->bytes(), cudaMemcpyHostToDevice));
    }
    for (const DeviceContext* device : {&device0, &device1}) {
        CUDA_CHECK(cudaSetDevice(device->device));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaSetDevice(device0.device));
    exec::output_logits_split_rank0({hidden0, hidden1}, {&head0, &head1}, {partial0, partial1},
                                    gathered, staging, {&work0, &work1}, execution, events);
    exec::project(hidden0, reference.text.output_head, single, reference_work, device0.stream);
    device0.synchronize();
    device1.synchronize();

    const auto whole  = read_bf16(gathered);
    const auto half0  = read_bf16(partial0);
    const auto half1  = read_bf16(partial1);
    const auto oracle = read_bf16(single);
    for (std::int32_t column = 0; column < kColumns; ++column) {
        const auto base = static_cast<std::size_t>(column) * static_cast<std::size_t>(vocab);
        require(std::equal(half0.begin() + column * rows0, half0.begin() + (column + 1) * rows0,
                           whole.begin() + static_cast<std::ptrdiff_t>(base)) &&
                    std::equal(half1.begin() + column * rows1, half1.begin() + (column + 1) * rows1,
                               whole.begin() + static_cast<std::ptrdiff_t>(base) + rows0),
                "gathered logits column " + std::to_string(column) +
                    " is not the two partial columns");
        std::vector<float> expected(static_cast<std::size_t>(vocab));
        std::vector<float> actual(static_cast<std::size_t>(vocab));
        for (std::size_t i = 0; i < expected.size(); ++i) {
            expected[i] = bf16_value(oracle[base + i]);
            actual[i]   = bf16_value(whole[base + i]);
        }
        compare(expected, actual, domain, "gathered logits column " + std::to_string(column));
    }
    reference_work.reset();
    work0.reset();
    work1.reset();
}

// A prompt chunk that fails inside its sliced FFN must leave no split-exchange lane open for the
// next chunk. A 512-column chunk over a HostExchange runs its block tails as four 128-column
// slices (tp2/prefill_slices.h): the first layer's mixer begins one all-reduce per slice, and the
// FFN finishes slice s's before it runs. Rank 0's copy of that FFN disagrees with rank 1's on the
// activation policy, so it throws at slice 0 with slices 1-3 still open. The same chunk then runs
// again from fresh state on intact Parameters, through the same exchange, and must reproduce the
// logits of the run before the failure bit for bit.
void failed_slice_closes_lanes(ExecutionContext& execution, const qwen::Model& split,
                               const exec::Parameters& rank0, const exec::Parameters& rank1) {
    constexpr std::uint32_t kTokens = 512;
    require(exec::prefill_overlap_split(kTokens) == 0 &&
                exec::prefill_slice_plan(kTokens, 0).count == 4,
            "the chunk no longer runs as four slices");
    ops::PeerEvents events(execution);
    ops::HostExchange exchange(execution);
    events.attach_host_exchange(&exchange);
    require(ops::allreduce_sum_split_available(execution, events), "no split exchange");

    exec::Parameters broken(split, 0);
    auto& ffn          = std::get<exec::DenseParameters>(broken.qwen4->layers.front().ffn);
    ffn.gate_up.policy = ffn.gate_up.policy == ops::LinearPolicy::A16Only
                             ? ops::LinearPolicy::AllowA8
                             : ops::LinearPolicy::A16Only;

    const auto& config               = split.config().text;
    const auto shard                 = exec::shard_text_config(config, exec::kTensorParallelWidth);
    constexpr std::size_t kWorkspace = std::size_t{256} << 20U;
    std::vector<int> prompt(kTokens);
    std::uint64_t state = 29;
    for (auto& token : prompt) { token = static_cast<int>(splitmix(state) % 256U); }
    // Each run starts from fresh KV and GDN state; only the exchange carries over.
    const auto prefill = [&](const exec::Parameters& parameters) {
        RankStorage storage0(*execution.dev[0], shard, config.vocab_size, KvCacheStorage::BFloat16,
                             kWorkspace, kTokens);
        RankStorage storage1(*execution.dev[1], shard, config.vocab_size, KvCacheStorage::BFloat16,
                             kWorkspace, kTokens);
        PeerBinding peer(execution, events, rank1, storage1);
        return Runner(execution, parameters, storage0, kTokens, &peer.tp, &storage1)
            .prefill(prompt);
    };
    const auto before = prefill(rank0);
    std::string failure;
    try {
        (void)prefill(broken);
    } catch (const std::exception& error) { failure = error.what(); }
    std::cout << "injected sliced FFN failure: " << failure << '\n';
    require(failure.find("text/layers/0 ") != std::string::npos &&
                failure.find("ranks disagree on the activation policy") != std::string::npos,
            "the mismatched rank policies did not fail the first layer's FFN");
    require(prefill(rank0) == before,
            "the chunk after a failed sliced FFN does not reproduce the logits before it");
}

int parity() {
    const char* scratch              = std::getenv("NINFER_TEST_SCRATCH_DIR");
    const std::filesystem::path root = scratch != nullptr && *scratch
                                           ? std::filesystem::path(scratch)
                                           : std::filesystem::temp_directory_path();
    SyntheticArtifact file(root / ("ninfer-tp2-parity-" + std::to_string(::getpid())));
    const auto path = write_synthetic_model(file);
    artifact::Reader reader(path);

    ExecutionContext execution({0, 1});
    (void)ops::enable_peer_access(execution);
    const ops::PeerEvents events(execution);

    auto single =
        qwen::materialize_model(qwen::plan_load(reader, models::LoadOptions{}), *execution.dev[0]);
    auto split =
        qwen::materialize_model(qwen::plan_load(reader, models::LoadOptions{.tp = 2}), execution);
    const exec::Parameters reference(*single, 0);
    const exec::Parameters rank0(*split, 0);
    const exec::Parameters rank1(*split, 1);

    const auto& config               = single->config().text;
    const auto shard                 = exec::shard_text_config(config, exec::kTensorParallelWidth);
    constexpr std::size_t kWorkspace = std::size_t{256} << 20U;
    const std::uint32_t rows         = config.vocab_size;
    RankStorage reference_storage(*execution.dev[0], config, rows, KvCacheStorage::BFloat16,
                                  kWorkspace);
    RankStorage storage0(*execution.dev[0], shard, rows, KvCacheStorage::BFloat16, kWorkspace);
    RankStorage storage1(*execution.dev[1], shard, rows, KvCacheStorage::BFloat16, kWorkspace);
    PeerBinding peer(execution, events, rank1, storage1);

    // Guard: the head-local attention has no FP8 KV route, so construction must reject it.
    {
        RankStorage fp8_0(*execution.dev[0], shard, rows, KvCacheStorage::Fp8E4M3Row256, 1U << 20U);
        RankStorage fp8_1(*execution.dev[1], shard, rows, KvCacheStorage::Fp8E4M3Row256, 1U << 20U);
        PeerBinding fp8_peer(execution, events, rank1, fp8_1);
        bool rejected = false;
        try {
            exec::TextContext card(*execution.dev[0], rank0, *fp8_0.work, qwen::PagedKVCacheView{},
                                   *fp8_0.state, *fp8_0.round, fp8_0.prefill_hidden, 64, 0, {},
                                   &fp8_0.decoder->text_kv, nullptr, &fp8_peer.tp);
        } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "tp2 TextContext accepted an FP8 KV cache");
    }

    // A 64-token prompt in chunks of 48 and 16: the second chunk continues the KV and GDN state.
    std::vector<int> prompt(64);
    std::uint64_t state = 17;
    for (auto& token : prompt) { token = static_cast<int>(splitmix(state) % 256U); }
    constexpr std::uint32_t kChunk = 48;
    const std::size_t domain       = single->resources().public_token_count;

    Runner one(execution, reference, reference_storage, kChunk, nullptr, nullptr);
    Runner two(execution, rank0, storage0, kChunk, &peer.tp, &storage1);
    const auto reference_prefill = one.prefill(prompt);
    compare(reference_prefill, two.prefill(prompt), domain, "prefill");
    // Both executions decode the single-device token stream, so every step compares the same
    // inputs. The sampled prefill tokens were checked through the logits argmax above.
    int token = one.prefill_token();
    for (int step = 0; step < 4; ++step) {
        const auto position         = static_cast<std::int32_t>(prompt.size()) + step;
        const auto reference_logits = one.decode(token, position);
        compare(reference_logits, two.decode(token, position), domain,
                "decode step " + std::to_string(step));
        token = static_cast<int>(argmax(reference_logits, domain));
    }
    // The second chunk continues a text-only suffix after a multimodal prefix: it must honor the
    // prefix's rope delta. Each prefill starts from fresh state, since decode leaves some behind.
    const auto prefill_with_delta = [&](std::int32_t rope_delta) {
        RankStorage fresh(*execution.dev[0], config, rows, KvCacheStorage::BFloat16, kWorkspace);
        return Runner(execution, reference, fresh, kChunk, nullptr, nullptr)
            .prefill(prompt, rope_delta);
    };
    const auto unshifted = prefill_with_delta(0);
    require(prefill_with_delta(0) == unshifted, "prefill is not repeatable");
    const auto shifted = prefill_with_delta(33);
    float moved        = 0.0F;
    for (std::size_t i = 0; i < domain; ++i) {
        moved = std::max(moved, std::abs(shifted[i] - unshifted[i]));
    }
    require(moved > 0.0F, "a later prefill chunk ignored the rope delta");
    gather_columns(execution, events, reference, rank0, rank1, *reference_storage.work,
                   *storage0.work, *storage1.work, domain);
    failed_slice_closes_lanes(execution, *split, rank0, rank1);
    std::cout << "qwen3_5 tensor-parallel Text parity passed\n";
    return 0;
}

int consistency() {
    const char* path = std::getenv("NINFER_TEST_ARTIFACT");
    if (path == nullptr || *path == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    artifact::Reader reader(path);
    ExecutionContext execution({0, 1});
    (void)ops::enable_peer_access(execution);
    const ops::PeerEvents events(execution);
    auto model =
        qwen::materialize_model(qwen::plan_load(reader, models::LoadOptions{.tp = 2}), execution);
    const exec::Parameters rank0(*model, 0);
    const exec::Parameters rank1(*model, 1);
    const auto shard = exec::shard_text_config(model->config().text, exec::kTensorParallelWidth);
    constexpr std::size_t kWorkspace = std::size_t{512} << 20U;
    const std::uint32_t rows         = model->config().text.vocab_size;
    RankStorage storage0(*execution.dev[0], shard, rows, KvCacheStorage::BFloat16, kWorkspace);
    RankStorage storage1(*execution.dev[1], shard, rows, KvCacheStorage::BFloat16, kWorkspace);
    PeerBinding peer(execution, events, rank1, storage1);

    const auto& tokenizer = *model->resources().tokenizer;
    const std::vector<int> prompt =
        tokenizer.encode("<|im_start|>user\nWhat is 17*23? Answer with the number only.<|im_end|>\n"
                         "<|im_start|>assistant\n<think>\n\n</think>\n\n");
    constexpr std::size_t kAnswerTokens = 64;
    require(!prompt.empty() && prompt.size() + kAnswerTokens < kCapacity,
            "prompt does not fit the test KV");
    const std::size_t domain = model->resources().public_token_count;
    const auto& stops        = tokenizer.default_stop_token_ids();

    Runner runner(execution, rank0, storage0, kCapacity, &peer.tp, &storage1);
    auto logits = runner.prefill(prompt);
    std::vector<int> generated;
    for (std::size_t step = 0; step < kAnswerTokens; ++step) {
        for (const float value : logits) { require(std::isfinite(value), "non-finite logit"); }
        const int token = static_cast<int>(argmax(logits, domain));
        if (std::find(stops.begin(), stops.end(), token) != stops.end()) { break; }
        generated.push_back(token);
        logits =
            runner.decode(token, static_cast<std::int32_t>(prompt.size() + generated.size() - 1));
    }
    const std::string answer = tokenizer.decode(generated);
    std::cout << "tp2 answer: " << answer << '\n';
    require(answer.find("391") != std::string::npos, "the tp2 answer does not contain 391");
    std::cout << "qwen3_5 tensor-parallel Text consistency passed\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    // `write DIR` writes the synthetic artifact to DIR/model.ninfer and keeps it: the tp 1 golden
    // gate (tools/golden) runs it through this fork and through upstream. No device is needed.
    if (argc > 2 && std::string(argv[1]) == "write") {
        try {
            SyntheticArtifact file{std::filesystem::path(argv[2])};
            const auto entry = write_synthetic_model(file);
            file.keep();
            std::cout << entry.string() << '\n';
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
    const bool real = argc > 1 && std::string(argv[1]) == "real";
    int devices     = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: tensor-parallel execution requires two CUDA devices\n";
        return 77;
    }
    try {
        return real ? consistency() : parity();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
