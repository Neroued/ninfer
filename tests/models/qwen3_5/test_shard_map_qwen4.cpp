// Qwen4Exp (Qwen3.8-Flash-Next) tensor-parallel placement (load/sharding.h), host only.
//
// 1. The logical rule of every Qwen4Exp parameter at tp 2, default and with hc_split_streams.
// 2. The combined placement of the real model's parents (48 blocks + MTP, both official recipes,
//    the converter's grouping as written to the artifact: q|k|gate|v, q|k|v|z, a|b, HC down|inject,
//    shared gate|up, router|shared_score, indexer query|key, PLE key|value): every rank shard is
//    accepted by the slicer at the stored format's tile boundaries and has the per-rank shape the
//    split Ops consume, and the per-rank device bytes are printed (the T30 dossier's table).
// 3. Rejections: a Qwen4Exp parameter without a rule, an odd stream split.
// 4. A dense Qwen4Exp FFN: mlp/* split like the Qwen3.5 dense block, no MoE rule applies, the
//    per-rank config halves the intermediate width.

#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/formats.h"
#include "artifact/slices.h"
#include "models/qwen3_5/execution/tp.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/load/sharding.h"

#include <array>
#include <cstdio>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace qwen    = ninfer::models::qwen3_5;
namespace loading = ninfer::models::qwen3_5::loading;
using artifact::ShardAxis;
using artifact::SliceRange;
using ninfer::test::artifact_fixture::rejects;
using ninfer::test::artifact_fixture::require;
using Ranges = std::vector<SliceRange>;

constexpr std::uint64_t kH = 2560, kWide = 10240, kVocab = 248320;

// Qwen3.8-Flash-Next text geometry: 12 x (3 GDN + 1 QSA) blocks, MoE 512 x top-10, HC 4 x 320.
qwen::Config flash_next(std::uint32_t layers = 48) {
    qwen::Config config;
    auto& text             = config.text;
    text.architecture      = Architecture::Qwen4Exp;
    text.hidden_size       = kH;
    text.vocab_size        = kVocab;
    text.num_hidden_layers = layers;
    for (std::uint32_t i = 0; i < layers; ++i) {
        text.layer_types.push_back(i % 4 == 3 ? qwen::MixerKind::FullAttention
                                              : qwen::MixerKind::LinearAttention);
    }
    text.attention        = qwen::AttentionConfig{24, 2, 256};
    text.gdn              = qwen::GdnConfig{16, 128, 48, 128, 4};
    text.ffn              = qwen::MoeConfig{512, 10, 640, 640};
    text.gdn_gate         = qwen::GateActivation::Sigmoid;
    text.hyper_connection = qwen::HyperConnectionConfig{4, 320};
    text.sparse_attention = qwen::SparseAttentionConfig{4, 128, 2048, 4};
    qwen::PleConfig ple;
    ple.layer           = 1;
    ple.ngram_size      = 3;
    ple.heads_per_ngram = 8;
    ple.embed_dim       = 2560;
    ple.conv_kernel     = 4;
    ple.table_rows      = 320001536;
    text.ple            = ple;
    config.mtp          = true;
    return config;
}

const LoadOptions kTp2{.speculative = SpeculativeBackend::Mtp, .tp = 2};
const LoadOptions kTp2Split{
    .speculative = SpeculativeBackend::Mtp, .tp = 2, .hc_split_streams = true};

Ranges halves(std::uint64_t total, int device) {
    return {{static_cast<std::uint64_t>(device) * total / 2, total / 2}};
}

void expect(const qwen::Config& config, const std::string& name, artifact::Shape shape,
            ShardAxis axis, const std::array<Ranges, 2>& ranges = {}, int holder = 0,
            const LoadOptions& options = kTp2) {
    const auto shard = loading::logical_shard(name, shape, config, options);
    if (shard.axis != axis || shard.device != holder || shard.ranges[0] != ranges[0] ||
        shard.ranges[1] != ranges[1]) {
        throw std::runtime_error(name + ": unexpected tensor-parallel placement (axis " +
                                 std::to_string(int(shard.axis)) + ", device " +
                                 std::to_string(shard.device) + ")");
    }
}

void logical_policy() {
    const auto config         = flash_next();
    constexpr std::uint64_t q = 6144, k = 512, gk = 2048, gv = 6144;
    for (const std::string p : {"text/layers/3/", "mtp/layers/0/"}) {
        // 12 query heads and their 1 KV head per rank.
        for (const auto* leaf : {"attention/query", "attention/gate"}) {
            expect(config, p + leaf, {q, kH}, ShardAxis::Rows, {halves(q, 0), halves(q, 1)});
        }
        for (const auto* leaf : {"attention/key", "attention/value"}) {
            expect(config, p + leaf, {k, kH}, ShardAxis::Rows, {halves(k, 0), halves(k, 1)});
        }
        expect(config, p + "attention/output", {kH, q}, ShardAxis::Columns,
               {halves(q, 0), halves(q, 1)});
        for (const auto* leaf : {"attention/query_norm", "attention/key_norm"}) {
            expect(config, p + leaf, {256}, ShardAxis::Replicated);
        }
        // The indexer is replicated in both modes.
        for (const auto& options : {kTp2, kTp2Split}) {
            expect(config, p + "attention/indexer_query", {512, kH}, ShardAxis::Replicated, {}, 0,
                   options);
            expect(config, p + "attention/indexer_key", {128, kH}, ShardAxis::Replicated, {}, 0,
                   options);
            expect(config, p + "attention/indexer_query_norm", {128}, ShardAxis::Replicated, {}, 0,
                   options);
            expect(config, p + "attention/indexer_key_norm", {128}, ShardAxis::Replicated, {}, 0,
                   options);
        }
    }
    // GDN: 8 key heads with their 24 value heads per rank.
    const std::string g = "text/layers/0/gdn/";
    for (const auto* leaf : {"query", "key"}) {
        expect(config, g + leaf, {gk, kH}, ShardAxis::Rows, {halves(gk, 0), halves(gk, 1)});
    }
    for (const auto* leaf : {"value", "z"}) {
        expect(config, g + leaf, {gv, kH}, ShardAxis::Rows, {halves(gv, 0), halves(gv, 1)});
    }
    for (const auto* leaf : {"a_projection", "b_projection"}) {
        expect(config, g + leaf, {48, kH}, ShardAxis::Rows, {halves(48, 0), halves(48, 1)});
    }
    for (const auto* leaf : {"a_log", "dt_bias"}) {
        expect(config, g + leaf, {48}, ShardAxis::Rows, {halves(48, 0), halves(48, 1)});
    }
    expect(config, g + "norm", {128}, ShardAxis::Replicated);
    expect(config, g + "output", {kH, gv}, ShardAxis::Columns, {halves(gv, 0), halves(gv, 1)});
    expect(config, g + "convolution", {4, kWide}, ShardAxis::Columns,
           {Ranges{{0, 1024}, {2048, 1024}, {4096, 3072}},
            Ranges{{1024, 1024}, {3072, 1024}, {7168, 3072}}});

    // MoE: router and shared-expert gate replicated, routed experts host-mapped (also at tp 1),
    // the shared expert whole on rank (block % 2); the MTP block is block 48.
    for (const std::string p : {"text/layers/0/", "text/layers/47/", "mtp/layers/0/"}) {
        expect(config, p + "moe/router", {512, kH}, ShardAxis::Replicated);
        expect(config, p + "moe/shared_score", {1, kH}, ShardAxis::Replicated);
        for (const auto* expert : {"moe/experts/0/gate", "moe/experts/511/down"}) {
            for (const int tp : {1, 2}) {
                expect(config, p + expert, {640, kH}, ShardAxis::HostMapped, {}, 0,
                       LoadOptions{.tp = tp});
            }
        }
    }
    for (const auto& [p, rank] : std::vector<std::pair<std::string, int>>{{"text/layers/0/", 0},
                                                                          {"text/layers/1/", 1},
                                                                          {"text/layers/46/", 0},
                                                                          {"text/layers/47/", 1},
                                                                          {"mtp/layers/0/", 0}}) {
        for (const auto* leaf : {"moe/shared/gate", "moe/shared/up"}) {
            expect(config, p + leaf, {640, kH}, ShardAxis::SingleDevice, {}, rank);
        }
        expect(config, p + "moe/shared/down", {kH, 640}, ShardAxis::SingleDevice, {}, rank);
    }

    // Hyper-connections: replicated by default; split by residual streams with the flag.
    for (const std::string p :
         {"text/layers/0/attn_hc/", "text/layers/3/ffn_hc/", "mtp/layers/0/attn_hc/"}) {
        expect(config, p + "norm", {kWide}, ShardAxis::Replicated);
        expect(config, p + "down", {320, kWide}, ShardAxis::Replicated);
        expect(config, p + "inject", {4, kWide}, ShardAxis::Replicated);
        expect(config, p + "up", {kWide, 320}, ShardAxis::Replicated);
        expect(config, p + "norm", {kWide}, ShardAxis::Rows, {halves(kWide, 0), halves(kWide, 1)},
               0, kTp2Split);
        expect(config, p + "up", {kWide, 320}, ShardAxis::Rows,
               {halves(kWide, 0), halves(kWide, 1)}, 0, kTp2Split);
        for (const auto* leaf : {"down", "inject"}) {
            expect(config, p + leaf, {320, kWide}, ShardAxis::Columns,
                   {halves(kWide, 0), halves(kWide, 1)}, 0, kTp2Split);
        }
    }
    for (const std::string p : {"text/hc_head/", "mtp/hc_head/"}) {
        expect(config, p + "down", {320, kWide}, ShardAxis::Replicated);
        expect(config, p + "down", {320, kWide}, ShardAxis::Columns,
               {halves(kWide, 0), halves(kWide, 1)}, 0, kTp2Split);
        expect(config, p + "up", {kWide, 320}, ShardAxis::Rows,
               {halves(kWide, 0), halves(kWide, 1)}, 0, kTp2Split);
    }
    // PLE: key|value projection and the table replicated in both modes; the per-lane norms and
    // convolution follow the residual streams.
    const std::string ple = "text/layers/1/ple/";
    for (const auto& options : {kTp2, kTp2Split}) {
        expect(config, ple + "key", {kWide, kH}, ShardAxis::Replicated, {}, 0, options);
        expect(config, ple + "value", {kH, kH}, ShardAxis::Replicated, {}, 0, options);
        expect(config, ple + "table", {320001536, 160}, ShardAxis::Replicated, {}, 0, options);
    }
    for (const auto* leaf : {"key_norm", "query_norm", "conv_norm"}) {
        expect(config, ple + leaf, {kWide}, ShardAxis::Replicated);
        expect(config, ple + leaf, {kWide}, ShardAxis::Rows, {halves(kWide, 0), halves(kWide, 1)},
               0, kTp2Split);
    }
    expect(config, ple + "convolution", {4, kWide}, ShardAxis::Replicated);
    expect(config, ple + "convolution", {4, kWide}, ShardAxis::Columns,
           {halves(kWide, 0), halves(kWide, 1)}, 0, kTp2Split);

    // Vocabulary, MTP stem.
    expect(config, "text/token_embedding", {kVocab, kH}, ShardAxis::Replicated);
    expect(config, "text/token_embedding", {kVocab, kH}, ShardAxis::HostMapped, {}, 0,
           LoadOptions{.tp = 2, .embedding_host = true});
    expect(config, "text/output_head", {kVocab, kH}, ShardAxis::Rows,
           {halves(kVocab, 0), halves(kVocab, 1)});
    for (const auto* name :
         {"mtp/embedding_norm", "mtp/embedding_projection", "mtp/hidden_projection"}) {
        expect(config, name, {kH, kH}, ShardAxis::Replicated, {}, 0, kTp2Split);
    }
    expect(config, "mtp/hidden_norm", {kWide}, ShardAxis::Replicated);
    expect(config, "mtp/hidden_norm", {kWide}, ShardAxis::Rows,
           {halves(kWide, 0), halves(kWide, 1)}, 0, kTp2Split);
    expect(config, "vision/layers/0/attention/query", {1152, 1152}, ShardAxis::SingleDevice, {}, 1,
           {.tp = 2, .vision_rank = 1});

    // Rejections.
    rejects<std::invalid_argument>(
        [&] { (void)loading::logical_shard("text/layers/0/input_norm", {kH}, config, kTp2); },
        "a Qwen3.5 block leaf was placed for Qwen4Exp");
    rejects<std::invalid_argument>(
        [&] { (void)loading::logical_shard("text/final_norm", {kH}, config, kTp2); },
        "a parameter Qwen4Exp does not have was placed");
    rejects<std::invalid_argument>(
        [&] { (void)loading::logical_shard("text/layersX/attn_hc/norm", {kWide}, config, kTp2); },
        "a malformed block name was placed");
    auto three                            = config;
    three.text.hyper_connection->hc_count = 3;
    rejects<std::invalid_argument>(
        [&] {
            (void)loading::logical_shard("text/layers/0/attn_hc/norm", {3 * kH}, three, kTp2Split);
        },
        "three streams were split over two ranks");

    // Per-rank Text config.
    const auto rank  = qwen::execution::shard_text_config(config.text, 2);
    const auto split = qwen::execution::shard_text_config(config.text, 2, true);
    require(rank.attention->num_attention_heads == 12 && rank.attention->num_key_value_heads == 1 &&
                rank.gdn->linear_num_key_heads == 8 && rank.gdn->linear_num_value_heads == 24 &&
                rank.vocab_size == kVocab / 2 &&
                std::get<qwen::MoeConfig>(rank.ffn).num_experts == 512 &&
                rank.residual_width() == kWide && rank.sparse_attention->indexer_heads == 4,
            "per-rank Qwen4Exp config");
    require(split.residual_width() == kWide / 2 && split.hyper_connection->hc_count == 2,
            "per-rank Qwen4Exp config with split streams");
}

// Dense FFN: gate/up rows and down columns halved per rank (Text and MTP blocks); the MoE leaves of
// a Flash-Next block have no placement in a dense model, and a dense model's mlp/* none in a MoE
// one.
void dense_policy() {
    for (const std::uint64_t hidden : {kH, std::uint64_t(5120)}) {
        auto config               = flash_next(4);
        config.text.hidden_size   = static_cast<std::uint32_t>(hidden);
        const std::uint64_t width = hidden == kH ? 6912 : 14336;
        config.text.ffn           = qwen::DenseConfig{static_cast<std::uint32_t>(width)};
        config.text.attention     = qwen::AttentionConfig{24, hidden == kH ? 2U : 4U, 256};
        for (const std::string p : {"text/layers/0/", "text/layers/3/", "mtp/layers/0/"}) {
            for (const auto* leaf : {"mlp/gate", "mlp/up"}) {
                expect(config, p + leaf, {width, hidden}, ShardAxis::Rows,
                       {halves(width, 0), halves(width, 1)});
            }
            expect(config, p + "mlp/down", {hidden, width}, ShardAxis::Columns,
                   {halves(width, 0), halves(width, 1)});
            rejects<std::invalid_argument>(
                [&] {
                    (void)loading::logical_shard(p + "moe/router", {512, hidden}, config, kTp2);
                },
                "a MoE router was placed for a dense Qwen4Exp FFN");
            rejects<std::invalid_argument>(
                [&] {
                    (void)loading::logical_shard(p + "moe/shared/gate", {640, hidden}, config,
                                                 kTp2);
                },
                "a shared expert was placed for a dense Qwen4Exp FFN");
        }
        const auto rank = qwen::execution::shard_text_config(config.text, 2);
        require(std::get<qwen::DenseConfig>(rank.ffn).intermediate_size == width / 2 &&
                    rank.attention->num_key_value_heads ==
                        config.text.attention->num_key_value_heads / 2 &&
                    !rank.qwen4_routed_experts() && rank.qwen4(),
                "per-rank dense Qwen4Exp config");
        auto no_ple     = config;
        no_ple.text.ple = std::nullopt;
        expect(no_ple, "text/layers/1/mlp/gate", {width, hidden}, ShardAxis::Rows,
               {halves(width, 0), halves(width, 1)});
    }
    rejects<std::invalid_argument>(
        [&] {
            (void)loading::logical_shard("text/layers/0/mlp/gate", {6912, kH}, flash_next(), kTp2);
        },
        "a dense FFN leaf was placed for a MoE Qwen4Exp");
}

// ---- combined placements of the real parents ---------------------------------------------------

struct Format {
    QType format;
    QuantLayout layout;
};

constexpr Format kNvfp4{QType::NVFP4, QuantLayout::BlockScaleK16M128x4};
constexpr Format kQ8{QType::Q8_G32_FP16, QuantLayout::RowSplit};
constexpr Format kQ6{QType::Q6_G64_FP16, QuantLayout::RowSplit};
constexpr Format kBf16{QType::BF16, QuantLayout::Contiguous};
constexpr Format kFp32{QType::FP32, QuantLayout::Contiguous};

struct ParentSpec {
    std::string id;
    Format format;
    artifact::Shape shape;
    std::vector<std::pair<std::string, std::uint64_t>> rows; // Logical parameter, row count.
    std::string kind;                                        // Inventory class.
};

struct Recipe {
    const char* name;
    Format dense; // attention q|k|gate|v and output, GDN q|k|v|z and output, shared expert
    Format head;
};

// The device parents of one block as the converter groups them (tools/convert/qwen4_exp.py).
void block_parents(std::vector<ParentSpec>& out, const std::string& p, bool attention, bool ple,
                   const Recipe& recipe) {
    for (const auto* mixer : {"attn_hc/", "ffn_hc/"}) {
        const std::string m = p + mixer;
        out.push_back({m + "norm", kBf16, {kWide}, {{m + "norm", kWide}}, "hc"});
        out.push_back(
            {m + "down|inject", kQ8, {324, kWide}, {{m + "down", 320}, {m + "inject", 4}}, "hc"});
        out.push_back({m + "up", kQ8, {kWide, 320}, {{m + "up", kWide}}, "hc"});
    }
    if (attention) {
        const std::string a = p + "attention/";
        out.push_back(
            {a + "q|k|gate|v",
             recipe.dense,
             {13312, kH},
             {{a + "query", 6144}, {a + "key", 512}, {a + "gate", 6144}, {a + "value", 512}},
             "attention"});
        out.push_back({a + "output", recipe.dense, {kH, 6144}, {{a + "output", kH}}, "attention"});
        out.push_back({a + "query_norm", kBf16, {256}, {{a + "query_norm", 256}}, "attention"});
        out.push_back({a + "key_norm", kBf16, {256}, {{a + "key_norm", 256}}, "attention"});
        out.push_back({a + "indexer",
                       kQ8,
                       {640, kH},
                       {{a + "indexer_query", 512}, {a + "indexer_key", 128}},
                       "indexer"});
        out.push_back(
            {a + "indexer_query_norm", kBf16, {128}, {{a + "indexer_query_norm", 128}}, "indexer"});
        out.push_back(
            {a + "indexer_key_norm", kBf16, {128}, {{a + "indexer_key_norm", 128}}, "indexer"});
    } else {
        const std::string g = p + "gdn/";
        out.push_back(
            {g + "q|k|v|z",
             recipe.dense,
             {16384, kH},
             {{g + "query", 2048}, {g + "key", 2048}, {g + "value", 6144}, {g + "z", 6144}},
             "gdn"});
        out.push_back({g + "a|b",
                       kQ8,
                       {96, kH},
                       {{g + "a_projection", 48}, {g + "b_projection", 48}},
                       "gdn"});
        out.push_back({g + "a_log", kFp32, {48}, {{g + "a_log", 48}}, "gdn"});
        out.push_back({g + "dt_bias", kFp32, {48}, {{g + "dt_bias", 48}}, "gdn"});
        out.push_back({g + "convolution", kBf16, {4, kWide}, {{g + "convolution", 4}}, "gdn"});
        out.push_back({g + "norm", kBf16, {128}, {{g + "norm", 128}}, "gdn"});
        out.push_back({g + "output", recipe.dense, {kH, 6144}, {{g + "output", kH}}, "gdn"});
    }
    const std::string m = p + "moe/";
    out.push_back({m + "router|shared_score",
                   kBf16,
                   {513, kH},
                   {{m + "router", 512}, {m + "shared_score", 1}},
                   "router"});
    out.push_back({m + "shared/gate|up",
                   recipe.dense,
                   {1280, kH},
                   {{m + "shared/gate", 640}, {m + "shared/up", 640}},
                   "shared expert"});
    out.push_back(
        {m + "shared/down", recipe.dense, {kH, 640}, {{m + "shared/down", kH}}, "shared expert"});
    if (ple) {
        const std::string e = p + "ple/";
        out.push_back(
            {e + "key|value", kQ8, {12800, kH}, {{e + "key", kWide}, {e + "value", kH}}, "ple"});
        for (const auto* leaf : {"key_norm", "query_norm", "conv_norm"}) {
            out.push_back({e + leaf, kBf16, {kWide}, {{e + leaf, kWide}}, "ple"});
        }
        out.push_back({e + "convolution", kBf16, {4, kWide}, {{e + "convolution", 4}}, "ple"});
    }
}

std::vector<ParentSpec> model_parents(const qwen::Config& config, const Recipe& recipe) {
    std::vector<ParentSpec> out;
    out.push_back({"text/token_embedding",
                   kQ8,
                   {kVocab, kH},
                   {{"text/token_embedding", kVocab}},
                   "embedding"});
    out.push_back({"text/output_head",
                   recipe.head,
                   {kVocab, kH},
                   {{"text/output_head", kVocab}},
                   "output head"});
    for (const std::string h : {"text/hc_head/", "mtp/hc_head/"}) {
        out.push_back({h + "norm", kBf16, {kWide}, {{h + "norm", kWide}}, "hc"});
        out.push_back({h + "down", kQ8, {320, kWide}, {{h + "down", 320}}, "hc"});
        out.push_back({h + "up", kQ8, {kWide, 320}, {{h + "up", kWide}}, "hc"});
    }
    const auto& text = config.text;
    for (std::uint32_t i = 0; i < text.num_hidden_layers; ++i) {
        block_parents(out, "text/layers/" + std::to_string(i) + "/",
                      text.layer_types[i] == qwen::MixerKind::FullAttention, i == text.ple->layer,
                      recipe);
    }
    block_parents(out, "mtp/layers/0/", true, false, recipe);
    out.push_back({"mtp/embedding_norm", kBf16, {kH}, {{"mtp/embedding_norm", kH}}, "mtp stem"});
    out.push_back({"mtp/hidden_norm", kBf16, {kWide}, {{"mtp/hidden_norm", kWide}}, "mtp stem"});
    for (const auto* name : {"mtp/embedding_projection", "mtp/hidden_projection"}) {
        out.push_back({name, kQ8, {kH, kH}, {{name, kH}}, "mtp stem"});
    }
    return out;
}

struct Totals {
    std::map<std::string, std::array<std::uint64_t, 2>> by_kind;
    std::array<std::uint64_t, 2> total{};
};

// Combines the placements of `parents`, slices every shard with the artifact slicer (which
// enforces the layout boundaries) and returns each device's placed bytes.
Totals place(const qwen::Config& config, const std::vector<ParentSpec>& parents,
             const LoadOptions& options, const std::map<std::string, artifact::Shape>& shards) {
    std::vector<WeightGeometry> geometry;
    std::vector<loading::PendingWeight> weights;
    for (std::size_t object = 0; object < parents.size(); ++object) {
        const auto& parent = parents[object];
        geometry.push_back(
            weight_geometry(parent.format.format, parent.format.layout, parent.shape));
        const auto row      = geometry.back().elements / parent.shape.front();
        std::uint64_t first = 0;
        for (const auto& [name, rows] : parent.rows) {
            auto shape    = parent.shape;
            shape.front() = rows;
            artifact::Binding binding{
                .whole_object = parent.rows.size() == 1,
                .parts    = {{artifact::ObjectHandle{object}, first * row, (first + rows) * row}},
                .elements = rows * row};
            weights.push_back({.reference = {name, shape, binding}, .source_objects = {parent.id}});
            first += rows;
        }
        require(first == parent.shape.front(), ("parent rows of " + parent.id).c_str());
    }
    const loading::GeometryLookup lookup =
        [&](artifact::ObjectHandle object) -> const WeightGeometry& {
        return geometry.at(object.index);
    };
    const auto placements =
        loading::parent_placements(weights, parents.size(), lookup, config, options);
    Totals totals;
    for (std::size_t object = 0; object < parents.size(); ++object) {
        const auto& placement = placements.at(object);
        require(placement.has_value(), ("no placement for " + parents[object].id).c_str());
        auto& kind = totals.by_kind[parents[object].kind];
        for (std::size_t device = 0; device < 2; ++device) {
            std::uint64_t bytes = 0;
            switch (placement->axis) {
            case ShardAxis::Rows:
            case ShardAxis::Columns: {
                const auto slice = artifact::tensor_slice(geometry[object], placement->axis,
                                                          placement->device_ranges[device]);
                require(slice.geometry.format == parents[object].format.format,
                        "a shard changed format");
                // Shards of one leaf name share their shape; check the ones the Ops consume.
                for (const auto& [suffix, shape] : shards) {
                    if (parents[object].id.ends_with(suffix)) {
                        require(slice.geometry.shape == shape,
                                ("rank shard shape of " + parents[object].id).c_str());
                    }
                }
                bytes = slice.geometry.bytes;
                break;
            }
            case ShardAxis::Replicated:
                bytes = geometry[object].bytes;
                break;
            case ShardAxis::SingleDevice:
                bytes = placement->device == int(device) ? geometry[object].bytes : 0;
                break;
            default:
                throw std::runtime_error(parents[object].id + ": unexpected placement kind");
            }
            kind[device] += bytes;
            totals.total[device] += bytes;
        }
    }
    return totals;
}

void print(const char* label, const Totals& totals, std::uint64_t whole) {
    std::printf("%s\n", label);
    for (const auto& [kind, bytes] : totals.by_kind) {
        std::printf("  %-14s rank0 %13llu B  rank1 %13llu B\n", kind.c_str(),
                    static_cast<unsigned long long>(bytes[0]),
                    static_cast<unsigned long long>(bytes[1]));
    }
    std::printf("  %-14s rank0 %13llu B  rank1 %13llu B  (tp 1: %llu B)\n", "total",
                static_cast<unsigned long long>(totals.total[0]),
                static_cast<unsigned long long>(totals.total[1]),
                static_cast<unsigned long long>(whole));
}

void combined_placements() {
    const auto config = flash_next();
    for (const Recipe& recipe :
         {Recipe{"qwen3_8_flash_next_nvfp4 (dense Q8, head Q6)", kQ8, kQ6},
          Recipe{"qwen3_8_flash_next_nvfp4_dense (dense/head NVFP4)", kNvfp4, kNvfp4}}) {
        const auto parents = model_parents(config, recipe);
        // Per-rank problems the split Ops are registered for.
        const std::map<std::string, artifact::Shape> shards{
            {"attention/q|k|gate|v", {6656, kH}},
            {"attention/output", {kH, 3072}},
            {"gdn/q|k|v|z", {8192, kH}},
            {"gdn/a|b", {48, kH}},
            {"gdn/a_log", {24}},
            {"gdn/convolution", {4, 5120}},
            {"gdn/output", {kH, 3072}},
            {"text/output_head", {kVocab / 2, kH}},
        };
        std::map<std::string, artifact::Shape> split_shards = shards;
        split_shards.insert({{"down|inject", {324, 5120}},
                             {"hc_head/down", {320, 5120}},
                             {"hc/up", {5120, 320}},
                             {"hc_head/up", {5120, 320}},
                             {"hc/norm", {5120}},
                             {"ple/convolution", {4, 5120}}});
        const auto tp1   = place(config, parents, LoadOptions{.tp = 1}, {});
        const auto tp2   = place(config, parents, kTp2, shards);
        const auto split = place(config, parents, kTp2Split, split_shards);
        std::printf("== %s, 48 blocks + MTP, device weights (routed experts host-mapped, PLE table "
                    "file-mapped)\n",
                    recipe.name);
        print("tp 2, HC replicated", tp2, tp1.total[0]);
        print("tp 2, HC split by streams", split, tp1.total[0]);
        const auto hc = tp2.by_kind.at("hc")[0] - split.by_kind.at("hc")[0];
        std::printf("  HC split frees %llu B per rank (HC %llu -> %llu B)\n",
                    static_cast<unsigned long long>(hc),
                    static_cast<unsigned long long>(tp2.by_kind.at("hc")[0]),
                    static_cast<unsigned long long>(split.by_kind.at("hc")[0]));
        // Shared experts alternate: 24 Text blocks + MTP on rank 0, 24 on rank 1.
        const auto& shared = tp2.by_kind.at("shared expert");
        require(shared[0] + shared[1] == tp1.by_kind.at("shared expert")[0] &&
                    shared[0] * 24 == shared[1] * 25,
                "shared experts do not alternate between the ranks");
        require(tp2.by_kind.at("hc")[0] == tp1.by_kind.at("hc")[0] &&
                    split.by_kind.at("hc")[0] * 2 == tp1.by_kind.at("hc")[0],
                "HC bytes per rank");
    }
}

} // namespace

int main() {
    try {
        logical_policy();
        dense_policy();
        combined_placements();
        std::cout << "qwen4 tensor-parallel shard map passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
