#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/formats.h"
#include "artifact/slices.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/load/sharding.h"

#include <array>
#include <cstdlib>
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
using ninfer::test::artifact_fixture::Json;
using ninfer::test::artifact_fixture::rejects;
using ninfer::test::artifact_fixture::require;
using Ranges = std::vector<SliceRange>;

// Qwen3.8-27B text geometry: 16 x (3 GDN + 1 attention) blocks, dense FFN, MTP.
qwen::Config qwen_27b() {
    qwen::Config config;
    auto& text             = config.text;
    text.architecture      = Architecture::Qwen3_5;
    text.hidden_size       = 5120;
    text.vocab_size        = 248320;
    text.num_hidden_layers = 64;
    for (std::uint32_t i = 0; i < text.num_hidden_layers; ++i) {
        text.layer_types.push_back(i % 4 == 3 ? qwen::MixerKind::FullAttention
                                              : qwen::MixerKind::LinearAttention);
    }
    text.attention = qwen::AttentionConfig{24, 4, 256};
    text.gdn       = qwen::GdnConfig{16, 128, 48, 128, 4};
    text.ffn       = qwen::DenseConfig{17408};
    config.mtp     = true;
    return config;
}

const LoadOptions kTp2{.tp = 2};

Ranges halves(std::uint64_t total, int device) {
    return {{static_cast<std::uint64_t>(device) * total / 2, total / 2}};
}

void expect(const qwen::Config& config, const std::string& name, artifact::Shape shape,
            ShardAxis axis, const std::array<Ranges, 2>& ranges = {}, int holder = 0,
            const LoadOptions& options = kTp2) {
    const auto shard = loading::logical_shard(name, shape, config, options);
    if (shard.axis != axis || shard.device != holder || shard.ranges[0] != ranges[0] ||
        shard.ranges[1] != ranges[1]) {
        throw std::runtime_error(name + ": unexpected tensor-parallel placement");
    }
}

void logical_policy_27b() {
    const auto config         = qwen_27b();
    constexpr std::uint64_t h = 5120, q = 6144, k = 1024, gk = 2048, gv = 6144, i = 17408;
    constexpr std::uint64_t vocab = 248320;
    for (const std::string p : {"text/layers/3/", "mtp/layers/0/"}) {
        // Head-sharded attention: 12 query and 2 KV heads of 256 per rank.
        for (const auto* leaf : {"attention/query", "attention/gate"}) {
            expect(config, p + leaf, {q, h}, ShardAxis::Rows, {halves(q, 0), halves(q, 1)});
        }
        for (const auto* leaf : {"attention/key", "attention/value"}) {
            expect(config, p + leaf, {k, h}, ShardAxis::Rows, {halves(k, 0), halves(k, 1)});
        }
        expect(config, p + "attention/output", {h, q}, ShardAxis::Columns,
               {halves(q, 0), halves(q, 1)});
        for (const auto* leaf : {"attention/query_norm", "attention/key_norm"}) {
            expect(config, p + leaf, {256}, ShardAxis::Replicated);
        }
    }
    for (const std::string p : {"text/layers/5/", "mtp/layers/0/"}) {
        for (const auto* leaf : {"input_norm", "post_attention_norm"}) {
            expect(config, p + leaf, {h}, ShardAxis::Replicated);
        }
        for (const auto* leaf : {"mlp/gate", "mlp/up"}) {
            expect(config, p + leaf, {i, h}, ShardAxis::Rows, {halves(i, 0), halves(i, 1)});
        }
        expect(config, p + "mlp/down", {h, i}, ShardAxis::Columns, {halves(i, 0), halves(i, 1)});
    }
    const std::string g = "text/layers/0/gdn/";
    for (const auto* leaf : {"query", "key"}) {
        expect(config, g + leaf, {gk, h}, ShardAxis::Rows, {halves(gk, 0), halves(gk, 1)});
    }
    for (const auto* leaf : {"value", "z"}) {
        expect(config, g + leaf, {gv, h}, ShardAxis::Rows, {halves(gv, 0), halves(gv, 1)});
    }
    // 48 value heads, 3 per key head: rank 0 holds value heads 0-23 with key heads 0-7.
    for (const auto* leaf : {"a_projection", "b_projection"}) {
        expect(config, g + leaf, {48, h}, ShardAxis::Rows, {halves(48, 0), halves(48, 1)});
    }
    for (const auto* leaf : {"a_log", "dt_bias"}) {
        expect(config, g + leaf, {48}, ShardAxis::Rows, {halves(48, 0), halves(48, 1)});
    }
    expect(config, g + "norm", {128}, ShardAxis::Replicated);
    expect(config, g + "output", {h, gv}, ShardAxis::Columns, {halves(gv, 0), halves(gv, 1)});
    // Depthwise channels Q|K|V: each rank keeps 1024 + 1024 + 3072 = 5120 of 10240.
    expect(config, g + "convolution", {4, 10240}, ShardAxis::Columns,
           {Ranges{{0, 1024}, {2048, 1024}, {4096, 3072}},
            Ranges{{1024, 1024}, {3072, 1024}, {7168, 3072}}});

    expect(config, "text/token_embedding", {vocab, h}, ShardAxis::Replicated);
    // --embedding-host: one mapped host copy that every rank reads, at tp 1 too.
    for (const int tp : {1, 2}) {
        const LoadOptions host{.tp = tp, .embedding_host = true};
        expect(config, "text/token_embedding", {vocab, h}, ShardAxis::HostMapped, {}, 0, host);
        expect(config, "text/final_norm", {h}, ShardAxis::Replicated, {}, 0, host);
    }
    expect(config, "text/final_norm", {h}, ShardAxis::Replicated);
    expect(config, "text/output_head", {vocab, h}, ShardAxis::Rows,
           {halves(vocab, 0), halves(vocab, 1)});
    // Rank 0 contracts the normalized embedding half, rank 1 the hidden half.
    expect(config, "mtp/input_projection", {h, 2 * h}, ShardAxis::Columns,
           {halves(2 * h, 0), halves(2 * h, 1)});
    for (const auto* name : {"mtp/embedding_norm", "mtp/hidden_norm", "mtp/final_norm"}) {
        expect(config, name, {h}, ShardAxis::Replicated);
    }
    for (const auto* name :
         {"dflash2/layers/0/mlp/gate", "dflash2/feature_projection", "dflash2/layers/1/input_norm",
          "dflash2/candidate_selector/successor_codebook", "proposal/head", "proposal/token_ids"}) {
        expect(config, name, {h}, ShardAxis::PrimaryOnly);
    }
    {
        // Under MTP the indexed optimized proposal head splits by vocabulary rows and its token
        // IDs stay on rank 0. Under DFlash2 the head and its token IDs split by the same rows
        // ([65536,5120] blocks for the split top sixteen). A head over the whole vocabulary,
        // NINFER_TP_DRAFT_HEAD=primary and, for DFlash2, NINFER_TP_DRAFTER=primary stay on rank 0.
        constexpr std::uint64_t proposal = 131072;
        const LoadOptions mtp{.speculative = SpeculativeBackend::Mtp, .tp = 2};
        const LoadOptions dflash2{.speculative = SpeculativeBackend::DFlash2, .tp = 2};
        expect(config, "proposal/head", {proposal, h}, ShardAxis::Rows,
               {halves(proposal, 0), halves(proposal, 1)}, 0, mtp);
        expect(config, "proposal/token_ids", {proposal}, ShardAxis::PrimaryOnly, {}, 0, mtp);
        expect(config, "proposal/head", {vocab, h}, ShardAxis::PrimaryOnly, {}, 0, mtp);
        expect(config, "proposal/head", {proposal, h}, ShardAxis::Rows,
               {halves(proposal, 0), halves(proposal, 1)}, 0, dflash2);
        expect(config, "proposal/token_ids", {proposal}, ShardAxis::Rows,
               {halves(proposal, 0), halves(proposal, 1)}, 0, dflash2);
        expect(config, "proposal/head", {vocab, h}, ShardAxis::PrimaryOnly, {}, 0, dflash2);
        expect(config, "proposal/head", {2 * proposal, h}, ShardAxis::PrimaryOnly, {}, 0, dflash2);
        ::setenv("NINFER_TP_DRAFT_HEAD", "primary", 1);
        expect(config, "proposal/head", {proposal, h}, ShardAxis::PrimaryOnly, {}, 0, mtp);
        expect(config, "proposal/head", {proposal, h}, ShardAxis::PrimaryOnly, {}, 0, dflash2);
        expect(config, "proposal/token_ids", {proposal}, ShardAxis::PrimaryOnly, {}, 0, dflash2);
        ::unsetenv("NINFER_TP_DRAFT_HEAD");
        ::setenv("NINFER_TP_DRAFTER", "primary", 1);
        expect(config, "proposal/head", {proposal, h}, ShardAxis::Rows,
               {halves(proposal, 0), halves(proposal, 1)}, 0, mtp);
        expect(config, "proposal/head", {proposal, h}, ShardAxis::PrimaryOnly, {}, 0, dflash2);
        expect(config, "proposal/token_ids", {proposal}, ShardAxis::PrimaryOnly, {}, 0, dflash2);
        ::unsetenv("NINFER_TP_DRAFTER");
    }
    {
        // The DFlash2 drafter splits under DFlash2 (NINFER_TP_DRAFTER=primary keeps it on rank 0):
        // attention by its 32/8 heads, MLP rows and down columns, conv kernels, norms and the
        // feature projection replicated, the candidate selector on rank 1 (rank 0 with Vision on
        // rank 1, NINFER_TP_DRAFT_HEAD=primary or NINFER_TP_SELECTOR=primary).
        auto drafted            = config;
        auto& draft             = drafted.draft.emplace();
        draft.attention         = qwen::AttentionConfig{32, 8, 128};
        draft.intermediate_size = 17408;
        draft.num_hidden_layers = 5;
        draft.sliding_window    = 2048;
        draft.dflash2           = qwen::DFlash2Config{2, 16, 256, 16};
        const LoadOptions dflash2{.speculative = SpeculativeBackend::DFlash2, .tp = 2};
        const std::string d = "dflash2/layers/3/";
        expect(drafted, d + "attention/query", {4096, h}, ShardAxis::Rows,
               {halves(4096, 0), halves(4096, 1)}, 0, dflash2);
        for (const auto* leaf : {"key", "value", "context_key", "context_value"}) {
            expect(drafted, d + "attention/" + leaf, {1024, h}, ShardAxis::Rows,
                   {halves(1024, 0), halves(1024, 1)}, 0, dflash2);
        }
        expect(drafted, d + "attention/output", {h, 4096}, ShardAxis::Columns,
               {halves(4096, 0), halves(4096, 1)}, 0, dflash2);
        for (const auto* leaf : {"mlp/gate", "mlp/up"}) {
            expect(drafted, d + leaf, {17408, h}, ShardAxis::Rows,
                   {halves(17408, 0), halves(17408, 1)}, 0, dflash2);
        }
        expect(drafted, d + "mlp/down", {h, 17408}, ShardAxis::Columns,
               {halves(17408, 0), halves(17408, 1)}, 0, dflash2);
        for (const auto* leaf :
             {"input_norm", "post_attention_norm", "attention/query_norm", "attention/key_norm",
              "attention_conv/base_kernel", "attention_conv/kernel_projection",
              "mlp_conv/base_kernel", "mlp_conv/kernel_projection"}) {
            expect(drafted, d + leaf, {h}, ShardAxis::Replicated, {}, 0, dflash2);
        }
        for (const auto* name :
             {"dflash2/feature_projection", "dflash2/context_norm", "dflash2/final_norm"}) {
            expect(drafted, name, {h}, ShardAxis::Replicated, {}, 0, dflash2);
        }
        for (const auto* name : {"dflash2/candidate_selector/successor_codebook",
                                 "dflash2/candidate_selector/hidden_projection"}) {
            expect(drafted, name, {vocab, 256}, ShardAxis::SingleDevice, {}, 1, dflash2);
            expect(drafted, name, {vocab, 256}, ShardAxis::SingleDevice, {}, 1,
                   {.vision = true, .speculative = SpeculativeBackend::DFlash2, .tp = 2});
            expect(drafted, name, {vocab, 256}, ShardAxis::PrimaryOnly, {}, 0,
                   {.vision      = true,
                    .speculative = SpeculativeBackend::DFlash2,
                    .tp          = 2,
                    .vision_rank = 1});
            for (const auto* variable : {"NINFER_TP_SELECTOR", "NINFER_TP_DRAFT_HEAD"}) {
                ::setenv(variable, "primary", 1);
                expect(drafted, name, {vocab, 256}, ShardAxis::PrimaryOnly, {}, 0, dflash2);
                ::unsetenv(variable);
            }
        }
        // Under MTP, or with NINFER_TP_DRAFTER=primary, the drafter stays whole on rank 0.
        expect(drafted, d + "mlp/gate", {17408, h}, ShardAxis::PrimaryOnly, {}, 0,
               {.speculative = SpeculativeBackend::Mtp, .tp = 2});
        ::setenv("NINFER_TP_DRAFTER", "primary", 1);
        expect(drafted, d + "attention/query", {4096, h}, ShardAxis::PrimaryOnly, {}, 0, dflash2);
        expect(drafted, "dflash2/feature_projection", {h}, ShardAxis::PrimaryOnly, {}, 0, dflash2);
        ::unsetenv("NINFER_TP_DRAFTER");
    }
    expect(config, "vision/layers/0/attention/query", {1152, 1152}, ShardAxis::SingleDevice, {}, 1,
           {.tp = 2, .vision_rank = 1});

    // tp 1 places nothing, including names outside the tensor-parallel policy.
    for (const auto* name : {"text/layers/3/attention/query", "text/unknown"}) {
        expect(config, name, {q, h}, ShardAxis::Replicated, {}, 0, {});
    }
    rejects<std::invalid_argument>(
        [&] { (void)loading::logical_shard("text/unknown", {h}, config, kTp2); },
        "a parameter without a tensor-parallel rule was placed");
    rejects<std::invalid_argument>(
        [&] { (void)loading::logical_shard("text/final_norm", {h}, config, {.tp = 3}); },
        "tensor parallelism above two was accepted");
    auto odd                                = config;
    odd.text.attention->num_key_value_heads = 3;
    rejects<std::invalid_argument>(
        [&] { (void)loading::logical_shard("text/layers/3/attention/key", {768, h}, odd, kTp2); },
        "KV heads that do not divide by tp were split");
}

// Fused parents of a Qwen3.8-27B NVFP4 artifact, with their logical parameters in row order.
struct ParentSpec {
    std::string id;
    QType format;
    QuantLayout layout;
    artifact::Shape shape;
    std::vector<std::pair<std::string, std::uint64_t>> rows; // Logical parameter, row count.
};

void combined_placements_27b() {
    const auto config      = qwen_27b();
    const auto nvfp4       = std::pair{QType::NVFP4, QuantLayout::BlockScaleK16M128x4};
    const auto fp8         = std::pair{QType::FP8_E4M3FN_ROW_BF16, QuantLayout::RowScale};
    const auto bf16        = std::pair{QType::BF16, QuantLayout::Contiguous};
    const auto q6          = std::pair{QType::Q6_G64_FP16, QuantLayout::RowSplit};
    const std::string a    = "text/layers/3/";
    const std::string g    = "text/layers/0/";
    const std::string late = "text/layers/60/";
    const auto spec        = [](std::string id, std::pair<QType, QuantLayout> format,
                         artifact::Shape shape,
                         std::vector<std::pair<std::string, std::uint64_t>> rows) {
        return ParentSpec{std::move(id), format.first, format.second, std::move(shape),
                          std::move(rows)};
    };
    const std::vector<ParentSpec> specs{
        spec("qkgv", nvfp4, {14336, 5120},
             {{a + "attention/query", 6144},
              {a + "attention/key", 1024},
              {a + "attention/gate", 6144},
              {a + "attention/value", 1024}}),
        spec("qkvz", fp8, {16384, 5120},
             {{g + "gdn/query", 2048},
              {g + "gdn/key", 2048},
              {g + "gdn/value", 6144},
              {g + "gdn/z", 6144}}),
        spec("ab", bf16, {96, 5120}, {{g + "gdn/a_projection", 48}, {g + "gdn/b_projection", 48}}),
        spec("a_log", {QType::FP32, QuantLayout::Contiguous}, {48}, {{g + "gdn/a_log", 48}}),
        spec("conv", bf16, {4, 10240}, {{g + "gdn/convolution", 4}}),
        spec("gdn_norm", bf16, {128}, {{g + "gdn/norm", 128}}),
        spec("attention_output", fp8, {5120, 6144}, {{a + "attention/output", 5120}}),
        spec("gdn_output", nvfp4, {5120, 6144}, {{g + "gdn/output", 5120}}),
        spec("gate_up", nvfp4, {34816, 5120}, {{a + "mlp/gate", 17408}, {a + "mlp/up", 17408}}),
        spec("down", nvfp4, {5120, 17408}, {{a + "mlp/down", 5120}}),
        spec("late_down", q6, {5120, 17408}, {{late + "mlp/down", 5120}}),
        spec("head", fp8, {248320, 5120}, {{"text/output_head", 248320}}),
        spec("embedding", bf16, {248320, 5120}, {{"text/token_embedding", 248320}}),
        spec("mtp_fc", fp8, {5120, 10240}, {{"mtp/input_projection", 5120}}),
        spec("proposal", q6, {131072, 5120}, {{"proposal/head", 131072}}),
    };
    std::vector<WeightGeometry> geometry;
    std::vector<loading::PendingWeight> weights;
    for (std::size_t object = 0; object < specs.size(); ++object) {
        const auto& parent = specs[object];
        geometry.push_back(weight_geometry(parent.format, parent.layout, parent.shape));
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
    }
    const loading::GeometryLookup lookup =
        [&](artifact::ObjectHandle object) -> const WeightGeometry& {
        return geometry.at(object.index);
    };
    const auto placements = loading::parent_placements(weights, specs.size(), lookup, config, kTp2);

    const std::map<std::string, std::pair<ShardAxis, std::array<Ranges, 2>>> expected{
        {"qkgv",
         {ShardAxis::Rows,
          {Ranges{{0, 3072}, {6144, 512}, {7168, 3072}, {13312, 512}},
           Ranges{{3072, 3072}, {6656, 512}, {10240, 3072}, {13824, 512}}}}},
        {"qkvz",
         {ShardAxis::Rows,
          {Ranges{{0, 1024}, {2048, 1024}, {4096, 3072}, {10240, 3072}},
           Ranges{{1024, 1024}, {3072, 1024}, {7168, 3072}, {13312, 3072}}}}},
        {"ab", {ShardAxis::Rows, {Ranges{{0, 24}, {48, 24}}, Ranges{{24, 24}, {72, 24}}}}},
        {"a_log", {ShardAxis::Rows, {halves(48, 0), halves(48, 1)}}},
        {"conv",
         {ShardAxis::Columns,
          {Ranges{{0, 1024}, {2048, 1024}, {4096, 3072}},
           Ranges{{1024, 1024}, {3072, 1024}, {7168, 3072}}}}},
        {"gdn_norm", {ShardAxis::Replicated, {}}},
        {"attention_output", {ShardAxis::Columns, {halves(6144, 0), halves(6144, 1)}}},
        {"gdn_output", {ShardAxis::Columns, {halves(6144, 0), halves(6144, 1)}}},
        {"gate_up",
         {ShardAxis::Rows,
          {Ranges{{0, 8704}, {17408, 8704}}, Ranges{{8704, 8704}, {26112, 8704}}}}},
        {"down", {ShardAxis::Columns, {halves(17408, 0), halves(17408, 1)}}},
        {"late_down", {ShardAxis::Columns, {halves(17408, 0), halves(17408, 1)}}},
        {"head", {ShardAxis::Rows, {halves(248320, 0), halves(248320, 1)}}},
        {"embedding", {ShardAxis::Replicated, {}}},
        {"mtp_fc", {ShardAxis::Columns, {halves(10240, 0), halves(10240, 1)}}},
        {"proposal", {ShardAxis::PrimaryOnly, {}}},
    };
    // Rank shard shapes that the split Ops consume.
    const std::map<std::string, artifact::Shape> shard_shapes{
        {"qkgv", {7168, 5120}},       {"qkvz", {8192, 5120}},
        {"ab", {48, 5120}},           {"a_log", {24}},
        {"conv", {4, 5120}},          {"attention_output", {5120, 3072}},
        {"gdn_output", {5120, 3072}}, {"gate_up", {17408, 5120}},
        {"down", {5120, 8704}},       {"late_down", {5120, 8704}},
        {"head", {124160, 5120}},     {"mtp_fc", {5120, 5120}},
    };
    for (std::size_t object = 0; object < specs.size(); ++object) {
        const auto& id             = specs[object].id;
        const auto& placement      = placements.at(object);
        const auto& [axis, ranges] = expected.at(id);
        if (!placement || placement->axis != axis || placement->device_ranges[0] != ranges[0] ||
            placement->device_ranges[1] != ranges[1]) {
            throw std::runtime_error(id + ": unexpected combined parent placement");
        }
        if (!artifact::is_sharded(axis)) { continue; }
        // The slicer enforces the layout boundaries: 128-row NVFP4 tiles, 64-column NVFP4 and
        // 128-column RowSplit groups. Every rank receives the same narrowed shape.
        for (std::size_t device = 0; device < 2; ++device) {
            const auto slice = artifact::tensor_slice(geometry[object], axis, ranges[device]);
            require(slice.geometry.shape == shard_shapes.at(id) &&
                        slice.geometry.format == specs[object].format,
                    "a rank shard has an unexpected shape");
        }
    }

    // A tied embedding and head share one parent with incompatible placements.
    auto tied = weights;
    for (auto& weight : tied) {
        if (weight.reference.name == "text/token_embedding") {
            weight.reference.binding.parts[0].object = artifact::ObjectHandle{11};
        }
    }
    rejects<std::invalid_argument>(
        [&] { (void)loading::parent_placements(tied, specs.size(), lookup, config, kTp2); },
        "a tied embedding and vocabulary-split head were combined");
    // A row split that does not reach whole rows of a flattened parent.
    auto misaligned = weights;
    for (auto& weight : misaligned) {
        if (weight.reference.name == g + "gdn/a_log") {
            weight.reference.shape                   = {48};
            weight.reference.binding.parts[0].object = artifact::ObjectHandle{2};
        }
    }
    rejects<std::invalid_argument>(
        [&] { (void)loading::parent_placements(misaligned, specs.size(), lookup, config, kTp2); },
        "a row split inside a parent row was accepted");
}

// A two-block dense Qwen3.5 artifact whose heads and widths divide by two.
struct ModelFixture {
    ninfer::test::artifact_fixture::Fixture file;

    void tensor(const std::string& id, artifact::Shape shape, QType format = QType::BF16,
                QuantLayout layout = QuantLayout::Contiguous) {
        const auto geometry = weight_geometry(format, layout, shape);
        const auto offset   = (file.payload.size() + 255) / 256 * 256;
        file.payload.resize(offset + geometry.bytes);
        file.root["objects"].push_back({{"id", id},
                                        {"kind", "tensor"},
                                        {"shape", shape},
                                        {"format", artifact::format_name(format)},
                                        {"layout", artifact::layout_name(layout)},
                                        {"offset", offset},
                                        {"bytes", geometry.bytes}});
    }

    void resource(const std::string& role, const std::string& bytes) {
        const auto offset = file.payload.size();
        for (const auto byte : bytes) { file.payload.push_back(std::byte(byte)); }
        file.root["objects"].push_back({{"id", role},
                                        {"kind", "resource"},
                                        {"encoding", "raw_bytes_v1"},
                                        {"offset", offset},
                                        {"bytes", bytes.size()}});
        file.root["components"]["text"]["resources"][role] = role;
    }

    void use(const std::string& name, const std::string& input) {
        file.root["uses"].push_back(
            {{"parameter", name}, {"input", input}, {"activation_policy", "A16Only"}});
    }

    // Logical parameters over consecutive rows of one parent.
    void rows(const std::string& id, artifact::Shape shape, QType format, QuantLayout layout,
              const std::vector<std::pair<std::string, std::uint64_t>>& parts,
              const std::string& input = {}) {
        tensor(id, shape, format, layout);
        const auto row      = shape.size() > 1 ? shape[1] : 1;
        std::uint64_t first = 0;
        for (const auto& [name, count] : parts) {
            file.root["bindings"][name] = {
                {"parts",
                 Json::array({{{"object", id}, {"range", {first * row, (first + count) * row}}}})}};
            if (!input.empty()) { use(name, input); }
            first += count;
        }
    }

    void parameter(const std::string& name, artifact::Shape shape, const std::string& input = {},
                   QType format = QType::BF16, QuantLayout layout = QuantLayout::Contiguous) {
        tensor(name, shape, format, layout);
        file.root["bindings"][name] = {{"object", name}};
        if (!input.empty()) { use(name, input); }
    }

    ModelFixture() {
        file.payload.clear();
        file.root      = {{"components",
                           {{"text",
                             {{"config",
                               {{"architectures", {"Qwen3_5ForCausalLM"}},
                                {"model_type", "qwen3_5_text"},
                                {"hidden_size", 128},
                                {"vocab_size", 272},
                                {"num_hidden_layers", 2},
                                {"max_position_embeddings", 128},
                                {"tie_word_embeddings", false},
                                {"rms_norm_eps", 1e-6},
                                {"layer_types", {"linear_attention", "full_attention"}},
                                {"num_attention_heads", 2},
                                {"num_key_value_heads", 2},
                                {"head_dim", 8},
                                {"linear_num_key_heads", 2},
                                {"linear_key_head_dim", 8},
                                {"linear_num_value_heads", 4},
                                {"linear_value_head_dim", 8},
                                {"linear_conv_kernel_dim", 4},
                                {"intermediate_size", 256},
                                {"rope_parameters",
                                 {{"rope_theta", 10000},
                                  {"partial_rotary_factor", 0.5},
                                  {"mrope_section", {1, 1, 0}}}}}}}}}},
                          {"objects", Json::array()},
                          {"bindings", Json::object()},
                          {"uses", Json::array()},
                          {"metadata", {{"name", "tensor-parallel-fixture"}}}};
        const auto fp8 = std::pair{QType::FP8_E4M3FN_ROW_BF16, QuantLayout::RowScale};
        parameter("text/token_embedding", {272, 128});
        parameter("text/output_head", {272, 128}, "text/final_hidden", fp8.first, fp8.second);
        parameter("text/final_norm", {128});
        for (int layer = 0; layer < 2; ++layer) {
            const auto p = "text/layers/" + std::to_string(layer) + "/";
            parameter(p + "input_norm", {128});
            parameter(p + "post_attention_norm", {128});
            if (layer == 0) {
                rows(p + "qkvz", {96, 128}, fp8.first, fp8.second,
                     {{p + "gdn/query", 16},
                      {p + "gdn/key", 16},
                      {p + "gdn/value", 32},
                      {p + "gdn/z", 32}},
                     p + "mixer_input");
                rows(p + "ab", {8, 128}, QType::BF16, QuantLayout::Contiguous,
                     {{p + "gdn/a_projection", 4}, {p + "gdn/b_projection", 4}}, p + "mixer_input");
                parameter(p + "gdn/a_log", {4}, {}, QType::FP32);
                parameter(p + "gdn/dt_bias", {4}, {}, QType::FP32);
                parameter(p + "gdn/convolution", {4, 64});
                parameter(p + "gdn/norm", {8});
                parameter(p + "gdn/output", {128, 32}, p + "gdn/gated_output", fp8.first,
                          fp8.second);
            } else {
                rows(p + "qkgv", {64, 128}, fp8.first, fp8.second,
                     {{p + "attention/query", 16},
                      {p + "attention/key", 16},
                      {p + "attention/gate", 16},
                      {p + "attention/value", 16}},
                     p + "mixer_input");
                parameter(p + "attention/query_norm", {8});
                parameter(p + "attention/key_norm", {8});
                parameter(p + "attention/output", {128, 16}, p + "attention/gated_output");
            }
            rows(p + "gate_up", {512, 128}, QType::BF16, QuantLayout::Contiguous,
                 {{p + "mlp/gate", 256}, {p + "mlp/up", 256}}, p + "ffn_input");
            parameter(p + "mlp/down", {128, 256}, p + "mlp/product", QType::Q6_G64_FP16,
                      QuantLayout::RowSplit);
        }

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
            const Json token{{"id", 256 + i},        {"content", specials[i]}, {"special", true},
                             {"single_word", false}, {"lstrip", false},        {"rstrip", false},
                             {"normalized", false}};
            if (i < 8) { added.push_back(token); }
            decoder[std::to_string(256 + i)] = token;
        }
        resource("tokenizer.json",
                 Json{{"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", Json::array()}}},
                      {"added_tokens", added}}
                     .dump());
        resource("tokenizer_config.json", Json{{"added_tokens_decoder", decoder}}.dump());
        resource("generation_config.json", Json{{"eos_token_id", {256, 258}}}.dump());
        resource("chat_template.jinja", "template");
        file.root["files"] =
            Json::array({{{"path", nullptr}, {"payload_bytes", file.payload.size()}}});
    }
};

void planned_fixture() {
    ModelFixture fixture;
    fixture.file.write();
    artifact::Reader reader(fixture.file.entry);

    const auto single = qwen::plan_load(reader);
    require(single.materialization().device_count == 1 &&
                single.materialization().per_device_capacity_bytes[1] == 0,
            "tp 1 planned a second device");
    for (const auto& item : single.materialization().device_objects) {
        require(item.device == 0 && item.axis == ShardAxis::Replicated && item.ranges.empty() &&
                    item.copies.empty(),
                "tp 1 placed a shard");
    }

    const auto plan             = qwen::plan_load(reader, kTp2);
    const auto& materialization = plan.materialization();
    require(materialization.device_count == 2, "tp 2 did not plan two devices");
    const std::map<std::string, std::pair<ShardAxis, std::array<Ranges, 2>>> expected{
        {"text/token_embedding", {ShardAxis::Replicated, {}}},
        {"text/output_head", {ShardAxis::Rows, {halves(272, 0), halves(272, 1)}}},
        {"text/layers/0/qkvz",
         {ShardAxis::Rows,
          {Ranges{{0, 8}, {16, 8}, {32, 16}, {64, 16}},
           Ranges{{8, 8}, {24, 8}, {48, 16}, {80, 16}}}}},
        {"text/layers/0/ab", {ShardAxis::Rows, {Ranges{{0, 2}, {4, 2}}, Ranges{{2, 2}, {6, 2}}}}},
        {"text/layers/0/gdn/a_log", {ShardAxis::Rows, {halves(4, 0), halves(4, 1)}}},
        {"text/layers/0/gdn/convolution",
         {ShardAxis::Columns,
          {Ranges{{0, 8}, {16, 8}, {32, 16}}, Ranges{{8, 8}, {24, 8}, {48, 16}}}}},
        {"text/layers/0/gdn/output", {ShardAxis::Columns, {halves(32, 0), halves(32, 1)}}},
        {"text/layers/1/qkgv",
         {ShardAxis::Rows,
          {Ranges{{0, 8}, {16, 8}, {32, 8}, {48, 8}}, Ranges{{8, 8}, {24, 8}, {40, 8}, {56, 8}}}}},
        {"text/layers/1/attention/output", {ShardAxis::Columns, {halves(16, 0), halves(16, 1)}}},
        {"text/layers/0/gdn/dt_bias", {ShardAxis::Rows, {halves(4, 0), halves(4, 1)}}},
        {"text/layers/0/gate_up",
         {ShardAxis::Rows, {Ranges{{0, 128}, {256, 128}}, Ranges{{128, 128}, {384, 128}}}}},
        {"text/layers/1/gate_up",
         {ShardAxis::Rows, {Ranges{{0, 128}, {256, 128}}, Ranges{{128, 128}, {384, 128}}}}},
        {"text/layers/0/mlp/down", {ShardAxis::Columns, {halves(256, 0), halves(256, 1)}}},
        {"text/layers/1/mlp/down", {ShardAxis::Columns, {halves(256, 0), halves(256, 1)}}},
        {"text/layers/1/attention/key_norm", {ShardAxis::Replicated, {}}},
    };
    std::map<std::string, int> seen;
    for (const auto& item : materialization.device_objects) {
        const auto& id    = artifact::object_id(reader.directory().object(item.object));
        const auto found  = expected.find(id);
        const auto device = static_cast<std::size_t>(item.device);
        ++seen[id];
        require(item.offset % item.alignment == 0 &&
                    item.offset + item.bytes <= materialization.per_device_capacity_bytes[device],
                "a device placement is outside its arena");
        if (found == expected.end()) {
            require(!artifact::is_sharded(item.axis), "an unlisted parent was sharded");
            continue;
        }
        const auto& [axis, ranges] = found->second;
        require(item.axis == axis && item.ranges == ranges[device],
                "a planned parent has an unexpected placement");
    }
    for (const auto& [id, count] : seen) {
        require(count == 2, "a tensor-parallel parent is missing on one device");
    }
    require(materialization.per_device_capacity_bytes[0] ==
                materialization.per_device_capacity_bytes[1],
            "symmetric shards planned unequal device arenas");

    // --embedding-host: the embedding parent leaves every device arena for one host-mapped copy.
    const auto host_mapped = [&](const qwen::LoadPlan& base, const qwen::LoadPlan& host) {
        const auto& moved     = host.materialization();
        const auto& reference = base.materialization();
        const auto embedding  = reader.find("text/token_embedding");
        const auto bytes      = reader.geometry(embedding).bytes;
        require(moved.host_mapped_objects.size() == 1 &&
                    moved.host_mapped_objects[0].object == embedding &&
                    moved.host_mapped_objects[0].offset == 0 && moved.host_mapped_bytes == bytes,
                "--embedding-host did not plan one host-mapped embedding copy");
        for (const auto& item : moved.device_objects) {
            require(item.object != embedding, "--embedding-host left an embedding device copy");
        }
        for (int device = 0; device < moved.device_count; ++device) {
            const auto d = static_cast<std::size_t>(device);
            require(moved.per_device_capacity_bytes[d] < reference.per_device_capacity_bytes[d] &&
                        reference.per_device_capacity_bytes[d] -
                                moved.per_device_capacity_bytes[d] >=
                            bytes,
                    "--embedding-host did not free the embedding bytes of a device arena");
        }
    };
    host_mapped(single, qwen::plan_load(reader, {.embedding_host = true}));
    host_mapped(plan, qwen::plan_load(reader, {.tp = 2, .embedding_host = true}));

    rejects<std::invalid_argument>([&] { (void)qwen::plan_load(reader, {.tp = 3}); },
                                   "tensor parallelism above two was planned");
    rejects<std::invalid_argument>(
        [&] { (void)qwen::plan_load(reader, {.tp = 2, .vision_rank = 2}); },
        "a Vision rank outside the plan was accepted");
    rejects<std::invalid_argument>(
        [&] {
            (void)qwen::plan_load(reader, {.speculative = SpeculativeBackend::DFlash2, .tp = 2});
        },
        "a rank-0 drafter reading the vocabulary-split head was planned");
}

} // namespace

int main() {
    try {
        logical_policy_27b();
        combined_placements_27b();
        planned_fixture();
        std::cout << "qwen3_5 tensor-parallel shard map passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
