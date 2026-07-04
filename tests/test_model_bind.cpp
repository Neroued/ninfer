#include "qus/core/arena.h"
#include "qus/core/device.h"
#include "qus/core/kv_cache.h"
#include "qus/core/state_store.h"
#include "qus/core/weight_store.h"
#include "qus/core/weight_store_parser.h"
#include "qus/model/config.h"
#include "qus/model/model.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int fail(std::string message) {
    std::cerr << message << '\n';
    return 1;
}

std::filesystem::path make_fixture() {
    const auto path = std::filesystem::temp_directory_path() / "qus_q5090_model_bind_fixture.qus";
    const std::filesystem::path script =
        std::filesystem::path(QUS_SOURCE_DIR) / "tests/fixtures/make_q5090_fixture.py";
    const std::string command =
        "python3 \"" + script.string() + "\" --profile model-bind --out \"" + path.string() + "\"";
    if (std::system(command.c_str()) != 0) { throw std::runtime_error("fixture generator failed"); }
    return path;
}

qus::Q5090Expectations expectations() {
    qus::Q5090Expectations expected;
    expected.layer_count             = 64;
    expected.hidden_size             = 5120;
    expected.intermediate_size       = 17408;
    expected.vocab_size              = 248320;
    expected.num_attention_heads     = 24;
    expected.num_key_value_heads     = 4;
    expected.head_dim                = 256;
    expected.gdn_key_heads           = 16;
    expected.gdn_value_heads         = 48;
    expected.gdn_key_head_dim        = 128;
    expected.gdn_value_head_dim      = 128;
    expected.gdn_conv_width          = 4;
    expected.full_attention_interval = 4;
    expected.max_position_embeddings = 262144;
    return expected;
}

int expect_weight(const qus::Weight* w, qus::SourceKind kind, std::uint32_t layer, qus::QType qtype,
                  qus::QuantLayout layout, std::int32_t n, std::int32_t k, std::string_view label,
                  qus::ModuleKind module = qus::ModuleKind::TextCore) {
    int failures = 0;
    if (w == nullptr) { return fail(std::string(label) + " is null"); }
    failures += w->module == module ? 0 : fail(std::string(label) + " module");
    failures += w->source_kind == static_cast<std::uint32_t>(kind)
                    ? 0
                    : fail(std::string(label) + " source_kind");
    failures += w->source_layer == layer ? 0 : fail(std::string(label) + " source_layer");
    failures += w->qtype == qtype ? 0 : fail(std::string(label) + " qtype");
    failures += w->layout == layout ? 0 : fail(std::string(label) + " layout");
    failures += w->n == n ? 0 : fail(std::string(label) + " n");
    failures += w->k == k ? 0 : fail(std::string(label) + " k");
    failures += w->shape[0] == n ? 0 : fail(std::string(label) + " shape[0]");
    failures += w->shape[1] == k ? 0 : fail(std::string(label) + " shape[1]");
    failures += w->payload != nullptr ? 0 : fail(std::string(label) + " payload");
    failures += w->qdata != nullptr ? 0 : fail(std::string(label) + " qdata");
    if (layout == qus::QuantLayout::RowSplit) {
        failures += w->scales != nullptr ? 0 : fail(std::string(label) + " scales");
        if (qtype == qus::QType::Q5G64_F16S || qtype == qus::QType::Q6G64_F16S) {
            failures += w->qhigh != nullptr ? 0 : fail(std::string(label) + " qhigh");
            failures +=
                w->high_plane_bytes > 0 ? 0 : fail(std::string(label) + " high_plane_bytes");
        } else {
            failures += w->qhigh == nullptr ? 0 : fail(std::string(label) + " qhigh null");
            failures +=
                w->high_plane_bytes == 0 ? 0 : fail(std::string(label) + " high_plane_bytes zero");
        }
    } else {
        failures += w->qhigh == nullptr ? 0 : fail(std::string(label) + " dense qhigh null");
        failures += w->high_plane_bytes == 0
                        ? 0
                        : fail(std::string(label) + " dense high_plane_bytes zero");
    }
    return failures;
}

int expect_tensor(const qus::Tensor* t, qus::DType dtype, std::initializer_list<std::int32_t> shape,
                  std::string_view label) {
    int failures = 0;
    if (t == nullptr) { return fail(std::string(label) + " is null"); }
    failures += t->data != nullptr ? 0 : fail(std::string(label) + " data");
    failures += t->dtype == dtype ? 0 : fail(std::string(label) + " dtype");
    int i = 0;
    for (const std::int32_t dim : shape) {
        failures += t->ne[i] == dim ? 0 : fail(std::string(label) + " shape");
        ++i;
    }
    failures += t->is_contiguous() ? 0 : fail(std::string(label) + " contiguous");
    return failures;
}

int expect_mlp(const qus::model::MlpW& mlp, std::uint32_t layer, std::string_view label) {
    int failures = 0;
    failures += expect_weight(mlp.gate, qus::SourceKind::MlpGate, layer, qus::QType::Q4G64_F16S,
                              qus::QuantLayout::RowSplit, 8, 8, std::string(label) + ".gate");
    failures += expect_weight(mlp.up, qus::SourceKind::MlpUp, layer, qus::QType::Q4G64_F16S,
                              qus::QuantLayout::RowSplit, 8, 8, std::string(label) + ".up");
    failures += expect_weight(mlp.down, qus::SourceKind::MlpDown, layer, qus::QType::Q5G64_F16S,
                              qus::QuantLayout::RowSplit, 8, 8, std::string(label) + ".down");
    return failures;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 0;
    }

    int failures                             = 0;
    const std::filesystem::path fixture_path = make_fixture();

    qus::DeviceContext ctx(0);
    qus::DeviceArena weights_arena(128ULL * 1024ULL * 1024ULL);
    qus::DeviceArena cache_arena(256ULL * 1024ULL * 1024ULL);
    qus::WorkspaceArena workspace(8ULL * 1024ULL * 1024ULL);
    qus::DeviceArena io_arena(1024ULL * 1024ULL);

    qus::WeightStore store(expectations());
    qus::LoadOptions load_options;
    load_options.load_mtp = true;
    store.load(fixture_path.c_str(), weights_arena, ctx, load_options);
    const qus::Tensor* stored_conv1d = store.tensor(
        qus::ModuleKind::TextCore, static_cast<std::uint32_t>(qus::SourceKind::GdnConv1d), 0);
    failures += expect_tensor(stored_conv1d, qus::DType::BF16, {10240, 4, 1}, "store.gdn.conv1d");
    qus::KVCache kv(cache_arena, qus::model::kCfg.n_full(), 1, qus::model::kCfg.n_kv,
                    qus::model::kCfg.head_dim);
    qus::GdnState state(cache_arena, qus::model::kCfg.n_gdn(), qus::model::kCfg.conv_dim,
                        qus::model::kCfg.gdn_conv_state_width, qus::model::kCfg.gdn_v_heads,
                        qus::model::kCfg.gdn_v_dim, qus::model::kCfg.gdn_k_dim);
    qus::KVCache mtp_kv(cache_arena, qus::model::kCfg.mtp_layers, 1, qus::model::kCfg.n_kv,
                        qus::model::kCfg.head_dim);
    qus::model::StepState io{};
    io.token            = io_arena.alloc(qus::DType::I32, {1});
    io.pos              = io_arena.alloc(qus::DType::I32, {1});
    io.logits           = io_arena.alloc(qus::DType::BF16, {qus::model::kCfg.vocab});
    io.gdn_initial_slot = io_arena.alloc(qus::DType::I32, {1});
    CUDA_CHECK(cudaMemsetAsync(io.gdn_initial_slot.data, 0, io.gdn_initial_slot.bytes(),
                               ctx.stream));
    qus::model::Qwen3_6_27B card(ctx, store, workspace, kv, state, io, 512, &mtp_kv);

    failures += expect_weight(card.embed(), qus::SourceKind::Embed, qus::kQ5090NoLayer,
                              qus::QType::Q6G64_F16S, qus::QuantLayout::RowSplit, 16, 8, "embed");
    failures += expect_tensor(card.final_norm(), qus::DType::BF16, {5120}, "final_norm");
    failures += expect_weight(card.lm_head(), qus::SourceKind::LmHead, qus::kQ5090NoLayer,
                              qus::QType::Q6G64_F16S, qus::QuantLayout::RowSplit, 16, 8, "lm_head");

    const qus::model::FullLayerW& full = card.full_layer(0);
    failures += expect_tensor(full.input_norm, qus::DType::BF16, {5120}, "full.input_norm");
    failures += expect_weight(full.q_proj, qus::SourceKind::AttnQ, 3, qus::QType::Q4G64_F16S,
                              qus::QuantLayout::RowSplit, 8, 8, "full.q_proj");
    failures += expect_weight(full.gate_proj, qus::SourceKind::AttnGate, 3, qus::QType::Q5G64_F16S,
                              qus::QuantLayout::RowSplit, 8, 8, "full.gate_proj");
    failures += expect_weight(full.k_proj, qus::SourceKind::AttnK, 3, qus::QType::Q4G64_F16S,
                              qus::QuantLayout::RowSplit, 8, 8, "full.k_proj");
    failures += expect_weight(full.v_proj, qus::SourceKind::AttnV, 3, qus::QType::Q5G64_F16S,
                              qus::QuantLayout::RowSplit, 8, 8, "full.v_proj");
    failures += expect_weight(full.o_proj, qus::SourceKind::AttnO, 3, qus::QType::Q5G64_F16S,
                              qus::QuantLayout::RowSplit, 8, 8, "full.o_proj");
    failures += expect_tensor(full.q_norm, qus::DType::BF16, {256}, "full.q_norm");
    failures += expect_tensor(full.k_norm, qus::DType::BF16, {256}, "full.k_norm");
    failures += expect_tensor(full.post_attn_norm, qus::DType::BF16, {5120}, "full.post_attn_norm");
    failures += expect_mlp(full.mlp, 3, "full.mlp");

    const qus::model::GdnLayerW& gdn = card.gdn_layer(0);
    failures += expect_tensor(gdn.input_norm, qus::DType::BF16, {5120}, "gdn.input_norm");
    failures += expect_weight(gdn.in_q, qus::SourceKind::GdnInProjQ, 0, qus::QType::Q4G64_F16S,
                              qus::QuantLayout::RowSplit, 8, 8, "gdn.in_q");
    failures += expect_weight(gdn.in_k, qus::SourceKind::GdnInProjK, 0, qus::QType::Q4G64_F16S,
                              qus::QuantLayout::RowSplit, 8, 8, "gdn.in_k");
    failures += expect_weight(gdn.in_v, qus::SourceKind::GdnInProjV, 0, qus::QType::Q5G64_F16S,
                              qus::QuantLayout::RowSplit, 8, 8, "gdn.in_v");
    failures += expect_weight(gdn.in_z, qus::SourceKind::GdnInProjZ, 0, qus::QType::Q5G64_F16S,
                              qus::QuantLayout::RowSplit, 8, 8, "gdn.in_z");
    failures += expect_weight(gdn.in_a, qus::SourceKind::GdnInProjA, 0, qus::QType::BF16_CTRL,
                              qus::QuantLayout::Contiguous, 48, 8, "gdn.in_a");
    failures += expect_weight(gdn.in_b, qus::SourceKind::GdnInProjB, 0, qus::QType::BF16_CTRL,
                              qus::QuantLayout::Contiguous, 48, 8, "gdn.in_b");
    failures += expect_tensor(gdn.conv1d, qus::DType::BF16, {10240, 4}, "gdn.conv1d");
    failures +=
        gdn.conv1d->data == stored_conv1d->data ? 0 : fail("gdn.conv1d must alias WeightStore");
    failures += expect_tensor(gdn.a_log, qus::DType::FP32, {48}, "gdn.a_log");
    failures += expect_tensor(gdn.dt_bias, qus::DType::FP32, {48}, "gdn.dt_bias");
    failures += expect_tensor(gdn.gdn_norm, qus::DType::BF16, {128}, "gdn.gdn_norm");
    failures += expect_weight(gdn.out_proj, qus::SourceKind::GdnOutProj, 0, qus::QType::Q5G64_F16S,
                              qus::QuantLayout::RowSplit, 8, 8, "gdn.out_proj");
    failures += expect_tensor(gdn.post_attn_norm, qus::DType::BF16, {5120}, "gdn.post_attn_norm");
    failures += expect_mlp(gdn.mlp, 0, "gdn.mlp");

    const qus::model::MtpW& mtp = card.mtp_weights();
    failures += expect_weight(mtp.fc, qus::SourceKind::MtpFc, qus::kQ5090NoLayer,
                              qus::QType::W8G32_F16S, qus::QuantLayout::RowSplit, 8, 16,
                              "mtp.fc", qus::ModuleKind::MtpDraft);
    failures += expect_tensor(mtp.pre_fc_norm_embedding, qus::DType::BF16, {8},
                              "mtp.pre_fc_norm_embedding");
    failures +=
        expect_tensor(mtp.pre_fc_norm_hidden, qus::DType::BF16, {8}, "mtp.pre_fc_norm_hidden");
    failures += expect_tensor(mtp.input_norm, qus::DType::BF16, {8}, "mtp.input_norm");
    failures += expect_weight(mtp.attn_in, qus::SourceKind::Other, 0, qus::QType::W8G32_F16S,
                              qus::QuantLayout::RowSplit, 24, 8, "mtp.attn_in",
                              qus::ModuleKind::MtpDraft);
    failures += expect_tensor(mtp.q_norm, qus::DType::BF16, {4}, "mtp.q_norm");
    failures += expect_tensor(mtp.k_norm, qus::DType::BF16, {4}, "mtp.k_norm");
    failures += expect_weight(mtp.o_proj, qus::SourceKind::AttnO, 0, qus::QType::W8G32_F16S,
                              qus::QuantLayout::RowSplit, 8, 8, "mtp.o_proj",
                              qus::ModuleKind::MtpDraft);
    failures +=
        expect_tensor(mtp.post_attn_norm, qus::DType::BF16, {8}, "mtp.post_attn_norm");
    failures += expect_weight(mtp.gate_up, qus::SourceKind::Other, 0, qus::QType::W8G32_F16S,
                              qus::QuantLayout::RowSplit, 20, 8, "mtp.gate_up",
                              qus::ModuleKind::MtpDraft);
    failures += expect_weight(mtp.down, qus::SourceKind::MlpDown, 0, qus::QType::W8G32_F16S,
                              qus::QuantLayout::RowSplit, 8, 10, "mtp.down",
                              qus::ModuleKind::MtpDraft);
    failures += expect_tensor(mtp.norm, qus::DType::BF16, {8}, "mtp.norm");

    return failures == 0 ? 0 : 1;
}
