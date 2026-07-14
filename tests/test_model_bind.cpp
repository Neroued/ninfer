#include "ninfer/core/arena.h"
#include "ninfer/core/device.h"
#include "ninfer/core/kv_cache.h"
#include "ninfer/core/state_store.h"
#include "ninfer/core/weight_store.h"
#include "ninfer/core/weight_store_parser.h"
#include "ninfer/model/config.h"
#include "ninfer/model/model.h"

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
    const auto path = std::filesystem::temp_directory_path() / "ninfer_q5090_model_bind_fixture.qus";
    const std::filesystem::path script =
        std::filesystem::path(NINFER_SOURCE_DIR) / "tests/fixtures/make_q5090_fixture.py";
    const std::string command =
        "python3 \"" + script.string() + "\" --profile model-bind --out \"" + path.string() + "\"";
    if (std::system(command.c_str()) != 0) { throw std::runtime_error("fixture generator failed"); }
    return path;
}

int expect_weight(const ninfer::Weight* w, ninfer::SourceKind kind, std::uint32_t layer, ninfer::QType qtype,
                  ninfer::QuantLayout layout, std::int32_t n, std::int32_t k, std::string_view label,
                  ninfer::ModuleKind module = ninfer::ModuleKind::TextCore) {
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
    if (layout == ninfer::QuantLayout::RowSplit) {
        failures += w->scales != nullptr ? 0 : fail(std::string(label) + " scales");
        if (qtype == ninfer::QType::Q5G64_F16S || qtype == ninfer::QType::Q6G64_F16S) {
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

int expect_tensor(const ninfer::Tensor* t, ninfer::DType dtype, std::initializer_list<std::int32_t> shape,
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

int expect_mlp(const ninfer::model::MlpW& mlp, std::uint32_t layer, std::string_view label) {
    int failures = 0;
    failures += expect_weight(mlp.gate, ninfer::SourceKind::MlpGate, layer, ninfer::QType::Q4G64_F16S,
                              ninfer::QuantLayout::RowSplit, 8, 8, std::string(label) + ".gate");
    failures += expect_weight(mlp.up, ninfer::SourceKind::MlpUp, layer, ninfer::QType::Q4G64_F16S,
                              ninfer::QuantLayout::RowSplit, 8, 8, std::string(label) + ".up");
    failures += expect_weight(mlp.down, ninfer::SourceKind::MlpDown, layer, ninfer::QType::Q5G64_F16S,
                              ninfer::QuantLayout::RowSplit, 8, 8, std::string(label) + ".down");
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

    ninfer::DeviceContext ctx(0);
    ninfer::DeviceArena cache_arena(256ULL * 1024ULL * 1024ULL);
    ninfer::WorkspaceArena workspace(8ULL * 1024ULL * 1024ULL);
    ninfer::DeviceArena io_arena(1024ULL * 1024ULL);

    ninfer::WeightStore store;
    ninfer::LoadOptions load_options;
    load_options.load_mtp = true;
    store.load(fixture_path.c_str(), ctx, load_options);
    const ninfer::Tensor* stored_conv1d = store.tensor(
        ninfer::ModuleKind::TextCore, static_cast<std::uint32_t>(ninfer::SourceKind::GdnConv1d), 0);
    failures += expect_tensor(stored_conv1d, ninfer::DType::BF16, {10240, 4, 1}, "store.gdn.conv1d");
    ninfer::KVCache kv(cache_arena, ninfer::model::kCfg.n_full(), 1, ninfer::model::kCfg.n_kv,
                    ninfer::model::kCfg.head_dim);
    ninfer::GdnState state(cache_arena, ninfer::model::kCfg.n_gdn(), ninfer::model::kCfg.conv_dim,
                        ninfer::model::kCfg.gdn_conv_state_width, ninfer::model::kCfg.gdn_v_heads,
                        ninfer::model::kCfg.gdn_v_dim, ninfer::model::kCfg.gdn_k_dim);
    ninfer::KVCache mtp_kv(cache_arena, ninfer::model::kCfg.mtp_layers, 1, ninfer::model::kCfg.n_kv,
                        ninfer::model::kCfg.head_dim);
    ninfer::model::StepState io{};
    io.token            = io_arena.alloc(ninfer::DType::I32, {1});
    io.pos              = io_arena.alloc(ninfer::DType::I32, {1});
    io.logits           = io_arena.alloc(ninfer::DType::BF16, {ninfer::model::kCfg.vocab});
    io.gdn_initial_slot = io_arena.alloc(ninfer::DType::I32, {1});
    CUDA_CHECK(
        cudaMemsetAsync(io.gdn_initial_slot.data, 0, io.gdn_initial_slot.bytes(), ctx.stream));
    ninfer::model::Qwen3_6_27B card(ctx, store, workspace, kv, state, io, 512, &mtp_kv);

    failures += expect_weight(card.embed(), ninfer::SourceKind::Embed, ninfer::kQ5090NoLayer,
                              ninfer::QType::Q6G64_F16S, ninfer::QuantLayout::RowSplit, 16, 8, "embed");
    failures += expect_tensor(card.final_norm(), ninfer::DType::BF16, {5120}, "final_norm");
    failures += expect_weight(card.lm_head(), ninfer::SourceKind::LmHead, ninfer::kQ5090NoLayer,
                              ninfer::QType::Q6G64_F16S, ninfer::QuantLayout::RowSplit, 16, 8, "lm_head");

    const ninfer::model::FullLayerW& full = card.full_layer(0);
    failures += expect_tensor(full.input_norm, ninfer::DType::BF16, {5120}, "full.input_norm");
    failures += expect_weight(full.q_proj, ninfer::SourceKind::AttnQ, 3, ninfer::QType::Q4G64_F16S,
                              ninfer::QuantLayout::RowSplit, 8, 8, "full.q_proj");
    failures += expect_weight(full.gate_proj, ninfer::SourceKind::AttnGate, 3, ninfer::QType::Q5G64_F16S,
                              ninfer::QuantLayout::RowSplit, 8, 8, "full.gate_proj");
    failures += expect_weight(full.k_proj, ninfer::SourceKind::AttnK, 3, ninfer::QType::Q4G64_F16S,
                              ninfer::QuantLayout::RowSplit, 8, 8, "full.k_proj");
    failures += expect_weight(full.v_proj, ninfer::SourceKind::AttnV, 3, ninfer::QType::Q5G64_F16S,
                              ninfer::QuantLayout::RowSplit, 8, 8, "full.v_proj");
    failures += expect_weight(full.o_proj, ninfer::SourceKind::AttnO, 3, ninfer::QType::Q5G64_F16S,
                              ninfer::QuantLayout::RowSplit, 8, 8, "full.o_proj");
    failures += expect_tensor(full.q_norm, ninfer::DType::BF16, {256}, "full.q_norm");
    failures += expect_tensor(full.k_norm, ninfer::DType::BF16, {256}, "full.k_norm");
    failures += expect_tensor(full.post_attn_norm, ninfer::DType::BF16, {5120}, "full.post_attn_norm");
    failures += expect_mlp(full.mlp, 3, "full.mlp");

    const ninfer::model::GdnLayerW& gdn = card.gdn_layer(0);
    failures += expect_tensor(gdn.input_norm, ninfer::DType::BF16, {5120}, "gdn.input_norm");
    failures += expect_weight(gdn.in_q, ninfer::SourceKind::GdnInProjQ, 0, ninfer::QType::Q4G64_F16S,
                              ninfer::QuantLayout::RowSplit, 8, 8, "gdn.in_q");
    failures += expect_weight(gdn.in_k, ninfer::SourceKind::GdnInProjK, 0, ninfer::QType::Q4G64_F16S,
                              ninfer::QuantLayout::RowSplit, 8, 8, "gdn.in_k");
    failures += expect_weight(gdn.in_v, ninfer::SourceKind::GdnInProjV, 0, ninfer::QType::Q5G64_F16S,
                              ninfer::QuantLayout::RowSplit, 8, 8, "gdn.in_v");
    failures += expect_weight(gdn.in_z, ninfer::SourceKind::GdnInProjZ, 0, ninfer::QType::Q5G64_F16S,
                              ninfer::QuantLayout::RowSplit, 8, 8, "gdn.in_z");
    failures += expect_weight(gdn.in_a, ninfer::SourceKind::GdnInProjA, 0, ninfer::QType::BF16_CTRL,
                              ninfer::QuantLayout::Contiguous, 48, 8, "gdn.in_a");
    failures += expect_weight(gdn.in_b, ninfer::SourceKind::GdnInProjB, 0, ninfer::QType::BF16_CTRL,
                              ninfer::QuantLayout::Contiguous, 48, 8, "gdn.in_b");
    failures += expect_tensor(gdn.conv1d, ninfer::DType::BF16, {10240, 4}, "gdn.conv1d");
    failures +=
        gdn.conv1d->data == stored_conv1d->data ? 0 : fail("gdn.conv1d must alias WeightStore");
    failures += expect_tensor(gdn.a_log, ninfer::DType::FP32, {48}, "gdn.a_log");
    failures += expect_tensor(gdn.dt_bias, ninfer::DType::FP32, {48}, "gdn.dt_bias");
    failures += expect_tensor(gdn.gdn_norm, ninfer::DType::BF16, {128}, "gdn.gdn_norm");
    failures += expect_weight(gdn.out_proj, ninfer::SourceKind::GdnOutProj, 0, ninfer::QType::Q5G64_F16S,
                              ninfer::QuantLayout::RowSplit, 8, 8, "gdn.out_proj");
    failures += expect_tensor(gdn.post_attn_norm, ninfer::DType::BF16, {5120}, "gdn.post_attn_norm");
    failures += expect_mlp(gdn.mlp, 0, "gdn.mlp");

    const ninfer::model::MtpW& mtp = card.mtp_weights();
    failures +=
        expect_weight(mtp.fc, ninfer::SourceKind::MtpFc, ninfer::kQ5090NoLayer, ninfer::QType::W8G32_F16S,
                      ninfer::QuantLayout::RowSplit, 8, 16, "mtp.fc", ninfer::ModuleKind::MtpDraft);
    failures += expect_tensor(mtp.pre_fc_norm_embedding, ninfer::DType::BF16, {8},
                              "mtp.pre_fc_norm_embedding");
    failures +=
        expect_tensor(mtp.pre_fc_norm_hidden, ninfer::DType::BF16, {8}, "mtp.pre_fc_norm_hidden");
    failures += expect_tensor(mtp.input_norm, ninfer::DType::BF16, {8}, "mtp.input_norm");
    failures +=
        expect_weight(mtp.attn_in, ninfer::SourceKind::Other, 0, ninfer::QType::W8G32_F16S,
                      ninfer::QuantLayout::RowSplit, 24, 8, "mtp.attn_in", ninfer::ModuleKind::MtpDraft);
    failures +=
        expect_weight(mtp.q_proj, ninfer::SourceKind::AttnQ, 0, ninfer::QType::W8G32_F16S,
                      ninfer::QuantLayout::RowSplit, 8, 8, "mtp.q_proj", ninfer::ModuleKind::MtpDraft);
    failures +=
        expect_weight(mtp.gate_proj, ninfer::SourceKind::AttnGate, 0, ninfer::QType::W8G32_F16S,
                      ninfer::QuantLayout::RowSplit, 8, 8, "mtp.gate_proj", ninfer::ModuleKind::MtpDraft);
    failures +=
        expect_weight(mtp.k_proj, ninfer::SourceKind::AttnK, 0, ninfer::QType::W8G32_F16S,
                      ninfer::QuantLayout::RowSplit, 4, 8, "mtp.k_proj", ninfer::ModuleKind::MtpDraft);
    failures +=
        expect_weight(mtp.v_proj, ninfer::SourceKind::AttnV, 0, ninfer::QType::W8G32_F16S,
                      ninfer::QuantLayout::RowSplit, 4, 8, "mtp.v_proj", ninfer::ModuleKind::MtpDraft);
    failures += expect_tensor(mtp.q_norm, ninfer::DType::BF16, {4}, "mtp.q_norm");
    failures += expect_tensor(mtp.k_norm, ninfer::DType::BF16, {4}, "mtp.k_norm");
    failures +=
        expect_weight(mtp.o_proj, ninfer::SourceKind::AttnO, 0, ninfer::QType::W8G32_F16S,
                      ninfer::QuantLayout::RowSplit, 8, 8, "mtp.o_proj", ninfer::ModuleKind::MtpDraft);
    failures += expect_tensor(mtp.post_attn_norm, ninfer::DType::BF16, {8}, "mtp.post_attn_norm");
    failures +=
        expect_weight(mtp.gate_up, ninfer::SourceKind::Other, 0, ninfer::QType::W8G32_F16S,
                      ninfer::QuantLayout::RowSplit, 20, 8, "mtp.gate_up", ninfer::ModuleKind::MtpDraft);
    failures +=
        expect_weight(mtp.down, ninfer::SourceKind::MlpDown, 0, ninfer::QType::W8G32_F16S,
                      ninfer::QuantLayout::RowSplit, 8, 10, "mtp.down", ninfer::ModuleKind::MtpDraft);
    failures += expect_tensor(mtp.norm, ninfer::DType::BF16, {8}, "mtp.norm");

    return failures == 0 ? 0 : 1;
}
