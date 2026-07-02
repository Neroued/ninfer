#include "qus/core/arena.h"
#include "qus/core/device.h"
#include "qus/core/kv_cache.h"
#include "qus/core/state_store.h"
#include "qus/core/weight_store.h"
#include "qus/core/weight_store_parser.h"
#include "qus/model/config.h"
#include "qus/model/model.h"
#include "kernels/op_tester.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int fail(const std::string& message) {
    std::cerr << message << '\n';
    return 1;
}

std::filesystem::path make_fixture() {
    const auto path = std::filesystem::temp_directory_path() / "qus_q5090_model_blocks_fixture.qus";
    const std::filesystem::path script =
        std::filesystem::path(QUS_SOURCE_DIR) / "tests/fixtures/make_q5090_fixture.py";
    const std::string command = "python3 \"" + script.string() +
                                "\" --profile model-blocks --out \"" + path.string() + "\"";
    if (std::system(command.c_str()) != 0) { throw std::runtime_error("fixture generator failed"); }
    return path;
}

qus::Q5090Expectations expectations() {
    qus::Q5090Expectations expected;
    expected.layer_count             = 64;
    expected.hidden_size             = qus::model::kCfg.hidden;
    expected.intermediate_size       = qus::model::kCfg.intermediate;
    expected.vocab_size              = qus::model::kCfg.vocab;
    expected.num_attention_heads     = qus::model::kCfg.n_q;
    expected.num_key_value_heads     = qus::model::kCfg.n_kv;
    expected.head_dim                = qus::model::kCfg.head_dim;
    expected.gdn_key_heads           = qus::model::kCfg.gdn_k_heads;
    expected.gdn_value_heads         = qus::model::kCfg.gdn_v_heads;
    expected.gdn_key_head_dim        = qus::model::kCfg.gdn_k_dim;
    expected.gdn_value_head_dim      = qus::model::kCfg.gdn_v_dim;
    expected.gdn_conv_width          = qus::model::kCfg.gdn_conv_k;
    expected.full_attention_interval = qus::model::kCfg.full_interval;
    expected.max_position_embeddings = 262144;
    return expected;
}

void fill_hidden(qus::Tensor& x, int T) {
    std::vector<float> host(static_cast<std::size_t>(qus::model::kCfg.hidden) * T);
    qus::test::fill_uniform(host, 1234u + static_cast<std::uint32_t>(T), -0.25f, 0.25f);
    qus::test::round_to_bf16(host);
    std::vector<std::uint16_t> bf16(host.size());
    for (std::size_t i = 0; i < host.size(); ++i) { bf16[i] = qus::test::f32_to_bf16(host[i]); }
    CUDA_CHECK(cudaMemcpy(x.data, bf16.data(), bf16.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));
}

int expect_finite_hidden(const qus::Tensor& x, const char* label) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(qus::model::kCfg.hidden) * x.ne[1]);
    CUDA_CHECK(cudaMemcpy(bits.data(), x.data, bits.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < bits.size(); ++i) {
        const float value = qus::test::bf16_to_f32(bits[i]);
        if (!std::isfinite(value)) {
            return fail(std::string(label) + " produced non-finite value at " + std::to_string(i));
        }
    }
    return 0;
}

void set_positions(qus::model::StepState& io, qus::test::DBuf& pos, int T) {
    std::vector<int> host(static_cast<std::size_t>(T));
    qus::test::fill_iota_i32(host);
    pos    = qus::test::to_device_i32(host);
    io.pos = qus::Tensor(pos.p, qus::DType::I32, {T});
}

qus::Weight invalid_dense_weight(std::int32_t n, std::int32_t k) {
    qus::Weight w{};
    w.qtype             = qus::QType::BF16_CTRL;
    w.layout            = qus::QuantLayout::Contiguous;
    w.q5090_scale_dtype = qus::ScaleDType::None;
    w.n                 = n;
    w.k                 = k;
    w.ndim              = 2;
    w.shape[0]          = n;
    w.shape[1]          = k;
    for (int d = 0; d < 4; ++d) { w.padded_shape[d] = w.shape[d]; }
    w.payload_bytes = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(k) * 2ULL;
    return w;
}

int fused_prefill_uses_gate_up(qus::model::Qwen3_6_27B& card, const qus::model::FullLayerW& full,
                               qus::WorkspaceArena& work, qus::DeviceArena& x_arena) {
    constexpr int T = 32;

    work.reset();
    x_arena.reset();
    qus::Tensor x = x_arena.alloc(qus::DType::BF16, {qus::model::kCfg.hidden, T});
    fill_hidden(x, T);

    qus::Weight bad_gate =
        invalid_dense_weight(qus::model::kCfg.intermediate, qus::model::kCfg.hidden);
    qus::Weight bad_up =
        invalid_dense_weight(qus::model::kCfg.intermediate, qus::model::kCfg.hidden);
    qus::model::MlpW fused_mlp = full.mlp;
    fused_mlp.gate             = &bad_gate;
    fused_mlp.up               = &bad_up;

    try {
        card.test_mlp_tail(full.post_attn_norm, fused_mlp, x, qus::model::Phase::Prefill);
        CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
        return fail(std::string("prefill T=32 fused_mlp_tail: unexpected exception: ") + e.what());
    }

    int failures = expect_finite_hidden(x, "prefill T=32 fused_mlp_tail");
    if (work.used() == 0) {
        failures += fail("prefill T=32 fused_mlp_tail: workspace was not used");
    }
    return failures;
}

int run_case(qus::model::Qwen3_6_27B& card, const qus::model::FullLayerW& full,
             const qus::model::GdnLayerW& gdn, qus::WorkspaceArena& work, qus::DeviceArena& x_arena,
             qus::KVCache& kv, qus::GdnState& state, qus::model::StepState& io,
             qus::model::Phase phase, int T, const char* label) {
    int failures = 0;
    qus::test::DBuf pos(4);
    set_positions(io, pos, T);

    auto run_one = [&](const char* suffix, auto&& fn) {
        work.reset();
        x_arena.reset();
        kv.reset();
        state.reset();
        qus::Tensor x = x_arena.alloc(qus::DType::BF16, {qus::model::kCfg.hidden, T});
        fill_hidden(x, T);
        try {
            fn(x);
            CUDA_CHECK(cudaDeviceSynchronize());
        } catch (const std::exception& e) {
            failures +=
                fail(std::string(label) + " " + suffix + ": unexpected exception: " + e.what());
            return;
        }
        if (x.ne[0] != qus::model::kCfg.hidden || x.ne[1] != T) {
            failures += fail(std::string(label) + " " + suffix + ": output shape changed");
        }
        failures += expect_finite_hidden(x, (std::string(label) + " " + suffix).c_str());
        if (work.used() == 0) {
            failures += fail(std::string(label) + " " + suffix + ": workspace was not used");
        }
        work.reset();
        if (work.used() != 0) {
            failures += fail(std::string(label) + " " + suffix + ": workspace reset failed");
        }
    };

    run_one("attn_mix", [&](qus::Tensor& x) { card.test_attn_mix(full, x, 0, phase); });
    run_one("gdn_mix", [&](qus::Tensor& x) { card.test_gdn_mix(gdn, x, 0, phase); });
    run_one("full_mlp_tail",
            [&](qus::Tensor& x) { card.test_mlp_tail(full.post_attn_norm, full.mlp, x, phase); });
    run_one("gdn_mlp_tail",
            [&](qus::Tensor& x) { card.test_mlp_tail(gdn.post_attn_norm, gdn.mlp, x, phase); });
    return failures;
}

int schedule_mapping_smoke() {
    int failures = 0;
    for (int layer = 0; layer < qus::model::kCfg.n_layers; ++layer) {
        const auto got  = qus::model::Qwen3_6_27B::test_schedule_entry(layer);
        const bool full = qus::model::ModelConfig::is_full(layer);
        const int index = full ? qus::model::ModelConfig::full_idx(layer)
                               : qus::model::ModelConfig::gdn_idx(layer);
        if (got.is_full != full || got.index != index) {
            failures += fail("schedule mapping mismatch at layer " + std::to_string(layer));
        }
    }
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

    qus::DeviceContext ctx(0);
    if (ctx.total_vram() < (3ULL * 1024ULL * 1024ULL * 1024ULL)) {
        std::cout << "SKIP: model block smoke needs at least 3 GiB VRAM\n";
        return 0;
    }

    int failures = schedule_mapping_smoke();

    const std::filesystem::path fixture_path = make_fixture();
    qus::DeviceArena fixture_weight_arena(768ULL * 1024ULL * 1024ULL);
    qus::DeviceArena cache_arena(256ULL * 1024ULL * 1024ULL);
    qus::WorkspaceArena workspace(128ULL * 1024ULL * 1024ULL);
    qus::DeviceArena x_arena(8ULL * 1024ULL * 1024ULL);
    qus::DeviceArena io_arena(qus::model::kCfg.vocab * 2ULL + 1024ULL);

    qus::WeightStore store(expectations());
    store.load(fixture_path.c_str(), fixture_weight_arena, ctx);

    qus::KVCache kv(cache_arena, qus::model::kCfg.n_full(), 4, qus::model::kCfg.n_kv,
                    qus::model::kCfg.head_dim);
    qus::GdnState state(cache_arena, qus::model::kCfg.n_gdn(), qus::model::kCfg.conv_dim,
                        qus::model::kCfg.gdn_conv_state_width, qus::model::kCfg.gdn_v_heads,
                        qus::model::kCfg.gdn_v_dim, qus::model::kCfg.gdn_k_dim);
    qus::model::StepState io{io_arena.alloc(qus::DType::I32, {1}),
                             io_arena.alloc(qus::DType::I32, {1}),
                             io_arena.alloc(qus::DType::BF16, {qus::model::kCfg.vocab})};

    qus::model::Qwen3_6_27B card(ctx, store, workspace, kv, state, io);
    const qus::model::FullLayerW& full = card.full_layer(0);
    const qus::model::GdnLayerW& gdn   = card.gdn_layer(0);

    failures += run_case(card, full, gdn, workspace, x_arena, kv, state, io,
                         qus::model::Phase::Decode, 1, "decode T=1");
    failures += run_case(card, full, gdn, workspace, x_arena, kv, state, io,
                         qus::model::Phase::Prefill, 4, "prefill T=4");
    failures += fused_prefill_uses_gate_up(card, full, workspace, x_arena);
    return failures == 0 ? 0 : 1;
}
