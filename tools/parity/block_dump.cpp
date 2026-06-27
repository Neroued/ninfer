#include "qus/core/arena.h"
#include "qus/core/device.h"
#include "qus/core/kv_cache.h"
#include "qus/core/state_store.h"
#include "qus/core/weight_store.h"
#include "qus/model/config.h"
#include "qus/model/model.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

float bf16_to_f32(std::uint16_t h) {
    const std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
    float f               = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

std::uint16_t f32_to_bf16(float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7fffffffu) > 0x7f800000u) { return static_cast<std::uint16_t>((u >> 16) | 0x0040u); }
    const std::uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(u >> 16);
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

void write_f32(const std::filesystem::path& path, const std::vector<float>& values) {
    std::ofstream out(path, std::ios::binary);
    if (!out) { throw std::runtime_error("failed to open output: " + path.string()); }
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
}

void fill_hidden(qus::Tensor& x, int T, std::uint32_t seed) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(qus::model::kCfg.hidden) * T);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    for (std::uint16_t& value : bits) { value = f32_to_bf16(dist(rng)); }
    CUDA_CHECK(cudaMemcpy(x.data, bits.data(), bits.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));
}

std::vector<float> tensor_to_f32(const qus::Tensor& x) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(x.ne[0]) * x.ne[1]);
    CUDA_CHECK(cudaMemcpy(bits.data(), x.data, bits.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));
    std::vector<float> out(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) { out[i] = bf16_to_f32(bits[i]); }
    return out;
}

void set_positions(qus::model::StepState& io, qus::DeviceArena& arena, int T) {
    std::vector<int> host(static_cast<std::size_t>(T));
    for (int i = 0; i < T; ++i) { host[static_cast<std::size_t>(i)] = i; }
    qus::Tensor pos = arena.alloc(qus::DType::I32, {T});
    CUDA_CHECK(
        cudaMemcpy(pos.data, host.data(), host.size() * sizeof(int), cudaMemcpyHostToDevice));
    io.pos = pos;
}

template <typename Fn>
void run_case(const std::filesystem::path& out_dir, const char* name, qus::model::Qwen3_6_27B& card,
              qus::WorkspaceArena& work, qus::DeviceArena& x_arena, qus::DeviceArena& pos_arena,
              qus::KVCache& kv, qus::GdnState& state, qus::model::StepState& io, int T,
              std::uint32_t seed, Fn&& fn) {
    work.reset();
    x_arena.reset();
    pos_arena.reset();
    kv.reset();
    state.reset();
    set_positions(io, pos_arena, T);

    qus::Tensor x = x_arena.alloc(qus::DType::BF16, {qus::model::kCfg.hidden, T});
    fill_hidden(x, T, seed);
    write_f32(out_dir / (std::string(name) + ".input.f32"), tensor_to_f32(x));
    fn(x);
    CUDA_CHECK(cudaDeviceSynchronize());
    write_f32(out_dir / (std::string(name) + ".out.f32"), tensor_to_f32(x));
    work.reset();
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " <fixture.qus> <out-dir>\n";
        return 2;
    }

    const std::filesystem::path fixture = argv[1];
    const std::filesystem::path out_dir = argv[2];
    std::filesystem::create_directories(out_dir);

    qus::DeviceContext ctx(0);
    qus::DeviceArena fixture_weight_arena(768ULL * 1024ULL * 1024ULL);
    qus::DeviceArena cache_arena(256ULL * 1024ULL * 1024ULL);
    qus::WorkspaceArena workspace(256ULL * 1024ULL * 1024ULL);
    qus::DeviceArena x_arena(64ULL * 1024ULL * 1024ULL);
    qus::DeviceArena io_arena(qus::model::kCfg.vocab * 2ULL + 1024ULL);
    qus::DeviceArena pos_arena(4096);

    qus::WeightStore store(expectations());
    store.load(fixture.string().c_str(), fixture_weight_arena, ctx);

    qus::KVCache kv(cache_arena, qus::model::kCfg.n_full(), 8, qus::model::kCfg.n_kv,
                    qus::model::kCfg.head_dim);
    qus::GdnState state(cache_arena, qus::model::kCfg.n_gdn(), qus::model::kCfg.conv_dim,
                        qus::model::kCfg.gdn_conv_k, qus::model::kCfg.gdn_v_heads,
                        qus::model::kCfg.gdn_v_dim, qus::model::kCfg.gdn_k_dim);
    qus::model::StepState io{io_arena.alloc(qus::DType::I32, {1}),
                             io_arena.alloc(qus::DType::I32, {1}),
                             io_arena.alloc(qus::DType::BF16, {qus::model::kCfg.vocab, 1})};
    qus::model::Qwen3_6_27B card(ctx, store, workspace, kv, state, io);
    const qus::model::FullLayerW& full = card.full_layer(0);
    const qus::model::GdnLayerW& gdn   = card.gdn_layer(0);

    run_case(out_dir, "prefill_t4_attn", card, workspace, x_arena, pos_arena, kv, state, io, 4,
             4104u,
             [&](qus::Tensor& x) { card.test_attn_mix(full, x, 0, qus::model::Phase::Prefill); });
    run_case(out_dir, "prefill_t4_gdn", card, workspace, x_arena, pos_arena, kv, state, io, 4,
             4105u,
             [&](qus::Tensor& x) { card.test_gdn_mix(gdn, x, 0, qus::model::Phase::Prefill); });
    run_case(out_dir, "prefill_t4_full_mlp", card, workspace, x_arena, pos_arena, kv, state, io, 4,
             4106u, [&](qus::Tensor& x) {
                 card.test_mlp_tail(full.post_attn_norm, full.mlp, x, qus::model::Phase::Prefill);
             });
    run_case(out_dir, "prefill_t4_gdn_mlp", card, workspace, x_arena, pos_arena, kv, state, io, 4,
             4107u, [&](qus::Tensor& x) {
                 card.test_mlp_tail(gdn.post_attn_norm, gdn.mlp, x, qus::model::Phase::Prefill);
             });
    return 0;
}
