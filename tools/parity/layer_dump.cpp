#include "qus/core/arena.h"
#include "qus/core/device.h"
#include "qus/core/kv_cache.h"
#include "qus/core/state_store.h"
#include "qus/core/weight_store.h"
#include "qus/kernels/embed_gather.h"
#include "qus/model/config.h"
#include "qus/model/model.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
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

qus::Q5090Expectations expectations() {
    qus::Q5090Expectations expected;
    expected.layer_count             = qus::model::kCfg.n_layers;
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

void copy_i32_to_device(std::span<const int> src, qus::Tensor& dst) {
    CUDA_CHECK(cudaMemcpy(dst.data, src.data(), src.size_bytes(), cudaMemcpyHostToDevice));
}

void write_f32(const std::filesystem::path& path, const qus::Tensor& x) {
    const std::size_t n = static_cast<std::size_t>(x.ne[0]) * static_cast<std::size_t>(x.ne[1]);
    std::vector<std::uint16_t> bits(n);
    CUDA_CHECK(cudaMemcpy(bits.data(), x.data, bits.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));
    std::vector<float> out(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) { out[i] = bf16_to_f32(bits[i]); }
    std::ofstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("failed to open output: " + path.string()); }
    file.write(reinterpret_cast<const char*>(out.data()),
               static_cast<std::streamsize>(out.size() * sizeof(float)));
}

void set_positions(qus::model::StepState& io, qus::DeviceArena& arena, int begin, int count) {
    std::vector<int> host(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) { host[static_cast<std::size_t>(i)] = begin + i; }
    io.pos = arena.alloc(qus::DType::I32, {count});
    CUDA_CHECK(
        cudaMemcpy(io.pos.data, host.data(), host.size() * sizeof(int), cudaMemcpyHostToDevice));
}

void dump_layers(const std::filesystem::path& out_dir, qus::model::Qwen3_6_27B& card,
                 qus::Tensor& x, qus::model::Phase phase) {
    for (int layer = 0; layer < qus::model::kCfg.n_layers; ++layer) {
        if (qus::model::ModelConfig::is_full(layer)) {
            const int fidx                     = qus::model::ModelConfig::full_idx(layer);
            const qus::model::FullLayerW& full = card.full_layer(static_cast<std::size_t>(fidx));
            card.test_attn_mix(full, x, fidx, phase);
            card.test_mlp_tail(full.post_attn_norm, full.mlp, x, phase);
        } else {
            const int gidx                   = qus::model::ModelConfig::gdn_idx(layer);
            const qus::model::GdnLayerW& gdn = card.gdn_layer(static_cast<std::size_t>(gidx));
            card.test_gdn_mix(gdn, x, gidx, phase);
            card.test_mlp_tail(gdn.post_attn_norm, gdn.mlp, x, phase);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        char name[32];
        std::snprintf(name, sizeof(name), "layer_%02d.f32", layer);
        write_f32(out_dir / name, x);
    }
}

int parse_int(const char* text, const char* label) {
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<int>(value);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: " << argv[0]
                  << " <weights.qus> <out-dir> <target-generated-index> <prompt ids...>\n";
        return 2;
    }

    try {
        const std::filesystem::path weights_path = argv[1];
        const std::filesystem::path out_dir      = argv[2];
        const int target                         = parse_int(argv[3], "target-generated-index");
        std::vector<int> prompt;
        for (int i = 4; i < argc; ++i) { prompt.push_back(parse_int(argv[i], "token-id")); }
        if (prompt.empty()) { throw std::invalid_argument("prompt must not be empty"); }
        std::filesystem::create_directories(out_dir);

        qus::DeviceContext ctx(0);
        const std::size_t weight_bytes =
            static_cast<std::size_t>(std::filesystem::file_size(weights_path)) +
            256ULL * 1024ULL * 1024ULL;
        qus::DeviceArena weight_arena(weight_bytes);
        qus::DeviceArena cache_arena(1024ULL * 1024ULL * 1024ULL);
        qus::WorkspaceArena work(2ULL * 1024ULL * 1024ULL * 1024ULL);
        qus::DeviceArena pos_arena(64ULL * 1024ULL);
        qus::WeightStore store(expectations());
        store.load(weights_path.string().c_str(), weight_arena, ctx);

        const auto max_ctx =
            static_cast<std::uint32_t>(prompt.size() + static_cast<std::size_t>(target) + 1);
        qus::KVCache kv(cache_arena, qus::model::kCfg.n_full(), max_ctx, qus::model::kCfg.n_kv,
                        qus::model::kCfg.head_dim);
        qus::GdnState state(cache_arena, qus::model::kCfg.n_gdn(), qus::model::kCfg.conv_dim,
                            qus::model::kCfg.gdn_conv_k, qus::model::kCfg.gdn_v_heads,
                            qus::model::kCfg.gdn_v_dim, qus::model::kCfg.gdn_k_dim);
        qus::model::StepState io{cache_arena.alloc(qus::DType::I32, {1}),
                                 cache_arena.alloc(qus::DType::I32, {1}),
                                 cache_arena.alloc(qus::DType::BF16, {qus::model::kCfg.vocab, 1})};
        qus::model::Qwen3_6_27B card(ctx, store, work, kv, state, io);

        kv.reset();
        state.reset(ctx.stream);
        if (target == 0) {
            const int T     = static_cast<int>(prompt.size());
            qus::Tensor ids = work.alloc(qus::DType::I32, {T});
            copy_i32_to_device(prompt, ids);
            set_positions(io, pos_arena, 0, T);
            qus::Tensor x = work.alloc(qus::DType::BF16, {qus::model::kCfg.hidden, T});
            qus::kernels::embed_gather(ids, *card.embed(), x, ctx.stream);
            dump_layers(out_dir, card, x, qus::model::Phase::Prefill);
        } else {
            card.prefill(prompt);
            for (int step = 1; step < target; ++step) { card.decode_step(); }
            work.reset();
            pos_arena.reset();
            qus::Tensor x = work.alloc(qus::DType::BF16, {qus::model::kCfg.hidden, 1});
            qus::kernels::embed_gather(io.token, *card.embed(), x, ctx.stream);
            dump_layers(out_dir, card, x, qus::model::Phase::Decode);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
