#include "qus/runtime/engine.h"

#include "qus/model/config.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace qus {
namespace {

constexpr std::size_t kMiB = 1024ULL * 1024ULL;

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t align_slack(std::size_t tensors) { return checked_mul(tensors, 256, "arena size"); }

std::size_t align_up(std::size_t value, std::size_t alignment, const char* label) {
    const std::size_t mask = alignment - 1;
    if (value > std::numeric_limits<std::size_t>::max() - mask) {
        throw std::overflow_error(label);
    }
    return (value + mask) & ~mask;
}

ArenaMemoryStats arena_stats(const std::optional<DeviceArena>& arena) noexcept {
    ArenaMemoryStats stats;
    if (!arena) { return stats; }
    stats.present = true;
    stats.capacity_bytes = arena->capacity();
    stats.used_bytes = arena->used();
    stats.peak_used_bytes = arena->peak_used();
    return stats;
}

} // namespace

Engine::Engine(EngineOptions options) : options_(options) {
    for (const int id : options_.stop_token_ids) {
        if (id < 0) {
            throw std::invalid_argument("Engine stop_token_ids must be nonnegative");
        }
    }
    std::sort(options_.stop_token_ids.begin(), options_.stop_token_ids.end());
    options_.stop_token_ids.erase(
        std::unique(options_.stop_token_ids.begin(), options_.stop_token_ids.end()),
        options_.stop_token_ids.end());
    if (options_.max_ctx == 0) { throw std::invalid_argument("Engine max_ctx must be nonzero"); }
    if (options_.work_bytes == 0) {
        throw std::invalid_argument("Engine work_bytes must be nonzero");
    }
}

Q5090Expectations Engine::expectations() {
    Q5090Expectations expected;
    expected.layer_count             = model::kCfg.n_layers;
    expected.hidden_size             = model::kCfg.hidden;
    expected.intermediate_size       = model::kCfg.intermediate;
    expected.vocab_size              = model::kCfg.vocab;
    expected.num_attention_heads     = model::kCfg.n_q;
    expected.num_key_value_heads     = model::kCfg.n_kv;
    expected.head_dim                = model::kCfg.head_dim;
    expected.gdn_key_heads           = model::kCfg.gdn_k_heads;
    expected.gdn_value_heads         = model::kCfg.gdn_v_heads;
    expected.gdn_key_head_dim        = model::kCfg.gdn_k_dim;
    expected.gdn_value_head_dim      = model::kCfg.gdn_v_dim;
    expected.gdn_conv_width          = model::kCfg.gdn_conv_k;
    expected.full_attention_interval = model::kCfg.full_interval;
    expected.max_position_embeddings = 262144;
    return expected;
}

std::size_t Engine::default_weight_bytes(const std::string& path) {
    const auto file_size = std::filesystem::file_size(path);
    return checked_add(static_cast<std::size_t>(file_size), 256ULL * kMiB, "weight arena size");
}

std::size_t Engine::default_cache_bytes(std::uint32_t max_ctx) {
    const auto padded_ctx_size = align_up(static_cast<std::size_t>(max_ctx), 128, "cache arena size");
    const std::size_t kv_elems = checked_mul(
        checked_mul(model::kCfg.n_full(), 2, "cache arena size"),
        checked_mul(checked_mul(model::kCfg.n_kv, model::kCfg.head_dim, "cache arena size"),
                    padded_ctx_size, "cache arena size"),
        "cache arena size");
    const std::size_t kv_bytes = checked_mul(kv_elems, dtype_size(DType::BF16), "cache arena size");

    const std::size_t conv_elems =
        checked_mul(checked_mul(model::kCfg.n_gdn(), model::kCfg.conv_dim, "cache arena size"),
                    model::kCfg.gdn_conv_state_width, "cache arena size");
    const std::size_t conv_bytes =
        checked_mul(conv_elems, dtype_size(DType::BF16), "cache arena size");

    const std::size_t ssm_elems =
        checked_mul(checked_mul(model::kCfg.n_gdn(), model::kCfg.gdn_k_dim, "cache arena size"),
                    checked_mul(model::kCfg.gdn_v_dim, model::kCfg.gdn_v_heads, "cache arena size"),
                    "cache arena size");
    const std::size_t ssm_bytes =
        checked_mul(ssm_elems, dtype_size(DType::FP32), "cache arena size");

    const std::size_t io_bytes =
        checked_add(checked_mul(model::kCfg.vocab, dtype_size(DType::BF16), "cache arena size"),
                    2 * dtype_size(DType::I32), "cache arena size");

    std::size_t total = 0;
    total             = checked_add(total, kv_bytes, "cache arena size");
    total             = checked_add(total, conv_bytes, "cache arena size");
    total             = checked_add(total, ssm_bytes, "cache arena size");
    total             = checked_add(total, io_bytes, "cache arena size");
    total = checked_add(total, align_slack(model::kCfg.n_full() * 2 + model::kCfg.n_gdn() * 2 + 3),
                        "cache arena size");
    return checked_add(total, 64ULL * kMiB, "cache arena size");
}

void Engine::load(const std::string& path) {
    reset_decode_graphs();
    card_.reset();
    state_.reset();
    kv_.reset();
    weights_.reset();
    work_.reset();
    cache_arena_.reset();
    weight_arena_.reset();
    ctx_.reset();
    io_ = {};

    ctx_.emplace(options_.device);
    weight_arena_.emplace(options_.weight_bytes == 0 ? default_weight_bytes(path)
                                                     : options_.weight_bytes);
    work_.emplace(options_.work_bytes);
    weights_.emplace(expectations());

    LoadOptions load_options;
    load_options.progress = options_.progress;
    weights_->load(path.c_str(), *weight_arena_, *ctx_, load_options);

    cache_arena_.emplace(options_.cache_bytes == 0 ? default_cache_bytes(options_.max_ctx)
                                                   : options_.cache_bytes);
    kv_.emplace(*cache_arena_, model::kCfg.n_full(), options_.max_ctx, model::kCfg.n_kv,
                model::kCfg.head_dim);
    state_.emplace(*cache_arena_, model::kCfg.n_gdn(), model::kCfg.conv_dim,
                   model::kCfg.gdn_conv_state_width, model::kCfg.gdn_v_heads, model::kCfg.gdn_v_dim,
                   model::kCfg.gdn_k_dim);
    io_ = model::StepState{
        cache_arena_->alloc(DType::I32, {1}),
        cache_arena_->alloc(DType::I32, {1}),
        cache_arena_->alloc(DType::BF16, {model::kCfg.vocab, 1}),
        cache_arena_->alloc(DType::I32, {kDecodeGraphChunk}),
    };
    card_.emplace(*ctx_, *weights_, *work_, *kv_, *state_, io_);
}

void Engine::require_loaded() const {
    if (!loaded()) { throw std::runtime_error("Engine is not loaded"); }
}

std::uint32_t Engine::position() const noexcept { return kv_ ? kv_->pos : 0; }

EngineMemoryStats Engine::memory_stats() const noexcept {
    EngineMemoryStats stats;
    stats.loaded = loaded();
    stats.device = options_.device;
    stats.max_context = options_.max_ctx;
    stats.position = position();
    stats.weights = arena_stats(weight_arena_);
    stats.cache = arena_stats(cache_arena_);
    stats.workspace = arena_stats(work_);
    if (weights_) {
        stats.q5090_loaded_payload_bytes = weights_->loaded_payload_bytes();
        stats.q5090_tensor_count = weights_->tensor_count();
        stats.q5090_quant_count = weights_->quant_count();
    }
    return stats;
}

void Engine::reset_memory_peaks() noexcept {
    if (weight_arena_) { weight_arena_->reset_peak(); }
    if (cache_arena_) { cache_arena_->reset_peak(); }
    if (work_) { work_->reset_peak(); }
}

int Engine::read_token() {
    int token = 0;
    ctx_->synchronize();
    CUDA_CHECK(cudaMemcpy(&token, io_.token.data, sizeof(token), cudaMemcpyDeviceToHost));
    return token;
}

std::vector<int> Engine::read_decode_tokens(int count) {
    if (count <= 0 || count > kDecodeGraphChunk) {
        throw std::invalid_argument("Engine::read_decode_tokens count out of range");
    }
    ctx_->synchronize();
    CUDA_CHECK(cudaMemcpy(host_decode_tokens_.data(), io_.decode_tokens.data,
                          static_cast<std::size_t>(count) * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    return std::vector<int>(host_decode_tokens_.begin(), host_decode_tokens_.begin() + count);
}

void Engine::record_decode_token(int slot) {
    if (slot < 0 || slot >= kDecodeGraphChunk) {
        throw std::invalid_argument("Engine::record_decode_token slot out of range");
    }
    card_->decode_step_record();
    auto* dst = static_cast<std::int32_t*>(io_.decode_tokens.data) + slot;
    CUDA_CHECK(cudaMemcpyAsync(dst, io_.token.data, sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, ctx_->stream));
}

void Engine::reset_decode_graphs() noexcept {
    for (DecodeGraph& graph : decode_graphs_) {
        graph.reset();
    }
    decode_warmed_ = false;
}

bool Engine::is_stop_token(int token) const noexcept {
    return std::binary_search(options_.stop_token_ids.begin(), options_.stop_token_ids.end(),
                              token);
}

int Engine::prefill(std::span<const int> ids) {
    require_loaded();
    if (ids.empty()) { throw std::invalid_argument("Engine::prefill requires tokens"); }
    if (ids.size() > options_.max_ctx) {
        throw std::invalid_argument("Engine::prefill exceeds max_ctx");
    }
    kv_->reset();
    state_->reset(ctx_->stream);
    card_->prefill(ids);
    return read_token();
}

int Engine::decode_step() {
    std::vector<int> tokens = decode_steps(1);
    return tokens.front();
}

std::vector<int> Engine::decode_steps(int max_steps) {
    require_loaded();
    if (max_steps <= 0) { throw std::invalid_argument("Engine::decode_steps requires steps"); }
    if (kv_->pos >= kv_->max_context) {
        throw std::out_of_range("Engine::decode_steps exceeds max_ctx");
    }
    int count = std::min(max_steps, kDecodeGraphChunk);
    count = std::min(count, static_cast<int>(kv_->max_context - kv_->pos));

    if (options_.use_cuda_graph && decode_warmed_) {
        DecodeGraph& graph = decode_graphs_[static_cast<std::size_t>(count - 1)];
        if (!graph.ready()) {
            graph.capture(ctx_->stream, [this, count] {
                for (int i = 0; i < count; ++i) {
                    record_decode_token(i);
                }
            });
        }
        graph.launch(ctx_->stream);
        for (int i = 0; i < count; ++i) {
            kv_->advance();
        }
    } else {
        count = 1;
        for (int i = 0; i < count; ++i) {
            record_decode_token(i);
            kv_->advance();
        }
        decode_warmed_ = true;
    }
    return read_decode_tokens(count);
}

std::vector<int> Engine::generate(std::span<const int> prompt, int max_new_tokens) {
    require_loaded();
    if (max_new_tokens < 0) {
        throw std::invalid_argument("Engine::generate max_new_tokens must be nonnegative");
    }
    std::vector<int> out;
    if (max_new_tokens == 0) { return out; }
    out.reserve(static_cast<std::size_t>(max_new_tokens));
    int token = prefill(prompt);
    out.push_back(token);
    if (is_stop_token(token)) { return out; }
    while (static_cast<int>(out.size()) < max_new_tokens) {
        const int remaining = max_new_tokens - static_cast<int>(out.size());
        std::vector<int> tokens = decode_steps(remaining);
        for (int next : tokens) {
            out.push_back(next);
            if (is_stop_token(next) || static_cast<int>(out.size()) >= max_new_tokens) { break; }
        }
        if (is_stop_token(out.back())) { break; }
    }
    return out;
}

} // namespace qus
