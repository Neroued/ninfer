#include "qus/runtime/engine.h"

#include "qus/model/config.h"

#include <cuda_runtime.h>

#include <algorithm>
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

} // namespace

Engine::Engine(EngineOptions options) : options_(options) {
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
    const auto max_ctx_size    = static_cast<std::size_t>(max_ctx);
    const std::size_t kv_elems = checked_mul(
        checked_mul(model::kCfg.n_full(), 2, "cache arena size"),
        checked_mul(checked_mul(model::kCfg.n_kv, model::kCfg.head_dim, "cache arena size"),
                    max_ctx_size, "cache arena size"),
        "cache arena size");
    const std::size_t kv_bytes = checked_mul(kv_elems, dtype_size(DType::BF16), "cache arena size");

    const std::size_t conv_elems =
        checked_mul(checked_mul(model::kCfg.n_gdn(), model::kCfg.conv_dim, "cache arena size"),
                    model::kCfg.gdn_conv_k, "cache arena size");
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
    state_.emplace(*cache_arena_, model::kCfg.n_gdn(), model::kCfg.conv_dim, model::kCfg.gdn_conv_k,
                   model::kCfg.gdn_v_heads, model::kCfg.gdn_v_dim, model::kCfg.gdn_k_dim);
    io_ = model::StepState{
        cache_arena_->alloc(DType::I32, {1}),
        cache_arena_->alloc(DType::I32, {1}),
        cache_arena_->alloc(DType::BF16, {model::kCfg.vocab, 1}),
    };
    card_.emplace(*ctx_, *weights_, *work_, *kv_, *state_, io_);
}

void Engine::require_loaded() const {
    if (!loaded()) { throw std::runtime_error("Engine is not loaded"); }
}

std::uint32_t Engine::position() const noexcept { return kv_ ? kv_->pos : 0; }

int Engine::read_token() {
    int token = 0;
    ctx_->synchronize();
    CUDA_CHECK(cudaMemcpy(&token, io_.token.data, sizeof(token), cudaMemcpyDeviceToHost));
    return token;
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
    require_loaded();
    if (kv_->pos >= kv_->max_context) {
        throw std::out_of_range("Engine::decode_step exceeds max_ctx");
    }
    card_->decode_step();
    return read_token();
}

std::vector<int> Engine::generate(std::span<const int> prompt, int max_new_tokens) {
    require_loaded();
    if (max_new_tokens < 0) {
        throw std::invalid_argument("Engine::generate max_new_tokens must be nonnegative");
    }
    std::vector<int> out;
    if (max_new_tokens == 0) { return out; }
    out.reserve(static_cast<std::size_t>(max_new_tokens));
    out.push_back(prefill(prompt));
    while (static_cast<int>(out.size()) < max_new_tokens) { out.push_back(decode_step()); }
    return out;
}

} // namespace qus
