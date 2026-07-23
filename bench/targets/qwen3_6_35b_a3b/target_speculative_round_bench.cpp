#include <ninfer/targets/qwen3_6_35b_a3b/package.h>
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include <ninfer/targets/qwen3_6/round_state.h>
#include <ninfer/targets/qwen3_6/startup_features.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/kv_cache.h"
#include "core/layout.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/speculative_round.h"
#include "targets/qwen3_6_35b_a3b/impl/load/bindings.h"
#include "targets/qwen3_6_35b_a3b/impl/variant.h"

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_35b_a3b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_35b_a3b_runtime
#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace target  = ninfer::targets::qwen3_6_35b_a3b;
namespace detail  = ninfer::targets::qwen3_6_35b_a3b::detail;
namespace family  = ninfer::targets::qwen3_6;
namespace runtime = ninfer::targets::qwen3_6::detail::qwen3_6_35b_a3b_runtime;

constexpr std::uint32_t kMaximumDraftTokens = 15;
constexpr std::size_t kArenaAlignment       = 256;

struct Options {
    std::filesystem::path artifact = "out/qwen3_6_35b_a3b.ninfer";
    int device                     = 0;
    int warmup                     = 3;
    int repetitions                = 10;
    std::uint32_t context_tokens   = 128;
    std::uint32_t prefill_chunk    = 128;
    std::vector<std::uint32_t> draft_windows;
    std::optional<std::uint32_t> accepted_drafts;
    ninfer::DType kv_dtype = ninfer::DType::I8;
    bool eager             = true;
    bool graph             = true;
};

std::uint32_t parse_u32(std::string_view text, const char* label) {
    std::size_t consumed      = 0;
    const unsigned long value = std::stoul(std::string(text), &consumed);
    if (consumed != text.size() || value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(std::string(label) + " is not a uint32");
    }
    return static_cast<std::uint32_t>(value);
}

std::vector<std::uint32_t> parse_draft_windows(std::string_view text) {
    if (text.empty()) { throw std::invalid_argument("--draft-tokens must not be empty"); }
    std::vector<std::uint32_t> out;
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t end = text.find(',', begin);
        const std::string_view item =
            text.substr(begin, end == std::string_view::npos ? text.size() - begin : end - begin);
        if (item.empty()) { throw std::invalid_argument("--draft-tokens contains an empty item"); }
        const std::uint32_t value = parse_u32(item, "draft token count");
        if (value == 0 || value > kMaximumDraftTokens) {
            throw std::invalid_argument("draft token counts must be in [1,15]");
        }
        out.push_back(value);
        if (end == std::string_view::npos) { break; }
        begin = end + 1;
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

void print_usage(const char* executable) {
    std::cout << "usage: " << executable
              << " [--artifact <model.ninfer>] [--device <id>] [--context <tokens>]"
                 " [--prefill-chunk <tokens>] [--draft-tokens <K[,K...]>]"
                 " [--accepted-drafts <A>] [--kv-dtype bf16|int8] [--mode eager|graph|both]"
                 " [--warmup <n>] [--reps <n>]\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (std::uint32_t k = 1; k <= kMaximumDraftTokens; ++k) { options.draft_windows.push_back(k); }
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto value = [&](const char* name) -> std::string_view {
            if (++index >= argc) {
                throw std::invalid_argument(std::string(name) + " needs value");
            }
            return argv[index];
        };
        if (argument == "--artifact") {
            options.artifact = value("--artifact");
        } else if (argument == "--device") {
            options.device = std::stoi(std::string(value("--device")));
        } else if (argument == "--context") {
            options.context_tokens = parse_u32(value("--context"), "context");
        } else if (argument == "--prefill-chunk") {
            options.prefill_chunk = parse_u32(value("--prefill-chunk"), "prefill chunk");
        } else if (argument == "--draft-tokens") {
            options.draft_windows = parse_draft_windows(value("--draft-tokens"));
        } else if (argument == "--accepted-drafts") {
            options.accepted_drafts = parse_u32(value("--accepted-drafts"), "accepted draft count");
        } else if (argument == "--kv-dtype") {
            const std::string_view dtype = value("--kv-dtype");
            if (dtype == "bf16") {
                options.kv_dtype = ninfer::DType::BF16;
            } else if (dtype == "int8") {
                options.kv_dtype = ninfer::DType::I8;
            } else {
                throw std::invalid_argument("--kv-dtype must be bf16 or int8");
            }
        } else if (argument == "--mode") {
            const std::string_view mode = value("--mode");
            options.eager               = mode == "eager" || mode == "both";
            options.graph               = mode == "graph" || mode == "both";
            if (!options.eager && !options.graph) {
                throw std::invalid_argument("--mode must be eager, graph, or both");
            }
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(std::string(value("--warmup")));
        } else if (argument == "--reps") {
            options.repetitions = std::stoi(std::string(value("--reps")));
        } else if (argument == "-h" || argument == "--help") {
            print_usage(argc > 0 ? argv[0]
                                 : "ninfer_qwen3_6_35b_a3b_target_speculative_round_bench");
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }

    if (options.device < 0) { throw std::invalid_argument("--device must be nonnegative"); }
    if (options.warmup < 0) { throw std::invalid_argument("--warmup must be nonnegative"); }
    if (options.repetitions <= 0) { throw std::invalid_argument("--reps must be positive"); }
    if (options.context_tokens == 0) { throw std::invalid_argument("--context must be positive"); }
    if (options.prefill_chunk == 0 || options.prefill_chunk % detail::kPrefillChunkAlignment != 0 ||
        options.prefill_chunk >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("--prefill-chunk must be a nonzero multiple of 128");
    }
    const std::uint32_t maximum_k =
        *std::max_element(options.draft_windows.begin(), options.draft_windows.end());
    if (options.context_tokens + static_cast<std::uint64_t>(maximum_k) + 1ULL >
        detail::kNativeContext) {
        throw std::invalid_argument("context plus verification block exceeds native capacity");
    }
    if (options.accepted_drafts &&
        *options.accepted_drafts >
            *std::min_element(options.draft_windows.begin(), options.draft_windows.end())) {
        throw std::invalid_argument(
            "--accepted-drafts must not exceed any selected draft-token count");
    }
    return options;
}

struct StateLayout {
    family::DecoderStateLayout decoder;
    family::RoundStateLayout round;
    ninfer::TensorRegion prefill_hidden;
    ninfer::TensorRegion tail_hidden;
    ninfer::TensorRegion boundary_hidden;
    std::size_t bytes = 0;
};

StateLayout plan_state(const Options& options, std::uint32_t maximum_k) {
    const std::uint32_t capacity = options.context_tokens + maximum_k + 1;
    ninfer::LayoutBuilder builder;
    StateLayout out;
    out.decoder = family::plan_decoder_state(
        builder,
        family::DecoderStateSpec{
            .full_attention_layers = detail::TextConfig::full_attention_layers(),
            .mtp_layers            = detail::TextConfig::mtp_layers,
            .capacity              = capacity,
            .kv_heads              = detail::TextConfig::kv_heads,
            .attention_head_dim    = detail::TextConfig::head_dim,
            .kv_dtype              = options.kv_dtype,
            .kv_quant_group = options.kv_dtype == ninfer::DType::I8 ? ninfer::kKvQuantGroup : 0,
            .enable_mtp     = false,
            .gdn =
                {
                    .layers         = detail::TextConfig::gdn_layers(),
                    .conv_dim       = detail::TextConfig::convolution_dim,
                    .conv_width     = detail::TextConfig::gdn_conv_state_width,
                    .value_heads    = detail::TextConfig::gdn_value_heads,
                    .value_head_dim = detail::TextConfig::gdn_value_head_dim,
                    .key_head_dim   = detail::TextConfig::gdn_key_head_dim,
                    .snapshot_slots = static_cast<std::int32_t>(maximum_k + 2),
                    .conv_dtype     = ninfer::DType::BF16,
                },
        });
    out.round = family::begin_round_state_layout(
        builder, family::RoundStateSpec{.hidden       = detail::TextConfig::hidden,
                                        .output_rows  = detail::TextConfig::output_rows,
                                        .draft_window = maximum_k,
                                        .enable_mtp   = false});
    out.prefill_hidden = builder.add_tensor(
        ninfer::DType::BF16,
        {detail::TextConfig::hidden, static_cast<std::int32_t>(options.prefill_chunk)},
        kArenaAlignment, "benchmark prefill hidden");
    family::complete_round_state_layout(builder, out.round);
    out.tail_hidden     = builder.add_tensor(ninfer::DType::BF16, {detail::TextConfig::hidden, 1},
                                             kArenaAlignment, "benchmark tail hidden");
    out.boundary_hidden = builder.add_tensor(ninfer::DType::BF16, {detail::TextConfig::hidden, 1},
                                             kArenaAlignment, "benchmark boundary hidden");
    out.bytes           = builder.finish(kArenaAlignment, "benchmark state");
    return out;
}

family::RoundState round_window(const family::RoundState& storage, std::uint32_t k) {
    const int columns             = static_cast<int>(k + 1);
    family::RoundState out        = storage;
    out.logits                    = storage.logits.slice(1, 0, columns);
    out.verify_hidden             = storage.verify_hidden.slice(1, 0, columns);
    out.speculative.target_argmax = storage.speculative.target_argmax.slice(0, 0, columns);
    out.speculative.draft_tokens =
        storage.speculative.draft_tokens.slice(0, 0, static_cast<int>(k));
    out.speculative.round_tokens     = storage.speculative.round_tokens.slice(0, 0, columns);
    out.speculative.target_input_ids = storage.speculative.target_input_ids.slice(0, 0, columns);
    out.speculative.target_positions = storage.speculative.target_positions.slice(0, 0, columns);
    out.speculative.stats = storage.speculative.stats.slice(0, 0, static_cast<int>(4 + k));
    return out;
}

std::vector<ninfer::TokenId> prompt_tokens(std::uint32_t count) {
    constexpr std::array<ninfer::TokenId, 8> seed{248045, 846, 198, 5834, 248046, 198, 3838, 374};
    std::vector<ninfer::TokenId> out(count);
    for (std::size_t i = 0; i < out.size(); ++i) { out[i] = seed[i % seed.size()]; }
    return out;
}

void copy_i32(ninfer::Tensor& tensor, std::int32_t value, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(tensor.data, &value, sizeof(value), cudaMemcpyHostToDevice, stream));
}

void copy_vector(ninfer::Tensor& tensor, const std::vector<std::int32_t>& values,
                 cudaStream_t stream) {
    if (tensor.bytes() != values.size() * sizeof(std::int32_t)) {
        throw std::logic_error("benchmark vector size does not match its tensor");
    }
    CUDA_CHECK(cudaMemcpyAsync(tensor.data, values.data(), tensor.bytes(), cudaMemcpyHostToDevice,
                               stream));
}

void reset_round_controls(family::RoundState& io, std::int32_t anchor, std::int32_t position,
                          std::int32_t initial_slot, cudaStream_t stream) {
    copy_i32(io.token, anchor, stream);
    copy_i32(io.pos, position, stream);
    copy_i32(io.gdn_initial_slot, initial_slot, stream);
}

std::vector<std::int32_t> prepare_drafts(runtime::schedule::State& state,
                                         runtime::schedule::TextContext& card, std::uint32_t k,
                                         std::uint32_t accepted_drafts, std::int32_t anchor,
                                         std::int32_t base_slot,
                                         ninfer::ops::GqaExecutionEnvelope envelope) {
    std::vector<std::int32_t> drafts(k);
    for (std::uint32_t i = 0; i < k; ++i) {
        drafts[i] = static_cast<std::int32_t>((17 + i * 7919) % detail::TextConfig::token_domain);
    }
    copy_vector(state.io.speculative.draft_tokens, drafts, state.device.stream);

    const auto target_at = [&](std::uint32_t index) {
        reset_round_controls(state.io, anchor, static_cast<std::int32_t>(state.text_kv_base),
                             base_slot, state.device.stream);
        ninfer::ops::speculative_prepare_verify_inputs(
            state.io.token, state.io.speculative.draft_tokens, state.io.pos,
            state.io.speculative.target_input_ids, state.io.speculative.target_positions,
            state.device.stream);
        runtime::schedule::target_verify(card, state, state.io.speculative.target_input_ids,
                                         state.io.speculative.target_positions, envelope);
        std::int32_t target_token = 0;
        const ninfer::Tensor source =
            state.io.speculative.target_argmax.slice(0, static_cast<int>(index), 1);
        CUDA_CHECK(cudaMemcpyAsync(&target_token, source.data, sizeof(target_token),
                                   cudaMemcpyDeviceToHost, state.device.stream));
        state.device.synchronize();
        return target_token;
    };

    for (std::uint32_t i = 0; i < accepted_drafts; ++i) {
        drafts[i] = target_at(i);
        copy_vector(state.io.speculative.draft_tokens, drafts, state.device.stream);
    }
    if (accepted_drafts < k) {
        const std::int32_t target_token = target_at(accepted_drafts);
        drafts[accepted_drafts]         = target_token == 0 ? 1 : 0;
        copy_vector(state.io.speculative.draft_tokens, drafts, state.device.stream);
    }

    reset_round_controls(state.io, anchor, static_cast<std::int32_t>(state.text_kv_base), base_slot,
                         state.device.stream);
    runtime::schedule::speculative_verify_and_accept(state, card, k, envelope);
    std::int32_t observed = -1;
    CUDA_CHECK(cudaMemcpyAsync(&observed, state.io.speculative.accepted_drafts.data,
                               sizeof(observed), cudaMemcpyDeviceToHost, state.device.stream));
    state.device.synchronize();
    if (observed != static_cast<std::int32_t>(accepted_drafts)) {
        throw std::runtime_error("failed to construct the requested acceptance frontier");
    }
    return drafts;
}

struct Timing {
    double mean_ms   = 0.0;
    double median_ms = 0.0;
    float minimum_ms = 0.0F;
    float maximum_ms = 0.0F;
};

template <class Launch, class Reset>
Timing measure(const Options& options, ninfer::DeviceContext& device, Launch&& launch,
               Reset&& reset) {
    for (int iteration = 0; iteration < options.warmup; ++iteration) {
        reset();
        launch();
        device.synchronize();
    }

    std::vector<float> samples;
    samples.reserve(static_cast<std::size_t>(options.repetitions));
    ninfer::CudaEventTimer timer(device);
    for (int iteration = 0; iteration < options.repetitions; ++iteration) {
        reset();
        timer.start();
        launch();
        samples.push_back(timer.stop_ms());
    }

    std::sort(samples.begin(), samples.end());
    const double sum         = std::accumulate(samples.begin(), samples.end(), 0.0);
    const std::size_t middle = samples.size() / 2;
    const double median      = samples.size() % 2 != 0
                                   ? samples[middle]
                                   : 0.5 * (static_cast<double>(samples[middle - 1]) + samples[middle]);
    return Timing{.mean_ms    = sum / static_cast<double>(samples.size()),
                  .median_ms  = median,
                  .minimum_ms = samples.front(),
                  .maximum_ms = samples.back()};
}

void print_row(const Options& options, std::uint32_t k, std::uint32_t accepted,
               std::string_view mode, const Timing& timing) {
    const double effective_tokens = static_cast<double>(accepted + 1);
    const double target_only_tps  = 1000.0 * effective_tokens / timing.mean_ms;
    std::cout << k << ',' << (k + 1) << ',' << accepted << ',' << mode << ','
              << (options.kv_dtype == ninfer::DType::I8 ? "int8-g64" : "bf16") << ','
              << options.context_tokens << ',' << timing.mean_ms << ',' << timing.median_ms << ','
              << timing.minimum_ms << ',' << timing.maximum_ms << ',' << target_only_tps << '\n';
}

int run(const Options& options) {
    if (!std::filesystem::exists(options.artifact)) {
        throw std::invalid_argument("artifact does not exist: " + options.artifact.string());
    }

    ninfer::DeviceContext device(options.device);
    ninfer::artifact::Reader reader(options.artifact);
    if (reader.model_id() != target::Package::model_id) {
        throw std::invalid_argument("artifact model_id does not match qwen3.6-35b-a3b");
    }
    ninfer::artifact::Binder binder(reader);
    detail::ArtifactLoadPlan load = detail::bind_artifact(
        binder, detail::LoadFeatures{.family = family::StartupFeatures{}, .dflash = false});
    auto materialized =
        ninfer::artifact::materialize(reader, load.materialization, device, nullptr);
    detail::LoadedModelData model(std::move(load.bindings), std::move(materialized));

    const std::uint32_t maximum_k =
        *std::max_element(options.draft_windows.begin(), options.draft_windows.end());
    const StateLayout layout = plan_state(options, maximum_k);
    ninfer::DeviceArena state_arena(layout.bytes);
    const ninfer::DeviceSpan backing{state_arena.base(), state_arena.capacity()};
    family::DecoderState decoder(backing, layout.decoder);
    family::RoundState round_storage(backing, layout.round);
    ninfer::Tensor prefill_hidden  = layout.prefill_hidden.bind(backing);
    ninfer::Tensor tail_hidden     = layout.tail_hidden.bind(backing);
    ninfer::Tensor boundary_hidden = layout.boundary_hidden.bind(backing);

    const std::size_t workspace_bytes =
        runtime::target_speculative_workspace_bytes(options.prefill_chunk, maximum_k);
    ninfer::WorkspaceArena workspace(workspace_bytes);
    ninfer::DeviceArena sampling_storage(kArenaAlignment);
    const ninfer::DeviceSpan sampling_span =
        sampling_storage.alloc_bytes(sizeof(ninfer::ops::SamplingConfig), kArenaAlignment);
    ninfer::ops::SamplingConfig sampling{};
    CUDA_CHECK(cudaMemcpyAsync(sampling_span.data, &sampling, sizeof(sampling),
                               cudaMemcpyHostToDevice, device.stream));
    CUDA_CHECK(cudaMemsetAsync(round_storage.speculative.stats.data, 0,
                               round_storage.speculative.stats.bytes(), device.stream));
    decoder.gdn.reset_running(device.stream);

    {
        runtime::schedule::TextContext prefill_card(
            device, model.runtime, workspace, decoder.text_kv, decoder.gdn, round_storage,
            prefill_hidden, options.prefill_chunk, 0, nullptr);
        prefill_card.set_sampling(
            static_cast<const ninfer::ops::SamplingConfig*>(sampling_span.data));
        const std::vector<ninfer::TokenId> prompt = prompt_tokens(options.context_tokens);
        prefill_card.prefill(prompt);
    }
    const std::int32_t base_slot = static_cast<std::int32_t>(maximum_k + 1);
    decoder.gdn.copy_slot(0, base_slot, device.stream);
    std::int32_t anchor = 0;
    CUDA_CHECK(cudaMemcpyAsync(&anchor, round_storage.token.data, sizeof(anchor),
                               cudaMemcpyDeviceToHost, device.stream));
    device.synchronize();

    std::cout << "format,ninfer_qwen3_6_35b_a3b_target_speculative_round_bench_v1\n";
    std::cout << "artifact," << options.artifact.string() << '\n';
    std::cout << "device," << device.props.name << '\n';
    std::cout << "state_bytes," << layout.bytes << '\n';
    std::cout << "workspace_bytes," << workspace_bytes << '\n';
    std::cout << "k,verify_tokens,accepted_drafts,mode,kv_dtype,context_tokens,mean_ms,median_ms,"
                 "min_ms,max_ms,target_side_effective_tok_s\n";

    for (const std::uint32_t k : options.draft_windows) {
        const std::uint32_t accepted = options.accepted_drafts.value_or(k);
        family::RoundState io        = round_window(round_storage, k);
        const ninfer::ops::GqaExecutionEnvelope envelope{options.context_tokens + k + 1,
                                                         options.context_tokens + k + 1};
        runtime::schedule::State state{
            device,
            model.runtime,
            workspace,
            decoder.text_kv,
            nullptr,
            decoder.gdn,
            io,
            prefill_hidden,
            options.prefill_chunk,
            options.context_tokens,
            static_cast<const ninfer::ops::SamplingConfig*>(sampling_span.data),
            ninfer::ProposalHead::Full,
            &tail_hidden,
            &boundary_hidden,
            nullptr,
            nullptr,
            nullptr};
        runtime::schedule::TextContext card(device, model.runtime, workspace, decoder.text_kv,
                                            decoder.gdn, io, prefill_hidden, options.prefill_chunk,
                                            options.context_tokens, nullptr);
        card.set_sampling(static_cast<const ninfer::ops::SamplingConfig*>(sampling_span.data));
        const std::vector<std::int32_t> drafts =
            prepare_drafts(state, card, k, accepted, anchor, base_slot, envelope);
        copy_vector(io.speculative.draft_tokens, drafts, device.stream);

        const auto reset = [&] {
            reset_round_controls(io, anchor, static_cast<std::int32_t>(options.context_tokens),
                                 base_slot, device.stream);
        };
        const auto body = [&] {
            runtime::schedule::speculative_verify_and_accept(state, card, k, envelope);
        };

        if (options.eager) {
            const Timing timing = measure(options, device, body, reset);
            print_row(options, k, accepted, "eager", timing);
        }
        if (options.graph) {
            ninfer::DecodeGraph graph;
            reset();
            device.synchronize();
            graph.capture(device.stream, body);
            const Timing timing =
                measure(options, device, [&] { graph.launch(device.stream); }, reset);
            print_row(options, k, accepted, "graph", timing);
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "ninfer_qwen3_6_35b_a3b_target_speculative_round_bench: " << error.what()
                  << '\n';
        return 1;
    }
}
