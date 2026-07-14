#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "product/prompt_input/prompt_input.h"
#include "runtime/engine/request_memory.h"
#include <ninfer/targets/qwen3_6_27b_rtx5090/package.h>
#include "targets/qwen3_6_27b_rtx5090/impl/diagnostic/activation_dump_access.h"
#include "targets/qwen3_6_27b_rtx5090/impl/frontend/frontend.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Json         = nlohmann::json;
namespace target   = ninfer::targets::qwen3_6_27b_rtx5090;
namespace detail   = target::detail;
namespace schedule = detail::schedule;

struct Options {
    bool help_requested = false;
    std::filesystem::path weights;
    std::vector<ninfer::TokenId> ids;
    std::filesystem::path messages;
    std::filesystem::path output;
    std::vector<ninfer::TokenId> stop_ids;
    std::uint32_t decode          = 0;
    std::uint32_t prefill_chunk   = 1024;
    std::uint32_t max_context     = 0;
    std::uint32_t mtp_drafts      = 0;
    ninfer::KvCacheStorage kv     = ninfer::KvCacheStorage::BFloat16;
    ninfer::ProposalHead proposal = ninfer::ProposalHead::Full;
    int device                    = 0;
    bool greedy                   = false;
    bool enable_thinking          = true;
    bool representative_only      = false;
};

constexpr std::string_view usage_text() {
    return "usage: ninfer-qwen3_6_27b-dump --weights MODEL.ninfer "
           "(--ids ID,...|--messages FILE.json)\n"
           "       --decode N --greedy --activation-dump DIR [--dump-level layer|vision-mtp]\n"
           "       [--prefill-chunk N] [--max-context N] [--kv-dtype bf16|int8]\n"
           "       [--stop-ids ID,...] [--mtp-draft-tokens 0..5]\n"
           "       [--proposal-head full|optimized] [--device N] [--no-thinking]\n"
           "\n"
           "Message content accepts text, image/image_url, and video/video_url parts; media\n"
           "sources may be local paths, HTTP(S) URLs, or base64 data URIs. The diagnostic\n"
           "uses the same owning message parser and media acquisition path as the product CLI.\n";
}

[[noreturn]] void usage_error(std::string message) {
    throw std::invalid_argument(std::move(message) + "\n" + std::string(usage_text()));
}

std::uint64_t parse_u64(std::string_view text, std::string_view label) {
    if (text.empty() || text.front() == '-') { usage_error("invalid " + std::string(label)); }
    std::uint64_t value     = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        usage_error("invalid " + std::string(label) + ": " + std::string(text));
    }
    return value;
}

std::uint32_t parse_u32(std::string_view text, std::string_view label, bool allow_zero = false) {
    const std::uint64_t value = parse_u64(text, label);
    if ((!allow_zero && value == 0) || value > std::numeric_limits<std::uint32_t>::max()) {
        usage_error("invalid " + std::string(label) + ": " + std::string(text));
    }
    return static_cast<std::uint32_t>(value);
}

std::vector<ninfer::TokenId> parse_ids(std::string_view text, bool allow_empty) {
    if (text.empty()) {
        if (allow_empty) { return {}; }
        usage_error("--ids must not be empty");
    }
    std::vector<ninfer::TokenId> result;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t comma     = text.find(',', begin);
        const std::size_t end       = comma == std::string_view::npos ? text.size() : comma;
        const std::string_view item = text.substr(begin, end - begin);
        const std::uint64_t value   = parse_u64(item, "token id");
        if (value > static_cast<std::uint64_t>(std::numeric_limits<ninfer::TokenId>::max())) {
            usage_error("token id exceeds int32: " + std::string(item));
        }
        result.push_back(static_cast<ninfer::TokenId>(value));
        if (comma == std::string_view::npos) { break; }
        begin = comma + 1;
    }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    std::optional<std::string> ids;
    std::optional<std::string> stop_ids;
    std::string dump_level = "layer";
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto value = [&](std::string_view flag) -> std::string_view {
            if (++index >= argc) { usage_error(std::string(flag) + " requires a value"); }
            return argv[index];
        };
        if (argument == "--weights") {
            options.weights = value(argument);
        } else if (argument == "--ids") {
            ids = value(argument);
        } else if (argument == "--messages") {
            options.messages = value(argument);
        } else if (argument == "--decode") {
            options.decode = parse_u32(value(argument), "decode");
        } else if (argument == "--prefill-chunk") {
            options.prefill_chunk = parse_u32(value(argument), "prefill-chunk");
        } else if (argument == "--max-context") {
            options.max_context = parse_u32(value(argument), "max-context");
        } else if (argument == "--kv-dtype") {
            const std::string_view kind = value(argument);
            if (kind == "bf16") {
                options.kv = ninfer::KvCacheStorage::BFloat16;
            } else if (kind == "int8") {
                options.kv = ninfer::KvCacheStorage::Int8Group64;
            } else {
                usage_error("--kv-dtype must be bf16 or int8");
            }
        } else if (argument == "--stop-ids") {
            stop_ids = value(argument);
        } else if (argument == "--activation-dump") {
            options.output = value(argument);
        } else if (argument == "--dump-level") {
            dump_level = value(argument);
        } else if (argument == "--mtp-draft-tokens") {
            options.mtp_drafts = parse_u32(value(argument), "mtp-draft-tokens", true);
        } else if (argument == "--proposal-head") {
            const std::string_view kind = value(argument);
            if (kind == "full") {
                options.proposal = ninfer::ProposalHead::Full;
            } else if (kind == "optimized") {
                options.proposal = ninfer::ProposalHead::Optimized;
            } else {
                usage_error("--proposal-head must be full or optimized");
            }
        } else if (argument == "--device") {
            const std::uint64_t device = parse_u64(value(argument), "device");
            if (device > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
                usage_error("--device exceeds int32");
            }
            options.device = static_cast<int>(device);
        } else if (argument == "--greedy") {
            options.greedy = true;
        } else if (argument == "--no-thinking") {
            options.enable_thinking = false;
        } else if (argument == "--help" || argument == "-h") {
            options.help_requested = true;
            return options;
        } else {
            usage_error("unknown argument: " + std::string(argument));
        }
    }
    if (options.weights.empty()) { usage_error("--weights is required"); }
    if (ids.has_value() == !options.messages.empty()) {
        usage_error("pass exactly one of --ids or --messages");
    }
    if (options.output.empty()) { usage_error("--activation-dump is required"); }
    if (options.decode == 0) { usage_error("--decode must be positive"); }
    if (!options.greedy) {
        usage_error("the initial diagnostic supports only explicit --greedy sampling");
    }
    if (dump_level != "layer" && dump_level != "vision-mtp") {
        usage_error("--dump-level must be layer or vision-mtp");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % 128 != 0) {
        usage_error("--prefill-chunk must be a nonzero multiple of 128");
    }
    if (options.mtp_drafts > 5) { usage_error("--mtp-draft-tokens must be in [0,5]"); }
    if (options.proposal == ninfer::ProposalHead::Optimized && options.mtp_drafts == 0) {
        usage_error("optimized proposal head requires MTP");
    }
    options.representative_only = dump_level == "vision-mtp";
    if (ids) { options.ids = parse_ids(*ids, false); }
    options.stop_ids = parse_ids(stop_ids.value_or(""), true);
    return options;
}

Json frontend_metadata(const detail::PreparedPrompt& prompt) {
    const detail::PreparedPromptData& data = detail::FrontendFactory::inspect(prompt);
    Json positions                         = Json::array();
    for (int axis = 0; axis < 3; ++axis) {
        const auto values = data.position_axis(axis);
        positions.push_back(std::vector<std::int32_t>(values.begin(), values.end()));
    }
    std::vector<std::uint32_t> token_types;
    token_types.reserve(data.token_types.size());
    for (const std::uint8_t value : data.token_types) { token_types.push_back(value); }
    return Json{{"token_ids", data.token_ids},
                {"token_types", std::move(token_types)},
                {"position_ids", std::move(positions)},
                {"rope_delta", data.rope_delta},
                {"media_items", data.prepare.media_items},
                {"raw_patches", data.prepare.raw_patches},
                {"vision_tokens", data.prepare.vision_tokens}};
}

std::string text_name(schedule::TapId id, int layer) {
    switch (id) {
    case schedule::TapId::AfterEmbed:
        return "embed";
    case schedule::TapId::AfterMixer:
        return "layer_" + (layer < 10 ? std::string("0") : std::string()) + std::to_string(layer) +
               "/mixer";
    case schedule::TapId::AfterMlp:
        return "layer_" + (layer < 10 ? std::string("0") : std::string()) + std::to_string(layer) +
               "/mlp";
    case schedule::TapId::AfterFinalNorm:
        return "final_norm";
    case schedule::TapId::AfterLogits:
        return "logits";
    }
    throw std::logic_error("unknown Text activation tap");
}

std::string vision_name(schedule::VisionTapId id, int layer) {
    switch (id) {
    case schedule::VisionTapId::PatchEmbed:
        return "vision/patch_embed";
    case schedule::VisionTapId::Block:
        return "vision/block_" + (layer < 10 ? std::string("0") : std::string()) +
               std::to_string(layer);
    case schedule::VisionTapId::Merger:
        return "vision/merger";
    }
    throw std::logic_error("unknown Vision activation tap");
}

std::string source_dtype(ninfer::DType dtype) {
    switch (dtype) {
    case ninfer::DType::BF16:
        return "bf16";
    case ninfer::DType::FP32:
        return "float32";
    default:
        throw std::invalid_argument("activation tap supports only BF16 and FP32 tensors");
    }
}

std::vector<float> copy_f32(const ninfer::Tensor& tensor, cudaStream_t stream) {
    if (tensor.data == nullptr || !tensor.is_contiguous() || tensor.numel() <= 0) {
        throw std::invalid_argument("activation tap requires a non-empty contiguous tensor");
    }
    const std::size_t elements = static_cast<std::size_t>(tensor.numel());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> result(elements);
    if (tensor.dtype == ninfer::DType::FP32) {
        CUDA_CHECK(cudaMemcpy(result.data(), tensor.data, elements * sizeof(float),
                              cudaMemcpyDeviceToHost));
        return result;
    }
    if (tensor.dtype != ninfer::DType::BF16) {
        throw std::invalid_argument("activation tap supports only BF16 and FP32 tensors");
    }
    std::vector<std::uint16_t> bits(elements);
    CUDA_CHECK(cudaMemcpy(bits.data(), tensor.data, elements * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));
    for (std::size_t index = 0; index < elements; ++index) {
        result[index] = std::bit_cast<float>(static_cast<std::uint32_t>(bits[index]) << 16U);
    }
    return result;
}

class ActivationWriter {
public:
    explicit ActivationWriter(std::filesystem::path root, bool representative_only)
        : root_(std::move(root)), representative_only_(representative_only) {
        std::filesystem::create_directories(root_);
    }

    void set_decode_context(std::uint32_t step, std::uint32_t position) noexcept {
        decode_step_     = step;
        decode_position_ = position;
    }

    static void text_callback(void* opaque, schedule::TapId id, int layer, schedule::Phase phase,
                              const ninfer::Tensor& tensor, cudaStream_t stream) {
        static_cast<ActivationWriter*>(opaque)->write_text(id, layer, phase, tensor, stream);
    }

    static void vision_callback(void* opaque, schedule::VisionTapId id, int layer,
                                const ninfer::Tensor& tensor, cudaStream_t stream) {
        static_cast<ActivationWriter*>(opaque)->write_vision(id, layer, tensor, stream);
    }

    void write_manifest(Json metadata) const {
        metadata["format"]  = "ninfer_activation_dump_v1";
        metadata["runtime"] = "cpp-ninfer";
        metadata["tensors"] = records_;
        std::ofstream output(root_ / "manifest.json");
        output << metadata.dump(2) << '\n';
        if (!output) { throw std::runtime_error("failed to write activation manifest"); }
    }

    [[nodiscard]] std::size_t record_count() const noexcept { return records_.size(); }

private:
    void write_text(schedule::TapId id, int layer, schedule::Phase phase,
                    const ninfer::Tensor& tensor, cudaStream_t stream) {
        std::string phase_name;
        std::uint32_t step     = 0;
        std::uint32_t chunk    = 0;
        std::uint32_t position = 0;
        if (phase == schedule::Phase::Prefill) {
            phase_name = "prefill";
            if (id == schedule::TapId::AfterEmbed) {
                current_prefill_chunk_    = next_prefill_chunk_++;
                current_prefill_position_ = prefill_position_;
                current_prefill_tokens_   = static_cast<std::uint32_t>(tensor.ne[1]);
                prefill_position_ += static_cast<std::uint32_t>(tensor.ne[1]);
            }
            chunk    = current_prefill_chunk_;
            position = current_prefill_position_;
            if (id == schedule::TapId::AfterLogits) { position += current_prefill_tokens_ - 1; }
        } else {
            // Program uses the verify schedule for both ordinary and speculative decode. The dump
            // labels it as decode because that is the externally comparable execution phase.
            phase_name = "decode";
            step       = decode_step_;
            position   = decode_position_;
        }
        if (representative_only_ &&
            (phase != schedule::Phase::Prefill || id != schedule::TapId::AfterEmbed ||
             current_prefill_chunk_ != 0)) {
            return;
        }
        write(text_name(id, layer), phase_name, step, chunk, position, tensor, stream);
    }

    void write_vision(schedule::VisionTapId id, int layer, const ninfer::Tensor& tensor,
                      cudaStream_t stream) {
        if (id == schedule::VisionTapId::Block && layer != 0 && layer != 13 && layer != 26) {
            return;
        }
        write(vision_name(id, layer), "prefill", 0, 0, 0, tensor, stream);
    }

    void write(const std::string& name, const std::string& phase, std::uint32_t step,
               std::uint32_t chunk, std::uint32_t position, const ninfer::Tensor& tensor,
               cudaStream_t stream) {
        if (tensor.ne[0] <= 0 || tensor.ne[1] <= 0 || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
            throw std::invalid_argument("activation tap expects a two-dimensional tensor");
        }
        const std::filesystem::path prefix =
            phase == "prefill" ? std::filesystem::path("prefill") / ("chunk_" + four_digits(chunk))
                               : std::filesystem::path("decode") / four_digits(step);
        const std::filesystem::path relative    = prefix / (name + ".f32");
        const std::filesystem::path destination = root_ / relative;
        std::filesystem::create_directories(destination.parent_path());
        const std::vector<float> values = copy_f32(tensor, stream);
        std::ofstream output(destination, std::ios::binary);
        output.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (!output) { throw std::runtime_error("failed to write activation tensor"); }
        records_.push_back(Json{{"name", name},
                                {"file", relative.generic_string()},
                                {"shape", {tensor.ne[1], tensor.ne[0]}},
                                {"source_dtype", source_dtype(tensor.dtype)},
                                {"phase", phase},
                                {"step", step},
                                {"chunk", chunk},
                                {"position_begin", position},
                                {"position_end", position + tensor.ne[1]}});
    }

    static std::string four_digits(std::uint32_t value) {
        std::string result = std::to_string(value);
        if (result.size() < 4) { result.insert(result.begin(), 4 - result.size(), '0'); }
        return result;
    }

    std::filesystem::path root_;
    bool representative_only_               = false;
    Json records_                           = Json::array();
    std::uint32_t prefill_position_         = 0;
    std::uint32_t next_prefill_chunk_       = 0;
    std::uint32_t current_prefill_chunk_    = 0;
    std::uint32_t current_prefill_position_ = 0;
    std::uint32_t current_prefill_tokens_   = 0;
    std::uint32_t decode_step_              = 0;
    std::uint32_t decode_position_          = 0;
};

bool is_stop(ninfer::TokenId token, std::span<const ninfer::TokenId> stops) {
    return std::find(stops.begin(), stops.end(), token) != stops.end();
}

std::uint32_t terminal_count(std::span<const ninfer::TokenId> tokens,
                             std::span<const ninfer::TokenId> stops, std::uint32_t remaining) {
    const std::uint32_t bounded =
        std::min<std::uint32_t>(remaining, static_cast<std::uint32_t>(tokens.size()));
    for (std::uint32_t index = 0; index < bounded; ++index) {
        if (is_stop(tokens[index], stops)) { return index + 1; }
    }
    return bounded;
}

int run(const Options& options) {
    ninfer::DeviceContext device(options.device);
    ninfer::artifact::Reader reader(options.weights);
    if (reader.model_id() != target::Package::model_id) {
        throw std::invalid_argument("artifact model_id does not match qwen3.6-27b");
    }
    ninfer::artifact::Binder binder(reader);
    auto load_plan = target::Package::plan_load(binder);
    auto materialized =
        ninfer::artifact::materialize(reader, load_plan.materialization(), device, nullptr);
    auto model =
        target::Package::construct_loaded_model(std::move(load_plan), std::move(materialized));
    auto frontend = target::Package::make_frontend(*model);
    auto prompt   = options.messages.empty() ? frontend.prepare_tokens(options.ids, false)
                                             : frontend.prepare(ninfer::product::prompt_from_messages(
                                                 options.messages, options.enable_thinking));
    const ninfer::PromptSummary prompt_summary = prompt.summary();
    Json frontend_dump                         = frontend_metadata(prompt);

    ninfer::EngineOptions engine;
    engine.artifact_path             = options.weights;
    engine.device                    = options.device;
    engine.prefill_chunk             = options.prefill_chunk;
    engine.kv_cache                  = options.kv;
    engine.speculative.draft_tokens  = options.mtp_drafts;
    engine.speculative.proposal_head = options.proposal;
    engine.use_cuda_graph            = false;
    const std::uint64_t required     = static_cast<std::uint64_t>(prompt_summary.prompt_tokens) +
                                   options.decode + 2ULL * options.mtp_drafts;
    engine.max_context = options.max_context != 0
                             ? options.max_context
                             : static_cast<std::uint32_t>(std::max<std::uint64_t>(2048, required));
    if (engine.max_context < required) {
        throw std::invalid_argument("--max-context is too small for the diagnostic schedule");
    }

    auto sequence_plan = target::Package::plan_sequence(*model, device, engine);
    auto program       = target::Package::create_program(*model, std::move(sequence_plan), device);
    ninfer::runtime::RequestMemory transient(device);
    ninfer::ExecutionOptions execution;
    execution.requested_output_tokens = options.decode;
    execution.sampling                = {};
    execution.allow_prefix_reuse      = false;
    auto request_plan                 = program->plan_request(prompt, execution);
    const auto summary                = request_plan.summary();
    transient.ensure(summary.transient_bytes, summary.transient_alignment);

    ActivationWriter writer(options.output, options.representative_only);
    detail::ActivationDumpAccess::attach(*program, &writer, &ActivationWriter::text_callback,
                                         &ActivationWriter::vision_callback);
    std::vector<ninfer::TokenId> generated;
    generated.reserve(options.decode);
    std::string stop_reason = "length";
    try {
        auto first = program->begin(std::move(prompt), std::move(request_plan), transient.region());
        const auto tokens         = first.round.tokens();
        const std::uint32_t count = terminal_count(tokens, options.stop_ids, options.decode);
        generated.insert(generated.end(), tokens.begin(),
                         tokens.begin() + static_cast<std::ptrdiff_t>(count));
        const bool stopped = is_stop(generated.back(), options.stop_ids);
        if (stopped || generated.size() == options.decode) {
            stop_reason = stopped ? "stop_token" : "length";
            (void)std::move(first.round).commit_prefix_and_finish(count);
        } else {
            std::move(first.round).commit_all();
        }

        while (generated.size() < options.decode && !is_stop(generated.back(), options.stop_ids)) {
            const auto sequence = program->sequence_summary();
            writer.set_decode_context(static_cast<std::uint32_t>(generated.size() - 1),
                                      sequence.materialized_tokens);
            const std::uint32_t remaining =
                options.decode - static_cast<std::uint32_t>(generated.size());
            auto round = program->decode_round(
                ninfer::runtime::RoundBudget{.generated_tokens_remaining = remaining});
            const auto tokens          = round.tokens();
            const std::uint32_t count  = terminal_count(tokens, options.stop_ids, remaining);
            const bool stopped         = is_stop(tokens[count - 1], options.stop_ids);
            const bool terminal_length = generated.size() + count == options.decode;
            generated.insert(generated.end(), tokens.begin(),
                             tokens.begin() + static_cast<std::ptrdiff_t>(count));
            if (stopped || terminal_length) {
                stop_reason = stopped ? "stop_token" : "length";
                (void)std::move(round).commit_prefix_and_finish(count);
            } else {
                std::move(round).commit_all();
            }
        }
    } catch (...) {
        detail::ActivationDumpAccess::detach(*program);
        throw;
    }
    detail::ActivationDumpAccess::detach(*program);

    const ninfer::SpeculativeStats speculative = program->speculative_stats();
    Json speculative_dump{{"enabled", speculative.enabled},
                          {"draft_window", speculative.draft_window},
                          {"rounds", speculative.rounds},
                          {"drafted_tokens", speculative.drafted_tokens},
                          {"accepted_tokens", speculative.accepted_tokens},
                          {"fallback_steps", speculative.fallback_steps},
                          {"accepted_per_position", speculative.accepted_per_position}};

    Json sampling{{"temperature", 0.0},
                  {"top_k", 0},
                  {"top_p", 1.0},
                  {"min_p", 0.0},
                  {"presence_penalty", 0.0},
                  {"frequency_penalty", 0.0},
                  {"seed", 0}};
    writer.write_manifest(
        Json{{"target", "qwen3_6_27b_rtx5090"},
             {"weights", std::filesystem::absolute(options.weights).string()},
             {"prompt_source", options.messages.empty() ? "ids" : "messages"},
             {"messages", options.messages.empty()
                              ? Json(nullptr)
                              : Json(std::filesystem::absolute(options.messages).string())},
             {"frontend", std::move(frontend_dump)},
             {"generated_token_ids", generated},
             {"stop_reason", stop_reason},
             {"sampling", std::move(sampling)},
             {"kv_dtype", options.kv == ninfer::KvCacheStorage::BFloat16 ? "bf16" : "int8"},
             {"prefill_chunk", options.prefill_chunk},
             {"mtp_draft_tokens", options.mtp_drafts},
             {"draft_head", options.proposal == ninfer::ProposalHead::Full ? "full" : "optimized"},
             {"speculative", std::move(speculative_dump)},
             {"vision", prompt_summary.has_media}});
    std::cout << Json{{"format", "ninfer_activation_dump_summary_v1"},
                      {"output", std::filesystem::absolute(options.output).string()},
                      {"prompt_tokens", prompt_summary.prompt_tokens},
                      {"vision", prompt_summary.has_media},
                      {"generated_token_ids", generated},
                      {"tensor_records", writer.record_count()}}
                     .dump()
              << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.help_requested) {
            std::cout << usage_text();
            return 0;
        }
        return run(options);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
