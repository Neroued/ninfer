#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>

#include "targets/qwen3_6/impl/frontend/chat_template.h"
#include "targets/qwen3_6/impl/frontend/digest.h"
#include "targets/qwen3_6/impl/frontend/media_cache.h"
#include "targets/qwen3_6/impl/frontend/processor.h"
#include "targets/qwen3_6/impl/frontend/test_access.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"
#include "text/unicode.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Frontend          = ninfer::targets::qwen3_6::Frontend;
using FrontendFactory   = ninfer::targets::qwen3_6::FrontendTestAccess;
using FrontendResources = ninfer::targets::qwen3_6::FrontendResources;
using PublishedOutput   = ninfer::targets::qwen3_6::PublishedOutput;
namespace fi            = ninfer::targets::qwen3_6::frontend_internal;

constexpr std::string_view kThinkingControlGuidance =
    "\n\n Considering the limited time by the user, I have to give the solution based on the "
    "thinking directly now.\n";
constexpr std::string_view kThinkingControl =
    "\n\n Considering the limited time by the user, I have to give the solution based on the "
    "thinking directly now.\n</think>\n\n";
constexpr std::string_view kUtf8Replacement = "\xef\xbf\xbd";

constexpr ninfer::TokenId kByte80Token = 13;
constexpr ninfer::TokenId kByteE0Token = 14;
constexpr ninfer::TokenId kByteEDToken = 15;
constexpr ninfer::TokenId kByteA0Token = 16;
constexpr ninfer::TokenId kByteF4Token = 17;
constexpr ninfer::TokenId kByte90Token = 18;
constexpr ninfer::TokenId kByteF5Token = 19;
constexpr ninfer::TokenId kByteF0Token = 20;
constexpr ninfer::TokenId kByte9FToken = 21;
constexpr ninfer::TokenId kByte98Token = 22;
constexpr ninfer::TokenId kByteC2Token = 23;
constexpr ninfer::TokenId kByteA2Token = 24;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

float bf16_value(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

std::uint16_t bf16_bits(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

std::string read_file(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error(std::string("failed to open test resource: ") + path); }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

std::string read_template_fixture(const char* path) {
    std::string source = read_file(path);
    if (!source.empty() && source.back() == '\n') { source.pop_back(); }
    return source;
}

std::string official_resource_path(std::string_view filename) {
    const char* directory = std::getenv("NINFER_QWEN3_6_FRONTEND_RESOURCES");
    if (directory == nullptr || *directory == '\0') {
        throw std::runtime_error(
            "NINFER_QWEN3_6_FRONTEND_RESOURCES must name a directory containing the official "
            "Qwen3.6 frontend JSON resources");
    }
    std::string path(directory);
    if (path.back() != '/') { path.push_back('/'); }
    path.append(filename);
    return path;
}

std::string read_official_resource(std::string_view filename) {
    return read_file(official_resource_path(filename).c_str());
}

const std::string& thinking_toggle_template_source() {
    static const std::string source = read_template_fixture(
        NINFER_SOURCE_DIR "/tests/fixtures/frontend/thinking_toggle_chat_template.jinja");
    return source;
}

const std::string& reasoning_effort_template_source() {
    static const std::string source = read_template_fixture(
        NINFER_SOURCE_DIR "/tests/fixtures/frontend/reasoning_effort_chat_template.jinja");
    return source;
}

const fi::CompiledChatTemplate& thinking_toggle_template() {
    static const fi::CompiledChatTemplate value =
        fi::CompiledChatTemplate::resolve(thinking_toggle_template_source());
    return value;
}

const fi::CompiledChatTemplate& reasoning_effort_template() {
    static const fi::CompiledChatTemplate value =
        fi::CompiledChatTemplate::resolve(reasoning_effort_template_source());
    return value;
}

const fi::Tokenizer& official_tokenizer() {
    static const std::string tokenizer_json =
        read_official_resource("tokenizer.json");
    static const std::string tokenizer_config_json =
        read_official_resource("tokenizer_config.json");
    static const std::string generation_config_json =
        read_official_resource("generation_config.json");
    static const fi::Tokenizer tokenizer({.tokenizer_json         = tokenizer_json,
                                          .tokenizer_config_json  = tokenizer_config_json,
                                          .generation_config_json = generation_config_json});
    return tokenizer;
}

nlohmann::json added(int id, std::string content, bool special = false) {
    return nlohmann::json{{"id", id},
                          {"content", std::move(content)},
                          {"single_word", false},
                          {"lstrip", false},
                          {"rstrip", false},
                          {"normalized", false},
                          {"special", special}};
}

nlohmann::json decoder_added(std::string content, bool special = false) {
    nlohmann::json value = added(0, std::move(content), special);
    value.erase("id");
    return value;
}

std::string byte_level_symbol(std::uint8_t target) {
    std::uint32_t next = 256;
    for (int value = 0; value <= 255; ++value) {
        const bool visible = (value >= 33 && value <= 126) || (value >= 161 && value <= 172) ||
                             (value >= 174 && value <= 255);
        const std::uint32_t codepoint = visible ? static_cast<std::uint32_t>(value) : next++;
        if (value == target) {
            return ninfer::text::unicode_internal::codepoint_to_utf8(
                static_cast<std::int32_t>(codepoint));
        }
    }
    throw std::logic_error("byte-level test symbol is outside one byte");
}

FrontendResources resources(const std::string& chat_template = thinking_toggle_template_source()) {
    FrontendResources result;
    result.chat_template_jinja  = chat_template;
    const nlohmann::json tokens = nlohmann::json::array(
        {added(1, "helloST"), added(2, "OPtail"), added(3, "thought</thi"),
         added(4, "nk>\n\nanswer"), added(6, "<eos>", true), added(7, "<0.0 seconds>"),
         added(8, std::string(kThinkingControlGuidance)), added(30, "user\n"),
         added(31, "assistant\n"), added(32, "\n"), added(248045, "<|im_start|>", true),
         added(248046, "<|im_end|>", true), added(248053, "<|vision_start|>", true),
         added(248054, "<|vision_end|>", true), added(248056, "<|image_pad|>", true),
         added(248057, "<|video_pad|>", true), added(248068, "<think>"),
         added(248069, "</think>")});
    nlohmann::json vocab           = {{"x", 0}, {"ä", 10}, {"¸", 11}, {"Ń", 12}};
    vocab[byte_level_symbol(0x80)] = kByte80Token;
    vocab[byte_level_symbol(0xe0)] = kByteE0Token;
    vocab[byte_level_symbol(0xed)] = kByteEDToken;
    vocab[byte_level_symbol(0xa0)] = kByteA0Token;
    vocab[byte_level_symbol(0xf4)] = kByteF4Token;
    vocab[byte_level_symbol(0x90)] = kByte90Token;
    vocab[byte_level_symbol(0xf5)] = kByteF5Token;
    vocab[byte_level_symbol(0xf0)] = kByteF0Token;
    vocab[byte_level_symbol(0x9f)] = kByte9FToken;
    vocab[byte_level_symbol(0x98)] = kByte98Token;
    vocab[byte_level_symbol(0xc2)] = kByteC2Token;
    vocab[byte_level_symbol(0xa2)] = kByteA2Token;
    result.tokenizer_json          = nlohmann::json{
                 {"model",
                  {{"type", "BPE"}, {"vocab", std::move(vocab)}, {"merges", nlohmann::json::array()}}},
                 {"added_tokens",
                  tokens}}.dump();

    nlohmann::json decoder = nlohmann::json::object();
    for (const nlohmann::json& token : tokens) {
        nlohmann::json value = token;
        const std::string id = std::to_string(value.at("id").get<int>());
        value.erase("id");
        decoder[id] = std::move(value);
    }
    decoder["248070"]            = decoder_added("<|audio_start|>", true);
    decoder["248071"]            = decoder_added("<|audio_end|>", true);
    decoder["248072"]            = decoder_added("<tts_pad>", true);
    decoder["248073"]            = decoder_added("<tts_text_bos>", true);
    decoder["248074"]            = decoder_added("<tts_text_eod>", true);
    decoder["248075"]            = decoder_added("<tts_text_bos_single>", true);
    decoder["248076"]            = decoder_added("<|audio_pad|>", true);
    result.tokenizer_config_json = nlohmann::json{
        {"add_bos_token", false},
        {"add_prefix_space", false},
        {"pad_token", "<|endoftext|>"},
        {"chat_template", result.chat_template_jinja},
        {"added_tokens_decoder",
         std::move(decoder)}}.dump();
    result.generation_config_json = R"({"eos_token_id":[6]})";
    result.preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":16777216}})";
    result.video_preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":25165824}})";
    return result;
}

std::vector<std::uint8_t> gradient_ppm() {
    std::vector<std::uint8_t> ppm;
    const std::string header = "P6\n64 64\n255\n";
    for (const char byte : header) {
        ppm.push_back(static_cast<std::uint8_t>(static_cast<unsigned char>(byte)));
    }
    for (int index = 0; index < 64 * 64; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    return ppm;
}

std::vector<std::uint8_t> block_ppm(int width, int height, std::uint8_t value) {
    const std::string header =
        "P6\n" + std::to_string(width) + ' ' + std::to_string(height) + "\n255\n";
    std::vector<std::uint8_t> ppm;
    ppm.reserve(header.size() +
                static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3);
    for (const char byte : header) {
        ppm.push_back(static_cast<std::uint8_t>(static_cast<unsigned char>(byte)));
    }
    ppm.insert(ppm.end(), static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3,
               value);
    return ppm;
}

ninfer::PromptInput image_text_input(std::vector<std::uint8_t> bytes, std::string text,
                                     std::string source_name) {
    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = std::move(bytes);
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = std::move(source_name);

    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(std::move(image));
    if (!text.empty()) {
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
    }
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    return input;
}

ninfer::PromptInput image_input() {
    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = gradient_ppm();
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "inline.ppm";
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(std::move(image));
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    return input;
}

bool near(float actual, float expected) { return std::abs(actual - expected) < 1.0e-6F; }

constexpr std::array<std::uint8_t, 32> kGradientDigest{
    0x1e, 0x8c, 0xd9, 0x22, 0x40, 0xfa, 0x10, 0x62, 0x7b, 0x60, 0x86, 0x8e, 0xe9, 0x66, 0x41, 0xa2,
    0x4d, 0x21, 0xff, 0xc7, 0xe9, 0xa2, 0x2b, 0x34, 0xc0, 0xec, 0x99, 0x84, 0x6c, 0xa9, 0xa4, 0x8a,
};

std::string channel_text(const PublishedOutput& output, ninfer::OutputChannel channel) {
    std::string result;
    for (const ninfer::OutputDelta& delta : output) {
        if (delta.channel == channel) { result += delta.text; }
    }
    return result;
}

fi::ChatMessage chat_message(ninfer::ChatRole role, std::string content) {
    fi::ChatMessage message;
    message.role = role;
    message.parts.push_back(fi::ChatPart::text_part(std::move(content)));
    return message;
}

fi::RenderedChat render_chat(std::vector<fi::ChatMessage> messages,
                             fi::ChatRenderOptions options = {}) {
    return thinking_toggle_template().render(messages, std::move(options));
}

std::string render_chat_text(std::vector<fi::ChatMessage> messages,
                             fi::ChatRenderOptions options = {}) {
    return render_chat(std::move(messages), std::move(options)).text;
}

template <class Callable>
bool throws_invalid_argument(Callable&& callable) {
    try {
        callable();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

int test_invalid_public_part_enums(const Frontend& frontend) {
    ninfer::ChatMessage invalid_part_message;
    invalid_part_message.role = ninfer::ChatRole::User;
    invalid_part_message.parts.push_back(ninfer::MessagePart{
        .kind = static_cast<ninfer::MessagePartKind>(255), .text = "invalid", .media = {}});
    ninfer::PromptInput invalid_part;
    invalid_part.messages.push_back(std::move(invalid_part_message));

    ninfer::MessagePart media;
    media.kind       = ninfer::MessagePartKind::Media;
    media.media.kind = static_cast<ninfer::MediaKind>(255);
    media.media.bytes.push_back(0);
    ninfer::ChatMessage invalid_media_message;
    invalid_media_message.role = ninfer::ChatRole::User;
    invalid_media_message.parts.push_back(std::move(media));
    ninfer::PromptInput invalid_media;
    invalid_media.messages.push_back(std::move(invalid_media_message));

    int failures =
        check(throws_invalid_argument([&] { (void)frontend.prepare(std::move(invalid_part)); }),
              "invalid public message-part kind was accepted");
    failures +=
        check(throws_invalid_argument([&] { (void)frontend.prepare(std::move(invalid_media)); }),
              "invalid public media kind was accepted as video");
    return failures;
}

template <class Callable>
bool throws_processor_budget(Callable&& callable) {
    try {
        callable();
    } catch (const fi::ProcessorError& error) {
        return error.kind() == fi::ProcessorErrorKind::BudgetExceeded;
    }
    return false;
}

template <class Callable>
bool throws_context_length(Callable&& callable) {
    try {
        callable();
    } catch (const ninfer::RequestError& error) {
        return error.kind() == ninfer::RequestErrorKind::ContextLengthExceeded;
    }
    return false;
}

int test_official_tokenizer_merge() {
    const fi::Tokenizer& tokenizer = official_tokenizer();

    constexpr std::array<std::pair<const char*, int>, 7> appended = {{
        {"<|audio_start|>", 248070},
        {"<|audio_end|>", 248071},
        {"<tts_pad>", 248072},
        {"<tts_text_bos>", 248073},
        {"<tts_text_eod>", 248074},
        {"<tts_text_bos_single>", 248075},
        {"<|audio_pad|>", 248076},
    }};
    int failures = check(tokenizer.has_exact_token_domain(248077),
                         "official tokenizer merge left a hole in the token domain");
    for (const auto& [text, id] : appended) {
        const std::vector<int> encoded = tokenizer.encode(text);
        failures += check(encoded == std::vector<int>{id} && tokenizer.is_special_token(id) &&
                              tokenizer.decode_token_bytes(id) == text,
                          "official tokenizer_config.json token did not merge exactly");
    }

    FrontendResources conflicting = resources();
    nlohmann::json config         = nlohmann::json::parse(conflicting.tokenizer_config_json);
    config["added_tokens_decoder"]["248045"]["special"] = false;
    conflicting.tokenizer_config_json                   = config.dump();
    failures += check(
        throws_invalid_argument([&] {
            fi::Tokenizer invalid({.tokenizer_json         = conflicting.tokenizer_json,
                                   .tokenizer_config_json  = conflicting.tokenizer_config_json,
                                   .generation_config_json = conflicting.generation_config_json});
        }),
        "conflicting tokenizer/tokenizer_config added-token definitions were accepted");
    return failures;
}

int test_bpe_merge_order() {
    const std::string tokenizer_json = nlohmann::json{
        {"model",
         {{"type", "BPE"},
          {"vocab", {{"a", 0}, {"aa", 1}, {"aaa", 2}, {"b", 3}, {"c", 4}, {"bc", 5}, {"abc", 6}}},
          {"merges",
           nlohmann::json::array(
               {nlohmann::json::array({"a", "a"}), nlohmann::json::array({"aa", "a"}),
                nlohmann::json::array({"b", "c"}), nlohmann::json::array({"a", "bc"})})}}},
        {"added_tokens",
         nlohmann::json::array()}}.dump();
    const std::string tokenizer_config_json =
        nlohmann::json{{"added_tokens_decoder", nlohmann::json::object()}}.dump();
    const fi::Tokenizer tokenizer({.tokenizer_json         = tokenizer_json,
                                   .tokenizer_config_json  = tokenizer_config_json,
                                   .generation_config_json = R"({"eos_token_id":0})"});
    return check(tokenizer.encode("aaa") == std::vector<int>{2} &&
                     tokenizer.encode("aaaa") == std::vector<int>({1, 1}) &&
                     tokenizer.encode("abc") == std::vector<int>{6},
                 "priority BPE changed rank or leftmost merge semantics");
}

int test_boundary_aware_tokenization() {
    const std::string tokenizer_json = nlohmann::json{
        {"model",
         {{"type", "BPE"},
          {"vocab", {{"a", 0}, {"b", 1}, {"c", 2}, {"bc", 3}, {"ab", 4}}},
          {"merges", nlohmann::json::array(
                         {nlohmann::json::array({"b", "c"}), nlohmann::json::array({"a", "b"})})}}},
        {"added_tokens",
         nlohmann::json::array()}}.dump();
    const std::string tokenizer_config_json =
        nlohmann::json{{"added_tokens_decoder", nlohmann::json::object()}}.dump();
    const fi::Tokenizer tokenizer({.tokenizer_json         = tokenizer_json,
                                   .tokenizer_config_json  = tokenizer_config_json,
                                   .generation_config_json = R"({"eos_token_id":0})"});
    constexpr std::array<std::size_t, 5> boundaries{2, 3, 0, 1, 2};
    const fi::BoundaryEncodedText encoded = tokenizer.encode_with_boundaries("abc", boundaries);
    int failures                          = check(
        encoded.input_ids == std::vector<int>({0, 3}) && encoded.boundaries.size() == 5 &&
            !encoded.boundaries[0].exact_frontier && encoded.boundaries[0].stable_frontier == 0 &&
            encoded.boundaries[1].exact_frontier == 2 &&
            encoded.boundaries[2].exact_frontier == 0 &&
            encoded.boundaries[3].exact_frontier == 1 && !encoded.boundaries[4].exact_frontier,
        "boundary-aware tokenizer changed crossing-token or result-order semantics");

    constexpr std::string_view decomposed = "e\xCC\x81x";
    constexpr std::array<std::size_t, 1> composition_boundary{1};
    const fi::Tokenizer& official = official_tokenizer();
    const fi::BoundaryEncodedText normalized =
        official.encode_with_boundaries(decomposed, composition_boundary);
    failures += check(normalized.input_ids == official.encode(decomposed) &&
                          !normalized.boundaries.front().exact_frontier &&
                          normalized.boundaries.front().stable_frontier == 0,
                      "boundary-aware tokenizer split an NFC composition sequence");
    constexpr std::array<fi::ByteSpan, 2> literal_spans{fi::ByteSpan{.begin = 0, .end = 1},
                                                        fi::ByteSpan{.begin = 1, .end = 2}};
    const fi::BoundaryEncodedText annotated =
        tokenizer.encode_with_boundaries("abc", {}, {}, literal_spans);
    failures += check(annotated.input_ids == encoded.input_ids,
                      "literal provenance introduced an artificial BPE boundary");
    return failures;
}

int test_literal_added_token_provenance() {
    const fi::Tokenizer& tokenizer    = official_tokenizer();
    constexpr std::string_view marker = "<|image_pad|>";
    constexpr std::array<fi::ByteSpan, 2> split_literal{
        fi::ByteSpan{.begin = 0, .end = 5},
        fi::ByteSpan{.begin = 5, .end = marker.size()},
    };
    const std::vector<int> ordinary =
        tokenizer.encode(marker, fi::EncodeOptions{.parse_added_tokens = false});
    const fi::BoundaryEncodedText annotated =
        tokenizer.encode_with_boundaries(marker, {}, {}, split_literal);
    int failures = check(annotated.input_ids == ordinary &&
                             std::find(annotated.input_ids.begin(), annotated.input_ids.end(),
                                       248056) == annotated.input_ids.end(),
                         "literal Vision token became an added token across text spans");

    const std::string mixed         = "<|im_start|>x<|image_pad|><|im_end|>";
    const std::size_t literal_begin = mixed.find(marker);
    const std::array<fi::ByteSpan, 1> literal{
        fi::ByteSpan{literal_begin, literal_begin + marker.size()}};
    const std::vector<int> mixed_tokens =
        tokenizer.encode_with_boundaries(mixed, {}, {}, literal).input_ids;
    failures += check(
        !mixed_tokens.empty() && mixed_tokens.front() == 248045 && mixed_tokens.back() == 248046 &&
            std::find(mixed_tokens.begin(), mixed_tokens.end(), 248056) == mixed_tokens.end(),
        "literal exclusion suppressed template-owned control tokens");
    return failures;
}

int test_repeated_special_tokens_scan_linearly() {
    constexpr std::string_view token = "<|image_pad|>";
    std::string text;
    text.reserve(token.size() * 5'000);
    for (int index = 0; index < 5'000; ++index) { text += token; }
    const std::vector<int> encoded = official_tokenizer().encode(text);
    return check(encoded.size() == 5'000 && std::all_of(encoded.begin(), encoded.end(),
                                                        [](int id) { return id == 248056; }),
                 "repeated special-token scan changed tokenization semantics");
}

int test_bounded_tokenizer_prefix() {
    const fi::Tokenizer& tokenizer = official_tokenizer();
    const std::string text =
        "<|im_start|>user\nA bounded tokenizer must preserve the exact ordinary and special-token "
        "prefix.<|im_end|>\n";
    const std::vector<int> full    = tokenizer.encode(text);
    constexpr std::size_t limit    = 7;
    const std::vector<int> bounded = tokenizer.encode(text, fi::EncodeOptions{.max_tokens = limit});
    const std::vector<int> roomy =
        tokenizer.encode(text, fi::EncodeOptions{.max_tokens = full.size() + 1U});
    return check(full.size() > limit && bounded.size() == limit &&
                     std::equal(bounded.begin(), bounded.end(), full.begin()) && roomy == full,
                 "bounded tokenizer output is not the exact prefix of unbounded tokenization");
}

int test_context_capacity_guard() {
    ninfer::PromptInput input;
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    input.messages.push_back(std::move(message));

    const Frontend counting         = FrontendFactory::create_component(resources(), false);
    const std::uint32_t exact_count = counting.count_tokens(input);
    ninfer::targets::qwen3_6::FrontendOptions exact_options;
    exact_options.vision_enabled = false;
    exact_options.max_context    = exact_count;
    const Frontend exact         = FrontendFactory::create_component(resources(), exact_options);
    int failures = check(exact.prepare(input).summary().prompt_tokens == exact_count,
                         "Frontend rejected a prompt exactly at max_context");

    ninfer::targets::qwen3_6::FrontendOptions short_options = exact_options;
    short_options.max_context                               = exact_count - 1U;
    const Frontend short_frontend = FrontendFactory::create_component(resources(), short_options);
    failures += check(short_frontend.count_tokens(input) == exact_count,
                      "exact token counting was incorrectly bounded by max_context");
    failures += check(throws_context_length([&] { (void)short_frontend.prepare(input); }),
                      "Frontend accepted a text prompt at max_context + 1");

    std::vector<ninfer::TokenId> exact_tokens(exact_count, 0);
    failures += check(exact.prepare_tokens(exact_tokens).summary().prompt_tokens == exact_count,
                      "prepare_tokens rejected an exact-capacity token vector");
    exact_tokens.push_back(0);
    failures +=
        check(throws_context_length([&] { (void)exact.prepare_tokens(std::move(exact_tokens)); }),
              "prepare_tokens accepted a token vector at max_context + 1");

    ninfer::targets::qwen3_6::FrontendOptions media_options = short_options;
    media_options.vision_enabled                            = true;
    const Frontend media_frontend = FrontendFactory::create_component(resources(), media_options);
    failures += check(throws_context_length([&] {
                          (void)media_frontend.prepare(
                              image_text_input({0}, std::string(64, 'x'), "must-not-decode.bin"));
                      }),
                      "over-capacity media prompt was not rejected before media decoding");

    std::atomic<int> control_checks{0};
    ninfer::PreparationControl cancelled_during_tokenization{
        .deadline     = {},
        .cancellation = ninfer::CancellationView(
            [&control_checks] { return control_checks.fetch_add(1) >= 2; }),
    };
    try {
        (void)media_frontend.prepare(
            image_text_input({0}, std::string(64, 'x'), "cancel-before-oversize.bin"),
            cancelled_during_tokenization);
        failures += check(false, "cancelled over-capacity media prompt completed successfully");
    } catch (const ninfer::RequestError& error) {
        failures +=
            check(error.kind() == ninfer::RequestErrorKind::Cancelled && control_checks.load() == 3,
                  "media over-capacity result took priority over tokenization cancellation");
    }
    return failures;
}

int test_official_chat_template() {
    int failures = 0;
    failures += check(render_chat_text({chat_message(ninfer::ChatRole::User, "hello")}) ==
                          "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n",
                      "ordinary user prompt differs from the official template");

    fi::ChatRenderOptions no_generation;
    no_generation.add_generation_prompt = false;
    failures +=
        check(render_chat_text({chat_message(ninfer::ChatRole::System, "  be concise  "),
                                chat_message(ninfer::ChatRole::User, "hello")},
                               no_generation) == "<|im_start|>system\nbe concise<|im_end|>\n"
                                                 "<|im_start|>user\nhello<|im_end|>\n",
              "leading system prompt differs from the official template");
    failures += check(render_chat_text({chat_message(ninfer::ChatRole::System, ""),
                                        chat_message(ninfer::ChatRole::User, "hello")},
                                       no_generation) ==
                          "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n",
                      "empty leading system prompt differs from the official template");

    fi::ChatMessage tool_assistant = chat_message(ninfer::ChatRole::Assistant, "");
    tool_assistant.tool_calls.push_back(
        {.id = "", .name = "f", .arguments_json = R"({"flag":true,"nested":{"x":[1,2]}})"});
    failures += check(render_chat_text({chat_message(ninfer::ChatRole::User, "hi"), tool_assistant},
                                       no_generation) ==
                          "<|im_start|>user\nhi<|im_end|>\n"
                          "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                          "<tool_call>\n<function=f>\n<parameter=flag>\ntrue\n</parameter>\n"
                          "<parameter=nested>\n{\"x\": [1, 2]}\n</parameter>\n"
                          "</function>\n</tool_call><|im_end|>\n",
                      "nested or boolean tool arguments differ from official JSON rendering");

    fi::ChatMessage preamble   = chat_message(ninfer::ChatRole::Assistant, "Let me check:");
    preamble.reasoning_content = "I should inspect.";
    preamble.tool_calls.push_back(
        {.id = "call_read", .name = "read_file", .arguments_json = R"({"path":"a"})"});
    failures += check(render_chat_text({chat_message(ninfer::ChatRole::User, "inspect"), preamble},
                                       no_generation) ==
                          "<|im_start|>user\ninspect<|im_end|>\n"
                          "<|im_start|>assistant\n<think>\nI should inspect.\n</think>\n\n"
                          "Let me check:\n\n<tool_call>\n<function=read_file>\n<parameter=path>\n"
                          "a\n</parameter>\n</function>\n</tool_call><|im_end|>\n",
                      "assistant reasoning, preamble and tool call did not share one exact turn");

    fi::ChatRenderOptions no_thinking;
    no_thinking.enable_thinking = false;
    failures +=
        check(render_chat_text({chat_message(ninfer::ChatRole::User, "q1"),
                                chat_message(ninfer::ChatRole::Assistant,
                                             "<think>\nold thought\n</think>\n\nold answer"),
                                chat_message(ninfer::ChatRole::User, "q2")},
                               no_thinking) == "<|im_start|>user\nq1<|im_end|>\n"
                                               "<|im_start|>assistant\nold answer<|im_end|>\n"
                                               "<|im_start|>user\nq2<|im_end|>\n"
                                               "<|im_start|>assistant\n<think>\n\n</think>\n\n",
              "thinking history differs from the official template");

    fi::ChatMessage lookup = chat_message(ninfer::ChatRole::Assistant, "");
    lookup.tool_calls.push_back(
        {.id = "", .name = "lookup", .arguments_json = R"({"city":"Paris"})"});
    failures +=
        check(render_chat_text({chat_message(ninfer::ChatRole::User, "weather?"), lookup,
                                chat_message(ninfer::ChatRole::Tool, "sunny"),
                                chat_message(ninfer::ChatRole::Tool, "20C"),
                                chat_message(ninfer::ChatRole::User, "thanks")},
                               no_generation) ==
                  "<|im_start|>user\nweather?<|im_end|>\n"
                  "<|im_start|>assistant\n<tool_call>\n<function=lookup>\n"
                  "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
                  "<|im_start|>user\n<tool_response>\nsunny\n</tool_response>\n"
                  "<tool_response>\n20C\n</tool_response><|im_end|>\n"
                  "<|im_start|>user\nthanks<|im_end|>\n",
              "tool-response grouping differs from the official template");

    fi::ChatRenderOptions tools = no_generation;
    tools.tool_jsons.push_back(
        R"({"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object","properties":{"flag":{"type":"boolean"}}}}})");
    const std::string tools_rendered =
        render_chat_text({chat_message(ninfer::ChatRole::System, "be exact"),
                          chat_message(ninfer::ChatRole::User, "hi")},
                         tools);
    failures += check(
        tools_rendered.find("\n{\"type\": \"function\", \"function\": {\"name\": \"f\", "
                            "\"description\": \"d\", \"parameters\": {\"type\": \"object\", "
                            "\"properties\": {\"flag\": {\"type\": \"boolean\"}}}}}\n</tools>") !=
                std::string::npos &&
            tools_rendered.ends_with(
                "</IMPORTANT>\n\nbe exact<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"),
        "tools system block differs from official tojson rendering");

    failures += check(throws_invalid_argument([&] {
                          (void)render_chat({chat_message(ninfer::ChatRole::System, "only")},
                                            no_generation);
                      }),
                      "message history without a user query was accepted");
    return failures;
}

int test_ordered_instruction_turns() {
    fi::ChatRenderOptions no_generation;
    no_generation.add_generation_prompt = false;

    const std::string leading_developer =
        render_chat_text({chat_message(ninfer::ChatRole::Developer, "policy"),
                          chat_message(ninfer::ChatRole::User, "hi")},
                         no_generation);
    int failures = check(leading_developer == "<|im_start|>system\npolicy<|im_end|>\n"
                                              "<|im_start|>user\nhi<|im_end|>\n",
                         "leading developer did not use the existing Qwen system path");

    const std::string late_system =
        render_chat_text({chat_message(ninfer::ChatRole::User, "hi"),
                          chat_message(ninfer::ChatRole::System, "  current diagnostics  ")},
                         no_generation);
    failures += check(late_system == "<|im_start|>user\nhi<|im_end|>\n"
                                     "<|im_start|>system\ncurrent diagnostics<|im_end|>\n",
                      "late system turn was not rendered at its original position");
    failures += check(
        render_chat_text({chat_message(ninfer::ChatRole::User, "hi"),
                          chat_message(ninfer::ChatRole::Developer, "  current diagnostics  ")},
                         no_generation) == late_system,
        "developer and system did not lower to the same in-place Qwen block");

    const std::string stable_history =
        render_chat_text({chat_message(ninfer::ChatRole::System, "stable policy"),
                          chat_message(ninfer::ChatRole::User, "hi")},
                         no_generation);
    const std::string appended_diagnostics =
        render_chat_text({chat_message(ninfer::ChatRole::System, "stable policy"),
                          chat_message(ninfer::ChatRole::User, "hi"),
                          chat_message(ninfer::ChatRole::System, "current diagnostics")},
                         no_generation);
    failures += check(appended_diagnostics.starts_with(stable_history) &&
                          appended_diagnostics.substr(stable_history.size()) ==
                              "<|im_start|>system\ncurrent diagnostics<|im_end|>\n",
                      "appended diagnostics changed the stable serialized history prefix");
    const std::vector<int> stable_tokens   = official_tokenizer().encode(stable_history);
    const std::vector<int> appended_tokens = official_tokenizer().encode(appended_diagnostics);
    failures +=
        check(appended_tokens.size() > stable_tokens.size() &&
                  std::equal(stable_tokens.begin(), stable_tokens.end(), appended_tokens.begin()),
              "appended diagnostics changed the stable token prefix");

    fi::ChatRenderOptions tools = no_generation;
    tools.tool_jsons.push_back(
        R"({"type":"function","function":{"name":"inspect","parameters":{"type":"object"}}})");
    const std::string tools_with_late_system =
        render_chat_text({chat_message(ninfer::ChatRole::System, "stable policy"),
                          chat_message(ninfer::ChatRole::User, "hi"),
                          chat_message(ninfer::ChatRole::System, "current diagnostics")},
                         tools);
    const std::size_t tools_position  = tools_with_late_system.find("# Tools");
    const std::size_t policy_position = tools_with_late_system.find("stable policy");
    const std::size_t user_position   = tools_with_late_system.find("<|im_start|>user\nhi");
    const std::size_t diagnostics_position =
        tools_with_late_system.find("<|im_start|>system\ncurrent diagnostics");
    failures +=
        check(tools_position != std::string::npos && policy_position != std::string::npos &&
                  user_position != std::string::npos && diagnostics_position != std::string::npos &&
                  tools_with_late_system.find("# Tools", tools_position + 1) == std::string::npos &&
                  tools_position < policy_position && policy_position < user_position &&
                  user_position < diagnostics_position,
              "late system duplicated or moved the leading tools/instruction block");

    const fi::RenderedChat generated =
        render_chat({chat_message(ninfer::ChatRole::User, "hi"),
                     chat_message(ninfer::ChatRole::System, "current diagnostics")});
    const std::string assistant_header = "<|im_start|>assistant\n";
    const std::size_t header           = generated.text.rfind(assistant_header);
    failures += check(header != std::string::npos && generated.rewrite_checkpoint &&
                          generated.rewrite_checkpoint->kind ==
                              ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure &&
                          generated.rewrite_checkpoint->offset == header &&
                          generated.text.find("current diagnostics<|im_end|>\n", 0) < header,
                      "late system was not included before the generation rewrite boundary");

    fi::ChatMessage invalid = chat_message(ninfer::ChatRole::System, "diagnostics");
    invalid.tool_calls.push_back({.id = "call", .name = "f", .arguments_json = "{}"});
    failures += check(throws_invalid_argument([&] {
                          (void)render_chat({chat_message(ninfer::ChatRole::User, "hi"), invalid},
                                            no_generation);
                      }),
                      "system turn carrying assistant tool metadata was accepted");

    fi::ChatMessage media_instruction = chat_message(ninfer::ChatRole::Developer, "diagnostics");
    media_instruction.parts.push_back(fi::ChatPart::image({}));
    failures +=
        check(throws_invalid_argument([&] {
                  (void)render_chat({chat_message(ninfer::ChatRole::User, "hi"), media_instruction},
                                    no_generation);
              }),
              "developer turn carrying media was accepted");

    fi::ChatMessage invalid_role = chat_message(ninfer::ChatRole::User, "bad");
    invalid_role.role            = static_cast<ninfer::ChatRole>(255);
    failures +=
        check(throws_invalid_argument([&] {
                  (void)render_chat({chat_message(ninfer::ChatRole::User, "hi"), invalid_role},
                                    no_generation);
              }),
              "invalid typed chat role was accepted");
    return failures;
}

int test_assistant_continuation() {
    fi::ChatRenderOptions options;
    options.continuation    = ninfer::PromptContinuationMode::ContinueFinalAssistant;
    options.enable_thinking = false;
    const fi::RenderedChat rendered =
        render_chat({chat_message(ninfer::ChatRole::User, "question"),
                     chat_message(ninfer::ChatRole::Assistant, "answer prefix")},
                    options);
    const std::string expected = "<|im_start|>user\nquestion<|im_end|>\n"
                                 "<|im_start|>assistant\nanswer prefix";
    int failures               = check(rendered.text == expected,
                                       "assistant continuation closed the turn or opened a second assistant");
    failures +=
        check(rendered.rewrite_checkpoint &&
                  rendered.rewrite_checkpoint->kind ==
                      ninfer::targets::qwen3_6::RewriteCheckpointKind::ResponseReplay &&
                  rendered.rewrite_checkpoint->offset == expected.find("<|im_start|>assistant"),
              "assistant continuation did not retain its replayable opener boundary");

    options.enable_thinking = true;
    failures += check(throws_invalid_argument([&] {
                          (void)render_chat({chat_message(ninfer::ChatRole::User, "question"),
                                             chat_message(ninfer::ChatRole::Assistant, "prefix")},
                                            options);
                      }),
                      "assistant continuation accepted an ambiguous Thinking opener");
    return failures;
}

int test_reasoning_effort_chat_template() {
    constexpr std::string_view low_instructions =
        "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly "
        "to the conclusion without unnecessary elaboration.";
    constexpr std::string_view xhigh_instructions =
        "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
        "assumptions, consider plausible alternatives, and prioritize correctness, consistency, "
        "and clarity in the final answer.";

    const ninfer::PromptCapabilities toggle_capabilities =
        thinking_toggle_template().capabilities();
    const ninfer::PromptCapabilities effort_capabilities =
        reasoning_effort_template().capabilities();
    int failures = check(toggle_capabilities.enable_thinking &&
                             !toggle_capabilities.reasoning_effort.default_effort &&
                             !toggle_capabilities.reasoning_effort.low &&
                             !toggle_capabilities.reasoning_effort.medium &&
                             !toggle_capabilities.reasoning_effort.xhigh,
                         "thinking-toggle template advertised reasoning effort");
    failures += check(
        effort_capabilities.enable_thinking && effort_capabilities.reasoning_effort.low &&
            effort_capabilities.reasoning_effort.medium &&
            effort_capabilities.reasoning_effort.xhigh &&
            effort_capabilities.reasoning_effort.default_effort == ninfer::ReasoningEffort::XHigh,
        "reasoning-effort template did not advertise its complete capability set");

    const auto render_effort = [](ninfer::ReasoningEffort effort) {
        fi::ChatRenderOptions options;
        options.reasoning_effort = effort;
        return reasoning_effort_template()
            .render({chat_message(ninfer::ChatRole::User, "hello")}, options)
            .text;
    };
    const std::string tail = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n";
    failures += check(
        reasoning_effort_template().render({chat_message(ninfer::ChatRole::User, "hello")}).text ==
            "<|im_start|>system\n" + std::string(xhigh_instructions) + "<|im_end|>\n" + tail,
        "reasoning-effort template did not apply its xhigh default");
    failures +=
        check(render_effort(ninfer::ReasoningEffort::Low) ==
                  "<|im_start|>system\n" + std::string(low_instructions) + "<|im_end|>\n" + tail,
              "low reasoning effort did not render the official instruction");
    failures += check(render_effort(ninfer::ReasoningEffort::Medium) == tail,
                      "medium reasoning effort injected an instruction");

    fi::ChatRenderOptions disabled;
    disabled.enable_thinking = false;
    failures += check(reasoning_effort_template()
                              .render({chat_message(ninfer::ChatRole::System, ""),
                                       chat_message(ninfer::ChatRole::User, "hello")},
                                      disabled)
                              .text == "<|im_start|>user\nhello<|im_end|>\n"
                                       "<|im_start|>assistant\n<think>\n\n</think>\n\n",
                      "disabled thinking did not suppress effort and an empty system turn");
    disabled.reasoning_effort = ninfer::ReasoningEffort::Low;
    failures += check(throws_invalid_argument([&] {
                          (void)reasoning_effort_template().render(
                              {chat_message(ninfer::ChatRole::User, "hello")}, disabled);
                      }),
                      "reasoning effort and disabled thinking were accepted together");

    fi::ChatRenderOptions unsupported;
    unsupported.reasoning_effort = ninfer::ReasoningEffort::Low;
    failures += check(throws_invalid_argument([&] {
                          (void)thinking_toggle_template().render(
                              {chat_message(ninfer::ChatRole::User, "hello")}, unsupported);
                      }),
                      "thinking-toggle template accepted reasoning effort");

    fi::ChatMessage previous   = chat_message(ninfer::ChatRole::Assistant, "old answer");
    previous.reasoning_content = "old thought";
    fi::ChatRenderOptions no_generation;
    no_generation.add_generation_prompt = false;
    no_generation.reasoning_effort      = ninfer::ReasoningEffort::Medium;
    const std::string preserved         = reasoning_effort_template()
                                      .render({chat_message(ninfer::ChatRole::User, "q1"), previous,
                                               chat_message(ninfer::ChatRole::User, "q2")},
                                              no_generation)
                                      .text;
    failures += check(
        preserved.find("<|im_start|>assistant\n<think>\nold thought\n</think>\n\nold answer") !=
            std::string::npos,
        "reasoning-effort template did not preserve prior thinking by default");
    no_generation.preserve_thinking = false;
    failures += check(reasoning_effort_template()
                              .render({chat_message(ninfer::ChatRole::User, "q1"), previous,
                                       chat_message(ninfer::ChatRole::User, "q2")},
                                      no_generation)
                              .text.find("old thought") == std::string::npos,
                      "explicit preserve_thinking=false did not remove prior thinking");

    fi::ChatMessage empty_arguments = chat_message(ninfer::ChatRole::Assistant, "");
    empty_arguments.tool_calls.push_back({.id = "", .name = "f", .arguments_json = ""});
    failures += check(
        reasoning_effort_template()
            .render({chat_message(ninfer::ChatRole::User, "call"), empty_arguments}, no_generation)
            .text.ends_with("<tool_call>\n<function=f>\n</function>\n"
                            "</tool_call><|im_end|>\n"),
        "empty tool arguments did not follow the reasoning-effort template");
    return failures;
}

int test_rewrite_checkpoint_trace() {
    const std::string assistant_header = "<|im_start|>assistant\n";
    fi::ChatMessage first              = chat_message(ninfer::ChatRole::Assistant, "");
    first.reasoning_content            = "first thought";
    first.parts.front().text           = "first answer";
    fi::ChatMessage second             = chat_message(ninfer::ChatRole::Assistant, "");
    second.reasoning_content           = "second thought";
    second.parts.front().text          = "second answer";

    const std::vector<fi::ChatMessage> tool_loop{
        chat_message(ninfer::ChatRole::User, "question"), first,
        chat_message(ninfer::ChatRole::Tool, "result one"), second,
        chat_message(ninfer::ChatRole::Tool, "result two")};
    const fi::RenderedChat open    = render_chat(tool_loop);
    const std::size_t first_header = open.text.find(assistant_header);
    int failures =
        check(first_header != std::string::npos && open.rewrite_checkpoint &&
                  open.rewrite_checkpoint->kind ==
                      ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure &&
                  open.rewrite_checkpoint->offset == first_header,
              "tool loop did not retain the stable prefix before its first assistant turn");

    fi::ChatRenderOptions preserve;
    preserve.preserve_thinking         = true;
    const fi::RenderedChat preserved   = render_chat(tool_loop, preserve);
    const std::size_t preserved_header = preserved.text.rfind(assistant_header);
    failures += check(preserved_header != std::string::npos && preserved.rewrite_checkpoint &&
                          preserved.rewrite_checkpoint->kind ==
                              ninfer::targets::qwen3_6::RewriteCheckpointKind::ResponseReplay &&
                          preserved.rewrite_checkpoint->offset == preserved_header &&
                          preserved.text.ends_with("<think>\n"),
                      "preserve_thinking did not checkpoint before the generation prologue");

    preserve.enable_thinking             = false;
    const fi::RenderedChat nonthinking   = render_chat(tool_loop, preserve);
    const std::size_t nonthinking_header = nonthinking.text.rfind(assistant_header);
    failures += check(nonthinking_header != std::string::npos && nonthinking.rewrite_checkpoint &&
                          nonthinking.rewrite_checkpoint->kind ==
                              ninfer::targets::qwen3_6::RewriteCheckpointKind::ResponseReplay &&
                          nonthinking.rewrite_checkpoint->offset == nonthinking_header &&
                          nonthinking.text.ends_with("<think>\n\n</think>\n\n"),
                      "non-thinking response replay did not checkpoint before its generation "
                      "prologue");

    std::vector<fi::ChatMessage> next_turn = tool_loop;
    next_turn.push_back(chat_message(ninfer::ChatRole::User, "next question"));
    const fi::RenderedChat next    = render_chat(next_turn);
    const std::size_t final_header = next.text.rfind(assistant_header);
    failures += check(final_header != std::string::npos && next.rewrite_checkpoint &&
                          next.rewrite_checkpoint->kind ==
                              ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure &&
                          next.rewrite_checkpoint->offset == final_header,
                      "new user turn did not move the rewrite boundary before its generation "
                      "opener");

    const fi::RenderedChat branch =
        render_chat({chat_message(ninfer::ChatRole::User, "question"),
                     chat_message(ninfer::ChatRole::User, "summarize the conversation")},
                    preserve);
    const fi::RenderedChat source =
        render_chat({chat_message(ninfer::ChatRole::User, "question")}, preserve);
    failures += check(
        source.rewrite_checkpoint &&
            source.rewrite_checkpoint->kind ==
                ninfer::targets::qwen3_6::RewriteCheckpointKind::ResponseReplay &&
            branch.text.starts_with(source.text.substr(0, source.rewrite_checkpoint->offset)) &&
            !branch.text.starts_with(
                source.text.substr(0, source.rewrite_checkpoint->offset + assistant_header.size())),
        "a replacement user suffix lost the stable pre-generation checkpoint");

    fi::ChatRenderOptions no_generation;
    no_generation.add_generation_prompt = false;
    const fi::RenderedChat no_assistant =
        render_chat({chat_message(ninfer::ChatRole::User, "question")}, no_generation);
    failures += check(!no_assistant.rewrite_checkpoint,
                      "boundary-less prompt unexpectedly published a rewrite boundary");

    no_generation.preserve_thinking                     = true;
    const fi::RenderedChat preserved_without_generation = render_chat(tool_loop, no_generation);
    failures += check(!preserved_without_generation.rewrite_checkpoint,
                      "response-replay boundary was published without a generation opener");
    no_generation.preserve_thinking = false;

    const fi::RenderedChat wrapped = render_chat(
        {chat_message(ninfer::ChatRole::User, "question"), first,
         chat_message(ninfer::ChatRole::User, "<tool_response>compat result</tool_response>"),
         second},
        no_generation);
    const std::size_t wrapped_first = wrapped.text.find(assistant_header);
    failures += check(wrapped.rewrite_checkpoint &&
                          wrapped.rewrite_checkpoint->kind ==
                              ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure &&
                          wrapped.rewrite_checkpoint->offset == wrapped_first,
                      "bare tool-response wrapper incorrectly advanced the real user turn");
    return failures;
}

int test_adjacent_tool_message_boundary() {
    fi::ChatMessage assistant = chat_message(ninfer::ChatRole::Assistant, "");
    assistant.tool_calls.push_back(
        {.id = "", .name = "lookup", .arguments_json = R"({"city":"Paris"})"});
    const fi::RenderedChat rendered =
        render_chat({chat_message(ninfer::ChatRole::User, "weather?"), std::move(assistant),
                     chat_message(ninfer::ChatRole::Tool, "sunny"),
                     chat_message(ninfer::ChatRole::Tool, "20C")});
    const fi::EncodedChat encoded = fi::encode_rendered_chat(official_tokenizer(), rendered);
    return check(rendered.message_boundaries.size() == 5 && rendered.message_boundaries[3] &&
                     encoded.message_boundaries.size() == 5 && encoded.message_boundaries[3] &&
                     encoded.message_boundaries[4] &&
                     *encoded.message_boundaries[3] < *encoded.message_boundaries[4],
                 "adjacent Tool messages lost their exact intermediate message boundary");
}

int test_official_resource_guards() {
    FrontendResources stale_pad     = resources();
    nlohmann::json tokenizer_config = nlohmann::json::parse(stale_pad.tokenizer_config_json);
    tokenizer_config["pad_token"]   = "<|vision_pad|>";
    stale_pad.tokenizer_config_json = tokenizer_config.dump();
    int failures =
        check(throws_invalid_argument([&] { (void)FrontendFactory::create_component(stale_pad); }),
              "stale Unsloth pad-token policy was accepted");

    FrontendResources mismatched       = resources();
    nlohmann::json mismatched_config   = nlohmann::json::parse(mismatched.tokenizer_config_json);
    mismatched_config["chat_template"] = reasoning_effort_template_source();
    mismatched.tokenizer_config_json   = mismatched_config.dump();
    failures +=
        check(throws_invalid_argument([&] { (void)FrontendFactory::create_component(mismatched); }),
              "different standalone and tokenizer-config chat templates were accepted");

    FrontendResources unknown = resources("{{ messages }}");
    failures +=
        check(throws_invalid_argument([&] { (void)FrontendFactory::create_component(unknown); }),
              "unknown chat template was accepted");

    const Frontend effort_frontend =
        FrontendFactory::create_component(resources(reasoning_effort_template_source()), false);
    const ninfer::PromptCapabilities capabilities = effort_frontend.prompt_capabilities();
    failures +=
        check(capabilities.reasoning_effort.low && capabilities.reasoning_effort.medium &&
                  capabilities.reasoning_effort.xhigh &&
                  capabilities.reasoning_effort.default_effort == ninfer::ReasoningEffort::XHigh,
              "Frontend did not expose capabilities from its loaded chat template");

    return failures;
}

int test_text_and_image_prepare(const Frontend& frontend) {
    ninfer::ChatMessage text_message;
    text_message.role = ninfer::ChatRole::User;
    text_message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::PromptInput text_input;
    text_input.messages.push_back(std::move(text_message));
    auto text             = frontend.prepare(std::move(text_input));
    const auto& text_data = FrontendFactory::inspect(text);
    const std::vector<ninfer::TokenId> expected{248045, 30, 0, 248046, 32, 248045, 31, 248068, 32};
    int failures =
        check(text_data.token_ids == expected, "text frontend did not render/tokenize chat");
    failures += check(text_data.identity.rewrite_checkpoint &&
                          text_data.identity.rewrite_checkpoint->kind ==
                              ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure &&
                          text_data.identity.rewrite_checkpoint->frontier == 5 &&
                          text_data.starts_in_reasoning && !text_data.has_media(),
                      "text frontend did not preserve prefix/thinking identity");
    failures +=
        check(text_data.position_axis(0).back() == 8 && text_data.position_axis(1).back() == 8 &&
                  text_data.position_axis(2).back() == 8,
              "text frontend did not construct axis-major positions");

    ninfer::ChatMessage preserved_message;
    preserved_message.role = ninfer::ChatRole::User;
    preserved_message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::PromptInput preserved_input;
    preserved_input.messages.push_back(std::move(preserved_message));
    preserved_input.options.preserve_thinking = true;
    const auto preserved_prompt               = frontend.prepare(std::move(preserved_input));
    const auto& preserved_data                = FrontendFactory::inspect(preserved_prompt);
    failures += check(preserved_data.identity.rewrite_checkpoint &&
                          preserved_data.identity.rewrite_checkpoint->kind ==
                              ninfer::targets::qwen3_6::RewriteCheckpointKind::ResponseReplay &&
                          preserved_data.identity.rewrite_checkpoint->frontier == 5 &&
                          preserved_data.identity.rewrite_checkpoint->frontier <
                              preserved_data.token_ids.size(),
                      "preserve-thinking prompt did not publish a pre-generation response "
                      "checkpoint");

    ninfer::ChatMessage nonthinking_message;
    nonthinking_message.role = ninfer::ChatRole::User;
    nonthinking_message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::PromptInput nonthinking_input;
    nonthinking_input.messages.push_back(std::move(nonthinking_message));
    nonthinking_input.options.preserve_thinking = true;
    nonthinking_input.options.enable_thinking   = false;
    const auto nonthinking_prompt               = frontend.prepare(std::move(nonthinking_input));
    const auto& nonthinking_data                = FrontendFactory::inspect(nonthinking_prompt);
    failures += check(nonthinking_data.identity.rewrite_checkpoint &&
                          nonthinking_data.identity.rewrite_checkpoint->kind ==
                              ninfer::targets::qwen3_6::RewriteCheckpointKind::ResponseReplay &&
                          nonthinking_data.identity.rewrite_checkpoint->frontier == 5 &&
                          nonthinking_data.identity.rewrite_checkpoint->frontier <
                              nonthinking_data.token_ids.size() &&
                          !nonthinking_data.starts_in_reasoning,
                      "non-thinking prompt did not publish a pre-generation response checkpoint");

    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = gradient_ppm();
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "inline.ppm";
    ninfer::ChatMessage image_message;
    image_message.role = ninfer::ChatRole::User;
    image_message.parts.push_back(std::move(image));
    ninfer::PromptInput image_input;
    image_input.messages.push_back(std::move(image_message));
    image_input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
        .after_message_count      = 1,
        .kind                     = ninfer::PromptCacheMarkerKind::PrivateLongAnchor,
        .location                 = ninfer::PromptCacheMarkerLocation::MessagePartBoundary,
        .after_message_part_count = 1,
    });
    auto prepared             = frontend.prepare(std::move(image_input));
    const auto& prepared_data = FrontendFactory::inspect(prepared);
    failures += check(prepared_data.has_media() && prepared_data.vision_items.size() == 1,
                      "image frontend did not retain one Vision item");
    if (!prepared_data.vision_items.empty()) {
        const auto& item = prepared_data.vision_items.front();
        failures +=
            check(item.grid.temporal == 1 && item.grid.height == 4 && item.grid.width == 4 &&
                      item.patch_count == 16 && item.content_digest == kGradientDigest &&
                      item.token_spans.size() == 1 && item.token_spans.front().count == 4,
                  "image frontend grid/patch/placeholder geometry is incorrect");
        if (!item.token_spans.empty()) {
            const std::size_t span = item.token_spans.front().begin;
            failures += check(
                prepared_data.position_axis(0)[span] == prepared_data.position_axis(1)[span] &&
                    prepared_data.position_axis(1)[span] == prepared_data.position_axis(2)[span] &&
                    prepared_data.position_axis(1)[span + 2] ==
                        prepared_data.position_axis(1)[span] + 1 &&
                    prepared_data.position_axis(2)[span + 1] ==
                        prepared_data.position_axis(2)[span] + 1,
                "image frontend MRoPE positions are incorrect");
        }
    }
    const std::span<const std::uint16_t> image_patches =
        prepared_data.media_payloads.size() == 1 && prepared_data.media_payloads.front()
            ? prepared_data.media_payloads.front()->span()
            : std::span<const std::uint16_t>{};
    failures += check(
        image_patches.size() == 16 * 1536 && prepared_data.prepare.raw_patches == 16 &&
            prepared_data.prepare.vision_tokens == 4 && prepared_data.identity.reusable &&
            prepared_data.identity.rewrite_checkpoint &&
            prepared_data.identity.rewrite_checkpoint->kind ==
                ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure &&
            prepared_data.identity.rewrite_checkpoint->frontier < prepared_data.token_ids.size(),
        "image frontend did not own the expected patch payload and identity");
    if (!prepared_data.vision_items.empty() &&
        !prepared_data.vision_items.front().token_spans.empty()) {
        const auto span            = prepared_data.vision_items.front().token_spans.front();
        const auto explicit_marker = std::find_if(
            prepared_data.context_cache.opportunities.begin(),
            prepared_data.context_cache.opportunities.end(), [](const auto& opportunity) {
                return ninfer::has_shared_candidate_evidence(
                    opportunity.evidence, ninfer::SharedCandidateEvidence::ExplicitBoundary);
            });
        failures += check(explicit_marker != prepared_data.context_cache.opportunities.end() &&
                              explicit_marker->frontier >= span.begin + span.count &&
                              explicit_marker->frontier < prepared_data.token_ids.size(),
                          "media expansion did not remap the following message cache boundary");
    }
    if (image_patches.size() == 16 * 1536) {
        failures += check(image_patches[0] == bf16_bits(-1.0F) &&
                              image_patches[1] == bf16_bits(1.0F / 127.5F - 1.0F) &&
                              image_patches[256] == bf16_bits(-1.0F) &&
                              image_patches[1536] == bf16_bits(16.0F / 127.5F - 1.0F),
                          "image frontend patch normalization/order is incorrect");
    }
    return failures;
}

int test_literal_control_tokens_with_media() {
    fi::ChatRenderOptions no_generation;
    no_generation.add_generation_prompt = false;
    const fi::RenderedChat literal_rendered =
        render_chat({chat_message(ninfer::ChatRole::User, "quoted <|image_pad|>")}, no_generation);
    const std::vector<int> literal_tokens =
        fi::encode_rendered_chat(official_tokenizer(), literal_rendered).input_ids;
    int failures = check(
        literal_rendered.text == "<|im_start|>user\nquoted <|image_pad|><|im_end|>\n" &&
            literal_rendered.text.find("\xE2\x81\xA0") == std::string::npos &&
            std::find(literal_tokens.begin(), literal_tokens.end(), 248056) == literal_tokens.end(),
        "renderer changed or structurally tokenized a literal Vision marker");

    fi::ChatMessage leading_tool;
    leading_tool.role = ninfer::ChatRole::Tool;
    leading_tool.parts.push_back(
        fi::ChatPart{.kind = fi::ChatPartKind::Text, .text = "imported result"});
    const fi::RenderedChat leading_tool_rendered = render_chat({leading_tool}, no_generation);
    failures += check(
        leading_tool_rendered.text ==
            "<|im_start|>user\n<tool_response>\nimported result\n</tool_response><|im_end|>\n",
        "leading tool result was rendered without its user-role envelope");

    FrontendResources official = resources();
    official.tokenizer_json          = read_official_resource("tokenizer.json");
    official.tokenizer_config_json   = read_official_resource("tokenizer_config.json");
    official.generation_config_json  = read_official_resource("generation_config.json");
    const Frontend frontend = FrontendFactory::create_component(official);

    auto text_part = [](std::string text) {
        return ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}};
    };
    auto image_part = [](std::vector<std::uint8_t> bytes, std::string source_name) {
        ninfer::MessagePart image;
        image.kind              = ninfer::MessagePartKind::Media;
        image.media.kind        = ninfer::MediaKind::Image;
        image.media.bytes       = std::move(bytes);
        image.media.media_type  = "image/x-portable-pixmap";
        image.media.source_name = std::move(source_name);
        return image;
    };

    std::vector<std::uint8_t> result_b_bytes  = gradient_ppm();
    std::vector<std::uint8_t> result_a1_bytes = result_b_bytes;
    std::vector<std::uint8_t> result_a2_bytes = result_b_bytes;
    result_a1_bytes.back() ^= 0x01U;
    result_a2_bytes.back() ^= 0x02U;
    const fi::Sha256Digest result_b_digest =
        fi::sha256(std::span<const std::uint8_t>(result_b_bytes));
    const fi::Sha256Digest result_a1_digest =
        fi::sha256(std::span<const std::uint8_t>(result_a1_bytes));
    const fi::Sha256Digest result_a2_digest =
        fi::sha256(std::span<const std::uint8_t>(result_a2_bytes));

    ninfer::ChatMessage system;
    system.role = ninfer::ChatRole::System;
    system.parts.push_back(
        text_part("The quoted template contains <|video_pad|>, <|vision_start|>, "
                  "<|image_pad|>, and <|vision_end|>."));

    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(text_part("inspect both files"));

    ninfer::ChatMessage assistant;
    assistant.role              = ninfer::ChatRole::Assistant;
    assistant.reasoning_content = "quoted reasoning <|video_pad|>";
    assistant.tool_calls.push_back(ninfer::ToolCall{
        .id             = "call_A",
        .name           = "read",
        .arguments_json = R"({"path":"quoted <|image_pad|>.png"})",
    });
    assistant.tool_calls.push_back(
        ninfer::ToolCall{.id = "call_B", .name = "read", .arguments_json = R"({"path":"b.png"})"});

    ninfer::ChatMessage result_b;
    result_b.role         = ninfer::ChatRole::Tool;
    result_b.tool_call_id = "call_B";
    result_b.parts.push_back(text_part("result B: literal <|image_"));
    result_b.parts.push_back(text_part("pad|> then image "));
    result_b.parts.push_back(image_part(std::move(result_b_bytes), "result-b.ppm"));

    ninfer::ChatMessage result_a;
    result_a.role         = ninfer::ChatRole::Tool;
    result_a.tool_call_id = "call_A";
    result_a.parts.push_back(text_part("result A first image "));
    result_a.parts.push_back(image_part(std::move(result_a1_bytes), "result-a1.ppm"));
    result_a.parts.push_back(text_part(" literal <|vision_start|> between images "));
    result_a.parts.push_back(image_part(std::move(result_a2_bytes), "result-a2.ppm"));

    ninfer::PromptInput input;
    input.messages.push_back(std::move(system));
    input.messages.push_back(std::move(user));
    input.messages.push_back(std::move(assistant));
    input.messages.push_back(std::move(result_b));
    input.messages.push_back(std::move(result_a));
    input.options.tool_jsons.push_back(
        R"({"type":"function","function":{"name":"read","description":"quoted <|vision_start|><|image_pad|><|vision_end|> and <|video_pad|>","parameters":{"type":"object"}}})");
    input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
        .after_message_count = static_cast<std::uint32_t>(input.messages.size()),
        .kind                = ninfer::PromptCacheMarkerKind::PrivateLongAnchor,
    });

    const std::uint32_t counted = frontend.count_tokens(input);
    const auto prepared         = frontend.prepare(std::move(input));
    const auto& data            = FrontendFactory::inspect(prepared);
    failures += check(data.token_ids.size() == counted,
                      "literal controls changed token-counting semantics");
    failures += check(data.vision_items.size() == 3 && data.media_payloads.size() == 3,
                      "literal controls changed the typed media count");
    const auto private_anchor = std::find_if(
        data.context_cache.opportunities.begin(), data.context_cache.opportunities.end(),
        [](const auto& opportunity) {
            return opportunity.kind == ninfer::PromptCacheMarkerKind::PrivateLongAnchor;
        });
    failures += check(private_anchor != data.context_cache.opportunities.end(),
                      "literal controls lost the following cache boundary");
    if (data.vision_items.size() == 3) {
        const auto& b  = data.vision_items[0];
        const auto& a1 = data.vision_items[1];
        const auto& a2 = data.vision_items[2];
        failures += check(
            b.content_digest == result_b_digest && a1.content_digest == result_a1_digest &&
                a2.content_digest == result_a2_digest && b.token_spans.size() == 1 &&
                a1.token_spans.size() == 1 && a2.token_spans.size() == 1 &&
                b.token_spans[0].count == 4 && a1.token_spans[0].count == 4 &&
                a2.token_spans[0].count == 4 && b.token_spans[0].begin < a1.token_spans[0].begin &&
                a1.token_spans[0].begin < a2.token_spans[0].begin,
            "parallel tool-result media lost request or nested-content order");
        if (private_anchor != data.context_cache.opportunities.end()) {
            failures +=
                check(private_anchor->frontier >= a2.token_spans[0].begin + a2.token_spans[0].count,
                      "media provenance broke the following cache boundary");
        }
    }
    failures += check(std::count(data.token_ids.begin(), data.token_ids.end(), 248056) == 12 &&
                          std::count(data.token_ids.begin(), data.token_ids.end(), 248057) == 0 &&
                          std::count(data.token_ids.begin(), data.token_ids.end(), 248053) == 3 &&
                          std::count(data.token_ids.begin(), data.token_ids.end(), 248054) == 3,
                      "literal Vision spellings became media tokens");
    return failures;
}

int test_image_resize_rejection_policy() {
    FrontendResources owned = resources();
    owned.preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":1048576}})";
    const Frontend frontend = FrontendFactory::create_component(owned);

    ninfer::PromptInput small = image_input();
    small.messages[0].parts[0].media.image_resize_policy =
        ninfer::ImageResizePolicy::RejectOversized;
    int failures = check(frontend.count_tokens(std::move(small)) != 0,
                         "oversized_image=error rejected an image that needed no downsize");

    ninfer::PromptInput oversized =
        image_text_input(block_ppm(2048, 1024, 127), {}, "oversized.ppm");
    oversized.messages[0].parts[0].media.image_resize_policy =
        ninfer::ImageResizePolicy::RejectOversized;
    try {
        (void)frontend.count_tokens(std::move(oversized));
        failures += check(false, "oversized_image=error allowed a required Vision downsize");
    } catch (const ninfer::RequestError& error) {
        failures += check(error.kind() == ninfer::RequestErrorKind::InvalidMedia,
                          "oversized_image=error used the wrong request-error classification");
    }
    return failures;
}

int test_explicit_leading_instruction_cache_boundary() {
    FrontendResources official = resources();
    official.tokenizer_json          = read_official_resource("tokenizer.json");
    official.tokenizer_config_json   = read_official_resource("tokenizer_config.json");
    official.generation_config_json  = read_official_resource("generation_config.json");
    const Frontend frontend           = FrontendFactory::create_component(official, false);
    constexpr std::string_view stable = "stable cache section.";
    ninfer::ChatMessage system;
    system.role = ninfer::ChatRole::System;
    system.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::string(stable), .media = {}});
    system.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "\ndynamic working directory", .media = {}});
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "question", .media = {}});

    ninfer::PromptInput input;
    input.messages.push_back(std::move(system));
    input.messages.push_back(std::move(user));
    input.options.tool_jsons.push_back(
        R"({"type":"function","function":{"name":"inspect","parameters":{"type":"object"}}})");
    input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
        .kind                      = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .location                  = ninfer::PromptCacheMarkerLocation::LeadingInstructionBoundary,
        .leading_instruction_bytes = static_cast<std::uint32_t>(stable.size()),
    });

    const auto prepared        = frontend.prepare(std::move(input));
    const auto& data           = FrontendFactory::inspect(prepared);
    const auto explicit_marker = std::find_if(
        data.context_cache.opportunities.begin(), data.context_cache.opportunities.end(),
        [](const auto& opportunity) {
            return ninfer::has_shared_candidate_evidence(
                opportunity.evidence, ninfer::SharedCandidateEvidence::ExplicitBoundary);
        });
    return check(explicit_marker != data.context_cache.opportunities.end() &&
                     explicit_marker->kind == ninfer::PromptCacheMarkerKind::SharedStablePrefix &&
                     explicit_marker->frontier != 0 &&
                     explicit_marker->frontier < data.token_ids.size(),
                 "explicit leading-system cache boundary was lost or shadowed by the automatic "
                 "full-system marker");
}

int test_media_admission_uses_aggregate_resources(const Frontend& frontend) {
    constexpr std::size_t kMediaItems     = 17;
    const std::vector<std::uint8_t> bytes = gradient_ppm();
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    for (std::size_t index = 0; index < kMediaItems; ++index) {
        ninfer::OwnedMedia media;
        media.kind        = ninfer::MediaKind::Image;
        media.bytes       = bytes;
        media.media_type  = "image/x-portable-pixmap";
        media.source_name = "aggregate-" + std::to_string(index) + ".ppm";
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Media, .text = {}, .media = std::move(media)});
    }
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    const auto prepared = frontend.prepare(std::move(input));
    const auto& data    = FrontendFactory::inspect(prepared);
    int failures        = check(data.prepare.media_items == kMediaItems &&
                                    data.prepare.media_bytes == kMediaItems * bytes.size() &&
                                    data.prepare.raw_patches == kMediaItems * 16 &&
                                    data.prepare.vision_tokens == kMediaItems * 4 &&
                                    data.vision_items.size() == kMediaItems,
                                "frontend retained an item-count admission limit");

    fi::ProcessorOptions options;
    options.max_encoded_media_bytes = bytes.size() * 2 - 1;
    auto cache = std::make_shared<fi::MediaPreprocessCache>(ninfer::kDefaultMediaCacheBytes,
                                                            ninfer::kDefaultMediaLiveBytes);
    fi::Processor processor(official_tokenizer(), thinking_toggle_template(), options,
                            std::move(cache));
    fi::ChatMessage internal_message;
    internal_message.role = ninfer::ChatRole::User;
    for (std::size_t index = 0; index < 2; ++index) {
        internal_message.parts.push_back(
            fi::ChatPart::image(fi::MediaData{.bytes       = bytes,
                                              .media_type  = "image/x-portable-pixmap",
                                              .source_name = "byte-budget.ppm"}));
    }
    failures += check(throws_processor_budget([&] {
                          (void)processor.process(std::vector<fi::ChatMessage>{internal_message});
                      }),
                      "processor did not enforce the aggregate encoded-media byte budget");
    return failures;
}

int test_multimodal_prompt_over_removed_32k_cap(const Frontend& frontend) {
    const std::string long_text(40'000, 'x');
    const ninfer::MediaCacheSummary before_count = frontend.media_cache_summary();
    const std::uint32_t counted =
        frontend.count_tokens(image_text_input(gradient_ppm(), long_text, "long-context.ppm"));
    const ninfer::MediaCacheSummary after_count = frontend.media_cache_summary();
    const auto prepared =
        frontend.prepare(image_text_input(gradient_ppm(), long_text, "long-context.ppm"));
    const auto& data = FrontendFactory::inspect(prepared);

    int failures = check(counted > 32'768 && data.token_ids.size() == counted,
                         "multimodal prompt retained the removed 32K frontend token cap");
    failures += check(after_count.entries == before_count.entries &&
                          after_count.live_bytes == before_count.live_bytes &&
                          after_count.hits == before_count.hits &&
                          after_count.misses == before_count.misses,
                      "multimodal token counting mutated the prepared-media cache");
    failures += check(data.has_media() && data.vision_items.size() == 1,
                      "long multimodal prompt lost its Vision item");
    return failures;
}

int test_attention_pairs_are_diagnostic(const Frontend& frontend) {
    constexpr std::uint64_t kRemovedAttentionPairLimit = 128ULL * 1024ULL * 1024ULL;
    const auto prepared =
        frontend.prepare(image_text_input(block_ppm(2048, 1536, 127), {}, "large-grid.ppm"));
    const auto& data = FrontendFactory::inspect(prepared);

    int failures = check(data.prepare.attention_pairs > kRemovedAttentionPairLimit,
                         "test image did not exceed the removed attention-pair threshold");
    failures += check(data.prepare.raw_patches == 12'288 && data.prepare.vision_tokens == 3'072 &&
                          data.vision_items.size() == 1,
                      "large image did not retain its expected Vision geometry");
    return failures;
}

int test_video_prepare(const Frontend& frontend) {
    ninfer::MessagePart video;
    video.kind              = ninfer::MessagePartKind::Media;
    video.media.kind        = ninfer::MediaKind::Video;
    video.media.bytes       = gradient_ppm();
    video.media.media_type  = "image/x-portable-pixmap";
    video.media.source_name = "single-frame.ppm";
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(std::move(video));
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));

    const ninfer::MediaCacheSummary before_count = frontend.media_cache_summary();
    const std::uint32_t counted                  = frontend.count_tokens(input);
    const ninfer::MediaCacheSummary after_count  = frontend.media_cache_summary();
    auto prepared                                = frontend.prepare(std::move(input));
    const auto& prepared_data                    = FrontendFactory::inspect(prepared);
    int failures = check(prepared_data.vision_items.size() == 1 && prepared_data.has_media() &&
                             prepared_data.token_ids.size() == counted,
                         "video token counting and preparation geometry diverged");
    failures += check(after_count.entries == before_count.entries &&
                          after_count.live_bytes == before_count.live_bytes &&
                          after_count.hits == before_count.hits &&
                          after_count.misses == before_count.misses,
                      "video token counting mutated the prepared-media cache");
    if (!prepared_data.vision_items.empty()) {
        const auto& item = prepared_data.vision_items.front();
        failures +=
            check(item.modality == ninfer::targets::qwen3_6::PromptModality::Video &&
                      item.grid.temporal == 1 && item.grid.height == 4 && item.grid.width == 4 &&
                      item.patch_count == 16 && item.content_digest == kGradientDigest &&
                      item.timestamps.size() == 1 && item.timestamps.front() == 0.0 &&
                      item.token_spans.size() == 1 && item.token_spans.front().count == 4,
                  "video frontend temporal/grid/placeholder metadata is incorrect");
    }
    const std::span<const std::uint16_t> video_patches =
        prepared_data.media_payloads.size() == 1 && prepared_data.media_payloads.front()
            ? prepared_data.media_payloads.front()->span()
            : std::span<const std::uint16_t>{};
    failures += check(
        video_patches.size() == 16 * 1536 && video_patches[0] == video_patches[256] &&
            prepared_data.prepare.raw_patches == 16 && prepared_data.prepare.vision_tokens == 4 &&
            prepared_data.prepare.media_cache_misses == 1 &&
            prepared_data.prepare.media_cache_hits == 0 && prepared_data.identity.reusable,
        "video frontend did not duplicate the odd temporal frame correctly");
    return failures;
}

int test_cross_round_stop(const Frontend& frontend) {
    auto prompt = frontend.prepare_tokens({0});
    ninfer::StopPolicy stop;
    stop.strings.push_back(ninfer::StopString{.text = "STOP"});
    auto session = frontend.make_output_session(prompt, stop);

    const auto first_decision = session.preview_model(std::array<ninfer::TokenId, 1>{1}, 2,
                                                      ninfer::FinishReason::OutputLimit);
    int failures     = check(first_decision.accepted_tokens == 1 && !first_decision.finished(),
                             "cross-round stop ended before the stop string was complete");
    const auto first = session.commit_preview();
    failures += check(channel_text(first, ninfer::OutputChannel::Content) == "hello",
                      "cross-round stop did not retain the ambiguous suffix");

    const auto second_decision = session.preview_model(std::array<ninfer::TokenId, 1>{2}, 1,
                                                       ninfer::FinishReason::OutputLimit);
    failures += check(second_decision.accepted_tokens == 1 &&
                          second_decision.finish_reason == ninfer::FinishReason::StopString,
                      "cross-round stop did not select the exact terminal token prefix");
    const auto second = session.commit_preview();
    failures += check(second.empty(), "stop marker or same-token suffix leaked to output");
    failures += check(session.matched_stop_string() == std::optional<std::string>("STOP"),
                      "terminal output session lost the matched stop declaration");
    return failures;
}

int test_same_token_stop_priority(const Frontend& frontend) {
    auto prompt = frontend.prepare_tokens({0});
    ninfer::StopPolicy stop;
    stop.strings = {
        ninfer::StopString{.text = "tail", .include_in_output = true},
        ninfer::StopString{.text = "OPtail"},
        ninfer::StopString{.text = "OP", .include_in_output = true},
    };
    auto session        = frontend.make_output_session(prompt, stop);
    const auto decision = session.preview_model(std::array<ninfer::TokenId, 1>{2}, 2,
                                                ninfer::FinishReason::OutputLimit);
    int failures        = check(decision.accepted_tokens == 1 &&
                                    decision.finish_reason == ninfer::FinishReason::StopString,
                                "same-token stop strings did not select a terminal prefix");
    const auto output   = session.commit_preview();
    failures += check(output.empty(),
                      "same-token stops did not prefer the earliest byte and declaration order");
    return failures;
}

int test_terminal_flush(const Frontend& frontend) {
    auto prompt = frontend.prepare_tokens({0});
    ninfer::StopPolicy stop;
    stop.strings.push_back(ninfer::StopString{.text = "STOP"});
    auto session = frontend.make_output_session(prompt, stop);

    const auto first_decision = session.preview_model(std::array<ninfer::TokenId, 1>{1}, 2,
                                                      ninfer::FinishReason::OutputLimit);
    int failures     = check(first_decision.accepted_tokens == 1 && !first_decision.finished(),
                             "terminal flush setup unexpectedly finished");
    const auto first = session.commit_preview();
    failures += check(channel_text(first, ninfer::OutputChannel::Content) == "hello",
                      "terminal flush setup did not retain the possible stop suffix");

    const auto terminal = session.preview_terminal(ninfer::FinishReason::Cancelled);
    failures += check(terminal.accepted_tokens == 0 &&
                          terminal.finish_reason == ninfer::FinishReason::Cancelled,
                      "between-round terminal preview returned the wrong decision");
    const auto flushed = session.commit_preview();
    failures += check(channel_text(flushed, ninfer::OutputChannel::Content) == "ST",
                      "between-round terminal preview lost the pending stop suffix");
    return failures;
}

int test_structured_tool_output() {
    FrontendResources owned = resources();
    owned.tokenizer_json          = read_official_resource("tokenizer.json");
    owned.tokenizer_config_json   = read_official_resource("tokenizer_config.json");
    owned.generation_config_json  = read_official_resource("generation_config.json");
    const Frontend frontend = FrontendFactory::create_component(owned);

    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    input.options.tool_jsons.push_back(
        R"({"type":"function","function":{"name":"TaskUpdate","parameters":{"type":"object","properties":{"taskId":{"type":"string"}}}}})");
    auto prompt = frontend.prepare(std::move(input));
    auto session =
        frontend.make_output_session(prompt, {}, ninfer::OutputOptions{.tool_name_max_length = 64});

    const std::string generated =
        "Calling.  \n<tool_call>\n<function=TaskUpdate>\n<parameter=taskId>\n1\n"
        "</parameter>\n</function>\n</tool_call>";
    const std::vector<ninfer::TokenId> tokens = official_tokenizer().encode(generated);
    const auto decision = session.preview_model(tokens, static_cast<std::uint32_t>(tokens.size()),
                                                ninfer::FinishReason::OutputLimit);
    int failures        = check(decision.finish_reason == ninfer::FinishReason::OutputLimit,
                                "tool output did not reach the terminal transaction");
    const auto output   = session.commit_preview();
    failures += check(channel_text(output, ninfer::OutputChannel::Content) == "Calling.",
                      "frontend did not hide the terminal tool-call suffix");
    const std::vector<ninfer::GeneratedToolCall> calls = session.take_tool_calls();
    failures += check(calls.size() == 1 && calls.front().name == "TaskUpdate",
                      "frontend did not publish the structured tool call");
    if (!calls.empty()) {
        const nlohmann::json arguments = nlohmann::json::parse(calls.front().arguments_json);
        failures += check(arguments.at("taskId").is_string() && arguments.at("taskId") == "1",
                          "frontend changed the declared string argument type");
    }
    return failures;
}

int test_reasoning_split(const Frontend& frontend) {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.continuation    = ninfer::PromptContinuationMode::NewAssistantTurn;
    input.options.enable_thinking = true;
    auto prompt                   = frontend.prepare(std::move(input));
    auto session                  = frontend.make_output_session(prompt, {});
    const std::array<ninfer::TokenId, 2> tokens{3, 4};
    const auto decision = session.preview_model(tokens, 2, ninfer::FinishReason::OutputLimit);
    int failures        = check(decision.accepted_tokens == 2 &&
                                    decision.finish_reason == ninfer::FinishReason::OutputLimit,
                                "reasoning output did not finish at the requested token limit");
    const auto output   = session.commit_preview();
    failures += check(channel_text(output, ninfer::OutputChannel::Reasoning) == "thought",
                      "reasoning channel did not remove the close marker");
    failures += check(channel_text(output, ninfer::OutputChannel::Content) == "answer",
                      "content channel did not strip the post-thinking separator");
    failures += check(session.reasoning_tokens() == 2,
                      "reasoning token usage did not count accepted reasoning tokens exactly");
    return failures;
}

ninfer::targets::qwen3_6::PreparedPrompt thinking_prompt(const Frontend& frontend) {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.continuation    = ninfer::PromptContinuationMode::NewAssistantTurn;
    input.options.enable_thinking = true;
    return frontend.prepare(std::move(input));
}

int test_thinking_budget_control(const Frontend& frontend) {
    auto prompt = thinking_prompt(frontend);
    ninfer::StopPolicy stop;
    stop.strings.push_back(
        ninfer::StopString{.text = "limited", .channel = ninfer::OutputChannel::Reasoning});
    auto session =
        frontend.make_output_session(prompt, stop, {}, ninfer::ThinkingControlOptions{.budget = 2});
    int failures = check(session.model_token_budget_remaining(20) == 2,
                         "thinking budget did not clamp the model round license");

    const std::array<ninfer::TokenId, 2> model_tokens{0, 0};
    const auto boundary =
        session.preview_model(model_tokens, 20, ninfer::FinishReason::OutputLimit);
    failures +=
        check(boundary.accepted_tokens == model_tokens.size() && !boundary.finished() &&
                  boundary.continuation == ninfer::runtime::ContinuationAction::ApplyTargetControl,
              "thinking boundary did not request target control");
    const auto model_output = session.commit_preview();
    failures += check(channel_text(model_output, ninfer::OutputChannel::Reasoning) == "xx",
                      "model-origin thinking output was not published before control");

    const std::span<const ninfer::TokenId> pending = session.pending_control_tokens();
    failures += check(pending.size() > 1,
                      "thinking boundary did not expose a multi-token canonical control span");
    const std::vector<ninfer::TokenId> control(pending.begin(), pending.end());
    failures += check(
        throws_invalid_argument([&] {
            session.validate_generation_capacity(static_cast<std::uint32_t>(2 + control.size()));
        }),
        "planning accepted a capacity that cannot fit control plus a post-close model token");
    session.validate_generation_capacity(static_cast<std::uint32_t>(3 + control.size()));

    const auto control_decision = session.preview_control(control, 18);
    failures +=
        check(control_decision.accepted_tokens == control.size() && !control_decision.finished(),
              "canonical thinking control was not accepted atomically");
    const auto control_output = session.commit_preview();
    failures += check(channel_text(control_output, ninfer::OutputChannel::Reasoning) ==
                              kThinkingControlGuidance &&
                          channel_text(control_output, ninfer::OutputChannel::Content).empty(),
                      "thinking control was truncated by caller stops or published to content");
    const ninfer::ThinkingBudgetStats stats = session.thinking_stats();
    failures += check(stats.configured_budget == 2 && stats.model_thinking_tokens == 2 &&
                          stats.injected_tokens == control.size() && stats.applied &&
                          session.pending_control_tokens().empty() &&
                          session.model_token_budget_remaining(17) == 17,
                      "thinking control accounting or post-close license is incorrect");

    const auto content_decision = session.preview_model(std::array<ninfer::TokenId, 1>{0}, 17,
                                                        ninfer::FinishReason::OutputLimit);
    failures +=
        check(!content_decision.finished(), "post-control model token unexpectedly terminated");
    const auto content_output = session.commit_preview();
    failures += check(channel_text(content_output, ninfer::OutputChannel::Content) == "x",
                      "post-control model output did not enter the content channel");

    auto natural_prompt  = thinking_prompt(frontend);
    auto natural_session = frontend.make_output_session(
        natural_prompt, {}, {}, ninfer::ThinkingControlOptions{.budget = 2});
    const auto natural = natural_session.preview_model(std::array<ninfer::TokenId, 2>{3, 4}, 20,
                                                       ninfer::FinishReason::OutputLimit);
    failures += check(!natural.finished() &&
                          natural.continuation == ninfer::runtime::ContinuationAction::Decode,
                      "natural thinking close at the budget boundary requested control");
    (void)natural_session.commit_preview();
    failures += check(natural_session.pending_control_tokens().empty() &&
                          !natural_session.thinking_stats().applied,
                      "natural thinking close left control armed");

    auto terminal_prompt  = thinking_prompt(frontend);
    auto terminal_session = frontend.make_output_session(
        terminal_prompt, {}, {}, ninfer::ThinkingControlOptions{.budget = 1});
    const auto terminal = terminal_session.preview_model(std::array<ninfer::TokenId, 1>{6}, 10,
                                                         ninfer::FinishReason::OutputLimit);
    failures += check(terminal.finish_reason == ninfer::FinishReason::StopToken &&
                          terminal.continuation == ninfer::runtime::ContinuationAction::Decode,
                      "terminal token at the thinking boundary did not take priority");
    (void)terminal_session.commit_preview();
    failures += check(terminal_session.pending_control_tokens().empty(),
                      "terminal thinking boundary left control pending");

    auto limit_prompt  = thinking_prompt(frontend);
    auto limit_session = frontend.make_output_session(limit_prompt, {}, {},
                                                      ninfer::ThinkingControlOptions{.budget = 1});
    const auto limited = limit_session.preview_model(std::array<ninfer::TokenId, 1>{0}, 1,
                                                     ninfer::FinishReason::ContextCapacity);
    failures += check(limited.finish_reason == ninfer::FinishReason::ContextCapacity &&
                          limited.continuation == ninfer::runtime::ContinuationAction::Decode,
                      "total capacity did not take priority at the thinking boundary");
    (void)limit_session.commit_preview();

    auto raw_prompt = thinking_prompt(frontend);
    auto raw_session =
        frontend.make_output_session(raw_prompt, {}, ninfer::OutputOptions{.raw = true},
                                     ninfer::ThinkingControlOptions{.budget = 1});
    const auto raw_boundary = raw_session.preview_model(std::array<ninfer::TokenId, 1>{0}, 10,
                                                        ninfer::FinishReason::OutputLimit);
    failures +=
        check(raw_boundary.continuation == ninfer::runtime::ContinuationAction::ApplyTargetControl,
              "raw presentation disabled semantic thinking control");
    (void)raw_session.commit_preview();
    const std::vector<ninfer::TokenId> raw_tokens(raw_session.pending_control_tokens().begin(),
                                                  raw_session.pending_control_tokens().end());
    (void)raw_session.preview_control(raw_tokens, 9);
    const auto raw_output = raw_session.commit_preview();
    failures += check(channel_text(raw_output, ninfer::OutputChannel::Content) == kThinkingControl,
                      "raw output did not preserve the inserted control representation");
    return failures;
}

int test_utf8_and_hidden_eos(const Frontend& frontend) {
    auto prompt             = frontend.prepare_tokens({0});
    auto session            = frontend.make_output_session(prompt, {});
    int failures            = 0;
    std::uint32_t remaining = 4;
    for (const ninfer::TokenId token : {10, 11}) {
        const auto decision = session.preview_model(std::array<ninfer::TokenId, 1>{token},
                                                    remaining, ninfer::FinishReason::OutputLimit);
        failures += check(decision.accepted_tokens == 1 && !decision.finished(),
                          "partial UTF-8 token unexpectedly ended generation");
        const auto output = session.commit_preview();
        remaining -= decision.accepted_tokens;
        failures += check(output.empty(), "partial UTF-8 codepoint was published");
    }
    const auto complete_decision = session.preview_model(
        std::array<ninfer::TokenId, 1>{12}, remaining, ninfer::FinishReason::OutputLimit);
    failures += check(complete_decision.accepted_tokens == 1 && !complete_decision.finished(),
                      "complete UTF-8 token unexpectedly ended generation");
    const auto complete = session.commit_preview();
    failures += check(channel_text(complete, ninfer::OutputChannel::Content) == "中",
                      "UTF-8 codepoint was not published when complete");

    const auto decode_generated = [&](const std::vector<ninfer::TokenId>& tokens,
                                      bool one_token_per_round) {
        auto generated_prompt  = frontend.prepare_tokens({0});
        auto generated_session = frontend.make_output_session(generated_prompt, {});
        std::string text;
        std::uint32_t budget = static_cast<std::uint32_t>(tokens.size());
        if (one_token_per_round) {
            for (const ninfer::TokenId token : tokens) {
                const auto decision =
                    generated_session.preview_model(std::array<ninfer::TokenId, 1>{token}, budget,
                                                    ninfer::FinishReason::OutputLimit);
                budget -= decision.accepted_tokens;
                text += channel_text(generated_session.commit_preview(),
                                     ninfer::OutputChannel::Content);
            }
        } else {
            (void)generated_session.preview_model(tokens, budget,
                                                  ninfer::FinishReason::OutputLimit);
            text = channel_text(generated_session.commit_preview(), ninfer::OutputChannel::Content);
        }
        return text;
    };

    struct Utf8Case {
        std::vector<ninfer::TokenId> tokens;
        std::string expected;
        const char* label;
    };

    const std::string replacement(kUtf8Replacement);
    const std::vector<Utf8Case> utf8_cases = {
        {{10, 1}, replacement + "helloST", "invalid continuation after leading byte"},
        {{10, 11, 1}, replacement + "helloST", "maximal incomplete subpart"},
        {{11, 1}, replacement + "helloST", "isolated continuation byte"},
        {{10, 11}, replacement, "terminal incomplete suffix"},
        {{kByteE0Token, kByte80Token, kByte80Token},
         replacement + replacement + replacement,
         "overlong codepoint"},
        {{kByteEDToken, kByteA0Token, kByte80Token},
         replacement + replacement + replacement,
         "surrogate codepoint"},
        {{kByteF4Token, kByte90Token, kByte80Token, kByte80Token},
         replacement + replacement + replacement + replacement,
         "out-of-range codepoint"},
        {{kByteF5Token, 1}, replacement + "helloST", "invalid leading byte"},
        {{kByteC2Token, kByteA2Token}, "¢", "valid two-byte codepoint"},
        {{kByteF0Token, kByte9FToken, kByte98Token, kByte80Token},
         "😀",
         "valid four-byte codepoint"},
    };
    for (const Utf8Case& test : utf8_cases) {
        const std::string batched = decode_generated(test.tokens, false);
        const std::string split   = decode_generated(test.tokens, true);
        failures += check(batched == test.expected, test.label);
        failures += check(split == test.expected, test.label);
        failures += check(split == batched,
                          "generated UTF-8 recovery changed across decode-round boundaries");
    }

    auto repaired_stop_prompt = frontend.prepare_tokens({0});
    ninfer::StopPolicy repaired_stop;
    repaired_stop.strings.push_back(ninfer::StopString{.text = "STOP"});
    auto repaired_stop_session = frontend.make_output_session(repaired_stop_prompt, repaired_stop);
    const auto repaired_stop_decision = repaired_stop_session.preview_model(
        std::array<ninfer::TokenId, 3>{10, 1, 2}, 3, ninfer::FinishReason::OutputLimit);
    failures += check(repaired_stop_decision.finish_reason == ninfer::FinishReason::StopString,
                      "UTF-8 recovery hid a following stop string");
    const auto repaired_stop_output = repaired_stop_session.commit_preview();
    failures += check(channel_text(repaired_stop_output, ninfer::OutputChannel::Content) ==
                          replacement + "hello",
                      "UTF-8 recovery changed stop-string publication");

    auto repaired_reasoning_prompt  = thinking_prompt(frontend);
    auto repaired_reasoning_session = frontend.make_output_session(repaired_reasoning_prompt, {});
    const auto repaired_reasoning_decision = repaired_reasoning_session.preview_model(
        std::array<ninfer::TokenId, 3>{10, 3, 4}, 3, ninfer::FinishReason::OutputLimit);
    failures +=
        check(repaired_reasoning_decision.finish_reason == ninfer::FinishReason::OutputLimit,
              "UTF-8 recovery changed reasoning termination");
    const auto repaired_reasoning_output = repaired_reasoning_session.commit_preview();
    failures += check(channel_text(repaired_reasoning_output, ninfer::OutputChannel::Reasoning) ==
                              replacement + "thought" &&
                          channel_text(repaired_reasoning_output, ninfer::OutputChannel::Content) ==
                              "answer",
                      "UTF-8 recovery changed reasoning/content channel routing");

    auto eos_prompt         = frontend.prepare_tokens({0});
    auto eos_session        = frontend.make_output_session(eos_prompt, {});
    const auto eos_decision = eos_session.preview_model(std::array<ninfer::TokenId, 1>{6}, 2,
                                                        ninfer::FinishReason::OutputLimit);
    failures += check(eos_decision.accepted_tokens == 1 &&
                          eos_decision.finish_reason == ninfer::FinishReason::StopToken,
                      "default EOS token did not end generation");
    const auto eos = eos_session.commit_preview();
    failures += check(eos.empty(), "default EOS token was published");

    auto raw_prompt  = frontend.prepare_tokens({0});
    auto raw_session = frontend.make_output_session(
        raw_prompt, {}, ninfer::OutputOptions{.raw = true, .preserve_special_tokens = false});
    const auto raw_eos_decision = raw_session.preview_model(std::array<ninfer::TokenId, 1>{6}, 2,
                                                            ninfer::FinishReason::OutputLimit);
    failures += check(raw_eos_decision.accepted_tokens == 1 &&
                          raw_eos_decision.finish_reason == ninfer::FinishReason::StopToken,
                      "raw EOS token did not end generation");
    const auto raw_eos = raw_session.commit_preview();
    failures += check(channel_text(raw_eos, ninfer::OutputChannel::Content) == "<eos>",
                      "raw output did not preserve the terminal special token");
    return failures;
}

int test_disabled_vision() {
    const Frontend frontend = FrontendFactory::create_component(resources(), false);
    int failures = check(throws_invalid_argument([&] { (void)frontend.prepare(image_input()); }),
                         "Vision-disabled frontend accepted media during prepare");
    failures += check(throws_invalid_argument([&] { (void)frontend.count_tokens(image_input()); }),
                      "Vision-disabled frontend accepted media during token counting");

    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    failures += check(frontend.prepare(std::move(input)).summary().prompt_tokens != 0,
                      "Vision-disabled frontend rejected a text prompt");
    return failures;
}

int test_invalid_media_classification() {
    const Frontend frontend = FrontendFactory::create_component(resources());
    auto invalid_image      = [] {
        ninfer::PromptInput input                    = image_input();
        input.messages[0].parts[0].media.bytes       = {0x00, 0x01, 0x02};
        input.messages[0].parts[0].media.source_name = "invalid-image.bin";
        return input;
    };
    auto is_invalid_media = [](const auto& operation) {
        try {
            operation();
        } catch (const ninfer::RequestError& error) {
            return error.kind() == ninfer::RequestErrorKind::InvalidMedia;
        }
        return false;
    };
    int failures = check(is_invalid_media([&] { (void)frontend.prepare(invalid_image()); }),
                         "invalid image preparation lost its typed request error");
    failures += check(is_invalid_media([&] { (void)frontend.count_tokens(invalid_image()); }),
                      "invalid image token counting lost its typed request error");
    return failures;
}

int test_media_cache_reuses_immutable_payload() {
    const Frontend frontend = FrontendFactory::create_component(resources());
    auto first              = frontend.prepare(image_input());
    auto second             = frontend.prepare(image_input());
    const auto& first_data  = FrontendFactory::inspect(first);
    const auto& second_data = FrontendFactory::inspect(second);
    int failures            = check(
        first_data.prepare.media_cache_misses == 1 && first_data.prepare.media_cache_hits == 0 &&
            first_data.prepare.built_patch_bytes == 16 * 1536 * sizeof(std::uint16_t),
        "first media preparation did not publish one cache miss");
    failures += check(
        second_data.prepare.media_cache_hits == 1 && second_data.prepare.media_cache_misses == 0 &&
            second_data.prepare.built_patch_bytes == 0 &&
            second_data.prepare.reused_patch_bytes == 16 * 1536 * sizeof(std::uint16_t),
        "second media preparation did not use the prepared-media cache");
    failures +=
        check(first_data.media_payloads.size() == 1 && second_data.media_payloads.size() == 1 &&
                  first_data.media_payloads.front() == second_data.media_payloads.front(),
              "cache hit did not share the immutable per-item patch payload");
    const ninfer::MediaCacheSummary cache = frontend.media_cache_summary();
    failures += check(cache.entries == 1 && cache.misses == 1 && cache.hits == 1 &&
                          cache.retained_bytes == 16 * 1536 * sizeof(std::uint16_t) &&
                          cache.live_bytes == cache.retained_bytes &&
                          cache.preprocess_threads >= 1 && cache.preprocess_threads <= 16,
                      "Frontend media-cache accounting does not describe the retained payload");
    return failures;
}

int test_media_payload_outlives_frontend_cache() {
    ninfer::targets::qwen3_6::PreparedPrompt survivor;
    {
        const Frontend frontend = FrontendFactory::create_component(resources());
        survivor                = frontend.prepare(image_input());
    }
    const auto& data = FrontendFactory::inspect(survivor);
    return check(data.media_payloads.size() == 1 && data.media_payloads.front() &&
                     data.media_payloads.front()->patch_elements == 16 * 1536 &&
                     near(bf16_value(data.media_payloads.front()->span().front()), -1.0F),
                 "request-pinned media payload did not survive its Frontend cache owner");
}

int test_media_live_bytes_follow_last_payload_reference() {
    fi::MediaPreprocessCache cache(0, 1ULL << 20, 1);
    auto payload = cache.allocate_payload(1536, {});
    int failures = check(cache.stats().live_bytes == 1536 * sizeof(std::uint16_t),
                         "media live-byte account did not charge the allocated payload");
    payload.reset();
    failures += check(cache.stats().live_bytes == 0,
                      "media live-byte account did not release the final payload reference");
    return failures;
}

int test_media_cache_singleflight() {
    auto cache = std::make_shared<fi::MediaPreprocessCache>(1ULL << 20, 2ULL << 20);
    fi::MediaCacheKey key;
    key.digest.front() = 0x5a;

    std::promise<void> producer_started;
    std::future<void> producer_started_future = producer_started.get_future();
    std::promise<void> release_producer;
    std::shared_future<void> release_future = release_producer.get_future().share();
    std::atomic<int> builders{0};
    std::array<fi::PreparedMedia, 2> results;
    std::array<std::exception_ptr, 2> errors;
    std::array<fi::MediaCacheRequestStats, 2> request_stats;

    const auto builder = [&]() {
        const int count = ++builders;
        if (count == 1) { producer_started.set_value(); }
        release_future.wait();
        auto payload                    = cache->allocate_payload(1536, {});
        payload->mutable_span().front() = 42;
        fi::VisionItem item;
        item.grid = {1, 1, 1};
        return fi::PreparedMedia{std::move(item), std::move(payload)};
    };
    const auto run = [&](std::size_t index) {
        try {
            results[index] = cache->get_or_prepare(key, {}, builder, request_stats[index]);
        } catch (...) { errors[index] = std::current_exception(); }
    };

    std::thread first(run, 0);
    producer_started_future.wait();
    std::thread second(run, 1);
    const auto wait_limit = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (cache->stats().singleflight_waits == 0 &&
           std::chrono::steady_clock::now() < wait_limit) {
        std::this_thread::yield();
    }
    release_producer.set_value();
    first.join();
    second.join();

    return check(!errors[0] && !errors[1] && builders == 1 && results[0].payload &&
                     results[0].payload == results[1].payload &&
                     cache->stats().singleflight_waits == 1,
                 "concurrent identical media did not collapse into one preprocessing flight");
}

int test_media_cache_runs_independent_misses_in_parallel() {
    auto cache = std::make_shared<fi::MediaPreprocessCache>(1ULL << 20, 2ULL << 20, 4);
    std::array<fi::PendingMedia, 4> pending;
    std::atomic<int> started{0};
    std::atomic<int> active{0};
    std::atomic<int> maximum_active{0};

    for (std::size_t index = 0; index < pending.size(); ++index) {
        fi::MediaCacheKey key;
        key.digest.front() = static_cast<std::uint8_t>(index + 1);
        pending[index]     = cache->begin_prepare(key, {}, [&, index] {
            ++started;
            const int now = ++active;
            int maximum   = maximum_active.load();
            while (now > maximum && !maximum_active.compare_exchange_weak(maximum, now)) {}
            const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            while (started.load() != static_cast<int>(pending.size()) &&
                   std::chrono::steady_clock::now() < limit) {
                std::this_thread::yield();
            }
            auto payload                    = cache->allocate_payload(1536, {});
            payload->mutable_span().front() = static_cast<std::uint16_t>(index);
            --active;
            fi::VisionItem item;
            item.grid = {1, 1, 1};
            return fi::PreparedMedia{std::move(item), std::move(payload)};
        });
    }

    fi::MediaCacheRequestStats request_stats;
    std::array<fi::PreparedMedia, 4> results;
    for (std::size_t index = 0; index < pending.size(); ++index) {
        results[index] = cache->await(pending[index], {}, request_stats);
    }
    return check(started == 4 && maximum_active == 4 && request_stats.misses == 4 &&
                     cache->stats().preprocess_threads == 4 && results.back().payload &&
                     results.back().payload->span().front() == 3,
                 "independent media misses did not use the bounded preprocessing pool");
}

int test_many_images_prepare_in_one_parallel_batch() {
    const Frontend frontend = FrontendFactory::create_component(resources());
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    for (int index = 0; index < 19; ++index) {
        ninfer::MessagePart image;
        image.kind               = ninfer::MessagePartKind::Media;
        image.media.kind         = ninfer::MediaKind::Image;
        image.media.bytes        = gradient_ppm();
        image.media.bytes.back() = static_cast<std::uint8_t>(index);
        image.media.media_type   = "image/x-portable-pixmap";
        image.media.source_name  = "parallel-" + std::to_string(index) + ".ppm";
        message.parts.push_back(std::move(image));
    }
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    const auto prepared = frontend.prepare(std::move(input));
    const auto& data    = FrontendFactory::inspect(prepared);
    return check(data.vision_items.size() == 19 && data.media_payloads.size() == 19 &&
                     data.prepare.media_cache_misses == 19 && data.prepare.raw_patches == 19 * 16 &&
                     frontend.media_cache_summary().preprocess_threads > 1,
                 "one request with 19 distinct images did not complete as a parallel media batch");
}

int test_media_preparation_cancellation() {
    const Frontend frontend = FrontendFactory::create_component(resources());
    ninfer::PreparationControl control{
        .deadline     = {},
        .cancellation = ninfer::CancellationView([] { return true; }),
    };
    try {
        (void)frontend.prepare(image_input(), control);
    } catch (const ninfer::RequestError& error) {
        return check(error.kind() == ninfer::RequestErrorKind::Cancelled &&
                         frontend.media_cache_summary().entries == 0,
                     "cancelled media preparation published a cache entry");
    }
    return check(false, "cancelled media preparation completed successfully");
}

} // namespace

int main() {
    const FrontendResources owned = resources();
    const Frontend frontend       = FrontendFactory::create_component(owned);
    int failures                  = 0;
    failures += test_official_tokenizer_merge();
    failures += test_bpe_merge_order();
    failures += test_boundary_aware_tokenization();
    failures += test_literal_added_token_provenance();
    failures += test_repeated_special_tokens_scan_linearly();
    failures += test_bounded_tokenizer_prefix();
    failures += test_context_capacity_guard();
    failures += test_official_chat_template();
    failures += test_ordered_instruction_turns();
    failures += test_assistant_continuation();
    failures += test_reasoning_effort_chat_template();
    failures += test_rewrite_checkpoint_trace();
    failures += test_adjacent_tool_message_boundary();
    failures += test_official_resource_guards();
    failures += test_invalid_public_part_enums(frontend);
    failures += test_text_and_image_prepare(frontend);
    failures += test_literal_control_tokens_with_media();
    failures += test_image_resize_rejection_policy();
    failures += test_explicit_leading_instruction_cache_boundary();
    failures += test_media_admission_uses_aggregate_resources(frontend);
    failures += test_multimodal_prompt_over_removed_32k_cap(frontend);
    failures += test_attention_pairs_are_diagnostic(frontend);
    failures += test_video_prepare(frontend);
    failures += test_cross_round_stop(frontend);
    failures += test_same_token_stop_priority(frontend);
    failures += test_terminal_flush(frontend);
    failures += test_structured_tool_output();
    failures += test_reasoning_split(frontend);
    failures += test_thinking_budget_control(frontend);
    failures += test_utf8_and_hidden_eos(frontend);
    failures += test_media_cache_reuses_immutable_payload();
    failures += test_media_payload_outlives_frontend_cache();
    failures += test_media_live_bytes_follow_last_payload_reference();
    failures += test_media_cache_singleflight();
    failures += test_media_cache_runs_independent_misses_in_parallel();
    failures += test_many_images_prepare_in_one_parallel_batch();
    failures += test_media_preparation_cancellation();
    failures += test_invalid_media_classification();
    failures += test_disabled_vision();
    return failures == 0 ? 0 : 1;
}
