#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>

#include "targets/qwen3_6/impl/frontend/chat_template.h"
#include "targets/qwen3_6/impl/frontend/test_access.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace {

using Frontend          = ninfer::targets::qwen3_6::Frontend;
using FrontendFactory   = ninfer::targets::qwen3_6::FrontendTestAccess;
using FrontendResources = ninfer::targets::qwen3_6::FrontendResources;
using PublishedOutput   = ninfer::targets::qwen3_6::PublishedOutput;
namespace fi            = ninfer::targets::qwen3_6::frontend_internal;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
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

FrontendResources resources(const std::string& chat_template = thinking_toggle_template_source()) {
    FrontendResources result;
    result.chat_template_jinja  = chat_template;
    const nlohmann::json tokens = nlohmann::json::array(
        {added(1, "helloST"), added(2, "OPtail"), added(3, "thought</thi"),
         added(4, "nk>\n\nanswer"), added(6, "<eos>", true), added(7, "<0.0 seconds>"),
         added(30, "user\n"), added(31, "assistant\n"), added(32, "\n"),
         added(248045, "<|im_start|>", true), added(248046, "<|im_end|>", true),
         added(248053, "<|vision_start|>", true), added(248054, "<|vision_end|>", true),
         added(248056, "<|image_pad|>", true), added(248057, "<|video_pad|>", true),
         added(248068, "<think>"), added(248069, "</think>")});
    result.tokenizer_json = nlohmann::json{
        {"model",
         {{"type", "BPE"},
          {"vocab", {{"x", 0}, {"ä", 10}, {"¸", 11}, {"Ń", 12}}},
          {"merges", nlohmann::json::array()}}},
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

std::vector<std::uint8_t> block_ppm(int width, int height, std::uint8_t seed) {
    std::vector<std::uint8_t> ppm;
    const std::string header =
        "P6\n" + std::to_string(width) + " " + std::to_string(height) + "\n255\n";
    for (const char byte : header) {
        ppm.push_back(static_cast<std::uint8_t>(static_cast<unsigned char>(byte)));
    }
    for (std::size_t index = 0; index < static_cast<std::size_t>(width) * height; ++index) {
        ppm.push_back(static_cast<std::uint8_t>((index + seed) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3 + seed) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7 + seed) & 0xff));
    }
    return ppm;
}

ninfer::MessagePart ppm_image_part(int width, int height, std::uint8_t seed) {
    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = block_ppm(width, height, seed);
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "inline.ppm";
    return image;
}

ninfer::PromptInput image_input() {
    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = gradient_ppm();
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "inline.ppm";
    ninfer::ChatMessage message;
    message.role = "user";
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

fi::ChatMessage chat_message(std::string role, std::string content) {
    fi::ChatMessage message;
    message.role = std::move(role);
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

int test_official_tokenizer_merge() {
    const std::string tokenizer_json =
        read_file("/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16/tokenizer.json");
    const std::string tokenizer_config_json =
        read_file("/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16/tokenizer_config.json");
    const std::string generation_config_json =
        read_file("/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16/generation_config.json");
    const fi::Tokenizer tokenizer({.tokenizer_json         = tokenizer_json,
                                   .tokenizer_config_json  = tokenizer_config_json,
                                   .generation_config_json = generation_config_json});

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

int test_official_chat_template() {
    int failures = 0;
    failures += check(render_chat_text({chat_message("user", "hello")}) ==
                          "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n",
                      "ordinary user prompt differs from the official template");

    fi::ChatRenderOptions no_generation;
    no_generation.add_generation_prompt = false;
    failures += check(
        render_chat_text({chat_message("system", "  be concise  "), chat_message("user", "hello")},
                         no_generation) == "<|im_start|>system\nbe concise<|im_end|>\n"
                                           "<|im_start|>user\nhello<|im_end|>\n",
        "leading system prompt differs from the official template");
    failures += check(render_chat_text({chat_message("system", ""), chat_message("user", "hello")},
                                       no_generation) ==
                          "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n",
                      "empty leading system prompt differs from the official template");

    fi::ChatMessage tool_assistant = chat_message("assistant", "");
    tool_assistant.tool_calls.push_back(
        {.id = "", .name = "f", .arguments_json = R"({"flag":true,"nested":{"x":[1,2]}})"});
    failures +=
        check(render_chat_text({chat_message("user", "hi"), tool_assistant}, no_generation) ==
                  "<|im_start|>user\nhi<|im_end|>\n"
                  "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                  "<tool_call>\n<function=f>\n<parameter=flag>\ntrue\n</parameter>\n"
                  "<parameter=nested>\n{\"x\": [1, 2]}\n</parameter>\n"
                  "</function>\n</tool_call><|im_end|>\n",
              "nested or boolean tool arguments differ from official JSON rendering");

    fi::ChatRenderOptions no_thinking;
    no_thinking.enable_thinking = false;
    failures += check(
        render_chat_text({chat_message("user", "q1"),
                          chat_message("assistant", "<think>\nold thought\n</think>\n\nold answer"),
                          chat_message("user", "q2")},
                         no_thinking) == "<|im_start|>user\nq1<|im_end|>\n"
                                         "<|im_start|>assistant\nold answer<|im_end|>\n"
                                         "<|im_start|>user\nq2<|im_end|>\n"
                                         "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "thinking history differs from the official template");

    fi::ChatMessage lookup = chat_message("assistant", "");
    lookup.tool_calls.push_back(
        {.id = "", .name = "lookup", .arguments_json = R"({"city":"Paris"})"});
    failures += check(
        render_chat_text({chat_message("user", "weather?"), lookup, chat_message("tool", "sunny"),
                          chat_message("tool", "20C"), chat_message("user", "thanks")},
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
        render_chat_text({chat_message("system", "be exact"), chat_message("user", "hi")}, tools);
    failures += check(
        tools_rendered.find("\n{\"type\": \"function\", \"function\": {\"name\": \"f\", "
                            "\"description\": \"d\", \"parameters\": {\"type\": \"object\", "
                            "\"properties\": {\"flag\": {\"type\": \"boolean\"}}}}}\n</tools>") !=
                std::string::npos &&
            tools_rendered.ends_with(
                "</IMPORTANT>\n\nbe exact<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"),
        "tools system block differs from official tojson rendering");

    failures += check(throws_invalid_argument([&] {
                          (void)render_chat(
                              {chat_message("developer", "policy"), chat_message("user", "hi")},
                              no_generation);
                      }),
                      "direct developer role was accepted by the model frontend");
    failures +=
        check(throws_invalid_argument([&] {
                  (void)render_chat({chat_message("user", "hi"), chat_message("system", "late")},
                                    no_generation);
              }),
              "late system role was accepted by the model frontend");
    failures += check(throws_invalid_argument([&] {
                          (void)render_chat({chat_message("system", "only")}, no_generation);
                      }),
                      "message history without a user query was accepted");
    failures +=
        check(throws_invalid_argument([&] {
                  (void)render_chat({chat_message("user", "hi"), chat_message("unexpected", "bad")},
                                    no_generation);
              }),
              "unexpected chat role was accepted");
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
        return reasoning_effort_template().render({chat_message("user", "hello")}, options).text;
    };
    const std::string tail = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n";
    failures +=
        check(reasoning_effort_template().render({chat_message("user", "hello")}).text ==
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
    failures +=
        check(reasoning_effort_template()
                      .render({chat_message("system", ""), chat_message("user", "hello")}, disabled)
                      .text == "<|im_start|>user\nhello<|im_end|>\n"
                               "<|im_start|>assistant\n<think>\n\n</think>\n\n",
              "disabled thinking did not suppress effort and an empty system turn");
    disabled.reasoning_effort = ninfer::ReasoningEffort::Low;
    failures += check(throws_invalid_argument([&] {
                          (void)reasoning_effort_template().render({chat_message("user", "hello")},
                                                                   disabled);
                      }),
                      "reasoning effort and disabled thinking were accepted together");

    fi::ChatRenderOptions unsupported;
    unsupported.reasoning_effort = ninfer::ReasoningEffort::Low;
    failures += check(throws_invalid_argument([&] {
                          (void)thinking_toggle_template().render({chat_message("user", "hello")},
                                                                  unsupported);
                      }),
                      "thinking-toggle template accepted reasoning effort");

    fi::ChatMessage previous   = chat_message("assistant", "old answer");
    previous.reasoning_content = "old thought";
    fi::ChatRenderOptions no_generation;
    no_generation.add_generation_prompt = false;
    no_generation.reasoning_effort      = ninfer::ReasoningEffort::Medium;
    const std::string preserved =
        reasoning_effort_template()
            .render({chat_message("user", "q1"), previous, chat_message("user", "q2")},
                    no_generation)
            .text;
    failures += check(
        preserved.find("<|im_start|>assistant\n<think>\nold thought\n</think>\n\nold answer") !=
            std::string::npos,
        "reasoning-effort template did not preserve prior thinking by default");
    no_generation.preserve_thinking = false;
    failures +=
        check(reasoning_effort_template()
                      .render({chat_message("user", "q1"), previous, chat_message("user", "q2")},
                              no_generation)
                      .text.find("old thought") == std::string::npos,
              "explicit preserve_thinking=false did not remove prior thinking");

    fi::ChatMessage empty_arguments = chat_message("assistant", "");
    empty_arguments.tool_calls.push_back({.id = "", .name = "f", .arguments_json = ""});
    failures += check(reasoning_effort_template()
                          .render({chat_message("user", "call"), empty_arguments}, no_generation)
                          .text.ends_with("<tool_call>\n<function=f>\n</function>\n"
                                          "</tool_call><|im_end|>\n"),
                      "empty tool arguments did not follow the reasoning-effort template");
    return failures;
}

int test_turn_rewrite_trace() {
    const std::string assistant_header = "<|im_start|>assistant\n";
    fi::ChatMessage first              = chat_message("assistant", "");
    first.reasoning_content            = "first thought";
    first.parts.front().text           = "first answer";
    fi::ChatMessage second             = chat_message("assistant", "");
    second.reasoning_content           = "second thought";
    second.parts.front().text          = "second answer";

    const std::vector<fi::ChatMessage> tool_loop{chat_message("user", "question"), first,
                                                 chat_message("tool", "result one"), second,
                                                 chat_message("tool", "result two")};
    const fi::RenderedChat open    = render_chat(tool_loop);
    const std::size_t first_header = open.text.find(assistant_header);
    int failures =
        check(first_header != std::string::npos && open.turn_rewrite_byte_offset &&
                  *open.turn_rewrite_byte_offset == first_header + assistant_header.size(),
              "tool loop did not retain its first assistant rewrite boundary");

    fi::ChatRenderOptions preserve;
    preserve.preserve_thinking       = true;
    const fi::RenderedChat preserved = render_chat(tool_loop, preserve);
    failures += check(preserved.turn_rewrite_byte_offset == open.turn_rewrite_byte_offset,
                      "preserve_thinking changed the turn rewrite boundary");

    std::vector<fi::ChatMessage> next_turn = tool_loop;
    next_turn.push_back(chat_message("user", "next question"));
    const fi::RenderedChat next    = render_chat(next_turn);
    const std::size_t final_header = next.text.rfind(assistant_header);
    failures += check(final_header != std::string::npos && next.turn_rewrite_byte_offset &&
                          *next.turn_rewrite_byte_offset == final_header + assistant_header.size(),
                      "new user turn did not move the rewrite boundary to its generation opener");

    fi::ChatRenderOptions no_generation;
    no_generation.add_generation_prompt = false;
    const fi::RenderedChat no_assistant =
        render_chat({chat_message("user", "question")}, no_generation);
    failures += check(!no_assistant.turn_rewrite_byte_offset,
                      "boundary-less prompt unexpectedly published a rewrite boundary");

    const fi::RenderedChat wrapped =
        render_chat({chat_message("user", "question"), first,
                     chat_message("user", "<tool_response>compat result</tool_response>"), second},
                    no_generation);
    const std::size_t wrapped_first = wrapped.text.find(assistant_header);
    failures +=
        check(wrapped.turn_rewrite_byte_offset &&
                  *wrapped.turn_rewrite_byte_offset == wrapped_first + assistant_header.size(),
              "bare tool-response wrapper incorrectly advanced the real user turn");
    return failures;
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
    text_message.role = "user";
    text_message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::PromptInput text_input;
    text_input.messages.push_back(std::move(text_message));
    auto text             = frontend.prepare(std::move(text_input));
    const auto& text_data = FrontendFactory::inspect(text);
    const std::vector<ninfer::TokenId> expected{248045, 30, 0, 248046, 32, 248045, 31, 248068, 32};
    int failures =
        check(text_data.token_ids == expected, "text frontend did not render/tokenize chat");
    failures += check(text_data.identity.turn_rewrite_boundary == 7 &&
                          text_data.starts_in_reasoning && !text_data.has_media(),
                      "text frontend did not preserve prefix/thinking identity");
    failures +=
        check(text_data.position_axis(0).back() == 8 && text_data.position_axis(1).back() == 8 &&
                  text_data.position_axis(2).back() == 8,
              "text frontend did not construct axis-major positions");

    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = gradient_ppm();
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "inline.ppm";
    ninfer::ChatMessage image_message;
    image_message.role = "user";
    image_message.parts.push_back(std::move(image));
    ninfer::PromptInput image_input;
    image_input.messages.push_back(std::move(image_message));
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
    failures += check(
        prepared_data.patches.size() == 16 * 1536 && prepared_data.prepare.raw_patches == 16 &&
            prepared_data.prepare.vision_tokens == 4 && prepared_data.identity.reusable &&
            prepared_data.identity.turn_rewrite_boundary &&
            *prepared_data.identity.turn_rewrite_boundary < prepared_data.token_ids.size(),
        "image frontend did not own the expected patch payload and identity");
    if (prepared_data.patches.size() == 16 * 1536) {
        failures += check(near(prepared_data.patches[0], -1.0F) &&
                              near(prepared_data.patches[1], 1.0F / 127.5F - 1.0F) &&
                              near(prepared_data.patches[256], -1.0F) &&
                              near(prepared_data.patches[1536], 16.0F / 127.5F - 1.0F),
                          "image frontend patch normalization/order is incorrect");
    }
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
    message.role = "user";
    message.parts.push_back(std::move(video));
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));

    auto prepared             = frontend.prepare(std::move(input));
    const auto& prepared_data = FrontendFactory::inspect(prepared);
    int failures = check(prepared_data.vision_items.size() == 1 && prepared_data.has_media(),
                         "video frontend did not retain one Vision item");
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
    failures +=
        check(prepared_data.patches.size() == 16 * 1536 &&
                  near(prepared_data.patches[0], prepared_data.patches[256]) &&
                  prepared_data.prepare.raw_patches == 16 &&
                  prepared_data.prepare.vision_tokens == 4 && prepared_data.identity.reusable,
              "video frontend did not duplicate the odd temporal frame correctly");
    return failures;
}

int test_multi_image_attention_pairs_not_enforced(const Frontend& frontend) {
    // The removed attention_pairs guard rejected media requests whose cumulative
    // t*(h*w)^2 exceeded 128 Mi. Two 2048x1600 images sum to ~328M pairs and must
    // now prepare cleanly, with the statistic still reported (informational only).
    constexpr std::uint64_t old_attention_cap = 128ULL * 1024ULL * 1024ULL;
    ninfer::ChatMessage message;
    message.role = "user";
    message.parts.push_back(ppm_image_part(2048, 1600, 11));
    message.parts.push_back(ppm_image_part(2048, 1600, 97));
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));

    int failures = 0;
    try {
        auto prepared    = frontend.prepare(std::move(input));
        const auto& data = FrontendFactory::inspect(prepared);
        failures += check(data.vision_items.size() == 2,
                          "multi-image prepare did not retain both vision items");
        std::uint64_t expected_pairs = 0;
        for (const auto& item : data.vision_items) {
            const std::uint64_t spatial =
                static_cast<std::uint64_t>(item.grid.height) * item.grid.width;
            expected_pairs += static_cast<std::uint64_t>(item.grid.temporal) * spatial * spatial;
        }
        failures += check(data.prepare.attention_pairs == expected_pairs,
                          "multi-image prepare misreported the attention_pairs statistic");
        failures += check(expected_pairs > old_attention_cap,
                          "multi-image fixture no longer exceeds the removed 128 Mi cap");
    } catch (const std::exception& error) {
        std::cerr << "multi-image prepare threw after guard removal: " << error.what() << '\n';
        ++failures;
    }
    return failures;
}

int test_cross_round_stop(const Frontend& frontend) {
    auto prompt = frontend.prepare_tokens({0});
    ninfer::StopPolicy stop;
    stop.strings.push_back(ninfer::StopString{.text = "STOP"});
    auto session = frontend.make_output_session(prompt, stop);

    const auto first_decision =
        session.preview(std::array<ninfer::TokenId, 1>{1}, 2, ninfer::FinishReason::OutputLimit);
    int failures     = check(first_decision.accepted_tokens == 1 && !first_decision.finished(),
                             "cross-round stop ended before the stop string was complete");
    const auto first = session.commit_preview();
    failures += check(channel_text(first, ninfer::OutputChannel::Content) == "hello",
                      "cross-round stop did not retain the ambiguous suffix");

    const auto second_decision =
        session.preview(std::array<ninfer::TokenId, 1>{2}, 1, ninfer::FinishReason::OutputLimit);
    failures += check(second_decision.accepted_tokens == 1 &&
                          second_decision.finish_reason == ninfer::FinishReason::StopString,
                      "cross-round stop did not select the exact terminal token prefix");
    const auto second = session.commit_preview();
    failures += check(second.empty(), "stop marker or same-token suffix leaked to output");
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
    auto session = frontend.make_output_session(prompt, stop);
    const auto decision =
        session.preview(std::array<ninfer::TokenId, 1>{2}, 2, ninfer::FinishReason::OutputLimit);
    int failures      = check(decision.accepted_tokens == 1 &&
                                  decision.finish_reason == ninfer::FinishReason::StopString,
                              "same-token stop strings did not select a terminal prefix");
    const auto output = session.commit_preview();
    failures += check(output.empty(),
                      "same-token stops did not prefer the earliest byte and declaration order");
    return failures;
}

int test_terminal_flush(const Frontend& frontend) {
    auto prompt = frontend.prepare_tokens({0});
    ninfer::StopPolicy stop;
    stop.strings.push_back(ninfer::StopString{.text = "STOP"});
    auto session = frontend.make_output_session(prompt, stop);

    const auto first_decision =
        session.preview(std::array<ninfer::TokenId, 1>{1}, 2, ninfer::FinishReason::OutputLimit);
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

int test_reasoning_split(const Frontend& frontend) {
    ninfer::ChatMessage message;
    message.role = "user";
    message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.add_generation_prompt = true;
    input.options.enable_thinking       = true;
    auto prompt                         = frontend.prepare(std::move(input));
    auto session                        = frontend.make_output_session(prompt, {});
    const std::array<ninfer::TokenId, 2> tokens{3, 4};
    const auto decision = session.preview(tokens, 2, ninfer::FinishReason::OutputLimit);
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

int test_utf8_and_hidden_eos(const Frontend& frontend) {
    auto prompt             = frontend.prepare_tokens({0});
    auto session            = frontend.make_output_session(prompt, {});
    int failures            = 0;
    std::uint32_t remaining = 4;
    for (const ninfer::TokenId token : {10, 11}) {
        const auto decision = session.preview(std::array<ninfer::TokenId, 1>{token}, remaining,
                                              ninfer::FinishReason::OutputLimit);
        failures += check(decision.accepted_tokens == 1 && !decision.finished(),
                          "partial UTF-8 token unexpectedly ended generation");
        const auto output = session.commit_preview();
        remaining -= decision.accepted_tokens;
        failures += check(output.empty(), "partial UTF-8 codepoint was published");
    }
    const auto complete_decision = session.preview(std::array<ninfer::TokenId, 1>{12}, remaining,
                                                   ninfer::FinishReason::OutputLimit);
    failures += check(complete_decision.accepted_tokens == 1 && !complete_decision.finished(),
                      "complete UTF-8 token unexpectedly ended generation");
    const auto complete = session.commit_preview();
    failures += check(channel_text(complete, ninfer::OutputChannel::Content) == "中",
                      "UTF-8 codepoint was not published when complete");

    auto eos_prompt         = frontend.prepare_tokens({0});
    auto eos_session        = frontend.make_output_session(eos_prompt, {});
    const auto eos_decision = eos_session.preview(std::array<ninfer::TokenId, 1>{6}, 2,
                                                  ninfer::FinishReason::OutputLimit);
    failures += check(eos_decision.accepted_tokens == 1 &&
                          eos_decision.finish_reason == ninfer::FinishReason::StopToken,
                      "default EOS token did not end generation");
    const auto eos = eos_session.commit_preview();
    failures += check(eos.empty(), "default EOS token was published");

    auto raw_prompt  = frontend.prepare_tokens({0});
    auto raw_session = frontend.make_output_session(
        raw_prompt, {}, ninfer::OutputOptions{.raw = true, .preserve_special_tokens = false});
    const auto raw_eos_decision = raw_session.preview(std::array<ninfer::TokenId, 1>{6}, 2,
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
    message.role = "user";
    message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    failures += check(frontend.prepare(std::move(input)).summary().prompt_tokens != 0,
                      "Vision-disabled frontend rejected a text prompt");
    return failures;
}

} // namespace

int main() {
    const FrontendResources owned = resources();
    const Frontend frontend       = FrontendFactory::create_component(owned);
    int failures                  = 0;
    failures += test_official_tokenizer_merge();
    failures += test_official_chat_template();
    failures += test_reasoning_effort_chat_template();
    failures += test_turn_rewrite_trace();
    failures += test_official_resource_guards();
    failures += test_text_and_image_prepare(frontend);
    failures += test_video_prepare(frontend);
    failures += test_multi_image_attention_pairs_not_enforced(frontend);
    failures += test_cross_round_stop(frontend);
    failures += test_same_token_stop_priority(frontend);
    failures += test_terminal_flush(frontend);
    failures += test_reasoning_split(frontend);
    failures += test_utf8_and_hidden_eos(frontend);
    failures += test_disabled_vision();
    return failures == 0 ? 0 : 1;
}
