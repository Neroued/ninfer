#include "models/inkling2/frontend/frontend.h"
#include "models/inkling2/frontend/prepared_prompt.h"
#include "models/inkling2/frontend/resources.h"
#include "models/inkling2/frontend/tool_contract.h"
#include "text/tokenizer.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
namespace inkling = ninfer::models::inkling2;
using Json        = nlohmann::ordered_json;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("test assertion failed: " + message);
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read file: " + path.string());
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

struct CollectedOutput {
    std::string reasoning;
    std::string content;
    std::vector<GeneratedToolCall> calls;

    void append(const models::frontend::PublishedOutput& events) {
        for (const auto& ev : events) {
            if (const auto* delta = std::get_if<OutputDelta>(&ev)) {
                if (delta->channel == OutputChannel::Reasoning)
                    reasoning += delta->text;
                else
                    content += delta->text;
            } else if (const auto* start = std::get_if<ToolCallStart>(&ev)) {
                require(start->index == calls.size(), "tool start index out of order");
                calls.push_back({start->name, {}, ToolCallStatus::InProgress});
            } else if (const auto* args = std::get_if<ToolArgumentsDelta>(&ev)) {
                require(args->index < calls.size(), "tool args index out of range");
                calls[args->index].arguments_json += args->text;
            } else if (const auto* end = std::get_if<ToolCallEnd>(&ev)) {
                require(end->index < calls.size(), "tool end index out of range");
                calls[end->index].status = end->status;
            }
        }
    }
};

runtime::OutputDecision commit_round(models::frontend::OutputSession& session,
                                     std::span<const TokenId> tokens, CollectedOutput& out,
                                     std::uint32_t budget = 1000) {
    const auto decision = session.preview_model(tokens, budget, FinishReason::OutputLimit);
    out.append(session.commit_preview());
    return decision;
}

bool mask_allows(std::span<const std::uint32_t> mask, TokenId id) {
    return (mask[static_cast<std::size_t>(id) / 32U] & (1U << (static_cast<std::size_t>(id) % 32U))) !=
           0U;
}

std::vector<std::uint32_t> next_mask(models::frontend::OutputSession& session,
                                     std::size_t vocab_size) {
    const std::size_t words = (vocab_size + 31U) / 32U;
    std::vector<std::uint32_t> mask(words, 0U);
    (void)session.grammar_masks({}, mask);
    return mask;
}

// Minimal uncompressed-DEFLATE PNG writer (supports RGB color_type=2 and RGBA color_type=6).
std::uint32_t crc32_calc(const std::uint8_t* data, std::size_t len) {
    std::uint32_t crc = 0xffffffffU;
    for (std::size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int b = 0; b < 8; ++b) { crc = (crc >> 1U) ^ (0xedb88320U & -(crc & 1U)); }
    }
    return ~crc;
}

void push_be32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    out.push_back(static_cast<std::uint8_t>((v >> 24U) & 0xffU));
    out.push_back(static_cast<std::uint8_t>((v >> 16U) & 0xffU));
    out.push_back(static_cast<std::uint8_t>((v >> 8U) & 0xffU));
    out.push_back(static_cast<std::uint8_t>(v & 0xffU));
}

void push_png_chunk(std::vector<std::uint8_t>& png, const char type[4],
                    const std::vector<std::uint8_t>& data) {
    push_be32(png, static_cast<std::uint32_t>(data.size()));
    const std::size_t crc_start = png.size();
    png.insert(png.end(), type, type + 4);
    png.insert(png.end(), data.begin(), data.end());
    push_be32(png, crc32_calc(png.data() + crc_start, png.size() - crc_start));
}

std::vector<std::uint8_t> make_png(int width, int height, int channels,
                                   const std::vector<std::uint8_t>& pixels) {
    require(channels == 3 || channels == 4, "make_png expects RGB or RGBA");
    std::vector<std::uint8_t> raw;
    raw.reserve(static_cast<std::size_t>(height) * (1U + static_cast<std::size_t>(width) * channels));
    for (int y = 0; y < height; ++y) {
        raw.push_back(0); // filter type None
        const std::size_t row = static_cast<std::size_t>(y) * width * channels;
        raw.insert(raw.end(), pixels.begin() + row, pixels.begin() + row + width * channels);
    }

    // Wrap raw scanlines in stored zlib blocks.
    std::vector<std::uint8_t> zlib;
    zlib.push_back(0x78);
    zlib.push_back(0x01);
    std::size_t pos = 0;
    while (pos < raw.size()) {
        const std::size_t n = std::min<std::size_t>(raw.size() - pos, 65535U);
        const bool last     = (pos + n == raw.size());
        zlib.push_back(last ? 0x01 : 0x00);
        const auto len16  = static_cast<std::uint16_t>(n);
        const auto nlen16 = static_cast<std::uint16_t>(~len16);
        zlib.push_back(static_cast<std::uint8_t>(len16 & 0xffU));
        zlib.push_back(static_cast<std::uint8_t>(len16 >> 8U));
        zlib.push_back(static_cast<std::uint8_t>(nlen16 & 0xffU));
        zlib.push_back(static_cast<std::uint8_t>(nlen16 >> 8U));
        zlib.insert(zlib.end(), raw.begin() + pos, raw.begin() + pos + n);
        pos += n;
    }
    std::uint32_t s1 = 1, s2 = 0;
    for (const std::uint8_t b : raw) {
        s1 = (s1 + b) % 65521U;
        s2 = (s2 + s1) % 65521U;
    }
    push_be32(zlib, (s2 << 16U) | s1);

    std::vector<std::uint8_t> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    std::vector<std::uint8_t> ihdr;
    push_be32(ihdr, static_cast<std::uint32_t>(width));
    push_be32(ihdr, static_cast<std::uint32_t>(height));
    ihdr.push_back(8);                                  // bit depth
    ihdr.push_back(channels == 4 ? 6 : 2);              // RGBA or RGB
    ihdr.push_back(0);                                  // compression
    ihdr.push_back(0);                                  // filter
    ihdr.push_back(0);                                  // interlace
    push_png_chunk(png, "IHDR", ihdr);
    push_png_chunk(png, "IDAT", zlib);
    push_png_chunk(png, "IEND", {});
    return png;
}

std::uint16_t ref_normalize_bf16(float raw_value, int channel) {
    constexpr std::array<float, 3> clip_mean{0.48145466F, 0.4578275F, 0.40821073F};
    constexpr std::array<float, 3> proc_std{0.26862954F, 0.2613026F, 0.2757771F};
    constexpr std::array<float, 3> clip_std{0.26862954F, 0.26130258F, 0.27577711F};
    const auto to_bf16 = [](float v) {
        const auto bits = std::bit_cast<std::uint32_t>(v);
        return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
    };
    const auto from_bf16 = [](std::uint16_t b) {
        return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16U);
    };
    const float clipped =
        from_bf16(to_bf16((raw_value / 255.0F - clip_mean[channel]) / proc_std[channel]));
    return to_bf16(2.0F * (clipped * clip_std[channel] + clip_mean[channel]) - 1.0F);
}

// Builds a self-contained byte-level Inkling2 tokenizer fixture so tests run without external files.
struct SyntheticFixture {
    std::string tokenizer_json;
    std::string tokenizer_config_json;
    std::string chat_template_jinja;
    std::string generation_config_json;
    std::string processor_config_json;
    inkling::FrontendResources resources;

    SyntheticFixture() {
        // Build byte-level BPE mapping for all 256 byte values (GPT-2/O200k byte table).
        std::array<int, 256> byte_to_cp{};
        int extra = 0;
        for (int b = 0; b < 256; ++b) {
            if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
                byte_to_cp[b] = b;
            } else {
                byte_to_cp[b] = 256 + extra++;
            }
        }
        auto utf8_cp = [](int cp) {
            std::string s;
            if (cp < 0x80) {
                s.push_back(static_cast<char>(cp));
            } else {
                s.push_back(static_cast<char>(0xc0 | (cp >> 6)));
                s.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
            }
            return s;
        };

        Json vocab = Json::object();
        for (int b = 0; b < 256; ++b) { vocab[utf8_cp(byte_to_cp[b])] = b; }

        const std::vector<std::string> specials = {
            "<|endoftext|>",
            "<|message_user|>",
            "<|message_model|>",
            "<|message_system|>",
            "<|message_tool|>",
            "<|content_text|>",
            "<|content_image|>",
            "<|content_model_end_sampling|>",
            "<|content_thinking|>",
            "<|end_message|>",
            "<|content_tool_error|>",
            "<|content_xml|>",
            "<|content_invoke_tool_json|>",
            "<|unused_200054|>",
            "<|content_invoke_tool_text|>",
        };
        Json added         = Json::array();
        Json added_decoder = Json::object();
        int next_id        = 256;
        int eos_id         = -1;
        for (const auto& sp : specials) {
            if (sp == "<|content_model_end_sampling|>") eos_id = next_id;
            added.push_back({{"id", next_id},
                             {"content", sp},
                             {"single_word", false},
                             {"lstrip", false},
                             {"rstrip", false},
                             {"normalized", false},
                             {"special", true}});
            added_decoder[std::to_string(next_id)] = {
                {"content", sp}, {"single_word", false}, {"lstrip", false},
                {"rstrip", false}, {"normalized", false}, {"special", true}};
            ++next_id;
        }

        Json tok = {
            {"version", "1.0"},
            {"added_tokens", added},
            {"normalizer", nullptr},
            {"pre_tokenizer",
             {{"type", "Sequence"},
              {"pretokenizers",
               Json::array({{{"type", "Split"},
                             {"pattern",
                              {{"Regex",
                                "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}}},
                             {"behavior", "Isolated"},
                             {"invert", false}},
                            {{"type", "ByteLevel"},
                             {"add_prefix_space", false},
                             {"trim_offsets", false},
                             {"use_regex", false}}})}}},
            {"post_processor", nullptr},
            {"decoder",
             {{"type", "ByteLevel"},
              {"add_prefix_space", false},
              {"trim_offsets", false},
              {"use_regex", false}}},
            {"model",
             {{"type", "BPE"},
              {"vocab", vocab},
              {"merges", Json::array()}}},
        };
        tokenizer_json = tok.dump();
        tokenizer_config_json =
            Json{{"added_tokens_decoder", added_decoder},
                 {"eos_token", "<|content_model_end_sampling|>"},
                 {"pad_token", "<|endoftext|>"}}
                .dump();
        generation_config_json = Json{{"do_sample", true},
                                      {"temperature", 1.0},
                                      {"eos_token_id", eos_id}}
                                     .dump();
        processor_config_json =
            Json{{"image_token", "<|unused_200054|>"},
                 {"image_processor",
                  {{"do_convert_rgb", true},
                   {"do_normalize", true},
                   {"do_rescale", true},
                   {"image_mean", {0.5, 0.5, 0.5}},
                   {"image_std", {0.5, 0.5, 0.5}},
                   {"rescale_factor", 1.0 / 255.0},
                   {"size", {{"height", 40}, {"width", 40}}}}}}
                .dump();

        // Exact official Inkling-2-Mini chat_template.jinja.
        chat_template_jinja = R"jinja({%- set effort_map = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99} -%}
{%- set role_token = {"user": "<|message_user|>", "assistant": "<|message_model|>", "system": "<|message_system|>", "tool": "<|message_tool|>"} -%}
{%- macro emit_thinking_effort() -%}
    {%- set eff = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 0.9 -%}
    {%- if eff is string -%}
        {%- set key = eff | trim -%}
        {%- if key not in effort_map -%}
            {{- raise_exception("Unknown reasoning_effort: " ~ eff) -}}
        {%- endif -%}
        {%- set num = effort_map[key] -%}
    {%- else -%}
        {%- set num = eff | float -%}
    {%- endif -%}
    {%- if num < 0.0 or num > 0.99 -%}
        {{- raise_exception("reasoning_effort must be in [0.0, 0.99]") -}}
    {%- endif -%}
    {{- "<|message_system|><|content_text|>Thinking effort level: " -}}
    {%- if num == 0.0 -%}0{%- else -%}{{ num }}{%- endif -%}
    {{- "<|end_message|>" -}}
{%- endmacro -%}
{%- if tools -%}
    {%- set tool_state = namespace(specs=[]) -%}
    {%- for tool in tools -%}
        {%- set fn = tool.function if tool.function is defined else tool -%}
        {%- set spec = {
            "description": (fn.description if fn.description is defined and fn.description else ""),
            "name": fn.name,
            "parameters": (fn.parameters if fn.parameters is defined and fn.parameters else {}),
            "type": (tool.type if tool.type is defined and tool.type else "function"),
        } -%}
        {%- set tool_state.specs = tool_state.specs + [spec] -%}
    {%- endfor -%}
    {{- "<|message_system|>tool_declare<|content_xml|>" -}}
    {{- tool_state.specs | tojson(sort_keys=true, separators=(",", ":")) -}}
    {{- "<|end_message|>" -}}
{%- endif -%}
{%- set state = namespace(effort_emitted=false) -%}
{%- for message in messages -%}
    {%- if message.role not in role_token -%}
        {{- raise_exception("Unknown message role: " ~ message.role) -}}
    {%- endif -%}
    {%- if not state.effort_emitted and message.role != "system" -%}
        {{- emit_thinking_effort() -}}
        {%- set state.effort_emitted = true -%}
    {%- endif -%}
    {%- set header = namespace(text=role_token[message.role]) -%}
    {%- if message.role == "tool" -%}
        {%- if message.name is defined and message.name -%}
            {%- set header.text = header.text ~ message.name -%}
        {%- elif message.tool_call_id is defined and message.tool_call_id -%}
            {%- for prev in messages -%}
                {%- if prev.role == "assistant" and prev.tool_calls -%}
                    {%- for tc in prev.tool_calls -%}
                        {%- if tc.id is defined and tc.id == message.tool_call_id and tc.function.name is defined -%}
                            {%- set header.text = header.text ~ tc.function.name -%}
                        {%- endif -%}
                    {%- endfor -%}
                {%- endif -%}
            {%- endfor -%}
        {%- endif -%}
    {%- elif message.role == "assistant" and message.name is defined and message.name -%}
        {%- set header.text = header.text ~ message.name -%}
    {%- endif -%}
    {%- set rtok = header.text -%}
    {%- set rendered -%}
    {%- if message.role == "assistant" and message.reasoning_content is defined and message.reasoning_content -%}
        {{- rtok ~ "<|content_thinking|>" ~ message.reasoning_content ~ "<|end_message|>" -}}
    {%- endif -%}
    {%- if message.content is string and (message.content or not (message.role == "assistant" and (message.tool_calls or message.reasoning_content))) -%}
        {{- rtok ~ "<|content_text|>" ~ message.content ~ "<|end_message|>" -}}
    {%- elif message.content -%}
        {%- for part in message.content -%}
            {%- if part is string -%}
                {{- rtok ~ "<|content_text|>" ~ part ~ "<|end_message|>" -}}
            {%- elif part.type is not defined or part.type in ("text", "input_text") -%}
                {%- set text_part = (part.text if part.text is defined and part.text is string else "") -%}
                {{- rtok ~ "<|content_text|>" ~ text_part ~ "<|end_message|>" -}}
            {%- elif part.type in ("image", "input_image", "image_url") -%}
                {%- if message.role not in ("user", "tool") -%}
                    {{- raise_exception("Image content is only supported for user and tool messages") -}}
                {%- endif -%}
                {{- rtok ~ "<|content_image|><|unused_200054|><|end_message|>" -}}
            {%- elif part.type in ("audio", "input_audio", "audio_url") -%}
                {{- raise_exception("Inkling 2 does not accept audio input") -}}
            {%- else -%}
                {{- raise_exception("Unsupported content part type: " ~ part.type) -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
    {%- if message.role == "assistant" and message.tool_calls -%}
        {%- for tc in message.tool_calls -%}
            {%- set fn = tc.function -%}
            {%- set args = fn.arguments if fn.arguments is defined and fn.arguments else {} -%}
            {{- "<|message_model|>" ~ fn.name ~ "<|content_invoke_tool_json|>" -}}
            {{- '{"name":' ~ (fn.name | tojson(sort_keys=true, separators=(",", ":"))) ~ ',"args":' -}}
            {{- "{" -}}
            {%- for key, value in args | dictsort(case_sensitive=true) -%}
                {{- (key | tojson) ~ ":" ~ (value | tojson(separators=(",", ":"))) -}}
                {%- if not loop.last -%},{%- endif -%}
            {%- endfor -%}
            {{- "}}<|end_message|>" -}}
        {%- endfor -%}
    {%- endif -%}
    {%- if message.role == "assistant" -%}
        {{- "<|content_model_end_sampling|>" -}}
    {%- endif -%}
    {%- endset -%}
    {%- if message.role == "assistant" -%}
        {%- generation -%}{{- rendered -}}{%- endgeneration -%}
    {%- else -%}
        {{- rendered -}}
    {%- endif -%}
{%- endfor -%}
{%- if not state.effort_emitted -%}
    {{- emit_thinking_effort() -}}
{%- endif -%})jinja";

        resources.text.tokenizer_json         = tokenizer_json;
        resources.text.tokenizer_config_json  = tokenizer_config_json;
        resources.text.chat_template_jinja    = chat_template_jinja;
        resources.text.generation_config_json = generation_config_json;
        resources.processor_config_json       = processor_config_json;
        inkling::parse_resources(resources, static_cast<std::uint32_t>(next_id));
    }
};

std::string weather_tool(bool strict = false) {
    return Json{{"type", "function"},
                {"function",
                 {{"name", "get_weather"},
                  {"description", "Get weather"},
                  {"strict", strict},
                  {"parameters",
                   {{"type", "object"},
                    {"properties", {{"city", {{"type", "string"}}}}},
                    {"required", {"city"}},
                    {"additionalProperties", false}}}}}}
        .dump();
}

std::string calc_tool(bool strict = false) {
    return Json{{"type", "function"},
                {"function",
                 {{"name", "add"},
                  {"description", "Add two integers"},
                  {"strict", strict},
                  {"parameters",
                   {{"type", "object"},
                    {"properties",
                     {{"a", {{"type", "integer"}}}, {"b", {{"type", "integer"}}}}},
                    {"required", {"a", "b"}},
                    {"additionalProperties", false}}}}}}
        .dump();
}

void test_chat_template(const inkling::Frontend& frontend, const text::Tokenizer& tokenizer,
                        const inkling::FramingTokens& framing) {
    // 1. Default high effort (0.9) and Developer -> System mapping.
    {
        PromptInput in;
        in.messages.push_back({.role = ChatRole::Developer, .parts = {{.text = "Be concise."}}});
        in.messages.push_back({.role = ChatRole::User, .parts = {{.text = "Hi"}}});
        const auto prepared = frontend.prepare(in);
        const auto& data    = inkling::PreparedPromptAccess::view(prepared);
        const std::string decoded = tokenizer.decode(data.token_ids);
        const std::string expected =
            "<|message_system|><|content_text|>Be concise.<|end_message|>"
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
            "<|message_user|><|content_text|>Hi<|end_message|>";
        require(decoded == expected, "default chat template mismatch: " + decoded);
        require(data.output.initial_block == inkling::InitialBlock::BetweenBlocks,
                "Inkling2 prompt must end between blocks");
        require(data.output.thinking_enabled, "default effort 0.9 must enable thinking");
    }

    // 2. All 7 reasoning effort levels + enable_thinking=false + conflict check.
    const std::vector<std::pair<ReasoningEffort, std::string>> efforts = {
        {ReasoningEffort::None, "0"},      {ReasoningEffort::Minimal, "0.1"},
        {ReasoningEffort::Low, "0.2"},     {ReasoningEffort::Medium, "0.7"},
        {ReasoningEffort::High, "0.9"},    {ReasoningEffort::XHigh, "0.99"},
        {ReasoningEffort::Max, "0.99"},
    };
    for (const auto& [level, label] : efforts) {
        PromptInput in;
        in.messages.push_back({.role = ChatRole::User, .parts = {{.text = "q"}}});
        in.options.reasoning_effort = level;
        const auto prepared         = frontend.prepare(in);
        const auto& data            = inkling::PreparedPromptAccess::view(prepared);
        const std::string decoded   = tokenizer.decode(data.token_ids);
        require(decoded.find("Thinking effort level: " + label + "<|end_message|>") !=
                    std::string::npos,
                "effort level mismatch for " + label + ": " + decoded);
        require(data.output.thinking_enabled == (level != ReasoningEffort::None),
                "thinking_enabled flag mismatch");
    }
    {
        PromptInput in;
        in.messages.push_back({.role = ChatRole::User, .parts = {{.text = "q"}}});
        in.options.enable_thinking = false;
        const auto prepared        = frontend.prepare(in);
        const auto& data           = inkling::PreparedPromptAccess::view(prepared);
        require(tokenizer.decode(data.token_ids).find("Thinking effort level: 0<|end_message|>") !=
                    std::string::npos,
                "enable_thinking=false did not map to effort 0");
        require(!data.output.thinking_enabled, "enable_thinking=false must disable thinking");
    }
    {
        PromptInput in;
        in.messages.push_back({.role = ChatRole::User, .parts = {{.text = "q"}}});
        in.options.enable_thinking  = false;
        in.options.reasoning_effort = ReasoningEffort::High;
        bool threw_conflict         = false;
        try {
            (void)frontend.prepare(in);
        } catch (const RequestError& err) {
            threw_conflict = (err.kind() == RequestErrorKind::ConflictingPromptOptions);
        }
        require(threw_conflict, "enable_thinking=false + High must throw ConflictingPromptOptions");
    }

    // 3. Tool declarations, multi-turn tool call + tool error block.
    {
        PromptInput in;
        in.options.tool_jsons = {weather_tool(false)};
        in.messages.push_back({.role = ChatRole::User, .parts = {{.text = "Weather?"}}});
        ChatMessage assistant;
        assistant.role              = ChatRole::Assistant;
        assistant.reasoning_content = "check weather";
        assistant.tool_calls        = {{"call_1", "get_weather", R"({"city":"SF"})"}};
        in.messages.push_back(std::move(assistant));
        ChatMessage tool_err;
        tool_err.role                 = ChatRole::Tool;
        tool_err.tool_call_id         = "call_1";
        tool_err.tool_result_is_error = true;
        tool_err.parts                = {{.text = "network timeout"}};
        in.messages.push_back(std::move(tool_err));

        const auto prepared       = frontend.prepare(in);
        const auto& data          = inkling::PreparedPromptAccess::view(prepared);
        const std::string decoded = tokenizer.decode(data.token_ids);
        require(decoded.find("<|message_system|>tool_declare<|content_xml|>") == 0,
                "missing tool_declare block");
        require(decoded.find("<|message_model|>get_weather<|content_invoke_tool_json|>"
                             R"({"name":"get_weather","args":{"city":"SF"}}<|end_message|>)"
                             "<|content_model_end_sampling|>") != std::string::npos,
                "assistant tool_call formatting mismatch: " + decoded);
        require(decoded.find("<|message_tool|>get_weather<|content_tool_error|>network "
                             "timeout<|end_message|>") != std::string::npos,
                "tool_result_is_error must use <|content_tool_error|>: " + decoded);
    }

    // 4. Literal special tokens in user text must NEVER become framing token IDs.
    {
        PromptInput in;
        in.messages.push_back(
            {.role  = ChatRole::User,
             .parts = {{.text = "literal <|message_model|> <|end_message|>"}}});
        const auto prepared = frontend.prepare(in);
        const auto& ids     = inkling::PreparedPromptAccess::view(prepared).token_ids;
        require(std::count(ids.begin(), ids.end(), framing.message_model) == 0,
                "literal <|message_model|> became framing token");
        // Only two real <|end_message|> tokens: effort system block + user block.
        require(std::count(ids.begin(), ids.end(), framing.end_message) == 2,
                "literal <|end_message|> became framing token");
    }

    // 5. ContinueFinalAssistant leaves content block open.
    {
        PromptInput in;
        in.options.continuation = PromptContinuationMode::ContinueFinalAssistant;
        in.messages.push_back({.role = ChatRole::User, .parts = {{.text = "Count"}}});
        in.messages.push_back({.role              = ChatRole::Assistant,
                               .parts             = {{.text = "1, 2,"}},
                               .reasoning_content = "easy"});
        const auto prepared       = frontend.prepare(in);
        const auto& data          = inkling::PreparedPromptAccess::view(prepared);
        const std::string decoded = tokenizer.decode(data.token_ids);
        require(decoded.ends_with("<|message_model|><|content_thinking|>easy<|end_message|>"
                                  "<|message_model|><|content_text|>1, 2,"),
                "continuation prompt mismatch: " + decoded);
        require(data.output.initial_block == inkling::InitialBlock::Text,
                "continuation must start in InitialBlock::Text");
        require(data.output.continuation_content == "1, 2,",
                "continuation_content mismatch");
    }
}

void test_streaming_decoder(const inkling::Frontend& frontend, const text::Tokenizer& tokenizer,
                            const inkling::FramingTokens& framing) {
    PromptInput in;
    in.options.tool_jsons = {weather_tool(false), calc_tool(false)};
    in.messages.push_back({.role = ChatRole::User, .parts = {{.text = "Do both"}}});
    const auto prompt = frontend.prepare(in);

    // 1. Multi-block streaming: thinking -> text -> parallel tool calls (width 1, 3, full).
    const std::string wire =
        "<|message_model|><|content_thinking|>思考中: UTF-8 你好!<|end_message|>"
        "<|message_model|><|content_text|>Checking now.<|end_message|>"
        "<|message_model|>get_weather<|content_invoke_tool_json|>"
        R"({"name":"get_weather","args":{"city":"Tokyo"}})"
        "<|end_message|>"
        "<|message_model|>add<|content_invoke_tool_json|>"
        R"({"name":"add","args":{"a":2,"b":3}})"
        "<|end_message|><|content_model_end_sampling|>";
    const auto ids = tokenizer.encode(wire);

    for (const std::size_t width : {std::size_t(1), std::size_t(3), ids.size()}) {
        auto session = frontend.make_output_session(
            prompt, {}, {}, {}, {}, {.constraints = ToolConstraintMode::Automatic});
        CollectedOutput out;
        for (std::size_t i = 0; i < ids.size(); i += width) {
            const auto d = commit_round(
                session, std::span(ids).subspan(i, std::min(width, ids.size() - i)), out);
            if (d.finished()) break;
        }
        require(out.reasoning == "思考中: UTF-8 你好!",
                "reasoning mismatch: " + out.reasoning);
        require(out.content == "Checking now.", "content mismatch: " + out.content);
        require(out.calls.size() == 2, "expected 2 tool calls");
        require(out.calls[0].name == "get_weather" &&
                    Json::parse(out.calls[0].arguments_json) == Json{{"city", "Tokyo"}} &&
                    out.calls[0].status == ToolCallStatus::Complete,
                "first tool call mismatch");
        require(out.calls[1].name == "add" &&
                    Json::parse(out.calls[1].arguments_json) == Json{{"a", 2}, {"b", 3}} &&
                    out.calls[1].status == ToolCallStatus::Complete,
                "second tool call mismatch");
        require(out.calls == session.take_tool_calls(), "take_tool_calls mismatch");
        require(!session.output_error().has_value(), "unexpected output error");
    }

    // 2. Speculative mid-batch thinking budget enforcement + discard rollback.
    {
        auto session = frontend.make_output_session(
            prompt, {}, {}, {.budget = 2}, {}, {.constraints = ToolConstraintMode::Automatic});
        session.validate_generation_capacity(6); // budget(2) + control(3) + post(1)
        bool rejected_small = false;
        try {
            session.validate_generation_capacity(5);
        } catch (const std::invalid_argument&) { rejected_small = true; }
        require(rejected_small, "validate_generation_capacity must require control+1 tokens");

        // Feed speculative batch [<|message_model|>, <|content_thinking|>, 'a', 'b', 'c', 'd'].
        std::vector<TokenId> batch = {framing.message_model, framing.content_thinking};
        for (const char* s : {"a", "b", "c", "d"}) {
            const auto t = tokenizer.encode(s);
            require(t.size() == 1, "expected 1-token ascii");
            batch.push_back(t.front());
        }
        // Preview & discard first to verify zero state leak.
        const auto disc = session.preview_model(batch, 100, FinishReason::OutputLimit);
        require(disc.accepted_tokens == 4 &&
                    disc.continuation == runtime::ContinuationAction::ApplyTargetControl,
                "speculative batch did not stop at thinking budget");
        session.discard_preview();
        require(session.thinking_stats().model_thinking_tokens == 0 &&
                    session.pending_control_tokens().empty(),
                "discard_preview leaked thinking state");

        CollectedOutput out;
        const auto d = commit_round(session, batch, out);
        require(d.accepted_tokens == 4 &&
                    d.continuation == runtime::ContinuationAction::ApplyTargetControl,
                "budget did not request target control");
        const auto ctrl = std::vector<TokenId>(session.pending_control_tokens().begin(),
                                               session.pending_control_tokens().end());
        require(ctrl.size() == 3 && ctrl[0] == framing.end_message &&
                    ctrl[1] == framing.message_model && ctrl[2] == framing.content_text,
                "unexpected Inkling2 control token sequence");
        const auto cd = session.preview_control(ctrl, 100);
        require(cd.accepted_tokens == 3 && !cd.finished(), "preview_control mismatch");
        out.append(session.commit_preview());

        const auto tail = tokenizer.encode("final answer<|end_message|><|content_model_end_sampling|>");
        const auto td   = commit_round(session, tail, out);
        require(td.finish_reason == FinishReason::StopToken, "expected StopToken");
        require(out.reasoning == "ab" && out.content == "final answer",
                "forced thinking close output mismatch: r=" + out.reasoning + " c=" + out.content);
        const auto stats = session.thinking_stats();
        require(stats.applied && stats.model_thinking_tokens == 2 && stats.injected_tokens == 3,
                "thinking_stats mismatch");
    }

    // 3. StopString split across tokens on Content channel.
    {
        auto session = frontend.make_output_session(
            prompt, {.strings = {{.text = "STOP", .channel = OutputChannel::Content}}}, {}, {}, {},
            {.constraints = ToolConstraintMode::Automatic});
        CollectedOutput out;
        (void)commit_round(
            session,
            tokenizer.encode("<|message_model|><|content_thinking|>STOP<|end_message|>"
                             "<|message_model|><|content_text|>hello ST"),
            out);
        const auto d = commit_round(session, tokenizer.encode("OP world"), out);
        require(d.finish_reason == FinishReason::StopString, "expected StopString");
        require(session.matched_stop_string() == "STOP", "matched_stop_string mismatch");
        require(out.reasoning == "STOP" && out.content == "hello ",
                "StopString channel isolation/trim mismatch: r=" + out.reasoning +
                    " c=" + out.content);
    }

    // 4. Raw mode publishes framing tokens and EOS verbatim.
    {
        auto session = frontend.make_output_session(
            prompt, {}, {.raw = true}, {}, {}, {.constraints = ToolConstraintMode::Automatic});
        CollectedOutput out;
        const std::string sample =
            "<|message_model|><|content_text|>raw<|end_message|><|content_model_end_sampling|>";
        (void)commit_round(session, tokenizer.encode(sample), out);
        require(out.reasoning.empty() && out.content == sample && out.calls.empty(),
                "raw mode mismatch: " + out.content);
    }
}

void test_xgrammar_constraints(const inkling::Frontend& frontend, const text::Tokenizer& tokenizer,
                               const inkling::FramingTokens& framing) {
    const std::size_t vocab = tokenizer.vocab_size();
    const std::array<TokenId, 5> all_framing = {
        framing.message_model, framing.content_thinking, framing.content_text,
        framing.content_invoke_tool_json, framing.end_message};

    // 1. JSON Schema wrapped in Inkling blocks + verify framing tokens NEVER leak into JSON string.
    {
        PromptInput in;
        in.messages.push_back({.role = ChatRole::User, .parts = {{.text = "JSON"}}});
        const auto prompt = frontend.prepare(in);
        OutputConstraint schema{
            .kind   = OutputConstraintKind::JsonSchema,
            .source = R"({"type":"object","properties":{"msg":{"type":"string"}},"required":["msg"],"additionalProperties":false})",
        };
        auto session = frontend.make_output_session(prompt, {}, {}, {}, schema, {});
        require(session.constrained(), "JSON schema session must be constrained");

        // Initially only <|message_model|> is allowed!
        auto mask = next_mask(session, vocab);
        require(mask_allows(mask, framing.message_model) &&
                    !mask_allows(mask, framing.content_text) &&
                    !mask_allows(mask, framing.content_thinking),
                "initial mask must require <|message_model|>");

        CollectedOutput out;
        // Walk through a thinking block, then open constrained text block and enter JSON string.
        const auto prefix = tokenizer.encode(
            "<|message_model|><|content_thinking|>plan<|end_message|>"
            "<|message_model|><|content_text|>{\"msg\":\"hi");
        (void)commit_round(session, prefix, out);

        // Inside the JSON string value, NONE of the 5 framing tokens may be allowed!
        mask = next_mask(session, vocab);
        for (const TokenId fid : all_framing) {
            require(!mask_allows(mask, fid), "framing token leaked into JSON string value!");
        }

        const auto rest = tokenizer.encode("\"}<|end_message|><|content_model_end_sampling|>");
        const auto d    = commit_round(session, rest, out);
        require(d.finish_reason == FinishReason::StopToken, "expected StopToken after JSON");
        require(out.reasoning == "plan" && out.content == R"({"msg":"hi"})",
                "constrained JSON output mismatch: " + out.content);
        const auto obs = session.constraint_observation();
        require(obs && obs->complete && obs->branch == ConstraintOutputBranch::Content,
                "constraint_observation mismatch for JSON");
    }

    // 2. Regex constraint: framing tokens must never leak into `.*`.
    {
        PromptInput in;
        in.messages.push_back({.role = ChatRole::User, .parts = {{.text = "Regex"}}});
        const auto prompt = frontend.prepare(in);
        OutputConstraint regex{.kind = OutputConstraintKind::Regex, .source = "ok:.*"};
        auto session = frontend.make_output_session(prompt, {}, {}, {}, regex, {});
        CollectedOutput out;
        (void)commit_round(session,
                           tokenizer.encode("<|message_model|><|content_text|>ok:hello"), out);
        const auto mask = next_mask(session, vocab);
        require(mask_allows(mask, framing.end_message), "end_message must close regex text block");
        require(!mask_allows(mask, framing.message_model) &&
                    !mask_allows(mask, framing.content_thinking) &&
                    !mask_allows(mask, framing.content_text) &&
                    !mask_allows(mask, framing.content_invoke_tool_json),
                "non-closing framing token leaked into Regex .*");
    }

    // 3. ToolChoice::Required + strict schema + thinking budget (2-token control_to_header).
    {
        PromptInput in;
        in.options.tool_jsons = {calc_tool(/*strict=*/true)};
        in.messages.push_back({.role = ChatRole::User, .parts = {{.text = "2+3"}}});
        const auto prompt = frontend.prepare(in);
        auto session      = frontend.make_output_session(
            prompt, {}, {}, {.budget = 2}, {},
            {.mode = ToolChoiceMode::Required, .parallel = false});
        CollectedOutput out;
        const auto t1 = tokenizer.encode("a");
        const auto t2 = tokenizer.encode("b");
        require(t1.size() == 1 && t2.size() == 1, "expected 1-token ascii");
        const std::vector<TokenId> think = {framing.message_model, framing.content_thinking,
                                            t1.front(), t2.front()};
        const auto d                     = commit_round(session, think, out);
        require(d.continuation == runtime::ContinuationAction::ApplyTargetControl,
                "required tool + thinking budget must trigger control");
        const auto ctrl = std::vector<TokenId>(session.pending_control_tokens().begin(),
                                               session.pending_control_tokens().end());
        require(ctrl.size() == 2 && ctrl[0] == framing.end_message &&
                    ctrl[1] == framing.message_model,
                "required tool control must be 2 tokens {<|end_message|>, <|message_model|>}");
        (void)session.preview_control(ctrl, 100);
        out.append(session.commit_preview());

        // After forced close under Required + budget, <|content_thinking|> and <|content_text|>
        // must both be masked out — only tool name "add" can proceed!
        const auto mask = next_mask(session, vocab);
        require(!mask_allows(mask, framing.content_thinking) &&
                    !mask_allows(mask, framing.content_text),
                "required tool mask after budget close must forbid thinking/text");

        const auto call = tokenizer.encode(
            R"(add<|content_invoke_tool_json|>{"name":"add","args":{"a":2,"b":3}}<|end_message|><|content_model_end_sampling|>)");
        const auto td = commit_round(session, call, out);
        require(td.finish_reason == FinishReason::StopToken, "expected StopToken after tool call");
        require(out.calls.size() == 1 && out.calls[0].name == "add" &&
                    Json::parse(out.calls[0].arguments_json) == Json{{"a", 2}, {"b", 3}},
                "required strict tool call mismatch");
        const auto obs = session.constraint_observation();
        require(obs && obs->complete && obs->branch == ConstraintOutputBranch::Tools,
                "required tool branch observation mismatch");
    }

    // 4. ToolChoice::None forbids <|content_invoke_tool_json|>.
    {
        PromptInput in;
        in.options.tool_jsons = {weather_tool(false)};
        in.messages.push_back({.role = ChatRole::User, .parts = {{.text = "No tools"}}});
        const auto prompt = frontend.prepare(in);
        auto session      = frontend.make_output_session(
            prompt, {}, {}, {}, {}, {.mode = ToolChoiceMode::None});
        CollectedOutput out;
        (void)commit_round(session, tokenizer.encode("<|message_model|>"), out);
        const auto mask = next_mask(session, vocab);
        require(mask_allows(mask, framing.content_text) &&
                    mask_allows(mask, framing.content_thinking) &&
                    !mask_allows(mask, framing.content_invoke_tool_json),
                "ToolChoice::None must forbid tool invocation token");
    }

    // 5. Continuation + JSON Schema advances grammar via AcceptString.
    {
        PromptInput in;
        in.options.continuation = PromptContinuationMode::ContinueFinalAssistant;
        in.messages.push_back({.role = ChatRole::User, .parts = {{.text = "JSON"}}});
        in.messages.push_back({.role = ChatRole::Assistant, .parts = {{.text = R"({"x":)"}}});
        const auto prompt = frontend.prepare(in);
        OutputConstraint schema{
            .kind   = OutputConstraintKind::JsonSchema,
            .source = R"({"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false})",
        };
        auto session = frontend.make_output_session(prompt, {}, {}, {}, schema, {});
        CollectedOutput out;
        const auto d = commit_round(
            session, tokenizer.encode("42}<|end_message|><|content_model_end_sampling|>"), out);
        require(d.finish_reason == FinishReason::StopToken && out.content == "42}",
                "continuation JSON schema mismatch: " + out.content);
    }
}

void test_image_processor(const inkling::Frontend& frontend, const text::Tokenizer& tokenizer,
                          const inkling::FramingTokens& framing) {
    // 1. Non-multiple-of-40 RGBA PNG with alpha=0, 128, 255 spanning checkerboard cells.
    const int w = 53, h = 41; // merged = ceil(41/40)=2 x ceil(53/40)=2 -> 4 tokens, 16 native 20x20
    std::vector<std::uint8_t> rgba(static_cast<std::size_t>(w) * h * 4, 255);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const std::size_t idx = (static_cast<std::size_t>(y) * w + x) * 4;
            rgba[idx + 0]         = 100;
            rgba[idx + 1]         = 50;
            rgba[idx + 2]         = 200;
            rgba[idx + 3]         = (y == 0 && x == 0) ? 0      // checkerboard cell (0,0) -> 204
                                  : (y == 0 && x == 8) ? 0      // checkerboard cell (1,0) -> 153
                                  : (y == 8 && x == 8) ? 128    // checkerboard cell (1,1) -> 204
                                                       : 255;   // opaque
        }
    }
    const auto png_bytes = make_png(w, h, 4, rgba);

    PromptInput in;
    ChatMessage user;
    user.role  = ChatRole::User;
    user.parts = {
        {.kind = MessagePartKind::Text, .text = "Describe:"},
        {.kind = MessagePartKind::Media, .media = {.kind = MediaKind::Image, .bytes = png_bytes}},
    };
    in.messages.push_back(std::move(user));

    const auto prepared = frontend.prepare(in);
    const auto& data    = inkling::PreparedPromptAccess::view(prepared);
    require(data.identity.media.size() == 1 && data.media_payloads.size() == 1,
            "expected 1 prepared image");
    const auto& item = data.identity.media[0].item;
    require(item.grid.temporal == 1 && item.grid.height == 4 && item.grid.width == 4 &&
                item.patch_count == 16,
            "unexpected native patch grid for 53x41 image");
    require(item.token_spans.size() == 1 && item.token_spans[0].count == 4,
            "expected 4 image pad tokens for 53x41 image");
    const auto span = item.token_spans[0];
    require(data.token_ids[span.begin - 1] == framing.content_image,
            "image pad span must immediately follow <|content_image|>");
    for (std::size_t i = 0; i < span.count; ++i) {
        require(data.token_ids[span.begin + i] == framing.image_pad,
                "token inside media span must be image_pad");
    }
    require(data.token_ids[span.begin + span.count] == framing.end_message,
            "image pad span must immediately precede <|end_message|>");
    require(frontend.count_tokens(in) == data.token_ids.size(),
            "count_tokens must match prepared token count for images");

    // Verify exact bit-level patch pixels: checkerboard alpha + opaque + -1 padding.
    const auto patches = data.media_payloads[0]->span();
    require(patches.size() == 16U * 1200U, "unexpected BF16 patch buffer length");
    const auto patch_at = [&](int ny, int nx, int py, int px, int c) {
        const std::size_t row = static_cast<std::size_t>(ny) * 4U + static_cast<std::size_t>(nx);
        return patches[row * 1200U + (static_cast<std::size_t>(py) * 20U + px) * 3U + c];
    };
    // (0,0): alpha=0 on cell (0,0) -> bg=204 for all RGB channels.
    for (int c = 0; c < 3; ++c) {
        require(patch_at(0, 0, 0, 0, c) == ref_normalize_bf16(204.0F, c),
                "checkerboard (0,0) alpha=0 mismatch");
    }
    // (0,8): alpha=0 on cell (1,0) -> bg=153 for all RGB channels.
    for (int c = 0; c < 3; ++c) {
        require(patch_at(0, 0, 0, 8, c) == ref_normalize_bf16(153.0F, c),
                "checkerboard (0,8) alpha=0 mismatch");
    }
    // (8,8): alpha=128 on cell (1,1) -> bg=204, blended = (rgb*128 + 204*127 + 127) / 255.
    const std::array<int, 3> rgb{100, 50, 200};
    for (int c = 0; c < 3; ++c) {
        const int blended = (rgb[c] * 128 + 204 * 127 + 127) / 255;
        require(patch_at(0, 0, 8, 8, c) == ref_normalize_bf16(static_cast<float>(blended), c),
                "checkerboard (8,8) alpha=128 mismatch");
    }
    // (40,52) -> ny=2, nx=2, py=0, px=12: last valid pixel (opaque).
    for (int c = 0; c < 3; ++c) {
        require(patch_at(2, 2, 0, 12, c) == ref_normalize_bf16(static_cast<float>(rgb[c]), c),
                "valid pixel (40,52) mismatch");
    }
    // (40,53) -> ny=2, nx=2, py=0, px=13: out-of-bounds -> -1 padding!
    for (int c = 0; c < 3; ++c) {
        require(patch_at(2, 2, 0, 13, c) == ref_normalize_bf16(-1.0F, c),
                "padding pixel (40,53) mismatch");
    }
    // (41,0) -> ny=2, nx=0, py=1, px=0: out-of-bounds -> -1 padding!
    for (int c = 0; c < 3; ++c) {
        require(patch_at(2, 0, 1, 0, c) == ref_normalize_bf16(-1.0F, c),
                "padding pixel (41,0) mismatch");
    }

    // 2. Second prepare hits MediaPreprocessCache; tool messages also support images.
    PromptInput tool_img;
    tool_img.messages.push_back({.role = ChatRole::User, .parts = {{.text = "screenshot"}}});
    ChatMessage tool_msg;
    tool_msg.role  = ChatRole::Tool;
    tool_msg.parts = {
        {.kind = MessagePartKind::Media, .media = {.kind = MediaKind::Image, .bytes = png_bytes}},
    };
    tool_img.messages.push_back(std::move(tool_msg));
    const auto prepared2 = frontend.prepare(tool_img);
    require(prepared2.preparation_stats().media_cache_hits == 1 &&
                prepared2.preparation_stats().media_cache_misses == 0,
            "second prepare must hit media cache");
    require(tokenizer.decode(inkling::PreparedPromptAccess::view(prepared2).token_ids)
                    .find("<|message_tool|><|content_image|><|unused_200054|>") != std::string::npos,
            "tool image rendering mismatch");

    // 3. Reject video media cleanly.
    PromptInput bad_video;
    bad_video.messages.push_back(
        {.role  = ChatRole::User,
         .parts = {{.kind  = MessagePartKind::Media,
                    .media = {.kind = MediaKind::Video, .bytes = {1, 2, 3}}}}});
    bool rejected_video = false;
    try {
        (void)frontend.prepare(bad_video);
    } catch (const RequestError& err) {
        rejected_video = (err.kind() == RequestErrorKind::InvalidMedia);
    }
    require(rejected_video, "Inkling2 must reject video media");
}

void test_real_checkpoint_if_present() {
    const std::filesystem::path dir = "/home/neroued/models/llm/inkling/Inkling-2-Mini/base-hf";
    if (!std::filesystem::exists(dir / "tokenizer.json")) {
        std::cout << "Skipping real Inkling-2-Mini checkpoint frontend check (not present)\n";
        return;
    }
    const std::string tok_json  = read_file(dir / "tokenizer.json");
    const std::string tok_cfg   = read_file(dir / "tokenizer_config.json");
    const std::string tmpl      = read_file(dir / "chat_template.jinja");
    const std::string gen_cfg   = read_file(dir / "generation_config.json");
    const std::string proc_cfg  = read_file(dir / "processor_config.json");

    inkling::FrontendResources res;
    res.text.tokenizer_json         = tok_json;
    res.text.tokenizer_config_json  = tok_cfg;
    res.text.chat_template_jinja    = tmpl;
    res.text.generation_config_json = gen_cfg;
    res.processor_config_json       = proc_cfg;
    inkling::parse_resources(res, 200064);

    require(res.text.public_token_count == 200058, "real public_token_count mismatch");
    require(res.framing.message_model == 200001 && res.framing.content_text == 200004 &&
                res.framing.content_image == 200005 && res.framing.content_thinking == 200008 &&
                res.framing.end_message == 200010 &&
                res.framing.content_invoke_tool_json == 200049 &&
                res.framing.image_pad == 200054,
            "real Inkling-2-Mini framing token IDs mismatch");
    require(res.text.stop_token_ids == std::vector<TokenId>{200006},
            "real Inkling-2-Mini stop_token_ids mismatch");

    auto frontend = inkling::make_frontend(res, {.vision_enabled = true, .max_context = 4096});
    test_chat_template(frontend, *res.text.tokenizer, res.framing);
    test_streaming_decoder(frontend, *res.text.tokenizer, res.framing);
    test_xgrammar_constraints(frontend, *res.text.tokenizer, res.framing);
    test_image_processor(frontend, *res.text.tokenizer, res.framing);
    std::cout << "Real Inkling-2-Mini frontend suite passed!\n";
}

} // namespace

int main() {
    SyntheticFixture fixture;
    auto frontend =
        inkling::make_frontend(fixture.resources, {.vision_enabled = true, .max_context = 4096});
    test_chat_template(frontend, *fixture.resources.text.tokenizer, fixture.resources.framing);
    test_streaming_decoder(frontend, *fixture.resources.text.tokenizer, fixture.resources.framing);
    test_xgrammar_constraints(frontend, *fixture.resources.text.tokenizer,
                              fixture.resources.framing);
    test_image_processor(frontend, *fixture.resources.text.tokenizer, fixture.resources.framing);
    std::cout << "Synthetic Inkling2 frontend suite passed!\n";

    test_real_checkpoint_if_present();
    return 0;
}
