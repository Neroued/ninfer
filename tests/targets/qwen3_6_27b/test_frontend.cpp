#include "targets/qwen3_6_27b_rtx5090/impl/frontend/frontend.h"
#include "targets/qwen3_6_27b_rtx5090/impl/load/bindings.h"
#include "targets/qwen3_6_27b_rtx5090/impl/schedule/schedule.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using Frontend          = ninfer::targets::qwen3_6_27b_rtx5090::detail::Frontend;
using FrontendFactory   = ninfer::targets::qwen3_6_27b_rtx5090::detail::FrontendFactory;
using FrontendResources = ninfer::targets::qwen3_6_27b_rtx5090::detail::FrontendResources;
using PublishedOutput   = ninfer::targets::qwen3_6_27b_rtx5090::detail::PublishedOutput;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
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

FrontendResources resources() {
    FrontendResources result;
    result.tokenizer_json =
        nlohmann::json{
            {"model",
             {{"type", "BPE"},
              {"vocab", {{"x", 0}, {"ä", 10}, {"¸", 11}, {"Ń", 12}}},
              {"merges", nlohmann::json::array()}}},
            {"added_tokens",
             nlohmann::json::array(
                 {added(1, "helloST"), added(2, "OPtail"), added(3, "thought</thi"),
                  added(4, "nk>\n\nanswer"), added(6, "<eos>", true), added(30, "user\n"),
                  added(31, "assistant\n"), added(32, "\n"), added(248045, "<|im_start|>", true),
                  added(248046, "<|im_end|>", true), added(248053, "<|vision_start|>", true),
                  added(248054, "<|vision_end|>", true), added(248056, "<|image_pad|>", true),
                  added(248057, "<|video_pad|>", true), added(248068, "<think>"),
                  added(248069, "</think>")})}}
            .dump();
    result.tokenizer_config_json  = R"({"add_bos_token":false,"add_prefix_space":false})";
    result.chat_template_jinja    = "enable_thinking <|im_start|> <|vision_start|> vision_start";
    result.generation_config_json = R"({"eos_token_id":[6]})";
    result.preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"rescale_factor":0.00392156862745098,"size":{"shortest_edge":65536,"longest_edge":16777216}})";
    result.video_preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"rescale_factor":0.00392156862745098,"size":{"shortest_edge":4096,"longest_edge":25165824},"fps":2,"min_frames":4,"max_frames":768})";
    return result;
}

PublishedOutput resolve(ninfer::targets::qwen3_6_27b_rtx5090::detail::OutputSession& session,
                        ninfer::targets::qwen3_6_27b_rtx5090::detail::StagedText staged,
                        std::uint32_t count, ninfer::runtime::RoundContinuation continuation,
                        ninfer::FinishReason reason,
                        std::optional<ninfer::runtime::StagedChoiceId> selected = std::nullopt) {
    const ninfer::runtime::StagedChoiceId choice =
        selected.value_or(staged.choice_for(count, continuation, reason));
    auto plan = session.prepare_commit(
        std::move(staged), ninfer::runtime::OutputResolution{.committed_tokens = count,
                                                             .continuation     = continuation,
                                                             .finish_reason    = reason,
                                                             .staged_choice    = choice});
    return session.commit(std::move(plan));
}

std::string channel_text(const PublishedOutput& output, ninfer::OutputChannel channel) {
    std::string result;
    for (const ninfer::OutputDelta& delta : output) {
        if (delta.channel == channel) { result += delta.text; }
    }
    return result;
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
    failures += check(text_data.identity.assistant_content_boundary == 7 &&
                          text_data.starts_in_reasoning && !text_data.has_media(),
                      "text frontend did not preserve prefix/thinking identity");
    failures +=
        check(text_data.position_axis(0).back() == 8 && text_data.position_axis(1).back() == 8 &&
                  text_data.position_axis(2).back() == 8,
              "text frontend did not construct axis-major positions");

    std::vector<std::uint8_t> ppm;
    const std::string header = "P6\n32 32\n255\n";
    for (const char byte : header) {
        ppm.push_back(static_cast<std::uint8_t>(static_cast<unsigned char>(byte)));
    }
    for (int index = 0; index < 32 * 32; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = std::move(ppm);
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
        failures += check(item.grid.temporal == 1 && item.grid.height == 16 &&
                              item.grid.width == 16 && item.patch_count == 256 &&
                              item.token_spans.size() == 1 && item.token_spans.front().count == 64,
                          "image frontend grid/patch/placeholder geometry is incorrect");
    }
    failures += check(
        prepared_data.patches.size() == 256 * 1536 && prepared_data.prepare.raw_patches == 256 &&
            prepared_data.prepare.vision_tokens == 64 && !prepared_data.identity.reusable,
        "image frontend did not own the expected patch payload and identity");
    failures += check(
        ninfer::targets::qwen3_6_27b_rtx5090::detail::schedule::vision_workspace_bytes(
            prepared_data) > 0,
        "image frontend metadata did not produce a Vision workspace plan");
    return failures;
}

int test_cross_round_stop(const Frontend& frontend) {
    auto prompt = frontend.prepare_tokens({0});
    ninfer::StopPolicy stop;
    stop.strings.push_back(ninfer::StopString{.text = "STOP"});
    auto session = frontend.make_output_session(prompt, stop);

    const auto first =
        resolve(session, session.stage(std::array<ninfer::TokenId, 1>{1}), 1,
                ninfer::runtime::RoundContinuation::Continue, ninfer::FinishReason::None);
    int failures = check(channel_text(first, ninfer::OutputChannel::Content) == "hello",
                         "cross-round stop did not retain the ambiguous suffix");

    auto staged = session.stage(std::array<ninfer::TokenId, 1>{2});
    std::optional<ninfer::runtime::StagedChoiceId> stop_choice;
    for (const auto& candidate : staged.candidates()) {
        if (candidate.finish_reason == ninfer::FinishReason::StopString) {
            stop_choice = candidate.id;
            failures +=
                check(candidate.committed_tokens == 1 && candidate.decoded_byte_cut_order == 5,
                      "stop-string candidate did not map to the exact token/byte cut");
            break;
        }
    }
    failures += check(stop_choice.has_value(), "cross-round stop candidate was not staged");
    if (stop_choice.has_value()) {
        const auto second =
            resolve(session, std::move(staged), 1, ninfer::runtime::RoundContinuation::Finish,
                    ninfer::FinishReason::StopString, stop_choice);
        failures += check(second.empty(), "stop marker or same-token suffix leaked to output");
    }
    return failures;
}

int test_reasoning_split(const Frontend& frontend) {
    auto prompt                                          = frontend.prepare_tokens({0});
    FrontendFactory::inspect(prompt).starts_in_reasoning = true;
    auto session                                         = frontend.make_output_session(prompt, {});
    const std::array<ninfer::TokenId, 2> tokens{3, 4};
    const auto output =
        resolve(session, session.stage(tokens), 2, ninfer::runtime::RoundContinuation::Finish,
                ninfer::FinishReason::OutputLimit);
    int failures = check(channel_text(output, ninfer::OutputChannel::Reasoning) == "thought",
                         "reasoning channel did not remove the close marker");
    failures += check(channel_text(output, ninfer::OutputChannel::Content) == "answer",
                      "content channel did not strip the post-thinking separator");
    return failures;
}

int test_utf8_and_hidden_eos(const Frontend& frontend) {
    auto prompt  = frontend.prepare_tokens({0});
    auto session = frontend.make_output_session(prompt, {});
    int failures = 0;
    for (const ninfer::TokenId token : {10, 11}) {
        const auto output =
            resolve(session, session.stage(std::array<ninfer::TokenId, 1>{token}), 1,
                    ninfer::runtime::RoundContinuation::Continue, ninfer::FinishReason::None);
        failures += check(output.empty(), "partial UTF-8 codepoint was published");
    }
    const auto complete =
        resolve(session, session.stage(std::array<ninfer::TokenId, 1>{12}), 1,
                ninfer::runtime::RoundContinuation::Continue, ninfer::FinishReason::None);
    failures += check(channel_text(complete, ninfer::OutputChannel::Content) == "中",
                      "UTF-8 codepoint was not published when complete");

    auto eos_prompt  = frontend.prepare_tokens({0});
    auto eos_session = frontend.make_output_session(eos_prompt, {});
    const auto eos =
        resolve(eos_session, eos_session.stage(std::array<ninfer::TokenId, 1>{6}), 1,
                ninfer::runtime::RoundContinuation::Finish, ninfer::FinishReason::StopToken);
    failures += check(eos.empty(), "default EOS token was published");

    auto raw_prompt  = frontend.prepare_tokens({0});
    auto raw_session = frontend.make_output_session(
        raw_prompt, {}, ninfer::OutputOptions{.raw = true, .preserve_special_tokens = false});
    const auto raw_eos =
        resolve(raw_session, raw_session.stage(std::array<ninfer::TokenId, 1>{6}), 1,
                ninfer::runtime::RoundContinuation::Finish, ninfer::FinishReason::StopToken);
    failures += check(channel_text(raw_eos, ninfer::OutputChannel::Content) == "<eos>",
                      "raw output did not preserve the terminal special token");
    return failures;
}

} // namespace

int main() {
    const FrontendResources owned = resources();
    const Frontend frontend       = FrontendFactory::create_component(owned);
    int failures                  = 0;
    failures += test_text_and_image_prepare(frontend);
    failures += test_cross_round_stop(frontend);
    failures += test_reasoning_split(frontend);
    failures += test_utf8_and_hidden_eos(frontend);
    return failures == 0 ? 0 : 1;
}
