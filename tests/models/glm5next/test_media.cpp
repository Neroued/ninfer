#include "chat_fixture.h"
#include "models/glm5next/frontend/prepared_prompt.h"
#include "models/glm5next/frontend/resources.h"
#include "models/glm5next/vision/control.h"
#include "models/common/frontend/digest.h"
#include "media/decode/decode.h"

#include <atomic>
#include <future>
#include <iostream>
#include <thread>

namespace {
using namespace ninfer;
using namespace glm_fixture;
namespace glm    = models::glm5next;
namespace common = models::frontend;

const auto fixtures = std::filesystem::path(NINFER_SOURCE_DIR) / "tests/models/glm5next/fixtures";

glm::Frontend make(const Json& processor, std::uint32_t max_context = 16384,
                   std::size_t live_bytes               = 16ULL << 20,
                   std::filesystem::path tokenizer_path = {}) {
    const auto tokenizer =
        read(tokenizer_path.empty() ? fixtures / "media-tokenizer.json" : tokenizer_path);
    const auto source  = read(fixtures / "chat_template.jinja");
    const auto tokens  = Json::parse(tokenizer);
    std::uint32_t rows = 0;
    for (const auto& id : tokens["model"]["vocab"].items())
        rows = std::max(rows, id.value().get<std::uint32_t>() + 1);
    for (const auto& id : tokens["added_tokens"])
        rows = std::max(rows, id["id"].get<std::uint32_t>() + 1);
    const auto eos = std::find_if(tokens["added_tokens"].begin(), tokens["added_tokens"].end(),
                                  [](const auto& t) { return t["content"] == "<|endoftext|>"; });
    const auto generation = Json{{"eos_token_id", (*eos)["id"]}}.dump();
    const auto config     = processor.dump();
    glm::FrontendResources resources{.text                  = {.tokenizer_json         = tokenizer,
                                                               .tokenizer_config_json  = "{}",
                                                               .chat_template_jinja    = source,
                                                               .generation_config_json = generation},
                                     .processor_config_json = config};
    glm::parse_resources(resources, rows);
    return glm::make_frontend(resources, {.max_context              = max_context,
                                          .media_cache_bytes        = live_bytes,
                                          .media_live_bytes         = live_bytes,
                                          .media_preprocess_threads = 2});
}

PromptInput media_prompt(const Json& test, const std::filesystem::path& root) {
    auto input = prompt(test.at("input"));
    for (const auto& media : test["media"]) {
        const auto bytes = read(root / media["file"].get<std::string>());
        input.messages.at(media["message"])
            .parts.at(media["part"])
            .media.bytes.assign(bytes.begin(), bytes.end());
    }
    return input;
}

void execution_input(const glm::PreparedPromptData& prompt) {
    const glm::VisionConfig config{24, 1024, 16, 4096, 3, 14, 2, 2, 4096, 10240, 1e-5F, 10.F};
    std::uint32_t capacity = 0;
    for (const auto& media : prompt.identity.media)
        for (const auto& span : media.item.token_spans)
            capacity = std::max(capacity, static_cast<std::uint32_t>(span.count));
    const auto input     = glm::vision::prepare_input(prompt, config, capacity);
    std::uint32_t begin  = 0;
    std::size_t consumed = 0;
    while (const auto plan = models::vision::plan_window(input->units, begin,
                                                         prompt.token_ids.size(), capacity, 2)) {
        std::size_t features = 0;
        for (std::uint32_t rank = 0; rank < 2; ++rank) {
            const auto batch =
                glm::vision::prepare_batch(input->items, input->payloads, *plan, rank, config);
            features += batch.features;
            require(batch.cu_seqlens.back() == int(batch.features * 4),
                    "encoder segments do not cover their features");
            std::size_t segment = 0;
            for (const auto& placement : plan->placements) {
                if (placement.rank != rank) continue;
                const auto& unit  = plan->units[placement.unit];
                const auto source = prompt.media_payloads[unit.item]->span().subspan(
                    unit.patch_begin * 1176, std::size_t(unit.patches) * 1176);
                require(std::equal(source.begin(), source.end(), batch.patches.at(segment).begin(),
                                   batch.patches.at(segment).end()),
                        "rank encoder input changed prepared patch values");
                ++segment;
            }
        }
        require(features == plan->features, "rank assignment lost visual features");
        consumed += features;
        begin = plan->consume_end;
    }
    require(consumed == prompt.prepare.vision_tokens, "bounded windows lost media tokens");
}

void golden(const std::filesystem::path& path, const std::filesystem::path& dump,
            const std::filesystem::path& tokenizer) {
    const auto fixture = Json::parse(read(path));
    const auto root    = path.parent_path();
    for (const auto& test : fixture["cases"]) {
        const auto name     = test["name"].get<std::string>();
        const auto f        = make(test["config"], 16384, 16ULL << 20, tokenizer);
        const auto input    = media_prompt(test, root);
        const auto prepared = f.prepare(input);
        const auto& data    = glm::PreparedPromptAccess::view(prepared);
        require(f.count_tokens(input) == data.token_ids.size(), name + ": count differs");
        require(data.token_ids == test["ids"].get<std::vector<TokenId>>(),
                name + ": expanded token IDs differ");
        require(data.identity.media.size() == test["expected"].size(),
                name + ": media count differs");
        for (std::size_t i = 0; i < data.identity.media.size(); ++i) {
            const auto& item     = data.identity.media[i].item;
            const auto& expected = test["expected"][i];
            require(Json::array({item.grid.temporal, item.grid.height, item.grid.width}) ==
                        expected["grid"],
                    name + ": grid differs");
            require(item.timestamps == expected["timestamps"].get<std::vector<double>>(),
                    name + ": timestamps differ");
            Json spans = Json::array();
            for (auto span : item.token_spans) spans.push_back({span.begin, span.count});
            require(spans == expected["spans"], name + ": spans differ");
            const auto& payload = *data.media_payloads[i];
            require(payload.features_per_patch == 1176 &&
                        payload.patch_elements == item.patch_count * 1176,
                    name + ": patch geometry differs");
            const auto bytes =
                std::span(reinterpret_cast<const std::uint8_t*>(payload.patches.get()),
                          payload.patch_elements * 2);
            if (!dump.empty()) {
                std::ofstream file(dump / (name + "-" + std::to_string(i) + ".bf16"),
                                   std::ios::binary);
                file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            }
            const auto digest = common::sha256_hex(common::sha256(bytes));
            require(digest == expected["patch_sha256"].get<std::string>(),
                    name + ": patch values differ, got " + digest);
            require(!glm::complete_input_frontier(data.prefix_view(),
                                                  item.token_spans.front().begin + 1),
                    name + ": media-internal frontier allowed");
        }
        const auto cached = f.prepare(input);
        const auto& hit   = glm::PreparedPromptAccess::view(cached);
        require(
            glm::input_prefix_equal(data.prefix_view(), hit.prefix_view(), data.token_ids.size()) &&
                hit.token_ids == data.token_ids &&
                cached.preparation_stats().media_cache_hits == data.identity.media.size(),
            name + ": cached input differs");
        for (std::size_t i = 0; i < data.media_payloads.size(); ++i)
            require(data.media_payloads[i] == hit.media_payloads[i],
                    name + ": cached payload not retained");
        execution_input(data);
        std::cout << name << " passed\n";
    }
}

template <class Action>
void request_error(Action action, RequestErrorKind kind) {
    try {
        action();
    } catch (const RequestError& e) {
        require(e.kind() == kind, "wrong request error kind: " + std::string(e.what()));
        return;
    }
    throw std::runtime_error("expected request rejection");
}

void behavior(const std::filesystem::path& path, const std::filesystem::path& tokenizer) {
    const auto fixture          = Json::parse(read(path));
    const auto& test            = fixture["cases"][0];
    const auto config           = test["config"];
    const auto root             = path.parent_path();
    auto input                  = media_prompt(test, root);
    const auto f                = make(config, 16384, 16ULL << 20, tokenizer);
    input.context_cache.markers = {{.after_message_count = 1,
                                    .location = PromptCacheMarkerLocation::MessagePartBoundary,
                                    .after_message_part_count = 2}};
    const auto prepared         = f.prepare(input);
    const auto& data            = glm::PreparedPromptAccess::view(prepared);
    const auto span             = data.identity.media[0].item.token_spans[0];
    bool explicit_boundary      = false;
    for (const auto& opportunity : data.context_cache.opportunities)
        if (has_shared_candidate_evidence(opportunity.evidence,
                                          SharedCandidateEvidence::ExplicitBoundary)) {
            require(opportunity.frontier == span.begin + span.count + 1,
                    "media part boundary was not remapped past its closing token");
            explicit_boundary = true;
        }
    require(explicit_boundary, "missing explicit media part boundary");
    auto changed         = input;
    const auto different = read(root / fixture["cases"][4]["media"][0]["file"].get<std::string>());
    changed.messages[0].parts[1].media.bytes.assign(different.begin(), different.end());
    const auto other       = f.prepare(changed);
    const auto& other_data = glm::PreparedPromptAccess::view(other);
    require(
        glm::input_prefix_equal(data.prefix_view(), other_data.prefix_view(), span.begin) &&
            !glm::input_prefix_equal(data.prefix_view(), other_data.prefix_view(), span.begin + 1),
        "changed image identity is not bound at its first embedding token");
    require(glm::PrefixDigests(data.prefix_view()).at(span.begin + 1) !=
                glm::PrefixDigests(other_data.prefix_view()).at(span.begin + 1),
            "index queries inside media omitted content identity");
    auto config_changed                                = config;
    config_changed["image_processor"]["image_mean"][0] = 0.5;
    const auto cf          = make(config_changed, 16384, 16ULL << 20, tokenizer);
    const auto reprocessed = cf.prepare(input);
    require(!glm::input_prefix_equal(data.prefix_view(),
                                     glm::PreparedPromptAccess::view(reprocessed).prefix_view(),
                                     span.begin + 1),
            "processor configuration is absent from input identity");

    auto repeated = input;
    repeated.messages[0].parts.push_back(input.messages[0].parts[1]);
    const auto rf          = make(config, 16384, 16ULL << 20, tokenizer);
    const auto duplicate   = rf.prepare(repeated);
    const auto& duplicates = glm::PreparedPromptAccess::view(duplicate);
    require(duplicates.media_payloads.size() == 2 &&
                duplicates.media_payloads[0] == duplicates.media_payloads[1],
            "repeated source did not share immutable payload");
    require(duplicates.identity.media[1].item.patch_begin ==
                    duplicates.identity.media[0].item.patch_count &&
                duplicates.identity.media[0].item.token_spans[0].begin !=
                    duplicates.identity.media[1].item.token_spans[0].begin,
            "repeated media lost its per-use patch or token coordinates");
    auto job   = [&] { return f.prepare(input); };
    auto first = std::async(std::launch::async, job), second = std::async(std::launch::async, job);
    const auto a = first.get(), b = second.get();
    require(glm::PreparedPromptAccess::view(a).media_payloads[0] ==
                glm::PreparedPromptAccess::view(b).media_payloads[0],
            "concurrent media requests did not share the cache");

    const auto bytes    = data.media_payloads[0]->patch_elements * 2;
    const auto limited  = make(config, 16384, bytes, tokenizer);
    const auto retained = limited.prepare(input);
    request_error([&] { (void)limited.prepare(repeated); }, RequestErrorKind::MediaBudgetExceeded);
    const auto recovered = limited.prepare(input);
    require(glm::PreparedPromptAccess::view(retained).media_payloads[0] ==
                glm::PreparedPromptAccess::view(recovered).media_payloads[0],
            "failed request damaged a live cached payload");
    const auto small = make(config, 32, 16ULL << 20, tokenizer);
    require(small.count_tokens(input) == data.token_ids.size(),
            "count_tokens enforced execution context capacity");
    request_error([&] { (void)small.prepare(input); }, RequestErrorKind::ContextLengthExceeded);
    std::atomic<int> checks{};
    const auto cancel_frontend = make(config, 16384, 16ULL << 20, tokenizer);
    const auto caller          = std::this_thread::get_id();
    const PreparationControl cancel{.cancellation = CancellationView([&] {
                                        return std::this_thread::get_id() != caller &&
                                               ++checks >= 3;
                                    })};
    request_error([&] { (void)cancel_frontend.prepare(repeated, cancel); },
                  RequestErrorKind::Cancelled);
    const auto retry = cancel_frontend.prepare(repeated);
    require(retry.summary().has_media, "cancelled preparation left the cache unusable");
    const auto& downscale      = fixture["cases"][3];
    const auto strict_frontend = make(downscale["config"], 16384, 16ULL << 20, tokenizer);
    auto down                  = media_prompt(downscale, root);
    const auto down_ready      = strict_frontend.prepare(down);
    down.messages[0].parts[1].media.image_resize_policy = ImageResizePolicy::RejectOversized;
    request_error([&] { (void)strict_frontend.prepare(down); },
                  RequestErrorKind::MediaBudgetExceeded);
    request_error([&] { (void)strict_frontend.count_tokens(down); },
                  RequestErrorKind::MediaBudgetExceeded);

    // Ordinary source text cannot become a second embedding placeholder.
    auto literal                      = input;
    literal.messages[0].parts[0].text = "literal <|image|>";
    const auto literal_ready          = f.prepare(literal);
    const auto& literal_data          = glm::PreparedPromptAccess::view(literal_ready);
    const auto image_token            = f.tokenize_text("<|image|>")[0];
    require(std::count(literal_data.token_ids.begin(), literal_data.token_ids.end(), image_token) ==
                span.count,
            "literal media marker became an embedding token");

    // The decoder reports only frames actually delivered when declared metadata has a short tail.
    const auto video_file = read(root / fixture["cases"][6]["media"][0]["file"].get<std::string>());
    const auto encoded =
        std::span(reinterpret_cast<const std::uint8_t*>(video_file.data()), video_file.size());
    auto source         = media::decode::probe_video(encoded, {});
    source.total_frames = 5;
    const media::decode::VideoSelection selection{source, {0, 1, 2, 4}, 1};
    const auto info  = media::decode::inspect_video(encoded, {}, selection);
    const auto video = media::decode::decode_video(encoded, {}, selection);
    require(info.indices == std::vector<int>({0, 1, 2}) && info.indices == video.indices &&
                video.frames.size() == 3,
            "decoder returned nonexistent tail indices");
    std::cout << "media cache, identity, boundaries, budgets and cancellation passed\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto path =
            argc > 1 ? std::filesystem::path(argv[1]) : fixtures / "media-golden.json";
        golden(path, argc > 2 ? argv[2] : "", argc > 3 ? argv[3] : "");
        behavior(path, argc > 3 ? argv[3] : "");
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
