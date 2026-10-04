// Real-artifact scenarios for the hybrid prefix cache (docs/maintainer/hybrid-prefix-cache.md
// §13.3). Requires NINFER_TEST_ARTIFACT; NINFER_HYBRID_REAL_SCENARIO selects one scenario.

#include "ninfer/engine.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kPrefillChunk = 512;
// Tokens per cached KV block.
constexpr std::uint32_t kBlock = 64;

// NINFER_HYBRID_KV_DTYPE selects the KV storage every scenario runs with (default bf16), so the
// Host tier's page records and restores are exercised for each profile.
ninfer::KvCacheStorage kv_storage = ninfer::KvCacheStorage::BFloat16;

bool select_kv_storage(std::string_view name) {
    if (name == "bf16") {
        kv_storage = ninfer::KvCacheStorage::BFloat16;
    } else if (name == "int8") {
        kv_storage = ninfer::KvCacheStorage::Int8Group64;
    } else if (name == "fp8") {
        kv_storage = ninfer::KvCacheStorage::Fp8E4M3Row256;
    } else if (name == "nvfp4") {
        kv_storage = ninfer::KvCacheStorage::Nvfp4Group16;
    } else if (name == "k8v4") {
        kv_storage = ninfer::KvCacheStorage::Fp8KeyNvfp4Value;
    } else {
        return false;
    }
    return true;
}

// Deterministic ordinary-vocabulary tokens; the content only has to be reproducible.
std::vector<ninfer::TokenId> synthetic_tokens(std::size_t count, std::uint32_t seed) {
    std::vector<ninfer::TokenId> tokens;
    tokens.reserve(count);
    std::uint32_t state = seed * 2654435761U + 1U;
    for (std::size_t index = 0; index < count; ++index) {
        state = state * 1664525U + 1013904223U;
        tokens.push_back(static_cast<ninfer::TokenId>(1000U + (state >> 8U) % 30000U));
    }
    return tokens;
}

ninfer::EngineOptions hybrid_options(const char* artifact, ninfer::SpeculativeBackend backend,
                                     std::uint32_t kv_tokens, std::size_t host_bytes,
                                     std::uint32_t device_slots) {
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = 4096;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(kv_tokens);
    options.prefill_chunk        = kPrefillChunk;
    options.kv_cache             = kv_storage;
    options.speculative.backend  = backend;
    options.max_concurrency      = 1;
    options.max_pending_requests = 1;
    if (backend == ninfer::SpeculativeBackend::Mtp) {
        options.speculative.draft_tokens  = 3;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    } else if (backend == ninfer::SpeculativeBackend::DFlash2) {
        options.speculative.draft_tokens = 7;
    }
    options.context_cache.host_cache_bytes      = host_bytes;
    options.context_cache.device_snapshot_slots = device_slots;
    return options;
}

ninfer::RequestOptions greedy(std::uint32_t outputs) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = true;
    options.stop.include_model_defaults       = false;
    return options;
}

struct Observed {
    std::vector<ninfer::TokenId> tokens;
    std::uint32_t reused = 0;
    ninfer::RuntimeStats stats;
};

// A later request restores a flexible prompt-tail snapshot the first request captured at its last
// prefill chunk boundary. The Device-resident engine restores it with Device copies; the
// constrained engine is first forced to evict the snapshot image and most of its blocks, so the
// same snapshot comes back through the Host slabs. Both restores carry identical bytes and resume
// at the same frontier with the same chunking, so greedy generation must match token for token.
int exercise_restore_exact(const char* artifact, ninfer::SpeculativeBackend backend) {
    const std::vector<ninfer::TokenId> first = synthetic_tokens(1500, 1);
    std::vector<ninfer::TokenId> second(first.begin(), first.begin() + 1400);
    const std::vector<ninfer::TokenId> suffix = synthetic_tokens(300, 2);
    second.insert(second.end(), suffix.begin(), suffix.end());
    // 3900 prompt tokens need 61 of the constrained engine's 64 pages, evicting all but the
    // first few of the first request's cached blocks.
    const std::vector<ninfer::TokenId> pressure = synthetic_tokens(3900, 3);

    const auto run = [&](ninfer::EngineOptions options, bool apply_pressure) {
        ninfer::Engine engine(std::move(options));
        (void)engine.generate(engine.prepare_tokens(first), greedy(8));
        if (apply_pressure) { (void)engine.generate(engine.prepare_tokens(pressure), greedy(4)); }
        const ninfer::GenerationResult result =
            engine.generate(engine.prepare_tokens(second), greedy(24));
        return Observed{result.generated_token_ids, result.reused_prompt_tokens,
                        engine.runtime_stats()};
    };
    const Observed device = run(hybrid_options(artifact, backend, 16384, 1ULL << 30, 8), false);
    const Observed host   = run(hybrid_options(artifact, backend, 4096, 1ULL << 30, 1), true);

    // The first prompt's chunks end at 512 and 1024; the final chunk [1024, 1500) holds the
    // flexible prompt-tail tap, realized at its start.
    constexpr std::uint32_t kExpectedReuse = 2 * kPrefillChunk;
    int failures                           = 0;
    if (device.reused != kExpectedReuse || host.reused != kExpectedReuse) {
        std::cerr << "restore-exact: reused device=" << device.reused << " host=" << host.reused
                  << ", expected the prompt-tail snapshot at " << kExpectedReuse << '\n';
        ++failures;
    }
    if (device.stats.host_image_restores != 0 || device.stats.host_block_restores != 0) {
        std::cerr << "restore-exact: the Device-resident engine restored from Host\n";
        ++failures;
    }
    if (host.stats.host_image_restores == 0 || host.stats.host_block_restores == 0) {
        std::cerr << "restore-exact: the constrained engine did not restore through Host slabs"
                  << " (images=" << host.stats.host_image_restores
                  << " blocks=" << host.stats.host_block_restores << ")\n";
        ++failures;
    }
    if (device.tokens.size() != 24 || device.tokens != host.tokens) {
        std::cerr << "restore-exact: Host and Device restores generated different tokens\n";
        ++failures;
    }
    return failures;
}

// Pseudo-prose of exactly `count` tokens: random words give a continuation with real
// probability ties, so a different floating-point decomposition shows up in greedy output
// within a few dozen tokens (random token ids continue too confidently to show it).
std::vector<ninfer::TokenId> prose_tokens(const ninfer::Engine& engine, std::size_t count,
                                          std::uint32_t seed) {
    static constexpr std::array<std::string_view, 24> kWords{
        "river",   "ledger",  "harbor",  "violet",  "copper", "winter",  "signal",  "garden",
        "market",  "engine",  "silver",  "morning", "bridge", "pattern", "lantern", "meadow",
        "thunder", "compass", "orchard", "canvas",  "ember",  "voyage",  "mirror",  "granite"};
    std::string text;
    std::uint32_t state = seed * 2654435761U + 1U;
    std::vector<ninfer::TokenId> tokens;
    while (tokens.size() < count) {
        for (int word = 0; word < 400; ++word) {
            state = state * 1664525U + 1013904223U;
            text += kWords[(state >> 8U) % kWords.size()];
            text += (state >> 20U) % 9U == 0 ? ". " : " ";
        }
        tokens = engine.tokenize_text(text);
    }
    tokens.resize(count);
    return tokens;
}

// Resuming from a tap is bit-exact with prefilling the same prompt cold: the snapshot is the state
// the first request computed on the same chunk grid, so a second Engine that never saw the first
// prompt generates the same tokens. Raw-token prompts carry no hints, so neither request splits a
// chunk. The comparison would not hold across chunk decompositions: another --prefill-chunk
// changes greedy output on its own (docs/maintainer/hybrid-prefix-cache.md, section 7.8).
int exercise_resume_exact(const char* artifact, ninfer::SpeculativeBackend backend) {
    constexpr std::uint32_t kOutputs = 160;
    const auto run                   = [&](bool prime) {
        ninfer::Engine engine(hybrid_options(artifact, backend, 16384, 1ULL << 30, 8));
        const std::vector<ninfer::TokenId> first = prose_tokens(engine, 3000, 6);
        std::vector<ninfer::TokenId> second(first.begin(), first.begin() + 2800);
        const std::vector<ninfer::TokenId> suffix = prose_tokens(engine, 400, 7);
        second.insert(second.end(), suffix.begin(), suffix.end());
        if (prime) { (void)engine.generate(engine.prepare_tokens(first), greedy(8)); }
        return engine.generate(engine.prepare_tokens(second), greedy(kOutputs));
    };
    const ninfer::GenerationResult warm = run(true);
    const ninfer::GenerationResult cold = run(false);

    // The first prompt's final chunk [2560, 3000) holds its flexible prompt-tail tap, realized at
    // its start, inside the 2800 tokens the prompts share.
    constexpr std::uint32_t kExpectedReuse = 5 * kPrefillChunk;
    int failures                           = 0;
    if (warm.reused_prompt_tokens != kExpectedReuse || cold.reused_prompt_tokens != 0) {
        std::cerr << "resume-exact: reused warm=" << warm.reused_prompt_tokens
                  << " cold=" << cold.reused_prompt_tokens << ", expected " << kExpectedReuse
                  << " and 0\n";
        ++failures;
    }
    if (warm.generated_token_ids.size() != kOutputs ||
        warm.generated_token_ids != cold.generated_token_ids) {
        std::cerr << "resume-exact: the tap resume generated different tokens than a cold "
                  << "prefill\n";
        ++failures;
    }
    return failures;
}

// A restarted Engine resumes from the Host tier its predecessor saved: the restored snapshot is
// the same bytes, so greedy generation matches an Engine that never restarted. A file written for
// a different build identity is ignored, and a Host tier one slab smaller than the file restores
// some but not all of its snapshots.
int exercise_persist(const char* artifact) {
    const std::filesystem::path file =
        std::filesystem::temp_directory_path() / "ninfer-hybrid-persist-real-test.bin";
    std::error_code ignored;
    std::filesystem::remove(file, ignored);
    const std::vector<ninfer::TokenId> first = synthetic_tokens(1500, 4);
    std::vector<ninfer::TokenId> second(first.begin(), first.begin() + 1400);
    const std::vector<ninfer::TokenId> suffix = synthetic_tokens(300, 5);
    second.insert(second.end(), suffix.begin(), suffix.end());
    const auto options = [&](const char* identity) {
        ninfer::EngineOptions result =
            hybrid_options(artifact, ninfer::SpeculativeBackend::None, 16384, 1ULL << 30, 8);
        if (identity != nullptr) {
            result.context_cache.persistent_file     = file;
            result.context_cache.persistent_identity = identity;
        }
        return result;
    };

    std::vector<ninfer::TokenId> reference;
    {
        ninfer::Engine engine(options(nullptr));
        (void)engine.generate(engine.prepare_tokens(first), greedy(8));
        reference = engine.generate(engine.prepare_tokens(second), greedy(24)).generated_token_ids;
    }
    {
        ninfer::Engine saver(options("persist-test"));
        (void)saver.generate(saver.prepare_tokens(first), greedy(8));
    }
    int failures = 0;
    if (!std::filesystem::exists(file)) {
        std::cerr << "persist: the Engine did not save its Host tier\n";
        return 1;
    }
    // The saver's file, for the smaller Host tier below (the loader saves over `file`).
    const std::filesystem::path partial =
        std::filesystem::temp_directory_path() / "ninfer-hybrid-persist-real-test-partial.bin";
    std::filesystem::copy_file(file, partial, std::filesystem::copy_options::overwrite_existing);
    // The file read is published as one startup phase: the whole file, never a failure (a failed
    // phase would be logged as a failed startup).
    std::vector<ninfer::StartupEvent> load_events;
    const auto observe = [&](ninfer::EngineOptions result) {
        load_events.clear();
        result.startup_observer.callback = [&](const ninfer::StartupEvent& event) {
            if (event.phase == ninfer::StartupPhase::PrefixCacheLoad) {
                load_events.push_back(event);
            }
        };
        return result;
    };
    const auto observed  = [&](const char* identity) { return observe(options(identity)); };
    const auto one_phase = [&](std::uint64_t expected_bytes) {
        bool monotonic = true;
        for (std::size_t index = 1; index < load_events.size(); ++index) {
            monotonic = monotonic && load_events[index].current >= load_events[index - 1].current &&
                        load_events[index].current <= expected_bytes;
        }
        return load_events.size() >= 2 &&
               load_events.front().status == ninfer::StartupStatus::Begin &&
               load_events.front().total == expected_bytes &&
               load_events.back().status == ninfer::StartupStatus::Complete &&
               load_events.back().current == expected_bytes && monotonic &&
               std::all_of(load_events.begin() + 1, load_events.end() - 1,
                           [](const ninfer::StartupEvent& event) {
                               return event.status == ninfer::StartupStatus::Progress;
                           });
    };
    std::uint64_t required_host_bytes = 0;
    std::uint64_t saved_snapshots     = 0;
    {
        const std::uint64_t file_bytes = std::filesystem::file_size(file);
        ninfer::Engine loader(observed("persist-test"));
        if (!one_phase(file_bytes)) {
            std::cerr << "persist: the file read was not published as one complete phase ("
                      << load_events.size() << " events)\n";
            ++failures;
        }
        const ninfer::LoadSummary summary = loader.load_summary();
        const ninfer::GenerationResult resumed =
            loader.generate(loader.prepare_tokens(second), greedy(24));
        const ninfer::RuntimeStats stats                     = loader.runtime_stats();
        const ninfer::LoadSummary::PrefixCacheRestore& cache = summary.prefix_cache;
        if (!cache.restored || cache.snapshots == 0 || cache.snapshots != cache.saved_snapshots ||
            cache.blocks != cache.saved_blocks || cache.required_host_bytes > cache.host_bytes) {
            std::cerr << "persist: the saved Host tier was not wholly restored (" << cache.message
                      << ")\n";
            ++failures;
        }
        required_host_bytes = cache.required_host_bytes;
        saved_snapshots     = cache.saved_snapshots;
        if (resumed.reused_prompt_tokens != 2 * kPrefillChunk || stats.host_block_restores == 0 ||
            stats.host_image_restores == 0) {
            std::cerr << "persist: the restarted Engine reused " << resumed.reused_prompt_tokens
                      << " tokens (blocks restored " << stats.host_block_restores << ")\n";
            ++failures;
        }
        if (resumed.generated_token_ids != reference) {
            std::cerr << "persist: the restarted Engine generated different tokens\n";
            ++failures;
        }
    }
    {
        ninfer::Engine foreign(observed("another-build"));
        const ninfer::LoadSummary summary = foreign.load_summary();
        if (!load_events.empty()) {
            std::cerr << "persist: a file from another build published a load phase\n";
            ++failures;
        }
        if (summary.prefix_cache.restored || summary.prefix_cache.message.empty() ||
            foreign.generate(foreign.prepare_tokens(second), greedy(4)).reused_prompt_tokens != 0) {
            std::cerr << "persist: a file from another build was not ignored\n";
            ++failures;
        }
    }
    if (saved_snapshots < 2) {
        std::cerr << "persist: the saved tier holds " << saved_snapshots
                  << " snapshots; a smaller tier cannot be checked\n";
        ++failures;
    } else {
        // One slab short of the file: whole snapshots are dropped, never all of them, and only the
        // chosen entries are read.
        ninfer::EngineOptions small             = observe(hybrid_options(
            artifact, ninfer::SpeculativeBackend::None, 16384, required_host_bytes - 1, 8));
        small.context_cache.persistent_file     = partial;
        small.context_cache.persistent_identity = "persist-test";
        const std::uint64_t file_bytes          = std::filesystem::file_size(partial);
        ninfer::Engine engine(std::move(small));
        const ninfer::LoadSummary::PrefixCacheRestore cache = engine.load_summary().prefix_cache;
        if (!cache.restored || cache.snapshots == 0 || cache.snapshots >= cache.saved_snapshots ||
            cache.blocks > cache.saved_blocks || cache.required_host_bytes <= cache.host_bytes) {
            std::cerr << "persist: a smaller Host tier restored " << cache.snapshots << " of "
                      << cache.saved_snapshots << " snapshots (" << cache.message << ")\n";
            ++failures;
        }
        if (!one_phase(cache.bytes) || cache.bytes >= file_bytes) {
            std::cerr << "persist: a smaller Host tier did not read only what it restored ("
                      << cache.bytes << " of " << file_bytes << " bytes)\n";
            ++failures;
        }
        (void)engine.generate(engine.prepare_tokens(second), greedy(4));
    }
    std::filesystem::remove(file, ignored);
    std::filesystem::remove(partial, ignored);
    return failures;
}

ninfer::ChatMessage text_message(ninfer::ChatRole role, std::string text) {
    ninfer::ChatMessage message;
    message.role = role;
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
    return message;
}

// Protocol adapters shape the hints the way OpenAI default caching does: the Engine's own
// tool-boundary tap is disabled and an automatic marker follows the last message. Taps must keep
// their semantic placement under those hints.
bool protocol_hints = false;

ninfer::PromptInput conversation(std::vector<ninfer::ChatMessage> messages) {
    ninfer::PromptInput input;
    input.messages                = std::move(messages);
    input.options.enable_thinking = false;
    if (protocol_hints) {
        input.context_cache.allow_engine_automatic_shared_prefixes = false;
        input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
            .after_message_count = static_cast<std::uint32_t>(input.messages.size()),
            .kind                = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence            = ninfer::SharedCandidateEvidence::DefaultAutomatic,
            .location            = ninfer::PromptCacheMarkerLocation::MessageBoundary,
        });
    }
    return input;
}

// Exact semantic taps: the next turn of a conversation resumes at the previous turn's generation
// opener (not 64-token-floored), and a new conversation with the same leading system block resumes
// at that block's end.
int exercise_turns(const char* artifact, std::size_t host_bytes) {
    ninfer::EngineOptions options =
        hybrid_options(artifact, ninfer::SpeculativeBackend::None, 8192, host_bytes, 0);
    options.context_cache.device_snapshot_slots.reset();
    ninfer::Engine engine(std::move(options));

    std::string system = "You are a careful assistant for a small engineering team.";
    for (int sentence = 0; sentence < 60; ++sentence) {
        system += " Rule " + std::to_string(sentence) +
                  ": answer precisely, cite the relevant component, and keep replies short.";
    }
    const ninfer::ChatMessage leading = text_message(ninfer::ChatRole::System, system);
    const ninfer::ChatMessage first_user =
        text_message(ninfer::ChatRole::User, "Name three prime numbers.");

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(conversation({leading, first_user})), greedy(24));
    const ninfer::GenerationResult second = engine.generate(
        engine.prepare(conversation({leading, first_user,
                                     text_message(ninfer::ChatRole::Assistant, first.content),
                                     text_message(ninfer::ChatRole::User, "Now name two more.")})),
        greedy(8));
    const ninfer::GenerationResult other = engine.generate(
        engine.prepare(
            conversation({leading, text_message(ninfer::ChatRole::User, "What is a hash table?")})),
        greedy(8));
    const std::uint32_t other_user_tokens =
        engine
            .prepare(conversation({text_message(ninfer::ChatRole::User, "What is a hash table?")}))
            .summary()
            .prompt_tokens;

    int failures = 0;
    // The next turn resumes at the first turn's generation opener, a few tokens before its end, or
    // at the earliest boundary of the opener's cluster (the system-block end after a short user
    // turn), at most the 64-token tap separation earlier.
    if (second.reused_prompt_tokens + 64U + 16U < first.prompt.prompt_tokens ||
        second.reused_prompt_tokens >= second.prompt.prompt_tokens) {
        std::cerr << "turns: second turn reused " << second.reused_prompt_tokens
                  << " of the first turn's " << first.prompt.prompt_tokens
                  << " prompt tokens; expected the generation opener\n";
        ++failures;
    }
    // The shared system block ends where the new user turn begins.
    if (other.reused_prompt_tokens + other_user_tokens + 16U < other.prompt.prompt_tokens) {
        std::cerr << "turns: a new conversation reused " << other.reused_prompt_tokens << " of "
                  << other.prompt.prompt_tokens << " tokens; expected the system block end\n";
        ++failures;
    }
    const ninfer::RuntimeStats stats = engine.runtime_stats();
    const std::uint64_t hits         = stats.endpoint_selections + stats.snapshot_selections;
    if (stats.taps_created == 0 || hits < 2) {
        std::cerr << "turns: taps=" << stats.taps_created << " hits=" << hits << '\n';
        ++failures;
    }
    return failures;
}

std::vector<std::uint8_t> gradient_ppm(int size, int phase) {
    std::vector<std::uint8_t> ppm;
    const std::string header =
        "P6\n" + std::to_string(size) + ' ' + std::to_string(size) + "\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < size * size; ++index) {
        ppm.push_back(static_cast<std::uint8_t>((index + phase) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>(((index / size) * 3 + phase) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7 + phase * 5) & 0xff));
    }
    return ppm;
}

// Vision identity: the same image and text resume at the generation opener, while identical text
// tokens with a different image cannot reuse anything from the image onwards (its blocks carry the
// image's content key) and never resume inside the image's span.
int exercise_vision(const char* artifact) {
    ninfer::EngineOptions options =
        hybrid_options(artifact, ninfer::SpeculativeBackend::None, 8192, 1ULL << 30, 0);
    options.context_cache.device_snapshot_slots.reset();
    options.enable_vision = true;
    ninfer::Engine engine(std::move(options));

    std::string system = "You describe images for an accessibility service.";
    for (int sentence = 0; sentence < 30; ++sentence) {
        system +=
            " Rule " + std::to_string(sentence) + ": name colours, shapes and layout plainly.";
    }
    const auto prompt = [&](const std::vector<std::uint8_t>& image, const char* question) {
        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        ninfer::MessagePart media;
        media.kind              = ninfer::MessagePartKind::Media;
        media.media.kind        = ninfer::MediaKind::Image;
        media.media.bytes       = image;
        media.media.media_type  = "image/x-portable-pixmap";
        media.media.source_name = "image.ppm";
        user.parts.push_back(std::move(media));
        user.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = question, .media = {}});
        return conversation({text_message(ninfer::ChatRole::System, system), std::move(user)});
    };
    const std::vector<std::uint8_t> first_image  = gradient_ppm(512, 0);
    const std::vector<std::uint8_t> second_image = gradient_ppm(512, 97);

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(prompt(first_image, "Describe the image.")), greedy(4));
    const ninfer::GenerationResult same =
        engine.generate(engine.prepare(prompt(first_image, "Describe the image.")), greedy(4));
    const ninfer::GenerationResult other =
        engine.generate(engine.prepare(prompt(second_image, "Describe the image.")), greedy(4));

    int failures = 0;
    if (same.reused_prompt_tokens + 64U < first.prompt.prompt_tokens) {
        std::cerr << "vision: the same image and text reused " << same.reused_prompt_tokens
                  << " of " << first.prompt.prompt_tokens << " tokens\n";
        ++failures;
    }
    // A 512x512 image is 256 merged tokens; a different image must lose at least those.
    if (other.prompt.prompt_tokens != first.prompt.prompt_tokens ||
        other.reused_prompt_tokens + 256U > first.prompt.prompt_tokens) {
        std::cerr << "vision: a different image reused " << other.reused_prompt_tokens << " of "
                  << other.prompt.prompt_tokens << " tokens\n";
        ++failures;
    }
    if (first.generated_token_ids.size() != 4 || other.generated_token_ids.size() != 4) {
        std::cerr << "vision: generation did not complete\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    if (const char* storage = std::getenv("NINFER_HYBRID_KV_DTYPE");
        storage != nullptr && *storage != '\0' && !select_kv_storage(storage)) {
        std::cerr << "unknown NINFER_HYBRID_KV_DTYPE " << storage << '\n';
        return 1;
    }
    const char* selected            = std::getenv("NINFER_HYBRID_REAL_SCENARIO");
    const std::string_view scenario = selected != nullptr && *selected != '\0' ? selected : "all";
    constexpr std::array<std::string_view, 11> kScenarios{"all",
                                                          "restore-exact",
                                                          "restore-exact-mtp",
                                                          "restore-exact-dflash2",
                                                          "resume-exact",
                                                          "resume-exact-dflash2",
                                                          "turns",
                                                          "turns-device-only",
                                                          "vision",
                                                          "persist",
                                                          "turns-protocol"};
    if (std::find(kScenarios.begin(), kScenarios.end(), scenario) == kScenarios.end()) {
        std::cerr << "unknown NINFER_HYBRID_REAL_SCENARIO " << scenario << '\n';
        return 1;
    }
    int failures = 0;
    try {
        const bool all = scenario == "all";
        if (all || scenario == "restore-exact") {
            failures += exercise_restore_exact(artifact, ninfer::SpeculativeBackend::None);
        }
        if (all || scenario == "restore-exact-mtp") {
            failures += exercise_restore_exact(artifact, ninfer::SpeculativeBackend::Mtp);
        }
        if (all || scenario == "restore-exact-dflash2") {
            failures += exercise_restore_exact(artifact, ninfer::SpeculativeBackend::DFlash2);
        }
        if (all || scenario == "resume-exact") {
            failures += exercise_resume_exact(artifact, ninfer::SpeculativeBackend::None);
        }
        if (all || scenario == "resume-exact-dflash2") {
            failures += exercise_resume_exact(artifact, ninfer::SpeculativeBackend::DFlash2);
        }
        if (all || scenario == "turns") { failures += exercise_turns(artifact, 1ULL << 30); }
        if (all || scenario == "turns-device-only") { failures += exercise_turns(artifact, 0); }
        if (all || scenario == "vision") { failures += exercise_vision(artifact); }
        if (all || scenario == "persist") { failures += exercise_persist(artifact); }
        if (all || scenario == "turns-protocol") {
            protocol_hints = true;
            failures += exercise_turns(artifact, 1ULL << 30);
            protocol_hints = false;
        }
    } catch (const std::exception& error) {
        std::cerr << "hybrid prefix real test failed: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "hybrid prefix real tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
