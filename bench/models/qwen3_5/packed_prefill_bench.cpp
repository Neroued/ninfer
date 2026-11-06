// Packed prefill microbenchmark. One packed ForwardWork carrying S windows of L tokens is compared
// with the same S windows executed as S single-row ForwardWork over a real .ninfer artifact. Both
// paths run the production RankProgram on fresh sequences; wall time covers submission through
// the retirement of the last execution. Output is CSV:
//   bench,segments,segment_tokens,total_tokens,packed_ms,sequential_ms
#include "models/qwen3_5/native_fixture.h"

#include "artifact/reader.h"
#include "models/qwen3_5/load.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace ninfer;
namespace q35 = models::qwen3_5;
namespace ctx = models::context;
using Native  = test::qwen3_5::NativeRankFixture;
using Clock   = std::chrono::steady_clock;

struct Options {
    std::string artifact;
    std::uint32_t chunk = 2048, warmup = 2, repetitions = 5, drafts = 3;
    bool mtp = true;
    std::vector<std::uint32_t> segments{1, 2, 4, 6}, tokens{128, 512, 1024, 2048};
};

std::vector<std::uint32_t> list(std::string_view text) {
    std::vector<std::uint32_t> out;
    std::stringstream stream{std::string(text)};
    for (std::string item; std::getline(stream, item, ',');) out.push_back(std::stoul(item));
    if (out.empty()) throw std::invalid_argument("empty list");
    return out;
}

Options parse(int argc, char** argv) {
    Options out;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        const auto value = [&]() -> std::string {
            if (++i >= argc) throw std::invalid_argument("missing option value");
            return argv[i];
        };
        if (argument == "--artifact")
            out.artifact = value();
        else if (argument == "--chunk")
            out.chunk = std::stoul(value());
        else if (argument == "--warmup")
            out.warmup = std::stoul(value());
        else if (argument == "--reps")
            out.repetitions = std::stoul(value());
        else if (argument == "--segments")
            out.segments = list(value());
        else if (argument == "--segment-tokens")
            out.tokens = list(value());
        else if (argument == "--no-mtp")
            out.mtp = false;
        else if (argument == "-h" || argument == "--help") {
            std::cout << "usage: " << argv[0]
                      << " --artifact <model.ninfer> [--chunk 2048] [--segments 1,2,4,6]"
                         " [--segment-tokens 128,512,1024,2048] [--warmup 2] [--reps 5]"
                         " [--no-mtp]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (out.artifact.empty()) throw std::invalid_argument("--artifact is required");
    if (out.repetitions == 0) throw std::invalid_argument("--reps must be positive");
    return out;
}

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

constexpr std::uint32_t kRows = 6, kPage = 64;

// Independent sequences over one Context; each repetition clears their States and restarts them
// at position zero.
class Rig {
public:
    Rig(Native& native, std::uint32_t context)
        : native_(native), pool_(0, native.rank().resources.context.size(),
                                 native.rank().resources.context_records) {
        const auto pages = context / kPage;
        const std::vector<RecordRequest> demand{{pool_.record_class(0), kRows},
                                                {pool_.record_class(1), kRows * pages},
                                                {pool_.record_class(2), kRows * pages}};
        auto result = pool_.reserve(demand);
        require(result.reservation && result.reservation->ready(),
                "Context capacity cannot hold the benchmark rows");
        auto& reservation = *result.reservation;
        for (auto& state : states_)
            state = pool_.resolve(reservation.materialize(pool_.record_class(0)));
        q35::ContextWork bindings;
        q35::Work work;
        for (std::uint32_t row = 0; row < kRows; ++row) {
            for (std::uint32_t kind = 0; kind < 2; ++kind) {
                std::vector<std::int64_t> offsets;
                for (std::uint32_t page = 0; page < pages; ++page)
                    offsets.push_back(static_cast<std::int64_t>(
                        pool_.resolve(reservation.materialize(pool_.record_class(kind + 1)))
                            .offset));
                work.context.operations.emplace_back(
                    ctx::MappingUpdate{kind, static_cast<std::int32_t>(row), 0, offsets});
            }
            rows_[row] = {.member            = {{row + 1, 1}, {{row}, 1}},
                          .source_state      = {row, 1},
                          .destination_state = {row, 1},
                          .state_offsets     = {static_cast<std::int64_t>(states_[row].offset),
                                                static_cast<std::int64_t>(states_[row].offset)},
                          .main_table        = {0, row, 1},
                          .backend_table     = ctx::TableRow{1, row, 1}};
            bindings.bindings.push_back({rows_[row].member, {}, {}});
        }
        work.operation = std::move(bindings);
        wait(submit(std::move(work), runtime::FrameKind::Context), false);
    }

    void reset() {
        q35::Work work;
        for (const auto& state : states_)
            work.context.operations.emplace_back(
                ctx::RecordClear{{ctx::ContextMemory::Device,
                                  static_cast<std::int64_t>(state.offset), state.bytes}});
        work.operation = q35::ContextWork{};
        wait(submit(std::move(work), runtime::FrameKind::Context), false);
        for (auto& row : rows_) {
            row.main_frontier = row.backend_frontier = 0;
            row.tail_rope_position                   = {};
        }
    }

    // One ForwardWork holding `rows` windows of `tokens`, committed and retired.
    void forward(std::uint32_t rows, std::uint32_t first, std::uint32_t tokens) {
        q35::ForwardWork work;
        work.purpose = q35::ForwardPurpose::Prefill;
        for (std::uint32_t r = first; r < first + rows; ++r) {
            q35::InputWindow input;
            input.tokens.resize(tokens);
            input.rope_positions.resize(3 * std::size_t(tokens));
            for (std::uint32_t i = 0; i < tokens; ++i) {
                input.tokens[i] =
                    static_cast<TokenId>(1000 + (r * 7919ULL + i * 40503ULL) % 200000);
                for (int axis = 0; axis < 3; ++axis)
                    input.rope_positions[axis * tokens + i] =
                        static_cast<std::int32_t>(i) + 2 * axis;
            }
            work.rows.push_back(rows_[r]);
            work.inputs.push_back(std::move(input));
        }
        wait(submit({.operation = std::move(work)}, runtime::FrameKind::Compute), true);
    }

    runtime::ExecutionTicket submit(q35::Work work, runtime::FrameKind kind) {
        auto reservation = native_.executor().try_reserve(kind);
        require(bool(reservation), "native frame is unexpectedly occupied");
        return native_.executor().submit(std::move(*reservation), std::move(work));
    }

    void wait(runtime::ExecutionTicket ticket, bool commit) {
        for (bool ready = false; !ready;) {
            native_.device().synchronize();
            auto event = native_.executor().poll();
            if (!event) continue;
            if (auto* failure = std::get_if<runtime::ExecutionFailed>(&*event))
                throw std::runtime_error(failure->message);
            ready = std::holds_alternative<runtime::ResultReady<q35::ExecutionTypes>>(*event);
        }
        if (commit) (void)native_.executor().commit(ticket, {});
        for (;;) {
            native_.device().synchronize();
            auto event = native_.executor().poll();
            if (!event) continue;
            if (auto* failure = std::get_if<runtime::ExecutionFailed>(&*event))
                throw std::runtime_error(failure->message);
            if (std::holds_alternative<runtime::DeviceRetired>(*event)) return;
        }
    }

    Native& native_;
    ContextPool pool_;
    std::array<RecordBinding, kRows> states_;
    std::array<q35::ExecutionRow, kRows> rows_;
};

double milliseconds(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

double mean_ms(Rig& rig, std::uint32_t segments, std::uint32_t tokens, bool packed,
               const Options& options) {
    double total = 0;
    for (std::uint32_t i = 0; i < options.warmup + options.repetitions; ++i) {
        rig.reset();
        const auto begin = Clock::now();
        if (packed) {
            rig.forward(segments, 0, tokens);
        } else {
            for (std::uint32_t row = 0; row < segments; ++row) rig.forward(1, row, tokens);
        }
        const auto elapsed = milliseconds(begin);
        if (i >= options.warmup) total += elapsed;
    }
    return total / options.repetitions;
}

int run(const Options& options) {
    DeviceContext loading;
    artifact::Reader reader(options.artifact);
    auto definition = q35::read_definition(
        reader, {.speculative = options.mtp ? SpeculativeBackend::Mtp : SpeculativeBackend::None});
    auto profile = q35::execution::describe_model(*definition);
    std::shared_ptr<const q35::ModelShard> model =
        q35::materialize_model(q35::resolve_load(reader, definition, loading.device()), loading);
    const auto pack = q35::resolve_prefill_pack_tokens(options.chunk, 2 * options.chunk);
    EngineOptions engine;
    engine.max_context                       = pack;
    engine.max_concurrency                   = kRows;
    engine.prefill_chunk                     = options.chunk;
    engine.kv_cache                          = KvCacheStorage::BFloat16;
    engine.use_cuda_graph                    = false;
    engine.context_cache.enabled             = false;
    engine.context_cache.host_capacity_bytes = 0;
    if (options.mtp) {
        engine.speculative.backend      = SpeculativeBackend::Mtp;
        engine.speculative.draft_tokens = options.drafts;
    }
    Native native(definition, model, profile, engine);
    Rig rig(native, pack);
    std::cout << "bench,segments,segment_tokens,total_tokens,packed_ms,sequential_ms\n";
    for (auto segments : options.segments) {
        for (auto tokens : options.tokens) {
            const std::uint64_t total = std::uint64_t{segments} * tokens;
            if (segments > kRows || tokens > options.chunk || total > pack) continue;
            const auto packed     = mean_ms(rig, segments, tokens, true, options);
            const auto sequential = mean_ms(rig, segments, tokens, false, options);
            std::cout << "packed_prefill," << segments << ',' << tokens << ',' << total << ','
                      << packed << ',' << sequential << '\n';
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "ninfer_qwen3_5_packed_prefill_bench: " << error.what() << '\n';
        return 1;
    }
}
