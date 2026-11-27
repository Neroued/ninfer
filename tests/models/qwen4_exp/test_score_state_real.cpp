#include "rank_fixture.h"
#include "models/qwen4_exp/program/internal.h"
#include "models/qwen4_exp/session/session.h"
#include "runtime/executor/local_executor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <thread>

namespace {
using namespace ninfer;
namespace q4 = models::qwen4_exp;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template <class T>
std::vector<T> read_tensor(const Tensor& tensor) {
    require(tensor.bytes() % sizeof(T) == 0, "invalid observation element size");
    std::vector<T> values(tensor.bytes() / sizeof(T));
    CUDA_CHECK(cudaMemcpy(values.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
    return values;
}

// Observes actual submissions and completed device state without adding a product test hook.
class ObservedRank {
public:
    ObservedRank(const q4::execution::Parameters& parameters, q4::ExecutionPlan plan,
                 DeviceContext& device)
        : implementation(parameters, std::move(plan), device) {
        implementation.initialize_graphs();
    }

    [[nodiscard]] runtime::ExecutionCapacity capacity() const noexcept { return {1, 1}; }

    void validate_work(const q4::Work& work, runtime::FrameKind kind) const {
        implementation.validate_work(work, kind);
    }

    void validate_decision(runtime::ExecutionTicket ticket,
                           const runtime::DecisionInput& input) const {
        implementation.validate_decision(ticket, input);
    }

    void prepare_commit(runtime::ExecutionTicket ticket, q4::Commit& commit) const {
        implementation.prepare_commit(ticket, commit);
    }

    void expect_score(std::span<const TokenId> tokens, std::size_t first_target) {
        expected_tokens = tokens;
        next_input      = 0;
        next_target     = first_target;
    }

    void start(runtime::ExecutionRequest<q4::ExecutionTypes>&& request) {
        if (const auto* score = std::get_if<q4::ScoreWork>(&request.work.operation)) {
            const auto& row = score->row;
            const auto n    = row.ids.size();
            require(starts == results && row.main_frontier == next_input && n &&
                        next_input + n < expected_tokens.size() &&
                        std::ranges::equal(row.ids, expected_tokens.subspan(next_input, n)),
                    "scoring chunk changed input membership or frontier");
            require(score->first_target_column <= n &&
                        score->targets.size() == n - score->first_target_column &&
                        next_target + score->targets.size() <= expected_tokens.size() &&
                        std::ranges::equal(score->targets, expected_tokens.subspan(
                                                               next_target, score->targets.size())),
                    "scoring chunk changed the requested target slice");
            if (!score->targets.empty())
                require(next_input + score->first_target_column + 1 == next_target,
                        "first score used the wrong causal predecessor");
            require(row.rope_delta == 0 && row.rope_positions.empty(),
                    "linear scoring unexpectedly uses explicit RoPE");
            require(!row.mtp_frontier && !row.mtp_table.valid(),
                    "scoring unexpectedly uses an MTP history");
            const auto eos = implementation.parameters.profile->config.text.ple.eos;
            const std::array<TokenId, 2> tail{next_input >= 2 ? expected_tokens[next_input - 2]
                                                              : eos,
                                              next_input ? expected_tokens[next_input - 1] : eos};
            require(row.ple_raw_tail == tail, "scoring chunk inherited the wrong PLE raw tail");
            if (!next_input) {
                if (first_request) {
                    require(first_state_written, "first score did not exercise StateImage writes");
                    bool stale = false;
                    try {
                        implementation.validate_work(first_request->work,
                                                     first_request->ticket.frame.kind);
                    } catch (const std::logic_error&) { stale = true; }
                    require(stale, "fresh scoring still accepts the previous execution binding");
                } else
                    first_request = request;
                const auto image = read_state(row.states.at(0).source);
                require(std::all_of(image.begin(), image.end(),
                                    [](std::byte v) { return v == std::byte{}; }),
                        "fresh scoring inherited nonzero GDN, QSA or PLE state");
            }
            next_input += n;
            next_target += score->targets.size();
            ++starts;
        }
        implementation.start(std::move(request));
    }

    void supply(runtime::ExecutionTicket ticket, runtime::CommandSequence command,
                runtime::DecisionInput&& input) {
        implementation.supply(ticket, command, std::move(input));
    }

    void commit(runtime::ExecutionTicket ticket, runtime::CommandSequence command,
                q4::Commit&& commit) {
        implementation.commit(ticket, command, std::move(commit));
    }

    [[nodiscard]] bool can_discard(runtime::ExecutionTicket ticket) const {
        return implementation.can_discard(ticket);
    }

    void discard(runtime::ExecutionTicket ticket, runtime::CommandSequence command) {
        implementation.discard(ticket, command);
    }

    [[nodiscard]] std::optional<runtime::ExecutionEvent<q4::ExecutionTypes>> poll() {
        auto event = implementation.poll();
        const auto* ready =
            event ? std::get_if<runtime::ResultReady<q4::ExecutionTypes>>(&*event) : nullptr;
        if (ready && std::holds_alternative<q4::ScoreResult>(ready->result)) {
            const auto& frame    = *implementation.compute;
            const auto& row      = std::get<q4::ScoreWork>(frame.request.work.operation).row;
            const auto bases     = read_tensor<std::int32_t>(frame.batch.base_positions);
            const auto valid     = read_tensor<std::int32_t>(frame.batch.valid_columns);
            const auto positions = read_tensor<std::int32_t>(frame.batch.positions);
            require(bases.size() == 1 && bases[0] == static_cast<std::int32_t>(row.main_frontier) &&
                        valid.size() == 1 &&
                        valid[0] == static_cast<std::int32_t>(row.ids.size()) &&
                        frame.batch.max_visible_tokens == row.main_frontier + row.ids.size(),
                    "scoring device batch exposes the wrong KV history");
            require(positions.size() == row.ids.size(), "scoring position extent mismatch");
            for (std::size_t t = 0; t < positions.size(); ++t)
                require(positions[t] == static_cast<std::int32_t>(row.main_frontier + t),
                        "scoring device positions do not follow the input frontier");
            const auto rope = read_tensor<std::int32_t>(frame.batch.rope_positions);
            require(rope.size() == 3 * row.ids.size(), "scoring omitted a device RoPE axis");
            for (std::size_t axis = 0; axis < 3; ++axis)
                for (std::size_t t = 0; t < row.ids.size(); ++t)
                    require(rope[axis * row.ids.size() + t] ==
                                static_cast<std::int32_t>(row.main_frontier + t),
                            "scoring RoPE positions do not follow the input frontier");
            if (!results) {
                const auto image    = read_state(row.states.at(0).destination);
                first_state_written = std::any_of(image.begin(), image.end(), [](std::byte value) {
                    return value != std::byte{};
                });
                require(first_state_written, "real scoring left its entire StateImage untouched");
            }
            ++results;
        }
        return event;
    }

    void drain() { implementation.drain(); }

    [[nodiscard]] RankMemoryUsage observe_resources() const {
        return implementation.observe_resources();
    }

    void reset_measurement_peaks() noexcept { implementation.reset_measurement_peaks(); }

    q4::detail::RankProgramImpl implementation;
    unsigned starts = 0, results = 0;

private:
    [[nodiscard]] std::vector<std::byte> read_state(std::int64_t offset) const {
        const auto image = implementation.resources.states->storage().image(offset);
        std::vector<std::byte> bytes(image.bytes);
        CUDA_CHECK(cudaMemcpy(bytes.data(), image.data, image.bytes, cudaMemcpyDeviceToHost));
        return bytes;
    }

    std::span<const TokenId> expected_tokens;
    std::size_t next_input = 0, next_target = 0;
    bool first_state_written = false;
    std::optional<runtime::ExecutionRequest<q4::ExecutionTypes>> first_request;
};

struct ObservedTypes {
    using ModelShard     = q4::ModelShard;
    using Parameters     = q4::execution::Parameters;
    using Profile        = q4::execution::ModelProfile;
    using RankProgram    = ObservedRank;
    using ExecutionTypes = q4::ExecutionTypes;
};

q4::ScoreComplete finish_score(q4::Session& session, runtime::ExecutionTicket ticket) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(2);
    while (std::chrono::steady_clock::now() < deadline) {
        auto event = session.poll();
        if (!event) {
            std::this_thread::yield();
            continue;
        }
        if (auto* score = std::get_if<q4::ScoreComplete>(&event->result)) {
            require(event->ticket == ticket, "scoring completion changed its public ticket");
            return std::move(*score);
        }
        throw std::runtime_error("unexpected scoring event");
    }
    throw std::runtime_error("fresh scoring did not complete before its deadline");
}
} // namespace

int main() {
    const char* path = std::getenv("NINFER_QWEN4_EXP_TEST_ARTIFACT");
    if (!path || !*path) return 77;
    try {
        int devices      = 0;
        const auto error = cudaGetDeviceCount(&devices);
        if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver || !devices)
            return 77;
        CUDA_CHECK(error);
        auto device = std::make_unique<DeviceContext>();
        EngineOptions options;
        options.purpose         = EnginePurpose::CausalScoring;
        options.use_cuda_graph  = false;
        options.enable_vision   = false;
        options.max_context     = 512;
        options.prefill_chunk   = 128;
        options.max_concurrency = 1;
        options.context_cache   = {.enabled = false, .host_capacity_bytes = 0};
        artifact::Reader reader(path);
        q4::Startup startup(reader, reader, options, device->device());
        auto selected          = test::flash_next::plan_rank(startup, *device, options);
        const auto& definition = startup.definition();
        auto plan              = q4::plan_session(selected.sequence);
        auto model             = startup.materialize(*device, selected.placement, {});
        auto instance          = std::make_unique<runtime::RankModelInstance<ObservedTypes>>(
            std::move(model), startup.profile);
        instance->program = std::make_unique<ObservedRank>(instance->parameters,
                                                           std::move(selected.sequence), *device);
        auto& rank        = *instance->program;
        auto worker       = std::make_unique<runtime::RankWorker<ObservedTypes>>(std::move(device),
                                                                                 std::move(instance));
        runtime::LocalExecutor<ObservedTypes> executor(std::move(worker), {1},
                                                       options.max_concurrency);
        q4::Session session(q4::prepare_session(definition, std::move(plan)), executor);
        const auto empty = session.physical_usage();
        ninfer::models::frontend::FrontendOptions frontend_options;
        frontend_options.vision_enabled = false;
        frontend_options.max_context    = options.max_context;
        auto frontend = models::qwen::make_frontend(definition->resources(), frontend_options, {});
        // 65 consumed inputs leave a partial QSA compression group as well as GDN/PLE history.
        std::vector<TokenId> ids(261);
        for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = 50 + (i * 17 + i / 3) % 89;
        for (auto [length, first] : {std::pair{66U, 1U}, {261U, 1U}, {261U, 126U}}) {
            const auto tokens = std::span(ids).first(length);
            rank.expect_score(tokens, first);
            const auto ticket =
                session.start_score(frontend.prepare_tokens({tokens.begin(), tokens.end()}), first);
            require(ticket.has_value(), "scoring could not reserve its fresh binding");
            const auto result = finish_score(session, *ticket);
            require(result.logprobs.size() == length - first &&
                        std::all_of(result.logprobs.begin(), result.logprobs.end(),
                                    [](float value) { return std::isfinite(value) && value <= 0; }),
                    "scoring returned invalid log probabilities");
            const auto usage = session.physical_usage();
            require(usage.state_objects == empty.state_objects &&
                        usage.kv_page_groups == empty.kv_page_groups &&
                        std::all_of(usage.pools.begin(), usage.pools.end(),
                                    [](const auto& pool) {
                                        return pool.live_bytes == 0 && pool.reserved_bytes == 0 &&
                                               pool.pending_retirement_bytes == 0;
                                    }) &&
                        !executor.has_pending_work(),
                    "scoring completion left context ownership or an execution frame alive");
        }
        require(rank.starts == 7 && rank.results == 7, "fresh scoring observation was incomplete");
        std::cout
            << "PASS: fresh scoring clears StateImage and preserves causal targets across chunks\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
