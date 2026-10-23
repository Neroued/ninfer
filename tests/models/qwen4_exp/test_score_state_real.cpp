#include "models/qwen4_exp/load.h"
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

    void start(runtime::ExecutionRequest<q4::ExecutionTypes>&& request) {
        if (const auto* score = std::get_if<q4::ScoreWork>(&request.work.operation)) {
            const auto& row = score->row;
            require(starts < 2 && starts == results, "unexpected scoring continuation");
            require(row.main_frontier == 0 && row.rope_delta == 0,
                    "fresh scoring inherited a prior history frontier");
            require(!row.mtp_frontier && !row.mtp_table.valid(),
                    "scoring unexpectedly uses an MTP history");
            require(row.rope_positions.size() == 3 * row.ids.size(),
                    "fresh scoring omitted its three RoPE axes");
            for (unsigned axis = 0; axis < 3; ++axis)
                for (std::size_t t = 0; t < row.ids.size(); ++t)
                    require(row.rope_positions[axis * row.ids.size() + t] ==
                                static_cast<std::int32_t>(t),
                            "fresh scoring inherited RoPE positions");
            const auto eos = implementation.parameters.profile->config.text.ple.eos;
            require(row.ple_raw_tail == std::array<TokenId, 2>{eos, eos},
                    "fresh scoring inherited the previous PLE raw tail");
            if (starts) {
                require(first_state_written, "first score did not exercise StateImage writes");
                bool stale = false;
                try {
                    implementation.validate_work(first_request->work,
                                                 first_request->ticket.frame.kind);
                } catch (const std::logic_error&) { stale = true; }
                require(stale, "fresh scoring still accepts the previous execution binding");
            } else {
                first_request = request;
            }
            const auto image = read_state(row.states.at(0).source);
            require(std::all_of(image.begin(), image.end(),
                                [](std::byte value) { return value == std::byte{}; }),
                    "fresh scoring inherited nonzero GDN, QSA or PLE state");
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
            require(bases.size() == 1 && bases[0] == 0 && valid.size() == 1 &&
                        valid[0] == static_cast<std::int32_t>(row.ids.size()) &&
                        frame.batch.max_visible_tokens == row.ids.size(),
                    "fresh scoring device batch exposes an earlier KV history");
            require(positions.size() == row.ids.size(), "scoring position extent mismatch");
            for (std::size_t t = 0; t < positions.size(); ++t)
                require(positions[t] == static_cast<std::int32_t>(t),
                        "fresh scoring device positions do not start at zero");
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
        options.purpose          = EnginePurpose::CausalScoring;
        options.use_cuda_graph   = false;
        options.enable_vision    = false;
        options.max_context      = 128;
        options.prefill_chunk    = 128;
        options.max_concurrency  = 1;
        options.context_capacity = ContextCapacityPolicy::explicit_capacity(256ULL << 20);
        options.context_cache    = {.enabled = false, .host_capacity_bytes = 0};
        artifact::Reader reader(path);
        auto definition = q4::read_definition(reader, models::load_options(options));
        auto profile    = q4::execution::describe_model(*definition);
        auto load       = q4::resolve_load(reader, definition, device->device(), device->device());
        const auto geometry = load.ple_table().geometry();
        auto model          = q4::materialize_model(std::move(load), *device);
        auto execution =
            q4::plan_execution(profile, geometry, options, *device,
                               device_memory_snapshot(device->device()).available_bytes());
        require(execution.has_value(), "fresh scoring execution resources do not fit");
        auto plan = q4::plan_session(*execution);
        auto instance =
            std::make_unique<runtime::RankModelInstance<ObservedTypes>>(std::move(model), profile);
        instance->program =
            std::make_unique<ObservedRank>(instance->parameters, std::move(*execution), *device);
        auto& rank  = *instance->program;
        auto worker = std::make_unique<runtime::RankWorker<ObservedTypes>>(std::move(device),
                                                                           std::move(instance));
        runtime::LocalExecutor<ObservedTypes> executor(std::move(worker), {1});
        q4::Session session(q4::prepare_session(definition, std::move(plan)), executor);
        const auto empty = session.physical_usage();
        models::qwen::FrontendOptions frontend_options;
        frontend_options.vision_enabled = false;
        frontend_options.max_context    = options.max_context;
        auto frontend = models::qwen::make_frontend(definition->resources(), frontend_options, {});
        // 65 consumed inputs leave a partial QSA compression group as well as GDN/PLE history.
        std::vector<TokenId> ids(66);
        for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = 100 + i % 27;
        for (unsigned iteration = 0; iteration < 2; ++iteration) {
            const auto ticket = session.start_score(frontend.prepare_tokens(ids), 1);
            require(ticket.has_value(), "scoring could not reserve its fresh binding");
            const auto result = finish_score(session, *ticket);
            require(result.logprobs.size() == ids.size() - 1 &&
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
        require(rank.starts == 2 && rank.results == 2, "fresh scoring observation was incomplete");
        std::cout << "PASS: fresh scoring clears StateImage and exposes no prior KV history\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
