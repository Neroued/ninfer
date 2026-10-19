#include "core/buffer.h"
#include "core/decode_graph.h"
#include "ninfer/ops/collective.h"
#include "product/config/options.h"
#include "runtime/executor/distributed_executor.h"
#include "runtime/executor/worker_loop.h"

#include <cuda_bf16.h>
#include <sys/wait.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <thread>

namespace {
using namespace ninfer;
namespace dist = ninfer::distributed;
namespace wire = dist::wire;
namespace rt   = ninfer::runtime;

enum class Operation : std::uint8_t {
    Sum,
    Words,
    Maximum,
    Gather,
    Scatter,
    Broadcast,
    Exchange,
    State
};

struct Work {
    Operation operation = Operation::Sum;
    DType dtype         = DType::FP32;
    std::uint32_t count = 257;
    bool graph = false, decision = false, delay = false;
    std::int32_t fail_rank = -1;
    std::int32_t only_rank = -1;
};

struct Result {
    std::uint64_t state = 0;
};

struct Commit {
    std::uint32_t accepted = 1;
};

struct Types {
    using Work   = ::Work;
    using Result = ::Result;
    using Commit = ::Commit;
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Codec {
    rt::protocol::ControlBudget control_budget() const noexcept {
        return {4096, 4096, 4096, 1024, 4096, 65536};
    }

    void validate_work(const Work& w, rt::FrameKind kind) const {
        require(w.count && w.count <= 4096, "fixture extent is invalid");
        require((w.operation == Operation::State) == (kind == rt::FrameKind::Context),
                "fixture frame kind differs");
    }

    rt::protocol::Scope scope(const Work& work, const dist::ExecutionTopology& topology) const {
        if (work.only_rank >= 0) {
            require(work.operation == Operation::State, "partial-owner work must be local");
            const auto owner = static_cast<dist::RankId>(work.only_rank);
            return {{owner}, owner};
        }
        rt::protocol::Scope out{{}, topology.output_rank};
        for (const auto& rank : topology.ranks) out.owners.push_back(rank.rank);
        return out;
    }

    void encode_work(wire::Encoder& out, const Work& w, dist::RankId) const {
        out.u8(static_cast<std::uint8_t>(w.operation));
        out.u8(static_cast<std::uint8_t>(w.dtype));
        out.u32(w.count);
        out.boolean(w.graph);
        out.boolean(w.decision);
        out.boolean(w.delay);
        out.i32(w.fail_rank);
        out.i32(w.only_rank);
    }

    Work decode_work(wire::Decoder& in) const {
        return {static_cast<Operation>(in.u8()),
                static_cast<DType>(in.u8()),
                in.u32(),
                in.boolean(),
                in.boolean(),
                in.boolean(),
                in.i32(),
                in.i32()};
    }

    void encode_result(wire::Encoder& out, const Result& r) const { out.u64(r.state); }

    Result decode_result(wire::Decoder& in) const { return {in.u64()}; }

    void encode_commit(wire::Encoder& out, const Commit& c, dist::RankId) const {
        out.u32(c.accepted);
    }

    Commit decode_commit(wire::Decoder& in) const { return {in.u32()}; }

    void validate_decision(const Work&, const rt::DecisionRequest& request,
                           const rt::DecisionInput& input) const {
        require(input.stage == request.stage && input.rows.size() == 1 &&
                    input.rows[0].words.size() == 1,
                "fixture decision is invalid");
    }

    void validate_commit(const Work&, const Result&, const Commit& c) const {
        require(c.accepted == 1, "invalid fixture commit");
    }

    Result merge_context_results(const rt::protocol::Scope& scope,
                                 const std::vector<std::optional<Result>>& results) const {
        const auto expected = results.at(scope.result_owner).value();
        for (auto rank : scope.owners)
            require(results.at(rank)->state == expected.state, "rank states disagree");
        return expected;
    }

    MemorySummary merge_memory(const dist::ExecutionTopology& topology,
                               const std::vector<std::optional<MemorySummary>>& reports) const {
        MemorySummary total;
        for (const auto& rank : topology.ranks) {
            const auto& value = reports.at(rank.rank).value();
            total.workspace.capacity_bytes += value.workspace.capacity_bytes;
            total.workspace.peak_used_bytes += value.workspace.peak_used_bytes;
        }
        return total;
    }
};

class Program {
public:
    explicit Program(dist::WorkerConnection& connection)
        : connection_(connection), device_(connection.device()),
          group_(connection.communication().group(0)), compute_(device_, 131072),
          context_(device_, 256), state_(device_.device(), sizeof(std::uint64_t)) {
        state_.fill();
    }

    ~Program() {
        try {
            drain();
        } catch (...) {}
    }

    rt::ExecutionCapacity capacity() const noexcept { return {1, 1}; }

    void validate_work(const Work& w, rt::FrameKind kind) const { Codec{}.validate_work(w, kind); }

    void start(rt::ExecutionRequest<Types>&& request) {
        auto& f = frame(request.ticket.frame);
        require(f.phase == Phase::Empty, "fixture frame is still occupied");
        f.request     = std::move(request);
        f.command     = f.request.command;
        f.phase       = Phase::Running;
        const auto& w = f.request.work;
        if (w.fail_rank == static_cast<std::int32_t>(group_.rank()))
            throw std::runtime_error("injected rank execution failure");
        f.ready_at = std::chrono::steady_clock::now() +
                     std::chrono::milliseconds(w.delay && group_.rank() == 1 ? 60 : 0);
        if (w.operation == Operation::State) {
            CUDA_CHECK(cudaMemcpyAsync(f.output.data(), state_.data(), sizeof(std::uint64_t),
                                       cudaMemcpyDeviceToHost, device_.stream()));
        } else {
            fill(f);
            CUDA_CHECK(cudaMemcpyAsync(f.source.data(), f.input.data(), f.input_bytes,
                                       cudaMemcpyHostToDevice, device_.stream()));
            if (w.graph) {
                f.definition.capture(device_.stream(), [&] { collective(f); });
                f.graph.instantiate(f.definition);
                for (int i = 0; i < 4; ++i) f.graph.launch(device_.stream());
            } else
                collective(f);
            CUDA_CHECK(cudaMemcpyAsync(f.output.data(), f.destination.data(), f.output_bytes,
                                       cudaMemcpyDeviceToHost, device_.stream()));
        }
        f.done.record(device_.stream());
    }

    void validate_decision(rt::ExecutionTicket ticket, const rt::DecisionInput& input) const {
        require(compute_.request.ticket == ticket && compute_.phase == Phase::Decision &&
                    input.stage == 0,
                "fixture decision arrived before its stage");
    }

    void supply(rt::ExecutionTicket ticket, rt::CommandSequence command,
                rt::DecisionInput&& input) {
        auto& f   = find(ticket);
        f.command = command;
        events_.emplace_back(
            rt::DecisionConsumed{ticket, input.stage, {{0, input.rows[0].words.size() * 4}}});
        // A second collective makes a missed or skipped decision stage observable.
        collective(f);
        f.done.record(device_.stream());
        f.phase = Phase::AfterDecision;
    }

    void prepare_commit(rt::ExecutionTicket ticket, Commit&) const {
        require(compute_.request.ticket == ticket && compute_.phase == Phase::Result,
                "fixture commit arrived before local result");
    }

    void commit(rt::ExecutionTicket ticket, rt::CommandSequence command, Commit&& value) {
        auto& f = find(ticket);
        state_value_ += value.accepted;
        std::memcpy(f.input.data(), &state_value_, sizeof(state_value_));
        CUDA_CHECK(cudaMemcpyAsync(state_.data(), f.input.data(), sizeof(state_value_),
                                   cudaMemcpyHostToDevice, device_.stream()));
        f.command = command;
        f.done.record(device_.stream());
        f.phase = Phase::Retiring;
    }

    bool can_discard(rt::ExecutionTicket ticket) const {
        return compute_.request.ticket == ticket &&
               (compute_.phase == Phase::Decision || compute_.phase == Phase::Result);
    }

    void discard(rt::ExecutionTicket ticket, rt::CommandSequence command) {
        auto& f = find(ticket);
        require(can_discard(ticket), "fixture cancelled an unfinished communication stage");
        f.command = command;
        f.done.record(device_.stream());
        f.phase = Phase::Retiring;
    }

    std::optional<rt::ExecutionEvent<Types>> poll() {
        for (auto* f : {&compute_, &context_}) {
            if (f->phase == Phase::Empty || f->phase == Phase::Decision ||
                f->phase == Phase::Result || !f->done.ready() ||
                std::chrono::steady_clock::now() < f->ready_at)
                continue;
            if (f->phase == Phase::Retiring) {
                events_.emplace_back(rt::DeviceRetired{f->request.ticket, f->command});
                f->graph.reset();
                f->definition.reset();
                f->phase = Phase::Empty;
                continue;
            }
            if (f->request.work.operation != Operation::State) verify(*f);
            if (f->phase == Phase::Running && f->request.work.decision) {
                f->phase = Phase::Decision;
                events_.emplace_back(rt::DecisionNeeded{f->request.ticket, {0, 32, {{0, {}}}}});
                continue;
            }
            std::uint64_t state = state_value_;
            if (f->request.work.operation == Operation::State)
                std::memcpy(&state, f->output.data(), sizeof(state));
            events_.emplace_back(rt::ResultReady<Types>{f->request.ticket, {state}});
            f->phase = f->request.ticket.frame.kind == rt::FrameKind::Context ? Phase::Retiring
                                                                              : Phase::Result;
        }
        if (events_.empty()) return {};
        auto value = std::move(events_.front());
        events_.pop_front();
        return value;
    }

    void drain() { device_.synchronize(); }

    MemorySummary memory_summary() const {
        MemorySummary result;
        result.device                   = connection_.bootstrap().identity.device;
        result.workspace.capacity_bytes = compute_.source.size() + compute_.destination.size() +
                                          context_.source.size() + context_.destination.size();
        result.workspace.peak_used_bytes = peak_bytes_;
        return result;
    }

    void reset_memory_peaks() noexcept { peak_bytes_ = 0; }

private:
    enum class Phase { Empty, Running, Decision, AfterDecision, Result, Retiring };

    struct Frame {
        Frame(DeviceContext& device, std::size_t bytes)
            : source(device.device(), bytes), destination(device.device(), bytes), input(bytes),
              output(bytes), done(device) {}

        DeviceBuffer source, destination;
        PinnedHostBuffer input, output;
        CudaCompletionEvent done;
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        rt::ExecutionRequest<Types> request;
        rt::CommandSequence command;
        Phase phase = Phase::Empty;
        std::chrono::steady_clock::time_point ready_at;
        std::size_t input_bytes = 0, output_bytes = 0;
    };

    Frame& frame(rt::FrameId id) { return id.kind == rt::FrameKind::Compute ? compute_ : context_; }

    Frame& find(rt::ExecutionTicket ticket) {
        auto& f = frame(ticket.frame);
        require(f.request.ticket == ticket, "stale fixture ticket");
        return f;
    }

    void fill(Frame& f) {
        const auto& w           = f.request.work;
        const auto input_count  = w.count * (w.operation == Operation::Scatter ? group_.size() : 1);
        const auto output_count = w.count * (w.operation == Operation::Gather ? group_.size() : 1);
        f.input_bytes           = input_count * dtype_size(w.dtype);
        f.output_bytes          = output_count * dtype_size(w.dtype);
        peak_bytes_             = std::max(peak_bytes_, f.input_bytes + f.output_bytes);
        require(f.input_bytes <= f.input.size() && f.output_bytes <= f.output.size(),
                "fixture buffers are too small");
        for (std::size_t i = 0; i < input_count; ++i) {
            const auto value = static_cast<float>(i % 7 + group_.rank() + 1);
            switch (w.dtype) {
            case DType::FP32:
                static_cast<float*>(f.input.data())[i] = value;
                break;
            case DType::BF16:
                static_cast<__nv_bfloat16*>(f.input.data())[i] = __float2bfloat16(value);
                break;
            case DType::U32:
                static_cast<std::uint32_t*>(f.input.data())[i] =
                    i % group_.size() == group_.rank() ? 0x80008000U + i : 0;
                break;
            case DType::U64:
                static_cast<std::uint64_t*>(f.input.data())[i] =
                    (group_.rank() == 0 ? (1ULL << 63) : 0) + i;
                break;
            case DType::U8:
                static_cast<std::uint8_t*>(f.input.data())[i] = (i + group_.rank()) % 256;
                break;
            default:
                throw std::logic_error("unsupported fixture dtype");
            }
        }
        if (w.operation == Operation::Broadcast)
            CUDA_CHECK(cudaMemcpyAsync(f.destination.data(), f.input.data(), f.input_bytes,
                                       cudaMemcpyHostToDevice, device_.stream()));
    }

    void collective(Frame& f) {
        const auto& w        = f.request.work;
        auto source          = Tensor(f.source.span(), w.dtype,
                                      {static_cast<int>(f.input_bytes / dtype_size(w.dtype))});
        auto destination     = Tensor(f.destination.span(), w.dtype,
                                      {static_cast<int>(f.output_bytes / dtype_size(w.dtype))});
        const auto execution = device_.execution_view();
        switch (w.operation) {
        case Operation::Sum:
        case Operation::Words:
            ops::all_reduce(source, destination, dist::Reduction::Sum, group_, execution);
            break;
        case Operation::Maximum:
            ops::all_reduce(source, destination, dist::Reduction::Max, group_, execution);
            break;
        case Operation::Gather:
            ops::all_gather(source, destination, group_, execution);
            break;
        case Operation::Scatter:
            ops::reduce_scatter(source, destination, dist::Reduction::Sum, group_, execution);
            break;
        case Operation::Broadcast:
            ops::broadcast(destination, group_.size() - 1, group_, execution);
            break;
        case Operation::Exchange: {
            require(group_.size() == 2, "exchange fixture requires two ranks");
            const dist::PeerTransfer transfer{1 - group_.rank(),
                                              f.source.span().subspan(0, f.input_bytes),
                                              f.destination.span().subspan(0, f.output_bytes)};
            group_.exchange({&transfer, 1}, execution);
            break;
        }
        default:
            throw std::logic_error("invalid fixture collective");
        }
    }

    void verify(const Frame& f) {
        const auto& w    = f.request.work;
        const auto count = f.output_bytes / dtype_size(w.dtype);
        for (std::size_t i = 0; i < count; ++i) {
            if (w.dtype == DType::U32) {
                require(static_cast<const std::uint32_t*>(f.output.data())[i] == 0x80008000U + i,
                        "PLE word reduction changed represented bits");
                continue;
            }
            if (w.dtype == DType::U64) {
                require(static_cast<const std::uint64_t*>(f.output.data())[i] == (1ULL << 63) + i,
                        "unsigned MAX reduction differs");
                continue;
            }
            if (w.dtype == DType::U8) {
                require(static_cast<const std::uint8_t*>(f.output.data())[i] ==
                            (i + 1 - group_.rank()) % 256,
                        "point-to-point payload differs");
                continue;
            }
            float expected = 0;
            if (w.operation == Operation::Gather)
                expected = (i % w.count) % 7 + (i / w.count) + 1;
            else if (w.operation == Operation::Broadcast)
                expected = i % 7 + group_.size();
            else {
                const auto index =
                    w.operation == Operation::Scatter ? group_.rank() * w.count + i : i;
                for (std::uint32_t rank = 0; rank < group_.size(); ++rank)
                    expected += index % 7 + rank + 1;
            }
            const float actual =
                w.dtype == DType::FP32
                    ? static_cast<const float*>(f.output.data())[i]
                    : __bfloat162float(static_cast<const __nv_bfloat16*>(f.output.data())[i]);
            require(actual == expected,
                    "collective differs from independent rank/element reference");
        }
    }

    dist::WorkerConnection& connection_;
    DeviceContext& device_;
    const dist::Group& group_;
    Frame compute_, context_;
    DeviceBuffer state_;
    std::uint64_t state_value_ = 0;
    std::size_t peak_bytes_    = 0;
    std::deque<rt::ExecutionEvent<Types>> events_;
};

int worker() {
    auto bootstrap = dist::read_worker_bootstrap();
    std::cerr << dist::rank_description(bootstrap.identity) << '\n';
    dist::WorkerConnection connection(std::move(bootstrap));
    try {
        {
            Program program(connection);
            connection.send(wire::Kind::Ready);
            rt::WorkerLoop<Types, Program, Codec> loop(connection, program, {});
            while (loop.advance()) std::this_thread::yield();
        }
        connection.shutdown();
        return 0;
    } catch (const std::exception& error) {
        connection.fail(error.what());
        throw;
    }
}

using Executor = rt::DistributedExecutor<Types, Codec>;

void run(Executor& executor, Work work, std::uint64_t& expected_state, bool cancel = false) {
    const auto kind =
        work.operation == Operation::State ? rt::FrameKind::Context : rt::FrameKind::Compute;
    auto reservation = executor.try_reserve(kind);
    require(bool(reservation), "execution frame was not retired");
    const auto ticket = executor.submit(std::move(*reservation), work);
    require(executor.memory_summary().workspace.capacity_bytes == 4 * (131072 + 256),
            "rank resource report differs from its actual buffers");
    if (cancel) executor.discard(ticket);
    bool result = false, retired = false;
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!retired) {
        require(std::chrono::steady_clock::now() < until, "distributed execution test timed out");
        if (auto event = executor.poll()) {
            if (const auto* failure = std::get_if<rt::ExecutionFailed>(&*event))
                throw std::runtime_error(failure->message);
            if (std::holds_alternative<rt::DecisionNeeded>(*event))
                executor.supply(ticket, {0, {{0, 1, 0, {1}}}});
            if (const auto* output = std::get_if<rt::ResultReady<Types>>(&*event)) {
                require(!cancel, "cancelled execution published a result");
                require(output->result.state == expected_state,
                        "committed state differs across transactions");
                result = true;
                if (kind == rt::FrameKind::Compute) {
                    (void)executor.commit(ticket, {1});
                    ++expected_state;
                    require(!executor.try_reserve(kind),
                            "frame was released before distributed retirement");
                }
            }
            if (std::holds_alternative<rt::DeviceRetired>(*event)) retired = true;
        } else
            std::this_thread::yield();
    }
    require(cancel || result, "execution retired without a result");
}

int coordinator(int argc, char** argv) {
    require(argc == 3 && std::string_view(argv[1]) == "--config",
            "usage: ninfer_distributed_execution_test --config FILE");
    const auto deployment = product::read_deployment_file(argv[2]);
    dist::ExecutionTopology topology;
    topology.plan        = 1;
    topology.output_rank = 0;
    const auto size      = static_cast<std::uint32_t>(deployment.ranks.size());
    require(size == 2, "this integration fixture requires two workers");
    topology.ranks.resize(size);
    for (const auto& r : deployment.ranks)
        topology.ranks[r.rank] = {r.rank,         r.node,         0,  r.device,
                                  {r.rank, size}, {r.rank, size}, {}, {}};
    topology.groups.push_back({0, {0, 1}});
    topology.roles = {{"tp", 0}, {"ep", 0}};
    const std::array<std::string, 1> arguments{"--worker"};
    const auto prepare = [&] {
        auto group = std::make_unique<dist::ExecutionGroup>(
            deployment, topology, [](auto, auto line) { std::cerr << line << '\n'; }, arguments);
        std::size_t ready = 0;
        const auto until  = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (ready != size) {
            require(std::chrono::steady_clock::now() < until, "worker preparation timed out");
            if (auto message = group->transport().poll()) {
                if (message->message.kind == wire::Kind::Failed) {
                    wire::Decoder in(message->message.payload);
                    throw std::runtime_error(in.string());
                }
                require(message->message.kind == wire::Kind::Ready,
                        "unexpected fixture preparation response");
                ++ready;
            } else {
                group->check_processes();
                std::this_thread::yield();
            }
        }
        return group;
    };
    Executor executor(prepare(), {1, 1}, {});
    executor.bind_to_current_thread();
    std::uint64_t state = 0;
    for (auto work : std::vector<Work>{{},
                                       {Operation::Sum, DType::BF16, 2560},
                                       {Operation::Words, DType::U32, 129},
                                       {Operation::Maximum, DType::U64, 37},
                                       {Operation::Gather, DType::FP32, 17},
                                       {Operation::Scatter, DType::FP32, 19},
                                       {Operation::Broadcast, DType::BF16, 32},
                                       {Operation::Exchange, DType::U8, 97},
                                       {Operation::Sum, DType::FP32, 128, true},
                                       {Operation::Sum, DType::FP32, 64, false, true},
                                       {Operation::Sum, DType::FP32, 64, false, false, true}})
        run(executor, work, state);
    run(executor, {Operation::Sum, DType::FP32, 64, false, true, true}, state, true);
    run(executor, {Operation::State, DType::U64, 1}, state);
    run(executor, {Operation::State, DType::U64, 1, false, false, false, -1, 1}, state);
    require(executor.memory_summary().workspace.peak_used_bytes > 0,
            "worker resource peaks were not reported");
    executor.reset_memory_peaks();
    require(executor.memory_summary().workspace.peak_used_bytes == 0,
            "resource reset did not reach every worker");
    executor.drain();
    {
        Executor failing(prepare(), {1, 1}, {});
        failing.bind_to_current_thread();
        auto reservation = failing.try_reserve(rt::FrameKind::Compute);
        Work work;
        work.fail_rank = 1;
        (void)failing.submit(std::move(*reservation), work);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
        bool observed       = false;
        while (!observed) {
            require(std::chrono::steady_clock::now() < deadline,
                    "worker failure did not terminate the collective");
            if (auto event = failing.poll()) {
                if (const auto* failure = std::get_if<rt::ExecutionFailed>(&*event)) {
                    require(!failure->message.empty(), "worker failure lost its diagnostic");
                    std::cout << "expected failure: " << failure->message << '\n';
                    observed = true;
                }
            } else
                std::this_thread::yield();
        }
        require(failing.failed(), "failed group still accepts work");
        failing.drain();
    }
    {
        auto unavailable                = deployment;
        unavailable.nodes.back().worker = "/ninfer-missing-worker";
        bool rejected                   = false;
        try {
            dist::ExecutionGroup missing(unavailable, topology, {}, arguments);
        } catch (const std::exception&) { rejected = true; }
        require(rejected, "failed remote launch was accepted");
        int status = 0;
        require(::waitpid(-1, &status, WNOHANG) == -1 && errno == ECHILD,
                "execution group left an unreaped child after failure");
    }
    std::cout << "distributed collectives, graph, decisions, delayed commit, cancellation and "
                 "retirement, and failed-rank abort passed\n";
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--worker") return worker();
        return coordinator(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
