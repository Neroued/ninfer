#include "distributed/launch.h"
#include "models/qwen4_exp/parallel/preparation.h"
#include "models/qwen4_exp/parallel/startup.h"
#include "product/config/options.h"
#include "runtime/executor/rank_preparation.h"
#include "../models/qwen4_exp/ple/lookup_reference.h"

#include <iostream>

namespace {
using namespace ninfer;
namespace rt   = runtime;
namespace dist = distributed;
namespace wire = distributed::wire;
namespace q    = models::qwen4_exp;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

// Inject an allocation failure after both the base and the selected table exist, in the
// model-only materialization wave. Every rank has to release before the next layout can load.
class Preparation : public q::RankPreparation {
public:
    using RankPreparation::RankPreparation;

    void materialize() {
        require(++loads == 1, "base weights reloaded on placement retry");
        RankPreparation::materialize();
    }

    void materialize_placement(std::size_t id) {
        require(!installed && (id == 0 || releases > 0), "placement switched before group release");
        RankPreparation::materialize_placement(id);
        installed = true;
        if (inject && id == 0) throw DeviceAllocationError("injected PLE table allocation failure");
    }

    void release_placement() {
        RankPreparation::release_placement();
        installed = false;
        ++releases;
    }

    static inline bool inject = false;
    bool installed            = false;
    unsigned loads = 0, releases = 0;
};

void check_lookup(const artifact::Reader& reader, const q::ModelShard& model,
                  dist::WorkerConnection& connection) {
    const auto& definition = model.definition();
    const auto& table      = std::get<q::DevicePleTable>(model.ple_table());
    const auto& params     = definition.ple_parameters();
    auto& device           = connection.device();
    constexpr int batch = 2, tokens = 3, width = q::ple::kHeads * q::ple::kRowElements;
    const auto eos = definition.config().text.ple.eos;
    // The two rows share an input ID but have distinct histories. The second row ends early.
    const std::array<TokenId, 6> ids{101, eos, 327, 101, 2253, 177};
    const std::array<TokenId, 4> tails{227, 53, 19, 177};
    const std::array<int, 2> valid{3, 2};
    const std::array<std::int64_t, 2> offsets{0, 8};
    const auto upload = [&](const auto& values) {
        DeviceBuffer buffer(device.device(), sizeof(values));
        buffer.copy_from_host(values.data(), sizeof(values));
        return buffer;
    };
    auto input = upload(ids), lengths = upload(valid), history = upload(tails),
         rows = upload(offsets);
    DeviceBuffer output(device.device(), width * batch * tokens * sizeof(std::uint16_t));
    Tensor embedding(output.span(), DType::BF16, {width, tokens, batch});
    ops::NgramHashParameters hash;
    std::ranges::copy(params.hash.multipliers, hash.multipliers.begin());
    std::ranges::copy(params.hash.vocab_sizes, hash.vocab_sizes.begin());
    std::ranges::copy(params.hash.offsets, hash.offsets.begin());
    ops::NgramTokenTailView before{{{history.span(), Tensor(rows.span(), DType::I64, {batch})},
                                    {{{0, 8, 4}, DType::I32, {2, 1, 1, 1}}, 1, 8}}};
    ops::ngram_embedding_lookup(Tensor(input.span(), DType::I32, {tokens, batch}),
                                Tensor(lengths.span(), DType::I32, {batch}), before, hash, eos,
                                table.shards, embedding, device.execution_view());
    if (model.rank_plan().ple_placement == q::ple::Placement::Sharded) {
        const auto& group = connection.topology().role_group("ple", connection.identity().rank);
        models::ParallelExecution merge({&connection.communication().group(group.id)});
        merge.merge_unique(embedding, device.execution_view());
    }
    device.synchronize();
    std::vector<std::uint16_t> words(width * batch * tokens);
    output.copy_to_host(words.data(), output.size());
    for (int row = 0; row < batch; ++row) {
        auto older = tails[2 * row], previous = tails[2 * row + 1];
        for (int col = 0; col < tokens; ++col) {
            const auto pos = row * tokens + col;
            auto actual    = std::span(words).subspan(pos * width, width);
            if (col >= valid[row])
                require(std::ranges::all_of(actual, [](auto word) { return word == 0; }),
                        "padded PLE column contains a nonzero word");
            else
                test::flash_next::check_ple_column(reader, definition, older, previous, ids[pos],
                                                   actual);
            older    = previous;
            previous = ids[pos];
        }
    }
}

int worker(bool failure) {
    dist::WorkerConnection connection(dist::read_worker_bootstrap());
    std::unique_ptr<Preparation> preparing;
    std::unique_ptr<Preparation::Instance> instance;
    try {
        const auto message = rt::receive_preparation_command(connection);
        require(message.kind == wire::Kind::Prepare, "expected model preparation");
        wire::Decoder in(message.payload);
        const auto request = rt::decode_preparation_request(in);
        in.finish();
        auto reader         = std::make_unique<artifact::Reader>(connection.bootstrap().artifact);
        Preparation::inject = failure && connection.identity().rank == 1;
        const auto attempt  = rt::prepare_rank(connection, std::move(reader), request.model_payload,
                                               request.memory_headroom_bytes, preparing);
        if (attempt) {
            require(preparing->loads == 1 && preparing->installed &&
                        (!failure || preparing->releases),
                    "placement retry lost ownership");
            instance = std::move(*preparing).finish();
            preparing.reset();
            artifact::Reader reference(connection.bootstrap().artifact);
            check_lookup(reference, *instance->model, connection);
            wire::Encoder ready;
            ready.u64(connection.topology().plan);
            ready.u32(*attempt);
            connection.send(wire::Kind::Ready, std::move(ready).take());
            require(rt::receive_preparation_command(connection).kind == wire::Kind::Shutdown,
                    "expected shutdown after the lookup check");
        }
        instance.reset();
        preparing.reset();
        connection.shutdown();
        return 0;
    } catch (const std::exception& error) {
        connection.fail(error.what());
        throw;
    }
}

int coordinator(const char* path, std::string_view mode) {
    EngineOptions options;
    options.deployment = product::read_deployment_file(path);
    require(options.deployment->parallel.tp == 2 && options.deployment->parallel.dp == 1,
            "lookup fixture requires one replica with two consumers");
    options.ple.storage    = PleStorage::Device;
    options.ple.preference = PlePreference::Latency;
    const bool failure     = mode != "replicated";
    if (mode == "explicit" || !failure)
        options.ple.device_layout = PleDeviceLayout::Replicated;
    else
        require(mode == "retry", "expected replicated, explicit or retry");
    options.max_context      = 512;
    options.prefill_chunk    = 128;
    options.max_concurrency  = 2;
    options.context_capacity = ContextCapacityPolicy::explicit_capacity(2ULL << 30);
    options.context_cache.host_capacity_bytes = 0;
    options.use_cuda_graph                    = false;
    artifact::Reader reader(options.deployment->nodes.front().artifact);
    q::ParallelStartup setup(reader, options);
    const std::array<std::string, 2> args{"--worker", failure ? "fail" : "normal"};
    dist::ExecutionGroup group(
        *options.deployment, setup.topology(), {},
        [](auto rank, auto message) { std::cerr << "[rank " << rank << "] " << message << '\n'; },
        args);
    const auto requests = setup.requests(options);
    try {
        auto prepared = rt::prepare_execution_group(
            group, requests, options.context_capacity,
            {.memory_headroom_bytes = options.memory_headroom_bytes,
             .frontend_host_bytes   = setup.coordinator_payload_bytes(options),
             .coordinator_node      = options.deployment->coordinator_node,
             .placement_names       = setup.placement_names(),
             .maximize_context      = false},
            std::chrono::minutes(10));
        require(mode != "explicit" && (!failure || prepared.attempt > 0),
                "explicit PLE placement silently changed");
        for (const auto& layout : prepared.layouts) {
            wire::Decoder in(layout);
            const auto report = q::decode_rank_execution(in);
            in.finish();
            require(report.observation.memory.ple_table &&
                        report.observation.memory.ple_table->placement ==
                            (failure ? "device/sharded" : "device/replicated"),
                    "rank did not install the selected PLE layout");
        }
        rt::activate_execution_group(group, prepared, std::chrono::minutes(2));
        group.shutdown();
    } catch (const rt::ContextBudgetError& e) {
        if (mode != "explicit" ||
            std::string_view(e.what()).find("injected PLE table allocation failure") ==
                std::string_view::npos)
            throw;
    }
    std::cout << "PASS " << mode << '\n';
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("expected DEPLOYMENT replicated|retry|explicit");
        if (std::string_view(argv[1]) == "--worker")
            return worker(std::string_view(argv[2]) == "fail");
        return coordinator(argv[1], argv[2]);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
