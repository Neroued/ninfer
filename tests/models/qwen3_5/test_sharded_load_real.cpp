#include "artifact/reader.h"
#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/load/sharding.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace qwen    = ninfer::models::qwen3_5;
namespace loading = ninfer::models::qwen3_5::loading;
using artifact::ShardAxis;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

// The logical shape one rank binds: the split axis narrowed to that rank's ranges.
artifact::Shape rank_shape(artifact::Shape shape, const loading::LogicalShard& shard,
                           std::size_t device) {
    std::uint64_t held = 0;
    for (const auto& range : shard.ranges[device]) { held += range.count; }
    if (shard.axis == ShardAxis::Rows) { shape.front() = held; }
    if (shard.axis == ShardAxis::Columns) { shape.back() = held; }
    return shape;
}

bool holds(const loading::LogicalShard& shard, int device) {
    switch (shard.axis) {
    case ShardAxis::PrimaryOnly:
        return device == 0;
    case ShardAxis::SingleDevice:
        return device == shard.device;
    default:
        return true;
    }
}

void check_plan(const artifact::Reader& reader, const qwen::LoadPlan& plan) {
    const auto& materialization = plan.materialization();
    require(materialization.device_count == 2, "tp 2 did not plan two devices");
    for (const auto& item : materialization.device_objects) {
        if (!artifact::is_sharded(item.axis)) { continue; }
        const auto& geometry = reader.geometry(item.object);
        const auto& id       = artifact::object_id(reader.directory().object(item.object));
        const auto tile      = [&]() -> std::uint64_t {
            if (geometry.layout == QuantLayout::BlockScaleK16M128x4) {
                return item.axis == ShardAxis::Rows ? 128 : 64;
            }
            return geometry.layout == QuantLayout::RowSplit && item.axis == ShardAxis::Columns ? 128
                                                                                                    : 1;
        }();
        for (const auto& range : item.ranges) {
            require(range.begin % tile == 0 && range.count % tile == 0,
                    id + ": shard boundary violates its layout tile");
        }
    }
    const auto& config = plan.config();
    const LoadOptions options{.tp = 2};
    for (std::size_t i = 0; i < plan.parameter_count(); ++i) {
        const auto& reference = plan.parameter(qwen::WeightId{i});
        const auto shard = loading::logical_shard(reference.name, reference.shape, config, options);
        if (!artifact::is_sharded(shard.axis)) { continue; }
        require(rank_shape(reference.shape, shard, 0) == rank_shape(reference.shape, shard, 1),
                reference.name + ": ranks hold different shard shapes");
    }
}

} // namespace

int main(int argc, char** argv) {
    const char* path = std::getenv("NINFER_TEST_ARTIFACT");
    if (!path || !*path) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        LoadOptions options{.tp = 2};
        if (argc > 1 && std::string(argv[1]) == "mtp") {
            options.speculative = SpeculativeBackend::Mtp;
        }
        artifact::Reader reader(path);
        auto plan = qwen::plan_load(reader, options);
        check_plan(reader, plan);
        const auto& capacity = plan.materialization().per_device_capacity_bytes;
        std::cout << "planned weights: device 0 " << capacity[0] << " B, device 1 " << capacity[1]
                  << " B\n";

        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
            std::cout << "skip: tensor-parallel materialization requires two CUDA devices\n";
            return 77;
        }
        // Expected per-rank shapes, recorded before the plan is consumed.
        const auto config = plan.config();
        std::vector<std::pair<std::string, std::array<artifact::Shape, 2>>> expected;
        std::vector<loading::LogicalShard> shards;
        for (std::size_t i = 0; i < plan.parameter_count(); ++i) {
            const auto& reference = plan.parameter(qwen::WeightId{i});
            shards.push_back(
                loading::logical_shard(reference.name, reference.shape, config, options));
            expected.push_back({reference.name,
                                {rank_shape(reference.shape, shards.back(), 0),
                                 rank_shape(reference.shape, shards.back(), 1)}});
        }

        ExecutionContext execution({0, 1});
        const auto model = qwen::materialize_model(std::move(plan), execution);
        require(model->device_count() == 2, "the model has no second rank");
        for (std::size_t i = 0; i < expected.size(); ++i) {
            const qwen::WeightId id{i};
            for (int device = 0; device < 2; ++device) {
                const auto& [name, shapes] = expected[i];
                const bool held            = holds(shards[i], device);
                require(model->resident(id, device) == held,
                        name + ": residency differs from its placement on device " +
                            std::to_string(device));
                if (held) {
                    require(model->weight(id, device).view.shape ==
                                shapes[static_cast<std::size_t>(device)],
                            name + ": unexpected shard shape on device " + std::to_string(device));
                }
            }
        }

        // Native preparation per rank, with the extents the split Ops consume.
        const auto& text  = config.text;
        const auto& dense = std::get<qwen::DenseConfig>(text.ffn);
        for (int device = 0; device < 2; ++device) {
            const qwen::execution::Parameters parameters(*model, device);
            require(parameters.text.output_head.weight.n == std::int32_t(text.vocab_size / 2),
                    "output head is not split by vocabulary");
            require(parameters.qwen4.has_value(), "the Text forward has no blocks");
            for (std::size_t layer = 0; layer < parameters.qwen4->layers.size(); ++layer) {
                const auto& block = parameters.qwen4->layers[layer];
                const auto& ffn   = std::get<qwen::execution::DenseParameters>(block.ffn);
                require(ffn.gate_up.weight.n == std::int32_t(dense.intermediate_size) &&
                            ffn.down.weight.k == std::int32_t(dense.intermediate_size / 2),
                        "MLP shard extents differ");
                if (const auto* a =
                        std::get_if<qwen::execution::AttentionParameters>(&block.mixer)) {
                    const auto& attention = text.attention.value();
                    const auto rows = (2 * attention.query_width() + 2 * attention.key_width()) / 2;
                    const auto* projection =
                        std::get_if<qwen::execution::LinearParameters>(&a->projection);
                    require(!projection || projection->weight.n == std::int32_t(rows),
                            "attention projection shard extent differs");
                    require(a->output.weight.k == std::int32_t(attention.query_width() / 2),
                            "attention output shard extent differs");
                } else {
                    const auto& g   = std::get<qwen::execution::GdnParameters>(block.mixer);
                    const auto& gdn = text.gdn.value();
                    const auto rows = gdn.key_width() + gdn.value_width();
                    const auto* single =
                        std::get_if<qwen::execution::LinearParameters>(&g.projection);
                    require(!single || single->weight.n == std::int32_t(rows),
                            "GDN projection shard extent differs");
                    require(g.output.weight.k == std::int32_t(gdn.value_width() / 2) &&
                                g.convolution.ne[0] == std::int32_t(gdn.conv_channels() / 2),
                            "GDN output or convolution shard extent differs");
                }
            }
            if (parameters.mtp) {
                require(parameters.mtp->input_projection.weight.k == std::int32_t(text.hidden_size),
                        "MTP input projection is not split over its packed input");
            }
        }
        const auto& stats = model->storage_stats();
        for (std::size_t device = 0; device < 2; ++device) {
            std::cout << "device " << device << ": capacity "
                      << stats.per_device_capacity_bytes[device] << " B, sharded "
                      << stats.sharded_bytes[device] << " B, replicated "
                      << stats.replicated_bytes[device] << " B, local " << stats.local_bytes[device]
                      << " B\n";
        }
        std::cout << "qwen3_5 tensor-parallel load passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
