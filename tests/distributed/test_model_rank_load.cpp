#include "models/glm5next/load.h"
#include "models/glm5next/parallel/wire.h"
#include "models/qwen4_exp/load.h"
#include "models/qwen4_exp/execution/profile.h"
#include "models/qwen4_exp/parallel/wire.h"
#include "models/common/moe/exchange.h"
#include "models/qwen4_exp/program/planning.h"
#include "artifact/reader.h"
#include "core/platform.h"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <optional>
#include <vector>
#include <stdexcept>

namespace {
using namespace ninfer;
namespace glm  = models::glm5next;
namespace qwen = models::qwen4_exp;
namespace dist = distributed;
namespace wire = distributed::wire;

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

DeploymentOptions deployment(std::uint32_t dp, std::uint32_t tp, std::uint32_t adp,
                             std::uint32_t ep, const std::filesystem::path& artifact) {
    DeploymentOptions d;
    d.control_host = "127.0.0.1";
    d.parallel     = {.tp = tp, .ep = ep, .dp = dp, .adp = adp};
    d.nodes.push_back({0, {}, "/worker", artifact, {}});
    for (std::uint32_t i = 0; i < dp * tp; ++i) d.ranks.push_back({i, 0, static_cast<int>(i)});
    return d;
}

template <class F>
void rejected(F&& fn, std::string_view reason) {
    try {
        fn();
    } catch (const std::invalid_argument& e) {
        require(std::string_view(e.what()).find(reason) != std::string_view::npos, e.what());
        return;
    }
    throw std::runtime_error("invalid rank load accepted");
}

template <class Request, class Decode>
auto roundtrip(Request request, const dist::ExecutionTopology& topology, Decode decode) {
    const auto id        = request.rank.placement.global_rank;
    const auto& identity = topology.rank(id);
    const auto encode    = [&] {
        wire::Encoder out;
        encode_rank_load(out, request);
        return std::move(out).take();
    };
    const auto read = [&](const wire::Bytes& bytes, const dist::RankIdentity& target) {
        wire::Decoder in(bytes);
        auto result = decode(in, target, request.artifact);
        in.finish();
        return result.rank;
    };
    auto bytes = encode();
    auto rank  = read(bytes, identity);
    rejected([&] { read(bytes, topology.rank((id + 1) % topology.ranks.size())); }, "global rank");
    const auto& domain = *rank.expert_domain;
    require(domain.members == topology.role_group("expert", id).members,
            "dispatch peer order differs");
    for (std::size_t bank = 0; bank < rank.experts.size(); ++bank) {
        const auto& local = domain.banks[bank].at(topology.role_group("expert", id).index(id));
        require(local.ids == rank.experts[bank].ids &&
                    local.intermediate == rank.experts[bank].intermediate,
                "dispatch slots differ from actual materialized partition");
    }
    const auto& g    = topology.role_group("expert", id);
    const auto stage = models::moe::prepare_expert_exchange(g, domain, {0}, 2560, 8, 2);
    wire::Encoder desc;
    stage.encode(desc, 5);
    const auto contract = std::move(desc).take();
    wire::Decoder accepted(contract);
    stage.expect(accepted, 5);
    accepted.finish();
    auto different = stage;
    different.dispatch_bytes.front() += 16;
    rejected(
        [&] {
            wire::Decoder in(contract);
            different.expect(in, 5);
        },
        "differs across participants");
    rejected(
        [&] {
            wire::Decoder in(contract);
            stage.expect(in, 4);
        },
        "differs across participants");
    auto changed = std::make_shared<models::moe::ExpertDomain>(domain);
    std::swap(changed->members.front(), changed->members.back());
    request.rank.expert_domain = changed;
    rejected([&] { read(encode(), identity); }, "shard digest");
    request.rank.expert_domain = rank.expert_domain;
    ++request.rank.experts.front().intermediate.begin;
    rejected([&] { read(encode(), identity); }, "shard digest");
    --request.rank.experts.front().intermediate.begin;
    std::swap(request.rank.experts.front().ids.front(), request.rank.experts.front().ids.back());
    rejected([&] { read(encode(), identity); }, "shard digest");
    return rank;
}

template <class T>
std::vector<T> from_device(const T* src, std::size_t count) {
    std::vector<T> values(count);
    const auto status = cudaMemcpy(values.data(), src, count * sizeof(T), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    return values;
}

// Select source coordinates from the fixture's global expert and intermediate axes. Read
// the original artifact directly rather than deriving expected bytes from the load plan.
void check_bank(const artifact::Reader& reader, const models::moe::BankDescription& source,
                const models::moe::ResidentExpertBank& storage,
                const models::moe::ExpertDescription& local, const models::ExpertPartition& part) {
    const auto& view  = storage.view();
    const auto subset = storage.subset();
    require(subset.kind == ops::ExpertSubsetKind::MaskOnes &&
                view.count == source.down.layout.region.count,
            "resident bank lost its global domain");
    auto mask = from_device(static_cast<const std::int32_t*>(subset.mask.data), view.count);
    std::vector<std::int32_t> expected_mask(view.count);
    for (auto id : part.ids) expected_mask.at(id) = 1;
    require(mask == expected_mask, "device mask differs from assigned experts");
    for (bool down : {false, true}) {
        const auto& input    = down ? source.down : source.gate_up;
        const auto& device   = down ? view.down : view.gate_up;
        const auto& target   = down ? local.down : local.gate_up;
        const auto& original = reader.geometry(input.object);
        const auto start     = reader.directory().tensor(input.object).offset;
        const auto rows = target.shape.at(target.shape.size() - 2), columns = target.shape.back();
        const auto intermediate = source.down.layout.columns;
        const auto bits =
            original.format == QType::NVFP4 || original.format == QType::NVFP4_MUL ? 4U : 8U;
        std::vector<std::uint64_t> offsets;
        if (device.offsets) {
            offsets = from_device(device.offsets, view.count);
            for (std::size_t id = 0; id < view.count; ++id)
                require((offsets[id] != kExpertAbsentOffset) == bool(mask[id]),
                        "incorrect device expert offset");
        } else {
            require(device.expert_indices && device.physical_count == part.ids.size(),
                    "missing compact expert map");
            const auto indices = from_device(device.expert_indices, view.count);
            std::vector<std::uint32_t> expected(view.count, UINT32_MAX);
            for (std::size_t s = 0; s < part.ids.size(); ++s) expected[part.ids[s]] = s;
            require(indices == expected, "incorrect device expert slot");
        }
        for (const auto slot : {std::size_t{0}, part.ids.size() - 1}) {
            const auto expert = part.ids[slot];
            const auto offset = device.offsets ? offsets[expert] : slot * device.code_stride;
            const auto first =
                input.layout.region.row_begin + expert * input.layout.region.row_stride;
            for (const auto row : {std::uint64_t{0}, rows / 2, rows - 1}) {
                const auto sr       = down ? row
                                           : (row / part.intermediate.size) * intermediate +
                                           part.intermediate.begin + row % part.intermediate.size;
                const auto col      = columns - 16;
                const auto sc       = down ? col + part.intermediate.begin : col;
                const auto expected = reader.read_range(
                    start + (first + sr) * original.code_bytes_per_row + sc * bits / 8,
                    16 * bits / 8);
                const auto actual = from_device(
                    device.codes + offset + row * target.code_bytes_per_row + col * bits / 8,
                    expected.size());
                require(actual == expected, "device expert codes differ from original source");
                if (original.layout == QuantLayout::Csf32) {
                    const auto base =
                        reader.read_range(start + original.csf.bases_offset + first + sr, 1);
                    const auto value = from_device(
                        device.csf.first.bases + slot * device.csf.bases_stride + row, 1);
                    require(value.front() == std::to_integer<std::uint8_t>(base.front()),
                            "CSF row base differs");
                } else {
                    const auto origin =
                        weight_scale_offset(original, first + sr, sc / original.group_size);
                    const auto local_scale =
                        weight_scale_offset(target, row, col / target.group_size) -
                        target.scale_offset;
                    const auto scale_base =
                        device.offsets ? offsets[expert] : slot * device.scale_stride;
                    const auto expected_scale =
                        reader.read_range(start + origin, original.scale_word_bytes);
                    const auto scale = from_device(device.scales + scale_base + local_scale,
                                                   target.scale_word_bytes);
                    require(scale == expected_scale,
                            "device expert block scale differs from original source");
                }
            }
            if (input.layout.has_matrix_scales) {
                const auto src_index =
                    input.layout.matrix_scale_begin + expert * input.layout.matrix_scale_stride;
                const auto scalar =
                    reader.read_range(start + original.matrix_scale_offset + 4 * src_index, 4);
                const auto local_index =
                    (device.offsets ? expert : slot) * device.matrix_scale_stride;
                const auto actual = from_device(device.matrix_scales + local_index, 1);
                require(std::memcmp(actual.data(), scalar.data(), 4) == 0,
                        "expert matrix scalar changed");
            }
        }
    }
}

template <class Model, class Bank>
void check_device(const artifact::Reader& reader, const Model& model, Bank bank) {
    const auto& source = model.definition().expert_banks();
    for (std::size_t b = 0; b < source.size(); ++b)
        check_bank(reader, source[b], bank(ExpertBankId{static_cast<std::uint32_t>(b)}),
                   model.expert_data()[b], model.rank_plan().experts[b]);
    for (std::size_t i = 0; i < model.weight_data().size(); ++i) {
        const auto& original = model.definition().weight_data()[i];
        const auto& local    = model.weight_data()[i];
        for (std::size_t j = 0; j < original.uses.size(); ++j) {
            const auto& before = original.uses[j];
            if (!before.calibration_source || !before.calibration ||
                before.calibration->values.size() <= 1)
                continue;
            if (std::ranges::find(model.rank_plan().omitted, models::WeightId{i}) !=
                model.rank_plan().omitted.end())
                continue;
            const auto& after = local.uses.at(j);
            require(after.calibration &&
                        after.calibration->operation == before.calibration->operation &&
                        after.calibration_device.bytes ==
                            before.calibration->values.size() * sizeof(float),
                    "Use calibration lost its logical domain");
            const auto values =
                from_device(static_cast<const float*>(after.calibration_device.data),
                            before.calibration->values.size());
            require(values == before.calibration->values,
                    "Use calibration was reordered by local slot");
        }
    }
    std::cout << "rank=" << model.rank_plan().placement.global_rank << " banks=" << source.size()
              << " weight_bytes=" << model.storage_stats().device_capacity_bytes
              << " metadata_bytes=" << model.metadata_bytes() << '\n';
}

void qwen_case(const artifact::Reader& reader, const std::filesystem::path& path,
               const EngineOptions& options, std::optional<std::uint32_t> materialize) {
    auto def = qwen::read_definition(reader, models::load_options(options));
    auto p   = qwen::plan_parallel(*def, deployment(2, 1, 1, 2, path), qwen::ple::Placement::Disk);
    require(def->weights().mtp.has_value(), "fixture requires MTP");
    const auto full    = qwen::execution::describe_model(*def);
    const auto offload = qwen::describe_moe_plan(*full, options.moe);
    require(bool(offload) == bool(NINFER_EXPERT_OFFLOAD) &&
                (!offload || offload->description().layers.size() == full->text.layers.size()),
            "full expert source lost its target-specific offload schedule");
    const auto main = def->weights().text.layers.front().moe.bank.index;
    const auto mtp  = def->weights().mtp->layer.moe.bank.index;
    for (const auto& planned : p.ranks) {
        auto rank = roundtrip(
            qwen::RankLoadRequest{.artifact = reader.artifact_id(),
                                  .rank     = planned,
                                  .options  = options,
                                  .batches  = models::make_batch_execution_spec(1, 1, false)},
            p.topology, qwen::decode_rank_load);
        const auto id = rank.placement.global_rank;
        require(rank.placement.tp.rank == 0 && rank.placement.dp.rank == id &&
                    rank.placement.ep.rank == id &&
                    rank.ple_placement == qwen::ple::Placement::Disk && rank.ple_shards.size == 0,
                "source replica or Disk table differs");
        require(p.topology.role_group("expert", id).members == std::vector<dist::RankId>{0, 1},
                "expert exchange membership differs");
        for (const auto& b : rank.experts)
            require(b.ids.size() == 256 && std::is_sorted(b.ids.begin(), b.ids.end()) &&
                        b.intermediate == models::IndexRange{0, 640},
                    "Qwen expert layout differs");
        const auto contains = [&](auto bank, auto expert) {
            return std::ranges::binary_search(rank.experts[bank].ids, expert);
        };
        require(contains(main, 93U) == (id == 0) && contains(main, 426U) == (id == 1) &&
                    contains(mtp, 93U) == (id == 1) && contains(mtp, 426U) == (id == 0),
                "Main/MTP expert ownership differs");
        for (std::size_t b = 0; b < rank.experts.size(); ++b) {
            const auto& other = p.ranks[1 - id].experts[b].ids;
            for (auto e : rank.experts[b].ids)
                require(e < 512 && !std::ranges::binary_search(other, e),
                        "expert duplicated between owners");
        }
        auto load    = qwen::resolve_rank_load(reader, def, std::move(rank), Device::cuda(0));
        auto profile = qwen::execution::describe_model(*def, load.weight_data(), load.expert_data(),
                                                       load.rank_plan());
        require(profile->local.size == 1 && !qwen::describe_moe_plan(*profile, options.moe) &&
                    profile->text.layers.front().moe.subset == ops::ExpertSubsetKind::MaskOnes &&
                    profile->mtp->layer.moe.subset == ops::ExpertSubsetKind::MaskOnes,
                "TP1 execution description rebound an unowned full bank");
        if (materialize == id) {
            DeviceContext device(0);
            auto model = qwen::materialize_model(std::move(load), device);
            check_device(reader, *model,
                         [&](auto b) -> const auto& { return *model->resident_experts(b); });
        }
    }
}

void glm_case(const artifact::Reader& reader, const std::filesystem::path& path,
              const EngineOptions& options, bool mixed, std::optional<std::uint32_t> materialize) {
    auto def = glm::read_definition(reader, models::load_options(options));
    for (auto ep : (mixed ? std::vector{2U} : std::vector{8U, 2U})) {
        auto d = mixed ? deployment(2, 4, 2, ep, path) : deployment(4, 2, 2, ep, path);
        auto p = glm::plan_parallel(*def, d);
        for (const auto& planned : p.ranks) {
            auto rank = roundtrip(
                glm::RankLoadRequest{.artifact = reader.artifact_id(),
                                     .rank     = planned,
                                     .options  = options,
                                     .batches  = models::make_batch_execution_spec(1, 2, false)},
                p.topology, glm::decode_rank_load);
            const auto id    = rank.placement.global_rank;
            const auto local = id % d.parallel.tp;
            const auto owner = mixed ? local / 2 : ep == 8 ? id : local;
            const auto lane  = mixed ? local % 2 : 0U;
            require(rank.placement.tp.rank == local && rank.placement.ep.rank == owner &&
                        rank.placement.dp.rank == id / d.parallel.tp &&
                        rank.placement.adp.rank == (mixed ? local / 2 : local),
                    "GLM role identity differs");
            const auto width = mixed ? 1024U : 2048U;
            const auto count = 288U / ep;
            for (const auto& b : rank.experts) {
                require(b.ids.size() == count &&
                            b.intermediate == models::IndexRange{lane * width, width},
                        "GLM intermediate extent differs");
                for (std::uint32_t i = 0; i < count; ++i)
                    require(b.ids[i] == owner * count + i, "GLM expert ID differs");
            }
            const auto heads = mixed ? 32U : 64U;
            require(rank.mla_heads == models::IndexRange{lane * heads, heads},
                    "GLM head interval differs");
            const auto& expert = p.topology.role_group("expert", id).members;
            require(expert.size() == (ep == 8 ? 8U : d.parallel.tp) &&
                        expert.front() == (ep == 8 ? 0U : id - local),
                    "GLM expert group differs");
            if (mixed)
                require(p.topology.role_group("attention_owner", id).members ==
                                std::vector<dist::RankId>{id - lane, id - lane + 1} &&
                            p.topology.role_group("expert_tensor", id).members ==
                                p.topology.role_group("attention_owner", id).members,
                        "GLM two-member shard group differs");
            if (materialize == id) {
                DeviceContext device(0);
                auto model = glm::materialize_model(
                    glm::resolve_rank_load(reader, def, std::move(rank), device.device()), device);
                check_device(reader, *model,
                             [&](auto b) -> const auto& { return model->experts(b); });
            }
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3 && argc != 4)
            throw std::invalid_argument("usage: ninfer_model_rank_load_test "
                                        "qwen-dp2|glm-rank7|glm-mixed ARTIFACT [MATERIALIZE_RANK]");
        const std::string_view name = argv[1];
        const auto path             = std::filesystem::absolute(argv[2]);
        std::optional<std::uint32_t> rank;
        if (argc == 4) {
            std::uint32_t value;
            const std::string_view arg = argv[3];
            const auto [end, err] = std::from_chars(arg.data(), arg.data() + arg.size(), value);
            require(err == std::errc{} && end == arg.data() + arg.size() &&
                        value < (name == "qwen-dp2" ? 2U : 8U),
                    "invalid materialization rank");
            rank = value;
        }
        EngineOptions options;
        options.enable_vision       = true;
        options.speculative.backend = SpeculativeBackend::Mtp;
        options.use_cuda_graph      = false;
        artifact::Reader reader(path);
        if (name == "qwen-dp2")
            qwen_case(reader, path, options, rank);
        else if (name == "glm-rank7" || name == "glm-mixed")
            glm_case(reader, path, options, name == "glm-mixed", rank);
        else
            throw std::invalid_argument("unknown test case");
        std::cout << "PASS " << name << '\n';
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
