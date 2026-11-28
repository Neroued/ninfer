#include "models/glm5next/load.h"
#include "models/glm5next/parallel/startup.h"
#include "models/glm5next/execution/profile.h"
#include "models/glm5next/parallel/wire.h"
#include "models/qwen4_exp/load.h"
#include "models/qwen4_exp/execution/profile.h"
#include "models/qwen4_exp/parallel/wire.h"
#include "models/qwen4_exp/parallel/startup.h"
#include "models/qwen4_exp/ple/table.h"
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
    if (!rank.expert_domain) {
        require(identity.tp.size == 1 || rank.experts.empty(), "missing prepared expert domain");
        if constexpr (requires { request.rank.layers; }) {
            ++request.rank.layers.begin;
            rejected([&] { read(encode(), identity); }, "shard digest");
        }
        return rank;
    }
    const auto& domain = *rank.expert_domain;
    require(domain.members == topology.role_group("expert", id).members,
            "dispatch peer order differs");
    const auto owners = models::expert_replica_members(topology, topology.role_group("expert", id));
    require(domain.owner_count() == owners.size(), "expert owner count differs from topology");
    for (std::size_t peer = 0; peer < domain.members.size(); ++peer)
        require(owners.at(domain.peer_owners.at(peer)) ==
                    topology.replica_for_rank(domain.members[peer]).replica_id,
                "expert peer assigned to the wrong source owner");
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
    using qwen::ple::Placement;
    const auto split = qwen::ple::shard_range;
    require(split(Placement::Sharded, 5, {0, 3}) == models::IndexRange{0, 2} &&
                split(Placement::Sharded, 5, {2, 3}) == models::IndexRange{4, 1} &&
                split(Placement::Sharded, 2, {2, 3}) == models::IndexRange{2, 0},
            "unequal PLE ranges lost coverage or an empty participant");
    const auto forced = qwen::ple::placements({PleStorage::Auto, PleDeviceLayout::Sharded}, 5, 1);
    require(forced == std::vector{Placement::Replicated},
            "one-consumer sharded PLE was not canonicalized");
    bool conflict = false;
    try {
        (void)qwen::ple::placements({PleStorage::Disk, PleDeviceLayout::Replicated}, 5, 2);
    } catch (const std::invalid_argument&) { conflict = true; }
    require(conflict, "explicit Disk and device layout conflict was accepted");
    for (unsigned ranks : {8U, 16U}) {
        auto options_ep            = options;
        options_ep.deployment      = deployment(ranks / 2, 2, 1, ranks, path);
        options_ep.max_concurrency = ranks;
        qwen::ParallelStartup startup(reader, options_ep);
        const auto& topology = startup.topology();
        const auto& experts  = topology.role_group("expert", 0);
        const auto replicas  = models::expert_replica_members(topology, experts);
        require(experts.members.size() == ranks && replicas.size() == ranks / 2,
                "replicated Qwen inputs need one source owner per TP pair");
        for (unsigned i = 0; i < ranks / 2; ++i)
            require(replicas[i] == i && topology.replica(i).ranks.size() == 2,
                    "expert service lost a source replica or its TP follower");
        const auto requests = startup.requests(options_ep, false);
        const auto& config  = def->config().text.moe;
        for (auto id : {0U, ranks - 1}) {
            wire::Decoder in(requests.at(id));
            auto request = qwen::decode_rank_load(in, topology.rank(id), reader.artifact_id());
            in.finish();
            const auto& domain = *request.rank.expert_domain;
            require(domain.peer_owners.at(experts.index(id)) == topology.rank(id).dp.rank &&
                        domain.owner_count() == replicas.size(),
                    "worker assigned its TP lane to a different source replica");
            auto load =
                qwen::resolve_rank_load(reader, def, std::move(request.rank), Device::cuda(0));
            const auto& local = load.rank_plan();
            const auto& w     = def->weights();
            const auto shared = config.shared_intermediate / local.placement.tp.size;
            for (const auto* layer : {&w.text.layers.front().moe, &w.mtp->layer.moe}) {
                const auto& part = local.experts.at(layer->bank.index);
                require(
                    part.ids.size() * ranks == config.experts &&
                        part.intermediate == models::IndexRange{0, config.intermediate} &&
                        local.shared_intermediate == shared &&
                        load.weight_data()[layer->shared_gate.index].view.shape.front() == shared &&
                        load.weight_data()[layer->shared_down.index].view.shape.back() == shared,
                    "Main or MTP rank bound the wrong routed or shared projection");
            }
        }
    }
    auto p = qwen::plan_parallel(*def, deployment(2, 1, 1, 2, path), qwen::ple::Placement::Disk);
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
        if (id == 0) {
            const auto& domain = load.rank_plan().expert_domain;
            const auto& group  = p.topology.role_group("expert", id);
            const auto& layer  = profile->text.layers.front().moe;
            const auto prepare = [&](bool overlap) {
                models::moe::ResidentExchangePlan plan(domain, group, group.index(id),
                                                       layer.math.routed.ffn.hidden,
                                                       layer.math.routed.top_k, {2, 2, overlap});
                const auto scratch = plan.bind_layer(layer.bank, layer.math, profile->local.size);
                return std::pair{plan.bytes(), scratch};
            };
            const auto serial = prepare(false), overlap = prepare(true);
            require(overlap.first > serial.first && overlap.second > serial.second,
                    "two live expert windows were budgeted as a single frame or workspace");
        }
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

void qwen_pipeline_capacity(const artifact::Reader& reader, const std::filesystem::path& path,
                            EngineOptions options) {
    auto deployment_options        = deployment(2, 1, 1, 1, path);
    deployment_options.parallel.dp = 1;
    deployment_options.parallel.pp = 2;
    options.deployment             = deployment_options;
    options.memory_headroom_bytes  = 0;
    options.context_capacity       = ContextCapacityPolicy::automatic();
    options.ple.storage            = PleStorage::Device;
    qwen::ParallelStartup startup(reader, options);
    const auto& def = startup.definition();
    const auto original =
        qwen::plan_parallel(*def, deployment_options, qwen::ple::Placement::Replicated);
    const auto choose       = startup.select_shards(reader, options);
    const auto weight_bytes = [&](const auto& rank) {
        return qwen::resolve_rank_load(reader, def, rank, Device::cuda(0))
            .materialization()
            .device_capacity_bytes;
    };
    const auto table = qwen::plan_ple_table(reader, *def, qwen::ple::Placement::Replicated,
                                            original.ranks[0].placement.tp, Device::cuda(0))
                           .device_bytes();
    std::array<dist::RankMemoryOffer, 2> offers{};
    for (unsigned i = 0; i < 2; ++i)
        offers[i].available_bytes =
            weight_bytes(original.ranks[i]) + (original.ranks[i].owns_ple ? table : 0);
    require(!choose(offers), "fitting Qwen4Exp pipeline changed its partition");
    const auto cut = def->config().text.ple.layer;
    if (!cut || !original.ranks[0].owns_ple) return;
    offers[1].available_bytes += offers[0].available_bytes;
    const auto first          = qwen::plan_rank_layers(*def, original.topology.rank(0),
                                                       qwen::ple::Placement::Replicated, {0, cut});
    offers[0].available_bytes = weight_bytes(first);
    const auto reassigned     = choose(offers);
    require(reassigned && reassigned->plan == startup.topology().plan &&
                reassigned->plan != original.topology.plan,
            "PLE device allowance did not change the shard assignment");
    const auto payloads = startup.requests(options);
    for (unsigned i = 0; i < 2; ++i) {
        wire::Decoder in(payloads[i]);
        const auto req =
            qwen::decode_rank_load(in, startup.topology().rank(i), reader.artifact_id());
        in.finish();
        require(req.rank.owns_ple == (i == 1) && (i || req.rank.layers.size <= cut) &&
                    weight_bytes(req.rank) + (req.rank.owns_ple ? table : 0) <=
                        offers[i].available_bytes,
                "reassigned table or layer weights exceed their physical allowance");
    }
    std::cout << "PP2 capacity moved the PLE table to stage 1" << std::endl;
}

void qwen_pipeline(const artifact::Reader& reader, const std::filesystem::path& path,
                   EngineOptions o) {
    o.speculative = {};
    qwen_pipeline_capacity(reader, path, o);
    const auto def   = qwen::read_definition(reader, models::load_options(o));
    const auto& c    = def->config().text;
    const auto& text = def->weights().text;
    const auto n     = static_cast<std::uint32_t>(c.layers.size());
    const auto check = [&](std::uint32_t stages, std::uint32_t tp,
                           std::vector<std::uint32_t> counts) {
        auto d                  = deployment(stages, tp, 1, tp, path);
        d.parallel.dp           = 1;
        d.parallel.pp           = stages;
        d.parallel.stage_layers = std::move(counts);
        auto plan               = qwen::plan_parallel(*def, d, qwen::ple::Placement::Disk);
        unsigned covered = 0, ple_owners = 0;
        for (const auto& proposed : plan.ranks) {
            const auto rank = roundtrip(
                qwen::RankLoadRequest{.artifact = reader.artifact_id(),
                                      .rank     = proposed,
                                      .options  = o,
                                      .batches  = models::make_batch_execution_spec(2, 1, false)},
                plan.topology, qwen::decode_rank_load);
            const auto stage = rank.placement.pp.rank;
            const auto own   = [&](std::size_t i) {
                return i >= rank.layers.begin && i - rank.layers.begin < rank.layers.size;
            };
            if (!rank.placement.tp.rank) {
                require(rank.layers.begin == covered, "pipeline intervals do not meet");
                covered += rank.layers.size;
                ple_owners += rank.owns_ple;
            }
            require(own(c.ple.layer) == rank.owns_ple,
                    "different stage owns the PLE table and injection");
            auto load        = qwen::resolve_rank_load(reader, def, rank, Device::cuda(0));
            const auto bound = load.weight_data();
            const auto has   = [&](auto id) { return !bound[id.index].view.parts.empty(); };
            require(has(text.embedding) == (stage == 0) &&
                        has(text.head) == (stage + 1 == stages) &&
                        has(text.ple.key) == rank.owns_ple &&
                        has(text.ple.convolution) == rank.owns_ple,
                    "pipeline endpoint or PLE parameters bound on the wrong stage");
            for (std::size_t i = 0; i < n; ++i)
                require(has(text.layers[i].attention_hc.norm) == own(i) &&
                            has(text.layers[i].moe.router) == own(i),
                        "HC or router binding differs from the stage interval");
            require(load.expert_data().size() == rank.layers.size,
                    "foreign expert bank was materialized");
            // Weight binding is CPU-only. The production projection profiles also query
            // the compiled Op catalog; multi-lane Flash Next is qualified on Spark.
            if (tp == 1 || NINFER_TARGET_SPARK) {
                const auto profile =
                    qwen::execution::describe_model(*def, bound, load.expert_data(), rank);
                require(profile->config.text.gdn_layers + profile->config.text.qsa_layers ==
                                rank.layers.size &&
                            bool(profile->vision) == (stage == 0 && o.enable_vision),
                        "rank state or visual ownership differs");
            }
        }
        require(covered == n && ple_owners == 1, "incomplete Qwen4Exp pipeline");
        std::cout << "PP" << stages << "/TP" << tp << " layers";
        for (const auto& stage : plan.topology.replica(0).stage_ranks)
            std::cout << ' ' << plan.ranks[stage.front()].layers.size;
        std::cout << " PLE layer " << c.ple.layer << std::endl;
    };
    check(2, 1, {});
    if (c.ple.layer) check(2, 1, {c.ple.layer, n - c.ple.layer});
    if (c.ple.layer + 1 < n) check(2, 1, {c.ple.layer + 1, n - c.ple.layer - 1});
    if (c.ple.layer != 1) check(2, 1, {1, n - 1});
    check(4, 2, {});
}

void glm_pipeline_capacity(const artifact::Reader& reader, const std::filesystem::path& path,
                           EngineOptions options) {
    auto d                        = deployment(2, 1, 1, 1, path);
    d.parallel.dp                 = 1;
    d.parallel.pp                 = 2;
    options.deployment            = d;
    options.speculative           = {};
    options.memory_headroom_bytes = 0;
    options.context_capacity      = ContextCapacityPolicy::automatic();
    glm::ParallelStartup startup(reader, options);
    const auto selected = startup.select_shards(reader, options);
    const auto original = glm::plan_parallel(*startup.definition(), d);
    std::array<dist::RankMemoryOffer, 2> offers{};
    std::array<std::uint64_t, 2> bytes{};
    for (unsigned i = 0; i < 2; ++i) {
        auto load = glm::resolve_rank_load(reader, startup.definition(), original.ranks.at(i),
                                           Device::cuda(i));
        bytes[i]  = load.materialization().device_capacity_bytes;
        offers[i] = {bytes[i], false};
    }
    require(!selected(offers), "fitting pipeline changed its work-optimal partition");
    const auto old_layers     = original.ranks.front().layers.size;
    offers[0].available_bytes = bytes[0] - 1;
    offers[1].available_bytes = bytes[0] + bytes[1];
    auto replacement          = selected(offers);
    require(replacement && replacement->plan != original.topology.plan &&
                replacement->plan == startup.topology().plan,
            "short first device did not trigger a new model shard assignment");
    auto payloads = startup.requests(options);
    wire::Decoder in(payloads.front());
    const auto req = glm::decode_rank_load(in, startup.topology().rank(0), reader.artifact_id());
    in.finish();
    require(req.rank.layers.size < old_layers, "physical allowance did not move a layer");
    auto load = glm::resolve_rank_load(reader, startup.definition(), req.rank, Device::cuda(0));
    require(load.materialization().device_capacity_bytes <= offers[0].available_bytes,
            "selected rank exceeds its advertised physical capacity");
    const auto half = std::max(bytes[0], bytes[1]);
    offers          = {dist::RankMemoryOffer{half, true}, dist::RankMemoryOffer{half, true}};
    rejected([&] { (void)selected(offers); }, "physical memory pools");
    std::cout << "PP2 capacity moved stage 0 from " << old_layers << " to " << req.rank.layers.size
              << " layers; shared-pool overcommit rejected" << std::endl;
}

void glm_pipeline(const artifact::Reader& reader, const std::filesystem::path& path,
                  EngineOptions options) {
    options.speculative = {};
    auto def            = glm::read_definition(reader, models::load_options(options));
    glm_pipeline_capacity(reader, path, options);
    for (auto [stages, tp] :
         {std::pair{2U, 1U}, std::pair{4U, 2U}, std::pair{2U, 4U}, std::pair{2U, 8U}}) {
        auto d                = deployment(stages, tp, tp, 1, path);
        d.parallel.dp         = 1;
        d.parallel.pp         = stages;
        auto plan             = glm::plan_parallel(*def, d);
        std::uint32_t covered = 0;
        for (const auto& r : plan.ranks) {
            const auto local = roundtrip(
                glm::RankLoadRequest{.artifact = reader.artifact_id(),
                                     .rank     = r,
                                     .options  = options,
                                     .batches  = models::make_batch_execution_spec(2, tp, false)},
                plan.topology, glm::decode_rank_load);
            const auto rank = local.placement.tp.rank, stage = local.placement.pp.rank;
            require(local.layers == r.layers && local.banks == r.banks && local.layers.size,
                    "lost pipeline ranges");
            if (rank == 0) {
                require(local.layers.begin == covered,
                        "pipeline layer intervals overlap or leave a gap");
                covered += local.layers.size;
            }
            const auto& w    = def->weights().text;
            auto loaded      = glm::resolve_rank_load(reader, def, local, Device::cuda(0));
            const auto bound = loaded.weight_data();
            require(bound[w.embedding.index].view.parts.empty() == (stage != 0) &&
                        bound[w.head.index].view.parts.empty() == (stage + 1 != stages),
                    "stage bound another endpoint's embedding or output head");
            require(loaded.expert_data().size() == local.banks.size,
                    "stage loaded an unowned expert bank");
            for (std::size_t i = 0; i < w.layers.size(); ++i) {
                const bool owns =
                    i >= local.layers.begin && i < local.layers.begin + local.layers.size;
                require(!bound[w.layers[i].input_norm.index].view.parts.empty() == owns,
                        "stage bound a foreign layer or lost an owned layer");
            }
            auto profile = glm::execution::describe_model(*def, bound, loaded.expert_data(), local);
            require(profile->text.layers.size() == local.layers.size &&
                        profile->stage.rank == stage &&
                        profile->config.text.kda_layers + profile->config.text.mla_layers ==
                            local.layers.size,
                    "rank profile did not localize its layer state indices");
            require((tp == 1 ||
                     plan.topology.role_group("tensor", local.placement.global_rank).members ==
                         plan.topology.replica(0).stage_ranks[stage]),
                    "tensor role crossed stages");
        }
        require(covered == def->weights().text.layers.size(), "pipeline omitted target layers");
        std::cout << "PP" << stages << "/TP" << tp << " planned and bound " << covered << " layers"
                  << std::endl;
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3 && argc != 4)
            throw std::invalid_argument("usage: ninfer_model_rank_load_test "
                                        "qwen-dp2|qwen-pipeline|glm-rank7|glm-mixed|glm-pipeline "
                                        "ARTIFACT [MATERIALIZE_RANK]");
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
        else if (name == "qwen-pipeline")
            qwen_pipeline(reader, path, options);
        else if (name == "glm-pipeline")
            glm_pipeline(reader, path, options);
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
