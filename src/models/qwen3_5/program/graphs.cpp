#include "models/qwen3_5/program/graphs.h"

#include "core/nvtx.h"
#include "models/load_options.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_5 {
namespace {

std::vector<GraphProfile> profiles_through(std::uint32_t capacity,
                                           std::span<const std::uint32_t> ends) {
    if (!capacity) throw std::invalid_argument("graph context capacity must be positive");
    std::vector<GraphProfile> result;
    std::uint32_t begin = 0;
    for (const auto end : ends) {
        if (end < begin) continue;
        const auto last = std::min(end, capacity - 1);
        result.push_back({begin, last, 0});
        if (last == capacity - 1) return result;
        begin = last + 1;
    }
    result.push_back({begin, capacity - 1, 0});
    return result;
}

GraphInvocation invocation(GraphPhase phase, std::uint32_t batch, std::uint32_t width,
                           GraphProfile profile, std::uint32_t capacity) {
    const auto visible = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(capacity, std::uint64_t(profile.max) + width));
    return {phase, batch, width, profile, {1, visible}};
}

class PreparationInputs {
public:
    PreparationInputs(Resources& resources, DeviceContext& device)
        : resources_(resources), device_(device),
          pool_(0, resources.context.size(), resources.context_records) {
        const auto batch = resources.frame.spec.batch_capacity;
        std::vector<RecordRequest> demand;
        for (std::size_t i = 0; i < resources.context_records.size(); ++i)
            demand.push_back({pool_.record_class(i), batch});
        auto reserved = pool_.reserve(demand);
        if (!reserved.reservation || !reserved.reservation->ready())
            throw std::logic_error("Graph preparation Context records do not fit");
        auto& reservation = *reserved.reservation;
        for (std::uint32_t row = 0; row < batch; ++row) {
            const auto id = reservation.materialize(pool_.record_class(0));
            records_.push_back(id);
            state_offsets_.push_back(static_cast<std::int64_t>(pool_.resolve(id).offset));
        }
        initialize_tables(resources.decoder.text_kv, 1, reservation);
        if (auto* backend = resources.backend_kv()) initialize_tables(*backend, 2, reservation);
        device.synchronize();
    }

    void prepare(const GraphInvocation& call) {
        auto& frame         = resources_.frame;
        const auto cuda     = device_.execution_view();
        *frame.host_ingress = {};
        *frame.host_commit  = {};
        *frame.host_egress  = {};
        const auto maximum  = resources_.decoder.text_kv.max_context() - 1;
        const auto frontier = std::min(maximum, std::max(1U, call.profile.min));
        const auto extent   = std::min(frame.spec.draft_window, maximum - frontier);
        const auto input    = frame.host_input_view(call.width, call.batch);
        auto* ids           = static_cast<std::int32_t*>(input.ids.data);
        auto* positions     = static_cast<std::int32_t*>(input.cache_positions.data);
        auto* rope          = static_cast<std::int32_t*>(input.rope_positions.data);
        for (std::uint32_t row = 0; row < call.batch; ++row) {
            auto& host                              = *frame.host_ingress;
            host.frontiers[row]                     = frontier;
            host.backend_frontiers[row]             = frame.spec.backend == SpeculativeBackend::Mtp
                                                          ? (frontier ? frontier - 1 : 0)
                                                          : frontier;
            host.extents[row]                       = extent;
            host.valid_columns[row]                 = std::min(call.width, extent + 1);
            host.proposal_valid_columns[row]        = frame.spec.draft_window + 1;
            host.budgets[row]                       = frame.spec.draft_window + 1;
            host.main_table_rows[row]               = row;
            host.backend_table_rows[row]            = row;
            host.state_source_offsets[row]          = state_offsets_[row];
            host.state_destination_offsets[row]     = state_offsets_[row];
            host.residency_slots[row]               = row;
            frame.host_commit->history_columns[row] = extent;
            frame.host_commit->state_destination_offsets[row] = state_offsets_[row];
            for (std::uint32_t column = 0; column < call.width; ++column) {
                const auto index = row * call.width + column;
                ids[index]       = 0;
                positions[index] = frontier + std::min(column, extent);
                for (std::uint32_t axis = 0; axis < 3; ++axis) {
                    rope[axis * call.width * call.batch + index] = positions[index];
                }
            }
        }
        const auto destination = frame.input_view(call.width, call.batch);
        const auto upload      = [&](const Tensor& source, const Tensor& target) {
            CUDA_CHECK(cudaMemcpyAsync(target.data, source.data, source.bytes(),
                                            cudaMemcpyHostToDevice, cuda.stream()));
        };
        upload(input.ids, destination.ids);
        upload(input.cache_positions, destination.cache_positions);
        upload(input.rope_positions, destination.rope_positions);
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, frame.host_ingress, sizeof(FrameIngress),
                                   cudaMemcpyHostToDevice, cuda.stream()));
        CUDA_CHECK(cudaMemcpyAsync(frame.commit.data, frame.host_commit, sizeof(FrameCommit),
                                   cudaMemcpyHostToDevice, cuda.stream()));
        CUDA_CHECK(cudaMemsetAsync(frame.egress.data, 0, frame.egress.bytes, cuda.stream()));
        for (const auto offset : state_offsets_)
            resources_.state_images.storage().clear(offset, cuda);
        if (resources_.token_counts.data) {
            CUDA_CHECK(cudaMemsetAsync(resources_.token_counts.data, 0,
                                       resources_.token_counts.bytes(), cuda.stream()));
        }
        if (frame.mtp) {
            auto& mtp = *frame.mtp;
            CUDA_CHECK(cudaMemsetAsync(mtp.base_conditioning.data, 0, mtp.base_conditioning.bytes(),
                                       cuda.stream()));
            CUDA_CHECK(cudaMemsetAsync(mtp.drafts.data, 0, mtp.drafts.bytes(), cuda.stream()));
            auto* cache       = static_cast<std::int32_t*>(frame.host_step_positions.data);
            auto* step_rope   = static_cast<std::int32_t*>(frame.host_step_rope_positions.data);
            auto* valid       = static_cast<std::int32_t*>(frame.host_step_valid_columns.data);
            const auto stride = frame.spec.batch_capacity;
            for (std::uint32_t step = 0; step < frame.spec.draft_window; ++step) {
                for (std::uint32_t row = 0; row < stride; ++row) {
                    const auto p = std::min(maximum, (frontier ? frontier - 1 : 0) + step);
                    cache[step * stride + row] = p;
                    valid[step * stride + row] = row < call.batch && (step == 0 || step < extent);
                }
                for (std::uint32_t axis = 0; axis < 3; ++axis) {
                    for (std::uint32_t row = 0; row < call.batch; ++row) {
                        step_rope[step * 3 * stride + axis * call.batch + row] =
                            cache[step * stride + row];
                    }
                }
            }
            upload(frame.host_step_positions, mtp.step_positions);
            upload(frame.host_step_rope_positions, mtp.step_rope_positions);
            upload(frame.host_step_valid_columns, mtp.step_valid_columns);
        }
        if (resources_.dflash) {
            auto& state = *resources_.dflash;
            CUDA_CHECK(cudaMemsetAsync(state.pending_features.data, 0,
                                       state.pending_features.bytes(), cuda.stream()));
        }
        device_.synchronize();
    }

    void finish() {
        const auto cuda = device_.execution_view();
        for (const auto offset : state_offsets_)
            resources_.state_images.storage().clear(offset, cuda);
        if (resources_.token_counts.data) {
            CUDA_CHECK(cudaMemsetAsync(resources_.token_counts.data, 0,
                                       resources_.token_counts.bytes(), cuda.stream()));
        }
        const auto clear_table = [&](PagedKVCache& cache) {
            const auto& table = cache.tables().matrix();
            CUDA_CHECK(cudaMemsetAsync(table.data, 0xff, table.bytes(), cuda.stream()));
        };
        clear_table(resources_.decoder.text_kv);
        if (auto* backend = resources_.backend_kv()) clear_table(*backend);
        device_.synchronize();
        for (const auto record : records_) pool_.release(record);
        records_.clear();
    }

private:
    void initialize_tables(PagedKVCache& cache, std::size_t spec_index,
                           PoolReservation& reservation) {
        const auto batch     = resources_.frame.spec.batch_capacity;
        const auto count     = cache.tables().logical_page_capacity();
        const auto row_bytes = std::size_t(count) * sizeof(std::int64_t);
        tables_.emplace_back(row_bytes * batch);
        const auto host = tables_.back().span();
        for (std::uint32_t row = 0; row < batch; ++row) {
            const auto allocation = reservation.materialize(pool_.record_class(spec_index));
            records_.push_back(allocation);
            const auto binding = pool_.resolve(allocation);
            const auto staging = host.subspan(row * row_bytes, row_bytes);
            auto* offsets      = static_cast<std::int64_t*>(staging.data);
            std::fill_n(offsets, count, static_cast<std::int64_t>(binding.offset));
            cache.tables().publish(row, 0, {offsets, count}, staging, device_.execution_view());
            const auto record = resources_.context.span().subspan(binding.offset, binding.bytes);
            CUDA_CHECK(cudaMemsetAsync(record.data, 0, record.bytes, device_.stream()));
        }
    }

    Resources& resources_;
    DeviceContext& device_;
    ContextPool pool_;
    std::vector<AllocationId> records_;
    std::vector<std::int64_t> state_offsets_;
    std::vector<PinnedHostBuffer> tables_;
};

} // namespace

std::vector<GraphProfile> target_graph_profiles(std::uint32_t capacity, std::uint32_t width) {
    if (!width) throw std::invalid_argument("graph width must be positive");
    constexpr std::array tiers{128U, 512U, 2048U, 4096U, 8192U, 16384U, 32768U};
    std::vector<std::uint32_t> ends;
    for (const auto visible : tiers) {
        if (visible >= width) ends.push_back(visible - width);
    }
    return profiles_through(capacity, ends);
}

std::vector<GraphProfile> draft_graph_profiles(const GraphSpec& spec) {
    if (spec.backend == SpeculativeBackend::Mtp) {
        return target_graph_profiles(spec.capacity, spec.draft_window + 1);
    }
    if (spec.backend == SpeculativeBackend::DFlash2) {
        constexpr std::array ends{96U, 511U, 2047U, 8191U, 32767U};
        auto profiles = profiles_through(spec.capacity, ends);
        for (std::size_t index = 0; index < profiles.size(); ++index) {
            profiles[index].topology_class = static_cast<std::uint32_t>(index);
        }
        return profiles;
    }
    if (spec.backend == SpeculativeBackend::DFlash) {
        constexpr std::array ends{96U,   127U,   511U,   1023U,  2047U,   4095U,
                                  8191U, 16383U, 32767U, 65536U, 131072U, 196608U};
        auto profiles = profiles_through(spec.capacity, ends);
        for (auto& profile : profiles) profile.topology_class = profile.max > 96 ? 1 : 0;
        return profiles;
    }
    return {};
}

Graphs::Graphs(GraphSpec spec) : spec_(spec) {
    if (!spec.capacity || !spec.batch_capacity || spec.batch_capacity > kMaximumConcurrency ||
        spec.draft_window > 15 ||
        (spec.backend == SpeculativeBackend::None) != (spec.draft_window == 0)) {
        throw std::invalid_argument("Qwen3.5 graph geometry is invalid");
    }
}

void Graphs::SegmentFamily::instantiate(CudaExecutionView cuda) {
    for (std::size_t index = 0; index < profiles.size(); ++index) {
        const auto id    = profiles[index].profile.topology_class;
        const auto found = std::find_if(topologies.begin(), topologies.end(),
                                        [&](const auto& topology) { return topology.id == id; });
        if (found == topologies.end()) {
            auto& topology     = topologies.emplace_back();
            topology.id        = id;
            topology.installed = index;
            topology.executable.instantiate(profiles[index].definition);
        }
    }
    for (auto& topology : topologies) {
        const auto original = topology.installed;
        auto installed      = original;
        for (std::size_t index = 0; index < profiles.size(); ++index) {
            if (profiles[index].profile.topology_class != topology.id) continue;
            if (index != installed) topology.executable.update(profiles[index].definition);
            topology.executable.upload(cuda.stream());
            CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
            installed = index;
        }
        if (installed != original) {
            topology.executable.update(profiles[original].definition);
            topology.executable.upload(cuda.stream());
        }
    }
}

void Graphs::SegmentFamily::launch(std::uint32_t frontier, CudaExecutionView cuda) {
    const auto found = std::find_if(profiles.begin(), profiles.end(), [&](const auto& profile) {
        return profile.profile.min <= frontier && frontier <= profile.profile.max;
    });
    if (found == profiles.end()) throw std::out_of_range("graph phase frontier exceeds capacity");
    const auto index = static_cast<std::size_t>(found - profiles.begin());
    auto topology    = std::find_if(topologies.begin(), topologies.end(), [&](const auto& value) {
        return value.id == found->profile.topology_class;
    });
    if (topology == topologies.end()) throw std::logic_error("graph phase topology is unprepared");
    if (topology->installed != index) {
        topology->executable.update(found->definition);
        topology->installed = index;
    }
    topology->executable.launch(cuda.stream());
}

void Graphs::prepare(Resources& resources, const execution::Parameters& parameters,
                     DeviceContext& device, const GraphBodies& bodies) {
    if (!batches_.empty()) throw std::logic_error("Qwen3.5 graphs are already prepared");
    if (!bodies.prologue || !bodies.text || !bodies.project || !bodies.decision ||
        !bodies.readback || (spec_.draft_window && !bodies.draft)) {
        throw std::invalid_argument("Qwen3.5 graph numerical bodies are incomplete");
    }
    nvtx::ScopedRange range(nvtx::Name::CudaGraphPrepare, nvtx::Category::Graph);
    std::optional<moe::Schedule::Preparation> preparation;
    if (resources.moe) preparation.emplace(*resources.moe);
    PreparationInputs inputs(resources, device);
    batches_.resize(spec_.batch_capacity);
    const auto ordinary_profiles = target_graph_profiles(spec_.capacity, 1);
    const auto verify_profiles   = target_graph_profiles(spec_.capacity, spec_.draft_window + 1);
    const auto draft_profiles    = draft_graph_profiles(spec_);
    for (std::uint32_t batch = 1; batch <= spec_.batch_capacity; ++batch) {
        auto& target            = batches_[batch - 1];
        const auto prepare_main = [&](GraphPhase phase, execution::TextGraph& graph,
                                      std::uint32_t width, const auto& profiles) {
            const auto call = invocation(phase, batch, width, profiles.front(), spec_.capacity);
            inputs.prepare(call);
            resources.work.reset();
            resources.work.reset_peak();
            auto run = bodies.text(call);
            if (!run) throw std::logic_error("graph target body returned no traversal");
            graph.prepare(
                *run, parameters, resources.moe.get(), profiles, width,
                [&] { bodies.prologue(call); },
                [&] {
                    auto scope = resources.work.scope();
                    bodies.project(call);
                },
                resources.work, device);
            workspace_peak_ = std::max(workspace_peak_, graph.workspace_peak());
        };
        prepare_main(GraphPhase::Main, target.main, 1, ordinary_profiles);
        if (spec_.draft_window) {
            prepare_main(GraphPhase::Verify, target.verify, spec_.draft_window + 1,
                         verify_profiles);
        }
        const auto prepare_segment = [&](SegmentFamily& family, GraphPhase phase,
                                         const auto& profiles, std::uint32_t width) {
            family.profiles.reserve(profiles.size());
            for (const auto profile : profiles) {
                const auto call = invocation(phase, batch, width, profile, spec_.capacity);
                inputs.prepare(call);
                resources.work.reset();
                resources.work.reset_peak();
                const auto body = [&] {
                    auto scope = resources.work.scope();
                    if (phase == GraphPhase::Draft)
                        bodies.draft(call);
                    else {
                        bodies.decision(call);
                        bodies.readback(call);
                    }
                };
                body();
                device.synchronize();
                auto& stored   = family.profiles.emplace_back();
                stored.profile = profile;
                stored.definition.capture(device.stream(), body);
                workspace_peak_ = std::max(workspace_peak_, resources.work.peak_used());
            }
            family.instantiate(device.execution_view());
            device.synchronize();
        };
        if (spec_.draft_window) {
            prepare_segment(target.draft, GraphPhase::Draft, draft_profiles,
                            spec_.draft_window + 1);
        }
        const std::array decision_profiles{GraphProfile{0, spec_.capacity - 1, 0}};
        prepare_segment(target.sample, GraphPhase::Sample, decision_profiles, 1);
        if (spec_.draft_window) {
            prepare_segment(target.accept, GraphPhase::Accept, decision_profiles,
                            spec_.draft_window + 1);
        }
        if (bodies.mtp_history && spec_.draft_window) {
            inputs.prepare(invocation(GraphPhase::Verify, batch, spec_.draft_window + 1,
                                      {0, spec_.capacity - 1, 0}, spec_.capacity));
            auto& frame = resources.frame;
            for (std::uint32_t row = 0; row < batch; ++row)
                frame.host_commit->history_columns[row] = spec_.draft_window;
            CUDA_CHECK(cudaMemcpyAsync(frame.commit.data, frame.host_commit, sizeof(FrameCommit),
                                       cudaMemcpyHostToDevice, device.stream()));
            resources.work.reset_peak();
            bodies.mtp_history(batch);
            device.synchronize();
            auto& history = target.mtp_history;
            history.definition.capture(device.stream(), [&] { bodies.mtp_history(batch); });
            history.executable.instantiate(history.definition);
            history.executable.upload(device.stream());
            workspace_peak_ = std::max(workspace_peak_, resources.work.peak_used());
            device.synchronize();
        }
    }
    if (resources.moe) resources.moe->restore_cache_after_preparation();
    inputs.finish();
    resources.work.reset();
}

bool Graphs::start(GraphPhase phase, std::uint32_t batch, std::uint32_t frontier,
                   CudaExecutionView cuda) {
    if (pending_) throw std::logic_error("a Qwen3.5 graph phase is already pending");
    if (!batch || batch > batches_.size()) throw std::out_of_range("graph batch is unprepared");
    auto& family               = batches_[batch - 1];
    execution::TextGraph* text = nullptr;
    switch (phase) {
    case GraphPhase::Main:
        text = &family.main;
        break;
    case GraphPhase::Verify:
        text = &family.verify;
        break;
    case GraphPhase::Draft:
        family.draft.launch(frontier, cuda);
        return true;
    case GraphPhase::Sample:
        family.sample.launch(frontier, cuda);
        return true;
    case GraphPhase::Accept:
        family.accept.launch(frontier, cuda);
        return true;
    }
    text->select(frontier);
    pending_ = text;
    try {
        if (!text->start(cuda)) return false;
        pending_ = nullptr;
        return true;
    } catch (...) {
        cancel();
        throw;
    }
}

bool Graphs::poll() {
    if (!pending_) return true;
    if (!pending_->poll()) return false;
    pending_ = nullptr;
    return true;
}

void Graphs::rebuild_mtp_history(std::uint32_t batch, CudaExecutionView cuda) {
    if (pending_) throw std::logic_error("MTP history cannot precede its completed target");
    batches_.at(batch - 1).mtp_history.executable.launch(cuda.stream());
}

void Graphs::cancel() {
    if (pending_) pending_->cancel();
    pending_ = nullptr;
}

void Graphs::clear() {
    if (pending_) throw std::logic_error("cannot release an active Qwen3.5 graph");
    batches_.clear();
    workspace_peak_ = 0;
}

} // namespace ninfer::models::qwen3_5
