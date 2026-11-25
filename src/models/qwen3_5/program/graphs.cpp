#include "models/qwen3_5/program/internal.h"

#include "core/nvtx.h"
#include "models/qwen3_5/execution/text.h"

#include <algorithm>
#include <stdexcept>
#include <thread>
#include <type_traits>

namespace ninfer::models::qwen3_5::detail {
namespace {

class PreparationInputs {
public:
    PreparationInputs(Resources& resources, DeviceContext& device, const BatchExecutionPlan& plan)
        : resources_(resources), device_(device), plan_(plan),
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

    void stage(const speculative::RoundRoute& route) {
        const auto batch = route.shape.rows, width = route.width;
        auto& frame     = resources_.frame;
        const auto cuda = device_.execution_view();
        frame.bind_rows(batch);
        frame.clear_ingress();
        frame.clear_commit();
        frame.clear_egress();
        const auto live     = plan_.representative_owners({batch, batch}).front();
        const auto maximum  = resources_.decoder.text_kv.max_context() - 1;
        const auto frontier = std::min(maximum, 1U);
        const auto room     = maximum - frontier;
        const auto extent   = resources_.dflash && room < frame.spec.draft_window
                                  ? 0U
                                  : std::min(route.verification_limit(frame.spec.backend), room);
        const auto input    = frame.host_decode_view(width, batch);
        auto* positions     = static_cast<std::int32_t*>(input.cache_positions.data);
        auto* rope          = static_cast<std::int32_t*>(input.rope_positions.data);
        for (std::uint32_t row = 0; row < live; ++row) {
            auto& host                              = frame.host_ingress;
            host.live_rows[row]                     = 1;
            host.frontiers[row]                     = frontier;
            host.sampling_positions[row]            = frontier;
            host.history_skip[row]                  = frontier == 0 || extent > 0;
            frame.host_commit.accepted_columns[row] = extent + 1;
            host.backend_frontiers[row]             = frame.spec.backend == SpeculativeBackend::Mtp
                                                          ? (frontier ? frontier - 1 : 0)
                                                          : frontier;
            host.extents[row]                       = extent;
            host.valid_columns[row]                 = std::min(width, extent + 1);
            host.proposal_valid_columns[row]        = extent ? frame.spec.draft_window + 1 : 0;
            host.proposal_live_rows[row]            = extent != 0;
            host.main_table_rows[row]               = row;
            host.backend_table_rows[row]            = row;
            host.state_source_offsets[row]          = state_offsets_[row];
            host.state_destination_offsets[row]     = state_offsets_[row];
            host.residency_slots[row]               = row;
            frame.host_commit.state_destination_offsets[row] = state_offsets_[row];
            for (std::uint32_t column = 0; column < width; ++column) {
                const auto index = row * width + column;
                positions[index] = frontier + std::min(column, extent);
                for (std::uint32_t axis = 0; axis < 3; ++axis) {
                    rope[axis * width * batch + index] = positions[index];
                }
            }
        }
        if (width > 1) {
            const auto ids = frame.decode_view(width, batch).ids;
            CUDA_CHECK(cudaMemsetAsync(ids.data, 0, ids.bytes(), cuda.stream()));
        }
        CUDA_CHECK(cudaMemcpyAsync(frame.commit.data, frame.host_commit_bytes.data,
                                   frame.commit.bytes, cudaMemcpyHostToDevice, cuda.stream()));
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
            const auto stride = batch;
            for (std::uint32_t step = 0; step < frame.spec.draft_window; ++step) {
                for (std::uint32_t row = 0; row < stride; ++row) {
                    const auto p = std::min(maximum, (frontier ? frontier - 1 : 0) + step);
                    cache[step * stride + row] = row < live ? p : 0;
                    valid[step * stride + row] = row < live && step < extent;
                }
                for (std::uint32_t axis = 0; axis < 3; ++axis) {
                    for (std::uint32_t row = 0; row < live; ++row) {
                        step_rope[step * 3 * stride + axis * batch + row] =
                            cache[step * stride + row];
                    }
                }
            }
        }
        if (resources_.dflash) {
            auto& state = *resources_.dflash;
            CUDA_CHECK(cudaMemsetAsync(state.pending_features.data, 0,
                                       state.pending_features.bytes(), cuda.stream()));
        }
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, frame.host_ingress_bytes.data,
                                   frame.ingress.bytes, cudaMemcpyHostToDevice, cuda.stream()));
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
    const BatchExecutionPlan& plan_;
    ContextPool pool_;
    std::vector<AllocationId> records_;
    std::vector<std::int64_t> state_offsets_;
    std::vector<PinnedHostBuffer> tables_;
};

template <class Readout>
auto capture_readout(const Readout& call) {
    return [call](ops::ContextBucket bucket, CudaExecutionView cuda) {
        call.execute(bucket, cuda);
    };
}

template <class Readout>
ContextBucketBinding bind_readout(DeviceContext& device, ops::ContextBucketRange buckets,
                                  const Readout& call, std::size_t offset = 0) {
    ContextBucketSite::Profile profile;
    if constexpr (std::is_same_v<Readout, execution::CausalAttentionReadout>)
        profile = [call, sms = device.multiprocessor_count()](auto bucket) {
            return call.profile(bucket, sms);
        };
    else
        profile = [call](auto bucket) { return call.profile(bucket); };
    return {buckets, std::move(profile), capture_readout(call), offset};
}

template <class Cache>
auto paged_operands(const Cache& cache) {
    return std::array<const void*, 5>{cache.k_pages.data(), cache.v_pages.data(),
                                      cache.k_scale_pages.data(), cache.v_scale_pages.data(),
                                      cache.k_pages.records.page_offsets.data};
}

template <class Readout>
auto readout_operands(const Readout& call) {
    if constexpr (std::is_same_v<Readout, execution::CausalAttentionReadout>) {
        return std::tuple{call.positions.data, call.table_rows.data, paged_operands(call.cache)};
    } else if constexpr (std::is_same_v<Readout, execution::ContextAttentionReadout>) {
        return std::tuple{call.frontiers.data, call.table_rows.data, paged_operands(call.context)};
    } else {
        const auto& k = call.context.k;
        const auto& v = call.context.v;
        return std::tuple{call.positions.data,    k.records.byte_offsets.data,
                          k.records.storage.data, k.field.tensor.region.offset,
                          v.records.storage.data, v.field.tensor.region.offset};
    }
}

template <class Readout>
void record_readout(const ReadoutRecorder& record, const Readout& expected,
                    const Readout& actual) {
    if (expected.query.data != actual.query.data || expected.key.data != actual.key.data ||
        expected.value.data != actual.value.data || expected.output.data != actual.output.data ||
        expected.valid_columns.data != actual.valid_columns.data ||
        expected.scratch.data != actual.scratch.data ||
        expected.scratch.bytes != actual.scratch.bytes ||
        readout_operands(expected) != readout_operands(actual))
        throw std::logic_error(
            "Qwen3.5 graph readout differs from its bound workspace or operands");
    record(capture_readout(actual));
}

ReadoutCapture bind_mtp_step(RankProgramImpl& p, ops::ContextBucketRange buckets,
                             std::uint32_t batch, std::uint32_t step) {
    ReadoutCapture capture;
    p.bind_mtp_draft_step(batch, step, [&](const execution::CausalAttentionReadout& call) {
        capture.readouts.push_back(bind_readout(p.device, buckets, call));
        capture.body = [&p, batch, step, call,
                        bucket = buckets.first](const ReadoutRecorder& record) {
            p.run_mtp_draft_step(batch, step, bucket, [&](const auto& actual) {
                record_readout(record, call, actual);
            });
        };
    });
    return capture;
}

ReadoutCapture bind_dflash(RankProgramImpl& p, ops::ContextBucketRange buckets,
                           const speculative::RoundRoute& route) {
    ReadoutCapture capture;
    std::vector<execution::SlidingWindowReadout> local;
    std::vector<execution::ContextAttentionReadout> full;
    p.bind_dflash_draft(route,
                        {buckets,
                         [&](const auto& call) {
                             local.push_back(call);
                             capture.readouts.push_back(bind_readout(p.device, buckets, call));
                         },
                         [&](const auto& call) {
                             full.push_back(call);
                             capture.readouts.push_back(bind_readout(p.device, buckets, call, 1));
                         }});
    capture.body = [&p, route, buckets, local = std::move(local),
                    full = std::move(full)](const ReadoutRecorder& record) {
        std::size_t next_local = 0, next_full = 0;
        p.run_dflash_draft(route, std::span(&buckets.first, 1),
                           {buckets,
                            [&](const auto& actual) {
                                record_readout(record, local.at(next_local++), actual);
                            },
                            [&](const auto& actual) {
                                record_readout(record, full.at(next_full++), actual);
                            }});
    };
    return capture;
}

void proposal_prefix(RankProgramImpl& p, const speculative::RoundRoute& route, bool capture) {
    p.gpu_timer.begin_draft(capture);
    if (p.backend == SpeculativeBackend::Mtp) {
        p.upload_ingress();
        p.gather_mtp_conditioning(route.shape.rows);
    } else {
        p.append_draft_history(route.shape.rows, {0, p.draft_window + 1});
    }
}

ProposalCapture bind_proposal(RankProgramImpl& p, ops::ContextBucketRange buckets,
                              const speculative::RoundRoute& route) {
    ProposalCapture capture;
    capture.prefix          = [&p, route] { proposal_prefix(p, route, true); };
    capture.before_readback = [&p] { p.gpu_timer.end_draft(true); };
    capture.readback        = [&p, route] { p.readback_drafts(route); };
    capture.drafts_ready    = [&p] { p.drafts_ready.record_external(p.device.stream()); };
    if (route.mtp_steps) {
        for (std::uint32_t step = 0; step < route.mtp_steps; ++step)
            capture.steps.push_back({.forward = bind_mtp_step(p, buckets, route.shape.rows, step)});
    } else {
        capture.steps.push_back({.forward = bind_dflash(p, buckets, route)});
    }
    return capture;
}

std::unique_ptr<execution::TextRun> target_run(RankProgramImpl& p,
                                               const speculative::RoundRoute& route,
                                               ops::ContextBucket bucket, std::uint32_t active) {
    auto& resources  = p.resources;
    auto& frame      = resources.frame;
    const auto batch = static_cast<int>(route.shape.rows), width = static_cast<int>(route.width);
    auto input = frame.decode_view(width, batch);
    execution::TextInputs inputs{input.ids,
                                 input.cache_positions,
                                 input.rope_positions,
                                 frame.valid_columns.slice(0, 0, batch),
                                 frame.main_table_rows.slice(0, 0, batch),
                                 frame.state_source_offsets.slice(0, 0, batch),
                                 frame.state_destination_offsets.slice(0, 0, batch),
                                 {},
                                 active};
    execution::TextStateUpdate update = execution::StepUpdate{};
    if (width > 1) update.emplace<execution::ReplayRecords>(resources.replay(width).records);
    std::optional<execution::DFlashFeatureSink> features;
    if (resources.dflash) {
        features.emplace();
        features->features        = resources.dflash->pending_features.slice(1, 0, width);
        features->residency_slots = frame.residency_slots.slice(0, 0, batch);
        features->layers          = p.parameters.profile->config.draft->target_layer_ids;
    }
    return std::make_unique<execution::TextRun>(
        execution::TextExecutionView{p.parameters, resources.work, p.device.execution_view(),
                                     resources.decoder.text_kv, resources.state_images.linear(),
                                     resources.moe.get()},
        std::move(inputs), update, bucket, input.hidden, features ? &*features : nullptr);
}

void target_prefix(RankProgramImpl& p, const speculative::RoundRoute& route, bool capture) {
    p.gpu_timer.begin_target(capture);
    if (route.width == 1 && !p.resources.dflash) p.upload_ingress();
    if (route.width > 1) p.prepare_verification(route, route);
}

GraphBody target_head(RankProgramImpl& p, const speculative::RoundRoute& route) {
    return [&p, route] {
        auto scope = p.resources.work.scope();
        p.project_decode(route.shape.rows, route.width);
    };
}

ResidentTargetCapture bind_resident(RankProgramImpl& p, ops::ContextBucketRange buckets,
                                    const speculative::RoundRoute& route) {
    ResidentTargetCapture capture;
    std::vector<execution::CausalAttentionReadout> expected;
    const auto& layers = p.parameters.text.layers;
    {
        // A fused proposal uses the same arena before the target residual becomes live.
        auto run = target_run(p, route, buckets.first, 0);
        for (std::uint32_t layer = 0; layer < layers.size(); ++layer)
            if (std::holds_alternative<execution::AttentionParameters>(layers[layer].mixer))
                run->bind_attention_readout(layer, [&](const auto& call) {
                    expected.push_back(call);
                    capture.readouts.push_back(bind_readout(p.device, buckets, call));
                });
    }
    capture.body = [&p, route, bucket = buckets.first, expected = std::move(expected)](
                       const ReadoutRecorder& record, const GraphBody&, const GraphBody& head,
                       const StreamSectionRecorder&) {
        auto run = target_run(p, route, bucket, 0);
        target_prefix(p, route, true);
        run->prefix();
        std::size_t next = 0;
        for (std::uint32_t layer = 0; layer < p.parameters.text.layers.size(); ++layer) {
            run->mixer(layer, [&](const auto& actual) {
                record_readout(record, expected.at(next++), actual);
            });
            run->post_mixer(layer);
            run->capture_features(layer);
        }
        run->finish();
        head();
        p.gpu_timer.end_target(true);
    };
    capture.head = target_head(p, route);
    return capture;
}

OffloadedTargetCapture bind_offloaded(RankProgramImpl& p, ops::ContextBucketRange buckets,
                                      const speculative::RoundRoute& route) {
    const auto geometry = moe::decode_geometry(*p.batch_execution, route.shape.rows, route.width);
    // The suite borrows this traversal during capture and owns the schedule activity at replay.
    auto run = std::shared_ptr(target_run(p, route, buckets.first, 0));
    OffloadedTargetCapture capture{.output          = run->residual(),
                                   .physical_tokens = geometry.physical_tokens};
    capture.prefix = [&p, route, run] {
        target_prefix(p, route, true);
        run->prefix();
    };
    const auto& layers = p.parameters.text.layers;
    for (std::uint32_t layer = 0; layer < layers.size(); ++layer) {
        const auto* routed = std::get_if<moe::Parameters>(&layers[layer].ffn);
        ReadoutCapture mixer;
        std::optional<execution::CausalAttentionReadout> expected;
        if (std::holds_alternative<execution::AttentionParameters>(layers[layer].mixer))
            run->bind_attention_readout(layer, [&](const auto& call) {
                expected = call;
                mixer.readouts.push_back(bind_readout(p.device, buckets, call));
            });
        mixer.body = [run, layer, routed = bool(routed), expected](const ReadoutRecorder& record) {
            execution::CausalAttentionSubmit readout;
            if (expected)
                readout = [&](const auto& actual) { record_readout(record, *expected, actual); };
            run->mixer(layer, readout);
            if (!routed) {
                run->post_mixer(layer);
                run->capture_features(layer);
            }
        };
        if (routed) {
            capture.layers.emplace_back(OffloadedLayerCapture{
                routed->bank, geometry, std::move(mixer),
                [run, layer](const moe::GpuCall& call) { run->post_mixer(layer, &call); },
                [run, layer] { run->capture_features(layer); }});
        } else {
            capture.layers.emplace_back(std::move(mixer));
        }
    }
    capture.suffix     = [run] { run->finish(); };
    capture.head       = target_head(p, route);
    capture.after_head = [&p] { p.gpu_timer.end_target(true); };
    return capture;
}

} // namespace

GraphPreparationSample RankProgramImpl::initialize_graphs(GraphPreparationMode mode) {
    if (!use_cuda_graph || resources.frame.spec.causal_scoring) return {};
    if (graphs) throw std::logic_error("Qwen3.5 graphs are already initialized");
    if (mode == GraphPreparationMode::Complete) {
        initialize_graphs(GraphPreparationMode::Warmup);
        return initialize_graphs(GraphPreparationMode::Capture);
    }
    nvtx::ScopedRange range(nvtx::Name::CudaGraphPrepare, nvtx::Category::Graph);
    GraphPreparation observation(device, mode);
    PreparationInputs inputs(resources, device, *batch_execution);
    graphs = std::make_unique<DecodeGraphSuite>(device, resources.work, backend, false, nullptr,
                                                resources.moe.get());
    const auto buckets = ops::ContextBucketRange::through(capacity);
    ProposalBuckets draft_buckets;
    ops::ContextBucket main_bucket = buckets.first;
    std::int32_t active_tokens     = 0;
    DecodeGraphBindings bindings;
    bindings.stage = [&](const speculative::RoundRoute& route) {
        inputs.stage(route);
        draft_buckets = proposal_buckets(route);
        main_bucket   = target_bucket();
        active_tokens = 0;
        for (const auto columns : resources.frame.host_ingress.valid_columns)
            active_tokens += columns;
        return TargetReplay{
            .bucket        = main_bucket,
            .active_tokens = active_tokens,
            .proposal      = {.buckets = draft_buckets.span(), .readback = route.width > 1}};
    };
    bindings.warmup_proposal = [&](const speculative::RoundRoute& route) {
        proposal_prefix(*this, route, false);
        if (route.mtp_steps) {
            for (std::uint32_t step = 0; step < route.mtp_steps; ++step)
                run_mtp_draft_step(route.shape.rows, step, draft_buckets.values[step]);
        } else {
            run_dflash_draft(route, draft_buckets.span());
        }
        gpu_timer.end_draft();
    };
    bindings.warmup_target = [&](const speculative::RoundRoute& route) {
        auto run = target_run(*this, route, main_bucket, active_tokens);
        target_prefix(*this, route, false);
        if (!run->start())
            while (!run->poll()) std::this_thread::yield();
        target_head(*this, route)();
        gpu_timer.end_target();
    };
    bindings.proposal = [&](const speculative::RoundRoute& route) {
        return bind_proposal(*this, buckets, route);
    };
    bindings.target = [&](const speculative::RoundRoute& route) -> TargetCapture {
        if (resources.moe) return bind_offloaded(*this, buckets, route);
        return bind_resident(*this, buckets, route);
    };
    bindings.decision = [&](const speculative::RoundRoute& route) {
        auto scope = resources.work.scope();
        decide_decode(route, route);
        readback_result(route.shape.rows, &route);
    };
    if (draft_window)
        bindings.commit = [&](const speculative::RoundRoute& route, const SectionRecorder& repair,
                              const SectionRecorder& counts) {
            execute_decode_commit(route.shape.rows, route.width, true, true, repair, counts);
        };
    graphs->prepare(*batch_execution, *decode_routes, observation, bindings);
    inputs.finish();
    observation.checkpoint();
    const auto result = observation.sample();
    if (!graph_memory) graph_memory.emplace();
    if (observation.capturing()) {
        graph_memory->preparation_increment_bytes = result.increment_bytes;
        graph_memory->preparation_peak_bytes      = result.peak_bytes;
    } else {
        graphs.reset();
        graph_memory->warmup_increment_bytes = result.increment_bytes;
    }
    resources.work.reset();
    reset_measurement_peaks();
    return result;
}

void RankProgramImpl::release_graphs() {
    if (compute || context_operation)
        throw std::logic_error("cannot release graphs with active work");
    if (resources.moe) resources.moe->synchronize();
    device.synchronize();
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream()));
    graphs.reset();
}

} // namespace ninfer::models::qwen3_5::detail
