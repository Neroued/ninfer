#include "models/common/graph/target_body.h"

#include "core/nvtx.h"

#include <stdexcept>

namespace ninfer::models {
namespace {

void select_readouts(DecodeGraphExecutable& executable, std::span<ContextBucketSite> sites,
                     ops::ContextBucket bucket) {
    for (auto& site : sites) site.select(executable, bucket);
}

} // namespace

ProposalBody::ProposalBody(DeviceContext& device, const ProposalCapture& capture) {
    const auto cuda = device.execution_view();
    steps_.reserve(capture.steps.size());
    for (const auto& source : capture.steps) {
        auto& step = steps_.emplace_back();
        step.sites = prepare_readouts(cuda, source.forward.readouts);
        step.bucket_offsets.reserve(step.sites.size());
        for (const auto& binding : source.forward.readouts)
            step.bucket_offsets.push_back(binding.bucket_offset);
        if (source.handoff) {
            if (!capture.handoff_stream || capture.handoff_stream == device.stream())
                throw std::logic_error("proposal handoff requires a secondary stream");
            step.handoff.emplace(cuda, source.handoff);
        }
    }
    for (const auto& step : steps_) {
        if (!step.handoff) continue;
        fork_.emplace(device);
        join_.emplace(device);
        break;
    }
}

void ProposalBody::record(CudaExecutionView cuda, ParallelExecution* parallel,
                          const ProposalCapture& capture, bool gate_steps) {
    const auto stream = cuda.stream();
    if (gate_steps && fork_) throw std::logic_error("gated proposal has a secondary handoff");
    if (capture.prefix) capture.prefix();
    for (std::size_t i = 0; i < steps_.size(); ++i) {
        auto& step         = steps_[i];
        const auto forward = [&] {
            record_readouts(cuda, step.sites, capture.steps[i].forward.body);
        };
        if (gate_steps)
            gates_.capture_step(stream, parallel, forward);
        else
            forward();
        if (step.handoff) {
            fork_->record(stream);
            fork_->wait(capture.handoff_stream);
            step.handoff->append(capture.handoff_stream);
        }
    }
    if (join_) {
        join_->record(capture.handoff_stream);
        join_->wait(stream);
    }
    if (capture.before_readback) capture.before_readback();
    if (capture.readback) draft_readback_.capture(stream, capture.readback);
    if (capture.drafts_ready) capture.drafts_ready();
    if (capture.verify_ids) capture.verify_ids();
}

void ProposalBody::finish(const CommunicationGraphWork& work) {
    if (!gates_.empty()) gates_.finish(work);
}

void ProposalBody::add_calibration(DecodeGraphExecutable& executable,
                                   GraphPatchCalibration& calibration) {
    for (auto& step : steps_) {
        calibration.add(executable, step.sites);
        if (step.handoff) calibration.add(executable, *step.handoff);
    }
}

const CommunicationGraphWork* ProposalBody::select(DecodeGraphExecutable& executable,
                                                   const ProposalReplay& replay,
                                                   std::uint32_t active_steps) {
    const CommunicationGraphWork* work = nullptr;
    if (!gates_.empty())
        work = &gates_.set_depth(executable, active_steps);
    else if (active_steps != steps_.size())
        throw std::logic_error("proposal replay differs from its captured step count");
    for (std::size_t i = 0; i < active_steps; ++i) {
        auto& step = steps_[i];
        for (std::size_t s = 0; s < step.sites.size(); ++s) {
            const auto offset = step.bucket_offsets[s];
            if (i >= replay.buckets.size() || offset >= replay.buckets.size() - i)
                throw std::logic_error("missing proposal context bucket");
            step.sites[s].select(executable, replay.buckets[i + offset]);
        }
        if (step.handoff) step.handoff->select(executable, replay.slot.value());
    }
    draft_readback_.set_enabled(executable, replay.readback);
    return work;
}

void ResidentBody::prepare(DeviceContext& device, ParallelExecution* parallel,
                           GraphPreparation& preparation, const ResidentTargetCapture& capture,
                           const ProposalCapture* proposal, bool gate_steps) {
    const auto cuda   = device.execution_view();
    const auto stream = cuda.stream();
    parallel_         = parallel;
    if (proposal) proposal_.emplace(device, *proposal);
    main_sites_ = prepare_readouts(cuda, capture.readouts);
    if (capture.handoff) handoff_.emplace(cuda, capture.handoff);
    graph_.prepare(
        cuda,
        [&] {
            if (proposal_) proposal_->record(cuda, parallel, *proposal, gate_steps);
            record_readouts(cuda, main_sites_, [&](const ReadoutRecorder& readout) {
                capture.body(
                    readout,
                    [&] {
                        if (!handoff_) throw std::logic_error("unbound target handoff");
                        handoff_->append(stream);
                    },
                    [&] {
                        if (capture.head) head_.capture(stream, capture.head);
                    },
                    [&](cudaStream_t readback_stream, const GraphBody& body) {
                        readback_.capture(readback_stream, body);
                    });
            });
        },
        parallel);
    if (handoff_ && !handoff_->appended())
        throw std::logic_error("target handoff was not recorded");
    if (proposal_) proposal_->finish(graph_.communication);
    GraphPatchCalibration calibration;
    if (proposal_) proposal_->add_calibration(graph_.executable, calibration);
    calibration.add(graph_.executable, main_sites_);
    if (handoff_) calibration.add(graph_.executable, *handoff_);
    calibration.run(preparation, [&] { graph_.executable.upload(stream); });
}

void ResidentBody::start(CudaExecutionView cuda, const TargetReplay& replay,
                         std::uint32_t active_steps) {
    auto& executable = graph_.executable;
    const auto* work =
        proposal_ ? proposal_->select(executable, replay.proposal, active_steps) : nullptr;
    select_readouts(executable, main_sites_, replay.bucket);
    if (handoff_) handoff_->select(executable, replay.slot.value());
    head_.set_enabled(executable, replay.head);
    readback_.set_enabled(executable, replay.readback || replay.proposal.readback);
    graph_.launch(cuda, parallel_, work);
}

void OffloadedLayeredBody::prepare(CudaExecutionView cuda, ParallelExecution* parallel,
                                   GraphPreparation& preparation, moe::Schedule& schedule,
                                   const OffloadedTargetCapture& capture) {
    parallel_        = parallel;
    schedule_        = &schedule;
    output_          = capture.output;
    physical_tokens_ = capture.physical_tokens;
    prefix_.prepare(cuda, capture.prefix, parallel);
    if (capture.hook) {
        hook_layer_ = capture.hook->layer;
        if (*hook_layer_ > capture.layers.size())
            throw std::logic_error("invalid graph hook layer");
        hook_slot_.emplace(cuda, capture.hook->body);
        hook_.prepare(cuda, [&] { hook_slot_->append(cuda.stream()); }, parallel);
        GraphPatchCalibration calibration;
        calibration.add(hook_.executable, *hook_slot_);
        calibration.run(preparation, [&] { hook_.executable.upload(cuda.stream()); });
    }
    layers_.reserve(capture.layers.size());
    for (const auto& source : capture.layers) {
        auto& layer = layers_.emplace_back();
        GraphPatchCalibration calibration;
        if (const auto* resident = std::get_if<ReadoutCapture>(&source)) {
            auto& segment = std::get<GraphSegment>(layer.body);
            layer.sites   = prepare_readouts(cuda, resident->readouts);
            segment.prepare(
                cuda, [&] { record_readouts(cuda, layer.sites, resident->body); }, parallel);
            calibration.add(segment.executable, layer.sites);
            calibration.run(preparation, [&] { segment.executable.upload(cuda.stream()); });
        } else {
            const auto& offloaded = std::get<OffloadedLayerCapture>(source);
            auto& graph           = layer.body.emplace<OffloadedLayer>();
            layer.sites           = prepare_readouts(cuda, offloaded.mixer.readouts);
            graph.definition      = schedule.capture_layer(
                offloaded.bank, cuda, offloaded.geometry,
                [&](const moe::MixerSliceRecorder& mixer) {
                    record_readouts(cuda, layer.sites, offloaded.mixer.body, &mixer);
                },
                offloaded.gpu, offloaded.feature);
            if (graph.definition->mixer_slice_count() != layer.sites.size())
                throw std::logic_error("MoE mixer graph differs from its readout bindings");
            graph.executable = graph.definition->instantiate();
            graph.executable->upload(cuda.stream());
            calibration.add(*graph.executable, layer.sites);
            calibration.run(preparation, [&] { graph.executable->upload_mixer(cuda.stream()); });
        }
    }
    suffix_.prepare(
        cuda,
        [&] {
            if (capture.suffix) capture.suffix();
            if (capture.head) head_.capture(cuda.stream(), capture.head);
            if (capture.after_head) capture.after_head();
        },
        parallel);
}

void OffloadedLayeredBody::begin_prefix(CudaExecutionView cuda, const TargetReplay& replay) {
    if (cuda_) throw std::logic_error("target graph traversal is already active");
    head_.set_enabled(suffix_.executable, replay.head);
    if (auto activity = schedule_->begin(physical_tokens_, replay.active_tokens))
        activity_.emplace(std::move(*activity));
    cuda_           = cuda;
    bucket_         = replay.bucket;
    next_layer_     = 0;
    hook_submitted_ = false;
    prefix_.launch(cuda, parallel_);
}

TargetProgress OffloadedLayeredBody::start(CudaExecutionView cuda, const TargetReplay& replay) {
    begin_prefix(cuda, replay);
    return poll();
}

TargetProgress OffloadedLayeredBody::advance_until(std::size_t end) {
    if (!cuda_ || end > layers_.size() || end < next_layer_)
        throw std::logic_error("invalid target graph traversal boundary");
    for (; next_layer_ < end; ++next_layer_) {
        if (hook_layer_ == next_layer_ && !hook_submitted_) return TargetProgress::AwaitingHook;
        const nvtx::ScopedRange range(nvtx::Name::TextLayer, nvtx::Category::Runtime, next_layer_);
        auto& layer = layers_[next_layer_];
        if (auto* resident = std::get_if<GraphSegment>(&layer.body)) {
            select_readouts(resident->executable, layer.sites, bucket_);
            resident->launch(*cuda_, parallel_);
        } else {
            auto& graph = *std::get<OffloadedLayer>(layer.body).executable;
            if (waiting_layer_) {
                if (!graph.poll()) return TargetProgress::WaitingLayer;
            } else {
                for (std::size_t i = 0; i < layer.sites.size(); ++i)
                    layer.sites[i].select(graph, i, bucket_);
                waiting_layer_ = true;
                if (!graph.start(output_, *cuda_)) return TargetProgress::WaitingLayer;
            }
            waiting_layer_ = false;
        }
    }
    return TargetProgress::Complete;
}

void OffloadedLayeredBody::launch_hook(std::uint32_t slot) {
    if (!cuda_ || hook_layer_ != next_layer_ || waiting_layer_ || hook_submitted_)
        throw std::logic_error("target graph is not waiting for its inter-layer hook");
    hook_slot_->select(hook_.executable, slot);
    hook_.launch(*cuda_, parallel_);
    hook_submitted_ = true;
}

TargetProgress OffloadedLayeredBody::poll() {
    if (!cuda_) return TargetProgress::Complete;
    const auto progress = advance_until(layers_.size());
    if (progress != TargetProgress::Complete) return progress;
    if (hook_layer_ == next_layer_ && !hook_submitted_) return TargetProgress::AwaitingHook;
    suffix_.launch(*cuda_, parallel_);
    activity_.reset();
    cuda_.reset();
    return TargetProgress::Complete;
}

void OffloadedLayeredBody::cancel() {
    if (!cuda_) return;
    auto activity = std::move(activity_);
    activity_.reset();
    cuda_.reset();
    if (std::exchange(waiting_layer_, false))
        std::get<OffloadedLayer>(layers_[next_layer_].body).executable->cancel();
}

void TargetGraphBody::prepare(DeviceContext& device, ParallelExecution* parallel,
                              GraphPreparation& preparation, moe::Schedule* schedule,
                              const TargetCapture& capture, const ProposalCapture* fused,
                              bool gate_steps) {
    if (schedule) {
        if (fused) throw std::logic_error("offloaded target cannot capture a fused proposal");
        body_.emplace<OffloadedLayeredBody>().prepare(device.execution_view(), parallel,
                                                      preparation, *schedule,
                                                      std::get<OffloadedTargetCapture>(capture));
    } else {
        body_.emplace<ResidentBody>().prepare(device, parallel, preparation,
                                              std::get<ResidentTargetCapture>(capture), fused,
                                              gate_steps);
    }
}

TargetProgress TargetGraphBody::start(CudaExecutionView cuda, const TargetReplay& replay,
                                      std::uint32_t active_steps) {
    if (auto* resident = std::get_if<ResidentBody>(&body_)) {
        resident->start(cuda, replay, active_steps);
        return TargetProgress::Complete;
    }
    return std::get<OffloadedLayeredBody>(body_).start(cuda, replay);
}

void TargetGraphBody::begin_prefix(CudaExecutionView cuda, const TargetReplay& replay) {
    std::get<OffloadedLayeredBody>(body_).begin_prefix(cuda, replay);
}

TargetProgress TargetGraphBody::advance_until(std::size_t end) {
    return std::get<OffloadedLayeredBody>(body_).advance_until(end);
}

void TargetGraphBody::launch_hook(std::uint32_t slot) {
    std::get<OffloadedLayeredBody>(body_).launch_hook(slot);
}

TargetProgress TargetGraphBody::poll() {
    if (auto* layered = std::get_if<OffloadedLayeredBody>(&body_)) return layered->poll();
    return TargetProgress::Complete;
}

void TargetGraphBody::cancel() {
    if (auto* layered = std::get_if<OffloadedLayeredBody>(&body_)) layered->cancel();
}

std::optional<std::size_t> TargetGraphBody::hook_layer() const noexcept {
    if (const auto* layered = std::get_if<OffloadedLayeredBody>(&body_))
        return layered->hook_layer();
    return {};
}

} // namespace ninfer::models
