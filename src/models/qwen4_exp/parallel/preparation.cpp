#include "models/qwen4_exp/parallel/preparation.h"

namespace ninfer::models::qwen4_exp {

RankPreparation::RankPreparation(const artifact::Reader& reader, std::span<const std::byte> payload,
                                 const distributed::RankIdentity& identity, DeviceContext& device)
    : device_(device) {
    distributed::wire::Decoder in(payload);
    auto request = decode_rank_load(in);
    in.finish();
    if (request.rank.rank != identity.rank || request.rank.size != identity.tp.size ||
        identity.tp.rank != identity.rank || identity.ep.size != request.rank.size ||
        identity.ep.rank != identity.rank || identity.dp.size != 1 || identity.pp.size != 1)
        throw std::invalid_argument("Qwen4Exp rank plan differs from worker identity");
    if (request.artifact != reader.artifact_id())
        throw artifact::ArtifactError("Qwen4Exp worker artifact differs from coordinator");
    options_                = std::move(request.options);
    host_reservation_bytes_ = request.host_reservation_bytes;
    if (options_.use_cuda_graph)
        throw std::invalid_argument(
            "Qwen4Exp distributed preparation currently uses eager execution");
    auto definition = read_definition(reader, models::load_options(options_));
    load_.emplace(resolve_rank_load(reader, std::move(definition), std::move(request.rank),
                                    device_.device()));
    profile_ =
        execution::describe_model(load_->definition(), load_->weight_data(), load_->rank_plan());
    curve_ = describe_execution_capacity(profile_, options_, device_);
}

runtime::PreparationMemory RankPreparation::memory() const {
    return {load_->materialization().device_capacity_bytes,
            options_.kv_capacity.automatic_headroom_bytes, host_reservation_bytes_};
}

void RankPreparation::materialize() {
    if (!load_) throw std::logic_error("Qwen4Exp rank weights are already materialized");
    instance_ = std::make_unique<Instance>(
        materialize_model(std::move(*load_), device_, &options_.startup_observer), profile_);
    load_.reset();
}

runtime::RankPreparationReport RankPreparation::report(std::size_t available) const {
    const auto& model = *instance_->model;
    const auto& s     = model.storage_stats();
    return {.capacity          = {curve_, available},
            .weight_bytes      = s.device_capacity_bytes + model.metadata_bytes(),
            .host_weight_bytes = s.host_capacity_bytes,
            .staging_bytes     = s.peak_staging_bytes,
            .read_bytes        = s.read_bytes,
            .upload_bytes      = s.h2d_bytes,
            .load_seconds      = s.upload_seconds};
}

void RankPreparation::prepare(std::uint32_t pages, std::size_t available) {
    if (instance_->program) throw std::logic_error("Qwen4Exp rank Program is already prepared");
    auto options        = options_;
    options.kv_capacity = KvCapacityPolicy::explicit_capacity(curve_.resolved_tokens(pages));
    auto plan           = plan_execution(profile_, {}, options, device_, available);
    if (!plan) throw DeviceAllocationError("rank runtime capacity exceeds available memory");
    instance_->program = std::make_unique<RankProgram>(instance_->parameters, std::move(*plan),
                                                       device_, options_.startup_observer);
}

void RankPreparation::release_program() {
    instance_->program.reset();
    device_.synchronize();
}

RankExecutionDescription RankPreparation::describe() const {
    const auto& program = *instance_->program;
    return {program.session_plan(), program.capacity(), program.memory_summary()};
}

ExecutionCodec RankPreparation::codec(const distributed::ExecutionTopology& topology) const {
    return ExecutionCodec(instance_->model->definition(), describe(), topology);
}

std::unique_ptr<RankPreparation::Instance> RankPreparation::finish() && {
    if (!instance_ || !instance_->program)
        throw std::logic_error("Qwen4Exp rank Program has not been prepared");
    return std::move(instance_);
}

} // namespace ninfer::models::qwen4_exp
