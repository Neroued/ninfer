#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/text.h"
#include "core/startup.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ninfer::models::qwen3_5 {

RankProgram::RankProgram(std::unique_ptr<detail::RankProgramImpl> impl) : impl_(std::move(impl)) {}

RankProgram::~RankProgram() = default;

runtime::ExecutionCapacity RankProgram::capacity() const noexcept { return {1, 1}; }

void RankProgram::validate_work(const Work& work, runtime::FrameKind required) const {
    impl_->validate_work(work, required);
}

void RankProgram::validate_decision(runtime::ExecutionTicket ticket,
                                    const runtime::DecisionInput& input) const {
    impl_->validate_decision(ticket, input);
}

void RankProgram::prepare_commit(runtime::ExecutionTicket ticket, Commit& commit) const {
    impl_->prepare_commit(ticket, commit);
}

void RankProgram::start(runtime::ExecutionRequest<ExecutionTypes>&& request) {
    impl_->start(std::move(request));
}

void RankProgram::supply(runtime::ExecutionTicket ticket, runtime::CommandSequence command,
                         runtime::DecisionInput&& input) {
    impl_->supply(ticket, command, std::move(input));
}

void RankProgram::commit(runtime::ExecutionTicket ticket, runtime::CommandSequence command,
                         Commit&& commit) {
    impl_->commit(ticket, command, std::move(commit));
}

void RankProgram::discard(runtime::ExecutionTicket ticket, runtime::CommandSequence command) {
    impl_->discard(ticket, command);
}

std::optional<runtime::ExecutionEvent<ExecutionTypes>> RankProgram::poll() { return impl_->poll(); }

void RankProgram::drain() { impl_->drain(); }

std::optional<GraphMemorySummary> RankProgram::graph_memory_observation() const {
    return impl_->graph_memory;
}

RankMemoryUsage RankProgram::observe_resources() const { return impl_->observe_resources(); }

void RankProgram::reset_measurement_peaks() noexcept { impl_->reset_measurement_peaks(); }

std::vector<context::KVTransferProfile> RankProgram::context_transfer_profiles() const {
    std::vector<context::KVTransferProfile> profiles{
        impl_->resources.decoder.text_kv.pages().transfer_profile()};
    if (const auto* backend = impl_->resources.backend_kv())
        profiles.push_back(backend->pages().transfer_profile());
    return profiles;
}

std::unique_ptr<RankProgram> create_rank_program(const execution::Parameters& parameters,
                                                 SequencePlan&& plan, DeviceContext& device,
                                                 const StartupObserver& observer) {
    if (!plan.impl_) throw std::invalid_argument("Qwen3.5 rank requires a sequence plan");
    auto impl =
        std::make_unique<detail::RankProgramImpl>(parameters, *plan.impl_, device, observer);
    if (plan.impl_->use_cuda_graph) {
        StartupPhaseScope phase(observer, StartupPhase::CudaGraphPrepare);
        impl->initialize_graphs();
        phase.complete();
    }
    plan.impl_.reset();
    return std::unique_ptr<RankProgram>(new RankProgram(std::move(impl)));
}

} // namespace ninfer::models::qwen3_5

namespace ninfer::models::qwen3_5::detail {

ComputeOperation::ComputeOperation(runtime::ExecutionRequest<ExecutionTypes>&& value)
    : request(std::move(value)), last_command(request.command) {}

ComputeOperation::~ComputeOperation() = default;

void RankProgramImpl::validate_row(const ExecutionRow& row) const {
    const auto lane = row.member.residency.lane.value;
    if (lane >= max_concurrency || !row.member.sequence.generation ||
        !row.member.residency.generation || !row.source_state.valid() ||
        !row.destination_state.valid() || row.state_slots.source < 0 ||
        row.state_slots.destination < 0 ||
        row.state_slots.source >= resources.state_images.storage().slot_count() ||
        row.state_slots.destination >= resources.state_images.storage().slot_count() ||
        !row.main_table.valid() || row.main_table.pool_id != 0 ||
        row.main_table.index >=
            static_cast<std::uint32_t>(resources.decoder.text_kv.tables().row_count()) ||
        row.main_frontier > capacity || row.backend_frontier > row.main_frontier) {
        throw std::invalid_argument("Qwen3.5 work has an invalid row binding");
    }
    const auto* cache = resources.backend_kv();
    if (bool(cache) != row.backend_table.has_value() ||
        (cache &&
         (!row.backend_table->valid() || row.backend_table->pool_id != 1 ||
          row.backend_table->index >= static_cast<std::uint32_t>(cache->tables().row_count())))) {
        throw std::invalid_argument("Qwen3.5 work has an invalid backend row binding");
    }
    if (backend == SpeculativeBackend::Mtp &&
        row.backend_frontier != (row.main_frontier ? row.main_frontier - 1 : 0)) {
        throw std::invalid_argument("MTP history must end immediately before the Main frontier");
    }
    const auto& installed = residencies[lane];
    if (!installed.installed || installed.id != row.member.residency ||
        installed.sequence != row.member.sequence) {
        throw std::logic_error("Qwen3.5 work references a retired residency");
    }
}

void RankProgramImpl::validate_work(const Work& work, runtime::FrameKind kind) const {
    const bool context = std::holds_alternative<ContextWork>(work.operation);
    if (kind != (context ? runtime::FrameKind::Context : runtime::FrameKind::Compute)) {
        throw std::invalid_argument("Qwen3.5 work is reserved in the wrong execution frame");
    }
    (context ? context_actions : compute_actions)->validate(work.context);
    auto validate_input = [&](const InputWindow& input) {
        if (input.tokens.size() > prefill_chunk ||
            input.rope_positions.size() != input.tokens.size() * 3) {
            throw std::invalid_argument("Qwen3.5 input window has invalid positions or extent");
        }
        std::uint32_t previous = 0;
        for (const auto split : input.execution_splits) {
            if (split <= previous || split >= input.tokens.size())
                throw std::invalid_argument(
                    "Qwen3.5 input segments are not strictly inside its window");
            previous = split;
        }
        for (TokenId id : input.tokens) {
            if (id < 0 || static_cast<std::uint32_t>(id) >= parameters.profile->public_tokens)
                throw std::invalid_argument("Qwen3.5 input token is outside the public domain");
        }
    };
    auto validate_members = [&](const auto& rows, auto binding) {
        if (rows.empty() || rows.size() > max_concurrency)
            throw std::invalid_argument("Qwen3.5 work has invalid batch membership");
        std::array<bool, kMaximumConcurrency> seen{};
        for (const auto& value : rows) {
            const auto& row = binding(value);
            validate_row(row);
            auto& present = seen[row.member.residency.lane.value];
            if (present) throw std::invalid_argument("Qwen3.5 work repeats a residency");
            present = true;
        }
    };
    std::visit(
        [&](const auto& op) {
            using T = std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<T, ContextWork>) {
                for (const auto& binding : op.bindings) {
                    if (binding.member.residency.lane.value >= max_concurrency ||
                        !binding.member.residency.generation || !binding.member.sequence.generation)
                        throw std::invalid_argument("Qwen3.5 residency installation is invalid");
                    for (auto id : binding.committed_generated_tokens) {
                        if (id < 0 ||
                            static_cast<std::uint32_t>(id) >= parameters.profile->public_tokens)
                            throw std::invalid_argument("Qwen3.5 restored count token is invalid");
                    }
                }
                for (auto id : op.releases) {
                    if (id.lane.value >= max_concurrency || !residencies[id.lane.value].installed ||
                        residencies[id.lane.value].id != id)
                        throw std::logic_error("Qwen3.5 release references a retired residency");
                }
            } else if constexpr (std::is_same_v<T, ForwardWork>) {
                validate_members(op.rows,
                                 [](const auto& row) -> const ExecutionRow& { return row; });
                if (op.inputs.size() != op.rows.size() ||
                    (op.sample != SamplePurpose::None && op.sampling.size() != op.rows.size()))
                    throw std::invalid_argument("Qwen3.5 forward inputs do not match membership");
                if (op.purpose == ForwardPurpose::Control) {
                    if (op.committed_tokens.size() != op.rows.size())
                        throw std::invalid_argument(
                            "Qwen3.5 control tokens do not match membership");
                    for (const auto& tokens : op.committed_tokens) {
                        if (tokens.size() > prefill_chunk)
                            throw std::invalid_argument("Qwen3.5 control exceeds staging capacity");
                        for (TokenId id : tokens)
                            if (id < 0 ||
                                static_cast<std::uint32_t>(id) >= parameters.profile->public_tokens)
                                throw std::invalid_argument(
                                    "Qwen3.5 control token is outside its vocabulary");
                    }
                }
                for (std::size_t i = 0; i < op.rows.size(); ++i) {
                    validate_input(op.inputs[i]);
                    if (op.rows[i].main_frontier + op.inputs[i].tokens.size() > capacity ||
                        (op.inputs[i].tokens.empty() && op.sample == SamplePurpose::None &&
                         op.purpose != ForwardPurpose::Replay))
                        throw std::invalid_argument("Qwen3.5 forward exceeds its context");
                }
            } else if constexpr (std::is_same_v<T, DecodeWork>) {
                validate_members(
                    op.rows, [](const auto& row) -> const ExecutionRow& { return row.binding; });
                for (const auto& row : op.rows) {
                    if (!row.binding.main_frontier || row.draft_extent > draft_window ||
                        row.binding.main_frontier + row.draft_extent + 1 > capacity ||
                        row.anchor < 0 ||
                        static_cast<std::uint32_t>(row.anchor) >=
                            parameters.profile->public_tokens ||
                        !row.output_tokens_remaining ||
                        row.draft_extent >= row.output_tokens_remaining)
                        throw std::invalid_argument(
                            "Qwen3.5 decode extent exceeds its execution permit");
                }
            } else if constexpr (std::is_same_v<T, NormalizeWork>) {
                validate_members(op.rows,
                                 [](const auto& row) -> const ExecutionRow& { return row; });
                if (!resources.dflash)
                    throw std::logic_error("normalization requires draft feature state");
            } else if constexpr (std::is_same_v<T, ScoreWork>) {
                validate_row(op.row);
                validate_input(op.input);
                if (!resources.frame.spec.causal_scoring || op.input.tokens.empty() ||
                    op.first_column > op.input.tokens.size() ||
                    op.targets.size() > op.input.tokens.size() - op.first_column)
                    throw std::invalid_argument("Qwen3.5 scoring span is invalid");
            }
        },
        work.operation);
}

ComputeOperation& RankProgramImpl::require_compute(runtime::ExecutionTicket ticket) {
    if (!compute || compute->request.ticket != ticket)
        throw std::logic_error("Qwen3.5 execution ticket is no longer active");
    return *compute;
}

const ComputeOperation& RankProgramImpl::require_compute(runtime::ExecutionTicket ticket) const {
    if (!compute || compute->request.ticket != ticket)
        throw std::logic_error("Qwen3.5 execution ticket is no longer active");
    return *compute;
}

void RankProgramImpl::validate_decision(runtime::ExecutionTicket ticket,
                                        const runtime::DecisionInput& input) const {
    const auto& op = require_compute(ticket);
    if (!op.decision_requested || op.decision_supplied || input.stage != op.decision_stage)
        throw std::logic_error("Qwen3.5 decision input does not match the pending stage");
    std::array<bool, kMaximumConcurrency> seen{};
    const auto words = (parameters.profile->public_tokens + 31) / 32;
    for (const auto& row : input.rows) {
        if (row.member >= op.batch || seen[row.member] || !op.sampling[row.member].constrained ||
            row.positions !=
                static_cast<std::uint32_t>(resources.frame.host_ingress->extents[row.member] + 1) ||
            row.words.size() != words * row.positions || (row.dead_positions >> row.positions) != 0)
            throw std::invalid_argument("Qwen3.5 decision mask has invalid shape or membership");
        seen[row.member] = true;
    }
    for (std::uint32_t row = 0; row < op.batch; ++row) {
        if (seen[row] != op.sampling[row].constrained)
            throw std::invalid_argument("Qwen3.5 decision omitted a constrained row");
    }
}

void RankProgramImpl::prepare_commit(runtime::ExecutionTicket ticket, Commit& value) const {
    const auto& op = require_compute(ticket);
    if (op.stage != ComputeStage::AwaitCommit || !op.result_published)
        throw std::logic_error("Qwen3.5 commit does not match the pending result");
    const auto* forward  = std::get_if<ForwardWork>(&op.request.work.operation);
    const bool generated = std::holds_alternative<DecodeWork>(op.request.work.operation) ||
                           (forward && forward->sample != SamplePurpose::None);
    if (!generated) {
        if (!value.rows.empty() && value.rows.size() != op.batch)
            throw std::invalid_argument("Qwen3.5 non-generating commit has invalid membership");
        compute_actions->validate(value.context);
        return;
    }
    const auto& output = *resources.frame.host_egress;
    if (value.rows.size() != op.batch)
        throw std::invalid_argument("Qwen3.5 generating commit has invalid membership");
    compute_actions->validate(value.context);
    for (std::uint32_t i = 0; i < op.batch; ++i) {
        const auto& row = value.rows[i];
        if (row.accepted_tokens > static_cast<std::uint32_t>(output.counts[i]) ||
            (row.discarded && row.accepted_tokens != 0) ||
            (!row.discarded && !row.accepted_tokens) ||
            (!row.terminal && !row.discarded &&
             row.accepted_tokens != static_cast<std::uint32_t>(output.counts[i])))
            throw std::invalid_argument("Qwen3.5 commit contains an invalid accepted prefix");
    }
}

} // namespace ninfer::models::qwen3_5::detail
