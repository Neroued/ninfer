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

std::unique_ptr<RankProgram> create_rank_program(const execution::Parameters& parameters,
                                                 SequencePlan&& plan, DeviceContext& device,
                                                 const StartupObserver& observer,
                                                 GraphPreparationMode mode) {
    if (!plan.impl_) throw std::invalid_argument("Qwen3.5 rank requires a sequence plan");
    auto impl =
        std::make_unique<detail::RankProgramImpl>(parameters, *plan.impl_, device, observer);
    if (plan.impl_->use_cuda_graph) {
        StartupPhaseScope phase(observer, StartupPhase::CudaGraphPrepare);
        impl->initialize_graphs(mode);
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

std::uint64_t RankProgramImpl::next_validation_mark() const {
    if (++validation_generation == 0) {
        std::fill(validation_marks.begin(), validation_marks.end(), 0);
        ++validation_generation;
    }
    return validation_generation;
}

void RankProgramImpl::validate_row(const ExecutionRow& row) const {
    const auto lane = row.member.residency.lane.value;
    if (lane >= max_concurrency || !row.member.sequence.generation ||
        !row.member.residency.generation || !row.source_state.valid() ||
        !row.destination_state.valid() || row.state_offsets.source < 0 ||
        row.state_offsets.destination < 0 || !row.main_table.valid() ||
        row.main_table.pool_id != 0 ||
        row.main_table.index >=
            static_cast<std::uint32_t>(resources.decoder.text_kv.tables().row_count()) ||
        row.main_frontier > capacity || row.backend_frontier > row.main_frontier) {
        throw std::invalid_argument("Qwen3.5 work has an invalid row binding");
    }
    (void)resources.state_images.storage().image(row.state_offsets.source);
    (void)resources.state_images.storage().image(row.state_offsets.destination);
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

void validate_prefill_work(const ForwardWork& op, std::uint32_t prefill_chunk) {
    if (op.rows.empty() || op.inputs.size() != op.rows.size() ||
        ((!op.sampling.empty() || op.sample != SamplePurpose::None) &&
         op.sampling.size() != op.rows.size()))
        throw std::invalid_argument("Qwen3.5 prefill inputs do not match membership");
    if (std::ranges::any_of(op.inputs, [](const auto& w) { return w.visual.has_value(); }) &&
        (op.rows.size() != 1 || op.inputs.front().tokens.empty()))
        throw std::invalid_argument("Qwen3.5 visual prefill requires one nonempty input window");
    std::size_t total = 0;
    bool any_sample   = false;
    for (std::size_t i = 0; i < op.rows.size(); ++i) {
        const auto& row = op.rows[i];
        const auto n    = op.inputs[i].tokens.size();
        if (n > prefill_chunk - total)
            throw std::invalid_argument("Qwen3.5 prefill exceeds its packed token budget");
        total += n;
        const bool sampled     = !op.sampling.empty() && op.sampling[i].sample;
        const bool constrained = !op.sampling.empty() && op.sampling[i].constrained;
        any_sample |= sampled;
        if ((!n && (!sampled || !row.main_frontier)) || (constrained && !sampled))
            throw std::invalid_argument("Qwen3.5 prefill row has invalid sampling or extent");
        for (std::size_t j = 0; j < i; ++j) {
            const auto& prev        = op.rows[j];
            const bool prev_updates = !op.inputs[j].tokens.empty();
            if ((n && prev_updates &&
                 (row.destination_state == prev.destination_state ||
                  row.state_offsets.destination == prev.state_offsets.destination)) ||
                (prev_updates && (row.source_state == prev.destination_state ||
                                  row.state_offsets.source == prev.state_offsets.destination)) ||
                (n && (row.destination_state == prev.source_state ||
                       row.state_offsets.destination == prev.state_offsets.source)) ||
                row.main_table.index == prev.main_table.index ||
                (row.backend_table && prev.backend_table &&
                 row.backend_table->index == prev.backend_table->index))
                throw std::invalid_argument("Qwen3.5 prefill rows overlap State or KV writes");
        }
    }
    if (op.sample != (any_sample ? SamplePurpose::FirstToken : SamplePurpose::None))
        throw std::invalid_argument("Qwen3.5 prefill sample purpose differs from its rows");
}

void RankProgramImpl::validate_work(const Work& work, runtime::FrameKind kind) const {
    const bool context = std::holds_alternative<ContextWork>(work.operation);
    if (kind != (context ? runtime::FrameKind::Context : runtime::FrameKind::Compute)) {
        throw std::invalid_argument("Qwen3.5 work is reserved in the wrong execution frame");
    }
    (context ? context_actions : compute_actions)->validate(work.context);
    auto validate_input = [&](const InputWindow& input, std::uint32_t limit) {
        if (input.tokens.size() > limit || input.rope_positions.size() != input.tokens.size() * 3) {
            throw std::invalid_argument("Qwen3.5 input window has invalid positions or extent");
        }
        for (TokenId id : input.tokens) {
            if (id < 0 || static_cast<std::uint32_t>(id) >= parameters.profile->public_tokens)
                throw std::invalid_argument("Qwen3.5 input token is outside the public domain");
        }
    };
    auto validate_members = [&](const auto& rows, auto binding) {
        if (rows.empty() || rows.size() > max_concurrency)
            throw std::invalid_argument("Qwen3.5 work has invalid batch membership");
        const auto mark = next_validation_mark();
        for (const auto& value : rows) {
            const auto& row = binding(value);
            validate_row(row);
            auto& present = validation_marks[row.member.residency.lane.value];
            if (present == mark) throw std::invalid_argument("Qwen3.5 work repeats a residency");
            present = mark;
        }
    };
    std::visit(
        [&](const auto& op) {
            using T = std::decay_t<decltype(op)>;
            if constexpr (!std::is_same_v<T, ContextWork>) {
                const auto members = [&] {
                    if constexpr (std::is_same_v<T, ScoreWork>)
                        return std::size_t{1};
                    else
                        return op.rows.size();
                }();
                const auto shape = [&] {
                    if constexpr (std::is_same_v<T, DecodeWork>) {
                        if (!op.plan || op.plan->verify_drafts().size() != members)
                            throw std::invalid_argument(
                                "Qwen3.5 decode plan differs from membership");
                        return op.plan->route().shape;
                    } else
                        return op.shape;
                }();
                const bool bucketed =
                    std::is_same_v<T, DecodeWork> && !batch_execution->graph_rows().empty();
                if (!members || members > shape.rows || shape.rows > max_concurrency ||
                    shape.attention_rows != shape.rows || (!bucketed && shape.rows != members) ||
                    (bucketed &&
                     !std::binary_search(batch_execution->graph_rows().begin(),
                                         batch_execution->graph_rows().end(), shape.rows)))
                    throw std::invalid_argument(
                        "Qwen3.5 work is outside its prepared batch geometry");
            }
            if constexpr (std::is_same_v<T, ContextWork>) {
                if (op.bindings.size() > 1 || op.releases.size() > max_concurrency)
                    throw std::invalid_argument("Qwen3.5 context work exceeds its frame capacity");
                for (const auto& binding : op.bindings) {
                    if (binding.member.residency.lane.value >= max_concurrency ||
                        !binding.member.residency.generation || !binding.member.sequence.generation)
                        throw std::invalid_argument("Qwen3.5 residency installation is invalid");
                    binding.counts.validate(parameters.profile->public_tokens, capacity);
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
                if (op.purpose == ForwardPurpose::Prefill) validate_prefill_work(op, prefill_chunk);
                if (op.purpose == ForwardPurpose::Replay && op.rows.size() != 1)
                    throw std::invalid_argument("Qwen3.5 Replay requires one sequence");
                if (op.purpose == ForwardPurpose::Control) {
                    if (op.control_last_tokens.size() != op.rows.size())
                        throw std::invalid_argument(
                            "Qwen3.5 control tokens do not match membership");
                    for (std::size_t i = 0; i < op.rows.size(); ++i) {
                        const auto n  = op.inputs[i].tokens.size();
                        const auto id = op.control_last_tokens[i];
                        if (!n || n > parameters.profile->control_token_limit)
                            throw std::invalid_argument("Qwen3.5 control exceeds prepared extent");
                        if (id < 0 || std::uint32_t(id) >= parameters.profile->public_tokens)
                            throw std::invalid_argument("Qwen3.5 control token exceeds vocabulary");
                    }
                } else if (!op.control_last_tokens.empty()) {
                    throw std::invalid_argument("Qwen3.5 control output outside control work");
                }
                for (std::size_t i = 0; i < op.rows.size(); ++i) {
                    validate_input(op.inputs[i], op.purpose == ForwardPurpose::Control
                                                     ? parameters.profile->control_token_limit
                                                     : prefill_chunk);
                    if (op.rows[i].main_frontier + op.inputs[i].tokens.size() > capacity ||
                        (op.inputs[i].tokens.empty() && op.sample == SamplePurpose::None &&
                         op.purpose != ForwardPurpose::Replay))
                        throw std::invalid_argument("Qwen3.5 forward exceeds its context");
                }
            } else if constexpr (std::is_same_v<T, DecodeWork>) {
                validate_members(
                    op.rows, [](const auto& row) -> const ExecutionRow& { return row.binding; });
                const auto& route = op.plan->route();
                route.validate(backend, draft_window);
                if (graphs && !graphs->contains(route))
                    throw std::invalid_argument("Qwen3.5 graph route is unprepared");
                if (route.graph_entry.has_value() != bool(graphs))
                    throw std::invalid_argument(
                        "Qwen3.5 decode route differs from the prepared path");
                for (std::size_t i = 0; i < op.rows.size(); ++i) {
                    const auto& row   = op.rows[i];
                    const auto extent = op.plan->verify_drafts()[i];
                    if ((!row.binding.main_frontier && extent &&
                         backend == SpeculativeBackend::Mtp) ||
                        std::uint64_t(row.binding.main_frontier) + extent + 1 > capacity ||
                        row.anchor < 0 ||
                        static_cast<std::uint32_t>(row.anchor) >= parameters.profile->public_tokens)
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
                validate_input(op.input, prefill_chunk);
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
    const auto mark       = next_validation_mark();
    const auto words      = (parameters.profile->public_tokens + 31) / 32;
    const auto* forward   = std::get_if<ForwardWork>(&op.request.work.operation);
    const bool prefill    = forward && forward->purpose == ForwardPurpose::Prefill;
    const auto needs_mask = [&](std::uint32_t member) {
        const auto& sampling = op.sampling[member];
        return sampling.constrained && (!prefill || sampling.sample);
    };
    for (const auto& row : input.rows) {
        if (row.member >= op.batch || validation_marks[row.member] == mark ||
            !needs_mask(row.member) ||
            row.positions != (prefill
                                  ? 1U
                                  : static_cast<std::uint32_t>(
                                        resources.frame.host_ingress.extents[row.member] + 1)) ||
            row.words.size() != words * row.positions || (row.dead_positions >> row.positions) != 0)
            throw std::invalid_argument("Qwen3.5 decision mask has invalid shape or membership");
        validation_marks[row.member] = mark;
    }
    for (std::uint32_t row = 0; row < op.batch; ++row) {
        if ((validation_marks[row] == mark) != needs_mask(row))
            throw std::invalid_argument("Qwen3.5 decision omitted a constrained row");
    }
}

void validate_prefill_commit(const ForwardWork& work, std::span<const CommitRow> rows) {
    if (rows.size() != work.rows.size())
        throw std::invalid_argument("Qwen3.5 prefill commit has invalid membership");
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& row     = rows[i];
        const bool sampled  = work.sampling[i].sample;
        const auto expected = row.discarded ? 0 : sampled ? 1 : work.inputs[i].tokens.size();
        if (row.accepted_tokens != expected || (!sampled && row.terminal != row.discarded))
            throw std::invalid_argument("Qwen3.5 prefill commit has an invalid accepted prefix");
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
    const auto& output = resources.frame.host_egress;
    if (value.rows.size() != op.batch)
        throw std::invalid_argument("Qwen3.5 generating commit has invalid membership");
    compute_actions->validate(value.context);
    if (forward && forward->purpose == ForwardPurpose::Prefill) {
        validate_prefill_commit(*forward, value.rows);
        return;
    }
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
