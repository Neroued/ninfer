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

bool RankProgram::can_discard(runtime::ExecutionTicket ticket) const noexcept {
    return impl_->can_discard(ticket);
}

MemorySpan RankProgram::input_storage() noexcept {
    return impl_->resources.vision ? impl_->resources.vision->staging() : MemorySpan{};
}

std::optional<runtime::ExecutionEvent<ExecutionTypes>> RankProgram::poll() { return impl_->poll(); }

void RankProgram::drain() { impl_->drain(); }

GraphPreparationSample RankProgram::prepare_execution(GraphPreparationMode mode) {
    return impl_->use_cuda_graph ? impl_->initialize_graphs(mode) : GraphPreparationSample{};
}

void RankProgram::release_execution() { impl_->release_graphs(); }

std::optional<GraphMemorySummary> RankProgram::graph_memory_observation() const {
    return impl_->graph_memory;
}

RankMemoryUsage RankProgram::observe_resources() const { return impl_->observe_resources(); }

void RankProgram::reset_measurement_peaks() noexcept { impl_->reset_measurement_peaks(); }

std::unique_ptr<RankProgram> create_uninitialized_rank_program(
    const execution::Parameters& parameters, SequencePlan&& plan, DeviceContext& device,
    const StartupObserver& observer) {
    if (!plan.impl_) throw std::invalid_argument("Qwen3.5 rank requires a sequence plan");
    auto impl =
        std::make_unique<detail::RankProgramImpl>(parameters, *plan.impl_, device, observer);
    plan.impl_.reset();
    return std::unique_ptr<RankProgram>(new RankProgram(std::move(impl)));
}

std::unique_ptr<RankProgram> create_rank_program(const execution::Parameters& parameters,
                                                 SequencePlan&& plan, DeviceContext& device,
                                                 const StartupObserver& observer,
                                                 GraphPreparationMode mode) {
    if (!plan.impl_) throw std::invalid_argument("Qwen3.5 rank requires a sequence plan");
    const bool use_graphs = plan.impl_->use_cuda_graph;
    auto program = create_uninitialized_rank_program(parameters, std::move(plan), device, observer);
    if (use_graphs) {
        StartupPhaseScope phase(observer, StartupPhase::CudaGraphPrepare);
        (void)program->prepare_execution(mode);
        phase.complete();
    }
    return program;
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
                validate_forward_work(op, prefill_chunk, parameters.profile->control_token_limit,
                                      parameters.profile->public_tokens);
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
                validate_narrowing(op, capacity);
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
    std::span<const std::uint32_t> verify_drafts;
    if (const auto* decode = std::get_if<DecodeWork>(&op.request.work.operation))
        verify_drafts = op.narrowed ? std::span<const std::uint32_t>(op.narrowed->verify_drafts)
                                    : decode->plan->verify_drafts();
    validate_decision_rows(op.request.work, input, parameters.profile->public_tokens,
                           verify_drafts);
}

void RankProgramImpl::prepare_commit(runtime::ExecutionTicket ticket, Commit& value) const {
    const auto& op = require_compute(ticket);
    if (op.stage != ComputeStage::AwaitCommit || !op.result_published)
        throw std::logic_error("Qwen3.5 commit does not match the pending result");
    // Packed prefill readback stores compact sampled IDs. Its counts are determined by Work.
    const auto counts =
        op.packed_prefill ? std::span<const std::int32_t>{} : resources.frame.host_egress.counts;
    validate_commit_rows(op.request.work, value, counts);
    compute_actions->validate(value.context);
}

} // namespace ninfer::models::qwen3_5::detail
