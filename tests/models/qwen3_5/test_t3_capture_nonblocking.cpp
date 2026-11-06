// T3/F4: prefix checkpoint capture must not block other lanes, and a capture that is
// cancelled in flight must return every Context resource. Only the public Session contract
// (terminal_blocked, finish/abort, physical_usage, start_capture, cancel_context) is used.
#include "core/device.h"
#include "models/common/qwen/frontend/frontend.h"
#include "models/common/qwen/frontend/prepared_prompt.h"
#include "models/qwen3_5/execution/profile.h"
#include "models/qwen3_5/load.h"
#include "native_fixture.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
namespace q35    = models::qwen3_5;
namespace native = ninfer::test::qwen3_5;

constexpr std::uint32_t kCapacity = 512;
constexpr std::uint32_t kChunk    = 128;

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

bool same_ownership(const q35::PhysicalUsageSnapshot& a, const q35::PhysicalUsageSnapshot& b) {
    if (a.state_objects != b.state_objects || a.history_objects != b.history_objects ||
        a.kv_page_groups != b.kv_page_groups || a.pools.size() != b.pools.size() ||
        a.metadata.reserved_bytes != b.metadata.reserved_bytes)
        return false;
    for (std::size_t i = 0; i < a.pools.size(); ++i) {
        const auto& x = a.pools[i];
        const auto& y = b.pools[i];
        if (x.rank != y.rank || x.domain != y.domain || x.live_bytes != y.live_bytes ||
            x.in_use_bytes != y.in_use_bytes || x.cached_bytes != y.cached_bytes ||
            x.reserved_bytes != y.reserved_bytes ||
            x.pending_retirement_bytes != y.pending_retirement_bytes ||
            x.future_reserved_bytes != y.future_reserved_bytes ||
            x.relocation_bytes != y.relocation_bytes)
            return false;
    }
    return true;
}

// F4 acceptance 1: capture stays on by default with the baseline (a754fcf3) interval.
void default_capture_policy() {
    const EngineOptions defaults;
    require(defaults.context_cache.enabled && defaults.context_cache.checkpoint_interval == 8192,
            "prefix reuse default or checkpoint_interval differs from the baseline");
}

class Scenario {
public:
    Scenario(native::NativeRankFixture& rank, q35::Session& session,
             models::qwen::Frontend& frontend)
        : rank_(rank), session_(session), frontend_(frontend), empty_(session.physical_usage()),
          driver_(session, rank) {}

    // Prompt with three explicit shared-prefix boundaries inside the first prefill chunk.
    q35::RequestBasePlan marked_request() {
        PromptInput input;
        ChatMessage system;
        system.role = ChatRole::System;
        std::string instructions;
        for (unsigned word = 0; word < 20; ++word) instructions += "a ";
        system.parts.push_back({.kind = MessagePartKind::Text, .text = instructions});
        input.messages.push_back(std::move(system));
        ChatMessage user;
        user.role = ChatRole::User;
        std::string content;
        for (unsigned word = 0; word < 350; ++word) content += "b ";
        user.parts.push_back({.kind = MessagePartKind::Text, .text = content});
        input.messages.push_back(std::move(user));
        input.context_cache.allow_engine_automatic_shared_prefixes = false;
        for (const auto bytes : {13U, 21U, 29U}) {
            input.context_cache.markers.push_back(
                {.kind                      = PromptCacheMarkerKind::SharedStablePrefix,
                 .evidence                  = SharedCandidateEvidence::ExplicitBoundary,
                 .location                  = PromptCacheMarkerLocation::LeadingInstructionBoundary,
                 .leading_instruction_bytes = bytes});
        }
        runtime::ResolvedExecutionOptions options;
        options.requested_output_tokens = 8;
        options.allow_prefix_reuse      = true;
        return session_.plan_request(
            frontend_.prepare(std::move(input)),
            options);
    }

    q35::RequestBasePlan plain_request(std::uint32_t tokens) {
        std::vector<TokenId> ids(tokens, 198);
        ids.front() = 1000;
        runtime::ResolvedExecutionOptions options;
        options.requested_output_tokens = 8;
        options.allow_prefix_reuse      = false;
        return session_.plan_request(
            frontend_.prepare_tokens(std::move(ids), true),
            options);
    }

    q35::SequenceHandle bind(const q35::RequestBasePlan& base, std::uint32_t lane) {
        auto source = session_.inspect_source(base, std::nullopt, false, {}, {}, {0}, nullptr);
        require(source.has_value(), "fresh request has no root source");
        require(static_cast<bool>(session_.start_binding(base, {lane}, *source, nullptr,
                                                         q35::ExecutionUnitKind::Decode, 1)),
                "request binding could not reserve its first unit");
        auto progress = driver_.settle();
        driver_.idle();
        require(progress.published && progress.sequence.has_value(),
                "binding did not publish a lane");
        return *progress.sequence;
    }

    void reserve_prefill(q35::SequenceHandle sequence) {
        const std::array<q35::ExecutionUnit, 1> units{
            {{sequence, q35::ExecutionUnitKind::Prefill, 0}}};
        require(static_cast<bool>(session_.reserve_units(units)),
                "prefill unit was not reserved");
    }

    // Runs A to its first capture boundary and starts (but does not settle) the capture.
    q35::SequenceHandle start_capture_of_a() {
        const auto a = bind(marked_request(), 0);
        reserve_prefill(a);
        const auto step = driver_.prefill(a);
        require(step.capture_ready && !step.complete,
                "marked prompt did not reach a capture boundary");
        require(session_.start_capture(a), "capture of a reached boundary could not start");
        require(session_.has_context_operation(), "capture is not an in-flight operation");
        return a;
    }

    // F4 acceptance 3: another lane starts prefill while A's capture is still in flight.
    void other_lane_advances_during_capture() {
        const auto a = start_capture_of_a();
        require(session_.terminal_blocked(a), "capturing lane is not terminal-blocked");
        // finish/abort decline for the blocked lane and leave it intact.
        require(session_.abort(a).status != runtime::ConsumeStatus::Consumed &&
                    session_.finish(a).status != runtime::ConsumeStatus::Consumed,
                "terminal request consumed a lane owned by a context operation");
        require(session_.has_context_operation(), "declined terminal cancelled the capture");

        // B is bound while A's capture has not been delivered; its state must advance.
        const auto b = bind_b_during_capture();
        require(!session_.terminal_blocked(b), "idle lane B is blocked by A's capture");
        reserve_prefill(b);
        const auto start = session_.start_prefill({&b, 1}, {});
        require(start.has_value() && session_.has_context_operation(),
                "lane B could not start prefill while A's capture was in flight");
        const auto step = driver_.wait<q35::PrefillProgress>(start->ticket);
        require(step.rows.at(0).processed_prompt_tokens != 0,
                "lane B made no progress during A's capture");

        const auto progress = driver_.wait<q35::ContextProgress>();
        require(progress.published && !progress.captured_checkpoints.empty(),
                "A's capture did not publish after B advanced");
        driver_.idle();
        require(!session_.terminal_blocked(a), "A stays blocked after its capture settled");
        for (const auto point : progress.captured_checkpoints)
            require(session_.release_checkpoint(point), "captured checkpoint retained a reader");
        require(driver_.abort(a).status == runtime::ConsumeStatus::Consumed &&
                    driver_.abort(b).status == runtime::ConsumeStatus::Consumed,
                "lanes could not be released after capture");
        driver_.idle();
        require(same_ownership(session_.physical_usage(), empty_),
                "concurrent capture leaked Context resources");
    }

    // F4 acceptance 4: cancelling a capture in flight restores the pre-request capacity.
    void cancel_during_capture() {
        const auto a = start_capture_of_a();
        require(session_.terminal_blocked(a), "capturing lane is not terminal-blocked");
        session_.cancel_context();
        auto progress = driver_.wait<q35::ContextProgress>();
        driver_.idle();
        require(!session_.has_context_operation() && !session_.terminal_blocked(a),
                "cancelled capture left the lane blocked");
        for (const auto point : progress.captured_checkpoints)
            require(session_.release_checkpoint(point), "captured checkpoint retained a reader");
        require(driver_.abort(a).status == runtime::ConsumeStatus::Consumed,
                "cancelled capture's lane could not be aborted");
        driver_.idle();
        require(same_ownership(session_.physical_usage(), empty_),
                "cancelled capture did not return Context capacity");
    }

private:
    q35::SequenceHandle bind_b_during_capture() { return bind(plain_request(kChunk), 1); }

    native::NativeRankFixture& rank_;
    q35::Session& session_;
    models::qwen::Frontend& frontend_;
    q35::PhysicalUsageSnapshot empty_;
    native::EventDriver<q35::Session> driver_;
};

} // namespace

int main() {
    try {
        default_capture_policy();
    } catch (const std::exception& error) {
        std::cerr << "FAIL T3 capture default: " << error.what() << '\n';
        return 1;
    }
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "SKIP: set NINFER_TEST_ARTIFACT to an explicit .ninfer artifact\n";
        return 77;
    }
    try {
        DeviceContext device;
        models::LoadOptions selected;
        selected.speculative = SpeculativeBackend::None;
        artifact::Reader reader(artifact);
        auto definition = q35::read_definition(reader, selected);
        auto profile    = q35::execution::describe_model(*definition);
        std::shared_ptr<const q35::ModelShard> model =
            q35::materialize_model(q35::resolve_load(reader, definition, device.device()), device);
        auto frontend = models::qwen::make_frontend(
            definition->resources(), {.vision_enabled = false, .max_context = kCapacity},
            q35::default_sampling(definition->config().text.architecture));
        EngineOptions options;
        options.max_context         = kCapacity;
        options.prefill_chunk       = kChunk;
        options.max_concurrency     = 2;
        options.kv_cache            = KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;
        options.use_cuda_graph      = false;
        options.speculative.backend = SpeculativeBackend::None;
        const auto capacity_plan    = q35::make_sequence_planner(profile, device, options);
        options.context_cache.host_capacity_bytes =
            4 * describe_context_pool(capacity_plan.capacity_plan().records).block_bytes;
        for (const bool cancel : {false, true}) {
            native::NativeRankFixture rank(definition, model, profile, options,
                                           {.retained_states = 4});
            q35::Session session(q35::prepare_session(definition, std::move(rank.session_plan)),
                                 rank.executor());
            Scenario scenario(rank, session, frontend);
            if (cancel)
                scenario.cancel_during_capture();
            else
                scenario.other_lane_advances_during_capture();
        }
        std::cout << "OK T3 capture does not block other lanes; cancelled capture frees capacity\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL T3 capture: " << error.what() << '\n';
        return 1;
    }
}
