#include "core/device.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/program_impl.h"
#include "single_device_fit.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
namespace qwen      = models::qwen3_5;
namespace execution = qwen::execution;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

// These addresses belong to the fixture, rather than to an admitted Program continuation.
// Release their page and table leases before Program destroys its backing cache pools.
struct AddressOwner {
    qwen::detail::KVAddressSpaceStore* store = nullptr;
    qwen::detail::KVAddressSpaceHandle handle;

    ~AddressOwner() {
        if (store && !store->release_after_deactivate(handle)) { std::terminate(); }
    }
};

void append_bytes(std::vector<std::byte>& bytes, const Tensor& tensor) {
    const auto offset = bytes.size();
    bytes.resize(offset + tensor.bytes());
    CUDA_CHECK(
        cudaMemcpy(bytes.data() + offset, tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
}

std::vector<std::byte> local_bytes(qwen::detail::ProgramImpl& program, std::int32_t slot) {
    std::vector<std::byte> bytes;
    auto& cache = program.dflash->local;
    for (std::uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
        const auto view = cache.layer_view(layer);
        append_bytes(bytes, view.k.slice(3, slot, 1));
        append_bytes(bytes, view.v.slice(3, slot, 1));
    }
    return bytes;
}

bool any_nonzero(const std::vector<std::byte>& bytes) {
    return std::any_of(bytes.begin(), bytes.end(),
                       [](std::byte value) { return value != std::byte{0}; });
}

models::LoadOptions selected_load() {
    models::LoadOptions selected;
    selected.speculative   = SpeculativeBackend::DFlash2;
    selected.proposal_head = ProposalHead::Full;
    return selected;
}

void run(const char* artifact) {
    DeviceContext device;
    auto model = qwen::load_model(artifact, selected_load(), device);
    const execution::Parameters parameters(*model);
    const auto window   = model->config().draft->sliding_window;
    const auto capacity = ((window + 255) / 128) * 128;
    const auto pages    = capacity / kPagedKVPageSize;
    EngineOptions options;
    options.max_context                      = capacity;
    options.prefill_chunk                    = capacity;
    options.kv_capacity                      = KvCapacityPolicy::explicit_capacity(2 * capacity);
    options.max_concurrency                  = 2;
    options.context_cache.device_state_slots = 1;
    options.context_cache.host_state_slots   = 0;
    options.context_cache.host_kv_capacity_bytes            = 0;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    options.use_cuda_graph                                  = false;
    options.speculative.backend                             = SpeculativeBackend::DFlash2;
    options.speculative.draft_tokens                        = 3;
    options.speculative.proposal_head                       = ProposalHead::Full;
    auto planner = qwen::make_sequence_planner(parameters, device, options);
    auto plan    = std::move(planner).finalize(2 * pages);
    qwen::detail::ProgramImpl program(parameters, *plan.impl_, device, {});

    auto& states      = *program.state_store;
    const auto source = states.reserve_reset(device.stream);
    const auto other  = states.reserve_reset(device.stream);
    require(source && other, "fixture state allocation failed");
    const auto source_slot = states.physical_slot(*source);
    const auto other_slot  = states.physical_slot(*other);
    const auto text        = program.text_kv_addresses->create_active(pages, 1, device.stream);
    require(text.has_value(), "fixture text KV allocation failed");
    AddressOwner text_owner{program.text_kv_addresses.get(), *text};
    program.text_kv_addresses->ensure_mapped_to_tokens(*text, capacity, device.stream);
    execution::PrefillContext context{
        {device, parameters, program.work, program.state_images->linear(), &*program.replay_records,
         program.io, program.prefill_hidden, program.prefill_chunk, program.proposal_head},
        program.decoder->text_kv.execution_view(program.text_kv_addresses->execution_row(*text)),
        {},
        program.decoder->text_kv,
        nullptr,
        &*program.dflash,
        0,
        nullptr,
        nullptr,
        source_slot,
        source_slot,
        0,
        program.dflash_prefill_host_ingress};
    std::vector<TokenId> tokens(capacity);
    for (std::size_t i = 0; i < tokens.size(); ++i) { tokens[i] = 1000 + (i * 17) % 100; }

    const auto decode_binding = [&](std::int32_t slot) {
        *program.dflash_host_ingress                            = {};
        program.dflash_host_ingress->state_destination_slots[0] = slot;
        CUDA_CHECK(cudaMemcpyAsync(program.io.dflash_decode->ingress.data,
                                   program.dflash_host_ingress, sizeof(qwen::DFlashDecodeIngress),
                                   cudaMemcpyHostToDevice, device.stream));
    };
    const auto prefill = [&](std::uint32_t nominal, std::optional<std::uint32_t> split = {}) {
        const auto result   = execution::prefill_text_chunk(context, tokens, nominal, split, false);
        const auto expected = split ? std::min(nominal, *split - context.text_kv_base) : nominal;
        require(result.processed_tokens == expected && !result.finalized,
                "prefill did not honor its actual chunk boundary");
        context.text_kv_base += result.processed_tokens;
    };

    decode_binding(source_slot);
    prefill(16);
    const auto frozen = local_bytes(program, source_slot);
    require(any_nonzero(frozen), "first prefill did not populate draft local KV");
    const auto untouched_other = local_bytes(program, other_slot);

    // A checkpoint freezes the source and forks subsequent execution into another physical slot.
    // Leave decode ingress pointing to the old slot, exactly as it did at request admission.
    states.freeze(*source);
    const auto destination = states.reserve_destination();
    require(destination.has_value(), "fixture fork destination allocation failed");
    const auto binding = states.begin_fork(*source, *destination);
    program.state_images->copy_dflash_local(binding.source, binding.destination, device.stream);
    context.state_source_slot      = binding.source;
    context.state_destination_slot = binding.destination;
    prefill(23, 23); // Actual chunk is seven tokens, shortened at the capture frontier.
    states.commit_fork(*source, *destination);
    require(local_bytes(program, source_slot) == frozen,
            "prefill mutated the frozen checkpoint after a StateImage fork");
    const auto active = local_bytes(program, binding.destination);
    require(active != frozen, "prefill did not append to the fork destination");

    // Another compact decode batch rebinds the decode ingress to another state slot while this
    // prefill is suspended. Supply that binding deterministically; the assertions are on the slots.
    decode_binding(other_slot);
    context.state_source_slot = binding.destination;
    prefill(5);
    require(local_bytes(program, binding.destination) != active,
            "resumed prefill did not update its active slot");
    require(local_bytes(program, source_slot) == frozen &&
                local_bytes(program, other_slot) == untouched_other,
            "resumed prefill used another request's decode binding");

    // Local storage takes only the tail of an oversized chunk.
    prefill(window + 17);
    require(local_bytes(program, source_slot) == frozen &&
                local_bytes(program, other_slot) == untouched_other,
            "oversized prefill wrote outside its owned state");
    std::cout << "prefill fork, conflicting decode binding, split and oversized append passed\n";
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        if (const auto shortfall =
                ninfer::test::single_device_shortfall(artifact, selected_load())) {
            std::cout << "skip: " << *shortfall << '\n';
            return 77;
        }
        run(artifact);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
