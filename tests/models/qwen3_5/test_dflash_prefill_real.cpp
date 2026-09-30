#include "core/device.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/program_impl.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string_view>
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

std::vector<std::byte> full_bytes(qwen::detail::ProgramImpl& program, std::int32_t row,
                                  std::uint32_t pages) {
    std::vector<std::byte> bytes;
    if (!program.dflash->full) { return bytes; }
    auto& cache      = *program.dflash->full;
    const auto table = cache.execution_tables().matrix().slice(1, row, 1);
    std::vector<std::int32_t> physical(pages);
    CUDA_CHECK(cudaMemcpy(physical.data(), table.data, physical.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    const auto& pool = cache.page_pool();
    require(pool.geometry().device_plane_order == PagedKVPlaneOrder::HeadMajor,
            "fixture requires the draft model's head-major KV storage");
    for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
        for (const auto page : physical) {
            require(page >= 0, "fixture KV page is not mapped");
            for (std::int32_t head = 0; head < pool.plane(plane).ne[3]; ++head) {
                append_bytes(bytes, pool.plane(plane).slice(3, head, 1).slice(2, page, 1));
            }
        }
    }
    return bytes;
}

bool any_nonzero(const std::vector<std::byte>& bytes) {
    return std::any_of(bytes.begin(), bytes.end(),
                       [](std::byte value) { return value != std::byte{0}; });
}

// Check each absolute position, including both ends of an oversized chunk. This detects a
// truncated full append as well as a write to the wrong execution row.
void require_full_positions(qwen::detail::ProgramImpl& program, std::int32_t row,
                            std::uint32_t begin, std::uint32_t end) {
    if (!program.dflash->full) { return; }
    auto& cache      = *program.dflash->full;
    const auto table = cache.execution_tables().matrix().slice(1, row, 1);
    std::vector<std::int32_t> physical((end + 63) / 64);
    CUDA_CHECK(cudaMemcpy(physical.data(), table.data, physical.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    for (std::uint32_t layer = 0; layer < cache.layers(); ++layer) {
        const auto view = cache.batch_layer_view(layer);
        for (const auto& plane : {view.k_pages, view.v_pages}) {
            std::vector<std::byte> host;
            append_bytes(host, plane);
            for (auto position = begin; position < end; ++position) {
                const auto offset =
                    physical[position / 64] * plane.nb[2] + (position % 64) * plane.nb[1];
                require(std::any_of(host.begin() + offset, host.begin() + offset + plane.nb[1],
                                    [](std::byte value) { return value != std::byte{0}; }),
                        "full KV omitted an appended absolute position");
            }
        }
    }
}

void run(const char* artifact, SpeculativeBackend backend) {
    DeviceContext device;
    models::LoadOptions selected;
    selected.speculative   = backend;
    selected.proposal_head = ProposalHead::Full;
    auto model             = qwen::load_model(artifact, selected, device);
    const execution::Parameters parameters(*model);
    const auto window   = *model->config().draft->sliding_window;
    const auto capacity = ((window + 255) / 128) * 128;
    const auto pages    = capacity / kPagedKVPageSize;
    EngineOptions options;
    options.max_context                      = capacity;
    options.prefill_chunk                    = capacity;
    options.kv_capacity                      = KvCapacityPolicy::explicit_capacity(2 * capacity);
    options.max_concurrency                  = 2;
    options.context_cache.host_cache_bytes      = 0;
    options.context_cache.device_snapshot_slots = 1;
    options.context_cache.max_new_taps          = 0;
    options.context_cache.tap_ladder_tokens     = 4096;
    options.context_cache.tap_min_gap_tokens    = 1024;
    options.use_cuda_graph                                  = false;
    options.speculative.backend                             = backend;
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
    const auto text        = program.text_kv_addresses->create_active(pages, 1);
    require(text.has_value(), "fixture text KV allocation failed");
    AddressOwner text_owner{program.text_kv_addresses.get(), *text};
    program.text_kv_addresses->ensure_mapped_to_tokens(*text, capacity, device.stream);
    std::array<AddressOwner, 2> full_owners;
    if (program.backend_kv_addresses) {
        for (const auto row : {0, 1}) {
            const auto address = program.backend_kv_addresses->create_active(pages, row);
            require(address.has_value(), "fixture full KV allocation failed");
            full_owners[row].store  = program.backend_kv_addresses.get();
            full_owners[row].handle = *address;
            program.backend_kv_addresses->ensure_mapped_to_tokens(*address, capacity,
                                                                  device.stream);
        }
        auto& pool = program.dflash->full->page_pool();
        for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
            CUDA_CHECK(cudaMemsetAsync(pool.plane(plane).data, 0, pool.plane(plane).bytes(),
                                       device.stream));
        }
    }
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
        1,
        program.dflash_prefill_host_ingress};
    std::vector<TokenId> tokens(capacity);
    for (std::size_t i = 0; i < tokens.size(); ++i) { tokens[i] = 1000 + (i * 17) % 100; }

    const auto decode_binding = [&](std::int32_t slot, std::int32_t row) {
        *program.dflash_host_ingress                            = {};
        program.dflash_host_ingress->state_destination_slots[0] = slot;
        program.dflash_host_ingress->dflash_kv_table_rows[0]    = row;
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

    decode_binding(source_slot, 1);
    prefill(16);
    const auto frozen = local_bytes(program, source_slot);
    require(any_nonzero(frozen), "first prefill did not populate draft local KV");
    const auto untouched_other = local_bytes(program, other_slot);
    const auto untouched_full  = full_bytes(program, 0, pages);
    require_full_positions(program, 1, 0, 16);

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
    require_full_positions(program, 1, 16, 23);

    // Another compact decode batch overwrites row zero while this prefill is suspended.
    // Supply the exact conflicting binding deterministically; the assertion is on KV contents.
    decode_binding(other_slot, 0);
    context.state_source_slot = binding.destination;
    prefill(5);
    require(local_bytes(program, binding.destination) != active,
            "resumed prefill did not update its active slot");
    require(local_bytes(program, source_slot) == frozen &&
                local_bytes(program, other_slot) == untouched_other &&
                full_bytes(program, 0, pages) == untouched_full,
            "resumed prefill used another request's decode binding");
    require_full_positions(program, 1, 23, 28);

    // Local storage takes only the tail; full KV must receive every position of the same chunk.
    prefill(window + 17);
    require(local_bytes(program, source_slot) == frozen &&
                local_bytes(program, other_slot) == untouched_other &&
                full_bytes(program, 0, pages) == untouched_full,
            "oversized prefill wrote outside its owned state or KV row");
    require_full_positions(program, 1, 28, context.text_kv_base);
    std::cout << "prefill fork, conflicting decode binding, split and oversized append passed ("
              << (backend == SpeculativeBackend::DFlash2 ? "dflash2" : "dflash") << ")\n";
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        const std::string_view backend = argc > 1 ? argv[1] : "dflash2";
        require(backend == "dflash" || backend == "dflash2", "expected dflash or dflash2");
        run(artifact,
            backend == "dflash" ? SpeculativeBackend::DFlash : SpeculativeBackend::DFlash2);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
