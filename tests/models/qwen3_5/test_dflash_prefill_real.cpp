#include "core/device.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/execution/profile.h"
#include "models/qwen3_5/execution/draft.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/program/resources.h"
#include "models/qwen3_5/program/planning/startup.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
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

void append_bytes(std::vector<std::byte>& bytes, const Tensor& tensor) {
    const auto offset = bytes.size();
    bytes.resize(offset + tensor.bytes());
    CUDA_CHECK(
        cudaMemcpy(bytes.data() + offset, tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
}

std::vector<std::byte> local_bytes(qwen::Resources& resources, std::int64_t offset) {
    std::vector<std::byte> bytes;
    auto& cache = resources.dflash->local;
    for (std::uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
        const auto view = cache.layer_view(layer, {});
        append_bytes(bytes, view.k.tensor(offset));
        append_bytes(bytes, view.v.tensor(offset));
    }
    return bytes;
}

std::vector<std::byte> full_bytes(qwen::Resources& resources, std::int32_t row,
                                  std::uint32_t pages) {
    std::vector<std::byte> bytes;
    if (!resources.dflash->full) { return bytes; }
    auto& cache      = *resources.dflash->full;
    const auto table = cache.tables().matrix().slice(1, row, 1);
    std::vector<std::int64_t> offsets(pages);
    CUDA_CHECK(cudaMemcpy(offsets.data(), table.data, offsets.size() * sizeof(std::int64_t),
                          cudaMemcpyDeviceToHost));
    for (std::uint32_t layer = 0; layer < cache.layers(); ++layer) {
        const auto view = cache.batch_layer_view(layer);
        for (const auto& plane : {view.k_pages, view.v_pages}) {
            for (const auto offset : offsets) {
                require(offset >= 0, "fixture KV record is not mapped");
                append_bytes(bytes, plane.tensor(offset));
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
void require_full_positions(qwen::Resources& resources, std::int32_t row, std::uint32_t begin,
                            std::uint32_t end) {
    if (!resources.dflash->full) { return; }
    auto& cache      = *resources.dflash->full;
    const auto table = cache.tables().matrix().slice(1, row, 1);
    std::vector<std::int64_t> offsets((end + 63) / 64);
    CUDA_CHECK(cudaMemcpy(offsets.data(), table.data, offsets.size() * sizeof(std::int64_t),
                          cudaMemcpyDeviceToHost));
    for (std::uint32_t layer = 0; layer < cache.layers(); ++layer) {
        const auto view = cache.batch_layer_view(layer);
        for (const auto& plane : {view.k_pages, view.v_pages}) {
            for (auto page = begin / 64; page < offsets.size(); ++page) {
                std::vector<std::byte> host;
                const auto field = plane.tensor(offsets[page]);
                append_bytes(host, field);
                for (auto position = std::max(begin, page * 64);
                     position < std::min(end, (page + 1) * 64); ++position) {
                    const auto offset = (position % 64) * field.nb[1];
                    require(std::any_of(host.begin() + offset, host.begin() + offset + field.nb[1],
                                        [](std::byte value) { return value != std::byte{0}; }),
                            "full KV omitted an appended absolute position");
                }
            }
        }
    }
}

void run(const char* artifact, SpeculativeBackend backend) {
    DeviceContext device;
    models::LoadOptions selected;
    selected.speculative   = backend;
    selected.proposal_head = ProposalHead::Full;
    artifact::Reader reader(artifact);
    auto definition = qwen::read_definition(reader, selected);
    auto profile    = execution::describe_model(*definition);
    auto model =
        qwen::materialize_model(qwen::resolve_load(reader, definition, device.device()), device);
    const execution::Parameters parameters(*model, profile);
    const auto window   = *profile->config.draft->sliding_window;
    const auto capacity = ((window + 255) / 128) * 128;
    const auto pages    = capacity / kPagedKVPageSize;
    EngineOptions options;
    options.max_context                       = capacity;
    options.prefill_chunk                     = capacity;
    options.max_concurrency                   = 2;
    options.context_cache.host_capacity_bytes = 0;
    options.use_cuda_graph                    = false;
    options.speculative.backend               = backend;
    options.speculative.draft_tokens          = 3;
    options.speculative.proposal_head         = ProposalHead::Full;
    auto planner = qwen::detail::make_sequence_planner_impl(parameters.profile, device, options);
    const auto minimum = planner->capacity_plan.minimum_capacity();
    auto plan          = qwen::detail::finalize_sequence_plan_impl(std::move(planner), minimum);
    qwen::Resources resources(device, plan->persistent, plan->workspace);

    ContextPool pool(0, resources.context.size(), resources.context_records);
    const auto state_class = pool.record_class(0);
    std::vector<RecordRequest> demand{{state_class, 3}, {pool.record_class(1), 2 * pages}};
    if (resources.dflash->full) demand.push_back({pool.record_class(2), 2 * pages});
    auto reservation = pool.reserve(demand);
    require(reservation.reservation && reservation.reservation->ready(),
            "fixture Context capacity does not hold its State and histories");
    const auto allocate = [&](RecordClassId record_class) {
        return pool.resolve(reservation.reservation->materialize(record_class));
    };
    const auto source_offset      = static_cast<std::int64_t>(allocate(state_class).offset);
    const auto other_offset       = static_cast<std::int64_t>(allocate(state_class).offset);
    const auto destination_offset = static_cast<std::int64_t>(allocate(state_class).offset);
    auto& states                  = resources.state_images;
    for (const auto offset : {source_offset, other_offset, destination_offset})
        states.storage().clear(offset, device.execution_view());

    const auto map_rows = [&](qwen::PagedKVCache& cache, RecordClassId record_class) {
        std::vector<std::int64_t> mapping(pages);
        for (const auto row : {0, 1}) {
            for (auto& offset : mapping) {
                const auto record = allocate(record_class);
                offset            = static_cast<std::int64_t>(record.offset);
                CUDA_CHECK(cudaMemsetAsync(cache.arena().subspan(record.offset, record.bytes).data,
                                           0, record.bytes, device.stream()));
            }
            const auto table = cache.tables().row(row);
            CUDA_CHECK(cudaMemcpy(table.data, mapping.data(), mapping.size() * sizeof(std::int64_t),
                                  cudaMemcpyHostToDevice));
        }
    };
    map_rows(resources.decoder.text_kv, pool.record_class(1));
    if (resources.dflash->full) map_rows(*resources.dflash->full, pool.record_class(2));

    // The prefill selectors have their own storage. An unrelated decode frame can change
    // its bindings without changing the prefill's StateImage or full-attention row.
    struct AppendSelectors {
        std::int64_t state_offset;
        std::int32_t table_row;
        std::int32_t count;
    };

    DeviceBuffer selector_storage(device.device(), sizeof(AppendSelectors));
    const Tensor append_offset(selector_storage.span(), DType::I64, {1});
    const Tensor table_row(selector_storage.span().subspan(offsetof(AppendSelectors, table_row)),
                           DType::I32, {1});
    const Tensor append_count(selector_storage.span().subspan(offsetof(AppendSelectors, count)),
                              DType::I32, {1});
    std::vector<TokenId> tokens(capacity);
    for (std::size_t i = 0; i < tokens.size(); ++i) tokens[i] = 1000 + (i * 17) % 100;
    std::uint32_t frontier = 0;
    auto state_source = source_offset, state_destination = source_offset;

    const auto decode_binding = [&](std::int64_t offset, std::int32_t row) {
        auto& frame                                      = resources.frame;
        *frame.host_ingress                              = {};
        frame.host_ingress->state_destination_offsets[0] = offset;
        frame.host_ingress->backend_table_rows[0]        = row;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, frame.host_ingress,
                                   sizeof(qwen::FrameIngress), cudaMemcpyHostToDevice,
                                   device.stream()));
    };
    const auto prefill = [&](std::uint32_t count) {
        require(count && frontier + count <= tokens.size(), "fixture prefill exceeds its prompt");
        const auto width = static_cast<std::int32_t>(count);
        const AppendSelectors selectors{state_destination, 1, width};
        CUDA_CHECK(cudaMemcpy(selector_storage.data(), &selectors, sizeof(selectors),
                              cudaMemcpyHostToDevice));
        auto& frame = resources.frame;
        auto host   = frame.host_input_view(width, 1);
        auto input  = frame.input_view(width, 1);
        std::copy_n(tokens.begin() + frontier, count, static_cast<TokenId*>(host.ids.data));
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto position = static_cast<std::int32_t>(frontier + i);
            static_cast<std::int32_t*>(host.cache_positions.data)[i] = position;
            for (int axis = 0; axis < 3; ++axis)
                static_cast<std::int32_t*>(host.rope_positions.data)[axis * count + i] = position;
        }
        for (const auto& pair : {std::pair{host.ids, input.ids},
                                 std::pair{host.cache_positions, input.cache_positions},
                                 std::pair{host.rope_positions, input.rope_positions}})
            CUDA_CHECK(cudaMemcpyAsync(pair.second.data, pair.first.data, pair.second.bytes(),
                                       cudaMemcpyHostToDevice, device.stream()));
        auto features  = resources.dflash->prefill_features.slice(1, 0, width);
        auto positions = resources.dflash->prefill_positions.slice(0, 0, width);
        const execution::DFlashFeatureSink sink{
            features, positions, {}, profile->config.draft->target_layer_ids};
        {
            execution::TextRun run({parameters, resources.work, device.execution_view(),
                                    resources.decoder.text_kv, states.linear()},
                                   {input.ids,
                                    input.cache_positions,
                                    input.rope_positions,
                                    append_count,
                                    table_row,
                                    {},
                                    {},
                                    {}},
                                   execution::SequenceUpdate{state_source, state_destination},
                                   {frontier + count, frontier + count}, input.hidden, &sink);
            require(run.start(), "resident Main unexpectedly yielded for offloaded weights");
        }
        const execution::DraftExecutionView draft{
            parameters, resources.work, device.execution_view(), resources.dflash->local,
            resources.dflash->full ? &*resources.dflash->full : nullptr};
        execution::draft_append_context(draft, features, positions, append_count, append_offset,
                                        table_row, {count, count});
        device.synchronize();
        frontier += count;
    };

    decode_binding(source_offset, 1);
    prefill(16);
    const auto frozen = local_bytes(resources, source_offset);
    require(any_nonzero(frozen), "first prefill did not populate draft local KV");
    const auto untouched_other = local_bytes(resources, other_offset);
    const auto untouched_full  = full_bytes(resources, 0, pages);
    require_full_positions(resources, 1, 0, 16);

    // A checkpoint freezes the source and forks execution into another State record.
    // Leave decode ingress pointing to the source, exactly as it did at request admission.
    states.storage().copy_group(states.dflash_local_group(), source_offset, destination_offset,
                                device.execution_view());
    state_source      = source_offset;
    state_destination = destination_offset;
    prefill(7); // A chunk shortened at the capture frontier 23.
    require(local_bytes(resources, source_offset) == frozen,
            "prefill mutated the frozen checkpoint after a StateImage fork");
    const auto active = local_bytes(resources, destination_offset);
    require(active != frozen, "prefill did not append to the fork destination");
    require_full_positions(resources, 1, 16, 23);

    // Another compact decode batch overwrites row zero while this prefill is suspended.
    // Supply the exact conflicting binding deterministically; the assertion is on KV contents.
    decode_binding(other_offset, 0);
    state_source = destination_offset;
    prefill(5);
    require(local_bytes(resources, destination_offset) != active,
            "resumed prefill did not update its active State");
    require(local_bytes(resources, source_offset) == frozen &&
                local_bytes(resources, other_offset) == untouched_other &&
                full_bytes(resources, 0, pages) == untouched_full,
            "resumed prefill used another request's decode binding");
    require_full_positions(resources, 1, 23, 28);

    // Local storage takes only the tail; full KV must receive every position of the same chunk.
    prefill(window + 17);
    require(local_bytes(resources, source_offset) == frozen &&
                local_bytes(resources, other_offset) == untouched_other &&
                full_bytes(resources, 0, pages) == untouched_full,
            "oversized prefill wrote outside its owned state or KV row");
    require_full_positions(resources, 1, 28, frontier);
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
