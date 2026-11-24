#include "model_context_fixture.h"
#include "core/buffer.h"

#include "artifact/reader.h"
#include "core/device.h"
#include "models/common/context/storage.h"
#include "models/common/context/transfer.h"
#include "core/paged_kv_cache.h"
#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace ninfer::bench::context_cost {
namespace {

constexpr std::uint64_t kMiB                 = 1ULL << 20U;
constexpr std::uint32_t kMaximumFixturePages = 1024;

std::size_t checked_size_mul(std::size_t left, std::size_t right, const char* label) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(label);
    }
    return left * right;
}

double elapsed_ns(float milliseconds) {
    return std::max(1.0, static_cast<double>(milliseconds) * 1'000'000.0);
}

void fill_transfer_payload(void* destination, std::size_t bytes, std::uint64_t seed) {
    auto* output       = static_cast<std::byte*>(destination);
    std::size_t offset = 0;
    while (offset < bytes) {
        seed += 0x9e3779b97f4a7c15ULL;
        std::uint64_t value = seed;
        value               = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value               = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        value ^= value >> 31U;
        const std::size_t count = std::min(sizeof(value), bytes - offset);
        std::memcpy(output + offset, &value, count);
        offset += count;
    }
}

std::filesystem::path existing_input_path(const std::filesystem::path& path) {
    if (std::filesystem::exists(path)) { return path; }
#ifdef NINFER_SOURCE_DIR
    if (path.is_relative()) {
        const std::filesystem::path source = std::filesystem::path(NINFER_SOURCE_DIR) / path;
        if (std::filesystem::exists(source)) { return source; }
    }
#endif
    throw std::runtime_error("input path does not exist: " + path.string());
}

std::vector<TokenId> load_corpus(const std::filesystem::path& path) {
    const std::filesystem::path resolved = existing_input_path(path);
    std::ifstream input(resolved);
    if (!input) { throw std::runtime_error("failed to open token corpus: " + resolved.string()); }
    std::vector<TokenId> tokens;
    std::int64_t value = 0;
    while (input >> value) {
        if (value < 0 || value > std::numeric_limits<TokenId>::max()) {
            throw std::invalid_argument("token corpus contains an out-of-range token id");
        }
        tokens.push_back(static_cast<TokenId>(value));
    }
    if (tokens.empty()) { throw std::invalid_argument("token corpus is empty"); }
    return tokens;
}

std::vector<TokenId> prompt_slice(const std::vector<TokenId>& corpus, std::uint32_t count) {
    if (count == 0 || count > corpus.size()) {
        throw std::invalid_argument("requested prompt slice exceeds the token corpus");
    }
    return {corpus.begin(), corpus.begin() + static_cast<std::ptrdiff_t>(count)};
}

RequestOptions one_token_request() {
    RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.allow_prefix_reuse      = false;
    request.execution.sampling.temperature    = 0.0F;
    request.stop.include_model_defaults       = false;
    request.output.raw                        = true;
    request.output.preserve_special_tokens    = true;
    return request;
}

struct TransferCase {
    TransferDirection direction = TransferDirection::DeviceToHost;
    std::uint32_t pages         = 0;
    std::uint32_t runs          = 0;
    bool validation             = false;
};

std::uint32_t pages_for_bytes(std::uint64_t target_bytes, std::size_t payload_bytes) {
    if (payload_bytes == 0) { throw std::logic_error("transfer fixture page payload is zero"); }
    const std::uint64_t pages =
        (target_bytes + static_cast<std::uint64_t>(payload_bytes) - 1U) / payload_bytes;
    return static_cast<std::uint32_t>(std::clamp<std::uint64_t>(pages, 1, kMaximumFixturePages));
}

std::vector<TransferCase> transfer_cases(std::size_t payload_bytes) {
    std::vector<TransferCase> out;
    std::set<std::tuple<TransferDirection, std::uint32_t, std::uint32_t>> seen;
    const auto append = [&](TransferDirection direction, std::uint32_t pages, std::uint32_t runs,
                            bool validation) {
        runs           = std::clamp(runs, 1U, pages);
        const auto key = std::tuple{direction, pages, runs};
        if (seen.insert(key).second) {
            out.push_back(TransferCase{
                .direction = direction, .pages = pages, .runs = runs, .validation = validation});
        }
    };
    for (const std::uint64_t bytes : {1 * kMiB, 8 * kMiB, 64 * kMiB}) {
        const std::uint32_t pages = pages_for_bytes(bytes, payload_bytes);
        for (const TransferDirection direction :
             {TransferDirection::DeviceToHost, TransferDirection::HostToDevice}) {
            append(direction, pages, 1, false);
            append(direction, pages, 8, false);
        }
        append(TransferDirection::DeviceToDevice, pages, pages, false);
    }
    for (const std::uint64_t bytes : {4 * kMiB, 32 * kMiB}) {
        const std::uint32_t pages = pages_for_bytes(bytes, payload_bytes);
        for (const TransferDirection direction :
             {TransferDirection::DeviceToHost, TransferDirection::HostToDevice}) {
            append(direction, pages, 2, true);
            append(direction, pages, 4, true);
        }
        append(TransferDirection::DeviceToDevice, pages, pages, true);
    }
    return out;
}

std::vector<std::int64_t> select_runs(const std::vector<std::int64_t>& physical, std::size_t base,
                                      std::uint32_t pages, std::uint32_t runs) {
    if (pages == 0 || runs == 0 || runs > pages ||
        base + static_cast<std::size_t>(pages) + runs - 1U > physical.size()) {
        throw std::invalid_argument("invalid fragmented transfer selection");
    }
    std::vector<std::int64_t> selected;
    selected.reserve(pages);
    std::size_t cursor           = base;
    std::uint32_t remaining      = pages;
    std::uint32_t remaining_runs = runs;
    while (remaining_runs != 0) {
        const std::uint32_t extent = remaining / remaining_runs;
        for (std::uint32_t page = 0; page < extent; ++page) {
            selected.push_back(physical[cursor + page]);
        }
        cursor += extent;
        remaining -= extent;
        --remaining_runs;
        if (remaining_runs != 0) { ++cursor; }
    }
    return selected;
}

struct GeometryCase {
    const char* label;
    KVPageGeometry geometry;
};

std::vector<GeometryCase> transfer_geometries() {
    const auto records = [](std::size_t plane_count) {
        KVPageGeometry geometry{
            .page_tokens = kPagedKVPageSize,
        };
        geometry.planes.assign(plane_count, KVPlaneGeometry{.dtype          = DType::I8,
                                                            .leading_extent = 128,
                                                            .head_extent    = 2,
                                                            .alignment      = 256});
        return geometry;
    };
    return {
        GeometryCase{.label = "record-4", .geometry = records(4)},
        GeometryCase{.label = "record-64", .geometry = records(64)},
        GeometryCase{
            .label = "draft-full-2x8",
            .geometry =
                {
                    .page_tokens = kPagedKVPageSize,
                    .planes =
                        {
                            {DType::BF16, 128, 8, 256},
                            {DType::BF16, 128, 8, 256},
                        },
                },
        },
    };
}

void measure_geometry(DeviceContext& device, const GeometryCase& fixture,
                      const MeasurementOptions& options, std::vector<TransferMeasurement>& output) {
    namespace context           = models::context;
    const auto layout           = plan_kv_page_layout(fixture.geometry);
    const auto record_bytes     = layout.page_bytes;
    const auto cases            = transfer_cases(record_bytes);
    std::uint32_t maximum_pages = 0;
    std::uint32_t maximum_runs  = 0;
    for (const auto& test : cases) {
        maximum_pages = std::max(maximum_pages, test.pages);
        maximum_runs  = std::max(maximum_runs, test.runs);
    }
    const std::uint32_t region_pages   = maximum_pages + maximum_runs;
    const std::uint32_t physical_pages = 2U * region_pages;
    DeviceBuffer backing(device.device(), checked_size_mul(record_bytes, physical_pages,
                                                           "transfer fixture Device bytes"));
    const auto host_bytes =
        checked_size_mul(record_bytes, region_pages, "transfer fixture Host bytes");
    PinnedHostBuffer host(host_bytes);
    fill_transfer_payload(host.data(), host_bytes, 0x7ac5d3e91b2468f0ULL);
    context::ContextStorage storage(backing.span(), host.span(), {});
    const runtime::ContextTransferResource resource{runtime::KvPoolId{0}};
    context::ContextActions initialize;
    for (std::uint32_t region = 0; region < 2; ++region) {
        initialize.operations.push_back(context::RecordCopy{
            .source      = {context::ContextMemory::Host, 0, host_bytes},
            .destination = {context::ContextMemory::Device,
                            static_cast<std::int64_t>(region * host_bytes), host_bytes},
            .resource    = resource,
            .page_groups = region_pages});
    }
    storage.execute(initialize, {}, device.transfer_view());
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream()));

    std::vector<std::int64_t> offsets(physical_pages);
    for (std::uint32_t i = 0; i < physical_pages; ++i)
        offsets[i] = static_cast<std::int64_t>(i * record_bytes);
    std::vector<context::ContextActions> actions;
    actions.reserve(cases.size());
    for (const auto& test : cases) {
        const auto source      = select_runs(offsets, 0, test.pages, test.runs);
        const auto destination = select_runs(offsets, region_pages, test.pages, test.runs);
        context::ContextActions action;
        action.operations.reserve(test.pages);
        for (std::uint32_t i = 0; i < test.pages; ++i) {
            const auto host_offset = static_cast<std::int64_t>(i * record_bytes);
            context::ContextRange from{context::ContextMemory::Device, source[i], record_bytes};
            context::ContextRange to{context::ContextMemory::Device, destination[i], record_bytes};
            if (test.direction == TransferDirection::DeviceToHost)
                to = {context::ContextMemory::Host, host_offset, record_bytes};
            else if (test.direction == TransferDirection::HostToDevice)
                from = {context::ContextMemory::Host, host_offset, record_bytes};
            action.operations.push_back(context::RecordCopy{from, to, resource, 0, 1});
        }
        actions.push_back(std::move(action));
    }
    context::ContextMigration migration(device, storage, 0, 1);
    const auto run = [&](std::size_t index) {
        try {
            migration.submit(actions[index]);
            CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream()));
            const auto observations = migration.observations();
            if (observations.size() != 1)
                throw std::logic_error("transfer fixture must produce one observation segment");
            const auto observed = observations.front();
            migration.finish();
            return std::pair{static_cast<double>(observed.elapsed_ns), observed.work};
        } catch (...) {
            migration.drain();
            throw;
        }
    };

    for (std::size_t index = 0; index < cases.size(); ++index) {
        for (int warmup = 0; warmup < options.transfer_warmup; ++warmup) { (void)run(index); }
    }
    std::vector<std::vector<double>> samples(cases.size());
    std::vector<TransferWork> measured_work(cases.size());
    for (int repetition = 0; repetition < options.transfer_repetitions; ++repetition) {
        for (std::size_t offset = 0; offset < cases.size(); ++offset) {
            const std::size_t index =
                (offset + static_cast<std::size_t>(repetition)) % cases.size();
            const auto [elapsed, work] = run(index);
            samples[index].push_back(elapsed);
            measured_work[index] = work;
        }
    }
    for (std::size_t index = 0; index < cases.size(); ++index) {
        const TransferCase& test = cases[index];
        output.push_back(TransferMeasurement{
            .label           = fixture.label,
            .direction       = test.direction,
            .work            = measured_work[index],
            .page_count      = test.pages,
            .contiguous_runs = test.runs,
            .validation      = test.validation,
            .timing          = summarize_samples(std::move(samples[index])),
        });
    }
}

void measure_contiguous_d2d(DeviceContext& device, const MeasurementOptions& options,
                            std::vector<TransferMeasurement>& output) {
    struct ContiguousCase {
        std::size_t payload_bytes;
        std::uint32_t copy_operations;
        bool validation;
    };

    constexpr std::array<ContiguousCase, 10> cases{{
        {.payload_bytes = 1 * kMiB, .copy_operations = 1, .validation = false},
        {.payload_bytes = 1 * kMiB, .copy_operations = 8, .validation = false},
        {.payload_bytes = 8 * kMiB, .copy_operations = 1, .validation = false},
        {.payload_bytes = 8 * kMiB, .copy_operations = 8, .validation = false},
        {.payload_bytes = 64 * kMiB, .copy_operations = 1, .validation = false},
        {.payload_bytes = 64 * kMiB, .copy_operations = 8, .validation = false},
        {.payload_bytes = 4 * kMiB, .copy_operations = 2, .validation = true},
        {.payload_bytes = 4 * kMiB, .copy_operations = 4, .validation = true},
        {.payload_bytes = 32 * kMiB, .copy_operations = 2, .validation = true},
        {.payload_bytes = 32 * kMiB, .copy_operations = 4, .validation = true},
    }};
    constexpr std::size_t maximum_bytes = 64 * kMiB;

    PinnedHostBuffer host_source(maximum_bytes);
    fill_transfer_payload(host_source.data(), host_source.size(), 0xb3f71d6c4a928e05ULL);
    DeviceBuffer source(device.device(), maximum_bytes);
    DeviceBuffer destination(device.device(), maximum_bytes);
    source.copy_from_host(host_source.data(), host_source.size());
    destination.fill(0xa5);

    CudaEventTimer timer(device, device.transfer_stream());
    const auto run = [&](const ContiguousCase& test) {
        const std::size_t operation_bytes = test.payload_bytes / test.copy_operations;
        timer.start();
        for (std::uint32_t operation = 0; operation < test.copy_operations; ++operation) {
            const std::size_t offset = static_cast<std::size_t>(operation) * operation_bytes;
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(destination.data()) + offset,
                                       static_cast<const std::byte*>(source.data()) + offset,
                                       operation_bytes, cudaMemcpyDeviceToDevice,
                                       device.transfer_stream()));
        }
        return elapsed_ns(timer.stop_ms());
    };

    for (const ContiguousCase& test : cases) {
        for (int warmup = 0; warmup < options.transfer_warmup; ++warmup) { (void)run(test); }
    }
    std::array<std::vector<double>, cases.size()> samples;
    for (int repetition = 0; repetition < options.transfer_repetitions; ++repetition) {
        for (std::size_t offset = 0; offset < cases.size(); ++offset) {
            const std::size_t index =
                (offset + static_cast<std::size_t>(repetition)) % cases.size();
            samples[index].push_back(run(cases[index]));
        }
    }
    for (std::size_t index = 0; index < cases.size(); ++index) {
        const ContiguousCase& test = cases[index];
        output.push_back(TransferMeasurement{
            .label = "contiguous-d2d-b" + std::to_string(test.payload_bytes) + "-o" +
                     std::to_string(test.copy_operations),
            .direction = TransferDirection::DeviceToDevice,
            .work = {.payload_bytes = test.payload_bytes, .copy_operations = test.copy_operations},
            .page_count      = 0,
            .contiguous_runs = 0,
            .validation      = test.validation,
            .timing          = summarize_samples(std::move(samples[index])),
        });
    }
}

struct TextCase {
    std::uint32_t prefix = 0;
    std::uint32_t suffix = 0;
    bool validation      = false;
};

std::vector<TextCase> text_cases(std::uint32_t chunk) {
    if (chunk < 8) { throw std::invalid_argument("prefill chunk is too small for calibration"); }
    return {
        {.prefix = 0, .suffix = chunk / 8, .validation = false},
        {.prefix = 0, .suffix = chunk / 2, .validation = false},
        {.prefix = 0, .suffix = chunk, .validation = false},
        {.prefix = 0, .suffix = 2 * chunk, .validation = false},
        {.prefix = chunk, .suffix = chunk / 8, .validation = false},
        {.prefix = chunk, .suffix = chunk / 2, .validation = false},
        {.prefix = 4 * chunk, .suffix = chunk / 8, .validation = false},
        {.prefix = 4 * chunk, .suffix = chunk, .validation = false},
        {.prefix = 0, .suffix = 3 * chunk / 4, .validation = true},
        {.prefix = 2 * chunk, .suffix = chunk / 4, .validation = true},
        {.prefix = 2 * chunk, .suffix = 3 * chunk / 2, .validation = true},
        {.prefix = 6 * chunk, .suffix = chunk / 2, .validation = true},
    };
}

std::uint64_t attention_pairs(std::uint32_t prefix, std::uint32_t suffix) {
    const unsigned __int128 pairs = static_cast<unsigned __int128>(prefix) * suffix +
                                    static_cast<unsigned __int128>(suffix) * (suffix + 1ULL) / 2U;
    if (pairs > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("prefill attention-pair count exceeds uint64");
    }
    return static_cast<std::uint64_t>(pairs);
}

std::vector<std::uint8_t> block_ppm(int width, int height, std::uint8_t value) {
    if (width <= 0 || height <= 0) { throw std::invalid_argument("invalid synthetic image size"); }
    const std::string header =
        "P6\n" + std::to_string(width) + ' ' + std::to_string(height) + "\n255\n";
    const std::size_t pixels =
        checked_size_mul(static_cast<std::size_t>(width), static_cast<std::size_t>(height),
                         "synthetic image pixel count");
    std::vector<std::uint8_t> ppm;
    ppm.reserve(header.size() + checked_size_mul(pixels, 3, "synthetic image bytes"));
    for (const char byte : header) {
        ppm.push_back(static_cast<std::uint8_t>(static_cast<unsigned char>(byte)));
    }
    ppm.insert(ppm.end(), pixels * 3U, value);
    return ppm;
}

PromptInput image_prompt(int dimension) {
    MessagePart image;
    image.kind       = MessagePartKind::Media;
    image.media.kind = MediaKind::Image;
    image.media.bytes =
        block_ppm(dimension, dimension, static_cast<std::uint8_t>(dimension & 0xff));
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "context-cost-" + std::to_string(dimension) + ".ppm";

    ChatMessage message;
    message.role = ChatRole::User;
    message.parts.push_back(std::move(image));
    message.parts.push_back(MessagePart{
        .kind = MessagePartKind::Text, .text = "Describe the image briefly.", .media = {}});
    PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

} // namespace

ArtifactProfile inspect_artifact(const std::filesystem::path& artifact_path) {
    const std::filesystem::path path = existing_input_path(artifact_path);
    ninfer::artifact::Reader reader(path);
    return ArtifactProfile{.path = path};
}

TransferSuiteResult measure_context_transfers(const MeasurementOptions& options) {
    DeviceContext device(options.device);
    TransferSuiteResult result;
    for (const GeometryCase& fixture : transfer_geometries()) {
        measure_geometry(device, fixture, options, result.measurements);
    }
    // Page copies exercise the real paged-KV paths, but their bytes and operation count are
    // coupled. StateImage copies include large contiguous components, so add the same physical
    // cudaMemcpyAsync shape with independent byte and operation axes to identify D2D bandwidth.
    measure_contiguous_d2d(device, options, result.measurements);
    return result;
}

PrefillSuiteResult measure_prefill(const ArtifactProfile& artifact,
                                   const MeasurementOptions& options) {
    const std::vector<TextCase> cases = text_cases(options.prefill_chunk);
    std::uint32_t required_context    = 0;
    std::set<std::uint32_t> root_lengths;
    for (const TextCase& test : cases) {
        if (test.prefix != 0) { root_lengths.insert(test.prefix); }
        root_lengths.insert(test.prefix + test.suffix);
        required_context = std::max(required_context, test.prefix + test.suffix);
    }
    if (required_context > options.max_context) {
        throw std::invalid_argument("prefill calibration grid requires max_context >= " +
                                    std::to_string(required_context));
    }
    const std::vector<TokenId> corpus = load_corpus(options.corpus);
    if (corpus.size() < required_context) {
        throw std::invalid_argument("token corpus is shorter than the prefill calibration grid");
    }

    EngineOptions engine_options;
    engine_options.artifact_path                     = artifact.path;
    engine_options.device                            = options.device;
    engine_options.max_context                       = options.max_context;
    engine_options.context_capacity                  = ContextCapacityPolicy::automatic();
    engine_options.max_concurrency                   = 1;
    engine_options.max_pending_requests              = 1;
    engine_options.prefill_chunk                     = options.prefill_chunk;
    engine_options.kv_cache                          = KvCacheStorage::BFloat16;
    engine_options.speculative.backend               = SpeculativeBackend::None;
    engine_options.enable_vision                     = true;
    engine_options.use_cuda_graph                    = true;
    engine_options.context_cache.enabled             = false;
    engine_options.context_cache.host_capacity_bytes = 0;

    Engine engine(engine_options);
    const auto run_root = [&](std::uint32_t tokens) {
        auto prompt                = engine.prepare_tokens(prompt_slice(corpus, tokens), false);
        GenerationResult generated = engine.generate(std::move(prompt), one_token_request());
        if (generated.generated_token_ids.size() != 1 ||
            generated.finish_reason != FinishReason::OutputLimit ||
            generated.stats.prefix_reuse_path != PrefixReusePath::Root ||
            generated.stats.reused_prompt_tokens != 0 ||
            !(generated.stats.timings.prefill_seconds > 0.0)) {
            throw std::runtime_error("text prefill calibration left the expected root path");
        }
        return generated.stats.timings.prefill_seconds * 1'000'000'000.0;
    };

    const std::vector<std::uint32_t> lengths(root_lengths.begin(), root_lengths.end());
    (void)run_root(lengths.front());
    if (lengths.back() != lengths.front()) { (void)run_root(lengths.back()); }
    std::map<std::uint32_t, std::vector<double>> root_samples;
    for (const std::uint32_t length : lengths) {
        root_samples.emplace(length, std::vector<double>{});
    }
    for (int repetition = 0; repetition < options.prefill_repetitions; ++repetition) {
        for (std::size_t offset = 0; offset < lengths.size(); ++offset) {
            const std::size_t index =
                (offset + static_cast<std::size_t>(repetition)) % lengths.size();
            root_samples.at(lengths[index]).push_back(run_root(lengths[index]));
        }
    }

    PrefillSuiteResult result;
    result.load = engine.startup_report().load;
    for (const TextCase& test : cases) {
        std::vector<double> samples;
        const auto& endpoint = root_samples.at(test.prefix + test.suffix);
        if (test.prefix == 0) {
            samples = endpoint;
        } else {
            const auto& base = root_samples.at(test.prefix);
            for (std::size_t index = 0; index < endpoint.size(); ++index) {
                const double difference = endpoint[index] - base[index];
                if (!(difference > 0.0)) {
                    throw std::runtime_error(
                        "paired root prefill produced a nonpositive suffix time");
                }
                samples.push_back(difference);
            }
        }
        result.measurements.push_back(PrefillMeasurement{
            .label = "text-b" + std::to_string(test.prefix) + "-s" + std::to_string(test.suffix),
            .prefix_tokens   = test.prefix,
            .suffix_tokens   = test.suffix,
            .chunks          = (test.suffix + options.prefill_chunk - 1U) / options.prefill_chunk,
            .attention_pairs = attention_pairs(test.prefix, test.suffix),
            .vision_items    = 0,
            .vision_patches  = 0,
            .validation      = test.validation,
            .timing          = summarize_samples(std::move(samples)),
        });
    }

    struct VisionCase {
        int dimension;
        bool validation;
    };

    constexpr std::array<VisionCase, 4> vision_cases{{
        {.dimension = 224, .validation = false},
        {.dimension = 448, .validation = false},
        {.dimension = 672, .validation = false},
        {.dimension = 560, .validation = true},
    }};
    const auto run_vision = [&](int dimension) {
        PreparedPrompt prompt       = engine.prepare(image_prompt(dimension));
        const std::uint64_t patches = prompt.preparation_stats().raw_patches;
        GenerationResult generated  = engine.generate(std::move(prompt), one_token_request());
        if (patches == 0 || generated.generated_token_ids.size() != 1 ||
            generated.finish_reason != FinishReason::OutputLimit || !generated.prompt.has_media ||
            !(generated.stats.timings.vision_seconds > 0.0)) {
            throw std::runtime_error("Vision prefill calibration did not execute Vision");
        }
        return std::pair{patches, generated.stats.timings.vision_seconds * 1'000'000'000.0};
    };
    (void)run_vision(vision_cases.back().dimension);
    std::array<std::vector<double>, vision_cases.size()> vision_samples;
    std::array<std::uint64_t, vision_cases.size()> patches{};
    for (int repetition = 0; repetition < options.prefill_repetitions; ++repetition) {
        for (std::size_t offset = 0; offset < vision_cases.size(); ++offset) {
            const std::size_t index =
                (offset + static_cast<std::size_t>(repetition)) % vision_cases.size();
            const auto [count, nanoseconds] = run_vision(vision_cases[index].dimension);
            if (patches[index] != 0 && patches[index] != count) {
                throw std::runtime_error("synthetic image patch count changed between runs");
            }
            patches[index] = count;
            vision_samples[index].push_back(nanoseconds);
        }
    }
    for (std::size_t index = 0; index < vision_cases.size(); ++index) {
        result.measurements.push_back(PrefillMeasurement{
            .label           = "vision-" + std::to_string(vision_cases[index].dimension),
            .prefix_tokens   = 0,
            .suffix_tokens   = 0,
            .chunks          = 0,
            .attention_pairs = 0,
            .vision_items    = 1,
            .vision_patches  = patches[index],
            .validation      = vision_cases[index].validation,
            .timing          = summarize_samples(std::move(vision_samples[index])),
        });
    }
    return result;
}

const char* transfer_direction_name(TransferDirection direction) noexcept {
    switch (direction) {
    case TransferDirection::DeviceToHost:
        return "d2h";
    case TransferDirection::HostToDevice:
        return "h2d";
    case TransferDirection::DeviceToDevice:
        return "d2d";
    }
    return "unknown";
}

} // namespace ninfer::bench::context_cost
