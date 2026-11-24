#include "core/buffer.h"
#include "core/decode_graph.h"
#include "core/expert_cache.h"

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace ninfer::test {
void expert_cache_cuda_errors(const ExpertBankSource&, const DeviceContext&);
}

namespace {
using namespace ninfer;

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

void check(cudaError_t status) {
    if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
}

struct Matrix {
    HostBuffer bytes;
    CudaHostRegistration registration;
    DeviceBuffer gpu;
    std::vector<float> divisors;
    WeightParent parent;

    Matrix(QType format, std::uint32_t count, std::uint64_t n, std::uint64_t k, unsigned seed) {
        const bool nv    = format == QType::NVFP4;
        const bool fp8   = format == QType::FP8_E4M3FN_BLOCK128X128_BF16;
        const auto shape = nv || fp8 ? std::vector<std::uint64_t>{count, n, k}
                                     : std::vector<std::uint64_t>{count * n, k};
        const auto g     = weight_geometry(format,
                                       nv    ? QuantLayout::BatchedBlockScaleK16M128x4
                                           : fp8 ? QuantLayout::BlockScaleM128K128
                                                 : QuantLayout::RowSplit,
                                           shape);
        bytes            = HostBuffer(g.bytes, 4096);
        auto* data       = static_cast<std::byte*>(bytes.data());
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            data[i] = std::byte((i * 17 + i / 4096 + seed) % 251);
        }
        if (nv) {
            for (unsigned e = 0; e < count; ++e) { divisors.push_back(seed + e + 1.0F); }
            std::memcpy(data + g.divisor_offset, divisors.data(), count * 4);
        }
        parent = {g, Device::cpu(), data, divisors};
    }

    WeightBankView view(std::uint32_t count, std::uint64_t n) {
        return bind_weight_bank(parent, describe_weight_bank(parent.geometry, {0, count, n, n}));
    }

    void upload(Device device) {
        gpu = DeviceBuffer(device, bytes.size());
        gpu.copy_from_host(bytes.data(), bytes.size());
        parent.device = device;
        parent.data   = static_cast<const std::byte*>(gpu.data());
    }
};

struct Bank {
    Matrix gu, down;
    std::uint32_t count;

    Bank(QType gu_type, QType down_type, std::uint32_t count = 6)
        : gu(gu_type, count, 256, 128, 3), down(down_type, count, 128, 128, 13), count(count) {}

    ExpertBankSource source(Device cuda) {
        const auto upload = [&](Matrix& matrix) {
            if (matrix.parent.device != Device::cpu()) { return CudaHostView{}; }
            if (!matrix.registration.view().host.bytes) {
                matrix.registration = CudaHostRegistration(matrix.bytes.span(), cuda);
            }
            return matrix.registration.view();
        };
        return {gu.view(count, 256), down.view(count, 128), upload(gu), upload(down)};
    }
};

struct Sizes {
    std::uint64_t bytes[6];
};

Sizes sizes(const ExpertBankSource& source) {
    return {{source.gate_up.layout.codes.expert_bytes, source.gate_up.layout.high.expert_bytes,
             source.gate_up.layout.scales.expert_bytes, source.down.layout.codes.expert_bytes,
             source.down.layout.high.expert_bytes, source.down.layout.scales.expert_bytes}};
}

std::size_t total(Sizes sizes) {
    std::size_t result = 8;
    for (const auto n : sizes.bytes) { result += n; }
    return result;
}

__global__ void gather(ExpertBankDeviceView bank, unsigned expert, Sizes sizes,
                       unsigned char* out) {
    const auto i        = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    std::uint64_t start = 0;
    for (int p = 0; p < 6; ++p) {
        const auto matrix = p < 3 ? bank.gate_up : bank.down;
        const auto plane  = p % 3;
        const auto* base  = plane == 0 ? matrix.codes : plane == 1 ? matrix.high : matrix.scales;
        const auto stride = plane == 0   ? matrix.code_stride
                            : plane == 1 ? matrix.high_stride
                                         : matrix.scale_stride;
        if (i >= start && i < start + sizes.bytes[p]) {
            const auto offset = matrix.offsets ? matrix.offsets[expert] : expert * stride;
            out[i]            = offset == UINT64_MAX
                                    ? 255
                                    : reinterpret_cast<const unsigned char*>(base)[offset + i - start];
        }
        start += sizes.bytes[p];
    }
    if (!i) {
        const float gu =
            bank.gate_up.divisors ? bank.gate_up.divisors[expert * bank.gate_up.divisor_stride] : 1;
        const float dn =
            bank.down.divisors ? bank.down.divisors[expert * bank.down.divisor_stride] : 1;
        // The byte output need not be aligned for typed stores.
        for (int b = 0; b < 4; ++b) {
            out[start + b]     = reinterpret_cast<const unsigned char*>(&gu)[b];
            out[start + 4 + b] = reinterpret_cast<const unsigned char*>(&dn)[b];
        }
    }
}

std::vector<std::byte> expected(const ExpertBankSource& source, unsigned expert) {
    std::vector<std::byte> out;
    for (const auto* matrix : {&source.gate_up, &source.down}) {
        for (const auto& plane :
             {matrix->layout.codes, matrix->layout.high, matrix->layout.scales}) {
            const auto data = matrix->plane(plane, expert);
            if (!data.bytes) { continue; }
            const auto* bytes = static_cast<const std::byte*>(data.data);
            if (data.device == Device::cpu()) {
                out.insert(out.end(), bytes, bytes + data.bytes);
            } else {
                const auto offset = out.size();
                out.resize(offset + data.bytes);
                check(
                    cudaMemcpy(out.data() + offset, data.data, data.bytes, cudaMemcpyDeviceToHost));
            }
        }
    }
    for (const auto* matrix : {&source.gate_up, &source.down}) {
        const auto value  = matrix->divisor(expert);
        const auto* bytes = reinterpret_cast<const std::byte*>(&value);
        out.insert(out.end(), bytes, bytes + 4);
    }
    return out;
}

ExpertCacheUse acquire(ExpertCache& cache, std::span<const ExpertKey> keys,
                       CudaExecutionView execution) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    do {
        cache.progress();
        if (auto use = cache.acquire(keys, execution)) { return *use; }
        std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < deadline);
    throw std::runtime_error("expert readiness did not progress");
}

void verify(ExpertCache& cache, const ExpertBankSource& source, ExpertKey key,
            const DeviceContext& device) {
    const std::array keys{key};
    const auto use  = acquire(cache, keys, device.execution_view());
    const auto view = cache.view(key.bank);
    require(view.device == device.device(), "expert view lost its device");
    for (const auto* matrix : {&view.gate_up, &view.down}) {
        if (matrix->offsets) {
            require(matrix->indexed_storage.device == device.device() &&
                        matrix->indexed_storage.data && matrix->indexed_storage.bytes,
                    "indexed expert view lost its backing range");
        }
    }
    const auto dimensions = sizes(source);
    DeviceBuffer output(device.device(), total(dimensions));
    gather<<<(output.size() + 255) / 256, 256, 0, device.stream()>>>(
        cache.view(key.bank), key.expert, dimensions, static_cast<unsigned char*>(output.data()));
    check(cudaGetLastError());
    cache.release(use, device.execution_view());
    device.synchronize();
    std::vector<std::byte> bytes(output.size());
    output.copy_to_host(bytes.data(), bytes.size());
    require(bytes == expected(source, key.expert),
            "cached planes or logical divisor differ from source");
}

std::vector<ExpertPlacement> packed(const ExpertCacheRequirements& req,
                                    std::span<const ExpertKey> keys) {
    std::vector<ExpertPlacement> out;
    std::size_t offset = 0;
    for (const auto key : keys) {
        if (std::any_of(out.begin(), out.end(),
                        [&](const auto& entry) { return entry.key == key; }))
            continue;
        out.push_back({key, offset});
        offset += req.banks[key.bank.index].entry_bytes;
    }
    return out;
}

void replace(ExpertCache& cache, const ExpertCacheRequirements& req,
             std::span<const ExpertKey> keys) {
    cache.replace({0, req.config.capacity_bytes, 256}, packed(req, keys));
    cache.prepare({keys});
}

void codecs_and_chunks(const DeviceContext& device) {
    for (const auto types :
         {std::array{QType::Q4_G64_FP16, QType::Q5_G64_FP16},
          std::array{QType::Q4_G64_FP16, QType::Q6_G64_FP16},
          std::array{QType::NVFP4, QType::NVFP4},
          std::array{QType::FP8_E4M3FN_BLOCK128X128_BF16, QType::FP8_E4M3FN_BLOCK128X128_BF16}}) {
        Bank bank(types[0], types[1]);
        const std::array sources{bank.source(device.device())};
        auto requirements = describe_expert_cache(
            sources, {.capacity_bytes = 65536, .max_batch_bytes = 4096, .upload_slots = 2});
        // Force multiple slots and partial-expert transfers, with room for only one expert.
        requirements.config.capacity_bytes = requirements.banks[0].entry_bytes;
        ExpertCache cache(sources, requirements, device, device.transfer_view());
        for (const auto e : {5U, 1U, 4U, 0U}) {
            const std::array keys{ExpertKey{{0}, e}};
            replace(cache, requirements, keys);
            verify(cache, sources[0], keys[0], device);
        }
        const auto transferred = cache.usage().data_bytes;
        cache.prepare({std::array{ExpertKey{{0}, 0}}});
        verify(cache, sources[0], {{0}, 0}, device);
        require(cache.usage().data_bytes == transferred, "cache hit uploaded data");
        bool rejected = false;
        try {
            replace(cache, requirements, std::array{ExpertKey{{0}, 0}, ExpertKey{{0}, 1}});
        } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "required set larger than capacity was accepted");
    }
}

class StreamGate {
public:
    explicit StreamGate(cudaStream_t stream) : stream_(stream) {
        check(cudaLaunchHostFunc(stream_, wait, this));
    }

    ~StreamGate() {
        open();
        (void)cudaStreamSynchronize(stream_);
    }

    StreamGate(const StreamGate&)            = delete;
    StreamGate& operator=(const StreamGate&) = delete;

    void open() { open_.store(true, std::memory_order_release); }

private:
    static void CUDART_CB wait(void* data) {
        const auto& gate = *static_cast<const StreamGate*>(data);
        while (!gate.open_.load(std::memory_order_acquire)) { std::this_thread::yield(); }
    }

    cudaStream_t stream_;
    std::atomic<bool> open_{false};
};

void lifetimes_and_graph(const DeviceContext& device) {
    Bank bank(QType::NVFP4, QType::NVFP4);
    const std::array sources{bank.source(device.device())};
    auto req = describe_expert_cache(sources, {.capacity_bytes = 131072, .max_batch_bytes = 65536});
    req.config.capacity_bytes = req.banks[0].entry_bytes * 2;
    ExpertCache cache(sources, req, device, device.transfer_view());
    // Repeated keys must not reserve space or pin the same entry twice.
    const std::array first{ExpertKey{{0}, 0}, ExpertKey{{0}, 0}};
    replace(cache, req, first);
    // A use can be acquired after publication is queued, before the upload completes.
    StreamGate upload(device.transfer_stream());
    cache.progress();
    auto early = cache.acquire(first, device.execution_view());
    upload.open();
    require(early.has_value(), "acquire incorrectly waited for Host-observed DMA completion");
    cache.release(*early, device.execution_view());
    device.synchronize();

    const auto dimensions = sizes(sources[0]);
    DeviceBuffer output(device.device(), total(dimensions));
    DecodeGraphDefinition definition;
    definition.capture(device.stream(), [&] {
        gather<<<(output.size() + 255) / 256, 256, 0, device.stream()>>>(
            cache.view({0}), 0, dimensions, static_cast<unsigned char*>(output.data()));
    });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    const auto use = acquire(cache, first, device.execution_view());
    StreamGate consumer(device.stream());
    graph.launch(device.stream());
    cache.release(use, device.execution_view());
    const std::array next{ExpertKey{{0}, 1}, ExpertKey{{0}, 2}};
    replace(cache, req, next);
    cache.progress({.max_batches = 8});
    const auto blocked = cache.acquire(next, device.execution_view());
    consumer.open();
    require(blocked.has_value(), "released reader prevented ordered address reuse");
    cache.release(*blocked, device.execution_view());
    device.synchronize();
    std::vector<std::byte> bytes(output.size());
    output.copy_to_host(bytes.data(), bytes.size());
    require(bytes == expected(sources[0], 0), "index publication or reuse raced a Graph reader");
    verify(cache, sources[0], next[0], device);
    verify(cache, sources[0], next[1], device);
    replace(cache, req, first);
    const auto replay = acquire(cache, first, device.execution_view());
    graph.launch(device.stream());
    cache.release(replay, device.execution_view());
    device.synchronize();
    output.copy_to_host(bytes.data(), bytes.size());
    require(bytes == expected(sources[0], 0), "Graph retained an obsolete physical expert address");
}

void independent_readiness(const DeviceContext& device) {
    Bank bank(QType::NVFP4, QType::NVFP4);
    const std::array sources{bank.source(device.device())};
    auto req = describe_expert_cache(sources, {.capacity_bytes = 131072, .max_batch_bytes = 65536});
    ExpertCache cache(sources, req, device, device.transfer_view());
    const std::array current{ExpertKey{{0}, 0}};
    replace(cache, req, current);
    verify(cache, sources[0], current[0], device);
    StreamGate next_upload(device.transfer_stream());
    cache.replace({req.banks[0].entry_bytes, req.banks[0].entry_bytes, 256},
                  std::array{ExpertPlacement{{{0}, 1}, 0}});
    cache.prepare({current, std::array{ExpertKey{{0}, 1}}});
    cache.progress();
    const auto use = cache.acquire(current, device.execution_view());
    CudaCompletionEvent consumed(device);
    consumed.record(device.stream());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    bool complete       = false;
    while (!(complete = consumed.ready()) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    next_upload.open();
    require(use.has_value() && complete, "current bank waited for an unrelated prefetch");
    cache.release(*use, device.execution_view());
}

void variable_entries_and_prefetch(const DeviceContext& device) {
    Bank q5(QType::Q4_G64_FP16, QType::Q5_G64_FP16);
    Bank q6(QType::Q4_G64_FP16, QType::Q6_G64_FP16);
    const std::array sources{q5.source(device.device()), q6.source(device.device())};
    auto req = describe_expert_cache(sources, {.capacity_bytes = 262144, .max_batch_bytes = 4096});
    require(req.banks[0].entry_bytes != req.banks[1].entry_bytes,
            "different banks were padded to a maximum slot");
    req.config.capacity_bytes = 2 * req.banks[0].entry_bytes + req.banks[1].entry_bytes;
    ExpertCache cache(sources, req, device, device.transfer_view());
    std::array initial{ExpertKey{{0}, 0}, ExpertKey{{0}, 1}};
    const std::array placed{initial[0], initial[1], ExpertKey{{1}, 0}};
    cache.replace({0, req.config.capacity_bytes, 256}, packed(req, placed));
    cache.prepare({initial, std::array{ExpertKey{{1}, 0}}});
    for (const auto key : initial) { verify(cache, sources[0], key, device); }
    verify(cache, sources[1], {{1}, 0}, device);
    for (unsigned round = 0; round < 12; ++round) {
        const std::array keys{ExpertKey{{1}, round % 6}, ExpertKey{{0}, (round + 1) % 6}};
        replace(cache, req, keys);
        for (const auto key : keys) { verify(cache, sources[key.bank.index], key, device); }
    }
}

void resident_fp8(const DeviceContext& device) {
    Bank bank(QType::FP8_E4M3FN_BLOCK128X128_BF16, QType::FP8_E4M3FN_BLOCK128X128_BF16);
    bank.gu.upload(device.device());
    bank.down.upload(device.device());
    const std::array sources{bank.source(device.device())};
    auto req = describe_expert_cache(sources, {});
    require(req.entry_count == 0 && req.pinned_bytes == 0 && req.divisor_bytes == 0,
            "resident source allocated dynamic metadata");
    ExpertCache cache(sources, req, device, device.transfer_view());
    cache.wait_for_progress();
    verify(cache, sources[0], {{0}, 5}, device);
    ExpertCache other(sources, req, device, device.transfer_view());
    const std::array keys{ExpertKey{{0}, 0}};
    const auto first  = acquire(cache, keys, device.execution_view());
    const auto second = acquire(other, keys, device.execution_view());
    bool rejected     = false;
    try {
        other.release(first, device.execution_view());
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "use token crossed Program cache ownership");
    cache.release(first, device.execution_view());
    other.release(second, device.execution_view());
    require(cache.usage().data_bytes == 0 && cache.view({0}).gate_up.offsets == nullptr,
            "resident source was copied through cache");
}

void upload_budgets(const DeviceContext& device) {
    Bank bank(QType::NVFP4, QType::NVFP4);
    const std::array sources{bank.source(device.device())};
    auto req = describe_expert_cache(sources, {.capacity_bytes   = 262144,
                                               .max_batch_bytes  = 16384,
                                               .max_batch_copies = 1,
                                               .upload_slots     = 2});
    const std::array keys{ExpertKey{{0}, 0}, ExpertKey{{0}, 5}};
    ExpertCache cache(sources, req, device, device.transfer_view());
    replace(cache, req, keys);
    const auto none = cache.progress({.max_bytes = 0});
    require(!none.submitted_batches && !none.required_published, "zero byte budget submitted work");
    StreamGate blocked(device.transfer_stream());
    const auto first  = cache.progress({.max_bytes = 1001, .max_batches = 1});
    const auto second = cache.progress({.max_bytes = 2003, .max_batches = 1});
    const auto busy   = cache.progress();
    const auto early  = cache.acquire(keys, device.execution_view());
    blocked.open();
    require(first.submitted_bytes == 1001 && second.submitted_bytes == 2003 &&
                busy.waiting_for_uploads && !early,
            "upload budget or partial-expert publication is incorrect");
    for (const auto key : keys) { verify(cache, sources[0], key, device); }
    require(cache.usage().data_bytes == req.banks[0].payload_bytes * keys.size(),
            "partial transfers included entry padding or duplicated payload");
    check(cudaStreamBeginCapture(device.transfer_stream(), cudaStreamCaptureModeThreadLocal));
    bool rejected = false;
    try {
        cache.progress();
    } catch (const std::logic_error&) { rejected = true; }
    cudaGraph_t captured = nullptr;
    check(cudaStreamEndCapture(device.transfer_stream(), &captured));
    check(cudaGraphDestroy(captured));
    require(rejected, "cache handoff entered CUDA capture");
    verify(cache, sources[0], keys[0], device);
}

void complete_bank(const DeviceContext& device) {
    Bank bank(QType::Q4_G64_FP16, QType::Q6_G64_FP16, 96);
    const std::array sources{bank.source(device.device())};
    auto req                  = describe_expert_cache(sources, {});
    req.config.capacity_bytes = req.banks[0].entry_bytes * bank.count;
    ExpertCache cache(sources, req, device, device.transfer_view());
    std::vector<ExpertKey> keys;
    for (unsigned i = 0; i < bank.count; ++i) { keys.push_back({{0}, i}); }
    replace(cache, req, keys);
    for (const auto key : keys) { verify(cache, sources[0], key, device); }
    require(cache.usage().data_bytes == req.banks[0].payload_bytes * bank.count,
            "complete bank transferred a different payload");
}

void fixed_regions_and_windows(const DeviceContext& device) {
    Bank small(QType::Q4_G64_FP16, QType::Q5_G64_FP16, 32),
        large(QType::Q4_G64_FP16, QType::Q6_G64_FP16, 32);
    const std::array sources{small.source(device.device()), large.source(device.device()),
                             small.source(device.device()), large.source(device.device())};
    auto req = describe_expert_cache(sources, {});
    const std::array fixed{ExpertKey{{0}, 0}, ExpertKey{{1}, 0}, ExpertKey{{2}, 0},
                           ExpertKey{{3}, 0}};
    const auto fixed_layout   = packed(req, fixed);
    req.config.fixed_bytes    = fixed_layout.back().offset + req.banks[3].entry_bytes;
    const auto window_bytes   = 31 * req.banks[1].entry_bytes;
    req.config.capacity_bytes = req.config.fixed_bytes + 2 * window_bytes;
    const LayoutRegion extra{req.config.fixed_bytes, 2 * window_bytes, 256};
    ExpertCache cache(sources, req, device, device.transfer_view(), fixed_layout);
    cache.prepare({fixed});
    for (const auto key : fixed) verify(cache, sources[key.bank.index], key, device);
    const auto fixed_use = acquire(cache, fixed, device.execution_view());
    std::vector<ExpertKey> extras;
    std::size_t extra_bytes = 0, extra_payload = 0;
    for (std::uint32_t e = 1; e < 32; ++e) {
        for (std::uint32_t b = 0; b < 4; ++b) {
            if (req.banks[b].entry_bytes > extra.bytes - extra_bytes) continue;
            extras.push_back({{b}, e});
            extra_bytes += req.banks[b].entry_bytes;
            extra_payload += req.banks[b].payload_bytes;
        }
    }
    const auto check_fixed = [&] {
        device.synchronize();
        for (const auto& placement : fixed_layout) {
            std::uint64_t actual = UINT64_MAX;
            check(cudaMemcpy(&actual, cache.view(placement.key.bank).gate_up.offsets,
                             sizeof(actual), cudaMemcpyDeviceToHost));
            require(actual == placement.offset, "window reuse changed a fixed expert address");
        }
    };
    for (int round = 0; round < 3; ++round) {
        const auto before = cache.uploaded_bytes();
        cache.replace(extra, packed(req, extras));
        cache.prepare({extras});
        for (const auto key : extras) verify(cache, sources[key.bank.index], key, device);
        require(cache.uploaded_bytes() - before == extra_payload,
                "small-T reconstruction transferred more than its target payload");
        check_fixed();
        cache.replace(extra, {});
        for (std::uint32_t b = 0; b < 4; ++b) {
            std::vector<ExpertKey> missing, all;
            for (std::uint32_t e = 0; e < 32; ++e) {
                all.push_back({{b}, e});
                if (e) missing.push_back({{b}, e});
            }
            const auto before_layer = cache.uploaded_bytes();
            cache.replace({extra.offset + (b % 2) * window_bytes, window_bytes, 256},
                          packed(req, missing));
            cache.prepare({all});
            for (const auto key : all) verify(cache, sources[b], key, device);
            require(cache.uploaded_bytes() - before_layer == 31 * req.banks[b].payload_bytes,
                    "streaming recopied a fixed expert");
            check_fixed();
        }
    }
    bool rejected = false;
    try {
        cache.replace({0, req.config.fixed_bytes, 256}, {});
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "replacement admitted a write to the fixed region");
    cache.release(fixed_use, device.execution_view());

    // An unfinished small-T restore is retired before either window receives a new layout.
    cache.replace(extra, packed(req, extras));
    cache.prepare({{}, extras});
    StreamGate producer(device.transfer_stream());
    const auto partial = cache.progress({1001, 1});
    const std::array new_keys{ExpertKey{{3}, 31}};
    cache.replace(extra, packed(req, new_keys));
    cache.prepare({new_keys});
    cache.progress();
    producer.open();
    verify(cache, sources[3], new_keys[0], device);
    require(partial.submitted_bytes == 1001, "restore fixture did not create a partial upload");
    for (const auto key : fixed) verify(cache, sources[key.bank.index], key, device);
}

void promote_partial_upload(const DeviceContext& device) {
    Bank bank(QType::NVFP4, QType::NVFP4);
    const std::array sources{bank.source(device.device())};
    auto req = describe_expert_cache(
        sources, {.capacity_bytes = 65536, .max_batch_bytes = 4096, .upload_slots = 2});
    req.config.capacity_bytes = req.banks[0].entry_bytes;
    ExpertCache cache(sources, req, device, device.transfer_view());
    const std::array keys{ExpertKey{{0}, 4}};
    cache.replace({0, req.config.capacity_bytes, 256}, packed(req, keys));
    cache.prepare({{}, keys});
    StreamGate producer(device.transfer_stream());
    const auto first = cache.progress({1001, 1});
    std::vector<ExpertKey> ready;
    cache.ready_keys({0}, ready);
    require(first.submitted_bytes == 1001 && ready.empty(),
            "partial payload became a ready replica");
    cache.prepare({keys});
    const auto second = cache.progress({1003, 1});
    producer.open();
    require(second.submitted_bytes == 1003, "promoted upload did not continue in place");
    verify(cache, sources[0], keys[0], device);
    require(cache.uploaded_bytes() == req.banks[0].payload_bytes,
            "promoting prefetch duplicated or relocated its payload");
}

} // namespace

int main() {
    try {
        DeviceContext device;
        codecs_and_chunks(device);
        lifetimes_and_graph(device);
        independent_readiness(device);
        variable_entries_and_prefetch(device);
        resident_fp8(device);
        upload_budgets(device);
        complete_bank(device);
        fixed_regions_and_windows(device);
        promote_partial_upload(device);
        Bank failure_source(QType::NVFP4, QType::NVFP4);
        ninfer::test::expert_cache_cuda_errors(failure_source.source(device.device()), device);
        std::cout << "expert cache bytes, lifetimes, Graph and resident access passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
