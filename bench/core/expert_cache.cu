#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/expert_cache.h"
#include "core/transfer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>

namespace {
using namespace ninfer;
using Clock = std::chrono::steady_clock;

void check(cudaError_t error) {
    if (error != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(error)); }
}

double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

__global__ void consume(const unsigned char* codes, const std::uint64_t* offsets, std::size_t bytes,
                        unsigned long long* output) {
    if (offsets) { codes += offsets[0]; }
    unsigned long long sum = 0;
    for (auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < bytes;
         i += gridDim.x * blockDim.x) {
        sum += codes[i];
    }
    atomicAdd(output, sum);
}

struct Measurement {
    double wall_ms, plan_ms, submit_ms, gpu_ms;
    ExpertCacheUsage delta;
};

ExpertCacheUse acquire(ExpertCache& cache, std::span<const ExpertKey> keys,
                       CudaExecutionView execution) {
    const auto deadline = Clock::now() + std::chrono::seconds(30);
    do {
        cache.progress();
        if (auto use = cache.acquire(keys, execution)) { return *use; }
        std::this_thread::yield();
    } while (Clock::now() < deadline);
    throw std::runtime_error("cache did not make progress");
}

void place(ExpertCache& cache, const ExpertCacheRequirements& req, std::span<const ExpertKey> keys,
           std::size_t begin = 0) {
    std::vector<ExpertPlacement> entries;
    std::size_t offset = 0;
    for (const auto key : keys) {
        entries.push_back({key, offset});
        offset += req.banks[key.bank.index].entry_bytes;
    }
    cache.replace({begin, req.config.capacity_bytes - begin, 256}, entries);
    cache.prepare({keys});
}

void verify(ExpertCache& cache, const ExpertBankSource& source, ExpertKey key,
            const DeviceContext& device) {
    const std::array keys{key};
    const auto use = acquire(cache, keys, device.execution_view());
    device.synchronize();
    const auto& view     = cache.view(key.bank);
    std::uint64_t offset = 0;
    if (view.gate_up.offsets) {
        check(cudaMemcpy(&offset, view.gate_up.offsets + key.expert, 8, cudaMemcpyDeviceToHost));
    }
    for (const auto gate : {true, false}) {
        const auto& bank   = gate ? source.gate_up : source.down;
        const auto& matrix = gate ? view.gate_up : view.down;
        const std::array planes{bank.layout.codes, bank.layout.high, bank.layout.scales};
        const std::array pointers{matrix.codes, matrix.high, matrix.scales};
        const std::array strides{matrix.code_stride, matrix.high_stride, matrix.scale_stride};
        for (std::size_t p = 0; p < planes.size(); ++p) {
            const auto from = bank.plane(planes[p], key.expert);
            if (!from.bytes) { continue; }
            std::vector<std::byte> expected(from.bytes), actual(from.bytes);
            if (from.device == Device::cpu()) {
                std::memcpy(expected.data(), from.data, from.bytes);
            } else {
                check(cudaMemcpy(expected.data(), from.data, from.bytes, cudaMemcpyDeviceToHost));
            }
            const auto local = matrix.offsets ? offset : key.expert * strides[p];
            check(
                cudaMemcpy(actual.data(), pointers[p] + local, from.bytes, cudaMemcpyDeviceToHost));
            if (actual != expected) { throw std::runtime_error("real expert plane bytes differ"); }
        }
        if (matrix.divisors) {
            float divisor;
            check(cudaMemcpy(&divisor, matrix.divisors + key.expert * matrix.divisor_stride, 4,
                             cudaMemcpyDeviceToHost));
            if (divisor != bank.divisor(key.expert)) {
                throw std::runtime_error("real expert divisor differs");
            }
        }
    }
    cache.release(use, device.execution_view());
    device.synchronize();
}

void print(const char* label, std::vector<Measurement> values) {
    std::sort(values.begin(), values.end(),
              [](const auto& a, const auto& b) { return a.wall_ms < b.wall_ms; });
    const auto& median = values[values.size() / 2];
    std::cout << label << " wall_ms[min,median,max]=" << values.front().wall_ms << ','
              << median.wall_ms << ',' << values.back().wall_ms << " plan_ms=" << median.plan_ms
              << " submit_ms=" << median.submit_ms << " upload_timeline_ms=" << median.gpu_ms
              << " bytes=" << median.delta.data_bytes << " index_bytes=" << median.delta.index_bytes
              << " copy_calls=" << (median.delta.data_copy_calls + median.delta.index_copy_calls)
              << " GBps=" << median.delta.data_bytes / median.wall_ms / 1e6 << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string path, prefix;
        std::uint32_t count     = 0;
        bool resident           = false;
        std::size_t batch_bytes = 96 * 1024 * 1024;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            const auto value      = [&]() {
                if (++i >= argc) { throw std::invalid_argument("missing argument value"); }
                return std::string(argv[i]);
            };
            if (arg == "--artifact") {
                path = value();
            } else if (arg == "--bank") {
                prefix = value();
            } else if (arg == "--experts") {
                count = std::stoul(value());
            } else if (arg == "--batch-mib") {
                batch_bytes = std::stoul(value()) * 1024 * 1024;
            } else if (arg == "--resident") {
                resident = true;
            } else {
                throw std::invalid_argument("--artifact PATH --bank LOGICAL_EXPERT_PREFIX "
                                            "--experts E [--resident] [--batch-mib N]");
            }
        }
        if (path.empty() || prefix.empty() || !count) {
            throw std::invalid_argument("artifact, bank prefix and expert count are required");
        }
        DeviceContext device;
        artifact::Reader reader(path);
        const auto object = [&](const char* role) {
            auto name = prefix + '/' + role;
            if (!reader.directory().bindings.contains(name)) { name = prefix + "/0/" + role; }
            return reader.directory().bindings.at(name).parts.at(0).object;
        };
        const auto gu = object("gate"), down = object("down");
        const auto location = resident ? device.device() : Device::cpu();
        const auto access =
            resident ? artifact::HostAccess::CpuOnly : artifact::HostAccess::CudaTransfer;
        const std::array requests{artifact::PlacementRequest{gu, location, access},
                                  artifact::PlacementRequest{down, location, access}};
        const auto load_start = Clock::now();
        auto backing          = artifact::materialize(
            reader, artifact::plan_materialization(reader, requests), device.transfer_view());
        const auto source_load_ms = milliseconds(load_start);
        const auto bind           = [&](artifact::ObjectHandle handle) {
            const auto& parent =
                resident ? backing.device_parent(handle) : backing.host_parent(handle);
            const auto& g   = parent.geometry;
            const auto rows = g.elements / g.shape.back();
            if (rows % count) {
                throw std::invalid_argument("expert count does not divide parent rows");
            }
            return bind_weight_bank(
                parent, describe_weight_bank(g, {0, count, rows / count, rows / count}));
        };
        const ExpertBankSource source{bind(gu), bind(down),
                                      resident ? CudaHostView{} : backing.host_transfer_view(gu),
                                      resident ? CudaHostView{} : backing.host_transfer_view(down)};
        // Two logical banks share immutable source bytes, exercising current/next bank handoff.
        const std::array sources{source, source};
        auto req              = describe_expert_cache(sources, {.max_batch_bytes = batch_bytes});
        const auto bank_bytes = req.banks[0].entry_bytes * count;
        req.config.capacity_bytes = resident ? 0 : bank_bytes * 2;
        const auto setup_start    = Clock::now();
        ExpertCache cache(sources, req, device, device.transfer_view());
        std::cout << std::fixed << std::setprecision(3) << device.properties().name
                  << " runtime=" << CUDART_VERSION << " source_load_ms=" << source_load_ms
                  << " registered_host_bytes=" << backing.stats().registered_host_bytes
                  << " registration_ms=" << backing.stats().host_registration_seconds * 1000
                  << " cache_setup_ms=" << milliseconds(setup_start) << " bank_bytes=" << bank_bytes
                  << " expert_bytes=" << req.banks[0].entry_bytes
                  << " cache_bytes=" << req.config.capacity_bytes
                  << " pinned_bytes=" << req.pinned_bytes << " offsets=" << req.offset_bytes
                  << " static_divisors=" << req.divisor_bytes << '\n';
        if (resident) {
            verify(cache, source, {{0}, count - 1}, device);
            std::cout << "resident encoded planes and divisors passed; dynamic_bytes="
                      << cache.usage().data_bytes << '\n';
            return 0;
        }
        const auto roof_bytes =
            std::min<std::size_t>(64 * 1024 * 1024, source.gate_up_upload.host.bytes);
        const auto roof_source = source.gate_up_upload.subspan(0, roof_bytes);
        DeviceBuffer transfer_target(device.device(), roof_bytes);
        CudaEventTimer h2d(device, device.transfer_stream());
        const auto warmup_transfers = std::max<std::size_t>(2, 128 * 1024 * 1024 / roof_bytes);
        for (std::size_t i = 0; i < warmup_transfers; ++i) {
            copy_bytes_async(transfer_target.span(), roof_source.host, device.transfer_view());
        }
        check(cudaStreamSynchronize(device.transfer_stream()));
        const auto roof_transfers = std::max<std::size_t>(4, 1024 * 1024 * 1024 / roof_bytes);
        h2d.start();
        for (std::size_t i = 0; i < roof_transfers; ++i) {
            copy_bytes_async(transfer_target.span(), roof_source.host, device.transfer_view());
        }
        const auto h2d_ms = h2d.stop_ms();
        const auto dma_bw = roof_transfers * roof_bytes / h2d_ms / 1e6;
        std::cout << "roofline registered_H2D_GBps=" << dma_bw
                  << " full_bank_floor_ms=" << req.banks[0].payload_bytes * count / dma_bw / 1e6
                  << '\n';
        std::vector<unsigned> shuffled(count);
        std::iota(shuffled.begin(), shuffled.end(), 0);
        std::mt19937 rng(42);
        std::shuffle(shuffled.begin(), shuffled.end(), rng);
        std::vector<ExpertKey> full;
        for (unsigned e = 0; e < count; ++e) { full.push_back({{0}, e}); }
        const auto one = std::array{ExpertKey{{0}, 0}};
        place(cache, req, one);
        verify(cache, source, one[0], device);
        DeviceBuffer warm_sum(device.device(), sizeof(unsigned long long));
        warm_sum.fill();
        consume<<<1, 256, 0, device.stream()>>>(
            static_cast<const unsigned char*>(transfer_target.data()), nullptr,
            transfer_target.size(), static_cast<unsigned long long*>(warm_sum.data()));
        device.synchronize();
        std::vector<double> hits;
        for (int i = 0; i < 101; ++i) {
            device.synchronize();
            const auto start = Clock::now();
            cache.prepare({one});
            const auto use = cache.acquire(one, device.execution_view());
            if (!use) { throw std::runtime_error("resident hit unavailable"); }
            const auto admission_us = milliseconds(start) * 1000;
            const auto& view        = cache.view({0}).gate_up;
            consume<<<32, 128, 0, device.stream()>>>(
                reinterpret_cast<const unsigned char*>(view.codes), view.offsets,
                source.gate_up.layout.codes.expert_bytes,
                static_cast<unsigned long long*>(warm_sum.data()));
            const auto release_start = Clock::now();
            cache.release(*use, device.execution_view());
            hits.push_back(admission_us + milliseconds(release_start) * 1000);
        }
        std::sort(hits.begin(), hits.end());
        std::cout << "all_hit_active_host_us[min,median,max]=" << hits.front() << ','
                  << hits[hits.size() / 2] << ',' << hits.back() << '\n';
        const auto measure = [&](std::size_t n, bool overlap, bool scattered = false) {
            std::vector<Measurement> samples;
            for (int repeat = 0; repeat < 5; ++repeat) {
                // A new cache gives an explicit cold replica set; source pages are warm RAM.
                ExpertCache instance(sources, req, device, device.transfer_view());
                std::vector<ExpertKey> keys;
                for (std::size_t e = 0; e < n; ++e) {
                    keys.push_back({{0}, scattered ? shuffled[e] : static_cast<unsigned>(e)});
                }
                DeviceBuffer sum(device.device(), sizeof(unsigned long long));
                std::optional<ExpertCacheUse> previous;
                std::uint64_t previous_offset = 0;
                sum.fill();
                if (overlap) {
                    place(instance, req, full);
                    previous = acquire(instance, full, device.execution_view());
                    device.synchronize();
                    check(cudaMemcpy(&previous_offset, instance.view({0}).gate_up.offsets, 8,
                                     cudaMemcpyDeviceToHost));
                    for (auto& key : keys) { key.bank = {1}; }
                }
                const auto before = instance.usage();
                CudaEventTimer timer(device, device.transfer_stream());
                timer.start();
                const auto start = Clock::now();
                if (previous) {
                    // Synthetic read consumer: four passes over the encoded bank data.
                    // Measures transfer/consumer overlap, not MoE mathematical throughput.
                    for (int pass = 0; pass < 4; ++pass) {
                        const auto& view = instance.view({0});
                        consume<<<512, 256, 0, device.stream()>>>(
                            reinterpret_cast<const unsigned char*>(view.gate_up.codes) +
                                previous_offset,
                            nullptr,
                            req.banks[0].entry_bytes * count - req.banks[0].gate_up.codes.offset,
                            static_cast<unsigned long long*>(sum.data()));
                    }
                    instance.release(*previous, device.execution_view());
                }
                place(instance, req, keys, overlap ? bank_bytes : 0);
                const auto use = acquire(instance, keys, device.execution_view());
                instance.release(use, device.execution_view());
                device.synchronize();
                const auto wall = milliseconds(start);
                const auto gpu  = timer.stop_ms();
                auto delta      = instance.usage();
                delta.data_bytes -= before.data_bytes;
                delta.index_bytes -= before.index_bytes;
                delta.data_copy_calls -= before.data_copy_calls;
                delta.index_copy_calls -= before.index_copy_calls;
                delta.planning_nanoseconds -= before.planning_nanoseconds;
                delta.submit_nanoseconds -= before.submit_nanoseconds;
                samples.push_back({wall, delta.planning_nanoseconds / 1e6,
                                   delta.submit_nanoseconds / 1e6, gpu, delta});
                if (!overlap) { verify(instance, source, keys.back(), device); }
            }
            return samples;
        };
        print("one_miss", measure(1, false));
        print("eight_misses", measure(std::min<std::uint32_t>(8, count), false));
        print("eight_scattered", measure(std::min<std::uint32_t>(8, count), false, true));
        print("thirty_two_scattered", measure(std::min<std::uint32_t>(32, count), false, true));
        print("one_twenty_eight_scattered",
              measure(std::min<std::uint32_t>(128, count), false, true));
        print("full_bank", measure(count, false));
        print("consumer_plus_next_bank", measure(count, true));
        // Explicit region reconstruction, including reuse after a released consumer.
        auto transition_req                  = req;
        transition_req.config.capacity_bytes = bank_bytes;
        ExpertCache transitions(sources, transition_req, device, device.transfer_view());
        const auto report_change = [&](const char* label, const std::vector<ExpertKey>& keys) {
            const auto before = transitions.usage();
            const auto start  = Clock::now();
            place(transitions, transition_req, keys);
            const auto use = acquire(transitions, keys, device.execution_view());
            transitions.release(use, device.execution_view());
            device.synchronize();
            const auto after = transitions.usage();
            std::cout << label << " ms=" << milliseconds(start)
                      << " bytes=" << after.data_bytes - before.data_bytes
                      << " used=" << after.used_bytes << '\n';
        };
        std::vector<ExpertKey> fixed(full.begin(),
                                     full.begin() + std::min<std::size_t>(8, full.size()));
        report_change("fixed", fixed);
        report_change("fixed_to_full", full);
        for (auto& key : fixed) { key.bank = {1}; }
        report_change("full_to_next_fixed", fixed);
        for (auto& key : full) { key.bank = {1}; }
        report_change("next_fixed_to_full", full);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
