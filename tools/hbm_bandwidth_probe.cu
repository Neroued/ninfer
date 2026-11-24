// Sustained device-memory bandwidth probe.
//
// Built as bench/ninfer_memory_bandwidth_bench with the selected NINFER_TARGET.
//
// The "bus GB/s" column counts logical streaming traffic: N bytes for a
// read or write and 2N bytes for a copy (N read + N written). This is the
// rate to compare with the advertised bandwidth; it is not a measured DRAM counter.

#include "common/fixture_data.cuh"
#include "common/roofline.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#define CUDA_CHECK(expr)                                                                           \
    do {                                                                                           \
        const cudaError_t error__ = (expr);                                                        \
        if (error__ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                  \
                         cudaGetErrorString(error__));                                             \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (false)

namespace {

constexpr int kThreads                 = 256;
constexpr double kDefaultTrialSeconds  = 0.25;
constexpr int kDefaultTrials           = 5;
constexpr std::size_t kDefaultMaxBytes = std::size_t{4} << 30;

struct Options {
    double size_gib  = 0.0; // 0 selects min(4 GiB, 20% of currently free memory).
    double seconds   = kDefaultTrialSeconds;
    int trials       = kDefaultTrials;
    double peak_gbps = ninfer::bench::kRoofline.memory_gbs;
};

struct Result {
    const char* name;
    double payload_gbps;
    double bus_gbps;
    double median_bus_gbps;
    double peak_fraction;
    double ns;
    int iterations;
};

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&data_, count_ * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) { cudaFree(data_); }
    }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return data_; }

    const T* get() const { return data_; }

    std::size_t size() const { return count_; }

private:
    T* data_           = nullptr;
    std::size_t count_ = 0;
};

class Event {
public:
    Event() { CUDA_CHECK(cudaEventCreate(&event_)); }

    ~Event() { cudaEventDestroy(event_); }

    Event(const Event&)            = delete;
    Event& operator=(const Event&) = delete;

    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_{};
};

__global__ void write_u128_kernel(uint4* __restrict__ dst, std::size_t count, uint4 value) {
    const std::size_t tid    = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = tid; i < count; i += stride) {
        const auto offset = static_cast<std::uint32_t>(i);
        dst[i] = make_uint4(value.x ^ (offset * 0x9e3779b9U), value.y ^ (offset * 0x85ebca6bU),
                            value.z ^ (offset * 0xc2b2ae35U), value.w ^ (offset * 0x27d4eb2fU));
    }
}

// The block checksum makes every load observable while adding only one tiny
// output store per block. Four independent lanes avoid a single long integer
// dependency chain in each thread.
__global__ void read_u128_kernel(const uint4* __restrict__ src, uint4* __restrict__ block_checksums,
                                 std::size_t count) {
    __shared__ uint4 sums[kThreads];

    const std::size_t tid    = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    uint4 sum                = make_uint4(0u, 0u, 0u, 0u);

    for (std::size_t i = tid; i < count; i += stride) {
        const uint4 value = src[i];
        sum.x += value.x;
        sum.y += value.y;
        sum.z += value.z;
        sum.w += value.w;
    }

    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset != 0; offset /= 2) {
        if (threadIdx.x < offset) {
            const uint4 other = sums[threadIdx.x + offset];
            sums[threadIdx.x].x += other.x;
            sums[threadIdx.x].y += other.y;
            sums[threadIdx.x].z += other.z;
            sums[threadIdx.x].w += other.w;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { block_checksums[blockIdx.x] = sums[0]; }
}

__global__ void copy_u128_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst,
                                 std::size_t count) {
    const std::size_t tid    = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = tid; i < count; i += stride) { dst[i] = src[i]; }
}

// Expose four independent loads before their stores. On some GPUs this raises
// the per-warp memory-level parallelism of a mixed read/write stream.
__global__ void copy_u128x4_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst,
                                   std::size_t count) {
    const std::size_t tid    = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    std::size_t i            = tid;
    for (; i + 3 * stride < count; i += 4 * stride) {
        const uint4 value0  = src[i];
        const uint4 value1  = src[i + stride];
        const uint4 value2  = src[i + 2 * stride];
        const uint4 value3  = src[i + 3 * stride];
        dst[i]              = value0;
        dst[i + stride]     = value1;
        dst[i + 2 * stride] = value2;
        dst[i + 3 * stride] = value3;
    }
    for (; i < count; i += stride) { dst[i] = src[i]; }
}

double parse_double(const char* text, const char* option) {
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(value) || value <= 0.0) {
        std::fprintf(stderr, "Invalid value for %s: %s\n", option, text);
        std::exit(EXIT_FAILURE);
    }
    return value;
}

int parse_int(const char* text, const char* option) {
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0 || value > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "Invalid value for %s: %s\n", option, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

void print_help(const char* argv0) {
    std::printf("Usage: %s [options]\n"
                "  --size-gib N    bytes per large buffer (default: min(4, 20%% free))\n"
                "  --seconds N     target duration of each timed trial (default: %.2f)\n"
                "  --trials N      timed trials per method (default: %d)\n"
                "  --peak-gbps N   advertised bus bandwidth (default: %.0f)\n"
                "  --help          show this text\n",
                argv0, kDefaultTrialSeconds, kDefaultTrials, ninfer::bench::kRoofline.memory_gbs);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "--help") == 0) {
            print_help(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (i + 1 >= argc) {
            std::fprintf(stderr, "Missing value after %s\n", arg);
            std::exit(EXIT_FAILURE);
        }
        if (std::strcmp(arg, "--size-gib") == 0) {
            options.size_gib = parse_double(argv[++i], arg);
        } else if (std::strcmp(arg, "--seconds") == 0) {
            options.seconds = parse_double(argv[++i], arg);
        } else if (std::strcmp(arg, "--trials") == 0) {
            options.trials = parse_int(argv[++i], arg);
        } else if (std::strcmp(arg, "--peak-gbps") == 0) {
            options.peak_gbps = parse_double(argv[++i], arg);
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", arg);
            print_help(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }
    return options;
}

template <typename Launch>
double elapsed_ms(cudaStream_t stream, int iterations, Launch&& launch) {
    Event start;
    Event stop;
    CUDA_CHECK(cudaEventRecord(start.get(), stream));
    for (int i = 0; i < iterations; ++i) { launch(); }
    CUDA_CHECK(cudaEventRecord(stop.get(), stream));
    CUDA_CHECK(cudaEventSynchronize(stop.get()));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()));
    return static_cast<double>(milliseconds);
}

template <typename Launch>
Result benchmark(const char* name, std::size_t payload_bytes, double traffic_multiplier,
                 double peak_gbps, double trial_seconds, int trials, cudaStream_t stream,
                 Launch&& launch) {
    // A few untimed sweeps settle clocks and page mappings before calibration.
    for (int i = 0; i < 8; ++i) { launch(); }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const double calibration_ms = elapsed_ms(stream, 2, launch) / 2.0;
    const int iterations =
        std::max(1, static_cast<int>(std::ceil(trial_seconds * 1000.0 / calibration_ms)));

    std::vector<double> trial_ns;
    trial_ns.reserve(trials);
    for (int trial = 0; trial < trials; ++trial) {
        const double milliseconds = elapsed_ms(stream, iterations, launch);
        trial_ns.push_back(milliseconds * 1.0e6 / iterations);
    }
    CUDA_CHECK(cudaGetLastError());

    std::sort(trial_ns.begin(), trial_ns.end());
    const double best_ns      = trial_ns.front();
    const double median_ns    = trial_ns[trial_ns.size() / 2];
    const double payload_gbps = static_cast<double>(payload_bytes) / best_ns;
    const double bus_gbps     = payload_gbps * traffic_multiplier;
    const double median_bus_gbps =
        static_cast<double>(payload_bytes) / median_ns * traffic_multiplier;
    return Result{name,    payload_gbps, bus_gbps, median_bus_gbps, bus_gbps / peak_gbps,
                  best_ns, iterations};
}

template <typename Kernel>
int resident_grid(Kernel kernel, int sm_count) {
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, kThreads, 0));
    return std::max(sm_count, sm_count * blocks_per_sm);
}

void print_result(const Result& result) {
    std::printf("%-28s %12.1f %12.1f %12.1f %9.1f%% %10.3f\n", result.name, result.payload_gbps,
                result.bus_gbps, result.median_bus_gbps, result.peak_fraction * 100.0,
                result.ns / 1.0e6);
}

} // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    std::size_t bytes =
        options.size_gib > 0.0
            ? static_cast<std::size_t>(options.size_gib * static_cast<double>(std::size_t{1} << 30))
            : std::min(kDefaultMaxBytes, free_bytes / 5);
    bytes = bytes / sizeof(uint4) * sizeof(uint4);
    if (bytes < static_cast<std::size_t>(prop.l2CacheSize) * 4) {
        std::fprintf(stderr,
                     "Working set must be at least 4x L2 (%d MiB); requested %.1f "
                     "MiB\n",
                     prop.l2CacheSize >> 20, static_cast<double>(bytes) / (1 << 20));
        return EXIT_FAILURE;
    }
    if (bytes > free_bytes / 3) {
        std::fprintf(stderr,
                     "Two buffers plus headroom need <= 2/3 of free VRAM; "
                     "requested %.2f GiB each with %.2f GiB free\n",
                     static_cast<double>(bytes) / (std::size_t{1} << 30),
                     static_cast<double>(free_bytes) / (std::size_t{1} << 30));
        return EXIT_FAILURE;
    }

    const std::size_t vector_count = bytes / sizeof(uint4);
    const int write_grid           = resident_grid(write_u128_kernel, prop.multiProcessorCount);
    const int read_grid            = resident_grid(read_u128_kernel, prop.multiProcessorCount);
    const int copy_grid            = resident_grid(copy_u128_kernel, prop.multiProcessorCount);
    const int copy_x4_grid         = resident_grid(copy_u128x4_kernel, prop.multiProcessorCount);

    std::printf("GPU:               %s\n", prop.name);
    std::printf("Target:            %s\n", ninfer::bench::kRoofline.target);
    std::printf("SMs / L2:          %d / %.1f MiB\n", prop.multiProcessorCount,
                static_cast<double>(prop.l2CacheSize) / (1 << 20));
    std::printf("Memory bus width:  %d bits\n", prop.memoryBusWidth);
    std::printf("VRAM total/free:   %.2f / %.2f GiB\n",
                static_cast<double>(total_bytes) / (std::size_t{1} << 30),
                static_cast<double>(free_bytes) / (std::size_t{1} << 30));
    std::printf("Working set:       %.2f GiB per buffer (%.1fx L2)\n",
                static_cast<double>(bytes) / (std::size_t{1} << 30),
                static_cast<double>(bytes) / prop.l2CacheSize);
    std::printf("Advertised peak:   %.1f GB/s\n", options.peak_gbps);
    std::printf("Trial policy:      %d trials, >= %.2f s each, best + median\n", options.trials,
                options.seconds);
    std::printf("Resident grids:    write=%d read=%d copy=%d copy-x4=%d blocks x %d threads\n\n",
                write_grid, read_grid, copy_grid, copy_x4_grid, kThreads);

    DeviceBuffer<uint4> source(vector_count);
    DeviceBuffer<uint4> destination(vector_count);
    DeviceBuffer<uint4> checksums(read_grid);

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(ninfer::bench::fixture::fill_bytes(source.get(), bytes, 101U, stream));

    // Long preconditioning phase: make the first reported method independent
    // of idle P-state ramp-up.
    for (int i = 0; i < 64; ++i) {
        copy_u128_kernel<<<copy_grid, kThreads, 0, stream>>>(source.get(), destination.get(),
                                                             vector_count);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<Result> results;
    results.reserve(6);

    results.push_back(benchmark(
        "cudaMemsetAsync (write)", bytes, 1.0, options.peak_gbps, options.seconds, options.trials,
        stream, [&] { CUDA_CHECK(cudaMemsetAsync(destination.get(), 0xa5, bytes, stream)); }));

    const uint4 write_value = make_uint4(0x55aa55aau, 0xaa55aa55u, 0x31415926u, 0x27182818u);
    results.push_back(benchmark("kernel uint4 write", bytes, 1.0, options.peak_gbps,
                                options.seconds, options.trials, stream, [&] {
                                    write_u128_kernel<<<write_grid, kThreads, 0, stream>>>(
                                        destination.get(), vector_count, write_value);
                                }));

    results.push_back(benchmark("kernel uint4 read", bytes, 1.0, options.peak_gbps, options.seconds,
                                options.trials, stream, [&] {
                                    read_u128_kernel<<<read_grid, kThreads, 0, stream>>>(
                                        source.get(), checksums.get(), vector_count);
                                }));

    results.push_back(benchmark("kernel uint4 copy", bytes, 2.0, options.peak_gbps, options.seconds,
                                options.trials, stream, [&] {
                                    copy_u128_kernel<<<copy_grid, kThreads, 0, stream>>>(
                                        source.get(), destination.get(), vector_count);
                                }));

    results.push_back(benchmark("kernel uint4x4 copy", bytes, 2.0, options.peak_gbps,
                                options.seconds, options.trials, stream, [&] {
                                    copy_u128x4_kernel<<<copy_x4_grid, kThreads, 0, stream>>>(
                                        source.get(), destination.get(), vector_count);
                                }));

    results.push_back(benchmark("cudaMemcpyAsync D2D", bytes, 2.0, options.peak_gbps,
                                options.seconds, options.trials, stream, [&] {
                                    CUDA_CHECK(cudaMemcpyAsync(destination.get(), source.get(),
                                                               bytes, cudaMemcpyDeviceToDevice,
                                                               stream));
                                }));

    uint4 checksum{};
    CUDA_CHECK(cudaMemcpyAsync(&checksum, checksums.get(), sizeof(checksum), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("%-28s %12s %12s %12s %10s %10s\n", "method", "payload", "best bus", "median bus",
                "of peak", "best ms");
    std::printf("%-28s %12s %12s %12s %10s %10s\n", "", "GB/s", "GB/s", "GB/s", "", "");
    for (const Result& result : results) { print_result(result); }

    const auto best =
        std::max_element(results.begin(), results.end(),
                         [](const Result& a, const Result& b) { return a.bus_gbps < b.bus_gbps; });
    std::printf("\nBest sustained bus rate: %.1f GB/s (%.1f%% of %.1f GB/s), %s\n", best->bus_gbps,
                best->peak_fraction * 100.0, options.peak_gbps, best->name);
    std::printf("Read checksum witness:   %08x:%08x:%08x:%08x\n", checksum.x, checksum.y,
                checksum.z, checksum.w);
    std::printf("Note: copy payload is half its bus rate because every copied byte is "
                "both read and written.\n");

    CUDA_CHECK(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}
