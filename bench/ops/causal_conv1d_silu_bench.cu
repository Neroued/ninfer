// Performance bench for the BF16 depthwise causal width-4 convolution + SiLU Op.
//
// Examples:
//   ./ninfer_causal_conv1d_silu_bench --decode --channels 8192
//   ./ninfer_causal_conv1d_silu_bench --prefill --channels 8192 --tokens 1024
//   ./ninfer_causal_conv1d_silu_bench --distinct --channels 8192 --tokens 6
//   ./ninfer_causal_conv1d_silu_bench --snapshot --channels 8192 --tokens 6 --slots 7
// Printed logical GB/s is informational; NCU determines the applicable resource roofline.
#include "ninfer/ops/causal_conv1d_silu.h"
#include "core/buffer.h"
#include "core/device.h"
#include "op_measurement.cuh"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <optional>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::size_t kFlushBytes = std::size_t{256} << 20;

struct Options {
    measurement::Options timing;
    std::int32_t channels     = 8192;
    std::int32_t tokens       = 1024;
    std::int32_t slots        = 7;
    std::int32_t initial_slot = 6;
    std::int32_t batch        = 1;
    std::vector<std::int32_t> valid_columns;
    std::vector<std::int32_t> token_list;
    bool decode   = false;
    bool prefill  = false;
    bool distinct = false;
    bool snapshot = false;
    bool split    = false;
    bool cold     = false;
    bool in_place = false;
};

// The split-output fixture uses two key ranges followed by the remaining value rows; the
// public Op validates the requested channel geometry.
constexpr std::int32_t kSplitKeyDim = 2048;

bool split_partition(std::int32_t channels, std::int32_t& key_dim, std::int32_t& value_dim) {
    key_dim   = kSplitKeyDim;
    value_dim = channels - 2 * kSplitKeyDim;
    return value_dim > 0;
}

Result time_stage(const Options& options, const launch_fn& launch, double bytes_moved,
                  const launch_fn& restore = {}) {
    bench::L2FlushBuffer flush(kFlushBytes);
    const auto cuda     = device_context().execution_view();
    const auto measured = measurement::measure(
        launch,
        [&](CudaExecutionView execution) {
            if (restore) restore(execution);
        },
        flush, cuda, options.timing, options.timing.execution, options.cold ? "cold" : "warm");
    const auto& timing = measured.gpu;
    Result result;
    result.n_runs        = options.timing.repeat;
    result.median_us     = timing.median_us;
    result.min_us        = timing.min_us;
    result.p95_us        = timing.p95_us;
    const double seconds = timing.median_us * 1e-6;
    result.gbs           = seconds > 0.0 ? bytes_moved / seconds / 1e9 : 0.0;
    return result;
}

std::string cache_tag(const Options& options, const char* entry) {
    return std::string(entry) + (options.cold ? "-cold-" : "-warm-") + options.timing.execution;
}

DeviceBuffer make_varied_bf16(std::size_t n, std::uint32_t seed) {
    std::vector<std::uint16_t> h(n);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < n; ++i) {
        state         = state * 1664525u + 1013904223u;
        const float u = static_cast<float>((state >> 8) & 0x00ffffffu) * (1.0f / 16777216.0f);
        h[i]          = f32_to_bf16(2.0f * u - 1.0f);
    }
    DeviceBuffer d(ninfer::bench::device_context().device(), n * 2u);
    cudaMemcpy(d.data(), h.data(), n * 2u, cudaMemcpyHostToDevice);
    return d;
}

std::string shape_tag(const char* mode, std::int32_t channels, std::int32_t tokens,
                      std::int32_t batch = 1) {
    return std::string("causal_conv1d ") + mode + " [" + std::to_string(channels) + "," +
           std::to_string(tokens) + "," + std::to_string(batch) + "]";
}

void run_prefill(const Options& options, bool distinct) {
    const std::size_t n       = static_cast<std::size_t>(options.channels) * options.tokens;
    const std::size_t state_n = static_cast<std::size_t>(options.channels) * 3u;

    DeviceBuffer x = make_varied_bf16(n, 0x12345678U);
    DeviceBuffer weight =
        make_varied_bf16(static_cast<std::size_t>(options.channels) * 4u, 0x87654321U);
    DeviceBuffer state_in  = make_varied_bf16(state_n, 0x31415926U);
    DeviceBuffer state_out = make_zeros(state_n * 2u);
    DeviceBuffer out       = make_zeros(n * 2u);

    SavedBuffer initial(state_in);
    const auto restore = [&](CudaExecutionView cuda) { initial.restore(cuda); };
    Tensor tx(x.span(), DType::BF16, {options.channels, options.tokens});
    Tensor tw(weight.span(), DType::BF16, {options.channels, 4});
    Tensor tin(state_in.span(), DType::BF16, {options.channels, 3});
    Tensor tout_state(state_out.span(), DType::BF16, {options.channels, 3});
    Tensor tout(out.span(), DType::BF16, {options.channels, options.tokens});

    // Informational compulsory traffic: x/out, the four-tap weight, and state read/write. NCU
    // counters remain the performance authority.
    const double bytes = 4.0 * static_cast<double>(n) + 20.0 * options.channels;
    const Result r     = time_stage(
        options,
        [&](CudaExecutionView cuda) {
            if (distinct) {
                ops::causal_conv1d_silu(tx, tw, tin, tout_state, tout, cuda);
            } else {
                ops::causal_conv1d_silu(tx, tw, tin, tout, cuda);
            }
        },
        bytes, restore);
    const std::string tag = shape_tag(cache_tag(options, distinct ? "distinct" : "prefill").c_str(),
                                      options.channels, options.tokens);
    print_result(tag.c_str(), r);
}

// Split-output prefill. The destinations partition the channel extent the way the GDN caller
// does: two key-sized ranges followed by the value range.
void run_split(const Options& options) {
    std::int32_t key_dim   = 0;
    std::int32_t value_dim = 0;
    if (!split_partition(options.channels, key_dim, value_dim)) {
        std::printf("SKIP: --channels %d leaves no value rows\n", options.channels);
        return;
    }
    const std::size_t n       = static_cast<std::size_t>(options.channels) * options.tokens;
    const std::size_t state_n = static_cast<std::size_t>(options.channels) * 3u;

    DeviceBuffer x = make_varied_bf16(n, 0x12345678U);
    DeviceBuffer weight =
        make_varied_bf16(static_cast<std::size_t>(options.channels) * 4u, 0x87654321U);
    DeviceBuffer state_in  = make_varied_bf16(state_n, 0x31415926U);
    DeviceBuffer state_out = options.in_place ? DeviceBuffer{} : make_zeros(state_n * 2u);
    std::optional<SavedBuffer> initial;
    if (options.in_place) initial.emplace(state_in);
    DeviceBuffer q = make_zeros(static_cast<std::size_t>(key_dim) * options.tokens * 2u);
    DeviceBuffer k = make_zeros(static_cast<std::size_t>(key_dim) * options.tokens * 2u);
    DeviceBuffer v = make_zeros(static_cast<std::size_t>(value_dim) * options.tokens * 2u);

    Tensor tx(x.span(), DType::BF16, {options.channels, options.tokens});
    Tensor tw(weight.span(), DType::BF16, {options.channels, 4});
    Tensor tin(state_in.span(), DType::BF16, {options.channels, 3});
    Tensor tout_state(options.in_place ? state_in.span() : state_out.span(), DType::BF16,
                      {options.channels, 3});
    Tensor tq(q.span(), DType::BF16, {key_dim, options.tokens});
    Tensor tk(k.span(), DType::BF16, {key_dim, options.tokens});
    Tensor tv(v.span(), DType::BF16, {value_dim, options.tokens});

    // x in, q+k+v out, the four-tap weight and both states. NCU counters remain the performance
    // authority.
    const double bytes = 4.0 * static_cast<double>(n) + 20.0 * options.channels;
    const launch_fn restore =
        options.in_place ? launch_fn([&](CudaExecutionView cuda) { initial->restore(cuda); })
                         : launch_fn{};
    const Result r = time_stage(
        options,
        [&](CudaExecutionView cuda) {
            ops::causal_conv1d_silu_split(tx, tw, tin, tout_state, tq, tk, tv, cuda);
        },
        bytes, restore);
    const std::string tag =
        shape_tag(cache_tag(options, options.in_place ? "split-alias" : "split").c_str(),
                  options.channels, options.tokens);
    print_result(tag.c_str(), r);
}

void run_decode(const Options& options) {
    const std::size_t channels = static_cast<std::size_t>(options.channels);

    DeviceBuffer x      = make_varied_bf16(channels, 0x12345678U);
    DeviceBuffer weight = make_varied_bf16(channels * 4u, 0x87654321U);
    DeviceBuffer state  = make_varied_bf16(channels * 3u, 0x31415926U);
    DeviceBuffer out    = make_zeros(channels * 2u);

    SavedBuffer initial(state);
    const auto restore = [&](CudaExecutionView cuda) { initial.restore(cuda); };
    Tensor tx(x.span(), DType::BF16, {options.channels, 1});
    Tensor tw(weight.span(), DType::BF16, {options.channels, 4});
    Tensor ts(state.span(), DType::BF16, {options.channels, 3});
    Tensor tout(out.span(), DType::BF16, {options.channels, 1});

    const double bytes = 24.0 * options.channels;
    const Result r     = time_stage(
        options, [&](CudaExecutionView cuda) { ops::causal_conv1d_silu(tx, tw, ts, tout, cuda); },
        bytes, restore);
    const std::string tag = shape_tag(cache_tag(options, "decode").c_str(), options.channels, 1);
    print_result(tag.c_str(), r);
}

void run_snapshot(const Options& options) {
    const std::size_t n =
        static_cast<std::size_t>(options.channels) * options.tokens * options.batch;
    const std::int32_t required_slots =
        options.batch == 1 ? options.slots : options.batch * options.tokens + options.batch;
    const std::int32_t slots  = std::max(options.slots, required_slots);
    const std::size_t state_n = static_cast<std::size_t>(options.channels) * 3u * slots;

    DeviceBuffer x = make_varied_bf16(n, 0x12345678U);
    DeviceBuffer weight =
        make_varied_bf16(static_cast<std::size_t>(options.channels) * 4u, 0x87654321U);
    DeviceBuffer states = make_varied_bf16(state_n, 0x31415926U);
    DeviceBuffer initial_slot =
        make_zeros(static_cast<std::size_t>(options.batch) * sizeof(std::int32_t));
    DeviceBuffer snapshot_base_slot =
        make_zeros(static_cast<std::size_t>(options.batch) * sizeof(std::int32_t));
    DeviceBuffer valid_columns;
    DeviceBuffer out = make_zeros(n * 2u);
    std::vector<std::int32_t> initial_host(static_cast<std::size_t>(options.batch));
    std::vector<std::int32_t> base_host(static_cast<std::size_t>(options.batch));
    if (options.batch == 1) {
        initial_host[0] = options.initial_slot;
    } else {
        for (std::int32_t row = 0; row < options.batch; ++row) {
            base_host[static_cast<std::size_t>(row)]    = row * options.tokens;
            initial_host[static_cast<std::size_t>(row)] = options.batch * options.tokens + row;
        }
    }
    CUDA_CHECK(cudaMemcpy(initial_slot.data(), initial_host.data(), initial_slot.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(snapshot_base_slot.data(), base_host.data(), snapshot_base_slot.size(),
                          cudaMemcpyHostToDevice));
    if (!options.valid_columns.empty()) {
        valid_columns = make_zeros(options.valid_columns.size() * sizeof(std::int32_t));
        CUDA_CHECK(cudaMemcpy(valid_columns.data(), options.valid_columns.data(),
                              valid_columns.size(), cudaMemcpyHostToDevice));
    }

    SavedBuffer initial(states);
    const auto restore = [&](CudaExecutionView cuda) { initial.restore(cuda); };
    Tensor tx(x.span(), DType::BF16, {options.channels, options.tokens, options.batch});
    Tensor tw(weight.span(), DType::BF16, {options.channels, 4});
    Tensor ts(states.span(), DType::BF16, {options.channels, 3, slots});
    Tensor tvalid;
    if (!options.valid_columns.empty()) {
        tvalid = Tensor(valid_columns.span(), DType::I32, {options.batch});
    }
    Tensor tslot(initial_slot.span(), DType::I32, {options.batch});
    Tensor tsnapshot_base(snapshot_base_slot.span(), DType::I32, {options.batch});
    Tensor tout(out.span(), DType::BF16, {options.channels, options.tokens, options.batch});

    // The selected history and four weights are reusable across T; each token reads x, writes out,
    // and publishes three BF16 state columns.
    const double bytes = 8.0 * options.channels + 6.0 * options.channels * options.batch +
                         10.0 * static_cast<double>(n);
    const Result r = time_stage(
        options,
        [&](CudaExecutionView cuda) {
            ops::causal_conv1d_silu_snapshot(tx, tw, ts, tvalid, tslot, tsnapshot_base, tout, cuda);
        },
        bytes, restore);
    const std::string tag = shape_tag(
        cache_tag(options, options.valid_columns.empty() ? "snapshot dense" : "snapshot masked")
            .c_str(),
        options.channels, options.tokens, options.batch);
    print_result(tag.c_str(), r);
}

void print_usage(const char* program) {
    std::fprintf(stderr,
                 "usage: %s [--decode] [--prefill] [--distinct] [--snapshot] [--split] "
                 "[--inplace] [--cache warm|cold] [--execution graph|eager] "
                 "[--warmup N] [--repeat N] [--channels C] [--tokens T[,T...]] "
                 "[--batch B] [--valid-columns V0,V1,...] [--slots S] "
                 "[--initial-slot I]\n",
                 program);
}

bool parse_options(int argc, char** argv, Options& options) try {
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--decode")) {
            options.decode = true;
        } else if (!std::strcmp(argv[i], "--prefill")) {
            options.prefill = true;
        } else if (!std::strcmp(argv[i], "--distinct")) {
            options.distinct = true;
        } else if (!std::strcmp(argv[i], "--snapshot")) {
            options.snapshot = true;
        } else if (!std::strcmp(argv[i], "--split")) {
            options.split = true;
        } else if (!std::strcmp(argv[i], "--inplace")) {
            options.in_place = true;
        } else if ((!std::strcmp(argv[i], "--execution") || !std::strcmp(argv[i], "--warmup") ||
                    !std::strcmp(argv[i], "--repeat")) &&
                   i + 1 < argc) {
            const std::string_view flag = argv[i++];
            if (!measurement::parse(options.timing, flag, argv[i])) return false;
            if (options.timing.execution == "both") return false;
        } else if (!std::strcmp(argv[i], "--cache") && i + 1 < argc) {
            const char* value = argv[++i];
            if (!std::strcmp(value, "cold")) {
                options.cold = true;
            } else if (!std::strcmp(value, "warm")) {
                options.cold = false;
            } else {
                return false;
            }
        } else if (!std::strcmp(argv[i], "--valid-columns") && i + 1 < argc) {
            options.valid_columns = measurement::integers(argv[++i]);
        } else if ((!std::strcmp(argv[i], "--channels") || !std::strcmp(argv[i], "--tokens") ||
                    !std::strcmp(argv[i], "--slots") || !std::strcmp(argv[i], "--initial-slot") ||
                    !std::strcmp(argv[i], "--batch")) &&
                   i + 1 < argc) {
            const char* flag = argv[i++];
            if (!std::strcmp(flag, "--initial-slot")) {
                options.initial_slot = measurement::integer(argv[i], 0);
            } else if (!std::strcmp(flag, "--tokens")) {
                options.token_list = measurement::integers(argv[i]);
            } else {
                std::int32_t* destination = !std::strcmp(flag, "--channels") ? &options.channels
                                            : !std::strcmp(flag, "--batch")  ? &options.batch
                                                                             : &options.slots;
                *destination              = measurement::integer(argv[i]);
            }
        } else {
            return false;
        }
    }
    if (!options.decode && !options.prefill && !options.distinct && !options.snapshot &&
        !options.split) {
        options.decode = options.prefill = true;
    }
    if (options.batch > 8 || (options.in_place && !options.split) ||
        ((!options.snapshot) && (options.batch != 1 || !options.valid_columns.empty())) ||
        (!options.valid_columns.empty() &&
         options.valid_columns.size() != static_cast<std::size_t>(options.batch))) {
        return false;
    }
    for (const std::int32_t tokens : options.token_list.empty()
                                         ? std::vector<std::int32_t>{options.tokens}
                                         : options.token_list) {
        if (options.batch > 1 && tokens > 16) { return false; }
        if (options.snapshot && options.batch == 1 && tokens > options.slots) { return false; }
        for (const std::int32_t valid : options.valid_columns) {
            if (valid > tokens) { return false; }
        }
    }
    return !options.snapshot || options.batch > 1 || options.initial_slot < options.slots;
} catch (const std::invalid_argument&) { return false; }

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    Options options;
    if (!parse_options(argc, argv, options)) {
        print_usage(argv[0]);
        return 2;
    }

    std::vector<std::int32_t> tokens = options.token_list;
    if (tokens.empty()) { tokens.push_back(options.tokens); }
    for (const std::int32_t t : tokens) {
        Options point = options;
        point.tokens  = t;
        if (point.decode) run_decode(point);
        if (point.prefill) run_prefill(point, false);
        if (point.distinct) run_prefill(point, true);
        if (point.snapshot) run_snapshot(point);
        if (point.split) run_split(point);
    }
    return 0;
}
