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
#include "ragged_workload.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <optional>
#include <limits>
#include <stdexcept>
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
    std::vector<ragged::SegmentSpec> ragged_specs;
    std::optional<ragged::StateOffsetForm> state_form;
    int scalar_state_align = 16;
    bool decode            = false;
    bool prefill           = false;
    bool distinct          = false;
    bool snapshot          = false;
    bool split             = false;
    bool in_place          = false;
};

// The split-output fixture uses two key ranges followed by the remaining value rows; the
// public Op validates the requested channel geometry.
constexpr std::int32_t kSplitKeyDim = 2048;

bool split_partition(std::int32_t channels, std::int32_t& key_dim, std::int32_t& value_dim) {
    key_dim   = channels == 5120 ? 1024 : kSplitKeyDim;
    value_dim = channels - 2 * key_dim;
    return value_dim > 0;
}

std::string csv_fields(std::uint64_t bytes, int calls, std::string_view form, int state_align) {
    return std::to_string(bytes) + ',' + std::to_string(calls) + ',' + std::string(form) + ',' +
           std::to_string(state_align);
}

Result time_stage(const Options& options, const launch_fn& launch, double bytes_moved,
                  const launch_fn& restore = {}, const measurement::Row* row = nullptr,
                  measurement::Csv* csv = nullptr, std::string_view form = {}) {
    bench::L2FlushBuffer flush(kFlushBytes);
    const auto cuda     = device_context().execution_view();
    const auto measured = measurement::measure(
        launch,
        [&](CudaExecutionView execution) {
            if (restore) restore(execution);
        },
        flush, cuda, options.timing, options.timing.execution, options.timing.cache);
    if (csv && row) {
        csv->write(*row, measured, options.timing.execution, options.timing.cache,
                   csv_fields(static_cast<std::uint64_t>(bytes_moved), row->batch, form, 16));
    }
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
    return std::string(entry) + '-' + options.timing.cache + '-' + options.timing.execution;
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
    d.copy_from_host(h.data(), d.size());
    return d;
}

std::string shape_tag(const char* mode, std::int32_t channels, std::int32_t tokens,
                      std::int32_t batch = 1) {
    return std::string("causal_conv1d ") + mode + " [" + std::to_string(channels) + "," +
           std::to_string(tokens) + "," + std::to_string(batch) + "]";
}

void run_prefill(const Options& options, bool distinct, measurement::Csv* csv) {
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
    measurement::Row row;
    row.op            = "causal_conv";
    row.format        = "bf16";
    row.policy        = "a16";
    row.conv_channels = options.channels;
    row.tokens        = options.tokens;
    const Result r    = time_stage(
        options,
        [&](CudaExecutionView cuda) {
            if (distinct) {
                ops::causal_conv1d_silu(tx, tw, tin, tout_state, tout, cuda);
            } else {
                ops::causal_conv1d_silu(tx, tw, tin, tout, cuda);
            }
        },
        bytes, restore, &row, csv, distinct ? "distinct" : "alias");
    const std::string tag = shape_tag(cache_tag(options, distinct ? "distinct" : "prefill").c_str(),
                                      options.channels, options.tokens);
    print_result(tag.c_str(), r);
}

// Split-output prefill. The destinations partition the channel extent the way the GDN caller
// does: two key-sized ranges followed by the value range.
void run_split(const Options& options, measurement::Csv* csv) {
    std::int32_t key_dim   = 0;
    std::int32_t value_dim = 0;
    if (!split_partition(options.channels, key_dim, value_dim)) {
        std::printf("SKIP: --channels %d leaves no value rows\n", options.channels);
        return;
    }
    const std::size_t n =
        static_cast<std::size_t>(options.channels) * options.tokens * options.batch;
    const std::size_t state_n = static_cast<std::size_t>(options.channels) * 3u * options.batch;

    DeviceBuffer x = make_varied_bf16(n, 0x12345678U);
    DeviceBuffer weight =
        make_varied_bf16(static_cast<std::size_t>(options.channels) * 4u, 0x87654321U);
    DeviceBuffer state_in  = make_varied_bf16(state_n, 0x31415926U);
    DeviceBuffer state_out = options.in_place ? DeviceBuffer{} : make_zeros(state_n * 2u);
    std::optional<SavedBuffer> initial;
    if (options.in_place) initial.emplace(state_in);
    DeviceBuffer q =
        make_zeros(static_cast<std::size_t>(key_dim) * options.tokens * options.batch * 2u);
    DeviceBuffer k =
        make_zeros(static_cast<std::size_t>(key_dim) * options.tokens * options.batch * 2u);
    DeviceBuffer v =
        make_zeros(static_cast<std::size_t>(value_dim) * options.tokens * options.batch * 2u);

    Tensor tw(weight.span(), DType::BF16, {options.channels, 4});

    struct Views {
        Tensor x, state_in, state_out, q, k, v;
    };

    std::vector<Views> rows;
    const auto x_bytes = std::size_t(options.channels) * options.tokens * 2;
    const auto h_bytes = std::size_t(options.channels) * 3 * 2;
    const auto q_bytes = std::size_t(key_dim) * options.tokens * 2;
    const auto v_bytes = std::size_t(value_dim) * options.tokens * 2;
    for (int b = 0; b < options.batch; ++b) {
        const auto dest = options.in_place ? state_in.span() : state_out.span();
        rows.push_back(
            {Tensor(x.span().subspan(b * x_bytes, x_bytes), DType::BF16,
                    {options.channels, options.tokens}),
             Tensor(state_in.span().subspan(b * h_bytes, h_bytes), DType::BF16,
                    {options.channels, 3}),
             Tensor(dest.subspan(b * h_bytes, h_bytes), DType::BF16, {options.channels, 3}),
             Tensor(q.span().subspan(b * q_bytes, q_bytes), DType::BF16, {key_dim, options.tokens}),
             Tensor(k.span().subspan(b * q_bytes, q_bytes), DType::BF16, {key_dim, options.tokens}),
             Tensor(v.span().subspan(b * v_bytes, v_bytes), DType::BF16,
                    {value_dim, options.tokens})});
    }
    // B native 2D calls. Count each call's boundary payload; cache reuse is not DRAM traffic.
    const double bytes =
        options.batch * (4.0 * options.channels * options.tokens + 20.0 * options.channels);
    measurement::Row row;
    row.op            = "causal_conv_split";
    row.format        = "bf16";
    row.policy        = "a16";
    row.conv_channels = options.channels;
    row.tokens        = options.tokens;
    row.batch         = options.batch;
    const launch_fn restore =
        options.in_place ? launch_fn([&](CudaExecutionView cuda) { initial->restore(cuda); })
                         : launch_fn{};
    const Result r = time_stage(
        options,
        [&](CudaExecutionView cuda) {
            for (auto& views : rows)
                ops::causal_conv1d_silu_split(views.x, tw, views.state_in, views.state_out, views.q,
                                              views.k, views.v, cuda);
        },
        bytes, restore, &row, csv, options.in_place ? "alias" : "distinct");
    const std::string tag =
        shape_tag(cache_tag(options, options.in_place ? "split-alias" : "split").c_str(),
                  options.channels, options.tokens, options.batch);
    print_result(tag.c_str(), r);
}

void run_ragged(const Options& options, const ragged::SegmentSpec& lengths, measurement::Csv& csv,
                L2FlushBuffer& flush) {
    const auto cuda = device_context().execution_view();
    const ragged::RaggedDeviceFixture workload(lengths, cuda.device());
    const auto seq = workload.view();
    const int n = seq.total_tokens(), s = seq.num_seqs(), c = options.channels;
    int key_dim = 0, value_dim = c;
    if (options.split && !split_partition(c, key_dim, value_dim))
        throw std::invalid_argument("split channels leave no value rows");

    const auto form = options.state_form.value_or(
        options.in_place ? ragged::StateOffsetForm::Alias : ragged::StateOffsetForm::Distinct);
    const auto history_bytes       = std::size_t(c) * 6;
    const std::size_t field_offset = options.scalar_state_align == 2 ? 2 : 0;
    const auto offsets = ragged::make_state_offsets(s, form, history_bytes + field_offset, 16);
    auto sources       = ragged::upload_offsets(cuda.device(), offsets.source_offsets_host);
    auto destinations  = ragged::upload_offsets(cuda.device(), offsets.destination_offsets_host);
    auto states        = make_varied_bf16(offsets.backing_bytes() / 2, 0x31415926U);
    std::optional<SavedBuffer> initial;
    if (offsets.mutates_source) initial.emplace(states);
    const StateFieldLayout layout{
        {{field_offset, history_bytes, std::size_t(options.scalar_state_align)},
         DType::BF16,
         {c, 3, 1, 1}},
        1,
        0};
    const ConstStateFieldView source{{states.span(), ragged::offset_tensor(sources)}, layout};
    const StateFieldView destination{{states.span(), ragged::offset_tensor(destinations)}, layout};
    auto x      = make_varied_bf16(std::size_t(c) * n, 0x12345678U);
    auto weight = make_varied_bf16(std::size_t(c) * 4, 0x87654321U);
    auto out    = make_zeros(std::size_t(value_dim) * n * 2);
    const Tensor tx(x.span(), DType::BF16, {c, n}), tw(weight.span(), DType::BF16, {c, 4});
    Tensor y(out.span(), DType::BF16, {value_dim, n});
    DeviceBuffer q, k;
    Tensor tq, tk;
    if (options.split) {
        q  = make_zeros(std::size_t(key_dim) * n * 2);
        k  = make_zeros(q.size());
        tq = Tensor(q.span(), DType::BF16, {key_dim, n});
        tk = Tensor(k.span(), DType::BF16, {key_dim, n});
    }
    const auto launch = [&](CudaExecutionView execution) {
        if (options.split)
            ops::causal_conv1d_silu_split(tx, tw, seq, source, destination, tq, tk, y, execution);
        else
            ops::causal_conv1d_silu(tx, tw, seq, source, destination, y, execution);
    };
    const auto restore = [&](CudaExecutionView execution) {
        if (initial) initial->restore(execution);
    };
    const auto bytes =
        4ULL * c * n + 8ULL * c + std::uint64_t(history_bytes) * (offsets.num_source_records + s);
    measurement::Row row;
    row.op            = options.split ? "causal_conv_split" : "causal_conv";
    row.format        = "bf16";
    row.policy        = "a16";
    row.conv_channels = c;
    row.tokens        = n;
    row.batch         = s;
    row.valid         = ragged::format_segments(lengths);
    const auto extra =
        csv_fields(bytes, 1, ragged::state_form_name(form), options.scalar_state_align);
    for (const auto& execution :
         measurement::selections(options.timing.execution, "graph", "eager"))
        for (const auto& cache : measurement::selections(options.timing.cache, "cold", "warm"))
            csv.write(row,
                      measurement::measure(launch, restore, flush, cuda, options.timing, execution,
                                           cache),
                      execution, cache, extra);
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
    const auto required = options.batch == 1
                              ? std::int64_t(options.slots)
                              : std::int64_t(options.batch) * (std::int64_t(options.tokens) + 1);
    if (required > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument("snapshot slot extent exceeds I32");
    const auto required_slots = static_cast<std::int32_t>(required);
    const std::int32_t slots  = std::max(options.slots, required_slots);
    const std::size_t state_n = static_cast<std::size_t>(options.channels) * 3u * slots;

    DeviceBuffer x = make_varied_bf16(n, 0x12345678U);
    DeviceBuffer weight =
        make_varied_bf16(static_cast<std::size_t>(options.channels) * 4u, 0x87654321U);
    DeviceBuffer states    = make_varied_bf16(state_n, 0x31415926U);
    DeviceBuffer snapshots = make_zeros(state_n * sizeof(std::uint16_t));
    DeviceBuffer initial_slot =
        make_zeros(static_cast<std::size_t>(options.batch) * sizeof(std::int64_t));
    DeviceBuffer snapshot_base_slot =
        make_zeros(static_cast<std::size_t>(options.batch) * sizeof(std::int32_t));
    DeviceBuffer valid_columns;
    DeviceBuffer out = make_zeros(n * 2u);
    std::vector<std::int64_t> initial_host(static_cast<std::size_t>(options.batch));
    std::vector<std::int32_t> base_host(static_cast<std::size_t>(options.batch));
    if (options.batch == 1) {
        initial_host[0] = options.initial_slot;
    } else {
        for (std::int32_t row = 0; row < options.batch; ++row) {
            base_host[static_cast<std::size_t>(row)]    = row * options.tokens;
            initial_host[static_cast<std::size_t>(row)] = options.batch * options.tokens + row;
        }
    }
    for (auto& offset : initial_host) offset *= 6LL * options.channels;
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
    Tensor tvalid;
    if (!options.valid_columns.empty()) {
        tvalid = Tensor(valid_columns.span(), DType::I32, {options.batch});
    }
    Tensor tslot(initial_slot.span(), DType::I64, {options.batch});
    const StateFieldView source{
        {states.span(), tslot},
        {{{0, std::size_t(options.channels) * 6, 2}, DType::BF16, {options.channels, 3, 1, 1}},
         1,
         0}};
    Tensor snapshot(snapshots.span(), DType::BF16, {options.channels, 3, slots});
    Tensor tsnapshot_base(snapshot_base_slot.span(), DType::I32, {options.batch});
    Tensor tout(out.span(), DType::BF16, {options.channels, options.tokens, options.batch});

    std::uint64_t live_rows = 0, live_tokens = 0;
    for (int row = 0; row < options.batch; ++row) {
        const int count =
            options.valid_columns.empty() ? options.tokens : options.valid_columns[row];
        live_rows += count != 0;
        live_tokens += count;
    }
    // Weights/history and input/snapshots are read or written only for active prefixes;
    // every physical output column is written, including zero padding.
    const double bytes =
        options.channels * ((live_rows ? 8.0 : 0.0) + 6.0 * live_rows + 8.0 * live_tokens) +
        2.0 * static_cast<double>(n);
    const Result r = time_stage(
        options,
        [&](CudaExecutionView cuda) {
            ops::causal_conv1d_silu_snapshot(tx, tw, source, snapshot, tvalid, tsnapshot_base, tout,
                                             cuda);
        },
        bytes, restore);
    const std::string tag = shape_tag(
        cache_tag(options, options.valid_columns.empty() ? "snapshot dense" : "snapshot masked")
            .c_str(),
        options.channels, options.tokens, options.batch);
    print_result(tag.c_str(), r);
}

void print_usage(const char* program) {
    std::fprintf(
        stderr,
        "usage: %s [--decode] [--prefill] [--distinct] [--snapshot] [--split] "
        "[--inplace] [--cache warm|cold|both] [--execution graph|eager|both] "
        "[--warmup N] [--repeat N] [--channels C] [--tokens T[,T...] | --sweep START:END[:STEP]] "
        "[--batch B] [--valid-columns V0,V1,...] [--slots S] "
        "[--initial-slot I] [--csv-out PATH (prefill or split)]\n"
        "  --ragged-segments SPEC --state-form distinct|alias|mixed|permuted|shared-source\n"
        "  --scalar-state-align 2|16 (ragged scalar prefill; default 16)\n"
        "Split --batch B measures B native 2D calls with independent histories.\n",
        program);
}

bool parse_options(int argc, char** argv, Options& options) try {
    bool explicit_tokens = false;
    std::optional<std::string_view> ragged_text;
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
        } else if (!std::strcmp(argv[i], "--inplace") || !std::strcmp(argv[i], "--in-place")) {
            options.in_place = true;
        } else if (!std::strcmp(argv[i], "--ragged-segments") && i + 1 < argc) {
            ragged_text = argv[++i];
        } else if (!std::strcmp(argv[i], "--state-form") && i + 1 < argc) {
            options.state_form = ragged::parse_state_form(argv[++i]);
        } else if (!std::strcmp(argv[i], "--scalar-state-align") && i + 1 < argc) {
            options.scalar_state_align = measurement::integer(argv[++i]);
            if (options.scalar_state_align != 2 && options.scalar_state_align != 16) return false;
        } else if ((!std::strcmp(argv[i], "--execution") || !std::strcmp(argv[i], "--warmup") ||
                    !std::strcmp(argv[i], "--repeat") || !std::strcmp(argv[i], "--csv-out") ||
                    !std::strcmp(argv[i], "--cache")) &&
                   i + 1 < argc) {
            const std::string_view flag = argv[i++];
            if (!measurement::parse(options.timing, flag, argv[i])) return false;
        } else if (!std::strcmp(argv[i], "--valid-columns") && i + 1 < argc) {
            options.valid_columns = measurement::integers(argv[++i], 0);
        } else if ((!std::strcmp(argv[i], "--tokens") || !std::strcmp(argv[i], "--sweep")) &&
                   i + 1 < argc) {
            const std::string_view flag = argv[i++];
            options.token_list =
                flag == "--tokens" ? measurement::integers(argv[i]) : measurement::sweep(argv[i]);
            explicit_tokens = true;
        } else if ((!std::strcmp(argv[i], "--channels") || !std::strcmp(argv[i], "--slots") ||
                    !std::strcmp(argv[i], "--initial-slot") || !std::strcmp(argv[i], "--batch")) &&
                   i + 1 < argc) {
            const char* flag = argv[i++];
            if (!std::strcmp(flag, "--initial-slot")) {
                options.initial_slot = measurement::integer(argv[i], 0);
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
    if (!options.timing.csv_out.empty() && (options.decode || options.snapshot))
        throw std::invalid_argument("--decode and --snapshot are incompatible with --csv-out");
    if (ragged_text) {
        if (explicit_tokens)
            throw std::invalid_argument(
                "--ragged-segments is mutually exclusive with --tokens and --sweep");
        if (options.decode || options.snapshot || options.batch != 1 ||
            !options.valid_columns.empty())
            throw std::invalid_argument(
                "ragged convolution requires prefill, batch 1 and dense columns");
        if (options.split && options.scalar_state_align == 2)
            throw std::invalid_argument("ragged split requires --scalar-state-align 16");
        options.ragged_specs = ragged::parse_ragged_specs(*ragged_text);
        return true;
    }
    if (options.state_form || options.scalar_state_align != 16) return false;
    if (!options.decode && !options.prefill && !options.distinct && !options.snapshot &&
        !options.split) {
        options.prefill = true;
        options.decode  = options.timing.csv_out.empty();
    }
    if ((options.in_place && !options.split) ||
        (!options.snapshot && !options.split && options.batch != 1) ||
        (!options.snapshot && !options.valid_columns.empty()) ||
        (!options.valid_columns.empty() &&
         options.valid_columns.size() != static_cast<std::size_t>(options.batch))) {
        return false;
    }
    for (const std::int32_t tokens : options.token_list.empty()
                                         ? std::vector<std::int32_t>{options.tokens}
                                         : options.token_list) {
        if (options.snapshot && options.batch > 1 && tokens > 16) { return false; }
        if (options.snapshot && options.batch == 1 && tokens > options.slots) { return false; }
        for (const std::int32_t valid : options.valid_columns) {
            if (valid > tokens) { return false; }
        }
    }
    return !options.snapshot || options.batch > 1 || options.initial_slot < options.slots;
} catch (const std::invalid_argument& error) {

    std::fprintf(stderr, "%s\n", error.what());
    return false;
}

} // namespace

int main(int argc, char** argv) try {
    Options options;
    options.timing.cache = "warm";
    if (!parse_options(argc, argv, options)) {
        print_usage(argv[0]);
        return 2;
    }

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    std::optional<measurement::Csv> csv;
    if (!options.timing.csv_out.empty() || !options.ragged_specs.empty())
        csv.emplace(options.timing.csv_out, "logical_bytes,calls,state_form,state_align");
    if (!options.ragged_specs.empty()) {
        L2FlushBuffer flush(kFlushBytes);
        for (const auto& lengths : options.ragged_specs) run_ragged(options, lengths, *csv, flush);
        return 0;
    }
    std::vector<std::int32_t> tokens = options.token_list;
    if (tokens.empty()) { tokens.push_back(options.tokens); }
    for (const std::int32_t t : tokens) {
        Options point = options;
        point.tokens  = t;
        for (const auto& execution :
             measurement::selections(options.timing.execution, "graph", "eager"))
            for (const auto& cache :
                 measurement::selections(options.timing.cache, "cold", "warm")) {
                point.timing.execution = execution;
                point.timing.cache     = cache;
                if (point.decode) run_decode(point);
                if (point.prefill) run_prefill(point, false, csv ? &*csv : nullptr);
                if (point.distinct) run_prefill(point, true, csv ? &*csv : nullptr);
                if (point.snapshot) run_snapshot(point);
                if (point.split) run_split(point, csv ? &*csv : nullptr);
            }
    }
    return 0;
} catch (const std::exception& error) {

    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
