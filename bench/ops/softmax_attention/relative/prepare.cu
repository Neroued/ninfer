#include "common/roofline.h"
#include "ninfer/ops/relative_attention.h"
#include "ops/op_measurement.cuh"
#include "ops/ragged_workload.h"
#include "../../../../tools/support/paged_kv_fixture.h"

#include <cuda_bf16.h>

#include <array>
#include <limits>
#include <numeric>
#include <optional>

using namespace ninfer;
namespace m = ninfer::bench::measurement;
namespace r = ninfer::bench::ragged;
using ops::relative_attention_commit_cyclic_kv;
using ops::relative_attention_prepare;
using ops::relative_attention_prepare_snapshot;

namespace {

constexpr int D = 128, R = 16, QH = 24, KH = 8, C = D * KH, Window = 1024;
constexpr std::array positions{0, 63, 8190, 8191, 8192, 8193, 32767, 1023, 2047, 64, 65};

struct Options {
    m::Options measure;
    std::string mode = "both", format = "both", phase = "all";
    std::optional<int> batch, width, sink, dynamic, tau, position;
    r::StateOffsetForm state_form      = r::StateOffsetForm::Distinct;
    bool edge                          = false;
    std::vector<r::SegmentSpec> ragged = r::parse_ragged_specs(
        "1,2,3,4,5,17,313,977,1021;1021;1024;977,1021,17,3,2,1;2x1024;4x1024;4093;4096");
};

void help() {
    std::cout << "Usage: ninfer_relative_attention_prepare_bench\n"
                 "  --mode swa|full|both --kv-format bf16|fp8|both\n"
                 "  --phase decode|direct|verify|ragged|commit|all --batch B --width W\n"
                 "  --ragged LENGTHS --state-form distinct|alias|mixed|permuted|shared-source\n"
                 "  --sink-channels H --dynamic 0|1 --tau 0|1 --position P --edge 0|1\n"
                 "Batched shapes default to decode and a sweep of verify and off-grid points.\n"
                 "SWA defaults to static without sink/tau and Full to dynamic with sink/tau.\n";
    m::help();
}

Options parse(int argc, char** argv) {
    Options options;
    options.measure.repeat = 30;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        const auto value = m::take(argc, argv, i);
        auto choice      = [&](std::initializer_list<std::string_view> valid) {
            if (std::find(valid.begin(), valid.end(), value) == valid.end())
                throw std::invalid_argument("invalid value for " + std::string(flag));
            return std::string(value);
        };
        if (m::parse(options.measure, flag, value)) continue;
        if (flag == "--mode")
            options.mode = choice({"swa", "full", "both"});
        else if (flag == "--kv-format")
            options.format = choice({"bf16", "fp8", "both"});
        else if (flag == "--phase")
            options.phase = choice({"decode", "direct", "verify", "ragged", "commit", "all"});
        else if (flag == "--batch")
            options.batch = m::integer(value);
        else if (flag == "--width")
            options.width = m::integer(value);
        else if (flag == "--sink-channels" || flag == "--sink")
            options.sink = m::integer(value, 0);
        else if (flag == "--dynamic")
            options.dynamic = m::integer(value, 0, 1);
        else if (flag == "--tau")
            options.tau = m::integer(value, 0, 1);
        else if (flag == "--position")
            options.position = m::integer(value, 0);
        else if (flag == "--edge")
            options.edge = m::integer(value, 0, 1);
        else if (flag == "--ragged")
            options.ragged = r::parse_ragged_specs(value);
        else if (flag == "--state-form")
            options.state_form = r::parse_state_form(value);
        else
            throw std::invalid_argument("unknown option " + std::string(flag));
    }
    if (options.phase == "decode" && options.width.value_or(1) != 1)
        throw std::invalid_argument("decode requires --width 1");
    if (options.mode == "full" && options.phase == "commit")
        throw std::invalid_argument("cyclic commit requires SWA");
    return options;
}

struct Workload {
    bool paged, fp8, snapshot;
    int width, sequences, columns;
    std::string phase;
    r::SegmentSpec lengths;
};

class Fixture {
public:
    Fixture(const Options& options, const Workload& work, Device device)
        : work(work), device(device),
          storage(work.fp8 ? KvCacheStorage::Fp8E4m3fnGroup128Ue8m0 : KvCacheStorage::BFloat16),
          sink(options.sink.value_or(work.paged ? 2304 : 0)),
          dynamic(options.dynamic.value_or(work.paged)), edge(options.edge) {
        params.use_log_scaling = options.tau.value_or(work.paged);
        if (edge) params.qk_clip_threshold = 1024;
        const bool packed = work.phase == "ragged";
        const int w = packed ? work.columns : work.width, b = packed ? 1 : work.sequences;
        in.q             = bf16({D, QH, w, b});
        in.r             = bf16({R, QH, w, b});
        in.k             = bf16({D, KH, w, b}, true);
        in.v             = bf16({D, KH, w, b}, true);
        in.q_scale       = bf16({D, QH});
        in.r_scale       = bf16({R, QH});
        in.k_conv_weight = bf16({C, 4}, false, edge);
        in.v_conv_weight = bf16({C, 4}, false, edge);
        if (dynamic) {
            in.k_dyn = bf16({4, KH, w, b}, false, edge);
            in.v_dyn = bf16({4, KH, w, b}, false, edge);
        }
        if (sink) {
            in.sink_x      = bf16({sink, w, b});
            in.sink_weight = bf16({sink, QH});
            out.sink_out   = empty(DType::BF16, {QH, w, b});
        }
        out.q_out = empty(DType::BF16, {D, QH, w, b});
        out.r_out = empty(DType::BF16, {R, QH, w, b});

        std::vector<int> pos, starts, row_map(work.sequences);
        std::vector<std::int64_t> ends;
        for (int s = 0; s < work.sequences; ++s) {
            const int start = options.position.value_or(positions[s % positions.size()]);
            const auto last = std::int64_t(start) + work.lengths[s] - 1;
            if (last > std::numeric_limits<int>::max())
                throw std::invalid_argument("position range exceeds I32");
            starts.push_back(start);
            ends.push_back(last);
            for (int j = 0; j < work.lengths[s]; ++j) pos.push_back(start + j);
            row_map[s] = work.sequences - 1 - s;
        }
        in.positions = upload(pos, DType::I32, {w, b});
        if (packed)
            ragged.emplace(work.lengths, device);
        else
            valid = upload(work.lengths, DType::I32, {work.sequences});

        StateImageBuilder builder;
        const auto k_field = builder.add(DType::BF16, {C, 3}, 1, "K history");
        const auto v_field = builder.add(DType::BF16, {C, 3}, 1, "V history");
        const auto layout  = builder.finish();
        auto offsets = r::make_state_offsets(work.sequences, options.state_form, layout.image_bytes,
                                             layout.alignment);
        source_bytes = offsets.source_bytes();
        state        = DeviceBuffer(device, offsets.backing_bytes());
        state.fill(0x3d);
        StateImageStorage image(state.span(), layout);
        auto before = upload(offsets.source_offsets_host, DType::I64, {work.sequences});
        k_in        = image.field(k_field, before);
        v_in        = image.field(v_field, before);
        if (work.snapshot) {
            std::vector<int> slots(work.sequences);
            for (int s = 0; s < work.sequences; ++s) slots[s] = row_map[s] * work.width;
            snapshot_slots = upload(slots, DType::I32, {work.sequences});
            k_snapshots    = empty(DType::BF16, {C, 3, w * b});
            v_snapshots    = empty(DType::BF16, {C, 3, w * b});
        } else {
            const auto after =
                upload(offsets.destination_offsets_host, DType::I64, {work.sequences});
            k_out          = image.field(k_field, after);
            v_out          = image.field(v_field, after);
            restore_source = offsets.mutates_source;
        }

        if (work.paged) {
            rows = upload(row_map, DType::I32, {work.sequences});
            KVPageGeometry geo{64,
                               {{work.fp8 ? DType::FP8_E4M3FN : DType::BF16, D, KH},
                                {work.fp8 ? DType::FP8_E4M3FN : DType::FP16, D, KH}}};
            if (work.fp8) geo.planes.insert(geo.planes.end(), 2, {DType::U8, 1, KH});
            std::int64_t logical = 0, physical = 0;
            for (int s = 0; s < work.sequences; ++s) {
                logical = std::max(logical, ends[s] / kPagedKVPageSize + 1);
                physical += ends[s] / kPagedKVPageSize - starts[s] / kPagedKVPageSize + 1;
            }
            if (physical > std::numeric_limits<int>::max())
                throw std::invalid_argument("physical page count exceeds I32");
            page_entries = physical;
            pages.emplace(device, geo, static_cast<int>(physical));
            std::vector<std::int64_t> table(std::size_t(logical) * work.sequences, -1);
            for (int s = 0; s < work.sequences; ++s)
                for (auto p = std::int64_t(starts[s]) / kPagedKVPageSize;
                     p <= ends[s] / kPagedKVPageSize; ++p)
                    table[std::size_t(row_map[s]) * logical + p] =
                        pages->physical_offset(static_cast<int>(--physical));
            const auto page_offsets =
                upload(table, DType::I64, {static_cast<int>(logical), work.sequences});
            paged = {pages->plane(0, page_offsets),
                     pages->plane(1, page_offsets),
                     work.fp8 ? pages->plane(2, page_offsets) : PagedPlaneView{},
                     work.fp8 ? pages->plane(3, page_offsets) : PagedPlaneView{},
                     D,
                     KH,
                     storage};
        } else {
            if (work.phase != "decode") {
                chunk = {storage, empty(work.fp8 ? DType::FP8_E4M3FN : DType::BF16, {D, KH, w, b}),
                         empty(work.fp8 ? DType::FP8_E4M3FN : DType::FP16, {D, KH, w, b}),
                         work.fp8 ? empty(DType::U8, {1, KH, w, b}) : Tensor{},
                         work.fp8 ? empty(DType::U8, {1, KH, w, b}) : Tensor{}};
            }
            StateImageBuilder cache_builder;
            const auto cache = plan_cyclic_kv_cache(cache_builder, 1, Window, KH, D, storage);
            const auto cache_layout = cache_builder.finish();
            std::vector<std::int64_t> cache_offsets(work.sequences);
            for (int s = 0; s < work.sequences; ++s)
                cache_offsets[s] = row_map[s] * cache_layout.image_bytes;
            cache_buffer = DeviceBuffer(device, cache_layout.image_bytes * work.sequences);
            cache_buffer.fill();
            StateImageStorage cache_image(cache_buffer.span(), cache_layout);
            cyclic = CyclicKVCacheView(cache_image, cache)
                         .layer_view(0, upload(cache_offsets, DType::I64, {work.sequences}));
        }
    }

    void restore(CudaExecutionView cuda) {
        if (restore_source)
            CUDA_CHECK(cudaMemsetAsync(state.data(), 0x3d, source_bytes, cuda.stream()));
    }

    void launch(CudaExecutionView cuda) {
        if (ragged) {
            if (work.paged)
                relative_attention_prepare(in, ragged->view(), rows, k_in, k_out, v_in, v_out,
                                           paged, params, out, cuda);
            else
                relative_attention_prepare(in, ragged->view(), k_in, k_out, v_in, v_out, chunk,
                                           params, out, cuda);
        } else if (work.snapshot) {
            if (work.paged)
                relative_attention_prepare_snapshot(in, valid, rows, snapshot_slots, k_in,
                                                    k_snapshots, v_in, v_snapshots, paged, params,
                                                    out, cuda);
            else
                relative_attention_prepare_snapshot(in, valid, snapshot_slots, k_in, k_snapshots,
                                                    v_in, v_snapshots, chunk, params, out, cuda);
        } else if (work.paged) {
            relative_attention_prepare(in, valid, rows, k_in, k_out, v_in, v_out, paged, params,
                                       out, cuda);
        } else if (chunk.k.data) {
            relative_attention_prepare(in, valid, k_in, k_out, v_in, v_out, chunk, params, out,
                                       cuda);
        } else {
            relative_attention_prepare(in, valid, k_in, k_out, v_in, v_out, cyclic, params, out,
                                       cuda);
        }
    }

    bool staged() const { return chunk.k.data != nullptr; }

    void commit(CudaExecutionView cuda) {
        if (ragged)
            relative_attention_commit_cyclic_kv(chunk, ragged->view(), in.positions, cyclic, cuda);
        else
            relative_attention_commit_cyclic_kv(chunk, in.positions, valid, cyclic, cuda);
    }

    std::size_t useful_bytes(bool committing) const {
        const auto n                 = std::size_t(work.columns);
        const auto s                 = std::size_t(work.sequences);
        const auto kv_per_token      = 2 * KH * (work.fp8 ? D + 1 : 2 * D);
        const auto sequence_metadata = ragged ? 4 * (n + 2 * s) : 4 * s;
        if (committing) {
            const auto surviving = std::accumulate(
                work.lengths.begin(), work.lengths.end(), std::size_t{0},
                [](std::size_t total, int length) { return total + std::min(length, Window); });
            // Commit reads and writes only surviving vectors, and never reads convolution history.
            const auto commit_metadata = (ragged ? 8 : 4) * s;
            return (2 * kv_per_token + 4) * surviving + commit_metadata + 8 * s;
        }

        // Every prepared chunk token is observable, including the first conv that reads
        // three history taps per live sequence. State/snapshot bytes are charged once.
        const auto activations = (4 * QH * (D + R) + 4 * C + (dynamic ? 16 * KH : 0)) * n;
        const auto weights     = 2 * (QH * (D + R) + 8 * C);
        const auto conv_state  = 4 * C * 3 * (s + (work.snapshot ? n : s));
        const auto sink_bytes  = sink ? 2 * (n * (sink + QH) + sink * QH) : 0;
        const auto positions   = params.use_log_scaling || !staged() ? 4 * n : 0;
        const auto selectors =
            (work.snapshot ? 12 : 16) * s + (work.paged ? 4 * s + 8 * page_entries
                                             : staged() ? 0
                                                        : 8 * s);
        return activations + weights + kv_per_token * n + conv_state + sink_bytes + positions +
               sequence_metadata + selectors;
    }

    const Workload& work;
    Device device;
    KvCacheStorage storage;
    int sink;
    bool dynamic, edge;
    ops::RelativeAttentionPrepareParams params;

private:
    Tensor empty(DType dtype, std::initializer_list<int> dims) {
        const auto size = Tensor::metadata(dtype, dims).bytes();
        auto& buffer    = buffers.emplace_back(device, size);
        return Tensor(buffer.span(), dtype, dims);
    }

    template <class Word>
    Tensor upload(const std::vector<Word>& values, DType dtype, std::initializer_list<int> dims) {
        auto tensor = empty(dtype, dims);
        buffers.back().copy_from_host(values.data(), values.size() * sizeof(Word));
        return tensor;
    }

    Tensor bf16(std::initializer_list<int> dims, bool kv_input = false, bool zero = false) {
        const auto count = Tensor::metadata(DType::BF16, dims).numel();
        std::vector<__nv_bfloat16> words(count);
        for (std::int64_t i = 0; i < count; ++i) {
            const auto hash = static_cast<unsigned>(i) * 2654435761u + 97u * buffers.size();
            float value     = float(int((hash >> 13) % 547) - 273) / 128.0f;
            if (zero) value = 0;
            if (edge && kv_input && i % C < 4 * D) {
                const float limit = dynamic ? 1344.0f : 1000.0f;
                const int d       = i % D;
                switch (i % C / D) {
                case 0:
                    value = 0;
                    break;
                case 1:
                    value = d == 0 ? limit : d == 1 ? -limit : 0;
                    break;
                case 2:
                    value = d == 0 ? limit + 8 : d == 1 ? -limit : 0;
                    break;
                case 3:
                    value = d == 0 ? std::ldexp(1.0f, -130) : 0;
                    break;
                }
            }
            words[i] = __float2bfloat16_rn(value);
        }
        return upload(words, DType::BF16, dims);
    }

    std::vector<DeviceBuffer> buffers;
    DeviceBuffer state, cache_buffer;
    std::size_t source_bytes = 0, page_entries = 0;
    bool restore_source = false;
    std::optional<r::RaggedDeviceFixture> ragged;
    std::optional<tool::PackedKVFixture> pages;
    ops::RelativeAttentionPrepareInputs in;
    ops::RelativeAttentionPrepareOutputs out;
    Tensor valid, rows, snapshot_slots, k_snapshots, v_snapshots;
    StateFieldView k_in, v_in, k_out, v_out;
    CyclicKVCacheLayerView cyclic{};
    ops::RelativeAttentionChunkKV chunk;
    PagedKVBatchLayerView paged{};
};

void run(const Options& options) {
    auto& device = bench::device_context();
    m::hardware();
    bench::L2FlushBuffer flush(options.measure.flush_bytes);
    m::Csv csv(options.measure.csv_out, "layout,phase,segments,sink,tau,state_form,data,logical_"
                                        "bytes,logical_gb_s,sustained_pct,destination");
    auto measure = [&](const Workload& w) {
        Fixture fixture(options, w, device.device());
        auto measure_op = [&](bool committing) {
            if (committing) {
                fixture.restore(device.execution_view());
                fixture.launch(device.execution_view());
                device.synchronize();
            }
            const auto bytes = fixture.useful_bytes(committing);
            for (const auto& exec : m::selections(options.measure.execution, "graph", "eager"))
                for (const auto& cache : m::selections(options.measure.cache, "cold", "warm")) {
                    const auto timing = m::measure(
                        [&](CudaExecutionView cuda) {
                            if (committing)
                                fixture.commit(cuda);
                            else
                                fixture.launch(cuda);
                        },
                        [&](CudaExecutionView cuda) {
                            if (!committing) fixture.restore(cuda);
                        },
                        flush, device.execution_view(), options.measure, exec, cache);
                    if (timing.kernels != 1 || timing.nodes != 3)
                        throw std::runtime_error("prepare and commit must each capture one kernel");
                    const double gb_s = bytes / (timing.gpu.median_us * 1000.0);
                    csv.write(
                        {.op            = committing ? "relative_attention_commit_cyclic_kv"
                                                     : "relative_attention_prepare",
                         .format        = w.fp8 ? "fp8" : "bf16",
                         .policy        = fixture.dynamic ? "dynamic" : "static",
                         .n             = D,
                         .k             = R,
                         .conv_channels = C,
                         .qk_heads      = QH,
                         .value_heads   = KH,
                         .tokens        = w.phase == "ragged" ? w.columns : w.width,
                         .batch         = w.sequences},
                        timing, exec, cache,
                        std::string(w.paged ? "full" : "swa") + ',' + w.phase + ',' +
                            (w.phase == "ragged" ? r::format_segments(w.lengths) : "-") + ',' +
                            std::to_string(fixture.sink) + ',' +
                            std::to_string(fixture.params.use_log_scaling) + ',' +
                            (w.snapshot ? "snapshot"
                                        : std::string(r::state_form_name(options.state_form))) +
                            ',' + (fixture.edge ? "edge" : "spread") + ',' + std::to_string(bytes) +
                            ',' + bench::format_metric(gb_s) + ',' +
                            bench::format_metric(100 * gb_s / bench::kRoofline.sustained_read_gbs) +
                            ',' +
                            (w.paged                           ? "paged"
                             : fixture.staged() && !committing ? "chunk"
                                                               : "cyclic"));
                }
        };
        if (options.phase != "commit") measure_op(false);
        if (fixture.staged()) measure_op(true);
    };

    for (const auto& mode : m::selections(options.mode, "swa", "full"))
        for (const auto& format : m::selections(options.format, "bf16", "fp8")) {
            if (mode == "full" && options.phase == "commit") continue;
            auto batched = [&](const std::string& phase, bool snapshot, int batch, int width) {
                batch = options.batch.value_or(batch);
                width = phase == "decode" ? 1 : options.width.value_or(width);
                if (std::int64_t(batch) * width > std::numeric_limits<int>::max())
                    throw std::invalid_argument("B*W exceeds I32");
                measure({mode == "full", format == "fp8", snapshot, width, batch, batch * width,
                         phase, r::SegmentSpec(batch, width)});
            };
            if (options.phase == "decode" || options.phase == "all") batched("decode", false, 1, 1);
            for (const auto* phase : {"direct", "verify"}) {
                if (options.phase != phase && options.phase != "all" &&
                    !(options.phase == "commit" && std::string_view(phase) == "direct"))
                    continue;
                const bool snapshot = std::string_view(phase) == "verify";
                if (options.batch && options.width) {
                    batched(phase, snapshot, 8, 8);
                } else {
                    for (auto [b, w] : {std::pair{8, 8}, {3, 2}, {5, 3}, {8, 7}, {13, 13}, {1, 16}})
                        batched(phase, snapshot, b, w);
                }
            }
            if (options.phase == "ragged" || options.phase == "all" || options.phase == "commit")
                for (const auto& lengths : options.ragged)
                    measure({mode == "full", format == "fp8", false, 0,
                             static_cast<int>(lengths.size()),
                             std::accumulate(lengths.begin(), lengths.end(), 0), "ragged",
                             lengths});
        }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            help();
            return 0;
        }
        run(parse(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
