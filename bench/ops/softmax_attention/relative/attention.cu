#include "ninfer/ops/relative_attention.h"
#include "ops/softmax_attention/relative/attention/plan.h"

#include "ops/op_measurement.cuh"
#include "ops/ragged_workload.h"
#include "../../../../tools/support/paged_kv_fixture.h"

#include <array>
#include <numeric>
#include <optional>

namespace {

using namespace ninfer;
using namespace ninfer::ops;
namespace bench   = ninfer::bench;
namespace measure = bench::measurement;

constexpr AttentionHeadGeometry kGeometry{128, 24, 8};
constexpr auto kDim     = kGeometry.head_dim;
constexpr auto kHeads   = kGeometry.query_heads;
constexpr auto kKvHeads = kGeometry.kv_heads;
constexpr int kWindow = 1024, kRel = 16;
constexpr float kScale = 0.08838834764831845f;

std::size_t split_traffic(bool swa, bool fp8, const relative::DecodePlan& plan, int history) {
    if (plan.splits == 1) return 0;
    const int splits = relative::active_decode_splits(history, plan.splits, plan.streaming_splits);
    const int tile   = !swa        ? relative::kKeyTile
                       : plan.simt ? relative::kSwaDecodeKeyTile
                       : fp8       ? relative::kKeyTile
                                   : relative::kSwaVerifyKeyTile;
    const int first_query = history - plan.width;
    const int first_tile  = swa ? std::max(0, first_query - kWindow + 1) / tile : 0;
    const int tiles       = (history - 1) / tile + 1 - first_tile;
    std::size_t active    = 0;
    for (int p = first_query; p < history; ++p) {
        const int lo = (swa ? std::max(0, p - kWindow + 1) / tile : 0) - first_tile;
        const int hi = p / tile + 1 - first_tile;
        for (int split = 0; split < splits; ++split) {
            const auto interval = relative::split_tile_interval(tiles, splits, split);
            active += std::max(lo, interval.first) < std::min(hi, interval.end);
        }
    }
    // Every split writes an LSE. Only splits intersecting a row's visible keys write
    // an output vector. The combine kernel reads the same logical operands.
    const auto elements = std::size_t(plan.width) * splits + active * kDim;
    return 2 * sizeof(float) * kHeads * plan.batch * elements;
}

struct Options {
    measure::Options timing;
    std::string mode = "swa", phase = "decode";
    KvCacheStorage storage = KvCacheStorage::BFloat16;
    int batch = 0, width = 0, history = 0, prefix = 0;
    std::vector<bench::ragged::SegmentSpec> partitions{{4096}};
    std::optional<ContextBucket> bucket;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        if (flag == "--help") {
            std::cout
                << "Usage: ninfer_relative_softmax_attention_bench\n"
                   "  --mode swa|full --kv-format bf16|fp8 --phase decode|verify|ragged\n"
                   "  --batch N --width W --kv-len L --bucket 2k|8k|16k|32k|64k|128k|256k|512k|1m\n"
                   "  --segments '4096;4x1024;313,977,1217,1589' --prefix N\n";
            measure::help();
            std::exit(0);
        }
        const auto value = measure::take(argc, argv, i);
        if (flag == "--mode")
            o.mode = value;
        else if (flag == "--phase")
            o.phase = value;
        else if (flag == "--kv-format")
            o.storage = kv_cache_storage_from_option(value);
        else if (flag == "--batch")
            o.batch = measure::integer(value);
        else if (flag == "--width")
            o.width = measure::integer(value, 1, 16);
        else if (flag == "--kv-len")
            o.history = measure::integer(value, 1, int(context_bucket_tokens(ContextBucket::M1)));
        else if (flag == "--prefix")
            o.prefix = measure::integer(value, 0);
        else if (flag == "--segments")
            o.partitions = bench::ragged::parse_ragged_specs(value);
        else if (flag == "--bucket")
            o.bucket = parse_context_bucket(value);
        else if (!measure::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option: " + std::string(flag));
    }
    if ((o.mode != "swa" && o.mode != "full") ||
        (o.phase != "decode" && o.phase != "verify" && o.phase != "ragged") ||
        (o.storage != KvCacheStorage::BFloat16 &&
         o.storage != KvCacheStorage::Fp8E4m3fnGroup128Ue8m0))
        throw std::invalid_argument("invalid mode, phase, or KV format");
    if (!o.batch) o.batch = o.phase == "verify" ? 8 : 1;
    if (!o.width) o.width = o.phase == "verify" ? 8 : 1;
    if (!o.history) o.history = o.mode == "swa" ? kWindow : 32768;
    return o;
}

KVPageGeometry record_schema(bool swa, KvCacheStorage storage) {
    const auto schema = paged_kv_storage_layout(storage, kDim);
    KVPageGeometry record{
        static_cast<unsigned>(swa ? kWindow : kPagedKVPageSize),
        {{schema.key.data_dtype, kDim, kKvHeads}, {schema.value.data_dtype, kDim, kKvHeads}}};
    if (schema.key.has_scale()) {
        record.planes.push_back({DType::U8, 1, kKvHeads});
        record.planes.push_back({DType::U8, 1, kKvHeads});
    }
    return record;
}

template <class Word>
__global__ void init_kv(unsigned char* data, std::size_t stride, std::size_t elements,
                        std::size_t count, unsigned seed) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        unsigned u = (unsigned(i) ^ seed) * 0x7feb352du;
        u          = (u ^ (u >> 16)) * 0x846ca68bu;
        auto* row  = reinterpret_cast<Word*>(data + (i / elements) * stride);
        if constexpr (sizeof(Word) == 1)
            row[i % elements] = (u % 100) | ((u >> 13) & 128);
        else
            row[i % elements] = Word(float(int(u % 1024) - 512) / 512.0f);
    }
}

class Benchmark {
public:
    explicit Benchmark(const Options& o)
        : options(o), flush(o.timing.flush_bytes),
          csv(o.timing.csv_out, "phase,segments,kv_len,bucket,kv_bytes,logical_bytes,logical_gb_s,"
                                "sustained_kv_pct,useful_gflops,bf16_tc_pct") {}

    void run(std::span<const int> lengths) {
        const auto cuda        = bench::device_context().execution_view();
        const auto device      = cuda.device();
        const bool swa         = options.mode == "swa";
        const bool ragged      = options.phase == "ragged";
        const bool fp8         = options.storage != KvCacheStorage::BFloat16;
        const int batch        = ragged ? static_cast<int>(lengths.size()) : options.batch;
        const auto max_history = ragged ? std::int64_t(options.prefix) +
                                              *std::max_element(lengths.begin(), lengths.end())
                                        : options.history;
        if (max_history > context_bucket_tokens(ContextBucket::M1))
            throw std::invalid_argument("history exceeds M1");
        const int history = static_cast<int>(max_history);
        const auto bucket = options.bucket.value_or(select_context_bucket(history));
        if (history > int(context_bucket_tokens(bucket)) || (!ragged && options.width > history))
            throw std::invalid_argument("history must cover width and fit its bucket");
        const int width =
            ragged ? std::accumulate(lengths.begin(), lengths.end(), 0) : options.width;
        const int tensor_batch = ragged ? 1 : batch;
        const int sms          = cuda.multiprocessor_count();

        std::vector<int> positions;
        std::optional<bench::ragged::RaggedDeviceFixture> ragged_layout;
        if (ragged) {
            ragged_layout.emplace(lengths, device);
            for (const int length : lengths)
                for (int i = 0; i < length; ++i) positions.push_back(options.prefix + i);
        } else {
            positions.resize(std::size_t(batch) * width);
            for (int b = 0; b < batch; ++b)
                std::iota(positions.begin() + b * width, positions.begin() + (b + 1) * width,
                          history - width);
        }

        const int logical_pages = (history + kPagedKVPageSize - 1) / kPagedKVPageSize;
        std::vector<int> ids(batch), physical;
        std::iota(ids.begin(), ids.end(), 0);
        int records = 0;
        for (int b = 0; b < batch; ++b) {
            const int live =
                swa ? 1 : (ragged ? (options.prefix + lengths[b] + 63) / 64 : logical_pages);
            for (int j = 0; j < (swa ? 1 : logical_pages); ++j)
                physical.push_back(j < live ? records++ : -1);
        }
        tool::PackedKVFixture cache(device, record_schema(swa, options.storage), records);
        const auto init = [&]<class Word>(int index, unsigned seed) {
            const auto& region = cache.layout().planes[index].tensor.region;
            auto* address      = static_cast<unsigned char*>(cache.storage().data) +
                            cache.physical_offset(0) + region.offset;
            const auto elements = region.bytes / sizeof(Word);
            init_kv<Word><<<sms * 8, 256, 0, cuda.stream()>>>(address, cache.record_stride(),
                                                              elements, elements * records, seed);
            CUDA_CHECK(cudaGetLastError());
        };
        if (fp8) {
            init.template operator()<std::uint8_t>(0, 172);
            init.template operator()<std::uint8_t>(1, 529);
            for (int i = 2; i < 4; ++i) {
                const auto& region = cache.layout().planes[i].tensor.region;
                auto* base         = static_cast<unsigned char*>(cache.storage().data) +
                             cache.physical_offset(0) + region.offset;
                CUDA_CHECK(cudaMemset2DAsync(base, cache.record_stride(), 120, region.bytes,
                                             records, cuda.stream()));
            }
        } else {
            init.template operator()<__nv_bfloat16>(0, 172);
            init.template operator()<__half>(1, 529);
        }
        const auto offsets = cache.table_offsets(physical);
        DeviceBuffer d_offsets(device, offsets.size() * sizeof(std::int64_t));
        d_offsets.copy_from_host(offsets.data(), offsets.size() * sizeof(std::int64_t));
        auto d_rows = measure::make_i32(ids), d_pos = measure::make_i32(positions);
        auto d_valid = measure::make_i32(std::vector<int>(batch, width));
        const Tensor table_rows(d_rows.span(), DType::I32, {batch});
        const Tensor table(d_offsets.span(), DType::I64,
                           swa ? std::initializer_list<int>{batch}
                               : std::initializer_list<int>{logical_pages, batch});
        CyclicKVCacheLayerView cyclic;
        PagedKVBatchLayerView paged;
        if (swa) {
            const auto field = [&](int index) -> StateFieldView {
                const auto region = cache.layout().planes[index].tensor;
                return {{cache.storage(), table}, {region, 1, region.region.bytes}};
            };
            cyclic = {.k               = field(0),
                      .v               = field(1),
                      .k_scales        = fp8 ? field(2) : StateFieldView{},
                      .v_scales        = fp8 ? field(3) : StateFieldView{},
                      .capacity        = kWindow,
                      .padded_capacity = kWindow,
                      .num_kv_heads    = kKvHeads,
                      .head_dim        = kDim,
                      .storage         = options.storage};
        } else {
            paged = {cache.plane(0, table),
                     cache.plane(1, table),
                     fp8 ? cache.plane(2, table) : PagedPlaneView{},
                     fp8 ? cache.plane(3, table) : PagedPlaneView{},
                     kDim,
                     kKvHeads,
                     options.storage};
        }

        RelativeAttentionChunkKV chunk{.storage = options.storage};
        std::array<DeviceBuffer, 4> encoded;
        if (swa && (ragged || width > 1)) {
            const auto schema     = paged_kv_storage_layout(options.storage, kDim);
            const auto vectors    = positions.size() * kKvHeads;
            const auto init_chunk = [&]<class Word>(int plane, DType dtype, unsigned seed) {
                const auto elements = vectors * kDim;
                encoded[plane]      = bench::make_buffer(elements * sizeof(Word));
                init_kv<Word><<<sms * 8, 256, 0, cuda.stream()>>>(
                    static_cast<unsigned char*>(encoded[plane].data()), elements * sizeof(Word),
                    elements, elements, seed);
                CUDA_CHECK(cudaGetLastError());
                return Tensor(encoded[plane].span(), dtype, {kDim, kKvHeads, width, tensor_batch});
            };
            if (fp8) {
                chunk.k =
                    init_chunk.template operator()<std::uint8_t>(0, schema.key.data_dtype, 373);
                chunk.v =
                    init_chunk.template operator()<std::uint8_t>(1, schema.value.data_dtype, 719);
                for (int i : {2, 3}) {
                    encoded[i] = bench::make_buffer(vectors);
                    CUDA_CHECK(cudaMemsetAsync(encoded[i].data(), 120, vectors, cuda.stream()));
                }
                chunk.k_scales =
                    Tensor(encoded[2].span(), DType::U8, {1, kKvHeads, width, tensor_batch});
                chunk.v_scales =
                    Tensor(encoded[3].span(), DType::U8, {1, kKvHeads, width, tensor_batch});
            } else {
                chunk.k =
                    init_chunk.template operator()<__nv_bfloat16>(0, schema.key.data_dtype, 373);
                chunk.v = init_chunk.template operator()<__half>(1, schema.value.data_dtype, 719);
            }
        }

        const auto rows   = positions.size() * kHeads;
        auto q            = bench::make_bf16(rows * kDim, 927);
        auto r            = bench::make_bf16(rows * kRel, 418);
        auto pr           = bench::make_bf16(kRel * kWindow, 862);
        auto g            = bench::make_bf16(rows * kDim, 902);
        auto s            = bench::make_bf16(rows, 302);
        auto d_out        = bench::make_buffer(rows * kDim * 2);
        const auto vector = {kDim, kHeads, width, tensor_batch};
        RelativeAttentionQueryInputs in{
            Tensor(q.span(), DType::BF16, vector),
            Tensor(r.span(), DType::BF16, {kRel, kHeads, width, tensor_batch}),
            Tensor(pr.span(), DType::BF16, {kRel, kWindow}),
            Tensor(g.span(), DType::BF16, vector),
            Tensor(s.span(), DType::BF16, {kHeads, width, tensor_batch}),
            Tensor(d_pos.span(), DType::I32, {width, tensor_batch})};
        Tensor out(d_out.span(), DType::BF16, vector);
        const Tensor valid(d_valid.span(), DType::I32, {batch});
        const auto bytes =
            ragged ? relative_softmax_attention_ragged_workspace_capacity_bytes(
                         kGeometry, options.storage, swa, bucket, width, width, batch, batch, sms)
            : swa ? relative_softmax_attention_swa_workspace_capacity_bytes(
                        kGeometry, options.storage, kWindow, bucket, batch, width, width, sms)
                  : relative_softmax_attention_full_workspace_capacity_bytes(
                        kGeometry, options.storage, bucket, batch, width, width, sms);
        auto scratch = bench::make_buffer(bytes);
        MemoryArena arena(scratch.span());

        const auto launch = [&](CudaExecutionView ex) {
            if (ragged) {
                const auto& view = ragged_layout->view();
                if (swa)
                    relative_softmax_attention(in, view, kGeometry, kWindow, kScale, cyclic, chunk,
                                               bucket, arena, out, ex);
                else
                    relative_softmax_attention(in, view, table_rows, kGeometry, kScale, paged,
                                               bucket, arena, out, ex);
            } else if (swa) {
                relative_softmax_attention(in, valid, kGeometry, kWindow, kScale, cyclic, chunk,
                                           bucket, arena, out, ex);
            } else {
                relative_softmax_attention(in, valid, table_rows, kGeometry, kScale, paged, bucket,
                                           arena, out, ex);
            }
        };
        launch(cuda);
        const auto retained      = [&](int len) { return swa ? std::min(kWindow, len) : len; };
        const auto chunk_history = [](int start) { return std::min(kWindow - 1, start); };
        const std::size_t kv_tokens =
            ragged ? std::accumulate(lengths.begin(), lengths.end(), std::size_t(0),
                                     [&](std::size_t n, int l) {
                                         return n + (swa ? chunk_history(options.prefix) + l
                                                         : options.prefix + l);
                                     })
                   : std::size_t(batch) * (swa && width > 1 ? chunk_history(history - width) + width
                                                            : retained(history));
        const std::size_t kv_bytes =
            kv_tokens * kKvHeads *
            paged_kv_storage_layout(options.storage, kDim).physical_bytes_per_token_head();
        double flops = 0;
        for (const auto p : positions) {
            const auto visible = swa ? std::min(kWindow, p + 1) : p + 1;
            flops += double(kHeads) * (4 * kDim * visible + 2 * kRel * std::min(kWindow, visible));
        }
        const std::size_t projection_bytes =
            std::min(history, kWindow) * kRel * sizeof(std::uint16_t);
        const bool cyclic_history     = ragged ? options.prefix > 0 : width == 1 || history > width;
        const std::size_t table_bytes = swa ? (cyclic_history ? batch * sizeof(std::int64_t) : 0)
                                            : records * sizeof(std::int64_t) + d_rows.size();
        std::size_t scratch_traffic   = 0;
        if (!ragged) {
            const auto plan =
                relative::select_decode(swa, options.storage, bucket, batch, width, sms);
            scratch_traffic = split_traffic(swa, fp8, plan, history);
        } else if (batch > 1) {
            scratch_traffic =
                2 * sizeof(relative::RaggedTile) * relative::ragged_tiles(width, batch) +
                (batch + 1) * sizeof(int);
        }
        const auto logical = kv_bytes + q.size() + r.size() + projection_bytes + g.size() +
                             s.size() + d_out.size() + d_pos.size() + table_bytes +
                             (ragged ? 0 : d_valid.size()) + scratch_traffic;
        measure::Row row{.op              = "relative_softmax_attention",
                         .format          = kv_cache_profile_info(options.storage).name,
                         .policy          = options.mode,
                         .n               = kDim,
                         .k               = kRel,
                         .qk_heads        = kHeads,
                         .value_heads     = kKvHeads,
                         .tokens          = width,
                         .batch           = batch,
                         .workspace_bytes = bytes,
                         .workspace_peak  = arena.peak_used()};
        const auto tag = ragged ? bench::ragged::format_segments(lengths) : "-";
        for (const auto& execution :
             measure::selections(options.timing.execution, "graph", "eager"))
            for (const auto& state : measure::selections(options.timing.cache, "cold", "warm")) {
                const auto result = measure::measure(
                    launch, [](CudaExecutionView) {}, flush, cuda, options.timing, execution,
                    state);
                const double us   = result.gpu.median_us;
                const auto metric = [](double x) { return bench::format_metric(x, 3, ""); };
                csv.write(row, result, execution, state,
                          options.phase + ',' + tag + ',' + std::to_string(history) + ',' +
                              std::string(context_bucket_name(bucket)) + ',' +
                              std::to_string(kv_bytes) + ',' + std::to_string(logical) + ',' +
                              metric(logical / (us * 1000)) + ',' +
                              metric(kv_bytes / (us * 10 * bench::kRoofline.sustained_read_gbs)) +
                              ',' + metric(flops / 1e9) + ',' +
                              metric(flops / (us * 1e4 * bench::kRoofline.bf16_f32_tflops)));
            }
    }

private:
    const Options& options;
    bench::L2FlushBuffer flush;
    measure::Csv csv;
};

} // namespace

int main(int argc, char** argv) try {
    const auto options = parse(argc, argv);
    measure::hardware();
    Benchmark bench(options);
    if (options.phase == "ragged") {
        for (const auto& lengths : options.partitions) bench.run(lengths);
    } else {
        bench.run({});
    }
    return 0;
} catch (const std::exception& e) {

    std::cerr << e.what() << '\n';
    return 1;
}
