#include "ragged_fixture.cuh"
#include "ops/context_bucket_workload.h"
#include "common/gpu_activity.cuh"
#include "../../../../tools/support/kpool_indexer_fixture.h"

using namespace ninfer;
using namespace ninfer::bench;
namespace kp = ninfer::tool::kpool;

namespace {
constexpr const char* indexer_columns = "latent_storage,end,live_queries,scored_queries,key_"
                                        "payload_bytes,query_payload_bytes,output_bytes,"
                                        "score_flops,key_payload_GBps,score_TFLOps,"
                                        "metadata_bytes,minimum_io_bytes,score_io_bytes,"
                                        "candidate_io_bytes,prepared_query_io_bytes,"
                                        "pipeline_bytes,minimum_io_GBps,pipeline_GBps,"
                                        "tensor_peak_TFLOps,tensor_peak_pct,"
                                        "minimum_io_read_reference_pct,roofline_lower_us,"
                                        "bucket,length_dist,valid_dist";

template <class Launch>
void report(measurement::Csv& csv, const measurement::Options& options, L2FlushBuffer& flush,
            CudaExecutionView cuda, ops::KpoolMlaStorage storage, KvCacheVectorFormat index_format,
            int tokens, int batch, int layers, int end, ops::ContextBucket bucket,
            std::string_view length_dist, std::string_view validity, std::string_view valid,
            std::span<const int> lengths, std::span<const int> ends, int shared, bool ties,
            const kpool::Partition* partition, std::size_t workspace_bytes, MemoryArena& workspace,
            Launch launch) {
    const auto columns = partition ? tokens : std::uint64_t(tokens) * batch;
    std::uint64_t live = 0, scored = 0, key_bytes = 0, query_bytes = 0,
                  output_bytes   = 8ULL * columns;
    std::uint64_t metadata_bytes = 4ULL * lengths.size();
    std::uint64_t score_io_bytes = 0, candidate_io_bytes = 0;
    double flops = 0;
    for (std::size_t b = 0; b < lengths.size(); ++b) {
        int largest = 0;
        for (int t = 0; t < lengths[b]; ++t) {
            const int count = (ends[b] - lengths[b] + t + 1) / 4;
            ++live;
            output_bytes += 4ULL * std::min(512, count);
            if (count <= 512) continue;
            ++scored;
            largest = std::max(largest, count);
            flops += 8192. * count;
            score_io_bytes += 8ULL * count;
            // Logical write/read volume of exact partition candidates.
            // These normally reuse L2; they are not measured DRAM bytes.
            if (count > 4096) {
                const int parts = (count + 4095) / 4096;
                candidate_io_bytes +=
                    16ULL * ((parts - 1) * 512 + std::min(512, (count - 1) % 4096 + 1));
                if (parts > 8) {
                    const int groups = (parts + 7) / 8;
                    candidate_io_bytes +=
                        16ULL * ((groups - 1) * 512 + std::min(512, (count - 1) % 32768 + 1));
                }
            }
        }
        key_bytes += (index_format == KvCacheVectorFormat::BFloat16 ? 256ULL : 129ULL) * largest;
        metadata_bytes += 4ULL * lengths[b];
        if (largest) metadata_bytes += 4 + 8ULL * ((largest + 15) / 16);
    }
    query_bytes            = scored * (8192 + 128);
    const bool fp8         = index_format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0;
    const auto prepared_io = fp8 ? scored * 2 * (4096 + 128) : 0;
    const auto minimum_io  = key_bytes + query_bytes + output_bytes + metadata_bytes;
    const auto pipeline    = minimum_io + score_io_bytes + candidate_io_bytes + prepared_io;
    const double peak      = fp8 ? 249.75 : 125.89;
    measurement::Row row;
    row.op              = "kpool_indexer";
    row.format          = index_format == KvCacheVectorFormat::BFloat16 ? "bf16" : "fp8";
    row.n               = 32;
    row.k               = 128;
    row.tokens          = tokens;
    row.batch           = batch;
    row.layers          = layers;
    row.valid           = valid;
    row.workspace_bytes = workspace_bytes;
    for (const auto& execution : measurement::selections(options.execution, "graph", "eager"))
        for (const auto& cache : measurement::selections(options.cache, "cold", "warm")) {
            const auto result = measurement::measure(
                launch, [&](auto c) { condition_gpu(c, 200); }, flush, cuda, options, execution,
                cache);
            row.workspace_peak = workspace.peak_used();
            const double us    = result.gpu.median_us;
            csv.write(
                row, result, execution, cache,
                std::string(storage == ops::KpoolMlaStorage::BFloat16 ? "bf16," : "fp8,") +
                    std::to_string(end) + "," + std::to_string(live) + "," +
                    std::to_string(scored) + "," + std::to_string(key_bytes) + "," +
                    std::to_string(query_bytes) + "," + std::to_string(output_bytes) + "," +
                    std::to_string(flops) + "," + std::to_string(key_bytes / us / 1000) + "," +
                    std::to_string(flops / us / 1e6) + "," + std::to_string(metadata_bytes) + "," +
                    std::to_string(minimum_io) + "," + std::to_string(score_io_bytes) + "," +
                    std::to_string(candidate_io_bytes) + "," + std::to_string(prepared_io) + "," +
                    std::to_string(pipeline) + "," + std::to_string(minimum_io / us / 1000) + "," +
                    std::to_string(pipeline / us / 1000) + "," + std::to_string(peak) + "," +
                    std::to_string(flops / us / 1e6 / peak * 100) + "," +
                    std::to_string(minimum_io / us / 257420 * 100) + "," +
                    std::to_string(std::max(flops / (peak * 1e6), minimum_io / 257420.)) + "," +
                    std::string(ops::context_bucket_name(bucket)) + "," + std::string(length_dist) +
                    "," + std::string(validity) +
                    (partition ? partition->csv(result) : kpool::graph_csv(result) + ",,") + ',' +
                    std::to_string(shared) + ',' + (ties ? "ties" : "random"));
        }
}

void run_job(int argc, char** argv, kpool::JobResources& resources) {
    measurement::Options options;
    kpool::PackedOptions packed;
    std::vector<int> tokens{1, 2, 4, 8, 16, 32, 64};
    int batch = 1, end = 65536, layers = 11, active_rows = -1;
    bool ragged = false, inactive = false, ties = false;
    std::optional<int> shared_prefix;
    std::optional<ops::ContextBucket> bucket_override;
    auto length_dist = workload::LengthDist::Uniform;
    std::optional<workload::ValidDist> valid_dist;
    auto storage      = ops::KpoolMlaStorage::BFloat16;
    auto index_format = KvCacheVectorFormat::BFloat16;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--help") {
            std::cout << "Complete Kpool weighted Indexer + exact Top512.\n"
                         "  --tokens N,... --batch B --end E --layers 1|11 --ragged 0|1\n"
                         "  --latent-cache bf16|fp8 --index-cache bf16|fp8\n"
                         "  --shared-prefix N --score-pattern random|ties\n"
                         "  --bucket auto|2k|8k|16k|32k|64k|128k|256k|512k|1m\n"
                         "  --length-dist uniform|skewed-1long|staggered|shared-prefix\n"
                         "  --valid-dist full|ragged|with-inert  overrides --ragged\n"
                         "  --inactive 0|1  zero valid columns in every physical row\n"
                         "  --active-rows 0..B  leading active request rows (default B)\n"
                         "E is the largest requested exclusive end; requires T<=E<=1048576.\n";
            std::cout << "  --jobs-file PATH (one tab-separated argument list per line)\n"
                         "  --ragged-segments L,KxL;... --ragged-histories P,KxP;... (default 0)\n";
            measurement::help();
            return;
        }
        const auto value = measurement::take(argc, argv, i);
        if (measurement::parse(options, flag, value) || packed.parse(flag, value)) continue;
        if (flag == "--tokens")
            tokens = measurement::integers(value);
        else if (flag == "--batch")
            batch = measurement::integer(value);
        else if (flag == "--end")
            end = measurement::integer(value, 1, 1048576);
        else if (flag == "--layers") {
            layers = measurement::integer(value, 1, 11);
            if (layers != 1 && layers != 11) throw std::invalid_argument("layers must be 1 or 11");
        } else if (flag == "--latent-cache") {
            if (value == "bf16")
                storage = ops::KpoolMlaStorage::BFloat16;
            else if (value == "fp8")
                storage = ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0;
            else
                throw std::invalid_argument("latent-cache: bf16|fp8");
        } else if (flag == "--index-cache" && (value == "bf16" || value == "fp8")) {
            index_format = value == "fp8" ? KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0
                                          : KvCacheVectorFormat::BFloat16;
        } else if (flag == "--score-pattern" && (value == "random" || value == "ties"))
            ties = value == "ties";
        else if (flag == "--shared-prefix")
            shared_prefix = measurement::integer(value, 0, kpool::max_context);
        else if (flag == "--bucket")
            bucket_override = workload::parse_bucket_override(value);
        else if (flag == "--length-dist")
            length_dist = workload::parse_length_dist(value);
        else if (flag == "--valid-dist")
            valid_dist = workload::parse_valid_dist(value);
        else if (flag == "--ragged")
            ragged = measurement::integer(value, 0, 1);
        else if (flag == "--inactive")
            inactive = measurement::integer(value, 0, 1);
        else if (flag == "--active-rows")
            active_rows = measurement::integer(value, 0);
        else
            throw std::invalid_argument("unknown option: " + flag);
    }
    const auto partitions = packed.partitions();
    if (!partitions.empty() && (batch != 1 || ragged || inactive || active_rows >= 0 ||
                                valid_dist || length_dist != workload::LengthDist::Uniform))
        throw std::invalid_argument("packed Indexer requires packed metadata");
    if (active_rows < 0) active_rows = batch;
    if (active_rows > batch) throw std::invalid_argument("active rows exceed batch");
    if (inactive) active_rows = 0;
    auto& device = resources.device();
    auto& flush  = resources.flush(options.flush_bytes);
    measurement::Csv csv(options.csv_out, std::string(indexer_columns) +
                                              std::string(kpool::packed_columns) +
                                              ",shared_prefix,score_pattern");
    if (!partitions.empty()) {
        const int shared = shared_prefix.value_or(0);
        for (const auto& partition : partitions) {
            partition.validate_shared_prefix(shared);
            const auto bucket = partition.bucket(bucket_override);
            const auto cuda   = device.execution_view();
            kpool::IndexerFixture f(cuda, partition, layers, shared, storage, index_format, bucket,
                                    ties);
            std::vector<int> ends(partition.segments());
            for (int s = 0; s < partition.segments(); ++s) ends[s] = partition.end(s);
            report(csv, options, flush, cuda, storage, index_format, partition.tokens(),
                   partition.segments(), layers, partition.max_end(), bucket, "packed", "full",
                   "packed", partition.lengths, ends, shared, ties, &partition,
                   f.workspace.storage.size(), f.workspace.arena, [&](auto c) { f.launch(c); });
        }
        return;
    }
    if (shared_prefix && *shared_prefix % kpool::page_tokens)
        throw std::invalid_argument("--shared-prefix must contain whole pages");
    const int max_tokens = *std::max_element(tokens.begin(), tokens.end());
    if (max_tokens > end) throw std::invalid_argument("tokens exceed --end");
    const auto validity =
        valid_dist.value_or(ragged ? workload::ValidDist::Ragged : workload::ValidDist::Full);
    const auto make_shape = [&](int T) {
        auto w = workload::generate_batch_workload(end, T, batch, length_dist, validity);
        kp::IndexerCase shape;
        shape.reserve_ends = w.row_contexts;
        if (length_dist == workload::LengthDist::Uniform)
            std::fill(shape.reserve_ends.begin(), shape.reserve_ends.end(), end);
        shape.ends          = std::move(w.row_contexts);
        shape.valid         = std::move(w.valid_columns);
        shape.shared_prefix = shared_prefix.value_or(w.shared_prefix_tokens);
        int max_visible     = 0;
        for (int b = 0; b < batch; ++b) {
            if (!valid_dist && ragged) {
                shape.valid[b] = b == batch - 1 ? 0 : std::max(1, T - b);
                shape.ends[b]  = std::max(shape.valid[b], shape.ends[b] - T + shape.valid[b]);
            }
            if (b >= active_rows) shape.valid[b] = 0;
            if (!shape.valid[b]) {
                shape.ends[b] = 0;
                continue;
            }
            max_visible = std::max(max_visible, shape.ends[b]);
            if (shared_prefix && *shared_prefix > shape.ends[b] - shape.valid[b])
                throw std::invalid_argument("--shared-prefix exceeds the live history");
            shape.shared_prefix =
                std::min(shape.shared_prefix, (shape.ends[b] - shape.valid[b]) / 64 * 64);
        }
        shape.bucket = bucket_override.value_or(ops::select_context_bucket(max_visible));
        if (ops::context_bucket_tokens(*shape.bucket) < std::uint32_t(max_visible))
            throw std::invalid_argument("--bucket is smaller than the live history");
        if (!max_visible) shape.shared_prefix = 0;
        return shape;
    };
    auto backing = make_shape(max_tokens);
    for (const int T : tokens) {
        const auto shape      = make_shape(T);
        backing.shared_prefix = std::min(backing.shared_prefix, shape.shared_prefix);
        for (int b = 0; b < batch; ++b)
            backing.reserve_ends[b] = std::max(backing.reserve_ends[b], shape.reserve_ends[b]);
    }
    kp::IndexerFixture f(device.execution_view(), max_tokens, batch, end, layers, false, storage,
                         backing, true, index_format);
    if (ties) {
        std::fill(f.coeff.host.begin(), f.coeff.host.end(), 0.F);
        f.coeff.upload();
    }
    for (int T : tokens) {
        const auto shape = make_shape(T);
        f.bucket         = *shape.bucket;
        f.resize_queries(T, device.execution_view());
        for (int b = 0; b < batch; ++b) {
            f.valid.at(b)     = shape.valid[b];
            f.selectors.at(b) = shape.valid[b] ? batch - 1 - b : -1;
            for (int t = 0; t < T; ++t)
                f.positions.at(t, b) = t < shape.valid[b] ? shape.ends[b] - shape.valid[b] + t : -1;
        }
        f.positions.upload();
        f.valid.upload();
        f.selectors.upload();
        std::string valid = inactive     ? "inactive"
                            : valid_dist ? std::string(workload::valid_dist_name(validity))
                            : ragged     ? "ragged"
                                         : "dense";
        if (!inactive && active_rows < batch) valid += "-rows" + std::to_string(active_rows);
        report(csv, options, flush, device.execution_view(), storage, index_format, T, batch,
               layers, end, f.bucket, workload::length_dist_name(length_dist),
               workload::valid_dist_name(validity), valid, shape.valid, shape.ends,
               backing.shared_prefix, ties, nullptr, f.workspace_bytes, *f.workspace,
               [&](auto c) { f.launch(c); });
    }
}
} // namespace

int main(int argc, char** argv) { return kpool::run_jobs(argc, argv, run_job); }
