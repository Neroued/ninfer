#include "ops/op_measurement.cuh"
#include "common/gpu_activity.cuh"
#include "../../../../tools/support/kpool_indexer_fixture.h"

using namespace ninfer;
using namespace ninfer::bench;
namespace kp = ninfer::tool::kpool;

int main(int argc, char** argv) {
    try {
        measurement::Options options;
        std::vector<int> tokens{1, 2, 4, 8, 16, 32, 64};
        int batch = 1, end = 65536, layers = 11, active_rows = -1;
        bool ragged = false, inactive = false;
        auto storage      = ops::KpoolMlaStorage::BFloat16;
        auto index_format = KvCacheVectorFormat::BFloat16;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--help") {
                std::cout
                    << "Complete Kpool weighted Indexer + exact Top512.\n"
                       "  --tokens N,... --batch 1..8 --end E --layers 1|11 --ragged 0|1\n"
                       "  --latent-cache bf16|fp8 --index-cache bf16|fp8\n"
                       "  --inactive 0|1  zero valid columns in every physical row\n"
                       "  --active-rows 0..B  leading active request rows (default B)\n"
                       "E is the last physical query's exclusive end; requires T<=E<=1048576.\n";
                measurement::help();
                return 0;
            }
            const auto value = measurement::take(argc, argv, i);
            if (measurement::parse(options, flag, value)) continue;
            if (flag == "--tokens")
                tokens = measurement::integers(value);
            else if (flag == "--batch")
                batch = measurement::integer(value, 1, 8);
            else if (flag == "--end")
                end = measurement::integer(value, 1, 1048576);
            else if (flag == "--layers") {
                layers = measurement::integer(value, 1, 11);
                if (layers != 1 && layers != 11)
                    throw std::invalid_argument("layers must be 1 or 11");
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
            } else if (flag == "--ragged")
                ragged = measurement::integer(value, 0, 1);
            else if (flag == "--inactive")
                inactive = measurement::integer(value, 0, 1);
            else if (flag == "--active-rows")
                active_rows = measurement::integer(value, 0, 8);
            else
                throw std::invalid_argument("unknown option: " + flag);
        }
        if (active_rows < 0) active_rows = batch;
        if (active_rows > batch) throw std::invalid_argument("active rows exceed batch");
        if (inactive) active_rows = 0;
        measurement::hardware();
        auto& device = device_context();
        L2FlushBuffer flush(options.flush_bytes);
        const int max_tokens = *std::max_element(tokens.begin(), tokens.end());
        if (max_tokens > end) throw std::invalid_argument("tokens exceed --end");
        kp::IndexerFixture f(device.execution_view(), max_tokens, batch, end, layers, ragged,
                             storage, {}, true, index_format);
        measurement::Csv csv(options.csv_out, "latent_storage,end,live_queries,scored_queries,key_"
                                              "payload_bytes,query_payload_bytes,output_bytes,"
                                              "score_flops,key_payload_GBps,score_TFLOps,"
                                              "metadata_bytes,minimum_io_bytes,score_io_bytes,"
                                              "candidate_io_bytes,prepared_query_io_bytes,"
                                              "pipeline_bytes,minimum_io_GBps,pipeline_GBps,"
                                              "tensor_peak_TFLOps,tensor_peak_pct,"
                                              "minimum_io_read_reference_pct,roofline_lower_us");
        for (int T : tokens) {
            f.reset_tokens(T, ragged, device.execution_view());
            if (active_rows < batch) {
                for (int b = active_rows; b < batch; ++b) {
                    f.valid.at(b)     = 0;
                    f.selectors.at(b) = -1;
                    for (int t = 0; t < T; ++t) f.positions.at(t, b) = -1;
                }
                f.positions.upload();
                f.valid.upload();
                f.selectors.upload();
            }
            std::uint64_t live = 0, scored = 0, key_bytes = 0, query_bytes = 0,
                          output_bytes   = 8ULL * T * batch;
            std::uint64_t metadata_bytes = 4ULL * batch;
            std::uint64_t score_io_bytes = 0, candidate_io_bytes = 0;
            double flops = 0;
            for (int b = 0; b < batch; ++b) {
                int largest = 0;
                for (int t = 0; t < f.valid.at(b); ++t) {
                    const int count = (f.positions.at(t, b) + 1) / 4;
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
                            candidate_io_bytes += 16ULL * ((groups - 1) * 512 +
                                                           std::min(512, (count - 1) % 32768 + 1));
                        }
                    }
                }
                key_bytes +=
                    (index_format == KvCacheVectorFormat::BFloat16 ? 256ULL : 129ULL) * largest;
                metadata_bytes += 4ULL * f.valid.at(b);
                if (largest) metadata_bytes += 4 + 8ULL * ((largest + 15) / 16);
            }
            query_bytes            = scored * (8192 + 128);
            const bool fp8         = index_format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0;
            const auto prepared_io = fp8 ? scored * 2 * (4096 + 128) : 0;
            const auto minimum_io  = key_bytes + query_bytes + output_bytes + metadata_bytes;
            const auto pipeline    = minimum_io + score_io_bytes + candidate_io_bytes + prepared_io;
            const double peak      = fp8 ? 249.75 : 125.89;
            measurement::Row row;
            row.op     = "kpool_indexer";
            row.format = index_format == KvCacheVectorFormat::BFloat16 ? "bf16" : "fp8";
            row.n      = 32;
            row.k      = 128;
            row.tokens = T;
            row.batch  = batch;
            row.layers = layers;
            row.valid  = inactive ? "inactive" : ragged ? "ragged" : "dense";
            if (!inactive && active_rows < batch)
                row.valid += "-rows" + std::to_string(active_rows);
            row.workspace_bytes = f.workspace_bytes;
            for (const auto& execution :
                 measurement::selections(options.execution, "graph", "eager"))
                for (const auto& cache : measurement::selections(options.cache, "cold", "warm")) {
                    const auto result = measurement::measure(
                        [&](auto c) { f.launch(c); }, [&](auto c) { condition_gpu(c, 200); }, flush,
                        device.execution_view(), options, execution, cache);
                    row.workspace_peak = f.workspace->peak_used();
                    const double us    = result.gpu.median_us;
                    csv.write(
                        row, result, execution, cache,
                        std::string(storage == ops::KpoolMlaStorage::BFloat16 ? "bf16," : "fp8,") +
                            std::to_string(end) + "," + std::to_string(live) + "," +
                            std::to_string(scored) + "," + std::to_string(key_bytes) + "," +
                            std::to_string(query_bytes) + "," + std::to_string(output_bytes) + "," +
                            std::to_string(flops) + "," + std::to_string(key_bytes / us / 1000) +
                            "," + std::to_string(flops / us / 1e6) + "," +
                            std::to_string(metadata_bytes) + "," + std::to_string(minimum_io) +
                            "," + std::to_string(score_io_bytes) + "," +
                            std::to_string(candidate_io_bytes) + "," + std::to_string(prepared_io) +
                            "," + std::to_string(pipeline) + "," +
                            std::to_string(minimum_io / us / 1000) + "," +
                            std::to_string(pipeline / us / 1000) + "," + std::to_string(peak) +
                            "," + std::to_string(flops / us / 1e6 / peak * 100) + "," +
                            std::to_string(minimum_io / us / 257420 * 100) + "," +
                            std::to_string(std::max(flops / (peak * 1e6), minimum_io / 257420.)));
                }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
