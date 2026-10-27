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
        int batch = 1, end = 65536, layers = 11;
        bool ragged  = false;
        auto storage = ops::KpoolMlaStorage::BFloat16;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--help") {
                std::cout
                    << "Complete BF16 Kpool weighted Indexer + exact Top512.\n"
                       "  --tokens N,... --batch 1..8 --end E --layers 1|11 --ragged 0|1\n"
                       "  --latent-cache bf16|fp8 (index keys are BF16 in both layouts)\n"
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
            } else if (flag == "--ragged")
                ragged = measurement::integer(value, 0, 1);
            else
                throw std::invalid_argument("unknown option: " + flag);
        }
        measurement::hardware();
        auto& device = device_context();
        L2FlushBuffer flush(options.flush_bytes);
        measurement::Csv csv(options.csv_out, "latent_storage,end,live_queries,scored_queries,key_"
                                              "payload_bytes,query_payload_bytes,output_bytes,"
                                              "score_flops,key_payload_GBps,score_TFLOps,"
                                              "metadata_bytes");
        for (int T : tokens) {
            if (T > end) throw std::invalid_argument("tokens exceed --end");
            kp::IndexerFixture f(device.execution_view(), T, batch, end, layers, ragged, storage);
            std::uint64_t live = 0, scored = 0, key_bytes = 0, query_bytes = 0,
                          output_bytes   = 8ULL * T * batch;
            std::uint64_t metadata_bytes = 4ULL * batch;
            double flops                 = 0;
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
                }
                key_bytes += 256ULL * largest;
                metadata_bytes += 4ULL * f.valid.at(b);
                if (largest) metadata_bytes += 4 + 8ULL * ((largest + 15) / 16);
            }
            query_bytes = scored * (8192 + 128);
            measurement::Row row;
            row.op              = "kpool_indexer";
            row.format          = "bf16";
            row.n               = 32;
            row.k               = 128;
            row.tokens          = T;
            row.batch           = batch;
            row.layers          = layers;
            row.valid           = ragged ? "ragged" : "dense";
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
                            std::to_string(metadata_bytes));
                }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
