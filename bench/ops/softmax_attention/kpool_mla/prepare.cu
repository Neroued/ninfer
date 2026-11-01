#include "ops/op_measurement.cuh"
#include "common/gpu_activity.cuh"
#include "../../../../tools/support/kpool_prepare_fixture.h"

using namespace ninfer;
using namespace ninfer::bench;
namespace kp = ninfer::tool::kpool;

int main(int argc, char** argv) {
    try {
        measurement::Options options;
        std::vector<int> tokens{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096};
        int batch = 1, phase = 0;
        bool update = false, ragged = false, inplace = false, inactive = false;
        auto projection = kp::Projection::Full;
        auto index_format = KvCacheVectorFormat::BFloat16;
        auto latent_storage = ops::KpoolMlaStorage::BFloat16;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--help") {
                std::cout
                    << "Kpool MLA complete Prepare entry, fixed "
                       "H4096/Dc512/Di128/Rq1536/Hi32.\n"
                       "  --projection full|query-history|history --form record|update\n"
                       "  --tokens N,... --batch 1..8 --phase 0..3 --ragged 0|1 --inplace 0|1\n"
                       "  --inactive 0|1  zero valid columns in every physical row\n"
                       "  --index-cache bf16|fp8 --latent-cache bf16|fp8\n";
                measurement::help();
                return 0;
            }
            const auto v = measurement::take(argc, argv, i);
            if (measurement::parse(options, flag, v)) continue;
            if (flag == "--tokens")
                tokens = measurement::integers(v);
            else if (flag == "--batch")
                batch = measurement::integer(v, 1, 8);
            else if (flag == "--phase")
                phase = measurement::integer(v, 0, 3);
            else if (flag == "--ragged")
                ragged = measurement::integer(v, 0, 1);
            else if (flag == "--inplace")
                inplace = measurement::integer(v, 0, 1);
            else if (flag == "--inactive")
                inactive = measurement::integer(v, 0, 1);
            else if (flag == "--index-cache" && (v == "bf16" || v == "fp8"))
                index_format = v == "fp8" ? KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0
                                          : KvCacheVectorFormat::BFloat16;
            else if (flag == "--latent-cache" && (v == "bf16" || v == "fp8"))
                latent_storage = v == "fp8" ? ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0
                                            : ops::KpoolMlaStorage::BFloat16;
            else if (flag == "--form" && (v == "record" || v == "update"))
                update = v == "update";
            else if (flag == "--projection" &&
                     (v == "full" || v == "query-history" || v == "history"))
                projection = v == "full"      ? kp::Projection::Full
                             : v == "history" ? kp::Projection::History
                                              : kp::Projection::QueryAndHistory;
            else
                throw std::invalid_argument("unknown option/value: " + flag + " " + std::string(v));
        }
        measurement::hardware();
        auto& device = device_context();
        kp::Weights weights(device.device(), projection);
        L2FlushBuffer flush(options.flush_bytes);
        measurement::Csv csv(
            options.csv_out,
            "index_storage,projection,form,phase,live_columns,weight_bytes,read_payload_bytes,"
            "write_payload_bytes,payload_bytes,projection_flops,"
            "physical_projection_flops,payload_GBps,"
            "projection_TFLOps");
        for (int T : tokens) {
            kp::Fixture f(weights, device.execution_view(), T, batch, update, ragged, false,
                          inplace, phase, index_format, latent_storage);
            if (inactive) {
                for (int b = 0; b < batch; ++b) {
                    f.valid.at(b)          = 0;
                    f.selectors.at(b)      = -1;
                    f.offsets.at(b)        = -1;
                    f.before_offsets.at(b) = -1;
                    f.after_offsets.at(b)  = -1;
                    for (int t = 0; t < T; ++t) f.positions.at(t, b) = -1;
                }
                for (auto* v : {&f.valid, &f.selectors, &f.offsets, &f.positions}) v->upload();
                f.before_offsets.upload();
                f.after_offsets.upload();
            }
            const std::size_t M = std::size_t(T) * batch;
            std::size_t live = 0, pools = 0, tail_read = 0, tail_write = 0;
            for (int b = 0; b < batch; ++b) {
                const int n = f.valid.at(b);
                live += n;
                if (n) {
                    const int completed = (f.bases[b] + n) / 4 - f.bases[b] / 4;
                    pools += completed;
                    if (update || completed) tail_read += f.bases[b] % 4 * 512;
                    if (update) tail_write += 1536;
                }
            }
            // One logical payload pass. Metadata, staging, padding compute and repeated loads
            // are separate costs; this is not a DRAM traffic counter or a residency claim.
            const std::size_t reads = live ? weights.weight.view.bytes() + live * 8192 + tail_read +
                                                 (512 + weights.q + 256 + 512) * 2
                                           : 0;
            const std::size_t writes =
                live * ((latent_storage == ops::KpoolMlaStorage::BFloat16 ? 1024 : 516) +
                        (update ? 0 : 512)) +
                M * (weights.q * 2 + weights.heads * 4) +
                pools * (index_format == KvCacheVectorFormat::BFloat16 ? 256 : 129) + tail_write;
            const std::size_t payload   = reads + writes;
            const double flops          = 2. * weights.rows * 4096 * live;
            const double physical_flops = 2. * weights.rows * 4096 * M;
            measurement::Row row;
            row.op              = "kpool_mla_prepare";
            row.format          = latent_storage == ops::KpoolMlaStorage::BFloat16 ? "bf16" : "fp8";
            row.n               = weights.rows;
            row.k               = 4096;
            row.tokens          = T;
            row.batch           = batch;
            row.layers          = 11;
            row.valid           = inactive ? "empty" : ragged ? "ragged" : "dense";
            row.workspace_bytes = f.workspace_bytes;
            for (const auto& execution :
                 measurement::selections(options.execution, "graph", "eager"))
                for (const auto& cache : measurement::selections(options.cache, "cold", "warm")) {
                    auto result        = measurement::measure([&](auto c) { f.launch(c); },
                                                       [&](auto c) {
                                                           f.restore(c);
                                                           condition_gpu(c, 200);
                                                       },
                                                       flush, device.execution_view(), options,
                                                       execution, cache);
                    row.workspace_peak = f.workspace->peak_used();
                    const double us    = result.gpu.median_us;
                    csv.write(row, result, execution, cache,
                              std::string(index_format == KvCacheVectorFormat::BFloat16 ? "bf16,"
                                                                                        : "fp8,") +
                                  kp::name(projection) + "," + (update ? "update" : "record") +
                                  "," + std::to_string(phase) + "," + std::to_string(live) + "," +
                                  std::to_string(weights.weight.view.bytes()) + "," +
                                  std::to_string(reads) + "," + std::to_string(writes) + "," +
                                  std::to_string(payload) + "," + std::to_string(flops) + "," +
                                  std::to_string(physical_flops) + "," +
                                  std::to_string(payload / us / 1000) + "," +
                                  std::to_string(flops / us / 1e6));
                }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
