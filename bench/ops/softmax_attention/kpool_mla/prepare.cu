#include "ops/op_measurement.cuh"
#include "common/gpu_activity.cuh"
#include "../../../../tools/support/kpool_prepare_fixture.h"
#include "prepare_traffic.cuh"

using namespace ninfer;
using namespace ninfer::bench;
namespace kp = ninfer::tool::kpool;

int main(int argc, char** argv) {
    try {
        measurement::Options options;
        std::vector<int> tokens{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096};
        int batch = 1, phase = 0;
        bool update = false, ragged = false, inplace = false, inactive = false;
        int active_rows     = -1;
        auto projection     = kp::Projection::Full;
        auto weight_format  = kp::ProjectionFormat::BFloat16;
        auto index_format   = KvCacheVectorFormat::BFloat16;
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
                       "  --active-rows 0..B  leading active request rows (default B)\n"
                       "  --index-cache bf16|fp8 --latent-cache bf16|fp8\n"
                       "  --weight-format bf16|block-fp8|mxfp8\n";
                measurement::help();
                return 0;
            }
            const auto v = measurement::take(argc, argv, i);
            if (measurement::parse(options, flag, v)) continue;
            if (flag == "--tokens")
                tokens = measurement::integers(v);
            else if (flag == "--weight-format" && (v == "bf16" || v == "block-fp8" || v == "mxfp8"))
                weight_format = v == "bf16"    ? kp::ProjectionFormat::BFloat16
                                : v == "mxfp8" ? kp::ProjectionFormat::Mxfp8
                                               : kp::ProjectionFormat::BlockFp8;
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
            else if (flag == "--active-rows")
                active_rows = measurement::integer(v, 0, 8);
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
        if (active_rows < 0) active_rows = batch;
        if (active_rows > batch) throw std::invalid_argument("active rows exceed batch");
        if (inactive) active_rows = 0;
        measurement::hardware();
        auto& device = device_context();
        kp::Weights weights(device.device(), projection, weight_format);
        L2FlushBuffer flush(options.flush_bytes);
        measurement::Csv csv(options.csv_out,
                             "weight_format,index_storage,projection,form,phase,"
                             "live_columns,weight_bytes,read_payload_bytes,"
                             "write_payload_bytes,payload_bytes,projection_flops,"
                             "physical_projection_flops,payload_GBps,"
                             "projection_TFLOps,workspace_read_bytes,workspace_write_bytes,"
                             "additional_weight_bytes,additional_input_bytes,pipeline_bytes,"
                             "pipeline_GBps,payload_read_reference_pct,pipeline_read_reference_pct,"
                             "pipeline_nominal_pct,matrix_tensor_reference_us,"
                             "matrix_tensor_reference_pct,uses_simt");
        for (int T : tokens) {
            kp::Fixture f(weights, device.execution_view(), T, batch, update, ragged, false,
                          inplace, phase, index_format, latent_storage);
            if (active_rows < batch) {
                for (int b = active_rows; b < batch; ++b) {
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
            const std::size_t reads =
                live ? weights.bytes() + live * 8192 + tail_read + (512 + weights.q + 256 + 512) * 2
                     : 0;
            const std::size_t writes =
                live * ((latent_storage == ops::KpoolMlaStorage::BFloat16 ? 1024 : 516) +
                        (update ? 0 : 512)) +
                M * (weights.q * 2 + weights.heads * 4) +
                pools * (index_format == KvCacheVectorFormat::BFloat16 ? 256 : 129) + tail_write;
            const std::size_t payload = reads + writes;
            const double flops        = 2. * weights.rows * 4096 * live;
            const auto traffic        = kpool_prepare_traffic(f, device.execution_view());
            const auto pipeline       = payload + traffic.workspace_read + traffic.workspace_write +
                                  traffic.repeated_weights + traffic.quantization_input;
            measurement::Row row;
            row.op     = "kpool_mla_prepare";
            row.format = latent_storage == ops::KpoolMlaStorage::BFloat16 ? "bf16" : "fp8";
            row.policy = weight_format == kp::ProjectionFormat::Mxfp8 ? "a8/a16" : "a16";
            row.n      = weights.rows;
            row.k      = 4096;
            row.tokens = T;
            row.batch  = batch;
            row.layers = 11;
            row.valid  = inactive ? "empty" : ragged ? "ragged" : "dense";
            if (!inactive && active_rows < batch)
                row.valid += "-rows" + std::to_string(active_rows);
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
                    csv.write(
                        row, result, execution, cache,
                        std::string(kp::name(weight_format)) + "," +
                            std::string(index_format == KvCacheVectorFormat::BFloat16 ? "bf16,"
                                                                                      : "fp8,") +
                            kp::name(projection) + "," + (update ? "update" : "record") + "," +
                            std::to_string(phase) + "," + std::to_string(live) + "," +
                            std::to_string(weights.bytes()) + "," + std::to_string(reads) + "," +
                            std::to_string(writes) + "," + std::to_string(payload) + "," +
                            std::to_string(flops) + "," +
                            std::to_string(traffic.executed_matrix_flops) + "," +
                            std::to_string(payload / us / 1000) + "," +
                            std::to_string(flops / us / 1e6) + "," +
                            std::to_string(traffic.workspace_read) + "," +
                            std::to_string(traffic.workspace_write) + "," +
                            std::to_string(traffic.repeated_weights) + "," +
                            std::to_string(traffic.quantization_input) + "," +
                            std::to_string(pipeline) + "," + std::to_string(pipeline / us / 1000) +
                            "," +
                            std::to_string(payload / us / 1000 / kRoofline.sustained_read_gbs *
                                           100) +
                            "," +
                            std::to_string(pipeline / us / 1000 / kRoofline.sustained_read_gbs *
                                           100) +
                            "," +
                            std::to_string(pipeline / us / 1000 / kRoofline.memory_gbs * 100) +
                            "," + std::to_string(traffic.matrix_tensor_us) + "," +
                            std::to_string(traffic.matrix_tensor_us / us * 100) + "," +
                            std::to_string(traffic.simt));
                }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
