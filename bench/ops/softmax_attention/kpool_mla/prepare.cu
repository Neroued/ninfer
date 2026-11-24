#include "ragged_fixture.cuh"
#include "common/gpu_activity.cuh"
#include "prepare_traffic.cuh"

#include <climits>

using namespace ninfer;
using namespace ninfer::bench;
namespace kp = ninfer::tool::kpool;

namespace {
constexpr const char* prepare_columns =
    "weight_format,index_storage,projection,form,phase,live_columns,weight_bytes,"
    "read_payload_bytes,write_payload_bytes,payload_bytes,projection_flops,"
    "physical_projection_flops,payload_GBps,projection_TFLOps,workspace_read_bytes,"
    "workspace_write_bytes,additional_weight_bytes,additional_input_bytes,pipeline_bytes,"
    "pipeline_GBps,payload_read_reference_pct,pipeline_read_reference_pct,pipeline_nominal_pct,"
    "matrix_tensor_reference_us,matrix_tensor_reference_pct,uses_simt";

template <class Launch, class Restore>
void report(measurement::Csv& csv, const measurement::Options& options, L2FlushBuffer& flush,
            CudaExecutionView cuda, const kp::Weights& weights, ops::KpoolMlaStorage storage,
            KvCacheVectorFormat index, int tokens, int batch, bool update, std::string valid,
            int active_rows, std::span<const int> lengths, std::span<const int> bases, int phase,
            const std::optional<KpoolPrepareTraffic>& traffic, const kpool::Partition* partition,
            kpool::StateAlias alias, std::size_t workspace_bytes, MemoryArena& workspace,
            Launch launch, Restore restore) {
    std::size_t live = 0, pools = 0, tail_read = 0, tail_write = 0;
    for (std::size_t b = 0; b < lengths.size(); ++b) {
        const int n = lengths[b];
        live += n;
        if (n) {
            const int completed = (bases[b] + n) / 4 - bases[b] / 4;
            pools += completed;
            if (update || completed) tail_read += bases[b] % 4 * 512;
            if (update) tail_write += 1536;
        }
    }
    const bool bf16           = storage == ops::KpoolMlaStorage::BFloat16;
    const bool index_bf16     = index == KvCacheVectorFormat::BFloat16;
    const std::size_t columns = partition ? tokens : std::size_t(tokens) * batch;
    // One logical payload pass. It excludes metadata, staging and repeated loads.
    const std::size_t reads =
        live ? weights.bytes() + live * 8192 + tail_read + (512 + weights.q + 256 + 512) * 2 : 0;
    const std::size_t writes = live * ((bf16 ? 1024 : 516) + (update ? 0 : 512)) +
                               columns * (weights.q * 2 + weights.heads * 4) +
                               pools * (index_bf16 ? 256 : 129) + tail_write;
    const auto payload = reads + writes;
    const double flops = 2. * weights.rows * 4096 * live;
    measurement::Row row;
    row.op     = "kpool_mla_prepare";
    row.format = bf16 ? "bf16" : "fp8";
    row.policy = weights.format == kp::ProjectionFormat::Mxfp8 ? "a8/a16" : "a16";
    row.n      = weights.rows;
    row.k      = 4096;
    row.tokens = tokens;
    row.batch  = batch;
    row.layers = 11;
    row.valid  = std::move(valid);
    if (active_rows >= 0) row.valid += "-rows" + std::to_string(active_rows);
    row.workspace_bytes = workspace_bytes;
    for (const auto& execution : measurement::selections(options.execution, "graph", "eager"))
        for (const auto& cache : measurement::selections(options.cache, "cold", "warm")) {
            const auto result = measurement::measure(
                launch,
                [&](auto c) {
                    restore(c);
                    condition_gpu(c, 200);
                },
                flush, cuda, options, execution, cache);
            row.workspace_peak     = workspace.peak_used();
            const auto us          = result.gpu.median_us;
            const auto payload_gbs = payload / us / 1000;
            std::string metrics =
                std::string(kp::name(weights.format)) + ',' + (index_bf16 ? "bf16," : "fp8,") +
                kp::name(weights.mode) + ',' + (update ? "update," : "record,") +
                (partition ? "" : std::to_string(phase)) + ',' + std::to_string(live) + ',' +
                std::to_string(weights.bytes()) + ',' + std::to_string(reads) + ',' +
                std::to_string(writes) + ',' + std::to_string(payload) + ',' +
                std::to_string(flops) + ',' +
                (traffic ? std::to_string(traffic->executed_matrix_flops) : "") + ',' +
                std::to_string(payload_gbs) + ',' + std::to_string(flops / us / 1e6);
            if (traffic) {
                const auto pipeline = payload + traffic->workspace_read + traffic->workspace_write +
                                      traffic->repeated_weights + traffic->quantization_input;
                const auto pipeline_gbs = pipeline / us / 1000;
                metrics += ',' + std::to_string(traffic->workspace_read) + ',' +
                           std::to_string(traffic->workspace_write) + ',' +
                           std::to_string(traffic->repeated_weights) + ',' +
                           std::to_string(traffic->quantization_input) + ',' +
                           std::to_string(pipeline) + ',' + std::to_string(pipeline_gbs) + ',' +
                           std::to_string(payload_gbs / kRoofline.sustained_read_gbs * 100) + ',' +
                           std::to_string(pipeline_gbs / kRoofline.sustained_read_gbs * 100) + ',' +
                           std::to_string(pipeline_gbs / kRoofline.memory_gbs * 100) + ',' +
                           std::to_string(traffic->matrix_tensor_us) + ',' +
                           std::to_string(traffic->matrix_tensor_us / us * 100) + ',' +
                           std::to_string(traffic->simt);
            } else {
                // Private projection-route estimates only describe the rectangular fixture.
                metrics += ",,,,,,," +
                           std::to_string(payload_gbs / kRoofline.sustained_read_gbs * 100) +
                           ",,,,,";
            }
            metrics += partition ? partition->csv(result) + ',' + kpool::name(alias)
                                 : kpool::graph_csv(result) + ",,,";
            csv.write(row, result, execution, cache, metrics);
        }
}

void run_job(int argc, char** argv, kpool::JobResources& resources,
             std::unique_ptr<kp::Weights>& previous) {
    measurement::Options options;
    kpool::PackedOptions packed;
    std::vector<int> tokens{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096};
    int batch = 1, phase = 0;
    bool update = false, ragged = false, inplace = false, inactive = false;
    std::optional<kpool::StateAlias> state_alias;
    int active_rows     = -1;
    auto projection     = kp::Projection::Full;
    auto weight_format  = kp::ProjectionFormat::BFloat16;
    auto index_format   = KvCacheVectorFormat::BFloat16;
    auto latent_storage = ops::KpoolMlaStorage::BFloat16;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--help") {
            std::cout << "Kpool MLA complete Prepare entry, fixed "
                         "H4096/Dc512/Di128/Rq1536/Hi32.\n"
                         "  --projection full|query-history|history --form record|update\n"
                         "  --tokens N,... --batch B --phase 0..3 --ragged 0|1 --inplace 0|1\n"
                         "  --inactive 0|1  zero valid columns in every physical row\n"
                         "  --active-rows 0..B  leading active request rows (default B)\n"
                         "  --index-cache bf16|fp8 --latent-cache bf16|fp8\n"
                         "  --weight-format bf16|block-fp8|mxfp8\n"
                         "  --state-alias disjoint|inplace|mixed (packed --form update)\n";
            std::cout << "  --jobs-file PATH (one tab-separated argument list per line)\n"
                         "  --ragged-segments L,KxL;... --ragged-histories P,KxP;... (default 0)\n";
            measurement::help();
            return;
        }
        const auto v = measurement::take(argc, argv, i);
        if (measurement::parse(options, flag, v) || packed.parse(flag, v)) continue;
        if (flag == "--tokens")
            tokens = measurement::integers(v);
        else if (flag == "--weight-format" && (v == "bf16" || v == "block-fp8" || v == "mxfp8"))
            weight_format = v == "bf16"    ? kp::ProjectionFormat::BFloat16
                            : v == "mxfp8" ? kp::ProjectionFormat::Mxfp8
                                           : kp::ProjectionFormat::BlockFp8;
        else if (flag == "--state-alias")
            state_alias = kpool::parse_state_alias(v);
        else if (flag == "--batch")
            batch = measurement::integer(v, 1, INT_MAX);
        else if (flag == "--phase")
            phase = measurement::integer(v, 0, 3);
        else if (flag == "--ragged")
            ragged = measurement::integer(v, 0, 1);
        else if (flag == "--inplace")
            inplace = measurement::integer(v, 0, 1);
        else if (flag == "--inactive")
            inactive = measurement::integer(v, 0, 1);
        else if (flag == "--active-rows")
            active_rows = measurement::integer(v, 0, INT_MAX);
        else if (flag == "--index-cache" && (v == "bf16" || v == "fp8"))
            index_format = v == "fp8" ? KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0
                                      : KvCacheVectorFormat::BFloat16;
        else if (flag == "--latent-cache" && (v == "bf16" || v == "fp8"))
            latent_storage = v == "fp8" ? ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0
                                        : ops::KpoolMlaStorage::BFloat16;
        else if (flag == "--form" && (v == "record" || v == "update"))
            update = v == "update";
        else if (flag == "--projection" && (v == "full" || v == "query-history" || v == "history"))
            projection = v == "full"      ? kp::Projection::Full
                         : v == "history" ? kp::Projection::History
                                          : kp::Projection::QueryAndHistory;
        else
            throw std::invalid_argument("unknown option/value: " + flag + " " + std::string(v));
    }
    const auto partitions = packed.partitions();
    if (!partitions.empty() &&
        (!update || batch != 1 || ragged || inplace || inactive || active_rows >= 0 || phase != 0))
        throw std::invalid_argument("packed Prepare requires --form update and packed metadata");
    if (partitions.empty() && state_alias)
        throw std::invalid_argument("--state-alias requires --ragged-segments");
    if (active_rows < 0) active_rows = batch;
    if (active_rows > batch) throw std::invalid_argument("active rows exceed batch");
    if (inactive) active_rows = 0;
    auto& device = resources.device();
    auto& flush  = resources.flush(options.flush_bytes);
    if (!previous || previous->mode != projection || previous->format != weight_format)
        previous = std::make_unique<kp::Weights>(device.device(), projection, weight_format);
    auto& weights = *previous;
    measurement::Csv csv(options.csv_out, std::string(prepare_columns) +
                                              std::string(kpool::packed_columns) + ",state_alias");
    if (!partitions.empty()) {
        const auto alias = state_alias.value_or(kpool::StateAlias::Disjoint);
        for (const auto& partition : partitions) {
            const auto cuda = device.execution_view();
            kpool::PrepareFixture f(cuda, partition, alias, weights, latent_storage, index_format);
            report(
                csv, options, flush, cuda, weights, latent_storage, index_format,
                partition.tokens(), partition.segments(), true, "packed", -1, partition.lengths,
                partition.histories, 0, std::nullopt, &partition, alias, f.workspace.storage.size(),
                f.workspace.arena, [&](auto c) { f.launch(c); },
                [&](auto c) { f.tail.restore(c); });
        }
        return;
    }
    for (int T : tokens) {
        kp::Fixture f(weights, device.execution_view(), T, batch, update, ragged, false, inplace,
                      phase, index_format, latent_storage);
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
        const auto traffic = kpool_prepare_traffic(f, device.execution_view());
        report(
            csv, options, flush, device.execution_view(), weights, latent_storage, index_format, T,
            batch, update,
            inactive ? "empty"
            : ragged ? "ragged"
                     : "dense",
            !inactive && active_rows < batch ? active_rows : -1, f.valid.host, f.bases, phase,
            traffic, nullptr, kpool::StateAlias::Disjoint, f.workspace_bytes, *f.workspace,
            [&](auto c) { f.launch(c); }, [&](auto c) { f.restore(c); });
    }
}
} // namespace

int main(int argc, char** argv) {
    std::unique_ptr<kp::Weights> previous;
    return kpool::run_jobs(argc, argv, [&](int count, char** args, auto& resources) {
        run_job(count, args, resources, previous);
    });
}
