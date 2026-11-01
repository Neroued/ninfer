#include "ops/op_measurement.cuh"
#include "common/gpu_activity.cuh"
#include "../../../../tools/support/kpool_chain_fixture.h"

#include <set>

using namespace ninfer;
using namespace ninfer::bench;
namespace kp = ninfer::tool::kpool;

namespace {
// Diagnostic events observe the actual ordered graph. Only the uninstrumented
// complete call below is used for performance acceptance; stages include event cost.
std::array<double, 4> stages(kp::ChainFixture& f, L2FlushBuffer& flush,
                             const measurement::Options& options, const std::string& cache) {
    auto cuda = device_context().execution_view();
    std::array<cudaEvent_t, 5> events{};
    for (auto& event : events) CUDA_CHECK(cudaEventCreate(&event));
    auto graph = std::make_unique<measurement::BodyTimedGraph>();
    graph->capture(cuda, [&](auto c) {
        CUDA_CHECK(cudaEventRecordWithFlags(events[0], c.stream(), cudaEventRecordExternal));
        f.prepare(c);
        CUDA_CHECK(cudaEventRecordWithFlags(events[1], c.stream(), cudaEventRecordExternal));
        f.queries(c);
        CUDA_CHECK(cudaEventRecordWithFlags(events[2], c.stream(), cudaEventRecordExternal));
        f.index.launch(c);
        CUDA_CHECK(cudaEventRecordWithFlags(events[3], c.stream(), cudaEventRecordExternal));
        f.attend(c);
        CUDA_CHECK(cudaEventRecordWithFlags(events[4], c.stream(), cudaEventRecordExternal));
    });
    std::array<std::vector<double>, 4> values;
    for (int rep = -options.warmup; rep < options.repeat; ++rep) {
        condition_gpu(cuda, 200);
        if (cache == "cold") flush_l2(flush, cuda);
        graph->launch(cuda);
        CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
        if (rep < 0) continue;
        for (int s = 0; s < 4; ++s) {
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, events[s], events[s + 1]));
            values[s].push_back(ms * 1000.);
        }
    }
    std::array<double, 4> result;
    for (int s = 0; s < 4; ++s) result[s] = summarize_timings(std::move(values[s])).median_us;
    graph.reset();
    for (auto event : events) CUDA_CHECK(cudaEventDestroy(event));
    return result;
}

struct Work {
    std::uint64_t minimum_bytes = 0, visits = 0, candidates = 0;
    double flops = 0, latent_qk_flops = 0;
};

Work work(kp::ChainFixture& f) {
    auto& i                         = f.index;
    const std::uint64_t index_bytes = i.cache.format == KvCacheVectorFormat::BFloat16 ? 256 : 129;
    const std::uint64_t latent_bytes =
        f.cache.latent.storage == ops::KpoolMlaStorage::BFloat16 ? 1024 : 516;
    const auto ids = i.ids.download(), counts = i.counts.download(), ends = i.ends.download();
    std::set<std::size_t> old_latent;
    Work result;
    const int F          = i.E - i.T;
    std::uint64_t active = 0;
    bool key_factor      = false;
    result.minimum_bytes = i.B * 4;
    for (int b = 0; b < i.B; ++b) {
        const int n = i.valid.at(b);
        active += n;
        if (!n) continue;
        result.minimum_bytes += 20 + std::uint64_t(n) * 8 + ((F + n + 63) / 64) * 8;
        // Only the newest indexer row determines the union of prior index keys.
        if ((F + n) / 4 > 512) result.minimum_bytes += std::uint64_t(F / 4) * index_bytes;
        result.minimum_bytes += std::uint64_t(n) * latent_bytes +
                                ((F + n) / 4 - F / 4) * index_bytes + (F % 4 + 3) * 128 * 4;
        for (int t = 0; t < n; ++t) {
            const int row = b * i.T + t, end = ends[row], pools = end / 4;
            if (pools > 512) result.candidates += pools;
            const int size = counts[row] * 4 + end % 4;
            key_factor |= size > 1;
            result.visits += size;
            result.flops += (1 + (size > 1)) * (2. * 64 * 512 * 256);
            if (size > 1) {
                result.flops += size * (4. * 64 * 512);
                result.latent_qk_flops += size * (2. * 64 * 512);
            }
            for (int g = 0; g < counts[row]; ++g)
                for (int r = 0; r < 4; ++r) {
                    const int token = ids[row * 512 + g] * 4 + r;
                    if (token < F) old_latent.insert(f.physical_token(b, token));
                }
            for (int token = end / 4 * 4; token < end; ++token)
                if (token < F) old_latent.insert(f.physical_token(b, token));
        }
    }
    // External chain inputs, persistent publications and final output. Intermediate
    // rank/query/selection/scratch traffic is not mandatory DRAM traffic in this bound.
    if (active)
        result.minimum_bytes += f.weights.weight.view.bytes() + f.w_b.view.bytes() +
                                f.value_up.view.bytes() + (key_factor ? f.key_up.view.bytes() : 0) +
                                (512 + 1536 + 128 + 128 + 512) * 2 + active * 4096 * 2;
    result.minimum_bytes += f.output.view.bytes() + old_latent.size() * latent_bytes;
    // Useful dense contractions. Normalization, selection and SFU time are diagnosed
    // separately; this is a global lower bound, not an attainable fused schedule claim.
    result.flops +=
        active * (2. * 4096 * 2336 + 2. * 1536 * 20480) + result.candidates * (2. * 32 * 128);
    return result;
}
} // namespace

int main(int argc, char** argv) {
    try {
        measurement::Options options;
        std::vector<int> tokens{1, 8, 64, 256, 1024};
        int B = 1, E = 8192, layers = 11;
        bool ragged = false, diagnostic = true;
        auto index_format = KvCacheVectorFormat::BFloat16;
        auto storage      = ops::KpoolMlaStorage::BFloat16;
        for (int arg = 1; arg < argc; ++arg) {
            const std::string flag = argv[arg];
            if (flag == "--help") {
                std::cout << "Latent composition: Prepare -> Linear W_B -> Indexer -> Sparse MLA.\n"
                             "Synthetic represented weights; no Engine/model throughput claim.\n"
                             "  --tokens N,... --batch 1..8 --end E --layers 1|11 --ragged 0|1\n"
                             "  --index-cache bf16|fp8 --latent-cache bf16|fp8\n"
                             "  --stages 0|1  event-instrumented ordered Graph diagnostics\n";
                measurement::help();
                return 0;
            }
            const auto value = measurement::take(argc, argv, arg);
            if (measurement::parse(options, flag, value)) continue;
            if (flag == "--tokens")
                tokens = measurement::integers(value);
            else if (flag == "--batch")
                B = measurement::integer(value, 1, 8);
            else if (flag == "--end")
                E = measurement::integer(value, 1, 1048576);
            else if (flag == "--layers") {
                layers = measurement::integer(value, 1, 11);
                if (layers != 1 && layers != 11)
                    throw std::invalid_argument("layers must be 1 or 11");
            } else if (flag == "--ragged")
                ragged = measurement::integer(value, 0, 1);
            else if (flag == "--stages")
                diagnostic = measurement::integer(value, 0, 1);
            else if (flag == "--index-cache" && (value == "bf16" || value == "fp8"))
                index_format = value == "fp8" ? KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0
                                              : KvCacheVectorFormat::BFloat16;
            else if (flag == "--latent-cache" && (value == "bf16" || value == "fp8"))
                storage = value == "fp8" ? ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0
                                         : ops::KpoolMlaStorage::BFloat16;
            else
                throw std::invalid_argument("unknown flag: " + flag);
        }
        measurement::hardware();
        auto& device = device_context();
        L2FlushBuffer flush(options.flush_bytes);
        measurement::Csv csv(
            options.csv_out,
            "index_storage,end,minimum_external_bytes,useful_flops,bf16_flops,fp8_flops,candidate_"
            "pools,selected_token_visits,"
            "diagnostic_graph_prepare_us,diagnostic_graph_wb_us,diagnostic_graph_indexer_us,"
            "diagnostic_graph_mla_us");
        for (int T : tokens) {
            if (T > E) throw std::invalid_argument("T exceeds end");
            kp::ChainFixture f(device.execution_view(), T, B, E, layers, ragged, {}, false, true,
                               index_format, storage);
            f.launch(device.execution_view());
            device.synchronize();
            const auto w = work(f);
            measurement::Row row;
            row.op              = "kpool_chain";
            row.format          = storage == ops::KpoolMlaStorage::BFloat16 ? "bf16" : "fp8";
            row.tokens          = T;
            row.batch           = B;
            row.layers          = layers;
            row.valid           = ragged ? "ragged" : "dense";
            row.workspace_bytes = f.index.workspace_bytes;
            for (const auto& cache : measurement::selections(options.cache, "cold", "warm")) {
                const auto breakdown =
                    diagnostic ? stages(f, flush, options, cache) : std::array<double, 4>{};
                for (const auto& execution :
                     measurement::selections(options.execution, "graph", "eager")) {
                    const auto result = measurement::measure(
                        [&](auto c) { f.launch(c); }, [&](auto c) { condition_gpu(c, 200); }, flush,
                        device.execution_view(), options, execution, cache);
                    row.workspace_peak = f.index.workspace->peak_used();
                    const bool fp8 = index_format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0;
                    const double fp8_flops =
                        (fp8 ? w.candidates * 8192.0 : 0) +
                        (storage == ops::KpoolMlaStorage::BFloat16 ? 0 : w.latent_qk_flops);
                    std::string values =
                        std::string(fp8 ? "fp8," : "bf16,") + std::to_string(E) + ',' +
                        std::to_string(w.minimum_bytes) + ',' + std::to_string(w.flops) + ',' +
                        std::to_string(w.flops - fp8_flops) + ',' + std::to_string(fp8_flops) +
                        ',' + std::to_string(w.candidates) + ',' + std::to_string(w.visits);
                    for (double us : breakdown) values += ',' + std::to_string(us);
                    csv.write(row, result, execution, cache, values);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
