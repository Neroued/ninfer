#include "ops/op_measurement.cuh"
#include "common/gpu_activity.cuh"
#include "../../../../tools/support/kpool_sparse_mla_fixture.h"

#include <unordered_set>

using namespace ninfer;
using namespace ninfer::bench;
namespace kp = ninfer::tool::kpool;

int main(int argc, char** argv) {
    try {
        measurement::Options options;
        std::vector<int> tokens{1, 2, 4, 8, 16, 32, 64}, reserve_tokens;
        int batch = 1, end = 8192, layers = 11, selected = 512, selection_span = 1;
        int shared_prefix = 0, envelope = 0, active_rows = -1;
        bool ragged = false, frozen = false, mxfp8 = false;
        auto storage      = ops::KpoolMlaStorage::BFloat16;
        auto index_format = KvCacheVectorFormat::BFloat16;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--help") {
                std::cout
                    << "Complete latent Sparse MLA: U_K, attention, U_V.\n"
                       "  --tokens N,... --batch 1..8 --end E --layers 1|11\n"
                       "  --reserve-tokens N,...  preserve the cache layout of a wider sweep\n"
                       "  --selected 0..512 --ragged 0|1 --frozen 0|1\n"
                       "  --active-rows 0..B  leading active request rows (default B)\n"
                       "  --selection-span N  share each selection across N consecutive queries\n"
                       "  --latent-cache bf16|fp8 --index-cache bf16|fp8\n"
                       "  --factor-format bf16|mxfp8\n"
                       "  --shared-prefix N  share N original tokens, in complete 64-token pages\n"
                       "  --envelope E       execution bound (default: --end)\n"
                       "E is the final query's exclusive end; T<=E<=1048576.\n";
                measurement::help();
                return 0;
            }
            if (i + 1 >= argc) throw std::invalid_argument("missing flag value: " + flag);
            const std::string value = argv[++i];
            if (measurement::parse(options, flag, value)) continue;
            if (flag == "--reserve-tokens")
                reserve_tokens = measurement::integers(value);
            else if (flag == "--tokens")
                tokens = measurement::integers(value);
            else if (flag == "--factor-format" && (value == "bf16" || value == "mxfp8"))
                mxfp8 = value == "mxfp8";
            else if (flag == "--batch")
                batch = measurement::integer(value, 1, 8);
            else if (flag == "--active-rows")
                active_rows = measurement::integer(value, 0, 8);
            else if (flag == "--end")
                end = measurement::integer(value, 1, 1048576);
            else if (flag == "--selected")
                selected = measurement::integer(value, 0, 512);
            else if (flag == "--selection-span")
                selection_span = measurement::integer(value);
            else if (flag == "--shared-prefix")
                shared_prefix = measurement::integer(value, 0, 1048576);
            else if (flag == "--envelope")
                envelope = measurement::integer(value, 1, 1048576);
            else if (flag == "--layers") {
                layers = measurement::integer(value, 1, 11);
                if (layers != 1 && layers != 11)
                    throw std::invalid_argument("layers must be 1 or 11");
            } else if (flag == "--ragged")
                ragged = measurement::integer(value, 0, 1);
            else if (flag == "--frozen")
                frozen = measurement::integer(value, 0, 1);
            else if (flag == "--latent-cache" && (value == "bf16" || value == "fp8"))
                storage = value == "fp8" ? ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0
                                         : ops::KpoolMlaStorage::BFloat16;
            else if (flag == "--index-cache" && (value == "bf16" || value == "fp8"))
                index_format = value == "fp8" ? KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0
                                              : KvCacheVectorFormat::BFloat16;
            else
                throw std::invalid_argument("unknown flag: " + flag);
        }
        if (active_rows < 0) active_rows = batch;
        if (active_rows > batch) throw std::invalid_argument("active rows exceed batch");
        measurement::hardware();
        auto& device = device_context();
        L2FlushBuffer flush(options.flush_bytes);
        reserve_tokens.insert(reserve_tokens.end(), tokens.begin(), tokens.end());
        std::sort(reserve_tokens.begin(), reserve_tokens.end());
        reserve_tokens.erase(std::unique(reserve_tokens.begin(), reserve_tokens.end()),
                             reserve_tokens.end());
        const int max_tokens = *std::max_element(reserve_tokens.begin(), reserve_tokens.end());
        if (max_tokens > end) throw std::invalid_argument("tokens exceed --end");
        kp::SparseMlaFixture f(device.execution_view(), max_tokens, batch, end, selected, layers,
                               ragged, frozen, 1, selection_span, storage, index_format,
                               shared_prefix, envelope, reserve_tokens);
        if (mxfp8) f.encode_factors(device.execution_view());
        measurement::Csv csv(
            options.csv_out,
            "factor_format,end,selected_cap,frozen,selection_span,nonempty_queries,nontrivial_"
            "queries,selected_"
            "token_visits,attention_token_visits,unique_"
            "latent_bytes,"
            "factor_bytes,query_bytes,output_bytes,metadata_bytes,factor_flops,attention_flops,"
            "useful_TFLOps,qk_flops,pv_flops,index_storage,shared_prefix,envelope");
        for (int T : tokens) {
            f.reset_tokens(T, selected, ragged, frozen, selection_span, device.execution_view());
            if (active_rows < batch) {
                for (int b = active_rows; b < batch; ++b) {
                    f.valid.at(b)     = 0;
                    f.selectors.at(b) = -1;
                    for (int t = 0; t < T; ++t) {
                        f.positions.at(t, b)      = -1;
                        f.selection_rows.at(t, b) = -1;
                    }
                }
                f.valid.upload();
                f.selectors.upload();
                f.positions.upload();
                f.selection_rows.upload();
            }
            std::uint64_t nonempty = 0, nontrivial = 0, visits = 0, attention_visits = 0;
            std::unordered_set<std::size_t> unique;
            std::unordered_set<int> selection, requests;
            std::set<std::pair<int, int>> pages;
            std::uint64_t metadata_bytes = batch * 4;
            for (int b = 0; b < batch; ++b)
                for (int t = 0; t < f.valid.at(b); ++t) {
                    const auto ids = f.selected_tokens(t, b);
                    metadata_bytes += 4; // Query-to-selection mapping.
                    selection.insert(f.selection_rows.at(t, b));
                    if (!ids.empty()) requests.insert(b);
                    nonempty += !ids.empty();
                    visits += ids.size();
                    nontrivial += ids.size() > 1;
                    if (ids.size() > 1) attention_visits += ids.size();
                    for (int token : ids) {
                        unique.insert(f.physical_token(b, token));
                        pages.emplace(b, token / 64);
                    }
                }
            metadata_bytes += selection.size() * 8 + requests.size() * 4 + pages.size() * 8;
            for (int row : selection)
                // Selecting every complete pool makes its ID permutation immaterial.
                if (f.counts.at(row) < f.ends.at(row) / 4) metadata_bytes += f.counts.at(row) * 4;
            const std::uint64_t factor_bytes =
                (nonempty ? (mxfp8 ? f.value_encoded->bytes() : f.value_up.view.bytes()) : 0) +
                (nontrivial ? (mxfp8 ? f.key_encoded->bytes() : f.key_up.view.bytes()) : 0);
            const std::uint64_t query_bytes  = nontrivial * 64 * 256 * 2;
            const std::uint64_t output_bytes = std::uint64_t(T) * batch * 64 * 256 * 2;
            const double factor_flops        = (nonempty + nontrivial) * (2. * 64 * 512 * 256);
            const double attention_flops     = attention_visits * (4. * 64 * 512);
            measurement::Row row;
            row.op          = "kpool_sparse_mla";
            row.format      = storage == ops::KpoolMlaStorage::BFloat16 ? "bf16" : "fp8";
            row.n           = 256;
            row.k           = 512;
            row.qk_heads    = 64;
            row.value_heads = 64;
            row.tokens      = T;
            row.batch       = batch;
            row.layers      = layers;
            row.valid       = ragged ? "ragged" : "dense";
            if (active_rows < batch) row.valid += "-rows" + std::to_string(active_rows);
            row.workspace_bytes = f.workspace_bytes;
            for (const auto& execution :
                 measurement::selections(options.execution, "graph", "eager"))
                for (const auto& cache : measurement::selections(options.cache, "cold", "warm")) {
                    const auto result = measurement::measure(
                        [&](auto c) { f.launch(c); }, [&](auto c) { condition_gpu(c, 200); }, flush,
                        device.execution_view(), options, execution, cache);
                    row.workspace_peak = f.workspace->peak_used();
                    csv.write(row, result, execution, cache,
                              std::string(mxfp8 ? "mxfp8," : "bf16,") + std::to_string(end) + "," +
                                  std::to_string(selected) + "," + std::to_string(frozen) + "," +
                                  std::to_string(selection_span) + "," + std::to_string(nonempty) +
                                  "," + std::to_string(nontrivial) + "," + std::to_string(visits) +
                                  "," + std::to_string(attention_visits) + "," +
                                  std::to_string(
                                      unique.size() *
                                      (storage == ops::KpoolMlaStorage::BFloat16 ? 1024 : 516)) +
                                  "," + std::to_string(factor_bytes) + "," +
                                  std::to_string(query_bytes) + "," + std::to_string(output_bytes) +
                                  "," + std::to_string(metadata_bytes) + "," +
                                  std::to_string(factor_flops) + "," +
                                  std::to_string(attention_flops) + "," +
                                  std::to_string((factor_flops + attention_flops) /
                                                 result.gpu.median_us / 1e6) +
                                  "," + std::to_string(attention_flops / 2) + "," +
                                  std::to_string(attention_flops / 2) + "," +
                                  (index_format == KvCacheVectorFormat::BFloat16 ? "bf16" : "fp8") +
                                  "," + std::to_string(shared_prefix) + "," +
                                  std::to_string(f.envelope.max_visible_tokens));
                }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
