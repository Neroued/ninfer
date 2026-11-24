#include "ragged_fixture.cuh"
#include "ops/context_bucket_workload.h"
#include "common/gpu_activity.cuh"
#include "../../../../tools/support/kpool_sparse_mla_fixture.h"

#include <unordered_set>

#include <climits>

using namespace ninfer;
using namespace ninfer::bench;
namespace kp = ninfer::tool::kpool;

namespace {
constexpr const char* sparse_columns =
    "factor_format,end,selected_cap,frozen,selection_span,nonempty_queries,nontrivial_"
    "queries,selected_"
    "token_visits,attention_token_visits,unique_"
    "latent_bytes,"
    "factor_bytes,query_bytes,output_bytes,metadata_bytes,factor_flops,attention_flops,"
    "useful_TFLOps,qk_flops,pv_flops,index_storage,shared_prefix,bucket,envelope,"
    "length_dist,valid_dist";

template <class Launch>
void report(measurement::Csv& csv, const measurement::Options& options, L2FlushBuffer& flush,
            CudaExecutionView cuda, int tokens, int batch, int layers, int end, int selected,
            bool frozen, int selection_span, bool mxfp8, ops::KpoolMlaStorage storage,
            KvCacheVectorFormat index_format, int shared_prefix, ops::ContextBucket bucket,
            std::string_view length_dist, std::string_view validity, std::string_view valid,
            std::uint64_t nonempty, std::uint64_t nontrivial, std::uint64_t visits,
            std::uint64_t attention_visits, std::uint64_t unique_tokens,
            std::uint64_t metadata_bytes, std::uint64_t factor_bytes,
            const kpool::Partition* partition, std::size_t workspace_bytes, MemoryArena& workspace,
            Launch launch) {
    const std::uint64_t query_bytes = nontrivial * 64 * 256 * 2;
    const std::uint64_t output_bytes =
        (partition ? tokens : std::uint64_t(tokens) * batch) * 64 * 256 * 2;
    const double factor_flops    = (nonempty + nontrivial) * (2. * 64 * 512 * 256);
    const double attention_flops = attention_visits * (4. * 64 * 512);
    measurement::Row row;
    row.op              = "kpool_sparse_mla";
    row.format          = storage == ops::KpoolMlaStorage::BFloat16 ? "bf16" : "fp8";
    row.n               = 256;
    row.k               = 512;
    row.qk_heads        = 64;
    row.value_heads     = 64;
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
            csv.write(
                row, result, execution, cache,
                std::string(mxfp8 ? "mxfp8," : "bf16,") + std::to_string(end) + "," +
                    std::to_string(selected) + "," + std::to_string(frozen) + "," +
                    std::to_string(selection_span) + "," + std::to_string(nonempty) + "," +
                    std::to_string(nontrivial) + "," + std::to_string(visits) + "," +
                    std::to_string(attention_visits) + "," +
                    std::to_string(unique_tokens *
                                   (storage == ops::KpoolMlaStorage::BFloat16 ? 1024 : 516)) +
                    "," + std::to_string(factor_bytes) + "," + std::to_string(query_bytes) + "," +
                    std::to_string(output_bytes) + "," + std::to_string(metadata_bytes) + "," +
                    std::to_string(factor_flops) + "," + std::to_string(attention_flops) + "," +
                    std::to_string((factor_flops + attention_flops) / result.gpu.median_us / 1e6) +
                    "," + std::to_string(attention_flops / 2) + "," +
                    std::to_string(attention_flops / 2) + "," +
                    (index_format == KvCacheVectorFormat::BFloat16 ? "bf16" : "fp8") + "," +
                    std::to_string(shared_prefix) + "," +
                    std::string(ops::context_bucket_name(bucket)) + "," +
                    std::to_string(ops::context_bucket_tokens(bucket)) + "," +
                    std::string(length_dist) + "," + std::string(validity) +
                    (partition ? partition->csv(result) : kpool::graph_csv(result) + ",,"));
        }
}

void run_job(int argc, char** argv, kpool::JobResources& resources,
             std::unique_ptr<kpool::SparseFactors>& previous) {
    measurement::Options options;
    kpool::PackedOptions packed;
    std::vector<int> tokens{1, 2, 4, 8, 16, 32, 64}, reserve_tokens;
    int batch = 1, end = 8192, layers = 11, selected = 512, selection_span = 1;
    int shared_prefix = -1, envelope = 0, active_rows = -1;
    std::optional<ops::ContextBucket> bucket_override;
    auto length_dist = workload::LengthDist::Uniform;
    std::optional<workload::ValidDist> valid_dist;
    bool ragged = false, frozen = false, mxfp8 = false;
    auto storage      = ops::KpoolMlaStorage::BFloat16;
    auto index_format = KvCacheVectorFormat::BFloat16;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--help") {
            std::cout
                << "Complete latent Sparse MLA: U_K, attention, U_V.\n"
                   "  --tokens N,... --batch B --end E --layers 1|11\n"
                   "  --reserve-tokens N,...  preserve the cache layout of a wider sweep\n"
                   "  --selected 0..512 --ragged 0|1 --frozen 0|1\n"
                   "  --active-rows 0..B  leading active request rows (default B)\n"
                   "  --selection-span N  share each selection across N consecutive queries\n"
                   "  --latent-cache bf16|fp8 --index-cache bf16|fp8\n"
                   "  --factor-format bf16|mxfp8\n"
                   "  --shared-prefix N  share N original tokens, in complete 64-token pages\n"
                   "  --bucket auto|2k|8k|16k|32k|64k|128k|256k|512k|1m (default auto)\n"
                   "  --envelope E       auto bucket selection (default: --end)\n"
                   "  --length-dist uniform|skewed-1long|staggered|shared-prefix\n"
                   "  --valid-dist full|ragged|with-inert (overrides --ragged)\n"
                   "E is the final query's exclusive end; T<=E<=1048576.\n";
            std::cout << "  --jobs-file PATH (one tab-separated argument list per line)\n"
                         "  --ragged-segments L,KxL;... --ragged-histories P,KxP;... (default 0)\n";
            measurement::help();
            return;
        }
        if (i + 1 >= argc) throw std::invalid_argument("missing flag value: " + flag);
        const std::string value = argv[++i];
        if (measurement::parse(options, flag, value) || packed.parse(flag, value)) continue;
        if (flag == "--reserve-tokens")
            reserve_tokens = measurement::integers(value);
        else if (flag == "--tokens")
            tokens = measurement::integers(value);
        else if (flag == "--factor-format" && (value == "bf16" || value == "mxfp8"))
            mxfp8 = value == "mxfp8";
        else if (flag == "--batch")
            batch = measurement::integer(value, 1, INT_MAX);
        else if (flag == "--active-rows")
            active_rows = measurement::integer(value, 0, INT_MAX);
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
        else if (flag == "--bucket")
            bucket_override = workload::parse_bucket_override(value);
        else if (flag == "--length-dist")
            length_dist = workload::parse_length_dist(value);
        else if (flag == "--valid-dist")
            valid_dist = workload::parse_valid_dist(value);
        else if (flag == "--layers") {
            layers = measurement::integer(value, 1, 11);
            if (layers != 1 && layers != 11) throw std::invalid_argument("layers must be 1 or 11");
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
    const auto partitions = packed.partitions();
    if (!partitions.empty() &&
        (batch != 1 || ragged || active_rows >= 0 || !reserve_tokens.empty() || valid_dist ||
         length_dist != workload::LengthDist::Uniform))
        throw std::invalid_argument("packed Sparse MLA requires packed metadata");
    if (active_rows < 0) active_rows = batch;
    if (active_rows > batch) throw std::invalid_argument("active rows exceed batch");
    auto& device = resources.device();
    auto& flush  = resources.flush(options.flush_bytes);
    measurement::Csv csv(options.csv_out,
                         std::string(sparse_columns) + std::string(kpool::packed_columns));
    if (!partitions.empty()) {
        shared_prefix = std::max(0, shared_prefix);
        for (const auto& partition : partitions) partition.validate_shared_prefix(shared_prefix);
        if (!previous || previous->mxfp8 != mxfp8)
            previous = std::make_unique<kpool::SparseFactors>(device.device(), mxfp8);

        // A single-sequence sweep can use the same union of history pages as the rectangular
        // sweep. The permutation and physical sample indices then match across the two routes.
        kpool::PageLists common_pages;
        const int first_end       = partitions.front().max_end();
        const bool common_history = std::all_of(partitions.begin(), partitions.end(), [&](auto& p) {
            return p.segments() == 1 && p.max_end() == first_end;
        });
        if (common_history) {
            std::set<int> pages;
            for (const auto& partition : partitions) {
                const kpool::SparseSelection input(partition, selected, frozen, selection_span,
                                                   shared_prefix);
                pages.insert(input.pages[0].begin(), input.pages[0].end());
            }
            common_pages.emplace_back(pages.begin(), pages.end());
        }
        for (const auto& partition : partitions) {
            const auto bucket = partition.bucket(bucket_override, envelope);
            const auto cuda   = device.execution_view();
            const kpool::SparseSelection input(partition, selected, frozen, selection_span,
                                               shared_prefix);
            kpool::SparseFixture f(cuda, partition, input,
                                   common_history ? common_pages : input.pages, layers,
                                   shared_prefix, storage, index_format, bucket, previous->view());
            const auto factor_bytes = (input.nonempty ? previous->value_bytes() : 0) +
                                      (input.nontrivial ? previous->key_bytes() : 0);
            report(csv, options, flush, cuda, partition.tokens(), partition.segments(), layers,
                   partition.max_end(), selected, frozen, selection_span, mxfp8, storage,
                   index_format, shared_prefix, bucket, "packed", "full", "packed", input.nonempty,
                   input.nontrivial, input.visits, input.attention_visits, input.unique_tokens,
                   input.metadata_bytes, factor_bytes, &partition, f.workspace.storage.size(),
                   f.workspace.arena, [&](auto c) { f.launch(c); });
        }
        return;
    }
    reserve_tokens.insert(reserve_tokens.end(), tokens.begin(), tokens.end());
    std::sort(reserve_tokens.begin(), reserve_tokens.end());
    reserve_tokens.erase(std::unique(reserve_tokens.begin(), reserve_tokens.end()),
                         reserve_tokens.end());
    const int max_tokens = reserve_tokens.back();
    if (max_tokens > end) throw std::invalid_argument("tokens exceed --end");
    const auto bucket =
        bucket_override ? *bucket_override : ops::select_context_bucket(envelope ? envelope : end);
    std::vector<kp::SparseMlaBatch> sweep;
    const bool legacy_ragged = ragged && !valid_dist;
    for (int t : reserve_tokens) {
        auto shape = workload::generate_batch_workload(
            end, t, batch, length_dist, valid_dist.value_or(workload::ValidDist::Full));
        if (shared_prefix < 0) shared_prefix = shape.shared_prefix_tokens;
        if (legacy_ragged) {
            for (int b = 0; b < batch; ++b) {
                const int live         = b == batch - 1 ? 0 : std::max(1, t - b);
                shape.valid_columns[b] = live;
                shape.row_contexts[b]  = live ? shape.row_contexts[b] - (t - live) : 0;
            }
        }
        sweep.push_back({t, std::move(shape.row_contexts), std::move(shape.valid_columns)});
    }
    kp::SparseMlaFixture f(device.execution_view(), max_tokens, batch, end, selected, layers, false,
                           frozen, 1, selection_span, storage, index_format, shared_prefix,
                           ops::ContextBucketRange{bucket}, sweep);
    if (mxfp8) f.encode_factors(device.execution_view());
    for (int T : tokens) {
        const auto shape =
            std::find_if(sweep.begin(), sweep.end(), [&](const auto& s) { return s.tokens == T; });
        f.reset_tokens(T, selected, false, frozen, selection_span, device.execution_view(),
                       &*shape);
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
        std::string valid = valid_dist      ? std::string(workload::valid_dist_name(*valid_dist))
                            : legacy_ragged ? "ragged"
                                            : "dense";
        if (active_rows < batch) valid += "-rows" + std::to_string(active_rows);
        report(csv, options, flush, device.execution_view(), T, batch, layers, end, selected,
               frozen, selection_span, mxfp8, storage, index_format, shared_prefix, bucket,
               workload::length_dist_name(length_dist),
               valid_dist      ? workload::valid_dist_name(*valid_dist)
               : legacy_ragged ? "ragged"
                               : "full",
               valid, nonempty, nontrivial, visits, attention_visits, unique.size(), metadata_bytes,
               factor_bytes, nullptr, f.workspace_bytes, *f.workspace,
               [&](auto c) { f.launch(c); });
    }
}
} // namespace

int main(int argc, char** argv) {
    std::unique_ptr<kpool::SparseFactors> previous;
    return kpool::run_jobs(argc, argv, [&](int count, char** args, auto& resources) {
        run_job(count, args, resources, previous);
    });
}
