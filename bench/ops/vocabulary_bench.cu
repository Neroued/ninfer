#include "ninfer/ops/vocabulary.h"
#include "op_measurement.cuh"
#include "common/roofline.h"

#include <numeric>
#include <optional>

using namespace ninfer;
using namespace ninfer::bench;

namespace {
struct Options {
    measurement::Options timing;
    std::string op = "candidates";
    std::vector<int> tokens{1};
    int batch = 1, rows = 124160, domain = 248077, ranks = 2, rank = 0;
    std::optional<int> candidates;
    bool stochastic = false, penalties = false, mask = false, accept_all = false;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help") {
            std::cout << "Vocabulary partition benchmark\n"
                         "  --op candidates|sample|accept|counts --tokens T[,T...] --batch 1..8\n"
                         "  --rows V --domain D --ranks 1..16 --rank R\n"
                         "  --candidates 1|20 (defaults to the sampling requirement)\n"
                         "  --stochastic --penalties --mask --accept-all\n";
            measurement::help();
            std::exit(0);
        }
        if (flag == "--stochastic") {
            o.stochastic = true;
            continue;
        }
        if (flag == "--penalties") {
            o.penalties = true;
            continue;
        }
        if (flag == "--mask") {
            o.mask = true;
            continue;
        }
        if (flag == "--accept-all") {
            o.accept_all = true;
            continue;
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--op")
            o.op = value;
        else if (flag == "--tokens")
            o.tokens = measurement::integers(value);
        else if (flag == "--batch")
            o.batch = measurement::integer(value, 1, 8);
        else if (flag == "--rows")
            o.rows = measurement::integer(value);
        else if (flag == "--domain")
            o.domain = measurement::integer(value);
        else if (flag == "--ranks")
            o.ranks = measurement::integer(value, 1, 16);
        else if (flag == "--rank")
            o.rank = measurement::integer(value, 0, 15);
        else if (flag == "--candidates")
            o.candidates = measurement::integer(value);
        else if (!measurement::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option");
    }
    if ((o.op != "candidates" && o.op != "sample" && o.op != "accept" && o.op != "counts") ||
        (o.candidates && *o.candidates != 1 && *o.candidates != 20) ||
        (o.stochastic && o.candidates == 1) || o.rank >= o.ranks ||
        std::int64_t(o.rows) * o.ranks < o.domain || (o.accept_all && o.op != "accept") ||
        (o.op == "accept" && o.domain < 24) ||
        (o.op == "counts" && (o.stochastic || o.penalties || o.mask)))
        throw std::invalid_argument("invalid Op or vocabulary partition");
    return o;
}

void bench_counts(const Options& o, int t, L2FlushBuffer& flush, measurement::Csv& csv) {
    std::vector<int> values(std::size_t(t) * o.batch), selected(o.batch);
    for (int b = 0; b < o.batch; ++b) {
        selected[b] = b;
        for (int col = 0; col < t; ++col)
            values[b * t + col] = (std::int64_t(col) * 17 + b * 97) % o.domain;
    }
    auto ids    = measurement::make_i32(values),
         valid  = measurement::make_i32(std::vector<int>(o.batch, t));
    auto slots  = measurement::make_i32(selected),
         counts = make_zeros(std::size_t(o.domain) * o.batch * 4);
    Tensor input(ids.span(), DType::I32, {t, o.batch}), n(valid.span(), DType::I32, {o.batch});
    Tensor lanes(slots.span(), DType::I32, {o.batch}),
        out(counts.span(), DType::I32, {o.domain, o.batch});
    measurement::Row row;
    row.op           = "increment_token_counts_batch";
    row.format       = "i32";
    row.tokens       = t;
    row.batch        = o.batch;
    const auto bytes = ids.size() + valid.size() + slots.size() +
                       std::size_t(std::min(t, o.domain / std::gcd(o.domain, 17))) * o.batch * 8;
    for (const auto& execution : measurement::selections(o.timing.execution, "graph", "eager"))
        for (const auto& cache : measurement::selections(o.timing.cache, "cold", "warm")) {
            const auto timing = measurement::measure(
                [&](auto cuda) { ops::increment_token_counts_batch(input, n, lanes, out, cuda); },
                [&](auto cuda) {
                    CUDA_CHECK(cudaMemsetAsync(counts.data(), 0, counts.size(), cuda.stream()));
                },
                flush, device_context().execution_view(), o.timing, execution, cache);
            csv.write(row, timing, execution, cache,
                      "0,1,0,0,0," + std::to_string(bytes) + ',' +
                          format_metric(bytes / (timing.gpu.median_us * 1000), 3, "") + ',' +
                          format_metric(bytes / (kRoofline.memory_gbs * 1000), 3, ""));
        }
}

struct Fixture {
    const Options& options;
    int t, k;
    DeviceBuffer logits, configs, counts, masks, valid, drafts, keys, work, positions, output;
    DeviceBuffer extents, lengths, licensed, licensed_counts;
    Tensor input, key_view, valid_view, draft_view;
    MemoryArena workspace;
    std::size_t logical_bytes;

    Fixture(const Options& o, int tokens)
        : options(o), t(tokens), k(o.candidates.value_or(o.stochastic ? 20 : 1)),
          logits(make_bf16(std::size_t(o.rows) * t * o.batch, 177, -8, 8)),
          configs(make_buffer(o.batch * sizeof(ops::SamplingConfig))),
          counts(o.penalties
                     ? measurement::make_i32(std::vector<int>(std::size_t(o.domain) * o.batch, 1))
                     : DeviceBuffer{}),
          masks(o.mask ? measurement::make_i32(
                             std::vector<int>(std::size_t((o.domain + 31) / 32) * t * o.batch, -1))
                       : DeviceBuffer{}),
          valid(measurement::make_i32(std::vector<int>(o.batch, t))),
          drafts(measurement::make_i32(std::vector<int>(std::max(1, t - 1) * o.batch, 17))),
          keys(make_buffer(std::size_t(k) * t * o.batch * o.ranks * 8)),
          work(make_buffer(std::max<std::size_t>(
              256, ops::vocabulary_candidates_workspace_capacity_bytes(o.rows, t * o.batch, k)))),
          positions(measurement::make_i32(std::vector<int>(o.batch, 1024))),
          output(make_buffer(o.batch * 4)),
          extents(measurement::make_i32(std::vector<int>(o.batch, t - 1))),
          lengths(measurement::make_i32(std::vector<int>(o.batch, 1024))),
          licensed(make_buffer(t * o.batch * 4)), licensed_counts(make_buffer(o.batch * 4)),
          input(logits.span(), DType::BF16, {o.rows, t, o.batch}),
          key_view(keys.span(), DType::I64, {k, t, o.batch, o.ranks}),
          valid_view(valid.span(), DType::I32, {o.batch}), workspace(work.span()) {
        if ((o.op == "sample" && t != 1) || (o.op == "accept" && (t < 2 || t > 16)))
            throw std::invalid_argument("invalid decision T");
        if (t > 1) draft_view = Tensor(drafts.span(), DType::I32, {t - 1, o.batch});
        std::vector<ops::SamplingConfig> cfg(o.batch);
        for (int b = 0; b < o.batch; ++b) {
            cfg[b].temperature       = o.stochastic ? .8F : 0;
            cfg[b].presence_penalty  = o.penalties ? .25F : 0;
            cfg[b].frequency_penalty = o.penalties ? .125F : 0;
            cfg[b].seed              = 913 + b;
            cfg[b].token_counts =
                o.penalties ? static_cast<int*>(counts.data()) + b * o.domain : nullptr;
            if (o.mask)
                cfg[b].mask = {static_cast<const std::uint32_t*>(masks.data()) +
                                   b * t * ((o.domain + 31) / 32),
                               (o.domain + 31) / 32};
        }
        configs.copy_from_host(cfg.data(), configs.size());
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        const auto cuda = device_context().execution_view();
        if (o.op == "sample") {
            for (int rank = 0; rank < o.ranks; ++rank) extract(rank, cuda);
            CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
        }
        if (o.op == "accept") {
            std::vector<std::uint64_t> candidates(std::size_t(k) * t * o.batch * o.ranks);
            for (int b = 0; b < o.batch; ++b)
                for (int col = 0; col < t; ++col) {
                    const int id                  = o.accept_all && col + 1 < t ? 17 : 23;
                    candidates[(b * t + col) * k] = (1ULL << 63) | (UINT32_MAX - id);
                }
            keys.copy_from_host(candidates.data(), keys.size());
        }
        const auto live = std::max(0, std::min(o.rows, o.domain - o.rank * o.rows));
        logical_bytes   = std::size_t(live) * t * o.batch * 2 + std::size_t(k) * t * o.batch * 8 +
                        o.batch * (4 + sizeof(ops::SamplingConfig));
        if (o.penalties) logical_bytes += std::size_t(live) * o.batch * 4 + drafts.size();
        if (o.mask) logical_bytes += std::size_t((live + 31) / 32) * t * o.batch * 4;
        if (o.op != "candidates")
            logical_bytes = keys.size() + configs.size() +
                            (o.op == "sample" ? positions.size() + output.size()
                                              : drafts.size() + extents.size() + lengths.size() +
                                                    licensed.size() + licensed_counts.size());
    }

    void extract(int rank, CudaExecutionView cuda) {
        Tensor out(keys.span().subspan(std::size_t(rank) * k * t * options.batch * 8,
                                       std::size_t(k) * t * options.batch * 8),
                   DType::I64, {k, t, options.batch});
        ops::vocabulary_candidates(input, rank * options.rows, options.domain,
                                   static_cast<const ops::SamplingConfig*>(configs.data()),
                                   valid_view, draft_view, out, workspace, cuda);
    }

    void launch(CudaExecutionView cuda) {
        const auto* cfg = static_cast<const ops::SamplingConfig*>(configs.data());
        const int b     = options.batch;
        if (options.op == "candidates")
            extract(options.rank, cuda);
        else if (options.op == "sample") {
            Tensor pos(positions.span(), DType::I32, {b}), out(output.span(), DType::I32, {b});
            ops::sample_vocabulary_candidates(key_view, cfg, pos, ops::kSamplePurposeDecode, out,
                                              cuda);
        } else {
            Tensor e(extents.span(), DType::I32, {b}), l(lengths.span(), DType::I32, {b}),
                tokens(licensed.span(), DType::I32, {t, b}),
                n(licensed_counts.span(), DType::I32, {b});
            ops::accept_vocabulary_candidates(key_view, cfg, draft_view, e, l, tokens, n, cuda);
        }
    }

    void restore(CudaExecutionView cuda) {
        if (options.op == "accept")
            CUDA_CHECK(cudaMemcpyAsync(lengths.data(), positions.data(), positions.size(),
                                       cudaMemcpyDeviceToDevice, cuda.stream()));
    }
};
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        measurement::hardware();
        L2FlushBuffer flush(options.timing.flush_bytes);
        measurement::Csv csv(options.timing.csv_out, "rank,ranks,candidates,penalties,mask,logical_"
                                                     "bytes,logical_gb_s,bandwidth_reference_us");
        for (int t : options.tokens) {
            if (options.op == "counts") {
                bench_counts(options, t, flush, csv);
                continue;
            }
            Fixture f(options, t);
            measurement::Row row;
            row.op     = options.op;
            row.format = "bf16";
            row.tokens = t;
            row.batch  = options.batch;
            row.n      = options.rows;
            if (options.op == "accept") row.commits = options.accept_all ? std::to_string(t) : "1";
            for (const auto& execution :
                 measurement::selections(options.timing.execution, "graph", "eager"))
                for (const auto& cache :
                     measurement::selections(options.timing.cache, "cold", "warm")) {
                    const auto timing = measurement::measure(
                        [&](auto c) { f.launch(c); }, [&](auto c) { f.restore(c); }, flush,
                        device_context().execution_view(), options.timing, execution, cache);
                    row.workspace_bytes = options.op == "candidates" ? f.work.size() : 0;
                    row.workspace_peak  = options.op == "candidates" ? f.workspace.peak_used() : 0;
                    csv.write(
                        row, timing, execution, cache,
                        std::to_string(options.rank) + ',' + std::to_string(options.ranks) + ',' +
                            std::to_string(f.k) + ',' + std::to_string(options.penalties) + ',' +
                            std::to_string(options.mask) + ',' + std::to_string(f.logical_bytes) +
                            ',' +
                            format_metric(f.logical_bytes / (timing.gpu.median_us * 1000), 3, "") +
                            ',' +
                            format_metric(f.logical_bytes / (kRoofline.memory_gbs * 1000), 3, ""));
                }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
