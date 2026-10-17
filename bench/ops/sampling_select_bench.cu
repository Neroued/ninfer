#include "core/device.h"
#include "core/buffer.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_accept.h"
#include "op_measurement.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::int32_t kPhysicalRows      = 248320;
constexpr std::int32_t kTokenDomain       = 248077;
constexpr std::int32_t kDFlash2Candidates = 16;

enum class Mode {
    Greedy,
    Stochastic,
};

struct Options {
    bool sample        = false;
    bool mtp           = false;
    bool dflash2       = false;
    bool matrix        = false;
    bool counts_active = true;
    bool mask          = false;
    std::vector<std::int32_t> row_extents;
    measurement::Options timing{.warmup = 8, .repeat = 60, .cache = "cold"};
    Mode mode          = Mode::Stochastic;
    int batch          = 1;
    int mtp_k          = 3;
    int drafts         = 7;
    int extent         = 0;
    bool extent_set    = false;
    bool force_general = false;
    bool mixed         = false;
    int reject_at      = -1;
    int top_k          = 20;
};

void usage(const char* argv0) {
    std::printf("usage: %s [--sample|--mtp|--dflash2|--matrix] [--mode greedy|stochastic] "
                "[--batch 1..8] [--mtp-k 1..5] [--drafts 1..15] [--extent 0..K] [--top-k 1..20] "
                "[--no-counts] [--general] [--mixed] [--reject-at J] [--warmup N] [--repeat N] "
                "[--row-extents N,...] [--mask]\n",
                argv0);
    measurement::help();
}

int parse_int(std::string_view value, const char* name) {
    try {
        std::size_t parsed = 0;
        const int out      = std::stoi(std::string(value), &parsed);
        if (parsed != value.size()) { throw std::invalid_argument("trailing"); }
        return out;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " expects an integer");
    }
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto need_value = [&](const char* name) -> std::string_view {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string(name) + " needs a value");
            }
            return argv[++i];
        };
        if (arg == "--sample") {
            options.sample = true;
        } else if (arg == "--mtp") {
            options.mtp = true;
        } else if (arg == "--dflash2") {
            options.dflash2 = true;
        } else if (arg == "--matrix") {
            options.matrix = true;
        } else if (arg == "--mode") {
            const std::string_view mode = need_value("--mode");
            if (mode == "greedy") {
                options.mode = Mode::Greedy;
            } else if (mode == "stochastic") {
                options.mode = Mode::Stochastic;
            } else {
                throw std::invalid_argument("--mode must be greedy or stochastic");
            }
        } else if (arg == "--batch") {
            options.batch = parse_int(need_value("--batch"), "--batch");
        } else if (arg == "--mtp-k") {
            options.mtp_k = parse_int(need_value("--mtp-k"), "--mtp-k");
        } else if (arg == "--extent") {
            options.extent     = parse_int(need_value("--extent"), "--extent");
            options.extent_set = true;
        } else if (arg == "--drafts") {
            options.drafts = parse_int(need_value("--drafts"), "--drafts");
        } else if (arg == "--general") {
            options.force_general = true;
        } else if (arg == "--mixed") {
            options.mixed = true;
        } else if (arg == "--reject-at") {
            options.reject_at = parse_int(need_value("--reject-at"), "--reject-at");
        } else if (arg == "--warmup") {
            options.timing.warmup = parse_int(need_value("--warmup"), "--warmup");
        } else if (arg == "--repeat") {
            options.timing.repeat = parse_int(need_value("--repeat"), "--repeat");

        } else if (arg == "--top-k") {
            options.top_k = parse_int(need_value("--top-k"), "--top-k");
        } else if (arg == "--mask") {
            options.mask = true;
        } else if (arg == "--row-extents") {
            options.row_extents = measurement::integers(need_value("--row-extents"), 0);
        } else if (arg == "--no-counts") {
            options.counts_active = false;
        } else if (arg == "-h" || arg == "--help") {
            usage(argc > 0 ? argv[0] : "ninfer_sampling_select_bench");
            std::exit(0);
        } else if (!measurement::parse(options.timing, arg, need_value("option"))) {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (!options.sample && !options.mtp && !options.dflash2 && !options.matrix) {
        options.matrix = true;
    }
    if (options.matrix && (options.sample || options.mtp || options.dflash2)) {
        throw std::invalid_argument(
            "--matrix cannot be combined with --sample, --mtp, or --dflash2");
    }
    if (options.dflash2 && (options.sample || options.mtp)) {
        throw std::invalid_argument("--dflash2 cannot be combined with --sample or --mtp");
    }
    if (options.mtp_k < 1 || options.mtp_k > 5) {
        throw std::invalid_argument("--mtp-k must be in [1,5]");
    }
    if (options.batch < 1 || options.batch > 8) {
        throw std::invalid_argument("--batch must be in [1,8]");
    }
    if (options.drafts < 1 || options.drafts > 15)
        throw std::invalid_argument("--drafts must be in [1,15]");
    const int drafts = options.dflash2 ? options.drafts : options.matrix ? 5 : options.mtp_k;
    if (!options.extent_set) options.extent = drafts;
    if (options.extent < 0 || options.extent > drafts)
        throw std::invalid_argument("--extent must fit K");
    if (options.reject_at < -1 || options.reject_at >= drafts)
        throw std::invalid_argument("--reject-at must fit K");
    if (!options.row_extents.empty() &&
        (options.row_extents.size() != std::size_t(options.batch) ||
         *std::max_element(options.row_extents.begin(), options.row_extents.end()) > drafts))
        throw std::invalid_argument("row extents must fit B and K");
    if (options.matrix &&
        (!options.row_extents.empty() || options.extent_set || options.reject_at >= 0))
        throw std::invalid_argument("explicit extents/rejection require one acceptance profile");
    if (options.force_general && !options.dflash2)
        throw std::invalid_argument("--general requires --dflash2");
    if (options.timing.warmup < 0 || options.timing.repeat < 1)
        throw std::invalid_argument("invalid timing counts");
    if (options.top_k < 1 || options.top_k > 20) {
        throw std::invalid_argument("--top-k must be in [1,20]");
    }
    return options;
}

DeviceBuffer make_logits(int cols) {
    std::vector<std::uint16_t> host(static_cast<std::size_t>(kPhysicalRows) * cols);
    for (int col = 0; col < cols; ++col) {
        const int hot = (17 + col * 7919) % kTokenDomain;
        for (int row = 0; row < kPhysicalRows; ++row) {
            float value = fixture::uniform(static_cast<std::uint64_t>(col) * kPhysicalRows + row,
                                           101U, -8.F, -7.F);
            if (row == hot) { value = 8.0f; }
            if (row >= kTokenDomain) { value = 20.0f; }
            host[static_cast<std::size_t>(col) * kPhysicalRows + row] = f32_to_bf16(value);
        }
    }
    DeviceBuffer device(ninfer::bench::device_context().device(),
                        host.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(device.data(), host.data(), device.size(), cudaMemcpyHostToDevice));
    return device;
}

DeviceBuffer make_i32(const std::vector<std::int32_t>& host) {
    DeviceBuffer device(ninfer::bench::device_context().device(),
                        host.size() * sizeof(std::int32_t));
    CUDA_CHECK(cudaMemcpy(device.data(), host.data(), device.size(), cudaMemcpyHostToDevice));
    return device;
}

DeviceBuffer make_batch_configs(DeviceBuffer& counts, int batch, Mode mode, bool counts_active,
                                int top_k, bool mixed, const DeviceBuffer& masks, int columns) {
    std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(batch));
    for (int row = 0; row < batch; ++row) {
        ops::SamplingConfig& config = configs[static_cast<std::size_t>(row)];
        config.temperature          = mode == Mode::Greedy ? 0.0f : 0.6f;
        config.top_k                = top_k;
        config.top_p                = 0.95f;
        config.presence_penalty     = mode == Mode::Stochastic && counts_active ? 1.0f : 0.0f;
        config.seed                 = 20260716ull + static_cast<unsigned long long>(row);
        config.token_counts         = mode == Mode::Stochastic && counts_active
                                          ? static_cast<std::int32_t*>(counts.data()) +
                                        static_cast<std::size_t>(row) * kTokenDomain
                                          : nullptr;
        if (masks.size()) {
            const int words = (kTokenDomain + 31) / 32;
            config.mask     = {static_cast<const std::uint32_t*>(masks.data()) +
                                   std::size_t(row) * columns * words,
                               words};
        }
        if (mixed) {
            config.temperature       = row % 3 == 2 ? 0.6f : 0.0f;
            config.presence_penalty  = row % 3 == 0 ? 0.0f : 1.0f;
            config.frequency_penalty = row % 3 == 1 ? 0.125f : 0.0f;
            config.token_counts =
                counts_active ? static_cast<int*>(counts.data()) + row * kTokenDomain : nullptr;
        }
    }
    DeviceBuffer device(ninfer::bench::device_context().device(),
                        configs.size() * sizeof(ops::SamplingConfig));
    CUDA_CHECK(cudaMemcpy(device.data(), configs.data(), device.size(), cudaMemcpyHostToDevice));
    return device;
}

std::string list_counts(const std::vector<std::int32_t>& values) {
    std::string result;
    for (int x : values) {
        if (!result.empty()) result += ':';
        result += std::to_string(x);
    }
    return result;
}

class DecisionBenchmark {
public:
    explicit DecisionBenchmark(const Options& o)
        : timing_(o.timing), flush_(o.timing.flush_bytes),
          csv_(o.timing.csv_out, "mode,top_k,counts,mask,reject_at,raw_greedy_envelope,logical_"
                                 "bytes,bandwidth_reference_us") {}

    template <class Launch, class Restore>
    void run(const measurement::Row& row, const Options& o, std::size_t bytes, Launch launch,
             Restore restore, bool raw_greedy = false) {
        for (const auto& execution : measurement::selections(timing_.execution, "graph", "eager"))
            for (const auto& cache : measurement::selections(timing_.cache, "cold", "warm")) {
                const auto result =
                    measurement::measure(launch, restore, flush_, device_context().execution_view(),
                                         timing_, execution, cache);
                const std::string mode = o.mixed                  ? "mixed"
                                         : o.mode == Mode::Greedy ? "greedy"
                                                                  : "stochastic";
                csv_.write(
                    row, result, execution, cache,
                    mode + ',' + std::to_string(o.top_k) + ',' +
                        std::to_string(o.counts_active && (o.mixed || o.mode == Mode::Stochastic)) +
                        ',' + std::to_string(o.mask) + ',' + std::to_string(o.reject_at) + ',' +
                        std::to_string(raw_greedy) + ',' + std::to_string(bytes) + ',' +
                        format_metric(bytes / (kRoofline.memory_gbs * 1000.), 3, ""));
            }
    }
private:
    measurement::Options timing_;
    L2FlushBuffer flush_;
    measurement::Csv csv_;
};

DeviceBuffer make_counts(int batch) {
    std::vector<std::int32_t> values(std::size_t(kTokenDomain) * batch, 0);
    for (int b = 0; b < batch; ++b)
        for (int v = b; v < kTokenDomain; v += 113)
            values[std::size_t(b) * kTokenDomain + v] = 1 + v % 5;
    return make_i32(values);
}

DeviceBuffer make_masks(int columns, int batch, bool enabled) {
    if (!enabled) return {};
    const int words = (kTokenDomain + 31) / 32;
    std::vector<std::int32_t> values(std::size_t(words) * columns * batch, -1);
    for (int b = 0; b < batch; ++b)
        for (int c = 0; c < columns; ++c)
            for (int v = (b + c) % 7; v < kTokenDomain; v += 7)
                values[(std::size_t(b) * columns + c) * words + v / 32] &=
                    ~(std::uint32_t{1} << (v % 32));
    return make_i32(values);
}

std::vector<std::int32_t> extents(const Options& o, int columns) {
    return o.row_extents.empty()
               ? std::vector<std::int32_t>(o.batch, std::min(o.extent, columns - 1))
               : o.row_extents;
}

std::size_t decision_bytes(const Options& o, int columns, const std::vector<std::int32_t>& active,
                           bool acceptance) {
    std::size_t bytes = std::size_t(o.batch) * sizeof(ops::SamplingConfig);
    for (int b = 0; b < o.batch; ++b) {
        const bool stochastic = o.mixed ? b % 3 == 2 : o.mode == Mode::Stochastic;
        const bool penalties  = o.mixed ? b % 3 != 0 : stochastic && o.counts_active;
        const int used        = acceptance ? active[b] + 1 : 1;
        if (!acceptance || stochastic || penalties || o.mask)
            bytes += std::size_t(used) * kTokenDomain * 2;
        if (o.counts_active && penalties) bytes += std::size_t(kTokenDomain) * 4;
        if (o.mask) bytes += std::size_t(used) * ((kTokenDomain + 31) / 32) * 4;
        if (acceptance) {
            // Full valid prefix: drafts, extent/length input, and all output slots.
            bytes += std::size_t(used - 1 + columns + 6) * 4;
            if (!stochastic && !penalties && !o.mask) bytes += std::size_t(used) * 4;
            if (o.dflash2 && stochastic) bytes += std::size_t(used - 1) * kDFlash2Candidates * 8;
        } else {
            bytes += 8; // Logical position and selected token.
        }
        if (!o.dflash2 && o.counts_active && (stochastic || o.mixed))
            bytes += std::size_t(used) * (penalties ? 4 : 8);
    }
    return bytes;
}

void run_sample(DeviceBuffer& logits, const Options& o, DecisionBenchmark& benchmark) {
    const int batch = o.batch;
    auto counts     = make_counts(batch);
    auto masks      = make_masks(1, batch, o.mask);
    auto configs =
        make_batch_configs(counts, batch, o.mode, o.counts_active, o.top_k, o.mixed, masks, 1);
    auto out       = make_zeros(std::size_t(batch) * 4);
    auto positions = make_i32(std::vector<std::int32_t>(batch, 4096));
    SavedBuffer initial_counts(counts);
    Tensor tl(logits.span(), DType::BF16, {kPhysicalRows, batch}),
        to(out.span(), DType::I32, {batch}), tp(positions.span(), DType::I32, {batch});
    const auto bytes = ops::sampling_workspace_capacity_bytes(kTokenDomain, batch, batch);
    DeviceBuffer backing(device_context().device(), bytes);
    MemoryArena workspace(backing.span());
    measurement::Row row{.op              = "sample",
                         .format          = "bf16",
                         .n               = kPhysicalRows,
                         .k               = kTokenDomain,
                         .batch           = batch,
                         .workspace_bytes = bytes};
    benchmark.run(
        row, o, decision_bytes(o, 1, {}, false),
        [&](CudaExecutionView cuda) {
            ops::sample(tl, to, kTokenDomain,
                        static_cast<const ops::SamplingConfig*>(configs.data()), tp,
                        ops::kSamplePurposeDecode, workspace, cuda);
            row.workspace_peak = workspace.peak_used();
        },
        [&](CudaExecutionView cuda) { initial_counts.restore(cuda); });
}

void run_mtp(DeviceBuffer& logits, const Options& o, DecisionBenchmark& benchmark) {
    const int batch = o.batch, k = o.mtp_k, columns = k + 1;
    auto counts  = make_counts(batch);
    auto masks   = make_masks(columns, batch, o.mask);
    auto configs = make_batch_configs(counts, batch, o.mode, o.counts_active, o.top_k, o.mixed,
                                      masks, columns);
    std::vector<std::int32_t> targets(std::size_t(columns) * batch), drafts(std::size_t(k) * batch);
    for (int b = 0; b < batch; ++b)
        for (int c = 0; c < columns; ++c) {
            const int id             = (17 + (b * columns + c) * 7919) % kTokenDomain;
            targets[b * columns + c] = id;
            if (c < k) drafts[b * k + c] = c == o.reject_at ? (id + 1) % kTokenDomain : id;
        }
    const auto active = extents(o, columns);
    auto dt = make_i32(targets), dd = make_i32(drafts), de = make_i32(active);
    auto lengths  = make_i32(std::vector<std::int32_t>(batch, 4096)),
         anchors  = make_i32(std::vector<std::int32_t>(batch, 17));
    auto licensed = make_zeros(std::size_t(columns) * batch * 4), num = make_zeros(batch * 4),
         accepted = make_zeros(batch * 4);
    SavedBuffer initial_counts(counts), initial_lengths(lengths), initial_anchors(anchors);
    Tensor tt(dt.span(), DType::I32, {columns, batch}),
        tl(logits.span(), DType::BF16, {kPhysicalRows, columns, batch}),
        td(dd.span(), DType::I32, {k, batch}), te(de.span(), DType::I32, {batch}),
        tlen(lengths.span(), DType::I32, {batch}), ta(anchors.span(), DType::I32, {batch}),
        tout(licensed.span(), DType::I32, {columns, batch}), tn(num.span(), DType::I32, {batch}),
        tac(accepted.span(), DType::I32, {batch});
    const auto bytes = ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
        kTokenDomain, k, k, batch, batch);
    DeviceBuffer backing(device_context().device(), bytes);
    MemoryArena workspace(backing.span());
    measurement::Row row{.op              = "accept-greedy-drafts",
                         .format          = "bf16",
                         .valid           = list_counts(active),
                         .n               = kPhysicalRows,
                         .k               = kTokenDomain,
                         .tokens          = columns,
                         .batch           = batch,
                         .workspace_bytes = bytes};
    benchmark.run(
        row, o, decision_bytes(o, columns, active, true),
        [&](CudaExecutionView cuda) {
            ops::speculative_accept_greedy_drafts(
                tt, tl, td, te, tlen, ta, tout, tn, tac, kTokenDomain,
                static_cast<const ops::SamplingConfig*>(configs.data()), workspace, cuda);
            row.workspace_peak = workspace.peak_used();
        },
        [&](CudaExecutionView cuda) {
            initial_counts.restore(cuda);
            initial_lengths.restore(cuda);
            initial_anchors.restore(cuda);
        });
}

void run_dflash2(DeviceBuffer& logits, const Options& options, DecisionBenchmark& benchmark) {
    const int batch          = options.batch;
    const int kDFlash2Drafts = options.drafts, kDFlash2Columns = options.drafts + 1;
    const Mode mode          = options.mode;
    const bool counts_active = options.counts_active;
    const int top_k          = options.top_k;
    auto counts              = make_counts(batch);
    auto masks               = make_masks(kDFlash2Columns, batch, options.mask);
    DeviceBuffer configs     = make_batch_configs(counts, batch, mode, counts_active, top_k,
                                                  options.mixed, masks, kDFlash2Columns);

    std::vector<std::int32_t> target_host(static_cast<std::size_t>(kDFlash2Columns) * batch);
    std::vector<std::int32_t> draft_host(static_cast<std::size_t>(kDFlash2Drafts) * batch);
    std::vector<std::int32_t> candidate_host(static_cast<std::size_t>(kDFlash2Candidates) *
                                             kDFlash2Drafts * batch);
    std::vector<float> q_host(candidate_host.size(), 0.0f);
    for (int row = 0; row < batch; ++row) {
        for (int column = 0; column < kDFlash2Columns; ++column) {
            const int flat                              = row * kDFlash2Columns + column;
            target_host[static_cast<std::size_t>(flat)] = (17 + flat * 7919) % kTokenDomain;
        }
        for (int draft = 0; draft < kDFlash2Drafts; ++draft) {
            const int selected = target_host[static_cast<std::size_t>(row) * kDFlash2Columns +
                                             static_cast<std::size_t>(draft)];
            draft_host[static_cast<std::size_t>(row) * kDFlash2Drafts + draft] = selected;
            const std::size_t base =
                (static_cast<std::size_t>(row) * kDFlash2Drafts + draft) * kDFlash2Candidates;
            for (int candidate = 0; candidate < kDFlash2Candidates; ++candidate) {
                candidate_host[base + candidate] = (selected + candidate) % kTokenDomain;
            }
            q_host[base] = mode == Mode::Greedy ? 1.0f : 0.5f;
            if (mode == Mode::Stochastic) q_host[base + 1] = 0.5f;
            if (draft == options.reject_at) {
                draft_host[static_cast<std::size_t>(row) * kDFlash2Drafts + draft] =
                    candidate_host[base + 1];
                if (mode == Mode::Greedy) {
                    q_host[base]     = 0.0f;
                    q_host[base + 1] = 1.0f;
                }
            }
        }
    }

    DeviceBuffer targets    = make_i32(target_host);
    DeviceBuffer drafts     = make_i32(draft_host);
    DeviceBuffer candidates = make_i32(candidate_host);
    DeviceBuffer proposal_q(ninfer::bench::device_context().device(),
                            q_host.size() * sizeof(float));
    CUDA_CHECK(
        cudaMemcpy(proposal_q.data(), q_host.data(), proposal_q.size(), cudaMemcpyHostToDevice));
    const auto active    = extents(options, kDFlash2Columns);
    DeviceBuffer extents = make_i32(active);
    DeviceBuffer lengths =
        make_i32(std::vector<std::int32_t>(static_cast<std::size_t>(batch), 4096));
    DeviceBuffer anchors = make_i32(std::vector<std::int32_t>(static_cast<std::size_t>(batch), -1));
    DeviceBuffer licensed(ninfer::bench::device_context().device(),
                          static_cast<std::size_t>(kDFlash2Columns) * batch * sizeof(std::int32_t));
    DeviceBuffer licensed_counts(ninfer::bench::device_context().device(),
                                 static_cast<std::size_t>(batch) * sizeof(std::int32_t));
    DeviceBuffer accepted(ninfer::bench::device_context().device(),
                          static_cast<std::size_t>(batch) * sizeof(std::int32_t));

    Tensor ttargets(targets.span(), DType::I32, {kDFlash2Columns, batch});
    Tensor tlogits(logits.span(), DType::BF16, {kPhysicalRows, kDFlash2Columns, batch});
    Tensor tdrafts(drafts.span(), DType::I32, {kDFlash2Drafts, batch});
    Tensor tcandidates(candidates.span(), DType::I32, {kDFlash2Candidates, kDFlash2Drafts, batch});
    Tensor tq(proposal_q.span(), DType::FP32, {kDFlash2Candidates, kDFlash2Drafts, batch});
    Tensor textents(extents.span(), DType::I32, {batch});
    Tensor tlengths(lengths.span(), DType::I32, {batch});
    Tensor tanchors(anchors.span(), DType::I32, {batch});
    Tensor tlicensed(licensed.span(), DType::I32, {kDFlash2Columns, batch});
    Tensor tlicensed_counts(licensed_counts.span(), DType::I32, {batch});
    Tensor taccepted(accepted.span(), DType::I32, {batch});
    const ops::SpeculativeAcceptExecutionEnvelope envelope{
        .all_rows_greedy_without_penalties =
            mode == Mode::Greedy && !options.force_general && !options.mixed && !options.mask,
    };
    const std::size_t workspace_bytes =
        ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
            kTokenDomain, envelope, kDFlash2Drafts, kDFlash2Drafts, batch, batch);
    ninfer::DeviceBuffer workspace_storage(ninfer::bench::device_context().device(),
                                           std::max<std::size_t>(workspace_bytes, 256));
    MemoryArena workspace(workspace_storage.span());
    const auto* config_ptr = static_cast<const ops::SamplingConfig*>(configs.data());
    const auto launch      = [&](CudaExecutionView cuda) {
        ops::speculative_accept_sparse_drafts(
            ttargets, tlogits, tdrafts, tcandidates, tq, textents, tlengths, tanchors, tlicensed,
            tlicensed_counts, taccepted, kTokenDomain, config_ptr, envelope, workspace, cuda);
    };

    SavedBuffer initial_lengths(lengths), initial_anchors(anchors);
    measurement::Row row{.op              = "accept-sparse-drafts",
                         .format          = "bf16",
                         .valid           = list_counts(active),
                         .n               = kPhysicalRows,
                         .k               = kTokenDomain,
                         .tokens          = kDFlash2Columns,
                         .batch           = batch,
                         .workspace_bytes = workspace_bytes};
    benchmark.run(
        row, options, decision_bytes(options, kDFlash2Columns, active, true),
        [&](CudaExecutionView cuda) {
            launch(cuda);
            row.workspace_peak = workspace.peak_used();
        },
        [&](CudaExecutionView cuda) {
            initial_lengths.restore(cuda);
            initial_anchors.restore(cuda);
        },
        envelope.all_rows_greedy_without_penalties);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    try {
        const Options options = parse_args(argc, argv);
        const int max_cols =
            options.matrix ? 48
                           : std::max({options.sample ? options.batch : 1,
                                       options.mtp ? options.batch * (options.mtp_k + 1) : 1,
                                       options.dflash2 ? options.batch * (options.drafts + 1) : 1});
        DeviceBuffer logits = make_logits(max_cols);
        measurement::hardware();
        DecisionBenchmark benchmark(options);
        if (options.matrix) {
            for (int batch : {1, 2, 4, 8}) {
                for (Mode mode : {Mode::Greedy, Mode::Stochastic}) {
                    auto o          = options;
                    o.batch         = batch;
                    o.mode          = mode;
                    o.counts_active = mode == Mode::Stochastic;
                    run_sample(logits, o, benchmark);
                    for (int k = 1; k <= 5; ++k) {
                        o.mtp_k  = k;
                        o.extent = k;
                        run_mtp(logits, o, benchmark);
                    }
                }
            }
        } else {
            if (options.sample) run_sample(logits, options, benchmark);
            if (options.mtp) run_mtp(logits, options, benchmark);
            if (options.dflash2) run_dflash2(logits, options, benchmark);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ninfer_sampling_select_bench: %s\n", e.what());
        return 2;
    }
    return 0;
}
