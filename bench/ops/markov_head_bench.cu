#include "ninfer/ops/markov_head.h"
#include "ninfer/ops/argmax.h"

#include "direct_bf16_weight.cuh"
#include "op_measurement.cuh"

#include <charconv>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

struct Options {
    measurement::Options timing;
    std::vector<int> batches{1, 2, 4, 8, 16, 32};
    std::vector<int> steps{7};
    std::string op = "all";
    int h = 2304, rank = 256, vocab = 200064, domain = 200058, embed_vocab = 201024;
    float softcap = 30.0f;
    int alias     = 0;
};

std::string softcap_text(float value) {
    char buffer[32];
    const auto [end, error] = std::to_chars(buffer, buffer + sizeof(buffer), value);
    if (error != std::errc{}) throw std::runtime_error("cannot format softcap");
    return std::string(buffer, end);
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help") {
            std::cout << "Markov head: --op all|context|step|logits|sample --batches B[,B...] "
                         "--steps D[,D...]\n"
                         "  --h H --rank R --vocab V --domain N --embed-vocab N --softcap F\n"
                         "  --batch and --hidden are also accepted.\n"
                         "  --alias 0|1 (state/logits output reuse, restored before each sample)\n"
                         "Default vocabulary and token domain are 200064 and 200058.\n";
            measurement::help();
            std::exit(0);
        }
        const auto value      = measurement::take(argc, argv, i);
        constexpr int kMinInt = std::numeric_limits<int>::min();
        if (flag == "--op")
            o.op = value;
        else if (flag == "--batch" || flag == "--batches")
            o.batches = measurement::integers(value);
        else if (flag == "--steps")
            o.steps = measurement::integers(value, kMinInt);
        else if (flag == "--hidden" || flag == "--h")
            o.h = measurement::integer(value, kMinInt);
        else if (flag == "--rank")
            o.rank = measurement::integer(value);
        else if (flag == "--vocab")
            o.vocab = measurement::integer(value, kMinInt);
        else if (flag == "--domain")
            o.domain = measurement::integer(value, kMinInt);
        else if (flag == "--embed-vocab")
            o.embed_vocab = measurement::integer(value, kMinInt);
        else if (flag == "--alias")
            o.alias = measurement::integer(value, kMinInt);
        else if (flag == "--softcap") {
            const auto [end, error] =
                std::from_chars(value.data(), value.data() + value.size(), o.softcap);
            if (error != std::errc{} || end != value.data() + value.size())
                throw std::invalid_argument("invalid softcap");
        } else if (!measurement::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option: " + std::string(flag));
    }
    if (o.op != "all" && o.op != "context" && o.op != "step" && o.op != "logits" &&
        o.op != "sample")
        throw std::invalid_argument("invalid markov op");
    const bool context = o.op == "all" || o.op == "context";
    const bool step    = o.op == "all" || o.op == "step";
    const bool logits  = o.op == "all" || o.op == "logits" || o.op == "sample";
    if ((context || step) && o.rank > std::numeric_limits<int>::max() / 3)
        throw std::invalid_argument("rank exceeds the benchmark fixture's row extent");
    if (context && (o.h <= 0 || std::ranges::any_of(o.steps, [](int d) { return d <= 0; })))
        throw std::invalid_argument("context fixture requires positive hidden and step extents");
    if (step && o.embed_vocab <= 0)
        throw std::invalid_argument("step fixture requires a positive embedding vocabulary");
    if (logits && o.vocab <= 0)
        throw std::invalid_argument("logits fixture requires a positive vocabulary");
    if ((step || logits) && (o.alias < 0 || o.alias > 1))
        throw std::invalid_argument("alias must be 0 or 1");
    return o;
}

struct Bench {
    explicit Bench(const Options& options)
        : o(options), softcap(softcap_text(o.softcap)), flush(o.timing.flush_bytes),
          csv(o.timing.csv_out, "token_domain,softcap,embed_vocab,operand_bytes,"
                                "workspace_traffic_bytes,route_bytes,logical_gb_s,"
                                "sustained_reference_us,pct_sustained_reference,useful_tflops") {}

    template <class Launch, class Restore>
    void run(measurement::Row row, std::uint64_t operand_bytes, double flops, Launch launch,
             Restore restore, MemoryArena* arena = nullptr) {
        for (const auto& execution : measurement::selections(o.timing.execution, "graph", "eager"))
            for (const auto& cache : measurement::selections(o.timing.cache, "cold", "warm")) {
                if (arena) arena->reset_peak();
                const auto result =
                    measurement::measure(launch, restore, flush, device_context().execution_view(),
                                         o.timing, execution, cache);
                if (arena) row.workspace_peak = arena->peak_used();
                const auto workspace_traffic_bytes = 2ULL * row.workspace_peak;
                const auto route_bytes             = operand_bytes + workspace_traffic_bytes;
                const auto gbps                    = route_bytes / (1000 * result.gpu.median_us);
                csv.write(
                    row, result, execution, cache,
                    std::to_string(o.domain) + ',' + softcap + ',' + std::to_string(o.embed_vocab) +
                        ',' + std::to_string(operand_bytes) + ',' +
                        std::to_string(workspace_traffic_bytes) + ',' +
                        std::to_string(route_bytes) + ',' + format_metric(gbps, 3, "") + ',' +
                        format_metric(route_bytes / (1000 * kRoofline.sustained_read_gbs), 3, "") +
                        ',' + format_metric(100 * gbps / kRoofline.sustained_read_gbs, 3, "") +
                        ',' + format_metric(flops / (1e6 * result.gpu.median_us), 3, ""));
            }
    }

    const Options& o;
    const std::string softcap;
    L2FlushBuffer flush;
    measurement::Csv csv;
};

void context_cases(Bench& bench) {
    const auto& o  = bench.o;
    const int rows = 3 * o.rank;
    const auto wh  = make_direct_bf16_weight(rows, o.h, 17);
    const auto wf  = make_direct_bf16_weight(rows, o.h, 19);
    auto bias      = make_f32(rows, 23, -0.1f, 0.1f);
    Tensor bias_view(bias.span(), DType::FP32, {rows});
    for (const int b : o.batches) {
        auto frnn = make_bf16(std::size_t(o.h) * b, 29);
        Tensor frnn_view(frnn.span(), DType::BF16, {o.h, b});
        for (const int d : o.steps) {
            const auto columns = std::size_t(b) * d;
            auto hidden        = make_bf16(std::size_t(o.h) * columns, 31);
            auto context       = make_buffer(4 * rows * columns);
            const auto capacity =
                ops::markov_rnn_context_workspace_capacity_bytes(o.h, o.rank, d, b);
            auto scratch = make_buffer(capacity);
            MemoryArena arena(scratch.span());
            Tensor x(hidden.span(), DType::BF16, {o.h, d, b});
            Tensor y(context.span(), DType::FP32, {rows, b, d});
            measurement::Row row{.op              = "markov_rnn_context",
                                 .format          = "bf16",
                                 .n               = rows,
                                 .k               = o.h,
                                 .tokens          = d,
                                 .batch           = b,
                                 .workspace_bytes = capacity};
            const auto bytes = wh.model_weight_bytes() + wf.model_weight_bytes() + bias.size() +
                               hidden.size() + frnn.size() + context.size();
            bench.run(
                row, bytes, 2.0 * rows * o.h * (b + columns),
                [&](auto cuda) {
                    ops::markov_rnn_context(x, frnn_view, wh.weight, wf.weight, bias_view, arena, y,
                                            cuda);
                },
                [](auto) {}, &arena);
        }
    }
}

void step_cases(Bench& bench) {
    const auto& o = bench.o;
    const int r = o.rank, rows = 3 * r;
    auto embed         = make_bf16(std::size_t(o.embed_vocab) * r, 41);
    const auto weights = make_direct_bf16_weight(rows, 2 * r, 37);
    Tensor e(embed.span(), DType::BF16, {r, o.embed_vocab});
    for (const int b : o.batches) {
        const auto count = std::size_t(r) * b;
        auto initial     = make_bf16(count, 43);
        auto state       = make_buffer(2 * count);
        auto latent      = make_buffer(2 * count);
        auto context     = make_f32(3 * count, 47, -1.0f, 1.0f);
        std::vector<int> token_values(b);
        for (int i = 0; i < b; ++i) token_values[i] = int((7919LL * i + 23) % o.embed_vocab);
        auto tokens = measurement::make_i32(token_values);
        Tensor out(state.span(), DType::BF16, {r, b});
        Tensor in(o.alias ? state.span() : initial.span(), DType::BF16, {r, b});
        Tensor z(latent.span(), DType::BF16, {r, b});
        Tensor c(context.span(), DType::FP32, {rows, b});
        Tensor t(tokens.span(), DType::I32, {b});
        measurement::Row row{.op     = "markov_rnn_step",
                             .format = "bf16",
                             .policy = o.alias ? "alias" : "distinct",
                             .n      = rows,
                             .k      = 2 * r,
                             .batch  = b};
        const auto bytes =
            weights.model_weight_bytes() + 8 * count + context.size() + tokens.size();
        bench.run(
            row, bytes, 4.0 * r * rows * b,
            [&](auto cuda) { ops::markov_rnn_step(in, t, c, e, weights.weight, {}, out, z, cuda); },
            [&](auto cuda) {
                if (o.alias)
                    CUDA_CHECK(cudaMemcpyAsync(state.data(), initial.data(), initial.size(),
                                               cudaMemcpyDeviceToDevice, cuda.stream()));
            });
    }
}

void logits_cases(Bench& bench) {
    const auto& o = bench.o;
    const int r = o.rank, v = o.vocab;
    const auto weights = make_direct_bf16_weight(v, r, 53);
    auto bias          = make_bf16(v, 59);
    Tensor bias_view(bias.span(), DType::BF16, {v});
    for (const int b : o.batches) {
        auto latent  = make_bf16(std::size_t(r) * b, 61);
        auto initial = make_bf16(std::size_t(v) * b, 67, -20.0f, 20.0f);
        auto output  = make_buffer(initial.size());
        Tensor z(latent.span(), DType::BF16, {r, b});
        Tensor in(o.alias ? output.span() : initial.span(), DType::BF16, {v, b});
        Tensor out(output.span(), DType::BF16, {v, b});
        measurement::Row row{.op     = "markov_logits_add_softcap",
                             .format = "bf16",
                             .policy = o.alias ? "alias" : "distinct",
                             .n      = v,
                             .k      = r,
                             .batch  = b};
        const auto bytes = 2ULL * o.domain * (1ULL + r + b) + latent.size() + output.size();
        bench.run(
            row, bytes, 2.0 * o.domain * r * b,
            [&](auto cuda) {
                ops::markov_logits_add_softcap(z, weights.weight, bias_view, in, o.domain,
                                               o.softcap, {}, out, cuda);
            },
            [&](auto cuda) {
                if (o.alias)
                    CUDA_CHECK(cudaMemcpyAsync(output.data(), initial.data(), initial.size(),
                                               cudaMemcpyDeviceToDevice, cuda.stream()));
            });
    }
}

void sampling_cases(Bench& bench) {
    const auto& o = bench.o;
    const int r = o.rank, v = o.vocab;
    const auto weights = make_direct_bf16_weight(v, r, 53);
    auto bias          = make_bf16(v, 59);
    const Tensor bias_view(bias.span(), DType::BF16, {v});
    for (const int b : o.batches) {
        auto latent  = make_bf16(std::size_t(r) * b, 61);
        auto initial = make_bf16(std::size_t(v) * b, 67, -20.0f, 20.0f);
        auto output  = make_buffer(initial.size());
        auto ids     = make_buffer(sizeof(int) * b);
        auto probs   = make_buffer(sizeof(float) * b);
        const Tensor z(latent.span(), DType::BF16, {r, b});
        const Tensor base(initial.span(), DType::BF16, {v, b});
        Tensor logits(output.span(), DType::BF16, {v, b});
        Tensor tokens(ids.span(), DType::I32, {b});
        Tensor probabilities(probs.span(), DType::FP32, {b});
        const auto separate_capacity =
            ops::argmax_with_probability_workspace_capacity_bytes(o.domain, b);
        const auto fused_capacity =
            ops::markov_logits_add_softcap_argmax_workspace_capacity_bytes(o.domain, b);
        auto scratch = make_buffer(std::max(separate_capacity, fused_capacity));
        MemoryArena arena(scratch.span());
        const auto input_bytes  = 2ULL * o.domain * (1ULL + r + b) + latent.size();
        const auto result_bytes = ids.size() + probs.size();
        const auto flops        = 2.0 * o.domain * r * b;
        measurement::Row row{.op              = "markov_logits_add_softcap_argmax",
                             .format          = "bf16",
                             .policy          = "separate",
                             .n               = v,
                             .k               = r,
                             .batch           = b,
                             .workspace_bytes = separate_capacity};
        bench.run(
            row, input_bytes + result_bytes + output.size() + 2ULL * o.domain * b, flops,
            [&](auto cuda) {
                ops::markov_logits_add_softcap(z, weights.weight, bias_view, logits, o.domain,
                                               o.softcap, {}, logits, cuda);
                ops::argmax_with_probability(logits, tokens, probabilities, o.domain, arena, cuda);
            },
            [&](auto cuda) {
                CUDA_CHECK(cudaMemcpyAsync(output.data(), initial.data(), initial.size(),
                                           cudaMemcpyDeviceToDevice, cuda.stream()));
            },
            &arena);
        row.policy          = "fused";
        row.workspace_bytes = fused_capacity;
        bench.run(
            row, input_bytes + result_bytes, flops,
            [&](auto cuda) {
                ops::markov_logits_add_softcap_argmax(z, weights.weight, bias_view, base, o.domain,
                                                      o.softcap, {}, tokens, probabilities, arena,
                                                      cuda);
            },
            [](auto) {}, &arena);
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        measurement::hardware();
        Bench bench(options);
        if (options.op == "all" || options.op == "context") context_cases(bench);
        if (options.op == "all" || options.op == "step") step_cases(bench);
        if (options.op == "all" || options.op == "logits") logits_cases(bench);
        if (options.op == "all" || options.op == "sample") sampling_cases(bench);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
