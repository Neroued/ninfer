#include "ninfer/ops/markov_head.h"

#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"

#include <cmath>
#include <limits>

namespace ninfer::test {
void write_markov_weight_taps(std::uint16_t* weights, const std::uint64_t* offsets,
                              const std::uint16_t* values, int count, cudaStream_t stream);
} // namespace ninfer::test

namespace {
using namespace ninfer;
using namespace ninfer::test;
using Words = std::vector<std::uint16_t>;

enum class Mask { Dense, All, Mixed, None };

Device device() { return cuda_environment().context.device(); }

CudaExecutionView execution() { return cuda_environment().context.execution_view(); }

bool active(Mask mask, int row) {
    return mask != Mask::None && (mask != Mask::Mixed || row % 3 != 1);
}

std::vector<int> row_mask(Mask mask, int batch) {
    std::vector<int> result(batch);
    for (int b = 0; b < batch; ++b) result[b] = active(mask, b);
    return result;
}

std::uint16_t word(double x) { return f32_to_bf16(static_cast<float>(x)); }

double real(std::uint16_t x) { return bf16_to_f32(x); }

double bf16(double x) { return real(word(x)); }

struct GateResult {
    std::uint16_t state, latent;
};

GateResult gate_reference(double raw_gate, double raw_candidate, double raw_readout,
                          std::uint16_t previous) {
    const auto round      = [](float value) { return bf16_to_f32(f32_to_bf16(value)); };
    const float gate      = static_cast<float>(bf16(raw_gate));
    const float candidate = static_cast<float>(bf16(raw_candidate));
    const float readout   = static_cast<float>(bf16(raw_readout));
    const float g         = round(1.0f / (1.0f + std::exp(-gate)));
    const float keep      = round(g * bf16_to_f32(previous));
    const float update    = round(round(1.0f - g) * round(std::tanh(candidate)));
    return {f32_to_bf16(keep + update), f32_to_bf16(std::tanh(readout))};
}

Words values(std::size_t n, std::uint32_t seed, double scale = 1.0 / 256) {
    Words result(n);
    parallel_elements(n, [&](std::size_t i) {
        std::uint32_t hash = (static_cast<std::uint32_t>(i) ^ seed) * 0x9e3779b9U;
        hash ^= hash >> 16;
        hash *= 0x85ebca6bU;
        hash ^= hash >> 13;
        const int centered = int((hash >> 8) & 127) - 63;
        result[i]          = word(centered * scale);
    });
    return result;
}

struct WeightFixture {
    WeightFixture(int n, int k, std::uint32_t seed)
        : data(values(std::size_t(n) * k, seed, 1.0 / 512)), storage(to_device(data)),
          view(direct_bf16_weight::device_weight(n, k, storage.span())) {}

    Words data;
    DeviceBuffer storage;
    Weight view;
};

double dot(const std::uint16_t* a, const std::uint16_t* b, int count) {
    double acc = 0.0;
    for (int j = 0; j < count; ++j) acc += real(a[j]) * real(b[j]);
    return acc;
}

template <class T>
std::vector<T> download(const Tensor& t) {
    std::vector<T> data(t.numel());
    CUDA_CHECK(cudaMemcpy(data.data(), t.data, t.bytes(), cudaMemcpyDeviceToHost));
    return data;
}

int compare_words(std::string_view label, const Words& actual, const Words& expected) {
    int failures = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] == expected[i]) continue;
        if (failures++ < 4)
            std::cerr << label << " at " << i << ": got " << real(actual[i]) << " (" << actual[i]
                      << "), expected " << real(expected[i]) << " (" << expected[i] << ")\n";
    }
    if (failures) std::cerr << label << ": " << failures << " mismatches\n";
    return failures != 0;
}

struct ContextFixture {
    ContextFixture(int h, int rank, int steps, int batch)
        : h(h), rows(3 * rank), steps(steps), batch(batch),
          hidden(values(std::size_t(h) * steps * batch, 1)),
          frnn(values(std::size_t(h) * batch, 3)), bias(rows), hidden_storage(to_device(hidden)),
          frnn_storage(to_device(frnn)), bias_storage(device(), sizeof(float) * rows),
          out_storage(device(), sizeof(float) * std::size_t(rows) * batch * steps),
          w_hidden(rows, h, 2), w_frnn(rows, h, 5),
          scratch(device(),
                  ops::markov_rnn_context_workspace_capacity_bytes(h, rank, steps, batch)),
          arena(scratch.span()), x(hidden_storage.span(), DType::BF16, {h, steps, batch}),
          f(frnn_storage.span(), DType::BF16, {h, batch}),
          bias_view(bias_storage.span(), DType::FP32, {rows}),
          out(out_storage.span(), DType::FP32, {rows, batch, steps}) {
        for (int c = 0; c < rows; ++c) bias[c] = (c % 23 - 11) / 64.0f;
        bias_storage.copy_from_host(bias.data(), bias_storage.size());
    }

    void run() {
        ops::markov_rnn_context(x, f, w_hidden.view, w_frnn.view, bias_view, arena, out,
                                execution());
    }

    std::vector<float> reference() const {
        std::vector<float> result(std::size_t(rows) * batch * steps);
        parallel_rows(
            rows * batch, std::uint64_t(rows) * batch * (steps + 1) * h, [&](int first, int end) {
                for (int i = first; i < end; ++i) {
                    const int b = i / rows, c = i % rows;
                    const auto w_col  = std::size_t(c) * h;
                    const double base = bias[c] + dot(w_frnn.data.data() + w_col,
                                                      frnn.data() + std::size_t(b) * h, h);
                    for (int s = 0; s < steps; ++s) {
                        const auto input = (std::size_t(b) * steps + s) * h;
                        result[(std::size_t(s) * batch + b) * rows + c] = static_cast<float>(
                            base + dot(w_hidden.data.data() + w_col, hidden.data() + input, h));
                    }
                }
            });
        return result;
    }

    int h, rows, steps, batch;
    Words hidden, frnn;
    std::vector<float> bias;
    DeviceBuffer hidden_storage, frnn_storage, bias_storage, out_storage;
    WeightFixture w_hidden, w_frnn;
    DeviceBuffer scratch;
    MemoryArena arena;
    Tensor x, f, bias_view, out;
};

struct StepFixture {
    StepFixture(int rank, int batch, int vocab, bool in_place)
        : rank(rank), batch(batch), vocab(vocab), initial(values(std::size_t(rank) * batch, 19)),
          embed(values(std::size_t(rank) * vocab, 13)), w(3 * rank, 2 * rank, 17),
          state(to_device(initial)), next(device(), state.size()), latent(device(), state.size()),
          embedding(to_device(embed)), tokens(device(), sizeof(int) * batch),
          mask(device(), sizeof(int) * batch), in(state.span(), DType::BF16, {rank, batch}),
          out(in_place ? state.span() : next.span(), DType::BF16, {rank, batch}),
          z(latent.span(), DType::BF16, {rank, batch}),
          e(embedding.span(), DType::BF16, {rank, vocab}), t(tokens.span(), DType::I32, {batch}),
          live(mask.span(), DType::I32, {batch}) {}

    std::vector<int> token_ids(Mask m, int phase = 0) const {
        std::vector<int> picks(batch, std::numeric_limits<int>::max());
        for (int b = 0; b < batch; ++b)
            if (active(m, b)) picks[b] = (vocab - 1 - (7919 * b + phase * 13) % vocab);
        return picks;
    }

    void reset(Mask m, int phase = 0) {
        auto origin = initial;
        for (int b = 0; b < batch; ++b)
            if (!active(m, b)) std::fill_n(origin.data() + b * rank, rank, std::uint16_t{0x7fc0});
        state.copy_from_host(origin.data(), state.size());
        const auto picks = token_ids(m, phase);
        tokens.copy_from_host(picks.data(), tokens.size());
        const auto mask_values = row_mask(m, batch);
        mask.copy_from_host(mask_values.data(), mask.size());
    }

    void run(const Tensor& ctx, bool dense) {
        ops::markov_rnn_step(in, t, ctx, e, w.view, dense ? Tensor{} : live, out, z, execution());
    }

    std::pair<Words, Words> reference(const Words& old, const float* ctx, Mask m,
                                      int phase = 0) const {
        Words state_out(old.size()), latent_out(old.size());
        const auto picks = token_ids(m, phase);
        const int k      = 2 * rank;
        for (int b = 0; b < batch; ++b) {
            if (!active(m, b)) continue;
            const auto* state_row = old.data() + b * rank;
            const auto* embedding = embed.data() + std::size_t(picks[b]) * rank;
            for (int c = 0; c < rank; ++c) {
                const auto raw = [&](int i) {
                    const auto* weights = w.data.data() + std::size_t(i) * k;
                    return ctx[b * 3 * rank + i] + dot(weights, state_row, rank) +
                           dot(weights + rank, embedding, rank);
                };
                const auto result =
                    gate_reference(raw(c), raw(c + rank), raw(c + 2 * rank), state_row[c]);
                state_out[b * rank + c]  = result.state;
                latent_out[b * rank + c] = result.latent;
            }
        }
        return {std::move(state_out), std::move(latent_out)};
    }

    int rank, batch, vocab;
    Words initial, embed;
    WeightFixture w;
    DeviceBuffer state, next, latent, embedding, tokens, mask;
    Tensor in, out, z, e, t, live;
};

struct LogitsFixture {
    LogitsFixture(int rank, int batch, int vocab, int domain, float softcap, bool in_place)
        : rank(rank), batch(batch), vocab(vocab), domain(domain), softcap(softcap),
          base(values(std::size_t(vocab) * batch, 23, 0.5)), bias(values(vocab, 29)),
          w(vocab, rank, 31), base_storage(to_device(base)), bias_storage(to_device(bias)),
          out_storage(device(), base_storage.size()),
          x(base_storage.span(), DType::BF16, {vocab, batch}),
          bias_view(bias_storage.span(), DType::BF16, {vocab}),
          out(in_place ? base_storage.span() : out_storage.span(), DType::BF16, {vocab, batch}) {}

    void reset(Mask m) {
        auto inputs = base;
        for (int b = 0; b < batch; ++b)
            for (int v = active(m, b) ? domain : 0; v < vocab; ++v)
                inputs[std::size_t(b) * vocab + v] = 0x7fc0;
        base_storage.copy_from_host(inputs.data(), base_storage.size());
    }

    void run(const Tensor& latent, const Tensor& live) {
        ops::markov_logits_add_softcap(latent, w.view, bias_view, x, domain, softcap, live, out,
                                       execution());
    }

    Words reference(const Words& latent, Mask mask) const {
        Words result(std::size_t(vocab) * batch);
        parallel_rows(batch * vocab, std::uint64_t(vocab) * batch * rank, [&](int first, int end) {
            for (int i = first; i < end; ++i) {
                const int b = i / vocab, v = i % vocab;
                if (!active(mask, b) || v >= domain) continue;
                const double delta = bf16(dot(w.data.data() + std::size_t(v) * rank,
                                              latent.data() + std::size_t(b) * rank, rank) +
                                          real(bias[v]));
                const double sum   = bf16(real(base[i]) + delta);
                const float x      = static_cast<float>(sum) / softcap;
                result[i]          = word(softcap * static_cast<float>(std::tanh(double(x))));
            }
        });
        return result;
    }

    int rank, batch, vocab, domain;
    float softcap;
    Words base, bias;
    WeightFixture w;
    DeviceBuffer base_storage, bias_storage, out_storage;
    Tensor x, bias_view, out;
};

int test_context(int h, int r, int d, int b) {
    ContextFixture f(h, r, d, b);
    f.run();
    cuda_synchronize(execution().stream());
    const auto actual = download<float>(f.out);
    if (actual != f.reference() || f.arena.used()) {
        std::cerr << "incorrect context or workspace cursor\n";
        return 1;
    }
    return 0;
}

int test_step(int r, int b, int vocab, Mask mask, bool alias) {
    StepFixture f(r, b, vocab, alias);
    f.reset(mask);
    const Words sentinel(std::size_t(r) * b, word(1.0));
    if (!alias) f.next.copy_from_host(sentinel.data(), f.next.size());
    f.latent.copy_from_host(sentinel.data(), f.latent.size());
    std::vector<float> context(3 * r * b);
    for (int i = 0; i < 3 * r * b; ++i)
        context[i] = active(mask, i / (3 * r)) ? (i % 113 - 56) / 16.0f
                                               : std::numeric_limits<float>::quiet_NaN();
    auto buffer = to_device(context);
    Tensor ctx(buffer.span(), DType::FP32, {3 * r, b});
    f.run(ctx, mask == Mask::Dense);
    cuda_synchronize(execution().stream());
    const auto [state, latent] = f.reference(f.initial, context.data(), mask);
    return compare_words("state", download<std::uint16_t>(f.out), state) +
           compare_words("latent", download<std::uint16_t>(f.z), latent);
}

int test_step_sigmoid_overflow() {
    constexpr int rank = 16;
    StepFixture f(rank, 1, 1, false);
    std::fill(f.w.data.begin(), f.w.data.end(), 0);
    f.w.storage.copy_from_host(f.w.data.data(), f.w.storage.size());
    std::fill(f.initial.begin(), f.initial.end(), word(1.0));
    f.reset(Mask::Dense);

    std::vector<float> context(3 * rank, 0.0f);
    Words expected_state(rank), expected_latent(rank);
    for (int c = 0; c < rank; ++c) {
        context[c]        = c % 2 == 0 ? -90.0f : 90.0f;
        expected_state[c] = c % 2 == 0 ? 0 : word(1.0);
    }
    // expf(90) overflows in FP32. FP64 sigmoid followed by BF16 rounding would keep 0x0009.
    const auto [oracle_state, oracle_latent] = f.reference(f.initial, context.data(), Mask::Dense);
    auto storage                             = to_device(context);
    Tensor ctx(storage.span(), DType::FP32, {3 * rank, 1});
    f.run(ctx, true);
    cuda_synchronize(execution().stream());
    return compare_words("sigmoid overflow oracle state", oracle_state, expected_state) +
           compare_words("sigmoid overflow oracle latent", oracle_latent, expected_latent) +
           compare_words("sigmoid overflow state", download<std::uint16_t>(f.out), expected_state) +
           compare_words("sigmoid overflow latent", download<std::uint16_t>(f.z), expected_latent);
}

int test_step_large_in_place() {
    constexpr int rank = 24576, prefix = 22528, tail = rank - prefix;
    constexpr int rows = 3 * rank, width = 2 * rank;
    constexpr int vocab = 3, token = 2;
    constexpr double prefix_coeff[]{0.5, -0.75, 0.25};
    constexpr double tail_coeff[]{-0.375, 1.0, 0.625};
    constexpr double embed_coeff[]{0.25, -0.5, 1.25};

    Words initial(rank), embedding(std::size_t(rank) * vocab);
    for (int c = 0; c < rank; ++c) initial[c] = word((1 + (c * 11) % 31) / 32.0);
    for (int i = 0; i < rank * vocab; ++i) embedding[i] = word(((i * 7) % 29 - 14) / 32.0);

    // Three taps per projection row. Tail channels depend on already updated prefix
    // channels and, except at the wrap, on already updated tail channels.
    std::vector<std::uint64_t> offsets;
    Words coefficients;
    offsets.reserve(3 * rows);
    coefficients.reserve(3 * rows);
    std::vector<float> context(rows);
    std::vector<double> raw(rows);
    for (int row = 0; row < rows; ++row) {
        const int c = row % rank, plane = row / rank;
        const int shared_col   = (c * 5 + 197 * (plane + 1)) % prefix;
        const int register_col = prefix + (c + tail - 37 * (plane + 1)) % tail;
        const int embed_col    = (c * 7 + 59 * (plane + 1)) % rank;
        context[row]           = (row % 23 - 11) / 32.0f;
        raw[row]               = context[row] + prefix_coeff[plane] * real(initial[shared_col]) +
                   tail_coeff[plane] * real(initial[register_col]) +
                   embed_coeff[plane] * real(embedding[token * rank + embed_col]);
        const auto offset = std::uint64_t(row) * width;
        offsets.insert(offsets.end(),
                       {offset + shared_col, offset + register_col, offset + rank + embed_col});
        coefficients.insert(coefficients.end(), {word(prefix_coeff[plane]), word(tail_coeff[plane]),
                                                 word(embed_coeff[plane])});
    }
    Words expected_state(rank), expected_latent(rank);
    for (int c = 0; c < rank; ++c) {
        const auto expected = gate_reference(raw[c], raw[c + rank], raw[c + 2 * rank], initial[c]);
        expected_state[c]   = expected.state;
        expected_latent[c]  = expected.latent;
    }

    auto state          = to_device(initial);
    auto embed          = to_device(embedding);
    auto tokens         = to_device(std::vector<int>{token});
    auto ctx_storage    = to_device(context);
    auto device_offsets = to_device(offsets);
    auto device_coeffs  = to_device(coefficients);
    DeviceBuffer latent(device(), state.size());
    DeviceBuffer weights(device(), std::size_t(rows) * width * sizeof(std::uint16_t));
    const auto exec = execution();
    CUDA_CHECK(cudaMemsetAsync(weights.data(), 0, weights.size(), exec.stream()));
    write_markov_weight_taps(static_cast<std::uint16_t*>(weights.data()),
                             static_cast<const std::uint64_t*>(device_offsets.data()),
                             static_cast<const std::uint16_t*>(device_coeffs.data()),
                             static_cast<int>(offsets.size()), exec.stream());

    Tensor state_view(state.span(), DType::BF16, {rank, 1});
    Tensor latent_view(latent.span(), DType::BF16, {rank, 1});
    const Tensor embed_view(embed.span(), DType::BF16, {rank, vocab});
    const Tensor token_view(tokens.span(), DType::I32, {1});
    const Tensor ctx(ctx_storage.span(), DType::FP32, {rows, 1});
    const auto w = direct_bf16_weight::device_weight(rows, width, weights.span());
    ops::markov_rnn_step(state_view, token_view, ctx, embed_view, w, {}, state_view, latent_view,
                         exec);
    cuda_synchronize(exec.stream());
    return compare_words("large in-place state", download<std::uint16_t>(state_view),
                         expected_state) +
           compare_words("large in-place latent", download<std::uint16_t>(latent_view),
                         expected_latent);
}

int test_logits(int r, int b, int v, int domain, float cap, Mask mask, bool alias) {
    LogitsFixture f(r, b, v, domain, cap, alias);
    auto latent = values(std::size_t(r) * b, 37);
    for (int row = 0; row < b; ++row)
        if (!active(mask, row)) std::fill_n(latent.data() + row * r, r, std::uint16_t{0x7fc0});
    auto zbuf    = to_device(latent);
    auto maskbuf = to_device(row_mask(mask, b));
    Tensor z(zbuf.span(), DType::BF16, {r, b}), live(maskbuf.span(), DType::I32, {b});
    f.reset(mask);
    f.run(z, mask == Mask::Dense ? Tensor{} : live);
    cuda_synchronize(execution().stream());
    return compare_words("logits", download<std::uint16_t>(f.out), f.reference(latent, mask));
}

struct Graph {
    ~Graph() {
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
    }

    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
};

int test_graph_chain() {
    constexpr int h = 512, r = 64, b = 5, d = 3, v = 1024, domain = 1019;
    ContextFixture context(h, r, d, b);
    StepFixture step(r, b, v, true);
    std::vector<LogitsFixture> logits;
    logits.reserve(d);
    for (int s = 0; s < d; ++s) logits.emplace_back(r, b, v, domain, 15.0f, true);
    const auto stream = execution().stream();
    Graph capture;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    context.run();
    for (int s = 0; s < d; ++s) {
        const auto slice = context.out.slice(2, s, 1).view({3 * r, b});
        step.run(slice, false);
        logits[s].run(step.z, step.live);
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &capture.graph));
    CUDA_CHECK(cudaGraphInstantiate(&capture.exec, capture.graph, 0));

    const auto ctx = context.reference();
    int failures   = 0;
    for (const Mask mask : {Mask::Mixed, Mask::All, Mask::None}) {
        const int phase = static_cast<int>(mask);
        step.reset(mask, phase);
        for (auto& logit : logits) logit.reset(mask);
        CUDA_CHECK(cudaGraphLaunch(capture.exec, stream));
        cuda_synchronize(stream);
        Words state = step.initial;
        for (int s = 0; s < d; ++s) {
            auto [next, latent] = step.reference(state, ctx.data() + s * b * 3 * r, mask, phase);
            failures += compare_words("chain logits", download<std::uint16_t>(logits[s].out),
                                      logits[s].reference(latent, mask));
            if (s + 1 == d)
                failures += compare_words("chain latent", download<std::uint16_t>(step.z), latent);
            state = std::move(next);
        }
        failures += compare_words("chain state", download<std::uint16_t>(step.out), state);
    }
    return failures;
}

template <class F>
void rejects(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("accepted invalid Markov operands");
}

int test_rejection() {
    ContextFixture c(512, 64, 3, 2);
    StepFixture s(64, 2, 1024, false);
    LogitsFixture l(64, 2, 1024, 1019, 15.0f, false);
    auto ctx                  = c.out.slice(2, 0, 1).view({192, 2});
    const auto submit_context = [&](const Tensor& hidden, const Weight& w, MemoryArena& scratch) {
        ops::markov_rnn_context(hidden, c.f, w, c.w_frnn.view, c.bias_view, scratch, c.out,
                                execution());
    };
    const auto submit_step = [&](const Tensor& tokens, const Tensor& mask, Tensor& state) {
        ops::markov_rnn_step(s.in, tokens, ctx, s.e, s.w.view, mask, state, s.z, execution());
    };
    const auto submit_logits = [&](int domain, float softcap, Tensor& dest) {
        ops::markov_logits_add_softcap(s.z, l.w.view, l.bias_view, l.x, domain, softcap, {}, dest,
                                       execution());
    };
    rejects([&] { (void)ops::markov_rnn_context_workspace_capacity_bytes(513, 64, 3, 2); });
    rejects([&] { (void)ops::markov_rnn_context_workspace_capacity_bytes(512, 63, 3, 2); });
    rejects([&] { (void)ops::markov_rnn_context_workspace_capacity_bytes(512, 64, 3, 0); });
    auto bad_weight = c.w_hidden.view;
    --bad_weight.payload_bytes;
    rejects([&] { submit_context(c.x, bad_weight, c.arena); });
    auto bad_hidden = c.x;
    bad_hidden.nb[2] += 16;
    rejects([&] { submit_context(bad_hidden, c.w_hidden.view, c.arena); });
    MemoryArena overlapping(c.scratch.span());
    Tensor overlapping_hidden(c.scratch.span(), DType::BF16, {512, 3, 2});
    rejects([&] { submit_context(overlapping_hidden, c.w_hidden.view, overlapping); });
    MemoryArena too_small(c.scratch.span().subspan(0, 16));
    try {
        submit_context(c.x, c.w_hidden.view, too_small);
        throw std::runtime_error("accepted undersized workspace");
    } catch (const std::bad_alloc&) {}
    auto invalid_tokens   = s.t;
    invalid_tokens.device = Device::cpu();
    rejects([&] { submit_step(invalid_tokens, {}, s.out); });
    rejects([&] { submit_step(s.t, s.in, s.out); });
    rejects([&] { submit_step(s.t, {}, s.z); });
    auto partial_state = s.in;
    partial_state.data = static_cast<std::byte*>(partial_state.data) + 16;
    rejects([&] { submit_step(s.t, {}, partial_state); });
    auto misaligned_state = s.out;
    misaligned_state.data = static_cast<std::byte*>(misaligned_state.data) + 2;
    rejects([&] { submit_step(s.t, {}, misaligned_state); });
    auto partial_logits = l.x;
    partial_logits.data = static_cast<std::byte*>(partial_logits.data) + 16;
    rejects([&] { submit_logits(1019, 15.0f, partial_logits); });
    rejects([&] { submit_logits(1025, 15.0f, l.out); });
    rejects([&] { submit_logits(0, 15.0f, l.out); });
    rejects([&] { submit_logits(1019, std::numeric_limits<float>::infinity(), l.out); });
    rejects([&] { submit_logits(1019, std::numeric_limits<float>::quiet_NaN(), l.out); });
    rejects([&] { submit_logits(1019, 0.0f, l.out); });
    return c.arena.used() || overlapping.used() || too_small.used();
}

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    suite.add("context_primary", [] { return test_context(2304, 256, 7, 1); });
    suite.add("context_primary_batched", [] { return test_context(2304, 256, 7, 8); });
    suite.add("context_single_step", [] { return test_context(512, 64, 1, 2); });
    suite.add("context_many_columns", [] { return test_context(512, 128, 11, 13); });
    suite.add("context_k_tail", [] { return test_context(520, 48, 3, 3); });
    suite.add("context_batch_tile", [] { return test_context(512, 64, 3, 24); });
    suite.add("context_batch_aligned", [] { return test_context(2304, 64, 1, 16); });
    suite.add("step_primary", [] { return test_step(256, 1, 201024, Mask::Dense, false); });
    suite.add("step_primary_mixed", [] { return test_step(256, 8, 1024, Mask::Mixed, false); });
    suite.add("step_mixed", [] { return test_step(128, 13, 1024, Mask::Mixed, false); });
    suite.add("step_rank_tail", [] { return test_step(48, 2, 33, Mask::All, true); });
    suite.add("step_odd_batch", [] { return test_step(64, 3, 65, Mask::Mixed, false); });
    suite.add("step_primary_in_place", [] { return test_step(256, 8, 1024, Mask::Mixed, true); });
    suite.add("step_all_inactive", [] { return test_step(64, 5, 1024, Mask::None, true); });
    suite.add("step_batch_aligned", [] { return test_step(128, 16, 1024, Mask::All, true); });
    suite.add("step_batch_tile", [] { return test_step(64, 24, 1019, Mask::Mixed, false); });
    suite.add("step_sigmoid_overflow", test_step_sigmoid_overflow);
    suite.add("step_large_in_place", test_step_large_in_place);
    suite.add("logits_small",
              [] { return test_logits(64, 1, 1024, 1019, 15, Mask::Dense, false); });
    suite.add("logits_rank_tail",
              [] { return test_logits(48, 2, 1021, 1021, 50, Mask::Dense, false); });
    suite.add("logits_rank_tiled",
              [] { return test_logits(288, 3, 1031, 1, 15, Mask::Mixed, false); });
    suite.add("logits_odd_batch",
              [] { return test_logits(128, 5, 32008, 32003, 50, Mask::Mixed, false); });
    suite.add("logits_primary",
              [] { return test_logits(256, 8, 200064, 200058, 30, Mask::Dense, false); });
    suite.add("logits_batch_tail",
              [] { return test_logits(64, 13, 1024, 1019, 15, Mask::Mixed, true); });
    suite.add("logits_full_domain",
              [] { return test_logits(64, 16, 201024, 201024, 30, Mask::All, true); });
    suite.add("logits_batch_tile",
              [] { return test_logits(256, 24, 1024, 1019, 50, Mask::Mixed, false); });
    suite.add("logits_all_inactive",
              [] { return test_logits(256, 24, 1024, 1019, 50, Mask::None, true); });
    suite.add("logits_batch_tiled",
              [] { return test_logits(64, 33, 1021, 1019, 15, Mask::Mixed, true); });
    suite.add("logits_subnormal_softcap", [] {
        return test_logits(64, 3, 1024, 1019, std::numeric_limits<float>::denorm_min(), Mask::Mixed,
                           true);
    });
    suite.add("graph_recurrent_chain", test_graph_chain);
    suite.add("rejection", test_rejection);
    return suite.main(argc, argv);
}
