#include "ninfer/ops/loop_message.h"
#include "core/buffer.h"
#include "cuda_fixture.h"
#include "ops/suite.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::ops;

namespace {
constexpr float kPairEps = 1.e-6F, kNormEps = 3.e-5F;
constexpr auto capacity = loop_message_blend_rmsnorm_workspace_capacity_bytes;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Round directly from FP64 to the BF16 grid, including ties and subnormals.
float bf16(double x) {
    int exponent;
    std::frexp(x, &exponent);
    const int quantum = std::max(exponent - 8, -133);
    return float(std::ldexp(std::nearbyint(std::ldexp(x, -quantum)), quantum));
}

std::uint16_t word(float x) {
    return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(x) >> 16);
}

float value(std::uint16_t x) { return std::bit_cast<float>(std::uint32_t(x) << 16); }

template <class T>
std::vector<T> read(const DeviceBuffer& b, std::size_t n, std::size_t offset = 0) {
    std::vector<T> result(n);
    b.copy_to_host(result.data(), n * sizeof(T), offset);
    return result;
}

void upload(DeviceBuffer& b, const std::vector<float>& x, bool narrow = false) {
    if (!narrow) {
        b.copy_from_host(x.data(), x.size() * sizeof(float));
        return;
    }
    std::vector<std::uint16_t> words(x.size());
    std::transform(x.begin(), x.end(), words.begin(), word);
    b.copy_from_host(words.data(), words.size() * sizeof(std::uint16_t));
}

struct Shape {
    int d, r, s, m, sd, sn;
};

struct Fixture {
    explicit Fixture(Shape shape, int last = 0)
        : g(shape), device(cuda_environment().context.device()), d(std::size_t(g.d) * g.m * g.sd),
          msg(std::size_t(g.d) * g.m * g.s), down(std::size_t(2) * g.d * g.r * g.s),
          up(down.size()), gamma(std::size_t(g.d) * g.sn), dd(device, d.size() * 4),
          dm(device, msg.size() * 4), dw(device, down.size() * 2), uw(device, up.size() * 2),
          gw(device, gamma.size() * 2), out(device, msg.size() * 2 + 32),
          bytes(capacity(g.d, g.r, g.s, g.m, last ? last : g.m)), scratch(device, bytes + 32),
          arena(scratch.span()), dest(dd.span(), DType::FP32, {g.d, g.m, g.sd}),
          message(dm.span(), DType::FP32, {g.d, g.m, g.s}),
          norm(gw.span(), DType::BF16, {g.d, g.sn}),
          output(out.span().subspan(16, msg.size() * 2), DType::BF16, {g.d, g.m, g.s}),
          weights{Tensor(dw.span(), DType::BF16, {2 * g.d, g.r, g.s}),
                  Tensor(uw.span(), DType::BF16, {g.r, 2 * g.d, g.s})} {
        scratch.fill(0x5a);
        (void)arena.alloc_bytes(16, 16);
        out.fill(0xa5);
    }

    void seed(unsigned phase, bool zero = false, bool saturated = false) {
        std::mt19937 rng(92813 + phase);
        auto fill = [&](std::vector<float>& x, float scale, bool narrow) {
            for (auto& v : x) {
                v = (int(rng() % 65537) - 32768) * (scale / 32768);
                if (narrow) v = bf16(v);
            }
        };
        fill(d, 1.2F, false);
        fill(msg, 0.9F, false);
        fill(down, 2.F / std::sqrt(float(g.d)), true);
        fill(up, (saturated ? 80.F : 1.8F) / std::sqrt(float(g.r)), true);
        fill(gamma, 1.25F, true);
        if (zero) {
            std::fill(d.begin(), d.end(), 0.F);
            std::fill(msg.begin(), msg.end(), 0.F);
        } else if (phase % 2) {
            for (std::size_t i = 0; i < d.size(); i += 5) d[i] *= 1.e-3F;
            for (std::size_t i = 0; i < msg.size(); i += 7) msg[i] *= 3.F;
        }
        upload_operands();
    }

    void upload_operands() {
        upload(dd, d);
        upload(dm, msg);
        upload(dw, down, true);
        upload(uw, up, true);
        upload(gw, gamma, true);
    }

    void run(CudaExecutionView cuda) {
        loop_message_blend_rmsnorm(dest, message, weights, norm, pair_eps, norm_eps, arena, output,
                                   cuda);
    }

    std::vector<float> oracle() const {
        const auto [width, rank, streams, tokens, sd, sn] = g;
        std::vector<float> expected(msg.size());
        parallel_rows(streams * tokens, std::uint64_t(streams) * tokens * width * rank * 4,
                      [&](int first, int last) {
                          std::vector<float> pair(2 * width), h(rank);
                          std::vector<double> blended(width);
                          for (int row = first; row < last; ++row) {
                              const int s = row / tokens, m = row % tokens;
                              const auto* dx = d.data() + std::size_t(sd == 1 ? m : row) * width;
                              const auto* mx = msg.data() + std::size_t(row) * width;
                              const auto* w  = down.data() + std::size_t(s) * rank * (2 * width);
                              const auto* u  = up.data() + std::size_t(s) * (2 * width) * rank;
                              const auto* norm_w = gamma.data() + (sn == 1 ? 0 : s) * width;
                              double ds = 0, ms = 0;
                              for (int i = 0; i < width; ++i) {
                                  ds += double(dx[i]) * dx[i];
                                  ms += double(mx[i]) * mx[i];
                              }
                              const double di = 1 / std::sqrt(ds / width + pair_eps);
                              const double mi = 1 / std::sqrt(ms / width + pair_eps);
                              for (int i = 0; i < width; ++i) {
                                  pair[i]         = bf16(dx[i] * di);
                                  pair[i + width] = bf16(mx[i] * mi);
                              }
                              for (int i = 0; i < rank; ++i) {
                                  double sum = 0;
                                  for (int j = 0; j < 2 * width; ++j)
                                      sum += double(w[i * (2 * width) + j]) * pair[j];
                                  h[i] = bf16(sum);
                              }
                              double squares = 0;
                              for (int i = 0; i < width; ++i) {
                                  double gain = 0, forget = 0;
                                  for (int j = 0; j < rank; ++j) {
                                      gain += double(u[i * rank + j]) * h[j];
                                      forget += double(u[(i + width) * rank + j]) * h[j];
                                  }
                                  const double x =
                                      (1 - std::tanh(forget)) * dx[i] + std::tanh(gain) * mx[i];
                                  blended[i] = x;
                                  squares += x * x;
                              }
                              const double inv = 1 / std::sqrt(squares / width + norm_eps);
                              for (int i = 0; i < width; ++i)
                                  expected[std::size_t(row) * width + i] =
                                      bf16(blended[i] * inv * norm_w[i]);
                          }
                      });
        return expected;
    }

    void verify(double absolute_floor = 0, bool exact = false) const {
        const auto expected = oracle();
        const auto payload  = read<std::uint16_t>(out, expected.size(), 16);
        double diff = 0, ref = 0, max_error = 0;
        for (std::size_t i = 0; i < payload.size(); ++i) {
            const float actual = value(payload[i]);
            require(std::isfinite(actual), "nonfinite result");
            if (exact)
                require(payload[i] == word(expected[i]), "BF16 word differs from FP64 oracle");
            const double delta = actual - double(expected[i]);
            diff += delta * delta;
            ref += double(expected[i]) * expected[i];
            max_error = std::max(max_error, std::abs(delta));
        }
        const double squared_limit =
            std::max(ref * 0.0025 * 0.0025, payload.size() * absolute_floor * absolute_floor);
        if (diff > squared_limit || max_error > 0.04)
            throw std::runtime_error("oracle mismatch, L2 relative " +
                                     std::to_string(std::sqrt(diff / std::max(ref, 1.e-20))) +
                                     ", max absolute " + std::to_string(max_error));
        for (auto offset : {std::size_t(0), 16 + expected.size() * 2}) {
            const auto guard = read<std::uint8_t>(out, 16, offset);
            require(std::all_of(guard.begin(), guard.end(), [](auto x) { return x == 0xa5; }),
                    "output guards changed");
        }
        for (auto offset : {std::size_t(0), 16 + bytes}) {
            const auto guard = read<std::uint8_t>(scratch, 16, offset);
            require(std::all_of(guard.begin(), guard.end(), [](auto x) { return x == 0x5a; }),
                    "scratch guards changed");
        }
        require(arena.used() == 16, "workspace cursor changed");
        require(arena.peak_used() == 16 + capacity(g.d, g.r, g.s, g.m, g.m),
                "queried capacity differs from actual high-water mark");
        require(read<float>(dd, d.size()) == d && read<float>(dm, msg.size()) == msg,
                "input was modified");
        auto same_bf16 = [](const DeviceBuffer& b, const std::vector<float>& x) {
            const auto actual = read<std::uint16_t>(b, x.size());
            return std::equal(actual.begin(), actual.end(), x.begin(),
                              [](auto got, float want) { return got == word(want); });
        };
        require(same_bf16(dw, down) && same_bf16(uw, up) && same_bf16(gw, gamma),
                "weight was modified");
    }

    Shape g;
    Device device;
    std::vector<float> d, msg, down, up, gamma;
    DeviceBuffer dd, dm, dw, uw, gw, out;
    std::size_t bytes;
    DeviceBuffer scratch;
    MemoryArena arena;
    Tensor dest, message, norm, output;
    LoopMessageWeights weights;
    float pair_eps = kPairEps, norm_eps = kNormEps;
};

template <class Error = std::invalid_argument, class F>
void rejects(F&& f) {
    try {
        f();
    } catch (const Error&) { return; }
    throw std::runtime_error("invalid loop message call was accepted");
}

int validation() {
    auto cuda = cuda_environment().context.execution_view();
    Fixture f({256, 32, 2, 3, 2, 1});
    auto call = [&](const Tensor& d, const Tensor& m, const LoopMessageWeights& w,
                    const Tensor& gamma, Tensor y, MemoryArena& ws, float ep = kPairEps,
                    float en = kNormEps) {
        loop_message_blend_rmsnorm(d, m, w, gamma, ep, en, ws, y, cuda);
    };
    Tensor y = f.output;
    for (const auto* input : {&f.dest, &f.message, &f.weights.down, &f.weights.up, &f.norm}) {
        y.data = input->data;
        rejects([&] { call(f.dest, f.message, f.weights, f.norm, y, f.arena); });
    }
    y.data = static_cast<char*>(f.message.data) + 16;
    rejects([&] { call(f.dest, f.message, f.weights, f.norm, y, f.arena); });
    auto y0 = f.output;
    for (auto* backing : {&f.dd, &f.dm, &f.dw, &f.uw, &f.gw, &f.out}) {
        MemoryArena alias(backing->span());
        rejects([&] { call(f.dest, f.message, f.weights, f.norm, y0, alias); });
        require(alias.used() == 0, "workspace cursor changed on failure");
    }
    MemoryArena short_arena(f.scratch.span().subspan(0, f.bytes - 1));
    rejects<std::bad_alloc>([&] { call(f.dest, f.message, f.weights, f.norm, y0, short_arena); });
    HostBuffer host(f.bytes);
    MemoryArena cpu(host.span());
    rejects([&] { call(f.dest, f.message, f.weights, f.norm, y0, cpu); });

    for (auto eps : {0.F, -1.F, std::numeric_limits<float>::infinity(),
                     std::numeric_limits<float>::quiet_NaN()}) {
        rejects([&] { call(f.dest, f.message, f.weights, f.norm, y0, f.arena, eps); });
        rejects([&] { call(f.dest, f.message, f.weights, f.norm, y0, f.arena, kPairEps, eps); });
    }
    auto invalid   = f.dest;
    invalid.device = Device::cpu();
    rejects([&] { call(invalid, f.message, f.weights, f.norm, y0, f.arena); });
    invalid      = f.dest;
    invalid.data = static_cast<char*>(invalid.data) + 4;
    rejects([&] { call(invalid, f.message, f.weights, f.norm, y0, f.arena); });
    invalid = f.dest;
    invalid.nb[1] += 16;
    rejects([&] { call(invalid, f.message, f.weights, f.norm, y0, f.arena); });
    invalid       = f.dest;
    invalid.ne[2] = 3;
    rejects([&] { call(invalid, f.message, f.weights, f.norm, y0, f.arena); });
    invalid       = f.norm;
    invalid.dtype = DType::FP32;
    rejects([&] { call(f.dest, f.message, f.weights, invalid, y0, f.arena); });
    auto weights = f.weights;
    weights.up.ne[1]--;
    rejects([&] { call(f.dest, f.message, weights, f.norm, y0, f.arena); });
    rejects([&] { (void)capacity(7, 32, 2, 1, 2); });
    rejects([&] { (void)capacity(8, 24, 2, 1, 2); });
    rejects([&] { (void)capacity(8, 32, 0, 1, 2); });
    rejects([&] { (void)capacity(8, 32, 2, 2, 1); });
    require(f.arena.used() == 16, "workspace cursor changed on rejection");

    // Both read-only activations may use the same allocation.
    f.seed(2);
    call(f.dest, f.dest, f.weights, f.norm, y0, f.arena);
    f.msg = f.d;
    upload(f.dm, f.msg);
    CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
    f.verify();
    return 0;
}

int rounding_boundaries(Shape shape) {
    Fixture f(shape);
    std::fill(f.d.begin(), f.d.end(), 1.F);
    std::fill(f.msg.begin(), f.msg.end(), -1.F);
    std::fill(f.gamma.begin(), f.gamma.end(), 1.F);
    for (int s = 0; s < shape.s; ++s) {
        auto* w        = f.down.data() + std::size_t(s) * 2 * shape.d * shape.r;
        w[0]           = 1.F;
        w[1]           = 0.001953125F;
        w[2 * shape.d] = 1.F;
        auto* u        = f.up.data() + std::size_t(s) * 2 * shape.d * shape.r;
        for (int i = 0; i < 2 * shape.d; ++i) {
            u[i * shape.r]     = 0.55078125F;
            u[i * shape.r + 1] = bf16(-0.00145);
        }
    }
    // The represented dot products place both gates near 0.5. Rounding either
    // logit to BF16 or omitting the latent boundary breaks the cancellation.
    upload(f.dd, f.d);
    upload(f.dm, f.msg);
    upload(f.dw, f.down, true);
    upload(f.uw, f.up, true);
    upload(f.gw, f.gamma, true);
    const auto cuda = cuda_environment().context.execution_view();
    f.run(cuda);
    CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
    f.verify(4.e-5);
    return 0;
}

int graph(Shape shape) {
    Fixture f(shape);
    f.seed(71);
    auto cuda            = cuda_environment().context.execution_view();
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(cuda.stream(), cudaStreamCaptureModeThreadLocal));
    f.run(cuda);
    CUDA_CHECK(cudaStreamEndCapture(cuda.stream(), &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    std::vector<std::uint16_t> first;
    for (unsigned phase : {71, 82, 93, 71}) {
        f.seed(phase);
        CUDA_CHECK(cudaGraphLaunch(exec, cuda.stream()));
        CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
        f.verify();
        if (phase == 71) {
            auto words = read<std::uint16_t>(f.out, f.msg.size(), 16);
            if (first.empty())
                first = std::move(words);
            else
                require(first == words, "replaying identical inputs changed output words");
        }
    }
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    return 0;
}

void check_envelope(int d, int r, int s, int begin, int last) {
    const auto cuda = cuda_environment().context.execution_view();
    std::vector<std::size_t> actual;
    for (int m = begin; m <= last; ++m) {
        Fixture f({d, r, s, m, 1, s}, last);
        f.dd.fill();
        f.dm.fill();
        f.dw.fill();
        f.uw.fill();
        f.gw.fill();
        f.run(cuda);
        CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
        require(f.arena.used() == 16, "workspace cursor changed");
        actual.push_back(f.arena.peak_used() - 16);
    }
    for (int first = begin; first <= last; ++first) {
        std::size_t peak = 0;
        for (int m = first; m <= last; ++m) {
            peak = std::max(peak, actual[m - begin]);
            require(capacity(d, r, s, first, m) == peak, "incorrect interval capacity");
        }
    }
}

int envelope() {
    check_envelope(2304, 64, 1, 1, 272);
    check_envelope(2304, 64, 3, 1, 104);
    check_envelope(2304, 64, 3, 193, 313);
    check_envelope(256, 32, 3, 1, 37);
    return 0;
}

int capacity_extent_overflow() {
    constexpr int max = std::numeric_limits<int>::max();
    rejects<std::overflow_error>([] { (void)capacity(2304, 64, max, 1, max); });
    rejects<std::overflow_error>([] { (void)capacity(1 << 29, 1 << 29, 8, 1, 1); });
    return 0;
}

int rms_overflow(Shape shape) {
    Fixture f(shape);
    f.seed(91);
    // Mix overflowing and ordinary reductions in the same launch. Both
    // input RMS and the final blended RMS must match the FP64 oracle.
    for (std::size_t i = 0; i < f.d.size(); ++i)
        if (i / shape.d % 4 <= 1) f.d[i] *= 1.e20F;
    for (std::size_t i = 0; i < f.msg.size(); ++i)
        if (i / shape.d % 2 == 0) f.msg[i] *= 1.e20F;
    upload(f.dd, f.d);
    upload(f.dm, f.msg);
    auto cuda = cuda_environment().context.execution_view();
    f.run(cuda);
    CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
    f.verify();
    return 0;
}

int subnormal_eps() {
    Fixture f({256, 32, 2, 3, 2, 1});
    f.seed(1, true);
    auto cuda  = cuda_environment().context.execution_view();
    f.pair_eps = f.norm_eps = std::numeric_limits<float>::denorm_min();
    f.run(cuda);
    CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
    f.verify();
    return 0;
}

int rms_extremes(Shape shape, float input, float epsilon) {
    Fixture f(shape);
    f.pair_eps = f.norm_eps = epsilon;
    std::fill(f.d.begin(), f.d.end(), input);
    std::fill(f.msg.begin(), f.msg.end(), input);
    std::fill(f.gamma.begin(), f.gamma.end(), 1.F);
    const auto cuda = cuda_environment().context.execution_view();
    f.upload_operands();
    f.run(cuda);
    CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
    f.verify(0, true);

    // Turn the pair normalization into an observable forget gate at feature zero.
    for (int s = 0; s < shape.s; ++s) {
        const std::size_t base         = std::size_t(s) * 2 * shape.d * shape.r;
        f.down[base]                   = 0.5F;
        f.down[base + shape.d]         = 0.5F;
        f.up[base + shape.d * shape.r] = 1.F;
    }
    f.upload_operands();
    f.run(cuda);
    CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
    f.verify(0, true);
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    for (const auto g :
         {Shape{2304, 64, 1, 1, 1, 1}, {2304, 64, 3, 3, 1, 1},    {2304, 64, 3, 7, 3, 3},
          {2304, 64, 3, 8, 1, 3},      {2304, 64, 1, 15, 1, 1},   {2304, 64, 3, 17, 3, 1},
          {2304, 64, 3, 31, 1, 3},     {2304, 64, 2, 63, 2, 1},   {2304, 64, 4, 65, 1, 4},
          {2304, 64, 3, 129, 3, 3},    {2304, 64, 3, 1021, 3, 1}, {2304, 64, 3, 4093, 1, 3},
          {1024, 128, 3, 509, 3, 1},   {256, 32, 4, 1021, 1, 4},  {256, 32, 1, 2053, 1, 1},
          {256, 32, 1, 4093, 1, 1},    {3072, 128, 2, 3, 2, 2},   {232, 16, 2, 17, 2, 1},
          {24, 80, 2, 7, 1, 2},        {8200, 16, 1, 1, 1, 1}}) {
        const auto label = std::to_string(g.d) + "/" + std::to_string(g.r) + "/" +
                           std::to_string(g.s) + "/" + std::to_string(g.m);
        suite.add(label, [g] {
            Fixture f(g);
            f.seed(unsigned(g.m), g.d == 24, g.d == 232);
            auto cuda = cuda_environment().context.execution_view();
            f.run(cuda);
            CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
            f.verify();
            return 0;
        });
    }
    suite.add("rounding-decode", [] { return rounding_boundaries({2304, 64, 1, 1, 1, 1}); });
    suite.add("rounding-prefill", [] { return rounding_boundaries({2304, 64, 3, 17, 1, 3}); });
    suite.add("rounding-generic", [] { return rounding_boundaries({256, 32, 2, 3, 2, 1}); });
    suite.add("validation", validation);
    suite.add("capacity-envelope", envelope);
    suite.add("capacity-extent-overflow", capacity_extent_overflow);
    suite.add("rms-overflow-decode", [] { return rms_overflow({2304, 64, 3, 1, 1, 3}); });
    suite.add("rms-overflow-last-cta", [] { return rms_overflow({2304, 64, 3, 2, 3, 1}); });
    suite.add("rms-overflow-medium", [] { return rms_overflow({2304, 64, 3, 8, 1, 3}); });
    suite.add("rms-overflow-short-prefill", [] { return rms_overflow({2304, 64, 3, 77, 1, 3}); });
    suite.add("rms-overflow-prefill", [] { return rms_overflow({2304, 64, 3, 129, 1, 3}); });
    suite.add("rms-overflow-narrow-cluster", [] { return rms_overflow({2304, 64, 3, 319, 3, 1}); });
    suite.add("rms-overflow-vector", [] { return rms_overflow({256, 32, 2, 5, 1, 2}); });
    suite.add("rms-overflow-wide", [] { return rms_overflow({8200, 16, 1, 1, 1, 1}); });
    suite.add("subnormal-eps", subnormal_eps);
    suite.add("rms-denominator-overflow", [] {
        return rms_extremes({8, 16, 1, 1, 1, 1}, 0x1p60F, std::numeric_limits<float>::max());
    });
    for (const auto g : {Shape{8, 16, 1, 1, 1, 1},
                         {8200, 16, 1, 1, 1, 1},
                         {2304, 64, 3, 1, 1, 3},
                         {2304, 64, 3, 2, 3, 1},
                         {2304, 64, 3, 8, 1, 3},
                         {2304, 64, 3, 77, 1, 3},
                         {2304, 64, 3, 129, 1, 3},
                         {2304, 64, 3, 319, 3, 1}}) {
        const auto label =
            "rms-square-underflow/" + std::to_string(g.d) + "/" + std::to_string(g.m);
        suite.add(label, [g] {
            return rms_extremes(g, 0x1p-75F, std::numeric_limits<float>::denorm_min());
        });
    }
    suite.add("graph-decode-separate-finish", [] { return graph({2304, 64, 3, 1, 1, 3}); });
    suite.add("graph-decode-last-cta", [] { return graph({2304, 64, 3, 2, 1, 3}); });
    suite.add("graph-medium", [] { return graph({2304, 64, 3, 17, 3, 1}); });
    suite.add("graph-short-prefill", [] { return graph({2304, 64, 3, 77, 3, 1}); });
    suite.add("graph-short-prefill-token-chunks", [] { return graph({2304, 64, 8, 97, 1, 8}); });
    suite.add("graph-short-prefill-stream-chunks", [] { return graph({2304, 64, 13, 49, 13, 1}); });
    suite.add("graph-prefill-wide-cluster", [] { return graph({2304, 64, 3, 829, 1, 3}); });
    suite.add("graph-prefill-narrow-cluster", [] { return graph({2304, 64, 3, 319, 3, 1}); });
    suite.add("graph-generic", [] { return graph({256, 32, 2, 7, 1, 2}); });
    return suite.main(argc, argv);
}
