#include "ninfer/ops/relative_attention.h"
#include "ops/op_tester.h"
#include "ops/ragged_fixture.h"

#include <cuda_fp16.h>

#include <array>
#include <bit>
#include <optional>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::ops;

namespace {

constexpr int D = 128, R = 16, QH = 24, KH = 8, C = D * KH, History = 3, Window = 1024;
constexpr std::uint8_t Poison = 0xa5;
using Word                    = std::uint16_t;
using Words                   = std::vector<Word>;
using Bytes                   = std::vector<std::uint8_t>;

struct Case {
    bool paged, fp8, snapshot, ragged, dynamic, tau;
    int width, sink;
    std::vector<int> lengths, starts;
    bool edge = false, alias = false, unmasked = false, shared_source = false,
         clip_boundary = false;
    bool chunk_decode = false, shared_arena = false, overflow_norm = false;
    bool split_history_selectors = false, overflow_conv = false, underflow_norm = false;
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

double decode(Word w) { return bf16_to_f32(w); }

double round_bf16(double x) {
    if (x == 0) return x;
    const double magnitude = std::abs(x);
    const int exponent     = std::max(-126, std::ilogb(magnitude));
    const double spacing   = std::ldexp(1.0, exponent - 7);
    return std::copysign(std::nearbyint(magnitude / spacing) * spacing, x);
}

double e4m3(unsigned code) {
    const double significand = code & 7;
    const int exponent       = int(code >> 3) & 15;
    const double x =
        exponent ? std::ldexp(1.0 + significand / 8, exponent - 7) : std::ldexp(significand, -9);
    return code & 128 ? -x : x;
}

std::uint8_t encode_fp8(double x) {
    static const auto grid = [] {
        std::array<double, 127> values{};
        for (unsigned i = 0; i < values.size(); ++i) values[i] = e4m3(i);
        return values;
    }();
    const bool negative = std::signbit(x);
    x                   = std::abs(x);
    const auto it       = std::lower_bound(grid.begin(), grid.end(), x);
    unsigned code       = it - grid.begin();
    if (it == grid.end()) {
        code = 126;
    } else if (it != grid.begin()) {
        const double lo = x - *(it - 1), hi = *it - x;
        if (lo < hi || (lo == hi && (code & 1))) --code;
    }
    return static_cast<std::uint8_t>(code | (negative ? 128 : 0));
}

struct Operands {
    explicit Operands(const Case& c, bool commit_only = false)
        : c(c), ctx(cuda_environment().context), device(ctx.device()), b(c.lengths.size()),
          w(c.ragged ? std::accumulate(c.lengths.begin(), c.lengths.end(), 0) : c.width),
          n(c.ragged ? w : w * b),
          storage(c.fp8 ? KvCacheStorage::Fp8E4m3fnGroup128Ue8m0 : KvCacheStorage::BFloat16),
          positions(n, -1), starts(b), rows_host(b, -1), slots(b, -1), state_src(b, -1),
          state_dst(b, -1), cyclic_offsets(b, -1) {
        params.use_log_scaling    = c.tau;
        params.qk_clip_threshold  = c.edge ? 2048.0f : 3.25f;
        params.rel_clip_threshold = 1.75f;
        if (c.clip_boundary) {
            params.qk_clip_threshold  = 3.255f;
            params.rel_clip_threshold = 1.6275f;
        }
        params.log_scaling_alpha = 0.37f;
        const int batch          = c.ragged ? 1 : b;
        int flat                 = 0;
        for (int s = 0; s < b; ++s) {
            starts[s] = c.ragged ? flat : s * w;
            flat += c.lengths[s];
            for (int j = 0; j < c.lengths[s]; ++j) positions[starts[s] + j] = c.starts[s] + j;
            if (c.lengths[s]) {
                rows_host[s] = b - 1 - s;
                slots[s]     = (b - 1 - s) * (w + 1) + 1;
            }
        }
        in.positions = upload(positions, DType::I32, {w, batch});
        if (c.ragged)
            ragged.emplace(c.lengths);
        else if (!c.unmasked)
            valid = upload(c.lengths, DType::I32, {b});

        if (commit_only) {
            make_cache();
            return;
        }

        if (c.underflow_norm) {
            params.qk_clip_threshold = params.rel_clip_threshold = 0x1p-84f;
            params.clip_eps                                      = 0x1p-100f;
            params.v_norm_eps                                    = 0x1p-149f;
        }

        q  = data(n * D * QH, 1, 6.0f);
        r  = data(n * R * QH, 2, 4.0f);
        qs = data(D * QH, 3, 1.0f);
        rs = data(R * QH, 4, 1.5f);
        std::fill_n(q.begin(), D, 0);
        std::fill_n(r.begin(), R, 0);
        if (c.clip_boundary) {
            q[0] = r[0] = f32_to_bf16(10.0f);
            qs[0] = rs[0] = f32_to_bf16(1.0f);
        }
        if (c.overflow_norm) {
            auto large_query = [](Words& x, Words& scale, int dim) {
                std::fill_n(x.begin(), 2 * dim, 0);
                std::fill_n(scale.begin(), dim, f32_to_bf16(1.0f));
                std::fill_n(scale.begin() + dim, dim, f32_to_bf16(64.0f));
                x[0]       = f32_to_bf16(0x1p64f);
                x[dim]     = f32_to_bf16(1.0e20f);
                x[dim + 1] = f32_to_bf16(-8.0e19f);
                x[dim + 2] = f32_to_bf16(4.0e19f);
            };
            large_query(q, qs, D);
            large_query(r, rs, R);
        }
        if (c.underflow_norm) {
            q[0] = r[0] = f32_to_bf16(0x1p-80f);
            qs[0] = rs[0] = f32_to_bf16(1.0f);
        }
        q_device   = upload(q, DType::BF16, {D, QH, w, batch});
        in.q       = q_device;
        in.r       = upload(r, DType::BF16, {R, QH, w, batch});
        in.q_scale = upload(qs, DType::BF16, {D, QH});
        in.r_scale = upload(rs, DType::BF16, {R, QH});
        for (int side = 0; side < 2; ++side) {
            kv[side]      = data(n * C, 5 + side, 1.75f);
            weights[side] = data(C * 4, 7 + side, 0.25f);
            dyn[side]     = data(n * KH * 4, 9 + side, 0.125f);
            if (c.edge) {
                std::fill_n(kv[side].begin(), D, 0);
                std::fill_n(weights[side].begin(), C * 4, 0);
                std::fill_n(dyn[side].begin(), n * KH * 4, 0);
                for (int token = 0; token < n; ++token) {
                    auto* x = kv[side].data() + token * C;
                    std::fill_n(x, 4 * D, 0);
                    const float e4m3_max = c.dynamic ? 1344.0f : 1000.0f;
                    x[D]                 = f32_to_bf16(e4m3_max);
                    x[D + 1]             = f32_to_bf16(-e4m3_max);
                    x[D + 2]             = f32_to_bf16(c.dynamic ? 3.1875f : 2.375f);
                    x[D + 3]             = f32_to_bf16(c.dynamic ? -3.1875f : -2.375f);
                    x[2 * D]             = f32_to_bf16(e4m3_max + 8.0f);
                    x[2 * D + 1]         = f32_to_bf16(-e4m3_max);
                    x[3 * D]             = f32_to_bf16(std::ldexp(1.0f, -130));
                    x[3 * D + 1]         = f32_to_bf16(-std::ldexp(1.0f, -130));
                }
            }
            if (c.overflow_norm) {
                std::fill(weights[side].begin(), weights[side].end(), 0);
                std::fill(dyn[side].begin(), dyn[side].end(), 0);
                std::fill_n(kv[side].begin(), 2 * D, 0);
                kv[side][0]     = f32_to_bf16(0x1p66f);
                kv[side][D]     = f32_to_bf16(1.0e20f);
                kv[side][D + 1] = f32_to_bf16(-8.0e19f);
                kv[side][D + 2] = f32_to_bf16(4.0e19f);
            }
            if (c.overflow_conv || c.underflow_norm) {
                // Isolate the tested head from history and other convolution coefficients.
                std::fill_n(kv[side].begin(), D, 0);
                std::fill(weights[side].begin(), weights[side].end(), 0);
                std::fill(dyn[side].begin(), dyn[side].end(), 0);
                if (c.overflow_conv) {
                    kv[side][0]                = f32_to_bf16(0x1p127f);
                    weights[side][History * C] = f32_to_bf16(1.0f);
                } else if (side == 0) {
                    kv[side][0] = f32_to_bf16(3.0f * 0x1p-80f);
                } else {
                    // Each post-convolution square is a tie at half the least FP32 subnormal.
                    std::fill_n(kv[side].begin(), D, f32_to_bf16(3.0f * 0x1p-75f));
                }
            }
            (side ? in.v : in.k) = upload(kv[side], DType::BF16, {D, KH, w, batch});
            (side ? in.v_conv_weight : in.k_conv_weight) =
                upload(weights[side], DType::BF16, {C, 4});
            if (c.dynamic)
                (side ? in.v_dyn : in.k_dyn) = upload(dyn[side], DType::BF16, {4, KH, w, batch});
        }
        if (c.sink) {
            sink_input     = data(n * c.sink, 11, 0.5f);
            sink_weight    = data(QH * c.sink, 12, 0.25f);
            in.sink_x      = upload(sink_input, DType::BF16, {c.sink, w, batch});
            in.sink_weight = upload(sink_weight, DType::BF16, {c.sink, QH});
            out.sink_out   = allocate(DType::BF16, {QH, w, batch});
        }
        out.q_out = allocate(DType::BF16, {D, QH, w, batch});
        out.r_out = allocate(DType::BF16, {R, QH, w, batch});
        make_state();
        make_cache();
    }

    void launch(RelativeAttentionPrepareInputs inputs, RelativeAttentionPrepareOutputs outputs,
                const RelativeAttentionPrepareParams& p) {
        const auto stream = ctx.execution_view();
        if (c.ragged) {
            if (c.paged)
                relative_attention_prepare(inputs, ragged->view, rows, ki, ko, vi, vo, full, p,
                                           outputs, stream);
            else
                relative_attention_prepare(inputs, ragged->view, ki, ko, vi, vo, chunk, p, outputs,
                                           stream);
        } else if (c.snapshot) {
            if (c.paged)
                relative_attention_prepare_snapshot(inputs, valid, rows, snapshot_slots, ki, ks, vi,
                                                    vs, full, p, outputs, stream);
            else
                relative_attention_prepare_snapshot(inputs, valid, snapshot_slots, ki, ks, vi, vs,
                                                    chunk, p, outputs, stream);
        } else if (c.paged) {
            relative_attention_prepare(inputs, valid, rows, ki, ko, vi, vo, full, p, outputs,
                                       stream);
        } else if (uses_chunk()) {
            relative_attention_prepare(inputs, valid, ki, ko, vi, vo, chunk, p, outputs, stream);
        } else {
            relative_attention_prepare(inputs, valid, ki, ko, vi, vo, swa, p, outputs, stream);
        }
    }

    void prepare() {
        CUDA_CHECK(cudaMemcpyAsync(ki.records.storage.data, initial_state.data(),
                                   initial_state.size(), cudaMemcpyHostToDevice, ctx.stream()));
        CUDA_CHECK(cudaMemsetAsync(cache.data, Poison, cache.bytes, ctx.stream()));
        if (c.snapshot)
            CUDA_CHECK(cudaMemsetAsync(snapshots.data(), Poison, snapshots.size(), ctx.stream()));
        for (const auto* t : {&out.q_out, &out.r_out, &out.sink_out})
            if (t->data) CUDA_CHECK(cudaMemsetAsync(t->data, Poison, t->bytes(), ctx.stream()));
        ctx.synchronize();
    }

    int check(const std::vector<int>& lengths) {
        std::vector<double> q_ref(q.size()), r_ref(r.size()),
            sink_ref(sink_input.empty() ? 0 : n * QH);
        Bytes expected_cache(cache.bytes, Poison);
        std::vector<std::uint8_t> live_cache(cache.bytes, 0);
        auto expected_state = initial_state;
        Bytes expected_snapshots(snapshots.size(), Poison);
        for (int seq = 0; seq < b; ++seq) {
            for (int j = 0; j < lengths[seq]; ++j) {
                const int token = starts[seq] + j, p = positions[token];
                const double tau =
                    c.tau ? 1 + double(params.log_scaling_alpha) *
                                    std::log(
                                        std::max(1.0, (double(p) + 1) / params.log_scaling_floor))
                          : 1.0;
                auto clip = [&](const Words& x, const Words& scale, int dim, double threshold,
                                std::vector<double>& output) {
                    std::array<double, D> values{};
                    for (int h = 0; h < QH; ++h) {
                        const int base = (token * QH + h) * dim;
                        double sum     = 0;
                        for (int d = 0; d < dim; ++d) {
                            const double a = decode(x[base + d]) * decode(scale[h * dim + d]);
                            values[d]      = a;
                            sum += a * a;
                        }
                        const double gain =
                            std::min(1.0, threshold / (std::sqrt(sum) + params.clip_eps));
                        for (int d = 0; d < dim; ++d)
                            output[base + d] = round_bf16(round_bf16(values[d] * gain) * tau);
                    }
                };
                clip(q, qs, D, params.qk_clip_threshold, q_ref);
                clip(r, rs, R, params.rel_clip_threshold, r_ref);
                for (int h = 0; h < QH && c.sink; ++h) {
                    double sum = 0;
                    for (int d = 0; d < c.sink; ++d)
                        sum += decode(sink_input[token * c.sink + d]) *
                               decode(sink_weight[h * c.sink + d]);
                    sink_ref[token * QH + h] = round_bf16(sum);
                }
                for (int side = 0; side < 2; ++side) {
                    auto sample = [&](int col, int channel) {
                        if (col >= 0) return kv[side][(starts[seq] + col) * C + channel];
                        return read_word(initial_state,
                                         state_src[seq] + history_record_delta[side] +
                                             history_offset[side] +
                                             ((col + History) * C + channel) * sizeof(Word));
                    };
                    if (c.snapshot || j + 1 == lengths[seq]) {
                        for (int h = 0; h < History; ++h)
                            for (int channel = 0; channel < C; ++channel) {
                                const auto value = sample(j - History + 1 + h, channel);
                                const int index  = h * C + channel;
                                if (c.snapshot) {
                                    const auto offset = (slots[seq] + j) * snapshot_stride +
                                                        (side * History * C + index) * sizeof(Word);
                                    write_word(expected_snapshots, offset, value);
                                } else {
                                    write_word(expected_state,
                                               state_dst[seq] + history_record_delta[side] +
                                                   history_offset[side] + index * sizeof(Word),
                                               value);
                                }
                            }
                    }
                    for (int h = 0; h < KH; ++h) {
                        std::array<double, D> values;
                        double sum_sq = 0;
                        for (int d = 0; d < D; ++d) {
                            const int channel = h * D + d;
                            double sum        = 0;
                            for (int tap = 0; tap <= History; ++tap) {
                                const int lag      = c.dynamic ? tap : History - tap;
                                double coefficient = decode(weights[side][tap * C + channel]);
                                if (c.dynamic)
                                    coefficient += decode(dyn[side][(token * KH + h) * 4 + tap]);
                                sum += coefficient * decode(sample(j - lag, channel));
                            }
                            sum += decode(sample(j, channel));
                            const double v = round_bf16(sum / std::sqrt(c.dynamic ? 9.0 : 5.0));
                            values[d]      = v;
                            sum_sq += v * v;
                        }
                        const double gain =
                            side ? 1.0 / std::sqrt(sum_sq / D + params.v_norm_eps)
                                 : std::min(1.0, double(params.qk_clip_threshold) /
                                                     (std::sqrt(sum_sq) + params.clip_eps));
                        for (auto& v : values) v *= gain;
                        const auto record = c.paged
                                                ? page_offsets[rows_host[seq] * logical + p / 64]
                                            : uses_chunk() ? 0
                                                           : cyclic_offsets[seq];
                        const auto vector = uses_chunk()
                                                ? token * KH + h
                                                : h * cache_rows + p % (c.paged ? 64 : Window);
                        const auto base =
                            record + cache_offset[side] + std::size_t(vector) * D * (c.fp8 ? 1 : 2);
                        if (c.fp8) {
                            double maximum = 0;
                            for (auto v : values) maximum = std::max(maximum, std::abs(v));
                            const int e =
                                maximum ? std::max(-127, int(std::ceil(std::log2(maximum / 448))))
                                        : 0;
                            const auto scale      = record + cache_offset[side + 2] + vector;
                            expected_cache[scale] = static_cast<std::uint8_t>(e + 127);
                            live_cache[scale]     = 1;
                            for (int d = 0; d < D; ++d)
                                expected_cache[base + d] =
                                    maximum ? encode_fp8(std::ldexp(values[d], -e)) : 0;
                            std::fill_n(live_cache.begin() + base, D, 1);
                        } else {
                            for (int d = 0; d < D; ++d) {
                                const auto raw =
                                    side ? std::bit_cast<Word>(__float2half_rn(values[d]))
                                         : f32_to_bf16(static_cast<float>(values[d]));
                                write_word(expected_cache, base + 2 * d, raw);
                            }
                            // 2 and 3 mark the beginning of typed BF16 and FP16 words.
                            for (int d = 0; d < D; ++d)
                                live_cache[base + 2 * d] =
                                    token == 0 && h == 0 && (c.underflow_norm || c.overflow_conv)
                                        ? 4
                                        : 2 + side;
                        }
                    }
                }
            }
        }
        auto compare = [&](const char* name, const Tensor& output, const std::vector<double>& ref,
                           int features) {
            const auto actual = from_device<Word>(output.data, ref.size());
            for (int s = 0; s < b; ++s)
                if (!c.ragged)
                    for (int j = lengths[s]; j < w; ++j)
                        require(std::all_of(actual.begin() + (starts[s] + j) * features,
                                            actual.begin() + (starts[s] + j + 1) * features,
                                            [](auto x) { return x == 0; }),
                                "inactive output is nonzero");
            std::vector<double> decoded(actual.size());
            std::transform(actual.begin(), actual.end(), decoded.begin(), bf16_to_f32);
            return verify_reduction(name, decoded, ref, {0.0045, 0.00005, 0.009});
        };
        int failures =
            compare("Q", out.q_out, q_ref, D * QH) + compare("R", out.r_out, r_ref, R * QH);
        if (c.sink) failures += compare("sink", out.sink_out, sink_ref, QH);
        if (c.underflow_norm && lengths.front()) {
            require(decode(from_device<Word>(out.q_out.data, 1).front()) == q_ref[0] &&
                        decode(from_device<Word>(out.r_out.data, 1).front()) == r_ref[0],
                    "exact Q/R clip when the FP32 square underflows");
        }
        if (c.clip_boundary && lengths.front()) {
            const auto verify_boundary = [](const Tensor& t, double rounded, double clipped) {
                const auto word = from_device<Word>(t.data, 1).front();
                require(rounded != clipped && decode(word) == rounded, "clipped BF16 before tau");
            };
            const double tau = 1 + params.log_scaling_alpha * std::log(4.0);
            verify_boundary(out.q_out, q_ref[0],
                            round_bf16(std::copysign(
                                (params.qk_clip_threshold - params.clip_eps) * tau, decode(q[0]))));
            verify_boundary(out.r_out, r_ref[0],
                            round_bf16((params.rel_clip_threshold - params.clip_eps) * tau));
        }
        const auto got_state =
            from_device<std::uint8_t>(ki.records.storage.data, initial_state.size());
        require(got_state == expected_state, "history write domain or words");
        if (c.snapshot)
            require(from_device<std::uint8_t>(snapshots.data(), snapshots.size()) ==
                        expected_snapshots,
                    "snapshot words, guard bytes, or write domain");
        const auto got = from_device<std::uint8_t>(cache.data, cache.bytes);
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (live_cache[i] < 2) {
                if (got[i] != expected_cache[i])
                    throw std::runtime_error("KV byte " + std::to_string(i) + " expected " +
                                             std::to_string(expected_cache[i]) + " got " +
                                             std::to_string(got[i]));
            } else {
                const auto expected = read_word(expected_cache, i);
                const auto actual   = read_word(got, i);
                if (live_cache[i] == 4) {
                    require(actual == expected, "exact K/V word for exceptional norm or conv");
                    ++i;
                    continue;
                }
                const auto decode_word = live_cache[i] == 2 ? decode : +[](Word x) -> double {
                    return __half2float(std::bit_cast<__half>(x));
                };
                const double y = decode_word(expected), x = decode_word(actual);
                require(std::isfinite(x) && std::abs(y - x) <= 0.009 * std::abs(y) + 1e-8,
                        "BF16/FP16 KV word");
                ++i;
            }
        }
        return failures;
    }

    void replay_input() {
        for (auto& word : q) word ^= 0x8000;
        CUDA_CHECK(cudaMemcpyAsync(q_device.data, q.data(), q.size() * sizeof(Word),
                                   cudaMemcpyHostToDevice, ctx.stream()));
    }

    void set_valid(const std::vector<int>& lengths) {
        CUDA_CHECK(cudaMemcpyAsync(valid.data, lengths.data(), lengths.size() * sizeof(int),
                                   cudaMemcpyHostToDevice, ctx.stream()));
    }

    std::array<Bytes, 4> output_image() const {
        auto read = [](const Tensor& t) {
            return t.data ? from_device<std::uint8_t>(t.data, t.bytes()) : Bytes{};
        };
        return {read(out.q_out), read(out.r_out), read(out.sink_out),
                from_device<std::uint8_t>(cache.data, cache.bytes)};
    }

    void seed_encoded_chunk() {
        // A pure copy oracle with distinct bytes along the token, head, and plane axes.
        Bytes encoded(cache.bytes);
        for (std::size_t i = 0; i < encoded.size(); ++i)
            encoded[i] = static_cast<std::uint8_t>((i ^ (i >> 7) ^ (i >> 19)) % 127);
        CUDA_CHECK(cudaMemcpyAsync(cache.data, encoded.data(), encoded.size(),
                                   cudaMemcpyHostToDevice, ctx.stream()));
        ctx.synchronize();
    }

    void init_commit() {
        if (!cyclic.data()) make_cyclic();
        cyclic.fill(Poison);
    }

    void submit_commit(const Tensor& counts) {
        if (c.ragged)
            relative_attention_commit_cyclic_kv(chunk, ragged->view, in.positions, swa,
                                                ctx.execution_view());
        else
            relative_attention_commit_cyclic_kv(chunk, in.positions, counts, swa,
                                                ctx.execution_view());
    }

    void reject_commit_operands() {
        auto reject = [&](const RelativeAttentionChunkKV& bad, const Tensor& bad_positions,
                          const Tensor& bad_counts) {
            try {
                relative_attention_commit_cyclic_kv(bad, bad_positions, bad_counts, swa,
                                                    ctx.execution_view());
            } catch (const std::invalid_argument&) { return; }
            throw std::runtime_error("accepted invalid cyclic commit operands");
        };
        auto bad    = chunk;
        bad.storage = c.fp8 ? KvCacheStorage::BFloat16 : KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;
        reject(bad, in.positions, valid);
        bad        = chunk;
        bad.k.data = static_cast<char*>(bad.k.data) + 1;
        reject(bad, in.positions, valid);
        bad        = chunk;
        bad.k.data = cyclic.data();
        reject(bad, in.positions, valid);
        auto bad_positions = in.positions;
        bad_positions.data = cyclic.data();
        reject(chunk, bad_positions, valid);
        auto bad_counts = valid;
        bad_counts.data = cyclic.data();
        reject(chunk, in.positions, bad_counts);
    }

    void reject_cyclic_prepare() {
        try {
            relative_attention_prepare(in, valid, ki, ko, vi, vo, swa, params, out,
                                       ctx.execution_view());
        } catch (const std::invalid_argument&) { return; }
        throw std::runtime_error("accepted multi-column direct cyclic prepare");
    }

    void check_commit(const std::vector<int>& counts) {
        Bytes expected(cyclic.size(), Poison);
        const auto encoded = from_device<std::uint8_t>(cache.data, cache.bytes);
        const auto planes  = ring_fields();
        for (int seq = 0; seq < b; ++seq)
            for (int j = std::max(0, counts[seq] - Window); j < counts[seq]; ++j) {
                const int token = starts[seq] + j;
                const int slot  = positions[token] % Window;
                for (int plane = 0; plane < (c.fp8 ? 4 : 2); ++plane) {
                    const int size = plane >= 2 ? 1 : D * (c.fp8 ? 1 : 2);
                    for (int h = 0; h < KH; ++h) {
                        const auto src = cache_offset[plane] + (token * KH + h) * size;
                        const auto dst = cyclic_offsets[seq] + planes[plane] +
                                         (h * swa.padded_capacity + slot) * size;
                        std::copy_n(encoded.begin() + src, size, expected.begin() + dst);
                    }
                }
            }
        require(from_device<std::uint8_t>(cyclic.data(), cyclic.size()) == expected,
                "cyclic commit bytes and write domain");
    }

    const Tensor& counts_tensor() const { return valid; }

    const Case& c;
    const DeviceContext& ctx;
    Device device;
    int b, w, n;
    KvCacheStorage storage;
    RelativeAttentionPrepareInputs in;
    RelativeAttentionPrepareOutputs out;
    RelativeAttentionPrepareParams params;

private:
    Words data(int n, unsigned seed, float scale) {
        Words words(n);
        for (int i = 0; i < n; ++i) {
            const auto hash = unsigned(i) * 2654435761u ^ seed * 97531u;
            words[i]        = f32_to_bf16(scale * (int((hash >> 17) % 257) - 128) / 128.0f);
        }
        return words;
    }

    Tensor allocate(DType dtype, std::initializer_list<int> dims) {
        auto& buffer = buffers.emplace_back(device, Tensor::metadata(dtype, dims).bytes());
        return Tensor(buffer.span(), dtype, dims);
    }

    template <class T>
    Tensor upload(const std::vector<T>& v, DType dtype, std::initializer_list<int> dims) {
        const auto t = allocate(dtype, dims);
        require(t.bytes() == v.size() * sizeof(T), "fixture upload size");
        buffers.back().copy_from_host(v.data(), t.bytes());
        return t;
    }

    static void write_word(Bytes& bytes, std::size_t offset, Word word) {
        std::memcpy(bytes.data() + offset, &word, sizeof(word));
    }

    static Word read_word(const Bytes& bytes, std::size_t offset) {
        Word word;
        std::memcpy(&word, bytes.data() + offset, sizeof(word));
        return word;
    }

    void make_state() {
        StateImageBuilder builder;
        const auto k = builder.add(DType::BF16, {C, History}, 1, "history K");
        const auto v =
            c.split_history_selectors ? k : builder.add(DType::BF16, {C, History}, 1, "history V");
        const auto layout = builder.finish();
        const auto stride = layout.image_bytes + 256;
        state             = DeviceBuffer(device, (2 * b + 1) * stride);
        initial_state.assign(state.size(), Poison);
        if (c.split_history_selectors) history_record_delta[1] = b * stride;
        for (int s = 0; s < b; ++s) {
            if (!c.lengths[s]) continue;
            state_src[s] = (c.shared_source ? 1 : b - s) * stride;
            state_dst[s] = c.split_history_selectors || (c.alias && s % 2 == 0)
                               ? state_src[s]
                               : (b + 1 + s) * stride;
        }
        history_offset = {layout.fields[k.index].tensor.region.offset,
                          layout.fields[v.index].tensor.region.offset};
        for (int side = 0; side < 2; ++side)
            for (int s = 0; s < b; ++s) {
                if (state_src[s] < 0) continue;
                const auto words = data(C * History, 16 + side + (c.shared_source ? 0 : s), 0.75f);
                std::memcpy(initial_state.data() + state_src[s] + history_record_delta[side] +
                                history_offset[side],
                            words.data(), words.size() * sizeof(Word));
            }
        StateImageStorage image(state.span(), layout);
        const auto src = upload(state_src, DType::I64, {b});
        ki             = image.field(k, src);
        vi             = image.field(v, src);
        if (c.split_history_selectors) {
            auto v_offsets = state_src;
            for (auto& off : v_offsets)
                if (off >= 0) off += history_record_delta[1];
            vi = image.field(v, upload(v_offsets, DType::I64, {b}));
            ko = ki;
            vo = vi;
            require(history_offset[0] == 0 && history_offset[1] == 0 &&
                        ki.records.byte_offsets.data != vi.records.byte_offsets.data,
                    "shared histories must select different records at local offset zero");
        } else if (c.snapshot) {
            snapshot_slots         = upload(slots, DType::I32, {b});
            const int count        = b * (w + 1) + 1;
            const auto plane_bytes = C * History * sizeof(Word);
            snapshot_stride        = 2 * plane_bytes + 256;
            snapshots              = DeviceBuffer(device, count * snapshot_stride);
            ks                     = Tensor(snapshots.span(), DType::BF16, {C, History, count});
            vs = Tensor(snapshots.span().subspan(plane_bytes), DType::BF16, {C, History, count});
            ks.nb[2] = vs.nb[2] = snapshot_stride;
        } else {
            const auto dst = upload(state_dst, DType::I64, {b});
            ko             = image.field(k, dst);
            vo             = image.field(v, dst);
        }
    }

    void make_cache() {
        if (c.paged) {
            KVPageGeometry geo{64,
                               {{c.fp8 ? DType::FP8_E4M3FN : DType::BF16, D, KH},
                                {c.fp8 ? DType::FP8_E4M3FN : DType::FP16, D, KH}}};
            if (c.fp8) geo.planes.insert(geo.planes.end(), 2, {DType::U8, 1, KH});
            int count = 0;
            logical   = 1;
            for (int s = 0; s < b; ++s) {
                if (!c.lengths[s]) continue;
                const auto last = (c.starts[s] + c.lengths[s] - 1) / 64;
                logical         = std::max(logical, last + 1);
                count += last - c.starts[s] / 64 + 1;
            }
            // Leave one physical record unassigned and permute the live mappings.
            const auto layout      = plan_kv_page_layout(geo);
            const auto stride      = layout.page_bytes + layout.alignment;
            const auto state_bytes = c.shared_arena ? state.size() : 0;
            cache_buffer =
                DeviceBuffer(device, state_bytes + layout.alignment + (count + 1) * stride);
            const auto backing = cache_buffer.span();
            cache              = backing.subspan(state_bytes);
            if (c.shared_arena) {
                require(history_offset[0] == 0 && layout.planes[0].tensor.region.offset == 0,
                        "shared arena fixture must exercise overlapping record-local fields");
                for (auto* field : {&ki, &vi, &ko, &vo})
                    if (field->records.storage.data) field->records.storage = backing;
            }
            cache_rows = 64;
            page_offsets.assign(logical * b, -1);
            for (int s = 0; s < b; ++s)
                if (c.lengths[s])
                    for (int page = c.starts[s] / 64; page <= (c.starts[s] + c.lengths[s] - 1) / 64;
                         ++page)
                        page_offsets[rows_host[s] * logical + page] =
                            layout.alignment + count-- * stride;
            auto device_offsets = page_offsets;
            for (auto& off : device_offsets)
                if (off >= 0) off += state_bytes;
            const auto table = upload(device_offsets, DType::I64, {logical, b});
            rows             = upload(rows_host, DType::I32, {b});
            auto plane       = [&](unsigned i) {
                cache_offset[i] = layout.planes[i].tensor.region.offset;
                return PagedPlaneView{{backing, table, 64}, layout.planes[i]};
            };
            full = {plane(0),
                    plane(1),
                    c.fp8 ? plane(2) : PagedPlaneView{},
                    c.fp8 ? plane(3) : PagedPlaneView{},
                    D,
                    KH,
                    storage};
        } else if (uses_chunk()) {
            const int batch = c.ragged ? 1 : b;
            LayoutBuilder builder;
            const auto k = builder.add_tensor(c.fp8 ? DType::FP8_E4M3FN : DType::BF16,
                                              {D, KH, w, batch}, 16, "chunk K");
            const auto v = builder.add_tensor(c.fp8 ? DType::FP8_E4M3FN : DType::FP16,
                                              {D, KH, w, batch}, 16, "chunk V");
            TensorRegion k_scales, v_scales;
            if (c.fp8) {
                k_scales = builder.add_tensor(DType::U8, {1, KH, w, batch}, 1, "chunk K scales");
                v_scales = builder.add_tensor(DType::U8, {1, KH, w, batch}, 1, "chunk V scales");
            }
            cache_buffer = DeviceBuffer(device, builder.finish());
            cache        = cache_buffer.span();
            chunk = {storage, k.bind(cache), v.bind(cache), c.fp8 ? k_scales.bind(cache) : Tensor{},
                     c.fp8 ? v_scales.bind(cache) : Tensor{}};
            cache_offset = {k.region.offset, v.region.offset, k_scales.region.offset,
                            v_scales.region.offset};
        } else {
            make_cyclic();
            cache        = cyclic.span();
            cache_rows   = swa.padded_capacity;
            cache_offset = ring_fields();
        }
    }

    void make_cyclic() {
        StateImageBuilder builder;
        const auto plan    = plan_cyclic_kv_cache(builder, 2, Window, KH, D, storage);
        const auto layout  = builder.finish();
        const auto stride  = layout.image_bytes + 256;
        const auto records = std::count_if(c.lengths.begin(), c.lengths.end(),
                                           [](int length) { return length > 0; });
        int index          = records;
        for (int s = 0; s < b; ++s)
            if (c.lengths[s]) cyclic_offsets[s] = index-- * stride;
        cyclic = DeviceBuffer(device, (records + 1) * stride);
        StateImageStorage image(cyclic.span(), layout);
        swa = CyclicKVCacheView(image, plan).layer_view(1, upload(cyclic_offsets, DType::I64, {b}));
    }

    std::array<std::size_t, 4> ring_fields() const {
        return {swa.k.field.tensor.region.offset, swa.v.field.tensor.region.offset,
                swa.k_scales.field.tensor.region.offset, swa.v_scales.field.tensor.region.offset};
    }

    bool uses_chunk() const {
        return !c.paged && (c.ragged || c.snapshot || w != 1 || c.chunk_decode);
    }

    std::vector<int> positions, starts, rows_host, slots;
    std::vector<std::int64_t> state_src, state_dst, cyclic_offsets, page_offsets;
    Words q, r, qs, rs, sink_input, sink_weight;
    std::array<Words, 2> kv, weights, dyn;
    Bytes initial_state;
    std::array<std::size_t, 2> history_offset{}, history_record_delta{};
    std::array<std::size_t, 4> cache_offset{};
    int cache_rows = 0, logical = 0;
    std::vector<DeviceBuffer> buffers;
    DeviceBuffer state, cyclic, cache_buffer, snapshots;
    std::size_t snapshot_stride = 0;
    MemorySpan cache;
    std::optional<RaggedFixture> ragged;
    RelativeAttentionChunkKV chunk;
    PagedKVBatchLayerView full{};
    CyclicKVCacheLayerView swa{};
    Tensor q_device, valid, rows, snapshot_slots, ks, vs;
    StateFieldView ki, vi, ko, vo;
};

class Graph {
public:
    template <class Launch>
    Graph(const DeviceContext& ctx, Launch&& launch) {
        CUDA_CHECK(cudaStreamBeginCapture(ctx.stream(), cudaStreamCaptureModeThreadLocal));
        try {
            launch();
        } catch (...) {
            cudaGraph_t discarded = nullptr;
            (void)cudaStreamEndCapture(ctx.stream(), &discarded);
            if (discarded) (void)cudaGraphDestroy(discarded);
            throw;
        }

        try {
            CUDA_CHECK(cudaStreamEndCapture(ctx.stream(), &graph));
            std::size_t nodes = 0;
            CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nodes));
            require(nodes == 1, "graph must contain exactly one node");
            cudaGraphNode_t node;
            CUDA_CHECK(cudaGraphGetNodes(graph, &node, &nodes));
            cudaGraphNodeType type;
            CUDA_CHECK(cudaGraphNodeGetType(node, &type));
            require(type == cudaGraphNodeTypeKernel, "graph node must be a kernel");
            cudaKernelNodeParams params{};
            CUDA_CHECK(cudaGraphKernelNodeGetParams(node, &params));
            blocks_ = std::uint64_t(params.gridDim.x) * params.gridDim.y * params.gridDim.z;
            CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
        } catch (...) {
            release();
            throw;
        }
    }

    ~Graph() { release(); }

    Graph(const Graph&)            = delete;
    Graph& operator=(const Graph&) = delete;

    std::uint64_t blocks() const { return blocks_; }

    void replay(const DeviceContext& ctx) {
        CUDA_CHECK(cudaGraphLaunch(exec, ctx.stream()));
        ctx.synchronize();
    }

private:
    void release() noexcept {
        if (exec) (void)cudaGraphExecDestroy(exec);
        if (graph) (void)cudaGraphDestroy(graph);
    }

    std::uint64_t blocks_ = 0;
    cudaGraph_t graph     = nullptr;
    cudaGraphExec_t exec  = nullptr;
};

int run_commit(const Case& c) {
    Operands f(c);
    f.init_commit();
    f.prepare();
    f.launch(f.in, f.out, f.params);
    f.ctx.synchronize();
    int failures = f.check(c.lengths);
    const std::vector<int> zero(c.lengths.size(), 0);
    f.check_commit(zero);
    if (!c.ragged) {
        if (!c.snapshot) f.reject_cyclic_prepare();
        f.reject_commit_operands();
        Graph graph(f.ctx, [&] { f.submit_commit(f.counts_tensor()); });
        require(graph.blocks() <= c.lengths.size() * std::min(c.width, Window),
                "batched commit grid is bounded by the surviving tail");
        f.set_valid(zero);
        graph.replay(f.ctx);
        f.check_commit(zero);
        auto prefix = c.lengths;
        --prefix.front();
        f.set_valid(prefix);
        graph.replay(f.ctx);
        f.check_commit(prefix);
        f.set_valid(c.lengths);
        graph.replay(f.ctx);
        f.check_commit(c.lengths);
        if (c.width == 2 && c.lengths.size() == 1) {
            f.init_commit();
            f.submit_commit({});
            f.ctx.synchronize();
            f.check_commit(c.lengths);
        }
    } else {
        Graph graph(f.ctx, [&] { f.submit_commit({}); });
        graph.replay(f.ctx);
        f.check_commit(c.lengths);
    }
    return failures;
}

int run_long_commit(const Case& c) {
    Operands f(c, true);
    f.init_commit();
    f.seed_encoded_chunk();

    struct CaptureFailure {};

    try {
        Graph failed(f.ctx, [&] {
            f.submit_commit({});
            throw CaptureFailure{};
        });
        throw std::runtime_error("capture did not propagate the injected exception");
    } catch (const CaptureFailure&) {
        cudaStreamCaptureStatus status;
        CUDA_CHECK(cudaStreamIsCapturing(f.ctx.stream(), &status));
        require(status == cudaStreamCaptureStatusNone, "exception left the stream capturing");
    }
    f.check_commit(std::vector<int>(c.lengths.size(), 0));
    Graph graph(f.ctx, [&] { f.submit_commit({}); });
    require(graph.blocks() <= c.lengths.size() * Window,
            "ragged commit grid is bounded by the surviving tail");
    graph.replay(f.ctx);
    f.check_commit(c.lengths);
    return 0;
}

int run(const Case& c) {
    Operands f(c);
    auto launch = [&] { f.launch(f.in, f.out, f.params); };
    f.prepare();
    launch();
    f.ctx.synchronize();
    int failures = f.check(c.lengths);

    if (c.shared_arena || c.split_history_selectors) {
        auto separate_case                    = c;
        separate_case.shared_arena            = false;
        separate_case.split_history_selectors = false;
        Operands separate(separate_case);
        separate.prepare();
        separate.launch(separate.in, separate.out, separate.params);
        separate.ctx.synchronize();
        require(f.output_image() == separate.output_image(), "shared arena changed output bytes");
    }

    Graph graph(f.ctx, launch);

    f.replay_input();
    f.prepare();
    graph.replay(f.ctx);
    failures += f.check(c.lengths);
    if (!c.ragged && !c.unmasked) {
        auto lengths = c.lengths;
        std::fill(lengths.begin(), lengths.end(), 0);
        f.set_valid(lengths);
        f.prepare();
        graph.replay(f.ctx);
        failures += f.check(lengths);
    }

    auto reject = [&](RelativeAttentionPrepareInputs in, RelativeAttentionPrepareOutputs out,
                      RelativeAttentionPrepareParams params) {
        try {
            f.launch(in, out, params);
        } catch (const std::invalid_argument&) { return; }
        throw std::runtime_error("accepted invalid prepare operands");
    };
    auto inputs   = f.in;
    auto outputs  = f.out;
    auto params   = f.params;
    outputs.q_out = inputs.q;
    reject(inputs, outputs, params);
    params.clip_eps = 0;
    reject(inputs, f.out, params);
    if (inputs.k_dyn.data)
        inputs.v_dyn = {};
    else
        inputs.k_dyn = inputs.q;
    reject(inputs, f.out, f.params);
    inputs = f.in;
    if (inputs.sink_x.data)
        inputs.sink_weight = {};
    else
        inputs.sink_weight = inputs.q_scale;
    reject(inputs, f.out, f.params);
    inputs                 = f.in;
    inputs.positions.dtype = DType::FP32;
    reject(inputs, f.out, f.params);
    inputs                  = f.in;
    inputs.positions.device = Device::cpu();
    reject(inputs, f.out, f.params);
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    for (bool fp8 : {false, true}) {
        const std::string format = fp8 ? "fp8" : "bf16";
        for (bool paged : {false, true}) {
            const std::string layout = paged ? "paged" : "cyclic";
            auto add                 = [&](std::string tag, Case c) {
                suite.add(format + "-" + layout + "-" + tag, [c] { return run(c); },
                                          {{"format", format}, {"suite", layout}});
            };
            for (int mode = 0; mode < 3; ++mode) {
                const std::string form = mode == 0 ? "direct" : mode == 1 ? "snapshot" : "ragged";
                for (int combination = 0; combination < 8; ++combination) {
                    const bool dynamic = combination & 1, tau = combination & 2;
                    const int sink = combination & 4 ? 72 : 0;
                    add(form + "-" + std::to_string(combination),
                        {paged, fp8, mode == 1, mode == 2, dynamic, tau, 7, sink,
                         mode == 2 ? std::vector<int>{1, 3, 5, 17} : std::vector<int>{1, 0, 5},
                         mode == 2 ? std::vector<int>{0, 63, 8190, 1023}
                                   : std::vector<int>{8191, 0, 8192},
                         false, mode != 1});
                }
            }
            const std::vector<int> starts{0,    63,   8190, 8191, 8192, 8193, 32767,
                                          1023, 2047, 64,   65,   127,  2048};
            for (auto [batch, width] :
                 {std::pair{1, 1}, {3, 2}, {5, 3}, {8, 8}, {13, 13}, {1, 16}}) {
                const std::vector<int> lengths(batch, width);
                add("off-grid-" + std::to_string(batch) + "x" + std::to_string(width),
                    {paged, fp8, width > 1, false, width % 2 == 0, true, width,
                     width == 8   ? 2304
                     : width == 1 ? 8
                                  : 232,
                     lengths, std::vector<int>(starts.begin(), starts.begin() + batch), true,
                     width == 1, true});
            }
            add("tau-bf16-boundary", {paged,
                                      fp8,
                                      false,
                                      false,
                                      false,
                                      true,
                                      1,
                                      0,
                                      {1},
                                      {32767},
                                      false,
                                      true,
                                      true,
                                      false,
                                      true});
            add("ragged-mixed", {paged,
                                 fp8,
                                 false,
                                 true,
                                 true,
                                 true,
                                 0,
                                 24,
                                 {1, 2, 3, 4, 5, 17, 313, 977, 1021},
                                 {0, 63, 64, 1023, 2047, 8190, 8193, 32767, 8192},
                                 false,
                                 false,
                                 false,
                                 true});
            add("sink-17", {paged, fp8, false, false, true, true, 7, 17, {5, 0, 7}, {63, 0, 8192}});
            for (int width : {17, 145, 200, 203})
                add("sink-tiled-" + std::to_string(width),
                    {paged,
                     fp8,
                     true,
                     false,
                     true,
                     true,
                     13,
                     width,
                     {13, 12, 13, 0, 9, 13, 13, 11, 13, 13, 8, 13, 13, 13, 13, 13, 13, 13, 12, 13},
                     {63, 0, 8190, 0,    8193, 1023, 2047, 0,  2,  63,
                      1,  4, 51,   1023, 63,   1024, 11,   12, 15, 63}});
            Case overflow{paged, fp8, false, false, true, true, 3, 0, {3}, {8191}};
            overflow.overflow_norm = true;
            add("overflow-norm", overflow);
            Case conv_overflow{paged, fp8, false, false, false, false, 1, 0, {1}, {63}};
            conv_overflow.overflow_conv = true;
            add("overflow-convolution-residual", conv_overflow);
            Case underflow{paged, fp8, false, false, true, false, 3, 0, {3}, {1023}};
            underflow.underflow_norm = true;
            add("underflow-norm", underflow);
            Case split{paged, fp8, false, false, true, true, 5, 0, {5, 0, 3}, {63, 0, 8191}};
            split.split_history_selectors = true;
            add("shared-history-distinct-selectors", split);
            if (paged) {
                Case shared{true, fp8, false, false, true, true, 7, 0, {3, 7}, {63, 8192}};
                shared.alias        = true;
                shared.shared_arena = true;
                add("shared-arena", shared);
            } else {
                Case staged{false, fp8, false, false, true, true, 1, 0, {1, 0, 1}, {1024, 0, 2048}};
                staged.chunk_decode = true;
                add("staged-width-one", staged);
                add("cyclic-max-position",
                    {false, fp8, false, false, false, true, 1, 0, {1}, {INT32_MAX}});
                for (bool snapshot : {false, true}) {
                    Case wrap{false, fp8, snapshot, false, true, false, 2, 0, {2}, {1023}};
                    suite.add(format + (snapshot ? "-commit-verify-wrap" : "-commit-direct-wrap"),
                              [wrap] { return run_commit(wrap); }, {{"format", format}});
                }
                for (bool mixed : {false, true}) {
                    Case large{false,
                               fp8,
                               false,
                               true,
                               false,
                               false,
                               0,
                               0,
                               mixed ? std::vector<int>{1, 8192, 17} : std::vector<int>{8192},
                               mixed ? std::vector<int>{63, 1023, 8191} : std::vector<int>{1023}};
                    suite.add(format + (mixed ? "-commit-large-mixed" : "-commit-large-single"),
                              [large] { return run_long_commit(large); }, {{"format", format}});
                }
                for (bool ragged : {false, true}) {
                    Case tail{false,
                              fp8,
                              false,
                              ragged,
                              false,
                              false,
                              1031,
                              0,
                              ragged ? std::vector<int>{1, 1031, 3} : std::vector<int>{1031, 0},
                              ragged ? std::vector<int>{63, 1023, 8191}
                                     : std::vector<int>{1023, 0}};
                    tail.edge = true;
                    suite.add(format + (ragged ? "-commit-ragged-tail" : "-commit-batched-tail"),
                              [tail] { return run_commit(tail); }, {{"format", format}});
                }
            }
            add("ragged-long",
                {paged, fp8, false, true, false, false, 0, 0, {2053}, {1023}, true, true});
        }
    }
    return suite.main(argc, argv);
}
