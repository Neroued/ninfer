// Packed prefill: several independent sequence windows share one Main traversal. The test drives
// the production RankProgram through ForwardWork and compares the packed execution with the same
// windows executed one by one, and each packed mixer call with the FP64 oracles of tests/ops.
#include "models/qwen3_5/native_fixture.h"

#include "artifact/reader.h"
#include "models/qwen3_5/load.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/softmax_attention.h"
#include "ops/float_codec.h"
#include "ops/linear_attention/gated_delta_net/gdn_ref.h"
#include "ops/op_check.h"
#include "ops/softmax_attention/oracle.h"
#include "symbol_interpose.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// Mixer taps. The test executable replaces the three mixer Ops through the dynamic symbol table
// (ninfer_test_interpose), forwards to the production Op, and records the represented inputs and
// outputs of the first mixer layer of each kind while armed. Verification runs afterwards.
namespace packed_tap {

struct Attention {
    int dim = 0, heads = 0, kv_heads = 0, tokens = 0;
    float scale = 0;
    std::vector<std::uint16_t> q, k, v, out;
    std::vector<std::int32_t> positions;
};

struct Gdn {
    int qk_heads = 0, value_heads = 0, tokens = 0;
    float scale    = 0;
    bool normalize = false;
    std::vector<std::uint16_t> q, k, v, out;
    std::vector<float> g, beta, state_in, state_out;
};

struct Conv {
    int channels = 0, tokens = 0;
    std::array<int, 3> rows{};
    std::vector<std::uint16_t> x, weight, state_in, state_out;
    std::array<std::vector<std::uint16_t>, 3> out;
};

struct Store {
    std::mutex mutex;
    bool armed        = false;
    std::size_t limit = 0;
    std::vector<Attention> attention;
    std::vector<Gdn> gdn;
    std::vector<Conv> conv;
};

inline Store& store() {
    static Store value;
    return value;
}

template <class Record>
bool wants(const std::vector<Record> Store::*member) {
    auto& s = store();
    std::lock_guard lock(s.mutex);
    return s.armed && (s.*member).size() < s.limit;
}

template <class Record>
void push(std::vector<Record> Store::*member, Record&& record) {
    auto& s = store();
    std::lock_guard lock(s.mutex);
    (s.*member).push_back(std::move(record));
}

template <class T>
std::vector<T> download(const ninfer::Tensor& tensor) {
    if (!tensor.is_contiguous() || tensor.bytes() % sizeof(T))
        throw std::runtime_error("mixer tap needs a contiguous tensor");
    std::vector<T> result(tensor.bytes() / sizeof(T));
    CUDA_CHECK(cudaMemcpy(result.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
    return result;
}

} // namespace packed_tap

namespace ninfer::ops {

void causal_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid_columns,
                              const Tensor& kv_table_rows, AttentionHeadGeometry geometry,
                              float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                              Tensor& out, CudaExecutionView execution) {
    using Function =
        void (*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                 const Tensor&, AttentionHeadGeometry, float, PagedKVBatchLayerView,
                 CausalAttentionExecutionEnvelope, MemoryArena&, Tensor&, CudaExecutionView);
    static const auto real = test::next_symbol<Function>(
        "_ZN6ninfer3ops24causal_softmax_attentionERKNS_6TensorES3_S3_S3_S3_S3_NS0_"
        "21AttentionHeadGeometryEfNS_21PagedKVBatchLayerViewENS0_32CausalAttentionExecution"
        "EnvelopeERNS_11MemoryArenaERS1_NS_17CudaExecutionViewE");
    const bool tap = packed_tap::wants<packed_tap::Attention>(&packed_tap::Store::attention);
    real(q, k, v, positions, valid_columns, kv_table_rows, geometry, scale, cache, envelope,
         workspace, out, execution);
    if (!tap) return;
    CUDA_CHECK(cudaStreamSynchronize(execution.stream()));
    packed_tap::Attention record;
    record.dim       = geometry.head_dim;
    record.heads     = geometry.query_heads;
    record.kv_heads  = geometry.kv_heads;
    record.tokens    = q.ne[2];
    record.scale     = scale;
    record.q         = packed_tap::download<std::uint16_t>(q);
    record.k         = packed_tap::download<std::uint16_t>(k);
    record.v         = packed_tap::download<std::uint16_t>(v);
    record.out       = packed_tap::download<std::uint16_t>(out);
    record.positions = packed_tap::download<std::int32_t>(positions);
    packed_tap::push(&packed_tap::Store::attention, std::move(record));
}

void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, MemoryArena& ws,
                     const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                     CudaExecutionView execution) {
    using Function =
        void (*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, float,
                 bool, MemoryArena&, const Tensor&, Tensor&, Tensor&, CudaExecutionView);
    static const auto real = test::next_symbol<Function>(
        "_ZN6ninfer3ops15gated_delta_netERKNS_6TensorES3_S3_S3_S3_fbRNS_11MemoryArenaES3_RS1_"
        "S6_NS_17CudaExecutionViewE");
    const bool tap = packed_tap::wants<packed_tap::Gdn>(&packed_tap::Store::gdn);
    packed_tap::Gdn record;
    if (tap) {
        CUDA_CHECK(cudaStreamSynchronize(execution.stream()));
        record.state_in = packed_tap::download<float>(ssm_state_in);
    }
    real(q, k, v, g, beta, scale, normalize_qk, ws, ssm_state_in, ssm_state_out, out, execution);
    if (!tap) return;
    CUDA_CHECK(cudaStreamSynchronize(execution.stream()));
    record.qk_heads    = q.ne[1];
    record.value_heads = v.ne[1];
    record.tokens      = q.ne[2];
    record.scale       = scale;
    record.normalize   = normalize_qk;
    record.q           = packed_tap::download<std::uint16_t>(q);
    record.k           = packed_tap::download<std::uint16_t>(k);
    record.v           = packed_tap::download<std::uint16_t>(v);
    record.g           = packed_tap::download<float>(g);
    record.beta        = packed_tap::download<float>(beta);
    record.out         = packed_tap::download<std::uint16_t>(out);
    record.state_out   = packed_tap::download<float>(ssm_state_out);
    packed_tap::push(&packed_tap::Store::gdn, std::move(record));
}

void causal_conv1d_silu_split(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                              Tensor& conv_state_out, Tensor& out0, Tensor& out1, Tensor& out2,
                              CudaExecutionView execution) {
    using Function         = void (*)(const Tensor&, const Tensor&, const Tensor&, Tensor&, Tensor&,
                              Tensor&, Tensor&, CudaExecutionView);
    static const auto real = test::next_symbol<Function>(
        "_ZN6ninfer3ops24causal_conv1d_silu_splitERKNS_6TensorES3_S3_RS1_S4_S4_S4_NS_"
        "17CudaExecutionViewE");
    const bool tap = packed_tap::wants<packed_tap::Conv>(&packed_tap::Store::conv);
    packed_tap::Conv record;
    if (tap) {
        CUDA_CHECK(cudaStreamSynchronize(execution.stream()));
        record.state_in = packed_tap::download<std::uint16_t>(conv_state_in);
    }
    real(x, weight, conv_state_in, conv_state_out, out0, out1, out2, execution);
    if (!tap) return;
    CUDA_CHECK(cudaStreamSynchronize(execution.stream()));
    record.channels  = x.ne[0];
    record.tokens    = x.ne[1];
    record.rows      = {out0.ne[0], out1.ne[0], out2.ne[0]};
    record.x         = packed_tap::download<std::uint16_t>(x);
    record.weight    = packed_tap::download<std::uint16_t>(weight);
    record.state_out = packed_tap::download<std::uint16_t>(conv_state_out);
    record.out[0]    = packed_tap::download<std::uint16_t>(out0);
    record.out[1]    = packed_tap::download<std::uint16_t>(out1);
    record.out[2]    = packed_tap::download<std::uint16_t>(out2);
    packed_tap::push(&packed_tap::Store::conv, std::move(record));
}

} // namespace ninfer::ops

namespace {
using namespace ninfer;
namespace q35 = models::qwen3_5;
namespace ctx = models::context;
using Native  = test::qwen3_5::NativeRankFixture;

constexpr std::uint32_t kRows = 7, kPackedRows = 6, kBystander = 6, kChunk = 2048, kContext = 4096,
                        kDrafts = 3, kPage = 64;
constexpr std::uint8_t kSentinel = 0xa5;

// Device allocations observed while a prefill executes (see the cudaMalloc replacements).
std::atomic<bool> g_alloc_active{false};
std::atomic<std::size_t> g_alloc_count{0};

void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

float bf16(std::uint16_t word) { return std::bit_cast<float>(std::uint32_t{word} << 16); }

template <class T>
std::vector<T> read(const Tensor& tensor) {
    require(tensor.is_contiguous() && tensor.bytes() % sizeof(T) == 0, "invalid readback");
    std::vector<T> result(tensor.bytes() / sizeof(T));
    CUDA_CHECK(cudaMemcpy(result.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
    return result;
}

// Packed and sequential execution differ through the batch-shape dependent choices of the
// token-wise projections and routed experts (activation-quantized profiles change with T). The
// budget is the numerical effect of cutting one sequence into different windows: the test runs
// that control (one sequence re-cut into half windows) and holds it to the same budget, so it
// bounds what packing may change without admitting a wrong segment, State or position. Those
// produce errors of order one, and the isolation check below is bitwise-tight.
// Unequal segment lengths change the packed token count of every step, so the final-token hidden
// of a one-token window (a single vector, the noisiest statistic here) reaches 0.12 against its
// B=1 execution while the re-cut control stays near 0.085; the budget leaves room for that.
constexpr test::ReductionCriterion kPackedCriterion{/*relative_l2*/ 0.15,
                                                    /*gross_absolute*/ 0.0,
                                                    /*gross_relative_to_max_reference*/ 0.3};
// Changing another segment's tokens must not move this segment's results.
constexpr test::ReductionCriterion kIsolationCriterion{1.0e-4, 0.0, 1.0e-3};

// The Op suites bound the worst element by a few 1e-3 of the largest reference for unit-RMS
// fixtures. Model activations reach |x| > 4, where one BF16 half-ulp of the output alone is 2^-8
// of the maximum, so the worst-element bound here is one BF16 ulp (2^-7) of the largest reference
// while the normwise budgets stay those of the Op suites.
constexpr test::ReductionCriterion kAttentionCriterion{2.8e-3, 1.0e-3, 0x1p-7};
constexpr test::ReductionCriterion kGdnOutputCriterion{6.0e-3, 5.0e-6, 8.0e-3};
constexpr test::ReductionCriterion kGdnStateCriterion{4.5e-3, 1.0e-5, 9.0e-3};
constexpr test::ReductionCriterion kConvCriterion{1.85e-3, 1.0e-3, 0x1p-7};

void compare(const std::string& label, const std::vector<double>& got,
             const std::vector<double>& reference, const test::ReductionCriterion& criterion) {
    require(got.size() == reference.size() && !got.empty(), label + ": size mismatch");
    const auto stats = test::compute_reduction_stats(got.data(), reference.data(), got.size());
    std::cout << "  " << label << ": rel_l2=" << stats.relative_l2
              << " max_abs=" << stats.maximum_absolute_error
              << " max_ref=" << stats.maximum_absolute_reference << '\n';
    require(test::reduction_passes(stats, got.size(), criterion), label + ": outside budget");
}

std::vector<double> widen(const std::vector<float>& values) {
    return {values.begin(), values.end()};
}

std::vector<double> widen_bf16(const std::vector<std::uint16_t>& words) {
    std::vector<double> result;
    result.reserve(words.size());
    for (auto word : words) result.push_back(bf16(word));
    return result;
}

// Attention is checked on a sample of query rows against the FP64 dense oracle. The first window
// of a sequence has no cached prefix, so its keys are the packed segment's own K and V.
void verify_attention(const packed_tap::Attention& c, const std::string& label) {
    if (c.positions.empty() || c.positions[0] != 0) return;
    std::vector<int> picks;
    for (int t = 0; t < c.tokens; ++t)
        if (c.tokens <= 64 || t < 4 || t >= c.tokens - 4 || t % 61 == 0) picks.push_back(t);
    const int D = c.dim, H = c.heads, K = c.kv_heads;
    const ops::AttentionHeadGeometry geometry{D, H, K};
    std::vector<double> reference(std::size_t(D) * H * picks.size()), actual(reference.size());
    test::naive_dense_softmax_attention(
        geometry, int(picks.size()), c.tokens, c.scale,
        [&](int d, int h, int i) {
            return double(bf16(c.q[d + D * (h + std::size_t(H) * picks[i])]));
        },
        [&](int d, int h, int j) { return double(bf16(c.k[d + D * (h + std::size_t(K) * j)])); },
        [&](int d, int h, int j) {
            // V is stored as FP16(RNE) of the BF16 input.
            return double(test::float_codec::f16_bits_to_f32(
                test::float_codec::f32_to_f16_bits(bf16(c.v[d + D * (h + std::size_t(K) * j)]))));
        },
        [&](int i, int j) { return j <= picks[i]; },
        [&](int d, int h, int i, double value) {
            reference[d + D * (h + std::size_t(H) * i)] = value;
        },
        true);
    for (std::size_t i = 0; i < picks.size(); ++i)
        for (int h = 0; h < H; ++h)
            for (int d = 0; d < D; ++d)
                actual[d + D * (h + std::size_t(H) * i)] =
                    bf16(c.out[d + D * (h + std::size_t(H) * picks[i])]);
    compare(label + " attention", actual, reference, kAttentionCriterion);
}

void verify_gdn(const packed_tap::Gdn& c, const std::string& label) {
    test::gdn_ref::Inputs in;
    in.head_dim    = 128;
    in.qk_heads    = c.qk_heads;
    in.value_heads = c.value_heads;
    in.tokens      = c.tokens;
    for (auto word : c.q) in.q.push_back(bf16(word));
    for (auto word : c.k) in.k.push_back(bf16(word));
    for (auto word : c.v) in.v.push_back(bf16(word));
    in.g                 = c.g;
    in.beta              = c.beta;
    in.state             = c.state_in;
    const auto reference = test::gdn_ref::evaluate(in, c.scale, c.normalize);
    compare(label + " gated delta net output", widen_bf16(c.out), reference.out,
            kGdnOutputCriterion);
    compare(label + " gated delta net state", widen(c.state_out), reference.final_state,
            kGdnStateCriterion);
}

void verify_conv(const packed_tap::Conv& c, const std::string& label) {
    const std::size_t C = c.channels, T = c.tokens;
    const auto silu = [](double x) { return x / (1.0 + std::exp(-x)); };
    std::vector<double> reference(C * T), actual(C * T);
    std::vector<std::uint16_t> expected_state(C * 3);
    for (std::size_t ch = 0; ch < C; ++ch) {
        const auto input = [&](long t) {
            return double(t < 0 ? bf16(c.state_in[ch + C * (t + 3)]) : bf16(c.x[ch + C * t]));
        };
        for (std::size_t t = 0; t < T; ++t) {
            double sum = 0;
            for (int j = 0; j < 4; ++j)
                sum += double(bf16(c.weight[ch + C * j])) * input(long(t) - 3 + j);
            reference[ch + C * t] = silu(sum);
            const int part        = ch < std::size_t(c.rows[0])               ? 0
                                    : ch < std::size_t(c.rows[0] + c.rows[1]) ? 1
                                                                              : 2;
            const std::size_t row = ch - (part == 0   ? 0
                                          : part == 1 ? c.rows[0]
                                                      : c.rows[0] + c.rows[1]);
            actual[ch + C * t]    = bf16(c.out[part][row + std::size_t(c.rows[part]) * t]);
        }
        for (int j = 0; j < 3; ++j) {
            const long t               = long(T) - 3 + j;
            expected_state[ch + C * j] = t < 0 ? c.state_in[ch + C * (t + 3)] : c.x[ch + C * t];
        }
    }
    compare(label + " causal conv", actual, reference, kConvCriterion);
    require(expected_state == c.state_out, label + " causal conv: trailing window differs");
}

struct Snapshot {
    std::vector<float> main_k, main_v, mtp_k, mtp_v, recurrent, conv, hidden, logits;
};

struct Segment {
    std::uint32_t row, tokens, seed;
};

// One Context holding kRows independent sequences. The last row is a bystander that no work
// addresses; its State and KV records keep their sentinel bytes throughout.
class Rig {
public:
    explicit Rig(Native& native)
        : native_(native),
          pool_(0, native.rank().resources.context.size(), native.rank().resources.context_records),
          scratch_(native.device().device(), 256ULL << 20), arena_(scratch_.span()) {
        const auto state_class    = pool_.record_class(0);
        const std::uint32_t pages = kContext / kPage;
        const std::vector<RecordRequest> demand{{state_class, kRows},
                                                {pool_.record_class(1), kRows * pages},
                                                {pool_.record_class(2), kRows * pages}};
        auto result = pool_.reserve(demand);
        require(result.reservation && result.reservation->ready(),
                "Context capacity cannot hold the packed prefill rows");
        auto& reservation = *result.reservation;
        for (auto& state : states_) state = pool_.resolve(reservation.materialize(state_class));
        q35::ContextWork bindings;
        q35::Work work;
        for (std::uint32_t row = 0; row < kRows; ++row) {
            for (std::uint32_t kind = 0; kind < 2; ++kind) {
                std::vector<std::int64_t> offsets;
                for (std::uint32_t page = 0; page < pages; ++page) {
                    const auto record =
                        pool_.resolve(reservation.materialize(pool_.record_class(kind + 1)));
                    pages_[kind][row].push_back(record);
                    offsets.push_back(static_cast<std::int64_t>(record.offset));
                }
                work.context.operations.emplace_back(
                    ctx::MappingUpdate{kind, static_cast<std::int32_t>(row), 0, offsets});
            }
            rows_[row] = {.member            = {{row + 1, 1}, {{row}, 1}},
                          .source_state      = {row, 1},
                          .destination_state = {row, 1},
                          .state_offsets     = {static_cast<std::int64_t>(states_[row].offset),
                                                static_cast<std::int64_t>(states_[row].offset)},
                          .main_table        = {0, row, 1},
                          .backend_table     = ctx::TableRow{1, row, 1}};
            bindings.bindings.push_back({rows_[row].member, {}, {}});
        }
        work.operation    = std::move(bindings);
        const auto ticket = submit(std::move(work), runtime::FrameKind::Context);
        (void)result_of(ticket);
        retire(ticket);
    }

    // Clears every State and fills every KV record with the sentinel, so that anything a run
    // writes outside its own sequence extent is visible.
    void reset() {
        q35::Work work;
        for (const auto& state : states_)
            work.context.operations.emplace_back(ctx::RecordClear{range(state)});
        work.operation    = q35::ContextWork{};
        const auto ticket = submit(std::move(work), runtime::FrameKind::Context);
        (void)result_of(ticket);
        retire(ticket);
        native_.device().synchronize();
        auto* base = static_cast<std::uint8_t*>(native_.rank().resources.context.data());
        for (const auto& kind : pages_)
            for (const auto& row : kind)
                for (const auto& page : row)
                    CUDA_CHECK(cudaMemset(base + page.offset, kSentinel, page.bytes));
        CUDA_CHECK(
            cudaMemset(base + states_[kBystander].offset, kSentinel, states_[kBystander].bytes));
        for (auto& row : rows_) {
            row.main_frontier = row.backend_frontier = 0;
            row.tail_rope_position                   = {};
        }
        bystander_ = bytes(states_[kBystander]);
    }

    [[nodiscard]] std::uint32_t frontier(std::uint32_t row) const {
        return rows_[row].main_frontier;
    }

    Native& native_;
    ContextPool pool_;
    DeviceBuffer scratch_;
    MemoryArena arena_;
    std::array<RecordBinding, kRows> states_;
    std::array<std::array<std::vector<RecordBinding>, kRows>, 2> pages_;
    std::array<q35::ExecutionRow, kRows> rows_;
    std::vector<std::uint8_t> bystander_;
    bool strict = false;

    static ctx::ContextRange range(const RecordBinding& binding) {
        return {ctx::ContextMemory::Device, static_cast<std::int64_t>(binding.offset),
                binding.bytes};
    }

    std::vector<std::uint8_t> bytes(const RecordBinding& binding) {
        std::vector<std::uint8_t> result(binding.bytes);
        CUDA_CHECK(cudaMemcpy(result.data(),
                              static_cast<std::uint8_t*>(native_.rank().resources.context.data()) +
                                  binding.offset,
                              binding.bytes, cudaMemcpyDeviceToHost));
        return result;
    }

    runtime::ExecutionTicket submit(q35::Work work, runtime::FrameKind kind) {
        auto reservation = native_.executor().try_reserve(kind);
        require(bool(reservation), "native frame is unexpectedly occupied");
        return native_.executor().submit(std::move(*reservation), std::move(work));
    }

    q35::Result result_of(runtime::ExecutionTicket ticket) {
        for (;;) {
            native_.device().synchronize();
            CUDA_CHECK(cudaStreamSynchronize(native_.device().transfer_stream()));
            auto event = native_.executor().poll();
            if (!event) continue;
            if (auto* failure = std::get_if<runtime::ExecutionFailed>(&*event))
                throw std::runtime_error(failure->message);
            if (auto* ready = std::get_if<runtime::ResultReady<q35::ExecutionTypes>>(&*event)) {
                require(ready->ticket == ticket, "unexpected native result ticket");
                return std::move(ready->result);
            }
            throw std::runtime_error("unexpected native event before result");
        }
    }

    void retire(runtime::ExecutionTicket ticket) {
        for (;;) {
            native_.device().synchronize();
            auto event = native_.executor().poll();
            if (!event) continue;
            if (auto* failure = std::get_if<runtime::ExecutionFailed>(&*event))
                throw std::runtime_error(failure->message);
            if (auto* retired = std::get_if<runtime::DeviceRetired>(&*event)) {
                require(retired->ticket == ticket, "unexpected native retirement");
                return;
            }
            throw std::runtime_error("unexpected native event before retirement");
        }
    }

    static std::vector<std::int64_t> table(const PagedKVBatchLayerView& view, int row,
                                           std::uint32_t pages) {
        auto offsets = read<std::int64_t>(view.k_pages.records.page_offsets.slice(1, row, 1));
        require(offsets.size() >= pages, "KV table is shorter than the sequence");
        offsets.resize(pages);
        return offsets;
    }

    // K as BF16 and V as FP16 words of positions [0, tokens), ordered [position][head][dim].
    static void append_kv(const PagedKVBatchLayerView& view, int row, std::uint32_t tokens,
                          std::vector<float>& k, std::vector<float>& v) {
        const auto offsets  = table(view, row, (tokens + kPage - 1) / kPage);
        const std::size_t D = view.head_dim;
        for (std::uint32_t page = 0; page < offsets.size(); ++page) {
            const auto kw = read<std::uint16_t>(view.k_pages.tensor(offsets[page]));
            const auto vw = read<std::uint16_t>(view.v_pages.tensor(offsets[page]));
            for (auto p = page * kPage; p < std::min(tokens, (page + 1) * kPage); ++p)
                for (int head = 0; head < view.num_kv_heads; ++head)
                    for (std::size_t d = 0; d < D; ++d) {
                        const auto at = d + D * (p % kPage + std::size_t(kPage) * head);
                        k.push_back(bf16(kw[at]));
                        v.push_back(test::float_codec::f16_bits_to_f32(vw[at]));
                    }
        }
    }

    // Every KV word at or beyond `tokens` in the row's mapped pages still holds the sentinel.
    static void require_untouched_tail(const PagedKVBatchLayerView& view, int row,
                                       std::uint32_t tokens, const std::string& label) {
        const auto offsets  = table(view, row, kContext / kPage);
        const std::size_t D = view.head_dim;
        for (std::uint32_t page = tokens / kPage; page < offsets.size(); ++page)
            for (const auto* plane : {&view.k_pages, &view.v_pages}) {
                const auto words = read<std::uint16_t>(plane->tensor(offsets[page]));
                for (auto p = std::max(tokens, page * kPage); p < (page + 1) * kPage; ++p)
                    for (int head = 0; head < view.num_kv_heads; ++head)
                        for (std::size_t d = 0; d < D; ++d)
                            require(words[d + D * (p % kPage + std::size_t(kPage) * head)] ==
                                        0xa5a5,
                                    label + ": KV written beyond the sequence at position " +
                                        std::to_string(p));
            }
    }

    Snapshot snapshot(std::uint32_t row, std::uint32_t tokens) {
        auto& rank = native_.rank();
        Snapshot out;
        auto& text = rank.resources.decoder.text_kv;
        for (std::uint32_t layer = 0; layer < text.layers(); ++layer)
            append_kv(text.batch_layer_view(layer), row, tokens, out.main_k, out.main_v);
        if (auto* mtp = rank.resources.decoder.mtp_cache())
            for (std::uint32_t layer = 0; layer < mtp->layers(); ++layer)
                append_kv(mtp->batch_layer_view(layer), row, tokens ? tokens - 1 : 0, out.mtp_k,
                          out.mtp_v);
        const auto offset = rows_[row].state_offsets.destination;
        auto& linear      = rank.resources.state_images.linear();
        for (std::uint32_t layer = 0; layer < linear.layer_count(); ++layer) {
            for (float value : read<float>(linear.recurrent_tensor(layer, offset)))
                out.recurrent.push_back(value);
            for (auto word : read<std::uint16_t>(linear.conv_tensor(layer, offset)))
                out.conv.push_back(bf16(word));
        }
        const auto hidden = rank.resources.state_images.continuation_hidden(offset);
        for (auto word : read<std::uint16_t>(hidden)) out.hidden.push_back(bf16(word));
        // The continuation hidden is the final-normalized last-token hidden. Project it through
        // the output head with the same Op the sampler uses.
        const int H = rank.resources.frame.spec.hidden, V = rank.resources.frame.spec.output_rows;
        DeviceBuffer input(native_.device().device(), std::size_t(H) * 2);
        CUDA_CHECK(cudaMemcpy(input.data(), hidden.data, hidden.bytes(), cudaMemcpyDeviceToDevice));
        DeviceBuffer output(native_.device().device(), std::size_t(V) * 2);
        const Tensor x(input.span(), DType::BF16, {H, 1});
        Tensor y(output.span(), DType::BF16, {V, 1});
        arena_.reset();
        const auto& head = rank.parameters.text.output_head;
        ops::linear(x, head.weight, y, head.policy, arena_, native_.device().execution_view());
        native_.device().synchronize();
        for (auto word : read<std::uint16_t>(y)) out.logits.push_back(bf16(word));
        return out;
    }

    q35::InputWindow window(const Segment& segment) const {
        const auto base = rows_[segment.row].main_frontier;
        q35::InputWindow out;
        out.tokens.resize(segment.tokens);
        out.rope_positions.resize(3 * std::size_t(segment.tokens));
        for (std::uint32_t i = 0; i < segment.tokens; ++i) {
            const std::uint64_t position = base + i;
            out.tokens[i] =
                static_cast<TokenId>(1000 + (segment.seed * 2654435761ULL + position * 40503ULL +
                                             position / 7 * 977ULL) %
                                                200000);
            for (int axis = 0; axis < 3; ++axis)
                out.rope_positions[axis * segment.tokens + i] =
                    static_cast<std::int32_t>(position) + 2 * axis;
        }
        return out;
    }

    // One ForwardWork. Several rows are executed as one packed traversal; one row is the
    // ordinary B=1 prefill.
    void execute(std::span<const Segment> group) {
        q35::ForwardWork work;
        work.purpose = q35::ForwardPurpose::Prefill;
        std::vector<q35::InputWindow> windows;
        for (const auto& segment : group) {
            work.rows.push_back(rows_[segment.row]);
            windows.push_back(window(segment));
        }
        work.inputs       = windows;
        const auto ticket = submit({.operation = std::move(work)}, runtime::FrameKind::Compute);
        const auto value  = std::get<q35::ForwardResult>(result_of(ticket).output);
        std::uint64_t processed = 0;
        for (auto count : value.processed_tokens) processed += count;
        std::uint64_t expected = 0;
        for (const auto& segment : group) expected += segment.tokens;
        require(processed == expected && !value.generated, "prefill processed the wrong tokens");
        (void)native_.executor().commit(ticket, {});
        retire(ticket);
        for (std::size_t i = 0; i < group.size(); ++i) {
            auto& row = rows_[group[i].row];
            row.main_frontier += group[i].tokens;
            row.backend_frontier = row.main_frontier - 1;
            for (int axis = 0; axis < 3; ++axis)
                row.tail_rope_position[axis] =
                    windows[i].rope_positions[axis * group[i].tokens + group[i].tokens - 1];
        }
    }

    // Executes the windows. A tapped run records the first mixer layer of each kind and checks
    // every segment's mixer against the FP64 oracles; every run must leave the device free memory
    // and the allocator call count unchanged and must not touch the bystander row.
    void run(std::span<const Segment> segments, bool packed, bool tapped) {
        auto& taps = packed_tap::store();
        if (tapped) {
            std::lock_guard lock(taps.mutex);
            taps.armed = true;
            taps.limit = segments.size();
            taps.attention.clear();
            taps.gdn.clear();
            taps.conv.clear();
        }
        std::size_t free_before = 0, free_after = 0, total = 0;
        native_.device().synchronize();
        CUDA_CHECK(cudaMemGetInfo(&free_before, &total));
        g_alloc_count  = 0;
        g_alloc_active = true;
        if (packed) {
            execute(segments);
        } else {
            for (const auto& segment : segments) execute({&segment, 1});
        }
        g_alloc_active = false;
        CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
        // The first pass lazily loads kernels and may allocate; once warm, execution must not.
        if (strict) {
            require(g_alloc_count == 0, "execution called the device allocator");
            require(free_after >= free_before, "device free memory shrank during execution");
        }
        require(bytes(states_[kBystander]) == bystander_, "bystander State was modified");
        if (!tapped) return;
        std::lock_guard lock(taps.mutex);
        taps.armed          = false;
        const auto expected = segments.size();
        require(taps.attention.size() == expected && taps.gdn.size() == expected &&
                    taps.conv.size() == expected,
                "mixer taps did not observe one call per segment");
        for (std::size_t i = 0; i < expected; ++i) {
            const auto label = "segment " + std::to_string(i) + " (" +
                               std::to_string(segments[i].tokens) + " tokens)";
            verify_attention(taps.attention[i], label);
            verify_gdn(taps.gdn[i], label);
            verify_conv(taps.conv[i], label);
        }
    }

    // After the windows, every row holds exactly its extent: nothing beyond it was written.
    void require_extents(std::span<const std::uint32_t> rows) {
        auto& rank = native_.rank();
        auto& text = rank.resources.decoder.text_kv;
        for (auto row : rows) {
            const auto label = "row " + std::to_string(row);
            for (std::uint32_t layer = 0; layer < text.layers(); ++layer)
                require_untouched_tail(text.batch_layer_view(layer), row, rows_[row].main_frontier,
                                       label + " main");
            if (auto* mtp = rank.resources.decoder.mtp_cache())
                require_untouched_tail(mtp->batch_layer_view(0), row, rows_[row].main_frontier - 1,
                                       label + " MTP");
        }
        require_untouched_tail(text.batch_layer_view(0), kBystander, 0, "bystander main");
    }
};

void compare_snapshots(const std::string& label, const Snapshot& got, const Snapshot& reference,
                       const test::ReductionCriterion& criterion = kPackedCriterion) {
    compare(label + " main K", widen(got.main_k), widen(reference.main_k), criterion);
    compare(label + " main V", widen(got.main_v), widen(reference.main_v), criterion);
    if (!reference.mtp_k.empty()) {
        compare(label + " MTP K", widen(got.mtp_k), widen(reference.mtp_k), criterion);
        compare(label + " MTP V", widen(got.mtp_v), widen(reference.mtp_v), criterion);
    }
    compare(label + " GDN recurrent state", widen(got.recurrent), widen(reference.recurrent),
            criterion);
    compare(label + " GDN conv state", widen(got.conv), widen(reference.conv), criterion);
    compare(label + " last-token hidden", widen(got.hidden), widen(reference.hidden), criterion);
    compare(label + " last-token logits", widen(got.logits), widen(reference.logits), criterion);
    const auto top = [](const std::vector<float>& logits) {
        return std::max_element(logits.begin(), logits.end()) - logits.begin();
    };
    std::cout << "  " << label << " top-1 token: packed " << top(got.logits) << " sequential "
              << top(reference.logits) << '\n';
}

struct CaseSpec {
    std::string name;
    // windows[w][i] is the length of segment i in the w-th packed execution. Segment i is the
    // same sequence (row) in every window and continues from the previous one.
    std::vector<std::vector<std::uint32_t>> windows;
};

// Tokens depend on the row and the absolute position only, never on how windows are cut.
std::vector<Segment> segments_of(const std::vector<std::uint32_t>& lengths,
                                 std::uint32_t perturb_row = kRows, std::uint32_t perturb = 0) {
    std::vector<Segment> result;
    for (std::uint32_t i = 0; i < lengths.size(); ++i)
        if (lengths[i])
            result.push_back({i, lengths[i], 17 + i + (i == perturb_row ? perturb : 0)});
    return result;
}

std::vector<Snapshot> execute_windows(Rig& rig, const CaseSpec& spec, bool packed, bool tapped,
                                      std::uint32_t perturb_row = kRows,
                                      std::uint32_t perturb     = 0) {
    rig.reset();
    for (std::uint32_t w = 0; w < spec.windows.size(); ++w)
        rig.run(segments_of(spec.windows[w], perturb_row, perturb), packed, tapped && w == 0);
    std::vector<std::uint32_t> rows(spec.windows.front().size());
    std::iota(rows.begin(), rows.end(), 0U);
    rig.require_extents(rows);
    std::vector<Snapshot> result;
    for (auto row : rows) result.push_back(rig.snapshot(row, rig.frontier(row)));
    return result;
}

void run_case(Rig& rig, const CaseSpec& spec, bool isolation) {
    std::cout << "case " << spec.name << '\n';
    const auto packed     = execute_windows(rig, spec, true, true);
    const auto sequential = execute_windows(rig, spec, false, false);
    {
        // Noise floor: the same sequences re-cut into half windows and run one by one.
        CaseSpec halves{spec.name + " resplit", {}};
        for (const auto& window : spec.windows) {
            std::vector<std::uint32_t> first, second;
            for (auto length : window) {
                first.push_back((length + 1) / 2);
                second.push_back(length / 2);
            }
            halves.windows.push_back(std::move(first));
            halves.windows.push_back(std::move(second));
        }
        const auto resplit = execute_windows(rig, halves, false, false);
        for (std::size_t i = 0; i < resplit.size(); ++i)
            compare_snapshots(halves.name + " row " + std::to_string(i), resplit[i], sequential[i]);
    }
    for (std::size_t i = 0; i < packed.size(); ++i)
        compare_snapshots(spec.name + " row " + std::to_string(i), packed[i], sequential[i]);
    if (!isolation) return;
    // One segment's tokens change at a time; no other segment's KV, State or logits may follow,
    // while the changed segment's own result does move.
    for (std::uint32_t row = 0; row < packed.size(); ++row) {
        const auto changed = execute_windows(rig, spec, true, false, row, 1000);
        for (std::size_t i = 0; i < packed.size(); ++i) {
            if (i == row) continue;
            compare_snapshots(spec.name + " perturbed row " + std::to_string(row) + ", row " +
                                  std::to_string(i),
                              changed[i], packed[i], kIsolationCriterion);
        }
        const auto other = test::compute_reduction_stats(widen(changed[row].hidden).data(),
                                                         widen(packed[row].hidden).data(),
                                                         changed[row].hidden.size());
        require(other.relative_l2 > 0.05, "changing a segment's tokens did not change its hidden");
    }
}

} // namespace

// Allocator probe: any device allocation between execute() entry and exit is a violation.
extern "C" {
cudaError_t CUDARTAPI cudaMalloc(void** pointer, std::size_t bytes) {
    static const auto real =
        ninfer::test::next_symbol<cudaError_t (*)(void**, std::size_t)>("cudaMalloc");
    if (g_alloc_active) ++g_alloc_count;
    return real(pointer, bytes);
}

cudaError_t CUDARTAPI cudaMallocAsync(void** pointer, std::size_t bytes, cudaStream_t stream) {
    static const auto real =
        ninfer::test::next_symbol<cudaError_t (*)(void**, std::size_t, cudaStream_t)>(
            "cudaMallocAsync");
    if (g_alloc_active) ++g_alloc_count;
    return real(pointer, bytes, stream);
}
}

int main() {
    std::cout << std::unitbuf;
    const char* path = std::getenv("NINFER_TEST_ARTIFACT");
    if (!path || !*path) {
        std::cout << "SKIP: set NINFER_TEST_ARTIFACT to an explicit MTP .ninfer artifact\n";
        return 77;
    }
    try {
        DeviceContext loading;
        artifact::Reader reader(path);
        auto definition = q35::read_definition(reader, {.speculative = SpeculativeBackend::Mtp});
        auto profile    = q35::execution::describe_model(*definition);
        std::shared_ptr<const q35::ModelShard> model = q35::materialize_model(
            q35::resolve_load(reader, definition, loading.device()), loading);
        EngineOptions options;
        options.max_context                       = kContext;
        options.max_concurrency                   = kRows;
        options.prefill_chunk                     = kChunk;
        options.kv_cache                          = KvCacheStorage::BFloat16;
        options.use_cuda_graph                    = false;
        options.speculative.backend               = SpeculativeBackend::Mtp;
        options.speculative.draft_tokens          = kDrafts;
        options.context_cache.enabled             = false;
        options.context_cache.host_capacity_bytes = 0;
        Native fixture(definition, model, profile, options);
        auto& spec = fixture.rank().resources.frame.spec;
        require(spec.pack_tokens == q35::resolve_prefill_pack_tokens(kChunk, kContext) &&
                    spec.pack_tokens >= 2 * kChunk && spec.pack_segments == kRows,
                "the frame does not plan the packed extent");
        Rig rig(fixture);
        const std::vector<CaseSpec> cases{
            {"one segment", {{2048}}},
            {"two segments with one token and 2048", {{1, 2048}}},
            {"two segments, no multiple of 128", {{129, 777}}},
            {"two segments cut by caps", {{2048, 2047}, {1, 777}}},
            {"six segments", {{1, 130, 257, 2048, 3, 600}, {5, 300, 1, 70, 129, 640}}},
            {"six segments filling the pack", {{683, 683, 683, 683, 682, 682}, {1, 1, 2, 3, 5, 8}}},
            {"six single tokens", {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1}}},
        };
        // The cases whose segments are unequal are also checked for segment isolation.
        const auto isolated = [](const std::string& name) {
            return name == "two segments with one token and 2048" ||
                   name == "two segments cut by caps" || name == "six segments" ||
                   name == "six segments filling the pack";
        };
        std::size_t first_peak = 0;
        for (int pass = 0; pass < 2; ++pass) {
            // The first pass warms every lazily loaded kernel; the second must find the
            // workspace high-water mark and the device memory unchanged.
            rig.strict = pass == 1;
            if (pass == 1) fixture.rank().reset_measurement_peaks();
            for (const auto& item : cases) run_case(rig, item, !pass && isolated(item.name));
            const auto peak = fixture.rank().workspace_peak;
            std::cout << "pass " << pass << " workspace peak " << peak << " of "
                      << fixture.rank().workspace_plan.text_prefill << " bytes\n";
            require(peak <= fixture.rank().workspace_plan.text_prefill,
                    "the workspace peak exceeds the planned prefill workspace");
            if (!pass) first_peak = peak;
            require(pass == 0 || peak <= first_peak, "the workspace peak grew after warm-up");
        }
        std::cout << "OK packed prefill matches sequential B=1 and the FP64 mixer oracles\n";
    } catch (const std::exception& error) {
        std::cerr << "packed prefill: " << error.what() << '\n';
        return 1;
    }
}
