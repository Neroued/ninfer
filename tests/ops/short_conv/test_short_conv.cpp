#include "ninfer/ops/short_conv.h"

#include "core/decode_graph.h"
#include "ops/block_attn_res/oracle.h"
#include "ops/ragged_fixture.h"
#include "ops/state_fixture.h"

#include <memory>

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::test;

namespace {

enum class Layout { Direct, Snapshot, Ragged };
enum class Residual { None, Addend, Separate, Alias };

struct Case {
    Layout layout;
    int d, width;
    std::vector<int> lengths;
    Residual residual;
    bool in_place, fused, norm;
    int stacked, q;
    bool graph            = false;
    bool shared_source    = false;
    bool same_arena       = false;
    int snapshot_capacity = 0;
};

std::uint16_t rounded(float x) { return f32_to_bf16(x); }

struct Expected {
    std::vector<std::uint16_t> states;
    std::vector<double> ideal;
    std::vector<int> active;
};

Expected reference(int d, const std::vector<int>& lengths, const std::vector<int>& starts,
                   const std::vector<int>& source, const std::vector<int>& destination,
                   const std::vector<std::uint16_t>& x, const std::vector<std::uint16_t>& w,
                   const std::vector<std::uint16_t>& residual,
                   const std::vector<std::uint16_t>& addend,
                   const std::vector<std::uint16_t>& history,
                   std::vector<std::uint16_t> output_state, bool snapshot) {
    Expected result{std::move(output_state), std::vector<double>(x.size()),
                    std::vector<int>(x.size() / d)};
    const auto statesize = std::size_t(d) * 3;
    for (std::size_t row = 0; row < lengths.size(); ++row) {
        if (lengths[row] == 0) continue;
        const auto old_state = history.begin() + source[row] * statesize;
        std::vector<std::uint16_t> window(old_state, old_state + statesize);
        for (int pos = 0; pos < lengths[row]; ++pos) {
            const auto t     = starts[row] + pos;
            result.active[t] = 1;
            const auto base  = std::size_t(t) * d;
            for (int c = 0; c < d; ++c) {
                const float current = bf16_to_f32(x[base + c]);
                double exact        = 0;
                for (int k = 0; k < 4; ++k) {
                    const auto u      = k == 3 ? current : bf16_to_f32(window[k * d + c]);
                    const auto weight = bf16_to_f32(w[k * d + c]);
                    exact += double(u) * weight;
                }
                const auto i          = base + c;
                const auto ideal_bits = rounded(float((exact + current) / std::sqrt(5.0)));
                double ideal          = bf16_to_f32(ideal_bits);
                if (!addend.empty()) {
                    const auto add = bf16_to_f32(addend[i]);
                    const auto res = residual.empty() ? 0.0f : bf16_to_f32(residual[i]);
                    ideal          = (double(res) + add) + ideal;
                }
                result.ideal[i] = ideal;
            }
            std::move(window.begin() + d, window.end(), window.begin());
            std::copy_n(x.begin() + base, d, window.begin() + 2 * d);
            if (snapshot || pos + 1 == lengths[row]) {
                const auto dst = destination[row] + (snapshot ? pos : 0);
                std::copy(window.begin(), window.end(), result.states.begin() + dst * statesize);
            }
        }
    }
    return result;
}

int run(const Case& tc) {
    const auto cuda      = cuda_environment().context.execution_view();
    const auto d         = tc.d;
    const auto rows      = static_cast<int>(tc.lengths.size());
    const bool is_ragged = tc.layout == Layout::Ragged;
    const bool snapshot  = tc.layout == Layout::Snapshot;
    const auto packed    = is_ragged ? std::make_unique<RaggedFixture>(tc.lengths) : nullptr;
    const auto m         = is_ragged ? packed->tokens() : tc.width * rows;
    const auto width     = is_ragged ? m : tc.width;
    const auto shape     = {d, width, is_ragged ? 1 : rows};
    const auto elements  = std::size_t(m) * d;

    const auto x   = bf16_words(attn_res::values(elements, 61));
    const auto w   = bf16_words(attn_res::values(std::size_t(d) * 4, 62));
    const auto res = bf16_words(attn_res::values(
        tc.residual == Residual::Separate || tc.residual == Residual::Alias ? elements : 0, 63));
    const auto add = bf16_words(attn_res::values(tc.residual == Residual::None ? 0 : elements, 64));
    auto dx = to_device(x), dw = to_device(w);
    const Tensor tx(dx.span(), DType::BF16, shape);
    const Tensor tw(dw.span(), DType::BF16, {d, 4});
    std::vector<std::uint16_t> poison(elements, 0x7fc0);
    auto dc = to_device(tc.residual == Residual::Alias ? res : poison);
    Tensor conv(dc.span(), DType::BF16, shape);
    const auto optional = [](DeviceBuffer& storage, const auto& words, auto dims) {
        if (words.empty()) return Tensor{};
        storage = to_device(words);
        return Tensor(storage.span(), DType::BF16, dims);
    };
    DeviceBuffer dr, da, ds, dat, dn, dq, dm;
    const auto residual = tc.residual == Residual::Alias ? conv : optional(dr, res, shape);
    const auto addend   = optional(da, add, shape);
    Tensor stack, attn_add, norm_weight, queries, mixed;
    std::vector<double> host_stack, host_attn_add, host_norm;
    std::vector<float> host_queries;
    if (tc.fused) {
        const auto s  = bf16_words(attn_res::values(elements * tc.stacked, 65));
        const auto a  = bf16_words(attn_res::values(elements, 66, 0.027f));
        const auto n  = bf16_words(attn_res::values(tc.norm ? std::size_t(d) * tc.q : 0, 67));
        host_stack    = decode_bf16_words(s);
        host_attn_add = decode_bf16_words(a);
        host_norm     = decode_bf16_words(n);
        stack         = optional(ds, s, std::initializer_list<int>{d, m, tc.stacked});
        attn_add      = optional(dat, a, shape);
        norm_weight   = optional(dn, n, std::initializer_list<int>{d, tc.q});
        host_queries  = attn_res::values(std::size_t(d) * tc.q, 68);
        dq            = to_device(host_queries);
        queries       = Tensor(dq.span(), DType::FP32, {d, tc.q});
        dm            = DeviceBuffer(cuda.device(), elements * tc.q * (tc.norm ? 2 : 4));
        mixed         = Tensor(dm.span(), tc.norm ? DType::BF16 : DType::FP32, {d, m, tc.q});
    }

    const auto state_words    = std::size_t(d) * 3;
    const auto slots          = tc.snapshot_capacity > 0 ? tc.snapshot_capacity
                                : snapshot               ? (tc.width + 1) * rows + 1
                                                         : rows + 2;
    const auto source_count   = tc.same_arena ? slots + rows : tc.in_place ? slots : rows + 2;
    const bool shared_storage = tc.in_place || tc.same_arena;
    const auto output_slots   = shared_storage ? source_count : slots;
    auto history              = bf16_words(attn_res::values(state_words * source_count, 69));
    std::vector<std::uint16_t> output_state(state_words * output_slots, 0x7fc0);
    std::vector<int> src(rows), dst(rows), starts(rows);
    std::vector<std::int64_t> src_bytes(rows), dst_bytes(rows);
    std::vector<int> snapshot_slots(rows);
    for (int b = 0; b < rows; ++b) {
        dst[b] = snapshot ? (rows - 1 - b) * (tc.width + 1) + 1 : rows - b;
        // Cover an alias in the initial group and one owned by a later column.
        src[b]            = tc.in_place     ? dst[b] + (snapshot ? (b == 0 ? tc.width - 1 : 0) : 0)
                            : tc.same_arena ? slots + b
                            : tc.shared_source ? 1
                                               : b + 1;
        starts[b]         = is_ragged ? packed->prefix[b] : b * width;
        src_bytes[b]      = tc.lengths[b] ? std::int64_t(src[b] * state_words * 2) : -2345678910LL;
        dst_bytes[b]      = tc.lengths[b] ? std::int64_t(dst[b] * state_words * 2) : -2345678910LL;
        snapshot_slots[b] = tc.lengths[b] ? dst[b] : -100000000;
    }
    auto dh          = to_device(history);
    auto dst_offsets = to_device(dst_bytes), src_offsets = to_device(src_bytes);
    auto base_slots = to_device(snapshot_slots), valid_columns = to_device(tc.lengths);
    DeviceBuffer separate;
    if (shared_storage)
        output_state = history;
    else
        separate = to_device(output_state);
    const Tensor source_table(src_offsets.span(), DType::I64, {rows});
    const Tensor destination_table(dst_offsets.span(), DType::I64, {rows});
    const bool dense = !tc.graph && std::all_of(tc.lengths.begin(), tc.lengths.end(),
                                                [width](int len) { return len == width; });
    const auto valid =
        is_ragged || dense ? Tensor{} : Tensor(valid_columns.span(), DType::I32, {rows});
    const Tensor bases(base_slots.span(), DType::I32, {rows});
    Tensor input_state(dh.span(), DType::BF16, {d, 3, source_count});
    auto& storage = shared_storage ? dh : separate;
    Tensor snapshots(storage.span(), DType::BF16, {d, 3, output_slots});
    const auto source      = dense_record_field(input_state, 2, source_table);
    const auto destination = dense_record_field(snapshots, 2, destination_table);
    const BlockAttnResParameters params{1.0f / std::sqrt(float(d)), 1e-5f, 1e-4f, tc.norm};

    const auto body = [&] {
        if (tc.fused) {
            if (is_ragged)
                short_conv_block_attn_res(tx, tw, packed->view, source, destination, residual,
                                          addend, stack, attn_add, queries, norm_weight, params,
                                          conv, mixed, cuda);
            else if (snapshot)
                short_conv_block_attn_res_snapshot(tx, tw, source, snapshots, valid, bases,
                                                   residual, addend, stack, attn_add, queries,
                                                   norm_weight, params, conv, mixed, cuda);
            else
                short_conv_block_attn_res(tx, tw, source, destination, valid, residual, addend,
                                          stack, attn_add, queries, norm_weight, params, conv,
                                          mixed, cuda);
        } else {
            if (is_ragged)
                short_conv_residual(tx, tw, packed->view, source, destination, residual, addend,
                                    conv, cuda);
            else if (snapshot)
                short_conv_residual_snapshot(tx, tw, source, snapshots, valid, bases, residual,
                                             addend, conv, cuda);
            else
                short_conv_residual(tx, tw, source, destination, valid, residual, addend, conv,
                                    cuda);
        }
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable replay;
    if (tc.graph) {
        definition.capture(cuda.stream(), body);
        replay.instantiate(definition);
    }

    int failed               = 0;
    const int trials         = tc.graph ? 2 : 1;
    std::vector<int> lengths = tc.lengths;
    for (int trial = 0; trial < trials; ++trial) {
        if (trial > 0 && !is_ragged && lengths.back() > 0) {
            lengths.back() = 0;
            valid_columns.copy_from_host(lengths.data(), valid_columns.size());
        }
        const auto expected = reference(d, lengths, starts, src, dst, x, w, res, add, history,
                                        output_state, snapshot);
        if (tc.graph)
            replay.launch(cuda.stream());
        else
            body();
        cuda_synchronize(cuda.stream());
        const auto actual_conv = from_device<std::uint16_t>(dc, elements);
        for (std::size_t i = 0; i < actual_conv.size(); ++i) {
            if (!expected.active[i / d] && actual_conv[i] != 0) {
                ++failed;
                std::cerr << "inactive convolution column was not positive BF16 zero\n";
                break;
            }
        }
        constexpr ReductionCriterion criterion{2.5e-3, 1.e-5, 4.e-3};
        failed += verify_reduction("conv FP64", decode_bf16_words(actual_conv), expected.ideal,
                                   criterion);
        failed += verify_exact("state and untouched slots",
                               from_device<std::uint16_t>(storage, state_words * output_slots),
                               expected.states);
        if (tc.fused) {
            std::vector<double> partial(elements);
            for (std::size_t i = 0; i < partial.size(); ++i)
                partial[i] = bf16_to_f32(rounded(float(expected.ideal[i])));
            const auto mix =
                attn_res::oracle(d, m, tc.q, tc.stacked, host_stack, partial, host_attn_add,
                                 host_queries, host_norm, params, expected.active);
            failed += attn_res::check("fused attention", dm, mix, tc.norm);
            const auto actual =
                tc.norm ? from_device_bf16(dm, mix.size()) : from_device_f32(dm, mix.size());
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (!expected.active[(i / d) % m] && (actual[i] != 0 || std::signbit(actual[i]))) {
                    ++failed;
                    std::cerr << "inactive mixed column was not positive zero\n";
                    break;
                }
            }
        }
        output_state = expected.states;
        if (shared_storage) history = output_state;
        if (tc.residual == Residual::Alias && trial + 1 < trials)
            dc.copy_from_host(res.data(), dc.size());
    }
    return failed;
}

int rounding_boundaries(bool fused) {
    const auto cuda = cuda_environment().context.execution_view();
    constexpr int d = 2304;
    auto x          = to_device_bf16(std::vector<float>(d, 1.0f));
    auto w          = to_device_bf16(std::vector<float>(4 * d));
    auto h          = to_device_bf16(std::vector<float>(3 * d));
    auto offset     = to_device(std::vector<std::int64_t>{0});
    std::vector<float> residual(d, 0.0f), addend(d, 0x1p-10f);
    std::vector<std::uint16_t> expected(d, 0x3ee6);
    // Half a BF16 step above the rounded convolution is a tie that rounds
    // up. Cancellation in channel 1 distinguishes the residual addition order.
    residual[1] = 0x1p24f;
    addend[1]   = -0x1p24f;
    expected[1] = 0x3ee5;
    auto y      = to_device_bf16(residual);
    auto da     = to_device_bf16(addend);
    const Tensor tx(x.span(), DType::BF16, {d, 1, 1}), tw(w.span(), DType::BF16, {d, 4});
    const Tensor add(da.span(), DType::BF16, {d, 1, 1});
    Tensor out(y.span(), DType::BF16, {d, 1, 1});
    const auto state = dense_record_field(Tensor(h.span(), DType::BF16, {d, 3, 1}), 2,
                                          Tensor(offset.span(), DType::I64, {1}));
    int failed       = 0;
    if (fused) {
        auto dq = to_device(std::vector<float>(d));
        DeviceBuffer mixed(cuda.device(), std::size_t(d) * sizeof(float));
        Tensor result(mixed.span(), DType::FP32, {d, 1, 1});
        const BlockAttnResParameters params{1.0f, 1.0e-6f, 1.0e-6f, false};
        short_conv_block_attn_res(tx, tw, state, state, {}, out, add, {}, {},
                                  Tensor(dq.span(), DType::FP32, {d, 1}), {}, params, out, result,
                                  cuda);
        cuda_synchronize(cuda.stream());
        std::vector<float> expected_mix(d);
        std::transform(expected.begin(), expected.end(), expected_mix.begin(), bf16_to_f32);
        failed += verify_exact("attention sees the rounded BF16 result",
                               from_device<float>(mixed, d), expected_mix);
    } else {
        short_conv_residual(tx, tw, state, state, {}, out, add, out, cuda);
        cuda_synchronize(cuda.stream());
    }
    return failed + verify_exact("BF16 residual rounding boundaries",
                                 from_device<std::uint16_t>(y, d), expected);
}

int validation() {
    const auto cuda = cuda_environment().context.execution_view();
    const int d     = 8;
    DeviceBuffer data(cuda.device(), 512), state(cuda.device(), 64);
    auto offsets = to_device(std::vector<std::int64_t>{0});
    Tensor source(state.span(), DType::BF16, {d, 3, 1});
    Tensor offset_tensor(offsets.span(), DType::I64, {1});
    const auto field = dense_record_field(source, 2, offset_tensor);
    const Tensor x(data.span().subspan(0), DType::BF16, {d, 1, 1});
    const Tensor w(data.span().subspan(16), DType::BF16, {d, 4});
    Tensor out(data.span().subspan(80), DType::BF16, {d, 1, 1});
    int failed = 0;
    failed += attn_res::rejects("residual without addend", [&] {
        short_conv_residual(x, w, field, field, {}, out, {}, out, cuda);
    });
    auto wrong = x;
    failed += attn_res::rejects(
        "x/out overlap", [&] { short_conv_residual(x, w, field, field, {}, {}, {}, wrong, cuda); });
    const Tensor x_odd(data.span().subspan(0), DType::BF16, {7, 1, 1});
    const Tensor w_odd(data.span().subspan(16), DType::BF16, {7, 4});
    Tensor out_odd(data.span().subspan(80), DType::BF16, {7, 1, 1});
    const Tensor res_odd(data.span().subspan(82), DType::BF16, {7, 1, 1});
    const Tensor add_odd(data.span().subspan(96), DType::BF16, {7, 1, 1});
    const Tensor state_odd(state.span(), DType::BF16, {7, 3, 1});
    const auto field_odd = dense_record_field(state_odd, 2, offset_tensor);
    failed += attn_res::rejects("partial residual alias", [&] {
        short_conv_residual(x_odd, w_odd, field_odd, field_odd, {}, res_odd, add_odd, out_odd,
                            cuda);
    });
    auto invalid_field         = field;
    invalid_field.field.layers = 2;
    failed += attn_res::rejects("unselected state field", [&] {
        short_conv_residual(x, w, invalid_field, field, {}, {}, {}, out, cuda);
    });
    auto bad_ragged = RaggedFixture({1});
    std::vector<int> bad_prefix{0, 0};
    bad_ragged.view.host_cu_seqlens = bad_prefix;
    failed += attn_res::rejects("ragged prefix", [&] {
        short_conv_residual(x, w, bad_ragged.view, field, field, {}, {}, out, cuda);
    });
    return failed;
}

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    const auto add = [&](std::string name, Case tc) { suite.add(name, [=] { return run(tc); }); };
    using enum Layout;
    add("direct_decode", {Direct, 2304, 1, {1}, Residual::None, true, false, false, 0, 1});
    add("fused_decode", {Direct, 2304, 1, {1}, Residual::None, true, true, false, 3, 1, true});
    add("direct_zero_valid", {Direct, 257, 1, {0}, Residual::None, true, true, false, 3, 5});
    add("direct_graph",
        {Direct, 2304, 7, {0, 2, 3, 7, 1}, Residual::Alias, true, true, true, 7, 3, true});
    add("snapshot_distinct",
        {Snapshot, 1024, 3, {3, 0, 2}, Residual::Addend, false, false, false, 0, 1});
    add("snapshot_fused_alias_graph",
        {Snapshot, 257, 8, {8, 0, 5, 1, 2}, Residual::Alias, true, true, false, 2, 5, true});
    add("snapshot_standalone_alias",
        {Snapshot, 1024, 5, {5, 0, 2}, Residual::Alias, true, false, false, 0, 1});
    add("snapshot_batched_off_grid", {Snapshot,
                                      256,
                                      16,
                                      {16, 15, 14, 13, 0, 11, 0, 2, 8, 7, 6, 5, 4},
                                      Residual::Separate,
                                      false,
                                      true,
                                      true,
                                      1,
                                      2});
    add("ragged_short_segments",
        {Ragged, 4096, 0, {1, 2, 3, 4, 5, 17}, Residual::None, true, true, true, 0, 4});
    add("ragged_mixed_shared_history", {Ragged,
                                        257,
                                        0,
                                        {1, 2, 3, 4, 5, 17, 313, 977, 1021},
                                        Residual::Separate,
                                        false,
                                        false,
                                        false,
                                        0,
                                        1,
                                        false,
                                        true});
    add("ragged_mixed_fused", {Ragged,
                               2304,
                               0,
                               {1, 2, 3, 4, 5, 17, 313, 977, 1021},
                               Residual::Addend,
                               true,
                               true,
                               true,
                               5,
                               1});
    add("ragged_single_graph",
        {Ragged, 1024, 0, {17}, Residual::Separate, true, true, false, 3, 2, true});
    add("ragged_unbounded_channels",
        {Ragged, 4104, 0, {1, 4}, Residual::Alias, true, true, true, 1, 2});
    add("snapshot_all_inactive_minimum_capacity", {Snapshot,
                                                   1024,
                                                   8,
                                                   {0},
                                                   Residual::Addend,
                                                   false,
                                                   false,
                                                   false,
                                                   0,
                                                   1,
                                                   false,
                                                   false,
                                                   false,
                                                   1});
    add("fused_snapshot_all_inactive_minimum_capacity", {Snapshot,
                                                         1024,
                                                         8,
                                                         {0},
                                                         Residual::Addend,
                                                         false,
                                                         true,
                                                         true,
                                                         3,
                                                         3,
                                                         false,
                                                         false,
                                                         false,
                                                         1});
    add("ragged_disjoint_records_in_same_arena", {Ragged,
                                                  2304,
                                                  0,
                                                  {2, 4, 17},
                                                  Residual::Alias,
                                                  false,
                                                  true,
                                                  true,
                                                  3,
                                                  1,
                                                  false,
                                                  false,
                                                  true});
    add("batched_disjoint_records_in_same_arena",
        {Direct, 257, 5, {2, 0, 5}, Residual::None, false, false, false, 0, 1, false, false, true});
    suite.add("rounding_boundaries", [] { return rounding_boundaries(false); });
    suite.add("fused_rounding_boundaries", [] { return rounding_boundaries(true); });
    suite.add("validation", validation);
    return suite.main(argc, argv);
}
