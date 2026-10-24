#include "ops/suite.h"
#include "cuda_fixture.h"
#include "core/decode_graph.h"
#include "../../../../tools/support/kpool_prepare_fixture.h"

#include <iostream>
using namespace ninfer;
using namespace ninfer::test;
namespace kp = ninfer::tool::kpool;

namespace {
int run(bool alias, int D) {
    auto& device    = cuda_environment().context;
    constexpr int B = 8, L = 11, W = 11;
    kp::Buffer<std::uint16_t> keys(device.device(), DType::BF16, {D, W, B, L}, true),
        gates(device.device(), DType::BF16, {D, W, B, L}, true);
    kp::Buffer<int> bases(device.device(), DType::I32, {B}),
        valid(device.device(), DType::I32, {B}), accepted(device.device(), DType::I32, {B});
    kp::Buffer<std::int64_t> src(device.device(), DType::I64, {B}),
        dst(device.device(), DType::I64, {B});
    StateImageBuilder builder;
    (void)builder.add(DType::I32, {64}, 1, "guard");
    auto fields       = ops::append_kpool_tail_fields(builder, D, L);
    auto layout       = builder.finish();
    const auto stride = std::int64_t(layout.image_bytes + 256);
    std::vector<std::uint16_t> initial((256 + 2 * B * stride) / 2, 0xa5a5);
    DeviceBuffer storage(device.device(), initial.size() * 2);
    StateImageStorage state(storage.span(), layout);
    ops::KpoolTailStateView before{state.field(fields.raw_keys, src.view),
                                   state.field(fields.raw_gates, src.view)},
        after{state.field(fields.raw_keys, dst.view), state.field(fields.raw_gates, dst.view)};
    for (int b = 0; b < B; ++b) {
        bases.at(b)    = 60 + b % 4;
        valid.at(b)    = W;
        accepted.at(b) = b == 0 ? 0 : b < 4 ? 1 : b + 3;
        src.at(b)      = b ? 256 + b * stride : -1;
        dst.at(b)      = b ? 256 + (alias ? b : b + B) * stride : -1;
    }
    const auto at = [&](bool output, int kind, int b, int l, int r, int d) {
        const auto& f = kind ? before.raw_gates.field : before.raw_keys.field;
        return ((output ? dst.at(b) : src.at(b)) + f.tensor.region.offset + l * f.layer_stride +
                r * f.byte_strides[1]) /
                   2 +
               d;
    };
    for (int b = 1; b < B; ++b)
        for (int l = 0; l < L; ++l)
            for (int r = 0; r < 3; ++r)
                for (int d = 0; d < D; ++d)
                    for (int kind = 0; kind < 2; ++kind)
                        initial[at(false, kind, b, l, r, d)] =
                            r < bases.at(b) % 4 ? std::uint16_t(kp::hash(d + 333 * r + 777 * l +
                                                                         99 * b + 101 * kind))
                                                : 0;
    for (std::size_t i = 0; i < keys.host.size(); ++i) keys.host[i] = kp::hash(i + 171);
    for (std::size_t i = 0; i < gates.host.size(); ++i) gates.host[i] = kp::hash(i + 923);
    auto expected = initial;
    for (int b = 1; b < B; ++b) {
        const int F = bases.at(b), end = F + accepted.at(b), rem = end % 4;
        for (int l = 0; l < L; ++l)
            for (int r = 0; r < 3; ++r)
                for (int d = 0; d < D; ++d)
                    for (int kind = 0; kind < 2; ++kind) {
                        std::uint16_t word = 0;
                        const int p        = end - rem + r;
                        if (r < rem)
                            word =
                                p < F ? initial[at(false, kind, b, l, p - (F - F % 4), d)]
                                      : (kind ? gates.at(d, p - F, b, l) : keys.at(d, p - F, b, l));
                        expected[at(true, kind, b, l, r, d)] = word;
                    }
    }
    bases.at(0) = valid.at(0) = -1;
    keys.upload();
    gates.upload();
    bases.upload();
    valid.upload();
    src.upload();
    dst.upload();
    // Capture at a different Device acceptance; replay must use the currently published value.
    auto desired = accepted.host;
    std::fill(accepted.host.begin(), accepted.host.end(), 0);
    accepted.upload();
    ops::KpoolRawRecordReadView records{keys.view, gates.view, bases.view, valid.view};
    DecodeGraphDefinition graph;
    graph.capture(device.stream(), [&] {
        ops::kpool_tail_extract(before, records, accepted.view, after, device.execution_view());
    });
    DecodeGraphExecutable executable;
    executable.instantiate(graph);
    accepted.host = desired;
    accepted.upload();
    storage.copy_from_host(initial.data(), storage.size());
    executable.launch(device.stream());
    device.synchronize();
    std::vector<std::uint16_t> actual(initial.size());
    storage.copy_to_host(actual.data(), storage.size());
    if (actual != expected || keys.download() != keys.host || gates.download() != gates.host ||
        !keys.guards() || !gates.guards()) {
        std::cerr << "tail extraction violated exact copy/write set\n";
        return 1;
    }
    return 0;
}

int layout_contract() {
    const auto bf =
        ops::plan_kpool_mla_cache_layout({ops::KpoolMlaStorage::BFloat16, 512, 128, 11});
    const auto fp = ops::plan_kpool_mla_cache_layout(
        {ops::KpoolMlaStorage::Fp8E4m3fnGroup128F32, 512, 128, 11});
    if (bf.page.page_bytes != 11 * 68 * 1024 || fp.page.page_bytes != 11 * 37 * 1024) return 1;
    LayoutBuilder builder;
    const auto raw       = ops::plan_kpool_raw_records(builder, 128, 17, 8, 11);
    const auto selection = ops::plan_kpool_block_selection(builder, 7);
    HostBuffer storage(builder.finish());
    const auto view = raw.write_view(storage.span(), 5, 3, {});
    if (view.raw_keys.ne[2] != 3 || view.raw_keys.ne[3] != 1 ||
        view.raw_keys.nb[2] != 128 * 17 * 2 || view.raw_keys.nb[3] != 128 * 17 * 8 * 2 ||
        selection.bind(storage.span()).pool_ids.ne[1] != 7)
        return 1;
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    suite.add("kpool/layout", layout_contract, {{"format", "bf16"}});
    for (bool alias : {false, true})
        for (int dimension : {128, 257})
            suite.add(std::string("kpool/tail/") + (alias ? "inplace/" : "disjoint/") +
                          std::to_string(dimension),
                      [=] { return run(alias, dimension); },
                      {{"format", "bf16"}, {"execution", "graph"}});
    return suite.main(argc, argv);
}
