// T98 probe (validation tool, not a test): the engine's own two routed-expert paths on the same
// real experts and the same real activations.
//   CPU  ops::host::HostMoeExecutor (W4A8, AVX-512 VNNI), the code the offload backend runs
//   GPU  ops::moe_slots_nvfp4 (W4A4 with the layer's calibrated input divisors)
// Inputs (written by a small exporter of real experts and activations, see T98 in the development
// log):
//   planes.bin  u32 'T98P', i32 n, f32 gate_up_input_divisor, f32 down_input_divisor, then n x
//               { i32 id; f32 div_gate, div_up, div_down; gate|up codes [1280,2560] E2M1 (1 638 400
//               B);
//                 gate|up swizzled scales (204 800 B); down codes [2560,640] (819 200 B);
//                 down swizzled scales (102 400 B) }   (one layer; ids ascending)
//   cases.bin   i32 n, then n x { i32 ids[10]; f32 alpha[10]; u16 x[2560] (BF16) }  (T = 1 each;
//               ids < 0 = not routed here)
// Output: out.bin  n x { f32 cpu[2560]; f32 gpu[2560] }.
// Usage: t98_expert_probe planes.bin cases.bin out.bin [cpu threads, default 6] [emu]
//   emu: the CPU column is the executor's T98 W4A4 emulation (Options::emulate_w4a4) instead of
//   W4A8.
#include "ninfer/ops/moe_slots.h"
#include "ops/sparse_moe/host/host_moe_executor.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;

namespace {

constexpr int kH = 2560, kI = 640, kK = 10, kE = 512;
constexpr std::size_t kGuC = 1638400, kGuS = 204800, kDnC = 819200, kDnS = 102400;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

std::vector<std::uint8_t> read_file(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) throw std::runtime_error(std::string("cannot open ") + path);
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<std::uint8_t> b(static_cast<std::size_t>(n));
    if (std::fread(b.data(), 1, b.size(), f) != b.size()) throw std::runtime_error("short read");
    std::fclose(f);
    return b;
}

template <class T>
T take(const std::uint8_t*& p) {
    T v;
    std::memcpy(&v, p, sizeof v);
    p += sizeof v;
    return v;
}

} // namespace

int main(int argc, char** argv) try {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s planes.bin cases.bin out.bin [threads]\n", argv[0]);
        return 2;
    }
    const int threads     = argc > 4 ? std::atoi(argv[4]) : 6;
    const auto planes     = read_file(argv[1]);
    const auto cases      = read_file(argv[2]);
    const std::uint8_t* p = planes.data();
    if (take<std::uint32_t>(p) != 0x50383954U) throw std::runtime_error("planes: bad magic");
    const int n       = take<std::int32_t>(p);
    const float gu_in = take<float>(p);
    const float dn_in = take<float>(p);
    std::vector<int> ids(n);
    std::vector<int> local(kE, -1);
    // Host bank of the n experts (local index = rank of the id, ids ascending so the executor's
    // ascending-id sum order is the engine's), device pool in the planar layout.
    std::vector<std::uint8_t> gu_codes(kGuC * n), gu_scales(kGuS * n), dn_codes(kDnC * n),
        dn_scales(kDnS * n);
    std::vector<float> host_div_gu(2 * n), host_div_dn(n), dev_div_gu(2 * kE, 1.0F),
        dev_div_dn(kE, 1.0F);
    for (int i = 0; i < n; ++i) {
        ids[i] = take<std::int32_t>(p);
        if (ids[i] < 0 || ids[i] >= kE || (i > 0 && ids[i] <= ids[i - 1]))
            throw std::runtime_error("planes: ids");
        local[ids[i]]  = i;
        const float dg = take<float>(p), du = take<float>(p), dd = take<float>(p);
        host_div_gu[2 * i] = dg, host_div_gu[2 * i + 1] = du, host_div_dn[i] = dd;
        dev_div_gu[2 * ids[i]] = dg, dev_div_gu[2 * ids[i] + 1] = du, dev_div_dn[ids[i]] = dd;
        std::memcpy(gu_codes.data() + kGuC * i, p, kGuC), p += kGuC;
        std::memcpy(gu_scales.data() + kGuS * i, p, kGuS), p += kGuS;
        std::memcpy(dn_codes.data() + kDnC * i, p, kDnC), p += kDnC;
        std::memcpy(dn_scales.data() + kDnS * i, p, kDnS), p += kDnS;
    }
    if (p != planes.data() + planes.size()) throw std::runtime_error("planes: trailing bytes");

    const std::uint8_t* c = cases.data();
    const int m           = take<std::int32_t>(c);

    struct Case {
        std::int32_t ids[kK];
        float alpha[kK];
        std::uint16_t x[kH];
    };

    std::vector<Case> cs(m);
    for (int i = 0; i < m; ++i) std::memcpy(&cs[i], c, sizeof(Case)), c += sizeof(Case);

    // ---- CPU: the engine's executor on a bank of these n experts
    ops::host::HostExpertLayer layer;
    layer.gate_up = ops::host::Nvfp4Plane{.codes            = gu_codes.data(),
                                          .scales           = gu_scales.data(),
                                          .divisors         = host_div_gu.data(),
                                          .rows             = 2LL * kI * n,
                                          .k                = kH,
                                          .rows_per_divisor = kI};
    layer.down    = ops::host::Nvfp4Plane{.codes            = dn_codes.data(),
                                          .scales           = dn_scales.data(),
                                          .divisors         = host_div_dn.data(),
                                          .rows             = 1LL * kH * n,
                                          .k                = kI,
                                          .rows_per_divisor = kH};
    layer.experts = n, layer.hidden = kH, layer.intermediate = kI;
    layer.gate_up_input_divisor = gu_in, layer.down_input_divisor = dn_in;
    ops::host::HostMoeExecutor::Options options{};
    options.emulate_w4a4        = argc > 5 && std::string(argv[5]) == "emu";
    options.a4                  = argc > 5 && std::string(argv[5]) == "a4";
    const std::vector<int> cpus = ops::host::spread_cpus(threads);
    std::vector<float> cpu_out(static_cast<std::size_t>(m) * kH),
        gpu_out(static_cast<std::size_t>(m) * kH);
    {
        ops::host::HostCpuPool pool(cpus);
        ops::host::HostMoeExecutor ex(pool, kH, kI, ops::host::kHostMaxTokens, kK, options);
        for (int i = 0; i < m; ++i) {
            std::int32_t lid[kK];
            for (int k = 0; k < kK; ++k) {
                const int e = cs[i].ids[k];
                lid[k]      = e >= 0 ? local.at(e) : -1;
                if (e >= 0 && lid[k] < 0)
                    throw std::runtime_error("case routes an expert not in planes");
            }
            const ops::host::HostMoeJob job{
                1,   kK,          cs[i].x,
                lid, cs[i].alpha, cpu_out.data() + static_cast<std::size_t>(i) * kH};
            (void)ex.run(layer, job);
        }
    }

    // ---- GPU: moe_slots_nvfp4 over a pool holding the n experts (slot = local index)
    void *d_guc, *d_gus, *d_dnc, *d_dns, *d_table, *d_dgu, *d_ddn, *d_x, *d_ids, *d_w, *d_out;
    check(cudaMalloc(&d_guc, gu_codes.size()), "malloc");
    check(cudaMalloc(&d_gus, gu_scales.size()), "malloc");
    check(cudaMalloc(&d_dnc, dn_codes.size()), "malloc");
    check(cudaMalloc(&d_dns, dn_scales.size()), "malloc");
    check(cudaMalloc(&d_table, 4 * kE), "malloc");
    check(cudaMalloc(&d_dgu, 4 * 2 * kE), "malloc");
    check(cudaMalloc(&d_ddn, 4 * kE), "malloc");
    check(cudaMalloc(&d_x, 2 * kH), "malloc");
    check(cudaMalloc(&d_ids, 4 * kK), "malloc");
    check(cudaMalloc(&d_w, 4 * kK), "malloc");
    check(cudaMalloc(&d_out, 4 * kH), "malloc");
    check(cudaMemcpy(d_guc, gu_codes.data(), gu_codes.size(), cudaMemcpyHostToDevice), "copy");
    check(cudaMemcpy(d_gus, gu_scales.data(), gu_scales.size(), cudaMemcpyHostToDevice), "copy");
    check(cudaMemcpy(d_dnc, dn_codes.data(), dn_codes.size(), cudaMemcpyHostToDevice), "copy");
    check(cudaMemcpy(d_dns, dn_scales.data(), dn_scales.size(), cudaMemcpyHostToDevice), "copy");
    check(cudaMemcpy(d_table, local.data(), 4 * kE, cudaMemcpyHostToDevice), "copy");
    check(cudaMemcpy(d_dgu, dev_div_gu.data(), 4 * 2 * kE, cudaMemcpyHostToDevice), "copy");
    check(cudaMemcpy(d_ddn, dev_div_dn.data(), 4 * kE, cudaMemcpyHostToDevice), "copy");
    ops::MoeSlotPool pool;
    pool.gate_up_codes  = {d_guc, static_cast<std::int64_t>(kGuC)};
    pool.gate_up_scales = {d_gus, static_cast<std::int64_t>(kGuS)};
    pool.down_codes     = {d_dnc, static_cast<std::int64_t>(kDnC)};
    pool.down_scales    = {d_dns, static_cast<std::int64_t>(kDnS)};
    pool.slots          = n;
    ops::MoeSlotsLayer view;
    view.slot_table            = Tensor(d_table, DType::I32, {kE});
    view.gate_up_divisors      = Tensor(d_dgu, DType::FP32, {2 * kE});
    view.down_divisors         = Tensor(d_ddn, DType::FP32, {kE});
    view.gate_up_input_divisor = gu_in;
    view.down_input_divisor    = dn_in;
    cudaStream_t stream        = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    WorkspaceArena work(ops::moe_slots_nvfp4_workspace_bytes(1));
    for (int i = 0; i < m; ++i) {
        check(cudaMemcpy(d_x, cs[i].x, 2 * kH, cudaMemcpyHostToDevice), "copy");
        check(cudaMemcpy(d_ids, cs[i].ids, 4 * kK, cudaMemcpyHostToDevice), "copy");
        check(cudaMemcpy(d_w, cs[i].alpha, 4 * kK, cudaMemcpyHostToDevice), "copy");
        // Pageable uploads may still be in flight; the non-blocking stream does not wait for them.
        check(cudaDeviceSynchronize(), "sync");
        Tensor tx(d_x, DType::BF16, {kH, 1}), ti(d_ids, DType::I32, {kK, 1}),
            tw(d_w, DType::FP32, {kK, 1}), to(d_out, DType::FP32, {kH, 1});
        ops::moe_slots_nvfp4(tx, ti, tw, view, pool, ops::MoeSlotsOutput::Store, to, work, stream);
        check(cudaGetLastError(), "moe_slots_nvfp4");
        check(cudaStreamSynchronize(stream), "sync");
        check(cudaMemcpy(gpu_out.data() + static_cast<std::size_t>(i) * kH, d_out, 4 * kH,
                         cudaMemcpyDeviceToHost),
              "copy");
    }
    std::FILE* f = std::fopen(argv[3], "wb");
    if (!f) throw std::runtime_error("cannot write output");
    for (int i = 0; i < m; ++i) {
        std::fwrite(cpu_out.data() + static_cast<std::size_t>(i) * kH, 4, kH, f);
        std::fwrite(gpu_out.data() + static_cast<std::size_t>(i) * kH, 4, kH, f);
    }
    std::fclose(f);
    std::printf("t98_expert_probe: %d experts, %d cases, gate_up_input_divisor %.6g "
                "down_input_divisor %.6g\n",
                n, m, gu_in, dn_in);
    return 0;
} catch (const std::exception& e) {

    std::fprintf(stderr, "t98_expert_probe: %s\n", e.what());
    return 1;
}
