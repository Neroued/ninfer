#include "workloads.h"
#include "common/gpu_activity.cuh"
#include "ninfer/ops/video_delta_attention.h"
#include <cuda_bf16.h>
#include <sstream>

using namespace ninfer;
using namespace ninfer::bench;

namespace {
using vda::Options;
using vda::Workload;

__device__ float random(std::uint64_t i, unsigned seed) {
    unsigned x = static_cast<unsigned>(i) ^ static_cast<unsigned>(i >> 32) ^ seed;
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return float(x & 0xffffff) * (2.f / 16777216.f) - 1.f;
}

__global__ void vectors(__nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v, __nv_bfloat16* gate,
                        std::int64_t rows) {
    const auto r = std::int64_t(blockIdx.x) * 8 + threadIdx.x / 32;
    if (r >= rows) return;
    const int lane = threadIdx.x % 32;
    float a[4], b[4], s = 0, t = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const auto i = r * 128 + lane + j * 32;
        a[j]         = random(i, 17);
        b[j]         = random(i, 73);
        s += a[j] * a[j];
        t += b[j] * b[j];
    }
#pragma unroll
    for (int x = 16; x; x /= 2) {
        s += __shfl_xor_sync(0xffffffff, s, x);
        t += __shfl_xor_sync(0xffffffff, t, x);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const auto i = r * 128 + lane + j * 32;
        if (q) q[i] = __float2bfloat16_rn(a[j] * rsqrtf(s + 1e-12f));
        k[i] = __float2bfloat16_rn(b[j] * rsqrtf(t + 1e-12f));
        v[i] = __float2bfloat16_rn(random(i, 213) * 0.5f);
        if (gate) gate[i] = __float2bfloat16_rn(random(i, 831) * 2.f);
    }
}

__global__ void controls(float* x, std::int64_t n, float low, float span) {
    for (auto i = std::int64_t(blockIdx.x) * 256 + threadIdx.x; i < n;
         i += std::int64_t(gridDim.x) * 256)
        x[i] = low + span * (random(i, 339) * 0.5f + 0.5f);
}

struct Fixture {
    int heads, frames, tokens_per_frame, tokens, batch, text_tokens;
    Options options;
    DeviceBuffer q, k, v, gate, alpha, beta, initial, output, weight;
    DeviceBuffer text_k, text_v, text_beta, scratch;
    std::unique_ptr<MemoryArena> workspace;
    std::size_t capacity;

    Fixture(const Options& configuration, const Workload& workload, int local_heads)
        : heads(local_heads), frames(workload.frames), tokens_per_frame(workload.tokens_per_frame),
          tokens(0), batch(configuration.batch), text_tokens(workload.text_tokens),
          options(configuration) {
        // Validate the extent with the public owner before multiplying dimensions or allocating.
        capacity = ops::video_delta_state_init_workspace_capacity_bytes(heads, text_tokens,
                                                                        text_tokens, batch);
        const bool only_state = options.op == "state";
        if (!only_state)
            capacity = std::max(
                capacity, ops::video_delta_attention_workspace_capacity_bytes(
                              heads, tokens_per_frame, frames, frames, batch, options.window));
        if (std::int64_t(frames) * tokens_per_frame > INT32_MAX)
            throw std::invalid_argument("VDN benchmark: token count overflow");
        tokens                      = frames * tokens_per_frame;
        const auto device           = device_context().device();
        const auto stream           = device_context().execution_view().stream();
        const std::size_t rows      = std::size_t(batch) * tokens * heads;
        const std::size_t text_rows = std::size_t(batch) * text_tokens * heads;
        auto allocate = [&](DeviceBuffer& buffer, std::size_t count, std::size_t element_bytes) {
            if (count > std::numeric_limits<std::size_t>::max() / element_bytes)
                throw std::overflow_error("VDN benchmark: allocation size overflow");
            buffer = DeviceBuffer(device, std::max<std::size_t>(count * element_bytes, 256));
        };
        for (auto* buffer : {&q, &k, &v, &gate, &output})
            allocate(*buffer, only_state ? 0 : rows, 256);
        allocate(alpha, only_state ? 0 : std::size_t(batch) * frames * heads, 128 * 4);
        allocate(beta, only_state ? 0 : rows, 4);
        allocate(initial, std::size_t(batch) * heads, 128 * 128 * 4);
        allocate(weight, 128, 2);
        // Tensor construction uses a one-token placeholder before setting L=0.
        const auto text_capacity_rows = std::max(text_rows, std::size_t(batch) * heads);
        allocate(text_k, text_capacity_rows, 256);
        allocate(text_v, text_capacity_rows, 256);
        allocate(text_beta, text_capacity_rows, 4);
        allocate(scratch, capacity, 1);
        workspace = std::make_unique<MemoryArena>(scratch.span());
        if (!only_state) {
            vectors<<<(rows + 7) / 8, 256, 0, stream>>>(
                static_cast<__nv_bfloat16*>(q.data()), static_cast<__nv_bfloat16*>(k.data()),
                static_cast<__nv_bfloat16*>(v.data()), static_cast<__nv_bfloat16*>(gate.data()),
                rows);
            controls<<<1024, 256, 0, stream>>>(static_cast<float*>(alpha.data()),
                                               std::int64_t(batch) * frames * heads * 128, .94f,
                                               .059f);
            controls<<<1024, 256, 0, stream>>>(static_cast<float*>(beta.data()), rows, .1f, .8f);
        }
        if (text_rows) {
            vectors<<<(text_rows + 7) / 8, 256, 0, stream>>>(
                nullptr, static_cast<__nv_bfloat16*>(text_k.data()),
                static_cast<__nv_bfloat16*>(text_v.data()), nullptr, text_rows);
            controls<<<128, 256, 0, stream>>>(static_cast<float*>(text_beta.data()), text_rows, .1f,
                                              .8f);
        }
        CUDA_CHECK(cudaGetLastError());
        const std::vector<std::uint16_t> ones(128, 0x3f80);
        weight.copy_from_host(ones.data(), 256);
        initialize_state(device_context().execution_view());
        device_context().synchronize();
    }

    Tensor vector_view(DeviceBuffer& buffer, int count) {
        Tensor result(buffer.span(), DType::BF16, {128, heads, std::max(count, 1), batch});
        result.ne[2] = count;
        return result;
    }

    Tensor state_view() { return Tensor(initial.span(), DType::FP32, {128, 128, heads, batch}); }

    void initialize_state(CudaExecutionView execution) {
        auto state = state_view();
        Tensor weights(text_beta.span(), DType::FP32, {heads, std::max(text_tokens, 1), batch});
        weights.ne[1] = text_tokens;
        ops::video_delta_state_init(vector_view(text_k, text_tokens),
                                    vector_view(text_v, text_tokens), weights, .5f, *workspace,
                                    state, execution);
    }

    void call(CudaExecutionView execution) {
        if (options.op == "state" || options.op == "chain") initialize_state(execution);
        if (options.op == "state") return;
        auto result = vector_view(output, tokens);
        const Tensor retention(alpha.span(), DType::FP32, {128, heads, frames, batch});
        const Tensor weights(beta.span(), DType::FP32, {heads, tokens, batch});
        if (options.op == "raw") {
            ops::video_delta_attention(vector_view(q, tokens), vector_view(k, tokens),
                                       vector_view(v, tokens), retention, weights, state_view(),
                                       tokens_per_frame, options.window, *workspace, result,
                                       execution);
        } else {
            ops::video_delta_attention_sigmoid_gated(
                vector_view(q, tokens), vector_view(k, tokens), vector_view(v, tokens), retention,
                weights, state_view(), Tensor(weight.span(), DType::BF16, {128}),
                vector_view(gate, tokens), 1e-6f, tokens_per_frame, options.window, *workspace,
                result, execution);
        }
    }
};

void measure_workload(const Options& options, const Workload& workload, int heads,
                      measurement::Csv& csv, L2FlushBuffer& flush) {
    Fixture fixture(options, workload, heads);
    const auto cost = vda::useful_cost(options, workload, heads);
    measurement::Row row;
    row.op       = "vdn-" + options.op;
    row.format   = "bf16";
    row.policy   = "fp32-state";
    row.tokens   = options.op == "state" ? workload.text_tokens : fixture.tokens;
    row.batch    = options.batch;
    row.qk_heads = row.value_heads = heads;
    row.workspace_bytes            = fixture.capacity;
    for (const auto& execution :
         measurement::selections(options.timing.execution, "graph", "eager")) {
        for (const auto& cache : measurement::selections(options.timing.cache, "cold", "warm")) {
            const auto result = measurement::measure(
                [&](auto view) { fixture.call(view); }, [](auto view) { condition_gpu(view, 200); },
                flush, device_context().execution_view(), options.timing, execution, cache,
                options.op == "state" && workload.text_tokens == 0);
            row.workspace_peak = fixture.workspace->peak_used();
            std::ostringstream extra;
            extra << workload.name << ',' << workload.video_frames << ',' << workload.width << ','
                  << workload.height << ',' << (56 % heads == 0 ? 56 / heads : 0) << ','
                  << workload.frames << ',' << workload.tokens_per_frame << ','
                  << options.window.chunk_frames << ',' << options.window.radius << ','
                  << options.window.frame_offset << ',' << workload.text_tokens << ',' << std::fixed
                  << cost.io_bytes << ',' << cost.io_bytes / (kRoofline.memory_gbs * 1000) << ','
                  << cost.contraction_flops << ','
                  << cost.contraction_flops / (result.gpu.median_us * 1e6);
            csv.write(row, result, execution, cache, extra.str());
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options   = vda::parse_options(argc, argv);
        const auto workloads = vda::select_workloads(options);
        measurement::hardware();
        L2FlushBuffer flush(options.timing.flush_bytes);
        measurement::Csv csv(options.timing.csv_out,
                             "case,video_frames,width,height,head_partition,frames,tokens_per_"
                             "frame,chunk,radius,offset,"
                             "text_tokens,io_bytes_lower_bound,io_floor_us,contraction_flops_excl_"
                             "inverse,useful_tflops");
        for (const auto& workload : workloads)
            for (const int heads : options.heads)
                measure_workload(options, workload, heads, csv, flush);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
