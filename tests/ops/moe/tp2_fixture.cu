#include "ops/moe/tp2_fixture.h"

#include <cuda_bf16.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace ninfer::test::moe_tp2 {
namespace {
__host__ __device__ unsigned hash(unsigned expert, unsigned row, unsigned column, unsigned seed) {
    unsigned x = expert * 0x9e3779b9U ^ row * 0x85ebca6bU ^ column * 0xc2b2ae35U ^ seed;
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    return x * 0x846ca68bU;
}

__host__ __device__ std::uint16_t bf16(float value) {
    union {
        float f;
        unsigned u;
    } bits{value};

    return (bits.u + 0x7fffU + ((bits.u >> 16) & 1U)) >> 16;
}

__host__ __device__ int original_row(int row, int intermediate, int rank, bool gate) {
    return !gate
               ? row
               : (row < intermediate ? rank * intermediate + row
                                     : 2 * intermediate + rank * intermediate + row - intermediate);
}

__host__ __device__ std::uint8_t source_code(int expert, int row, int column, unsigned seed,
                                             bool fp8) {
    auto code = std::uint8_t(hash(expert, row, column, seed));
    if (fp8 && (code & 127) == 127) --code;
    return fp8 ? code : code & 15;
}

__host__ __device__ std::uint16_t source_scale(int expert, int row, int column, unsigned seed,
                                               bool fp8) {
    const auto value = hash(expert, row / (fp8 ? 128 : 1), column / (fp8 ? 128 : 16), seed + 17);
    return fp8 ? bf16(float(1 + value % 13) * (1.0F / 262144)) : std::uint16_t(0x18 + value % 8);
}

__global__ void fill_codes(std::uint8_t* codes, int n, int k, int experts, int intermediate,
                           int rank, bool gate, bool fp8, unsigned seed) {
    const std::int64_t count = std::int64_t(experts) * n * (fp8 ? k : k / 2);
    for (auto i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += std::int64_t(gridDim.x) * blockDim.x) {
        const int column = i % (fp8 ? k : k / 2) * (fp8 ? 1 : 2);
        const int row = i / (fp8 ? k : k / 2) % n, e = i / (std::int64_t(n) * (fp8 ? k : k / 2));
        const int sr   = original_row(row, intermediate, rank, gate);
        const int sc   = column + (gate ? 0 : rank * intermediate);
        const auto low = source_code(e, sr, sc, seed, fp8);
        codes[i]       = fp8 ? low : low | (source_code(e, sr, sc + 1, seed, false) << 4);
    }
}

__global__ void fill_scales(std::uint8_t* scales, int n, int k, int experts, int intermediate,
                            int rank, bool gate, int block, unsigned seed) {
    const bool fp8 = block != 16;
    const std::int64_t per_expert =
        fp8 ? std::int64_t(n / block) * (k / block) : std::int64_t(n) * (k / 16);
    for (auto i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < experts * per_expert;
         i += std::int64_t(gridDim.x) * blockDim.x) {
        const int e     = i / per_expert;
        const auto cell = i % per_expert;
        const int row = cell / (k / block) * (fp8 ? block : 1), column = cell % (k / block) * block;
        const int sr     = original_row(row, intermediate, rank, gate);
        const int sc     = column + (gate ? 0 : rank * intermediate);
        const auto scale = source_scale(e, sr, sc, seed, fp8);
        if (fp8)
            reinterpret_cast<std::uint16_t*>(scales)[i] = scale;
        else {
            const int group   = column / 16;
            const auto offset = (row / 128 * (k / 64) + group / 4) * 512 + row % 32 * 16 +
                                (row % 128) / 32 * 4 + group % 4;
            scales[e * per_expert + offset] = scale;
        }
    }
}

__global__ void fill_shared(std::uint8_t* codes, std::uint16_t* scales, int n, int k, bool q8,
                            unsigned seed, int intermediate, int rank, bool gate) {
    for (auto i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < std::int64_t(n) * k;
         i += std::int64_t(gridDim.x) * blockDim.x) {
        const int row = i / k, column = i % k;
        const int sr    = original_row(row, intermediate, rank, gate);
        const int sc    = column + (gate ? 0 : rank * intermediate);
        const int value = int(hash(0, sr, sc, seed) % 32) - 16;
        if (q8)
            codes[i] = std::uint8_t(value);
        else
            reinterpret_cast<std::uint16_t*>(codes)[i] = bf16(value * (1.0F / 1024));
        if (q8 && column % 32 == 0)
            scales[i / 32] = 0x1400 + (hash(0, sr, sc / 32, seed + 1) % 3) * 0x400;
    }
}

struct Matrix {
    int n, k, count, block;
    WeightGeometry geometry;
    DeviceBuffer storage;
    std::vector<float> divisors;
    WeightParent parent;
    WeightBankView bank;

    Matrix(CudaExecutionView execution, const Shape& s, int rank, bool gate)
        : n(gate ? 2 * s.intermediate : s.hidden), k(gate ? s.hidden : s.intermediate),
          count(s.experts), block(s.format == QType::NVFP4                        ? 16
                                  : s.format == QType::FP8_E4M3FN_BLOCK64X64_BF16 ? 64
                                                                                  : 128),
          geometry(weight_geometry(s.format,
                                   block == 16   ? QuantLayout::BatchedBlockScaleK16M128x4
                                   : block == 64 ? QuantLayout::BlockScaleM64K64
                                                 : QuantLayout::BlockScaleM128K128,
                                   std::array<std::uint64_t, 3>{
                                       std::uint64_t(count), std::uint64_t(n), std::uint64_t(k)})),
          storage(execution.device(), geometry.bytes), divisors(geometry.divisor_count),
          parent{geometry, execution.device(), static_cast<const std::byte*>(storage.data()),
                 divisors} {
        const auto seed = gate ? 101U : 211U;
        fill_codes<<<768, 256, 0, execution.stream()>>>(static_cast<std::uint8_t*>(storage.data()),
                                                        n, k, count, s.intermediate, rank, gate,
                                                        block != 16, seed);
        CUDA_CHECK(cudaGetLastError());
        fill_scales<<<384, 256, 0, execution.stream()>>>(
            static_cast<std::uint8_t*>(storage.data()) + geometry.scale_offset, n, k, count,
            s.intermediate, rank, gate, block, seed);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(execution.stream()));
        if (!divisors.empty()) {
            for (int e = 0; e < count; ++e) divisors[e] = (gate ? 6.F : 3.F) + (e % 8) * .5F;
            storage.copy_from_host(divisors.data(), divisors.size() * 4, geometry.divisor_offset);
        }
        bank = bind_weight_bank(
            parent, describe_weight_bank(geometry, {0, unsigned(count), unsigned(n), unsigned(n)}));
    }

    Fixture::MatrixSnapshot snapshot(int e) const {
        Fixture::MatrixSnapshot out{n, k, block, geometry.format,
                                    divisors.empty() ? 1 : divisors[e]};
        const auto code  = bank.plane(bank.layout.codes, e),
                   scale = bank.plane(bank.layout.scales, e);
        out.codes.resize(code.bytes);
        out.scales.resize(scale.bytes);
        CUDA_CHECK(cudaMemcpy(out.codes.data(), code.data, code.bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out.scales.data(), scale.data, scale.bytes, cudaMemcpyDeviceToHost));
        return out;
    }
};

struct Shared {
    int n, k;
    bool q8;
    WeightGeometry geometry;
    DeviceBuffer storage;
    WeightParent parent;
    Weight weight;

    Shared(CudaExecutionView execution, const Shape& s, int rank, bool gate)
        : n(gate ? 2 * s.intermediate : s.hidden), k(gate ? s.hidden : s.intermediate),
          q8(s.clamped), geometry(weight_geometry(
                             q8 ? QType::Q8_G32_FP16 : QType::BF16,
                             q8 ? QuantLayout::RowSplit : QuantLayout::Contiguous,
                             std::array<std::uint64_t, 2>{std::uint64_t(n), std::uint64_t(k)})),
          storage(execution.device(), geometry.bytes),
          parent{geometry, execution.device(), static_cast<const std::byte*>(storage.data())} {
        fill_shared<<<128, 256, 0, execution.stream()>>>(
            static_cast<std::uint8_t*>(storage.data()),
            q8 ? reinterpret_cast<std::uint16_t*>(static_cast<std::byte*>(storage.data()) +
                                                  geometry.scale_offset)
               : nullptr,
            n, k, q8, gate ? 311 : 317, s.intermediate, rank, gate);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(execution.stream()));
        weight = native_weight(
            {{std::uint64_t(n), std::uint64_t(k)}, {{&parent, 0, geometry.elements}}});
    }

    Fixture::MatrixSnapshot snapshot() const {
        Fixture::MatrixSnapshot out{n, k, q8 ? 32 : 1, geometry.format, 1};
        out.codes.resize(geometry.code_bytes);
        out.scales.resize(geometry.scale_bytes);
        storage.copy_to_host(out.codes.data(), out.codes.size());
        if (!out.scales.empty())
            storage.copy_to_host(out.scales.data(), out.scales.size(), geometry.scale_offset);
        return out;
    }
};
} // namespace

struct Fixture::Impl {
    CudaExecutionView execution;
    Matrix gate_up, down;
    Shared shared_gate_up, shared_down;
    DeviceBuffer input, control, bias, ids, coefficients, shared_coefficients;
    std::vector<std::uint16_t> input_host, control_host;
    std::vector<float> bias_host;
    std::vector<int> expected;
    std::vector<std::vector<std::uint8_t>> preserved;

    Impl(CudaExecutionView cuda, Shape s, int rank, int tokens)
        : execution(cuda), gate_up(cuda, s, rank, true), down(cuda, s, rank, false),
          shared_gate_up(cuda, s, rank, true), shared_down(cuda, s, rank, false),
          input(cuda.device(), std::size_t(s.hidden) * tokens * 2),
          control(cuda.device(), std::size_t(s.experts + !s.clamped) * s.hidden * 2),
          bias(cuda.device(), s.clamped ? s.experts * 4 : 0),
          ids(cuda.device(), std::size_t(s.top_k) * tokens * 4),
          coefficients(cuda.device(), std::size_t(s.top_k) * tokens * 4),
          shared_coefficients(cuda.device(), tokens * 4), control_host(control.size() / 2),
          bias_host(s.clamped ? s.experts : 0) {
        for (int e = 0; e < s.experts; ++e) control_host[std::size_t(e) * s.hidden + e] = 0x3f80;
        if (!s.clamped) control_host[std::size_t(s.experts) * s.hidden + s.experts] = 0x3f80;
        for (int e = 0; e < int(bias_host.size()); ++e) bias_host[e] = (e % 7 - 3) * .0001F;
        control.copy_from_host(control_host.data(), control.size());
        if (bias.size()) bias.copy_from_host(bias_host.data(), bias.size());
    }
};

Fixture::Fixture(CudaExecutionView cuda, Shape s, int side, int tokens)
    : shape(s), rank(side), max_tokens(tokens),
      impl_(std::make_unique<Impl>(cuda, s, side, tokens)) {
    if (side < 0 || side > 1 || tokens <= 0) throw std::invalid_argument("invalid TP2 fixture");
}

Fixture::~Fixture() = default;

void Fixture::input(int tokens, std::string_view distribution, int phase, bool stress) {
    if (tokens <= 0 || tokens > max_tokens) throw std::invalid_argument("fixture token extent");
    auto& f      = *impl_;
    const auto s = shape;
    f.input_host.assign(std::size_t(s.hidden) * tokens, 0);
    f.expected.resize(std::size_t(s.top_k) * tokens);
    for (int t = 0; t < tokens; ++t) {
        for (int c = s.experts; c < s.hidden; ++c) {
            const float x = (int(hash(t + phase * 4099, c, 0, 719) % 1024) - 512) * (1.F / 2048);
            f.input_host[std::size_t(t) * s.hidden + c] = bf16(x * (stress ? 64.F : 1.F));
        }
        f.input_host[std::size_t(t) * s.hidden + s.experts] = bf16((t % 5 - 2) * .25F);
        for (int j = 0; j < s.top_k; ++j) {
            const int position = phase && (t & 1)
                                     ? (distribution == "tail" ? j / (s.top_k / 2) * (s.top_k / 2) +
                                                                     (j + phase) % (s.top_k / 2)
                                                               : (j + phase) % s.top_k)
                                     : j;
            int ordinal;
            if (distribution == "reuse")
                ordinal = position;
            else if (distribution == "pool")
                ordinal = (t * s.top_k + position) % (2 * s.top_k);
            else if (distribution == "tail")
                ordinal = position < s.top_k / 2
                              ? position
                              : s.top_k / 2 + (t * (s.top_k / 2) + position - s.top_k / 2) %
                                                  (s.experts - s.top_k / 2);
            else if (distribution == "spread")
                ordinal = (t * s.top_k + position) % s.experts;
            else
                throw std::invalid_argument("unknown TP2 distribution");
            const int e = (107 * (ordinal + phase * 37) + 19) % s.experts;
            f.expected[std::size_t(t) * s.top_k + j]    = e;
            f.input_host[std::size_t(t) * s.hidden + e] = bf16(1.F + (s.top_k - j) * .0625F);
        }
    }
    f.input.copy_from_host(f.input_host.data(), f.input_host.size() * 2);
}

Tensor Fixture::x(int tokens) const {
    return Tensor(impl_->input.span(), DType::BF16, {shape.hidden, tokens});
}

ops::ExpertRoutes Fixture::routes(int tokens) const {
    return {Tensor(impl_->ids.span(), DType::I32, {shape.top_k, tokens}),
            Tensor(impl_->coefficients.span(), DType::FP32, {shape.top_k, tokens})};
}

ops::RoutedSwiGLUProfile Fixture::routed_profile() const {
    const bool fp8 = shape.format != QType::NVFP4;
    auto p =
        ops::routed_swiglu_profile(impl_->gate_up.bank, impl_->down.bank, shape.top_k,
                                   fp8 ? ops::LinearPolicy::AllowA8 : ops::LinearPolicy::AllowA4,
                                   fp8 ? ops::LinearPolicy::AllowA8 : ops::LinearPolicy::AllowA4);
    if (shape.clamped) p.ffn.clamp_limit = 10;
    return p;
}

ops::CudaSwiGLUExperts Fixture::experts() const {
    return {device_expert_bank(impl_->gate_up.bank, impl_->down.bank),
            shape.format == QType::NVFP4 ? ops::ExpertActivationDivisors{3.5F, 5.25F}
                                         : ops::ExpertActivationDivisors{}};
}

ops::SigmoidMoeProfile Fixture::sigmoid_profile() const {
    auto shared        = ops::swiglu_profile(sigmoid_weights().shared);
    shared.clamp_limit = 10;
    return {{{shape.hidden, shape.experts, shape.top_k}, 2.5F, 1e-20F}, routed_profile(), shared};
}

ops::MoeProfile Fixture::softmax_profile() const {
    return {routed_profile(), ops::swiglu_profile(softmax_weights().shared)};
}

ops::SigmoidMoeWeights Fixture::sigmoid_weights() const {
    WeightParent parent{weight_geometry(QType::BF16, QuantLayout::Contiguous,
                                        std::array<std::uint64_t, 2>{std::uint64_t(shape.experts),
                                                                     std::uint64_t(shape.hidden)}),
                        impl_->execution.device(),
                        static_cast<const std::byte*>(impl_->control.data())};
    return {native_weight({parent.geometry.shape, {{&parent, 0, parent.geometry.elements}}}),
            Tensor(impl_->bias.span(), DType::FP32, {shape.experts}),
            experts(),
            {{impl_->shared_gate_up.weight, ops::LinearPolicy::A16Only},
             {impl_->shared_down.weight, ops::LinearPolicy::A16Only}}};
}

ops::MoeWeights Fixture::softmax_weights() const {
    WeightParent parent{
        weight_geometry(QType::BF16, QuantLayout::Contiguous,
                        std::array<std::uint64_t, 2>{std::uint64_t(shape.experts + 1),
                                                     std::uint64_t(shape.hidden)}),
        impl_->execution.device(), static_cast<const std::byte*>(impl_->control.data())};
    return {native_weight({parent.geometry.shape, {{&parent, 0, parent.geometry.elements}}}),
            experts(),
            {{impl_->shared_gate_up.weight, ops::LinearPolicy::A16Only},
             {impl_->shared_down.weight, ops::LinearPolicy::A16Only}}};
}

std::size_t Fixture::encoded_expert_bytes() const {
    return (impl_->gate_up.geometry.bytes + impl_->down.geometry.bytes) / shape.experts;
}

std::size_t Fixture::router_bytes() const { return impl_->control.size(); }

std::size_t Fixture::shared_bytes() const {
    return impl_->shared_gate_up.storage.size() + impl_->shared_down.storage.size();
}

const std::vector<int>& Fixture::expected_ids() const { return impl_->expected; }

const std::vector<std::uint16_t>& Fixture::input_words() const { return impl_->input_host; }

Fixture::MatrixSnapshot Fixture::snapshot(bool gate, int expert) const {
    return (gate ? impl_->gate_up : impl_->down).snapshot(expert);
}

Fixture::MatrixSnapshot Fixture::shared_snapshot(bool gate) const {
    return (gate ? impl_->shared_gate_up : impl_->shared_down).snapshot();
}

std::vector<std::uint16_t> Fixture::control_words() const { return impl_->control_host; }

std::vector<float> Fixture::bias_words() const { return impl_->bias_host; }

void Fixture::publish_routes(int tokens, bool signed_coefficients) {
    std::vector<float> coefficients(std::size_t(shape.top_k) * tokens);
    for (int t = 0; t < tokens; ++t)
        for (int j = 0; j < shape.top_k; ++j)
            coefficients[std::size_t(t) * shape.top_k + j] =
                (signed_coefficients && j % 3 == 0 ? -.375F : .75F) * (1 + j * .125F);
    impl_->ids.copy_from_host(impl_->expected.data(), coefficients.size() * 4);
    impl_->coefficients.copy_from_host(coefficients.data(), coefficients.size() * 4);
}

std::vector<int> Fixture::check_routes(int tokens, MemoryArena& workspace, CudaExecutionView cuda) {
    auto selected = routes(tokens);
    if (shape.clamped)
        ops::topk_sigmoid_route(x(tokens), sigmoid_profile().routing, sigmoid_weights().control,
                                sigmoid_weights().correction_bias, selected, workspace, cuda);
    else {
        Tensor shared(impl_->shared_coefficients.span(), DType::FP32, {tokens});
        ops::topk_softmax_route(x(tokens), {shape.hidden, shape.experts, shape.top_k},
                                softmax_weights().control, selected, shared, workspace, cuda);
    }
    CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
    std::vector<int> got(std::size_t(shape.top_k) * tokens), histogram(shape.experts);
    impl_->ids.copy_to_host(got.data(), got.size() * 4);
    if (got != impl_->expected) throw std::runtime_error("TP2 fixture router distribution differs");
    for (int e : got) ++histogram[e];
    return histogram;
}

int Fixture::verify_source_coordinates() const {
    for (const auto gate : {false, true})
        for (const auto e : {0, shape.experts / 2, shape.experts - 1}) {
            const auto m = snapshot(gate, e);
            for (const int r : {0, 63, 127, m.n / 2 - 1, m.n / 2, m.n - 1})
                for (const int c : {0, 63, 127, m.k - 65, m.k - 1}) {
                    const bool fp8   = m.format != QType::NVFP4;
                    const auto index = std::size_t(r) * m.k + c;
                    const auto code =
                        fp8 ? m.codes[index] : (m.codes[index / 2] >> (4 * (c & 1))) & 15;
                    if (code != source_code(e, original_row(r, shape.intermediate, rank, gate),
                                            c + (gate ? 0 : rank * shape.intermediate),
                                            gate ? 101 : 211, fp8))
                        return 1;
                    const auto sr             = original_row(r, shape.intermediate, rank, gate);
                    const auto sc             = c + (gate ? 0 : rank * shape.intermediate);
                    const auto expected_scale = source_scale(e, sr, sc, gate ? 101 : 211, fp8);
                    if (fp8) {
                        const auto offset = ((r / m.block) * (m.k / m.block) + c / m.block) * 2;
                        std::uint16_t word;
                        std::memcpy(&word, m.scales.data() + offset, 2);
                        if (word != expected_scale) return 1;
                    } else {
                        const int group   = c / 16;
                        const auto offset = (r / 128 * (m.k / 64) + group / 4) * 512 + r % 32 * 16 +
                                            (r % 128) / 32 * 4 + group % 4;
                        if (m.scales[offset] != expected_scale) return 1;
                    }
                }
        }
    return 0;
}

void Fixture::preserve_weights() {
    auto& f = *impl_;
    f.preserved.clear();
    for (const DeviceBuffer* buffer :
         {&f.gate_up.storage, &f.down.storage, &f.shared_gate_up.storage, &f.shared_down.storage,
          &f.control, &f.bias}) {
        f.preserved.emplace_back(buffer->size());
        if (buffer->size()) buffer->copy_to_host(f.preserved.back().data(), buffer->size());
    }
}

int Fixture::verify_preserved() const {
    const auto& f = *impl_;
    const std::array<const DeviceBuffer*, 6> buffers{
        &f.gate_up.storage,     &f.down.storage, &f.shared_gate_up.storage,
        &f.shared_down.storage, &f.control,      &f.bias};
    if (f.preserved.size() != buffers.size())
        throw std::runtime_error("missing TP2 preservation baseline");
    std::vector<std::uint8_t> chunk(2ULL << 20);
    for (std::size_t i = 0; i < buffers.size(); ++i)
        for (std::size_t offset = 0; offset < buffers[i]->size(); offset += chunk.size()) {
            const auto bytes = std::min(chunk.size(), buffers[i]->size() - offset);
            buffers[i]->copy_to_host(chunk.data(), bytes, offset);
            if (std::memcmp(chunk.data(), f.preserved[i].data() + offset, bytes)) return 1;
        }
    return 0;
}

void Fixture::zero_routed() {
    for (Matrix* matrix : {&impl_->gate_up, &impl_->down})
        CUDA_CHECK(cudaMemsetAsync(matrix->storage.data(), 0, matrix->geometry.code_bytes,
                                   impl_->execution.stream()));
    CUDA_CHECK(cudaStreamSynchronize(impl_->execution.stream()));
}
} // namespace ninfer::test::moe_tp2
