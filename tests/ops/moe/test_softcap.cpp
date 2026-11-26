#include "core/decode_graph.h"
#include "ninfer/ops/moe.h"
#include "ops/moe/softcap_routes.h"
#include "ops/op_tester.h"
#include "ops/suite.h"

#include <limits>

namespace ninfer::test::softcap {
namespace {
using namespace ops;
constexpr int H = 2304, I = 1152, E = 128, K = 8, input_patterns = 4;
constexpr auto write_mode = MoeEpilogue::Write;
constexpr auto a16        = LinearPolicy::A16Only;
constexpr auto a4         = LinearPolicy::AllowA4;
constexpr auto a8         = LinearPolicy::AllowA8;
constexpr ReductionCriterion a16_criterion{1.0 / 256, 1.0 / 256, 2.0 / 256};
constexpr ReductionCriterion a4_criterion{.25, 1e-6, .35, 1e-6};
constexpr ReductionCriterion a8_criterion{.075, 1e-6, .09, 1e-6};

RoutedSwiGLUProfile profile(LinearPolicy arithmetic) {
    constexpr auto q      = QType::NVFP4_MUL;
    constexpr auto layout = QuantLayout::BlockScaleK16M128x4;
    return {
        {H, I, q, q, layout, layout, arithmetic, arithmetic, {}, SwiGLUSoftcap{4.F, 25.F}}, E, K};
}

RoutedSwiGLUProfile mxfp8_profile(LinearPolicy gate, LinearPolicy down) {
    constexpr auto q      = QType::MXFP8;
    constexpr auto layout = QuantLayout::RowGroupScale;
    return {{H, I, q, q, layout, layout, gate, down, {}, SwiGLUSoftcap{4.F, 25.F}}, E, K};
}

std::uint32_t hash(std::uint32_t value) {
    value = (value ^ (value >> 16)) * 0x7feb352dU;
    value = (value ^ (value >> 15)) * 0x846ca68bU;
    return value ^ (value >> 16);
}

std::size_t scale_offset(int row, int group, int columns) {
    return (((row / 128) * (columns / 64) + group / 4) * 32 + row % 32) * 16 +
           ((row / 32) % 4) * 4 + group % 4;
}

double decode_scale(std::uint8_t value) {
    const int exponent = (value >> 3) & 15, mantissa = value & 7;
    const double magnitude =
        exponent ? std::ldexp(8.0 + mantissa, exponent - 10) : std::ldexp(mantissa, -9);
    return value & 128 ? -magnitude : magnitude;
}

constexpr std::array fp4{0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.};

struct Matrix {
    int rows, columns, count;
    std::size_t code_bytes, scale_bytes;
    std::vector<std::uint8_t> codes, scales;
    std::vector<float> multipliers;
    DeviceBuffer device_codes, device_scales, device_multipliers;

    Matrix(Device device, int n, int k, unsigned seed, bool special)
        : rows(n), columns(k), count(special ? 1 : E), code_bytes(std::size_t(n) * k / 2),
          scale_bytes(std::size_t(n) * k / 16), codes(count * code_bytes),
          scales(count * scale_bytes), multipliers(count), device_codes(device, codes.size()),
          device_scales(device, scales.size()),
          device_multipliers(device, multipliers.size() * sizeof(float)) {
        if (special) {
            multipliers[0] = 1.F;
            if (k == H) {
                for (int r : {0, 1, I, I + 1}) {
                    codes[r * k / 2]              = r == 1 ? 0xf : 7;
                    scales[scale_offset(r, 0, k)] = 0x50; // 8.0 in E4M3.
                }
            } else {
                codes[0]                      = 2;
                codes[k / 2]                  = 0x20;
                scales[scale_offset(0, 0, k)] = scales[scale_offset(1, 0, k)] = 0x38;
            }
        } else {
            parallel_rows(count, std::uint64_t(count) * code_bytes, [&](int first, int last) {
                for (int e = first; e < last; ++e) {
                    multipliers[e] = (5 + e % 13) * (k == H ? .008F : .003F);
                    for (std::size_t j = 0; j < code_bytes; ++j)
                        codes[e * code_bytes + j] = hash(e * code_bytes + j + seed) >> 24;
                    for (std::size_t j = 0; j < scale_bytes; ++j)
                        scales[e * scale_bytes + j] = 0x20 + hash(e * scale_bytes + j + seed) % 21;
                }
            });
        }
        device_codes.copy_from_host(codes.data(), codes.size());
        device_scales.copy_from_host(scales.data(), scales.size());
        device_multipliers.copy_from_host(multipliers.data(), device_multipliers.size());
    }

    ExpertMatrixDeviceView view(const std::uint32_t* indices) const {
        ExpertMatrixDeviceView v;
        v.codes               = static_cast<const std::byte*>(device_codes.data());
        v.scales              = static_cast<const std::byte*>(device_scales.data());
        v.code_stride         = code_bytes;
        v.scale_stride        = scale_bytes;
        v.matrix_scales       = static_cast<const float*>(device_multipliers.data());
        v.matrix_scale_stride = 1;
        v.format              = QType::NVFP4_MUL;
        v.layout              = QuantLayout::BlockScaleK16M128x4;
        v.expert_indices      = indices;
        v.physical_count      = count;
        return v;
    }

    // Read the represented nibbles and E4M3 scales independently of device codecs.
    double dot(int e, int row, const double* x) const {
        const auto* packed = codes.data() + e * code_bytes + row * columns / 2;
        const auto* block  = scales.data() + e * scale_bytes;
        double result      = 0;
        for (int group = 0; group < columns / 16; ++group) {
            double sum = 0;
            for (int j = 0; j < 8; ++j) {
                const unsigned byte = packed[group * 8 + j];
                sum += fp4[byte & 15] * x[group * 16 + j * 2];
                sum += fp4[byte >> 4] * x[group * 16 + j * 2 + 1];
            }
            result += sum * decode_scale(block[scale_offset(row, group, columns)]);
        }
        return result * multipliers[e];
    }
};

// Indexed codes and scales share a byte offset for each packed expert. Multipliers
// follow the logical route ID, independently of the order of those packed experts.
template <class MatrixType>
struct IndexedMatrix {
    const MatrixType& matrix;
    std::size_t stride;
    DeviceBuffer storage, offsets, multipliers;

    IndexedMatrix(Device device, const MatrixType& source)
        : matrix(source), stride(source.code_bytes + source.scale_bytes),
          storage(device, source.count * stride), offsets(device, E * sizeof(std::uint64_t)),
          multipliers(device, source.view(nullptr).matrix_scales ? E * sizeof(float) : 0) {
        std::vector<std::uint8_t> packed(storage.size());
        for (int e = 0; e < matrix.count; ++e) {
            auto* slot = packed.data() + e * stride;
            std::copy_n(matrix.codes.data() + e * matrix.code_bytes, matrix.code_bytes, slot);
            std::copy_n(matrix.scales.data() + e * matrix.scale_bytes, matrix.scale_bytes,
                        slot + matrix.code_bytes);
        }
        storage.copy_from_host(packed.data(), packed.size());
    }

    ExpertMatrixDeviceView view() const {
        auto v        = matrix.view(nullptr);
        v.codes       = static_cast<const std::byte*>(storage.data());
        v.scales      = v.codes + matrix.code_bytes;
        v.code_stride = v.scale_stride = 0;
        v.physical_count               = 0;
        v.offsets                      = static_cast<const std::uint64_t*>(offsets.data());
        if (v.matrix_scales) v.matrix_scales = static_cast<const float*>(multipliers.data());
        v.indexed_storage = storage.span();
        return v;
    }

    void publish(const std::array<std::uint32_t, E>& slots) {
        std::array<std::uint64_t, E> byte_offsets;
        [[maybe_unused]] std::array<float, E> logical_scales;
        for (int e = 0; e < E; ++e) {
            const auto slot    = slots[e];
            const bool present = slot < static_cast<unsigned>(matrix.count);
            byte_offsets[e]    = present ? slot * stride : kExpertAbsentOffset;
            if constexpr (requires { matrix.multipliers; })
                logical_scales[e] = present ? matrix.multipliers[slot] : 1.F;
        }
        offsets.copy_from_host(byte_offsets.data(), sizeof(byte_offsets));
        if constexpr (requires { matrix.multipliers; })
            multipliers.copy_from_host(logical_scales.data(), sizeof(logical_scales));
    }
};

struct Fixture {
    DeviceContext context;
    Device device = context.device();
    Matrix gate, down;
    std::array<std::uint16_t, H * input_patterns> input{};
    std::array<float, E> input_scale{}, product_scale{};
    DeviceBuffer input_scales{device, sizeof(input_scale)},
        product_scales{device, sizeof(product_scale)};
    bool special;

    explicit Fixture(bool saturate = false)
        : gate(device, 2 * I, H, 1783, saturate), down(device, H, I, 61897, saturate),
          special(saturate) {
        for (int p = 0; p < input_patterns; ++p) {
            if (special) {
                input[p * H] = f32_to_bf16(1.F);
                continue;
            }
            for (int i = 0; i < H; ++i) {
                const int value  = int(hash(i + H * p + 49) % 193) - 96;
                input[p * H + i] = f32_to_bf16(value * (p == 0 ? .00390625F : .0078125F));
            }
        }
        for (int e = 0; e < E; ++e) {
            input_scale[e]   = .002F * (2 + (e % 5) * (e % 5));
            product_scale[e] = .0390625F * (1 + (e % 7));
        }
        input_scales.copy_from_host(input_scale.data(), input_scales.size());
        product_scales.copy_from_host(product_scale.data(), product_scales.size());
    }

    CudaSwiGLUExperts operands(const RoutedSwiGLUProfile& p, bool per_expert,
                               const std::uint32_t* indices = nullptr) const {
        using Calibration       = ActivationCalibrationView;
        constexpr auto multiply = ActivationScaleOperation::Multiply;
        CudaSwiGLUExperts w{{gate.view(indices), down.view(indices), E, device}, {}};
        if (p.ffn.gate_up_policy == a4) {
            w.activation_calibration.gate_up =
                per_expert ? Calibration::per_matrix(multiply, input_scale, input_scales.span())
                           : Calibration::broadcast(multiply, .03125F);
        }
        if (p.ffn.down_policy == a4) {
            w.activation_calibration.down =
                per_expert ? Calibration::per_matrix(multiply, product_scale, product_scales.span())
                           : Calibration::broadcast(multiply, .125F);
        }
        return w;
    }
};

struct Mxfp8Matrix {
    int columns, count;
    std::size_t code_bytes, scale_bytes;
    std::vector<std::uint8_t> codes, scales;
    DeviceBuffer device_codes, device_scales;

    Mxfp8Matrix(Device device, int rows, int k, unsigned seed, bool special)
        : columns(k), count(special ? 1 : 16), code_bytes(std::size_t(rows) * k),
          scale_bytes(code_bytes / 32), codes(count * code_bytes), scales(count * scale_bytes, 127),
          device_codes(device, codes.size()), device_scales(device, scales.size()) {
        if (special) {
            if (k == H) {
                for (int r : {0, 1, I, I + 1}) {
                    codes[r * k]         = r == 1 ? 0xfe : 0x7e;
                    scales[r * (k / 32)] = 124;
                }
            } else {
                codes[0] = codes[k + 1] = 0x38;
            }
        } else {
            parallel_rows(count, std::uint64_t(count) * code_bytes, [&](int first, int last) {
                for (int e = first; e < last; ++e) {
                    for (std::size_t j = 0; j < code_bytes; ++j) {
                        const unsigned bits       = hash(e * code_bytes + j + seed);
                        codes[e * code_bytes + j] = (bits & 128) | (bits % 127);
                    }
                    for (std::size_t j = 0; j < scale_bytes; ++j)
                        scales[e * scale_bytes + j] = 113 + hash(e * scale_bytes + j + seed) % 4;
                }
            });
        }
        device_codes.copy_from_host(codes.data(), codes.size());
        device_scales.copy_from_host(scales.data(), scales.size());
    }

    ExpertMatrixDeviceView view(const std::uint32_t* indices) const {
        ExpertMatrixDeviceView out;
        out.codes          = static_cast<const std::byte*>(device_codes.data());
        out.scales         = static_cast<const std::byte*>(device_scales.data());
        out.code_stride    = code_bytes;
        out.scale_stride   = scale_bytes;
        out.format         = QType::MXFP8;
        out.layout         = QuantLayout::RowGroupScale;
        out.expert_indices = indices;
        out.physical_count = count;
        return out;
    }

    double dot(int expert, int row, const double* x) const {
        static const auto decoded = [] {
            std::array<double, 256> values{};
            for (unsigned code = 0; code < values.size(); ++code) values[code] = decode_scale(code);
            return values;
        }();
        const auto* w = codes.data() + expert * code_bytes + row * columns;
        const auto* s = scales.data() + expert * scale_bytes + row * (columns / 32);
        double result = 0;
        for (int g = 0; g < columns / 32; ++g) {
            double sum = 0;
            for (int k = g * 32; k < (g + 1) * 32; ++k) sum += decoded[w[k]] * x[k];
            result += std::ldexp(sum, int(s[g]) - 127);
        }
        return result;
    }
};

struct Mxfp8Fixture {
    DeviceContext context;
    Device device = context.device();
    Mxfp8Matrix gate, down;
    std::array<std::uint16_t, H * input_patterns> input{};
    bool special;

    explicit Mxfp8Fixture(bool saturate = false)
        : gate(device, 2 * I, H, 4321, saturate), down(device, H, I, 8765, saturate),
          special(saturate) {
        for (int p = 0; p < input_patterns; ++p) {
            if (special) {
                input[p * H] = f32_to_bf16(1.F);
                continue;
            }
            for (int i = 0; i < H; ++i) {
                const int v      = int(hash(i + H * p + 97) % 193) - 96;
                input[p * H + i] = f32_to_bf16(v * (p == 0 ? .00390625F : .0078125F));
            }
        }
    }

    CudaSwiGLUExperts operands(const RoutedSwiGLUProfile&, bool,
                               const std::uint32_t* indices) const {
        return {{gate.view(indices), down.view(indices), E, device}, {}};
    }
};

struct Oracle {
    std::vector<double> projection;

    template <class TestFixture>
    Oracle(const TestFixture& f, SwiGLUSoftcap caps)
        : projection(std::size_t(f.gate.count) * input_patterns * H) {
        std::array<double, H * input_patterns> x;
        std::transform(f.input.begin(), f.input.end(), x.begin(), bf16_to_f32);
        parallel_rows(f.gate.count, std::uint64_t(f.gate.count) * input_patterns * H * I * 3,
                      [&](int first, int last) {
                          for (int e = first; e < last; ++e)
                              for (int p = 0; p < input_patterns; ++p) {
                                  std::array<double, I> product;
                                  for (int row = 0; row < I; ++row) {
                                      const double g = f.gate.dot(e, row, x.data() + p * H);
                                      const double u = f.gate.dot(e, row + I, x.data() + p * H);
                                      product[row] =
                                          (caps.gate_cap * std::tanh(g / caps.gate_cap)) *
                                          (1.0 / (1.0 + std::exp(-g))) *
                                          (caps.up_cap * std::tanh(u / caps.up_cap));
                                  }
                                  for (int row = 0; row < H; ++row)
                                      projection[(e * input_patterns + p) * H + row] =
                                          f.down.dot(e, row, product.data());
                              }
                      });
    }

    const double* column(int physical, int input) const {
        return projection.data() + (physical * input_patterns + input) * H;
    }
};

struct Case {
    int tokens;
    const char* distribution;
    LinearPolicy policy;
    DType output            = DType::BF16;
    MoeEpilogue epilogue    = write_mode;
    ExpertSubsetKind subset = ExpertSubsetKind::All;
    bool graph              = false;
    bool per_expert         = false;
    bool missing_routes     = false;
    bool indexed            = false;
    bool overflowing_scales = false;
    bool missing_experts    = false;
    bool singleton_routes   = false;
};

void require(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}

std::size_t capacity(const RoutedSwiGLUProfile& p, ExpertSubsetKind subset, DType type,
                     MoeEpilogue mode, int first, int last) {
    return routed_swiglu_workspace_capacity_bytes(p, subset, type, mode, first, last);
}

template <class TestFixture>
int run(const TestFixture& f, const Oracle& oracle, RoutedSwiGLUProfile p, const Case& c,
        const std::string& label) {
    const int tokens = c.tokens, assignments = tokens * K;
    const std::size_t elements = std::size_t(tokens) * H;
    const auto cuda            = f.context.execution_view();
    const auto scratch_bytes   = capacity(p, c.subset, c.output, c.epilogue, tokens, tokens);
    GuardedDeviceBuffer x(elements * 2), y(elements * dtype_size(c.output)),
        workspace(scratch_bytes), ids(assignments * 4), coefficients(assignments * 4), mask(E * 4),
        indices(E * 4);
    auto output = Tensor(y.span(), c.output, {H, tokens});
    const Tensor input(x.span(), DType::BF16, {H, tokens});
    const ExpertRoutes routes =
        c.subset == ExpertSubsetKind::None
            ? ExpertRoutes{}
            : ExpertRoutes{Tensor(ids.span(), DType::I32, {K, tokens}),
                           Tensor(coefficients.span(), DType::FP32, {K, tokens})};
    ExpertSubset subset{c.subset, {}};
    if (c.subset == ExpertSubsetKind::MaskOnes || c.subset == ExpertSubsetKind::MaskZeros)
        subset.mask = Tensor(mask.span(), DType::I32, {E});
    auto w = c.subset == ExpertSubsetKind::None
                 ? CudaSwiGLUExperts{}
                 : f.operands(p, c.per_expert, static_cast<const std::uint32_t*>(indices.data()));
    std::optional<IndexedMatrix<std::remove_cvref_t<decltype(f.gate)>>> indexed_gate, indexed_down;
    if (c.indexed) {
        indexed_gate.emplace(f.device, f.gate);
        indexed_down.emplace(f.device, f.down);
        w.bank.gate_up = indexed_gate->view();
        w.bank.down    = indexed_down->view();
    }
    if (c.overflowing_scales && allows_a4(p.ffn.down_policy)) {
        constexpr auto multiply = ActivationScaleOperation::Multiply;
        const auto huge =
            ActivationCalibrationView::broadcast(multiply, std::numeric_limits<float>::max());
        w.activation_calibration.gate_up = huge;
        w.activation_calibration.down    = huge;
    }
    MemoryArena arena(workspace.span());
    const auto call = [&] {
        routed_swiglu(input, p, w, routes, subset, c.epilogue, output, arena, cuda);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (c.graph) {
        definition.capture(cuda.stream(), call);
        graph.instantiate(definition);
    }
    int failures = 0;
    for (int phase = 0; phase < (c.graph ? 3 : 1); ++phase) {
        std::vector<std::uint16_t> host_x(elements), residual(elements);
        for (int t = 0; t < tokens; ++t) {
            const int pattern = (t + phase) % input_patterns;
            std::copy_n(f.input.data() + pattern * H, H, host_x.data() + std::size_t(t) * H);
        }
        auto host_ids = make_routes(phase == 1   ? "reuse"
                                    : phase == 2 ? "tail"
                                                 : c.distribution,
                                    tokens, E, K);
        std::vector<float> host_coeff(assignments);
        std::array<int, E> host_mask;
        std::array<std::uint32_t, E> host_indices;
        const auto selected = [&](int e) {
            return c.subset == ExpertSubsetKind::All ||
                   (c.subset != ExpertSubsetKind::None &&
                    host_mask[e] == (c.subset == ExpertSubsetKind::MaskOnes));
        };
        for (int e = 0; e < E; ++e) {
            host_mask[e]    = (e + phase) % 3 == 0;
            host_indices[e] = !selected(e)                      ? UINT32_MAX - 1
                              : c.missing_experts && e % 5 == 4 ? UINT32_MAX
                              : f.special                       ? 0
                                          : (e * 17 + phase * 31) % f.gate.count;
        }
        for (int a = 0; a < assignments; ++a) {
            if (c.singleton_routes)
                host_ids[a] = a % K == 0 ? a / K : -1;
            else if (f.special || (c.missing_routes && (a / K % 7 == 2 || a % 5 == 2)))
                host_ids[a] = f.special && a % K == 0 ? 0 : -1;
            host_coeff[a] = host_ids[a] < 0
                                ? (a % 2 ? -.375F : std::numeric_limits<float>::quiet_NaN())
                            : c.overflowing_scales && !allows_a4(p.ffn.down_policy)
                                ? std::numeric_limits<float>::max()
                            : f.special ? 1.F
                                        : .11F * ((a % K) - 2);
        }
        for (std::size_t i = 0; i < elements; ++i) {
            if (c.epilogue == MoeEpilogue::AddResidual)
                residual[i] = f32_to_bf16((int(i % 17) - 8) * .005F);
        }
        x.copy_from_host(host_x.data(), x.bytes());
        ids.copy_from_host(host_ids.data(), ids.bytes());
        coefficients.copy_from_host(host_coeff.data(), coefficients.bytes());
        mask.copy_from_host(host_mask.data(), mask.bytes());
        if (c.indexed) {
            indexed_gate->publish(host_indices);
            indexed_down->publish(host_indices);
        } else {
            indices.copy_from_host(host_indices.data(), indices.bytes());
        }
        if (c.epilogue == MoeEpilogue::AddResidual)
            y.copy_from_host(residual.data(), y.bytes());
        else
            y.fill(0xff);
        if (c.graph)
            graph.launch(cuda.stream());
        else
            call();
        CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
        std::vector<double> expected(elements);
        parallel_rows(tokens, std::uint64_t(elements) * K, [&](int first, int last) {
            for (int t = first; t < last; ++t) {
                const auto base = std::size_t(t) * H;
                auto* out       = expected.data() + base;
                for (int j = 0; j < K; ++j) {
                    const auto e = host_ids[t * K + j];
                    if (e < 0 || !selected(e) || host_indices[e] == UINT32_MAX) continue;
                    const auto* value =
                        oracle.column(host_indices[e], (t + phase) % input_patterns);
                    const double coefficient = host_coeff[t * K + j];
                    for (int r = 0; r < H; ++r) out[r] += coefficient * value[r];
                }
                if (c.epilogue == MoeEpilogue::AddResidual)
                    for (int r = 0; r < H; ++r) out[r] += bf16_to_f32(residual[base + r]);
            }
        });
        std::vector<double> actual(elements);
        if (c.output == DType::BF16) {
            std::vector<std::uint16_t> words(elements);
            y.copy_to_host(words.data(), y.bytes());
            std::transform(words.begin(), words.end(), actual.begin(), bf16_to_f32);
            if (c.subset == ExpertSubsetKind::None && c.epilogue == MoeEpilogue::AddResidual)
                failures += verify_exact("None preserves residual bits", words, residual);
        } else {
            std::vector<float> values(elements);
            y.copy_to_host(values.data(), y.bytes());
            std::copy(values.begin(), values.end(), actual.begin());
        }
        if (c.overflowing_scales)
            require(
                std::all_of(actual.begin(), actual.end(), [](double value) { return value == 0; }),
                "overflowing finite scales must preserve exact zero output");
        if (f.special && !c.overflowing_scales && !c.singleton_routes) {
            // The capped-gate sigmoid would give a spurious response at row 1
            // of about -1.7. Both FP4 and BF16 must retain the raw negative gate.
            require(actual[0] > 70.0 && std::abs(actual[1]) < .01,
                    "softcap sigmoid must use the raw gate");
        }
        for (int t = 0; t < tokens; ++t) {
            const auto begin = host_ids.begin() + t * K;
            if (std::any_of(begin, begin + K, [&](int e) {
                    return e >= 0 && selected(e) && host_indices[e] != UINT32_MAX;
                }))
                continue;
            for (int r = 0; r < H; ++r) {
                const std::size_t index = std::size_t(t) * H + r;
                require(actual[index] == bf16_to_f32(residual[index]),
                        "inactive route polluted output");
            }
        }
        const bool allow_a4 = allows_a4(p.ffn.gate_up_policy) || allows_a4(p.ffn.down_policy);
        const bool allow_a8 = allows_a8(p.ffn.gate_up_policy) || allows_a8(p.ffn.down_policy);
        failures += verify_reduction(label + "/phase" + std::to_string(phase), actual, expected,
                                     allow_a4   ? a4_criterion
                                     : allow_a8 ? a8_criterion
                                                : a16_criterion);
        failures +=
            y.verify_guards(label) + workspace.verify_guards(label) + x.verify_guards(label);
        require(arena.used() == 0 && arena.peak_used() == scratch_bytes, "workspace high water");
    }
    if (scratch_bytes) {
        MemorySpan short_span = workspace.span();
        --short_span.bytes;
        MemoryArena short_arena(short_span);
        try {
            routed_swiglu(input, p, w, routes, subset, c.epilogue, output, short_arena, cuda);
            throw std::runtime_error("capacity minus one was accepted");
        } catch (const std::bad_alloc&) {
            require(short_arena.used() == 0, "failed call retained the workspace cursor");
        }
    }
    return failures;
}

template <class Function>
void rejects(Function fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid softcap call was accepted");
}

int contracts() {
    auto p           = profile(a16);
    const auto bytes = [&](const auto& f) {
        return capacity(f, ExpertSubsetKind::All, DType::BF16, write_mode, 1, 1);
    };
    for (bool gate : {true, false})
        for (float cap : {0.F, -1.F, std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()}) {
            auto invalid                                                         = p;
            (gate ? invalid.ffn.softcap->gate_cap : invalid.ffn.softcap->up_cap) = cap;
            rejects([&] { (void)bytes(invalid); });
        }
    auto invalid            = p;
    invalid.ffn.clamp_limit = 4.F;
    rejects([&] { (void)bytes(invalid); });
    invalid = p;
    invalid.ffn.softcap.reset();
    rejects([&] { (void)bytes(invalid); });
    invalid               = p;
    invalid.ffn.down_type = QType::NVFP4;
    rejects([&] { (void)bytes(invalid); });
    invalid                    = p;
    invalid.ffn.gate_up_layout = QuantLayout::RowSplit;
    rejects([&] { (void)bytes(invalid); });
    invalid                    = p;
    invalid.ffn.gate_up_policy = LinearPolicy::AllowA8;
    rejects([&] { (void)bytes(invalid); });
    invalid                 = p;
    invalid.ffn.down_policy = static_cast<LinearPolicy>(255);
    rejects([&] { (void)bytes(invalid); });
    for (const int h : {2048, 2560, 4096}) {
        invalid            = p;
        invalid.ffn.hidden = h;
        rejects([&] { (void)bytes(invalid); });
    }
    rejects([&] { (void)swiglu_workspace_capacity_bytes(p.ffn, 1, 1); });
    rejects([&] {
        (void)routed_swiglu_workspace_capacity_bytes(p, SwiGLUProfile{}, ExpertSubsetKind::All,
                                                     DType::BF16, write_mode, 1, 1);
    });
    rejects([&] {
        (void)capacity(p, ExpertSubsetKind::All, DType::FP32, MoeEpilogue::AddResidual, 1, 1);
    });
    rejects([&] { (void)capacity(p, ExpertSubsetKind::All, DType::BF16, write_mode, 0, 1); });
    rejects([&] { (void)capacity(p, ExpertSubsetKind::All, DType::BF16, write_mode, 2, 1); });
    for (auto policy : {a16, a4}) {
        p = profile(policy);
        for (auto [lo, hi] :
             {std::pair{1, 129}, {60, 68}, {4093, 4101}, {4097, 4219}, {7900, 8350}, {1, 12289}}) {
            std::size_t expected = 0;
            for (int t = lo; t <= hi; ++t)
                expected = std::max(
                    expected, capacity(p, ExpertSubsetKind::All, DType::BF16, write_mode, t, t));
            require(expected == capacity(p, ExpertSubsetKind::All, DType::BF16, write_mode, lo, hi),
                    "interval did not report the exact high water");
        }
    }
    Fixture f(true);
    const auto cuda = f.context.execution_view();
    DeviceBuffer x(f.device, H * 2), y(f.device, H * 2), ids(f.device, K * 4),
        coefficients(f.device, K * 4), work(f.device, bytes(p));
    Tensor input(x.span(), DType::BF16, {H, 1}), output(y.span(), DType::BF16, {H, 1});
    const ExpertRoutes routes{Tensor(ids.span(), DType::I32, {K, 1}),
                              Tensor(coefficients.span(), DType::FP32, {K, 1})};
    const auto* invalid_indices = reinterpret_cast<const std::uint32_t*>(0x10000);
    auto w                      = f.operands(p, false, invalid_indices);
    MemoryArena arena(work.span());
    rejects([&] {
        routed_swiglu(input, p, w, routes, {}, ScaledSwiGLUTerm{}, write_mode, output, arena, cuda);
    });
    rejects([&] { routed_swiglu(input, p, w, routes, {}, write_mode, input, arena, cuda); });
    MemoryArena overlapping({x.data(), work.size(), f.device});
    rejects([&] { routed_swiglu(input, p, w, routes, {}, write_mode, output, overlapping, cuda); });
    auto incorrect                        = w;
    incorrect.activation_calibration.down = {};
    rejects(
        [&] { routed_swiglu(input, p, incorrect, routes, {}, write_mode, output, arena, cuda); });
    incorrect = w;
    incorrect.activation_calibration.gate_up =
        ActivationCalibrationView::broadcast(ActivationScaleOperation::Divide, 1.F);
    rejects(
        [&] { routed_swiglu(input, p, incorrect, routes, {}, write_mode, output, arena, cuda); });
    incorrect                                = w;
    incorrect.activation_calibration.gate_up = ActivationCalibrationView::broadcast(
        ActivationScaleOperation::Multiply, std::numeric_limits<float>::infinity());
    rejects(
        [&] { routed_swiglu(input, p, incorrect, routes, {}, write_mode, output, arena, cuda); });

    // With all -1 assignments, even an unmapped mask or index address is unused.
    ids.fill(0xff);
    coefficients.fill(0xff);
    x.fill(0xff);
    const ExpertSubset empty{ExpertSubsetKind::MaskOnes,
                             Tensor(reinterpret_cast<void*>(0x20000), f.device, DType::I32, {E})};
    routed_swiglu(input, p, w, routes, empty, write_mode, output, arena, cuda);
    CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
    return verify_exact("all -1 skips mask and expert index", from_device<std::uint16_t>(y, H),
                        std::vector<std::uint16_t>(H));
}

int mxfp8_contracts() {
    constexpr auto all = ExpertSubsetKind::All;
    const auto bytes   = [](const RoutedSwiGLUProfile& p, int lo, int hi) {
        return capacity(p, all, DType::BF16, write_mode, lo, hi);
    };
    auto p = mxfp8_profile(a8, a8);
    for (const auto& policies : {std::pair{a16, a16}, {a8, a8}, {a16, a8}, {a8, a16}}) {
        const auto profile = mxfp8_profile(policies.first, policies.second);
        for (auto [lo, hi] : {std::pair{1, 260},
                              {7, 9},
                              {60, 68},
                              {2047, 2049},
                              {4093, 4101},
                              {4097, 4219},
                              {7900, 8350},
                              {1, 12289}}) {
            std::size_t expected = 0;
            for (int t = lo; t <= hi; ++t) expected = std::max(expected, bytes(profile, t, t));
            require(bytes(profile, lo, hi) == expected, "MXFP8 workspace interval high water");
        }
    }
    auto invalid            = p;
    invalid.ffn.down_policy = a4;
    rejects([&] { (void)bytes(invalid, 1, 1); });
    invalid                    = p;
    invalid.ffn.gate_up_policy = a4;
    rejects([&] { (void)bytes(invalid, 1, 1); });
    for (float cap : {0.F, -1.F, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
        invalid                       = p;
        invalid.ffn.softcap->gate_cap = cap;
        rejects([&] { (void)bytes(invalid, 1, 1); });
        invalid                     = p;
        invalid.ffn.softcap->up_cap = cap;
        rejects([&] { (void)bytes(invalid, 1, 1); });
    }
    invalid                 = p;
    invalid.ffn.clamp_limit = 1;
    rejects([&] { (void)bytes(invalid, 1, 1); });
    invalid                 = p;
    invalid.ffn.down_layout = QuantLayout::BlockScaleK16M128x4;
    rejects([&] { (void)bytes(invalid, 1, 1); });
    rejects([&] { (void)capacity(p, all, DType::FP32, MoeEpilogue::AddResidual, 1, 1); });
    require(capacity(p, ExpertSubsetKind::None, DType::BF16, write_mode, 1, 8193) == 0,
            "MXFP8 None has no operand workspace");

    const Mxfp8Fixture f(true);
    const auto cuda = f.context.execution_view();
    DeviceBuffer x(f.device, H * 2), y(f.device, H * 2), ids(f.device, K * 4),
        coefficients(f.device, K * 4), indices(f.device, E * 4), work(f.device, bytes(p, 1, 1));
    const Tensor input(x.span(), DType::BF16, {H, 1});
    Tensor output(y.span(), DType::BF16, {H, 1});
    const ExpertRoutes routes{Tensor(ids.span(), DType::I32, {K, 1}),
                              Tensor(coefficients.span(), DType::FP32, {K, 1})};
    const auto good = f.operands(p, false, static_cast<const std::uint32_t*>(indices.data()));
    auto w          = good;
    MemoryArena arena(work.span());
    const auto call = [&] {
        routed_swiglu(input, p, w, routes, {}, write_mode, output, arena, cuda);
    };
    w.bank.gate_up.matrix_scales = static_cast<const float*>(coefficients.data());
    rejects(call);
    w = good;
    w.activation_calibration.down =
        ActivationCalibrationView::broadcast(ActivationScaleOperation::Multiply, 1.F);
    rejects(call);
    w                          = good;
    w.bank.down.physical_count = 0;
    rejects(call);
    w                  = good;
    w.bank.down.format = QType::NVFP4_MUL;
    rejects(call);
    w = good;
    rejects([&] {
        routed_swiglu(input, p, w, routes, {}, ScaledSwiGLUTerm{}, write_mode, output, arena, cuda);
    });

    ids.fill(0xff);
    coefficients.fill(0xff);
    x.fill(0xff);
    w.bank.gate_up.expert_indices = w.bank.down.expert_indices =
        reinterpret_cast<const std::uint32_t*>(0x10000);
    const ExpertSubset empty{ExpertSubsetKind::MaskOnes,
                             Tensor(reinterpret_cast<void*>(0x20000), f.device, DType::I32, {E})};
    routed_swiglu(input, p, w, routes, empty, write_mode, output, arena, cuda);
    CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
    return verify_exact("MXFP8 -1 routes skip mask and index map", from_device<std::uint16_t>(y, H),
                        std::vector<std::uint16_t>(H));
}

void add_mxfp8_cases(OpSuite& suite) {
    suite.add("mxfp8-softcap/contracts", mxfp8_contracts);
    suite.add("mxfp8-softcap/raw-gate", [] {
        const Mxfp8Fixture f(true);
        const auto p = mxfp8_profile(a16, a16);
        const Oracle oracle(f, *p.ffn.softcap);
        int fails = run(f, oracle, p, {1, "reuse", a16, DType::FP32}, "mxfp8/raw-gate/a16");
        fails += run(f, oracle, mxfp8_profile(a8, a8), {24, "reuse", a8, DType::FP32},
                     "mxfp8/raw-gate/a8");
        return fails;
    });
    for (auto policy : {a16, a8}) {
        for (int tokens : {1, 4, 8, 65}) {
            const std::string id = "mxfp8-softcap/pre-scale-overflow/" +
                                   std::string(policy == a16 ? "a16" : "a8") + "/T" +
                                   std::to_string(tokens);
            suite.add(id, [id, policy, tokens] {
                Mxfp8Fixture f(true);
                // 448 * 2^120 exceeds FP32 before scaling. The represented dot is 3.5.
                for (int r : {0, 1, I, I + 1}) f.gate.scales[r * (H / 32)] = 0;
                f.gate.device_scales.copy_from_host(f.gate.scales.data(), f.gate.scales.size());
                for (int i = 0; i < input_patterns; ++i)
                    f.input[i * H] = f32_to_bf16(std::ldexp(1.F, 120));
                const auto p = mxfp8_profile(policy, policy);
                const Oracle oracle(f, *p.ffn.softcap);
                return run(f, oracle, p,
                           {.tokens           = tokens,
                            .distribution     = "spread",
                            .policy           = policy,
                            .output           = DType::FP32,
                            .singleton_routes = true},
                           id);
            });
        }
    }

    struct Mxfp8Case {
        Case test;
        LinearPolicy down;
    };

    std::vector<Mxfp8Case> cases;
    for (auto policy : {a16, a8}) {
        for (int tokens : {1, 2, 4, 8, 9, 16, 20, 64, 65, 128, 192, 1024})
            cases.push_back({{tokens, tokens % 2 ? "uniform" : "tail", policy}, policy});
        for (int tokens : {20, 64, 192, 1024})
            for (bool indexed : {false, true})
                cases.push_back({{.tokens          = tokens,
                                  .distribution    = "tail",
                                  .policy          = policy,
                                  .output          = DType::FP32,
                                  .subset          = ExpertSubsetKind::MaskZeros,
                                  .missing_routes  = true,
                                  .indexed         = indexed,
                                  .missing_experts = true},
                                 policy});
    }
    for (auto subset : {ExpertSubsetKind::None, ExpertSubsetKind::All, ExpertSubsetKind::MaskOnes,
                        ExpertSubsetKind::MaskZeros})
        for (auto mode : {write_mode, MoeEpilogue::AddResidual})
            cases.push_back({{.tokens         = 8,
                              .distribution   = "spread",
                              .policy         = a8,
                              .epilogue       = mode,
                              .subset         = subset,
                              .graph          = true,
                              .missing_routes = true},
                             a8});
    cases.push_back({{.tokens         = 4097,
                      .distribution   = "reuse",
                      .policy         = a8,
                      .output         = DType::FP32,
                      .missing_routes = true},
                     a8});
    cases.push_back({{.tokens         = 1024,
                      .distribution   = "reuse",
                      .policy         = a16,
                      .output         = DType::FP32,
                      .subset         = ExpertSubsetKind::MaskOnes,
                      .missing_routes = true,
                      .indexed        = true},
                     a8});
    cases.push_back({{.tokens       = 1024,
                      .distribution = "tail",
                      .policy       = a8,
                      .epilogue     = MoeEpilogue::AddResidual},
                     a16});
    for (int tokens : {17, 2047, 2048})
        cases.push_back({{.tokens       = tokens,
                          .distribution = "pool",
                          .policy       = a8,
                          .output       = DType::FP32,
                          .indexed      = true},
                         a8});
    // Replay crosses from dispersed rows to full expert coverage and back to a tail.
    // This exercises both gathered and contiguous MXFP8 input staging at one address.
    for (bool indexed : {false, true})
        cases.push_back({{.tokens       = 192,
                          .distribution = "spread",
                          .policy       = a8,
                          .epilogue     = MoeEpilogue::AddResidual,
                          .graph        = true,
                          .indexed      = indexed},
                         a8});

    std::vector<OpCase> catalog;
    for (std::size_t j = 0; j < cases.size(); ++j) {
        const auto& [c, down]    = cases[j];
        const std::string policy = c.policy != down  ? (c.policy == a16 ? "a16-a8" : "a8-a16")
                                   : c.policy == a16 ? "a16"
                                                     : "a8";
        catalog.push_back({"mxfp8-softcap/" + policy + "/T" + std::to_string(c.tokens) + "/" +
                               c.distribution + (c.indexed ? "/indexed/" : "/affine/") +
                               std::to_string(j),
                           {{"format", "mxfp8"},
                            {"policy", policy},
                            {"execution", c.graph ? "graph" : "eager"}}});
    }
    suite.add(catalog, [cases, catalog](std::span<const std::size_t> selection) {
        const Mxfp8Fixture f;
        const Oracle oracle(f, *mxfp8_profile(a16, a16).ffn.softcap);
        int fails = 0;
        for (const auto j : selection) {
            const auto& [c, down] = cases[j];
            fails += run(f, oracle, mxfp8_profile(c.policy, down), c, catalog[j].id);
        }
        return fails;
    });
    suite.add("mxfp8-softcap/custom-caps", [] {
        const Mxfp8Fixture f;
        auto p        = mxfp8_profile(a16, a8);
        p.ffn.softcap = SwiGLUSoftcap{3.25F, 15.75F};
        const Oracle oracle(f, *p.ffn.softcap);
        return run(f, oracle, p, {192, "pool", a16, DType::FP32}, "mxfp8/custom-caps");
    });
}
} // namespace
} // namespace ninfer::test::softcap

int main(int argc, char** argv) {
    using namespace ninfer;
    using namespace ninfer::test;
    using namespace ninfer::test::softcap;
    OpSuite suite;
    add_mxfp8_cases(suite);
    suite.add("softcap/contracts", contracts);
    suite.add("softcap/raw-gate", [] {
        Fixture f(true);
        auto p = profile(a16);
        const Oracle oracle(f, *p.ffn.softcap);
        int failures = run(f, oracle, p, {1, "reuse", a16, DType::FP32}, "raw-gate/a16");
        p            = profile(a4);
        for (int tokens : {2, 24})
            failures += run(f, oracle, p, {tokens, "reuse", a4, DType::FP32}, "raw-gate/a4");
        return failures;
    });
    suite.add("softcap/reuse-dispatch-graph", [] {
        const Fixture f;
        const auto p = profile(a4);
        const Oracle oracle(f, *p.ffn.softcap);
        // Spread, Hot-8 and tail replay on the same graph with the same scratch.
        return run(f, oracle, p,
                   {.tokens         = 8,
                    .distribution   = "spread",
                    .policy         = a4,
                    .epilogue       = MoeEpilogue::AddResidual,
                    .subset         = ExpertSubsetKind::MaskZeros,
                    .graph          = true,
                    .per_expert     = true,
                    .missing_routes = true},
                   "reuse-dispatch/graph");
    });
    suite.add("softcap/indexed-bank", [] {
        const Fixture f;
        const auto p = profile(a4);
        const Oracle oracle(f, *p.ffn.softcap);
        int failures = 0;
        for (int tokens : {24, 192})
            failures += run(f, oracle, p,
                            {.tokens         = tokens,
                             .distribution   = "tail",
                             .policy         = a4,
                             .output         = DType::FP32,
                             .subset         = ExpertSubsetKind::MaskZeros,
                             .graph          = true,
                             .per_expert     = true,
                             .missing_routes = true,
                             .indexed        = true},
                            "indexed-bank/T" + std::to_string(tokens));
        return failures;
    });
    suite.add("softcap/zero-with-overflowing-scales", [] {
        Fixture f(true);
        f.input.fill(0);
        for (auto* matrix : {&f.gate, &f.down}) {
            matrix->multipliers[0] = 2.F;
            matrix->device_multipliers.copy_from_host(matrix->multipliers.data(), sizeof(float));
        }
        const Oracle oracle(f, *profile(a4).ffn.softcap);
        int failures = 0;
        for (int tokens : {24, 192})
            for (bool indexed : {false, true})
                failures += run(f, oracle, profile(a4),
                                {.tokens             = tokens,
                                 .distribution       = "reuse",
                                 .policy             = a4,
                                 .output             = DType::FP32,
                                 .indexed            = indexed,
                                 .overflowing_scales = true},
                                "zero-scales/a4/T" + std::to_string(tokens));
        for (int tokens : {1, 4, 16, 64})
            failures += run(f, oracle, profile(a16),
                            {.tokens             = tokens,
                             .distribution       = "reuse",
                             .policy             = a16,
                             .output             = DType::FP32,
                             .overflowing_scales = true},
                            "zero-scales/a16/T" + std::to_string(tokens));
        return failures;
    });
    std::vector<Case> cases;
    // Exercise small decode, both grouping sizes, prefill, and slice boundaries.
    constexpr int token_points[]{1,   2,   3,   4,   8,   9,    10,   11,   12,   15,  16, 17,
                                 18,  19,  20,  21,  32,  42,   43,   44,   64,   65,  66, 128,
                                 191, 192, 193, 256, 512, 1024, 2048, 4095, 4096, 4097};
    for (auto policy : {a16, a4})
        for (int t : token_points)
            for (const auto* distribution : distributions)
                cases.push_back({.tokens       = t,
                                 .distribution = distribution,
                                 .policy       = policy,
                                 .per_expert   = (t & 1) == 0});
    for (auto policy : {a16, a4})
        for (auto subset : {ExpertSubsetKind::All, ExpertSubsetKind::None,
                            ExpertSubsetKind::MaskOnes, ExpertSubsetKind::MaskZeros})
            for (auto mode : {write_mode, MoeEpilogue::AddResidual})
                cases.push_back({.tokens         = policy == a16 ? 17 : 129,
                                 .distribution   = "spread",
                                 .policy         = policy,
                                 .epilogue       = mode,
                                 .subset         = subset,
                                 .graph          = true,
                                 .per_expert     = true,
                                 .missing_routes = true});
    cases.push_back({.tokens         = 4097,
                     .distribution   = "tail",
                     .policy         = a4,
                     .output         = DType::FP32,
                     .subset         = ExpertSubsetKind::MaskZeros,
                     .graph          = true,
                     .missing_routes = true});
    for (auto policy : {a16, a4})
        for (int tokens : {8, 129, 1024})
            cases.push_back({.tokens         = tokens,
                             .distribution   = "tail",
                             .policy         = policy,
                             .output         = DType::FP32,
                             .subset         = ExpertSubsetKind::MaskOnes,
                             .graph          = true,
                             .per_expert     = true,
                             .missing_routes = true});

    std::vector<OpCase> catalog;
    for (std::size_t j = 0; j < cases.size(); ++j) {
        const auto& c      = cases[j];
        const char* policy = c.policy == a16 ? "a16" : "a4";
        catalog.push_back({"softcap/" + std::string(policy) + "/T" + std::to_string(c.tokens) +
                               "/" + c.distribution + "/" + std::to_string(j),
                           {{"policy", policy},
                            {"execution", c.graph ? "graph" : "eager"},
                            {"format", "nvfp4-mul"}}});
    }
    suite.add(catalog, [cases, catalog](std::span<const std::size_t> selection) {
        Fixture f;
        const Oracle oracle(f, *profile(a16).ffn.softcap);
        int failures = 0;
        for (const auto j : selection)
            failures += run(f, oracle, profile(cases[j].policy), cases[j], catalog[j].id);
        return failures;
    });
    suite.add("softcap/custom-caps-and-mixed-policy", [] {
        Fixture f;
        auto p            = profile(a16);
        p.ffn.softcap     = SwiGLUSoftcap{3.25F, 15.75F};
        p.ffn.down_policy = a4;
        const Oracle oracle(f, *p.ffn.softcap);
        return run(f, oracle, p,
                   {.tokens       = 8,
                    .distribution = "pool",
                    .policy       = a16,
                    .output       = DType::FP32,
                    .graph        = true,
                    .per_expert   = true},
                   "custom-caps/mixed");
    });
    return suite.main(argc, argv);
}
