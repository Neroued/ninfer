#pragma once

#include "core/buffer.h"
#include "ninfer/ops/moe.h"

#include <memory>
#include <string_view>
#include <vector>

namespace ninfer::test::moe_tp2 {

struct Shape {
    int hidden, intermediate, experts, top_k;
    bool clamped;
    QType format;
};

// All logical experts have distinct payloads. Encoded values are sliced from a
// deterministic two-rank source by original matrix coordinates, then packed locally.
class Fixture {
public:
    Fixture(CudaExecutionView execution, Shape shape, int rank, int max_tokens);
    ~Fixture();
    Fixture(const Fixture&)            = delete;
    Fixture& operator=(const Fixture&) = delete;

    const Shape shape;
    const int rank, max_tokens;
    void input(int tokens, std::string_view distribution, int phase = 0, bool stress = false);
    Tensor x(int tokens) const;
    ops::ExpertRoutes routes(int tokens) const;
    Tensor shared_coefficients(int tokens) const;
    ops::RoutedSwiGLUProfile routed_profile() const;
    ops::SigmoidMoeProfile sigmoid_profile() const;
    ops::MoeProfile softmax_profile() const;
    ops::CudaSwiGLUExperts experts() const;
    ops::SigmoidMoeWeights sigmoid_weights() const;
    ops::MoeWeights softmax_weights() const;

    std::size_t encoded_expert_bytes() const;
    std::size_t router_bytes() const;
    std::size_t shared_bytes() const;
    const std::vector<int>& expected_ids() const;
    const std::vector<std::uint16_t>& input_words() const;
    // Publish coefficients directly, without a router, for standalone tests/benchmarks.
    void publish_routes(int tokens, bool signed_coefficients = false);
    std::vector<int> check_routes(int tokens, MemoryArena&, CudaExecutionView);

    struct MatrixSnapshot {
        int n, k, block;
        QType format;
        float divisor;
        std::vector<std::uint8_t> codes, scales;
    };

    MatrixSnapshot snapshot(bool gate, int expert) const;
    MatrixSnapshot shared_snapshot(bool gate) const;
    std::vector<std::uint16_t> control_words() const;
    std::vector<float> bias_words() const;
    int verify_source_coordinates() const;
    void preserve_weights();
    int verify_preserved() const;
    void zero_routed();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Strict C++ oracle independently decodes the represented fixture and evaluates
// the complete formula in FP64; it never reproduces private activation quantization.
class Oracle {
public:
    explicit Oracle(const Fixture&);
    ~Oracle();
    std::pair<std::vector<int>, std::vector<double>> routing(int token);
    double output(int row, int token, bool complete, bool signed_coefficients = false);
    double shared_coefficient(int token);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::test::moe_tp2
