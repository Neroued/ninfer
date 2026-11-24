#pragma once

#include "core/buffer.h"
#include "ninfer/ops/moe.h"
#include "ops/op_tester.h"
#include "ops/suite.h"
#include "ops/quantized_weight.h"

#include <array>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::test::moe {

inline constexpr int kHidden = 2048, kIntermediate = 512, kExperts = 256, kTopK = 8;
inline constexpr ReductionCriterion kCompleteA16{1.2e-2, 4.0e-3, 0.0};
inline constexpr ReductionCriterion kContributionA16{1.2e-2, 4.0e-3, 0.0};
inline constexpr ReductionCriterion kRouteAccuracy{3.0e-5, 5.0e-6, 0.0};

struct HostExpert {
    int id;
    quantized_weight::PackedWeight gate_up, down;
};

struct RouteReference {
    std::array<int, kTopK> ids;
    std::array<double, kTopK> coefficients;
    double shared;
};

struct LogicalCase {
    std::vector<float> input, residual;
    RouteReference route;
    std::vector<int> expert_ids;
    std::vector<std::vector<double>> expert_values;
    std::vector<double> shared;
    std::vector<double> complete;
    const std::vector<double>& expert(int id) const;
};

struct Batch {
    std::vector<std::uint16_t> input, residual;
    std::vector<int> ids, patterns;
    std::vector<float> coefficients, shared_coefficients;
    std::vector<double> complete;
};

class Fixture {
public:
    Fixture(QType gate_up, QType down);
    ~Fixture();
    const ops::MoeProfile& profile() const;
    const ops::MoeWeights& weights() const;
    std::span<const HostExpert> experts() const;
    const LogicalCase& logical(int pattern) const;
    Batch batch(int tokens, int first_pattern = 0) const;
    std::vector<double> contribution(const Batch& batch, ops::ExpertSubsetKind subset,
                                     std::span<const int> mask, bool shared, bool residual = false,
                                     std::span<const int> weight_aliases = {}) const;
    int verify_preserved() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class DeviceBatch {
public:
    explicit DeviceBatch(const Batch& batch);
    Tensor x, residual, shared;
    ops::ExpertRoutes routes;

private:
    DeviceBuffer input_, residual_, ids_, coefficients_, shared_;
};

class HostBank {
public:
    explicit HostBank(const Fixture& fixture);
    ~HostBank();
    HostBank(const HostBank&)            = delete;
    HostBank& operator=(const HostBank&) = delete;

    void register_upload(Device device);
    [[nodiscard]] ExpertBankSource source() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct FixtureCase {
    OpCase metadata;
    std::function<int(Fixture&, CudaExecutionView)> run;
};

void add_complete_cases(std::vector<FixtureCase>& cases, QType gate_up, QType down);
void add_route_cases(std::vector<FixtureCase>& cases);
void add_routed_cases(std::vector<FixtureCase>& cases, QType gate_up);
int test_weights();

} // namespace ninfer::test::moe
