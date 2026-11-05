#pragma once

#include "core/buffer.h"
#include "ninfer/ops/moe.h"
#include "ops/quantized_weight.h"

#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::test {
class OpSuite;

namespace native_moe {

enum class Profile { MainTp, MainEp, MtpTp, MtpEp };
enum class ScalePattern { Typical, Stress };

struct WeightBanks {
    quantized_weight::PackedWeightBank gate_up, down;
};

Profile parse_profile(std::string_view);
std::string_view profile_name(Profile);

class Fixture {
public:
    static constexpr int hidden = 4096, experts_count = 288, top_k = 8, shared_intermediate = 1024;
    Fixture(CudaExecutionView, Profile, int max_tokens, int rank = 0,
            ScalePattern = ScalePattern::Typical, std::optional<WeightBanks> = {});
    ~Fixture();
    Fixture(const Fixture&)            = delete;
    Fixture& operator=(const Fixture&) = delete;

    const Profile profile;
    const ScalePattern scale_pattern;
    const int intermediate, rank, max_tokens;
    bool main() const;
    bool ep() const;
    void input(int tokens, std::string_view distribution, int phase = 0, bool dense = false,
               std::string_view locality = "mixed", int expert_offset = 0, float input_scale = 1.F);
    void publish_routes(bool signed_coefficients = false);
    void release_host_weights();
    Tensor x(int tokens) const;
    ops::ExpertRoutes routes(int tokens) const;
    ops::ExpertSubset subset() const;
    ops::RoutedSwiGLUProfile routed_profile() const;
    ops::SigmoidMoeProfile complete_profile() const;
    ops::CudaSwiGLUExperts expert_weights() const;
    ops::SigmoidMoeWeights complete_weights() const;
    const std::vector<std::uint16_t>& input_words() const;
    const std::vector<int>& route_ids() const;
    const std::vector<float>& route_coefficients() const;
    std::vector<int> local_counts() const;
    std::uint64_t active_weight_bytes() const;
    std::uint64_t router_bytes() const;
    std::uint64_t shared_bytes() const;
    bool resident(int expert) const;
    int verify_input() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class Oracle;
};

class Oracle {
public:
    explicit Oracle(const Fixture&);
    ~Oracle();
    double output(int row, int token, bool complete, std::span<const int> mask = {});
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

void add_native_cases(OpSuite&);
} // namespace native_moe
} // namespace ninfer::test
