#pragma once
#include "ninfer/ops/moe.h"
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::test::clamped_ep {
inline constexpr int H = 4096, I = 2048, E = 288, Local = 144, K = 8;
enum class Placement { Contiguous, Interleaved, Permuted };

// Test/benchmark-owned encoded weights, exactly 144 actual payloads per projection.
// The producer and independent host oracle compile as C++, outside the CUDA benchmark TU.
class Weights {
public:
    Weights(Device device, Placement placement, int side, bool fp8 = false);
    ~Weights();
    Weights(const Weights&)            = delete;
    Weights& operator=(const Weights&) = delete;
    ops::RoutedSwiGLUProfile profile() const;
    ops::CudaSwiGLUExperts experts() const;
    ops::ExpertSubset subset() const;
    void publish(int index_phase = 0, bool drop_half = false);
    void set_limit(float limit);
    void clamp_probe(int recipe, bool norm_anchor = false);
    bool active(int global_id) const;
    int local_id(int ordinal) const;
    int remote_id(int ordinal) const;
    int side() const;
    Placement placement() const;
    std::size_t encoded_expert_bytes() const;
    std::size_t backing_bytes() const;
    const std::vector<double>& oracle(int pattern, int global_id, std::span<const float> x);
    int verify_controls() const;
    int verify_payloads() const;
    void release_host_oracle();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct Routes {
    std::vector<int> ids;
    std::vector<float> coefficients;
    std::uint64_t hits = 0, active_tokens = 0;
    std::vector<int> per_expert;
};

Routes make_routes(const Weights&, int tokens, const char* distribution, int fixed_hits = -1);
} // namespace ninfer::test::clamped_ep
