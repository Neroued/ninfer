#include "runtime/contract/sampling.h"

#include "models/qwen3_5/frontend/frontend.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

bool same_preset(const ninfer::SamplingPreset& actual, const ninfer::SamplingPreset& expected) {
    return actual.temperature == expected.temperature && actual.top_k == expected.top_k &&
           actual.top_p == expected.top_p && actual.min_p == expected.min_p &&
           actual.presence_penalty == expected.presence_penalty &&
           actual.frequency_penalty == expected.frequency_penalty;
}

bool throws_invalid(const auto& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}


} // namespace

int main() {
    using ninfer::models::qwen3_5::default_sampling;
    int failures        = 0;
    const auto defaults = default_sampling();

    const ninfer::SamplingPreset expected_thinking{
        .temperature = 1.0F, .top_k = 20, .top_p = 0.95F, .min_p = 0.0F};
    const ninfer::SamplingPreset expected_non_thinking{
        .temperature      = 0.7F,
        .top_k            = 20,
        .top_p            = 0.8F,
        .min_p            = 0.0F,
        .presence_penalty = 1.5F,
    };

    failures +=
        check(same_preset(defaults.thinking, expected_thinking), "thinking defaults mismatch");
    failures += check(same_preset(defaults.non_thinking, expected_non_thinking),
                      "non-thinking defaults mismatch");

    const ninfer::ResolvedSamplingParameters thinking = ninfer::runtime::resolve_sampling(
        defaults, ninfer::SamplingMode::Thinking, ninfer::SamplingOverrides{});
    const ninfer::ResolvedSamplingParameters non_thinking = ninfer::runtime::resolve_sampling(
        defaults, ninfer::SamplingMode::NonThinking, ninfer::SamplingOverrides{});
    failures += check(thinking.temperature == 1.0F && thinking.top_p == 0.95F &&
                          thinking.presence_penalty == 0.0F && thinking.seed == 0,
                      "omitted overrides did not select the thinking defaults");
    failures += check(non_thinking.temperature == 0.7F && non_thinking.top_p == 0.8F &&
                          non_thinking.presence_penalty == 1.5F,
                      "omitted overrides did not select the non-thinking defaults");

    ninfer::SamplingOverrides overrides;
    overrides.temperature       = 0.0F;
    overrides.top_k             = 0;
    overrides.top_p             = 0.0F;
    overrides.min_p             = 0.0F;
    overrides.presence_penalty  = 0.0F;
    overrides.frequency_penalty = -1.0F;
    overrides.seed              = 123;
    const ninfer::ResolvedSamplingParameters overridden =
        ninfer::runtime::resolve_sampling(defaults, ninfer::SamplingMode::NonThinking, overrides);
    failures += check(overridden.temperature == 0.0F && overridden.top_k == 20 &&
                          overridden.top_p == 0.0F && overridden.presence_penalty == 0.0F &&
                          overridden.frequency_penalty == -1.0F && overridden.seed == 123,
                      "explicit zero sampling overrides were not normalized to the target cap");

    overrides.top_k = 21;
    failures += check(throws_invalid([&] {
                          (void)ninfer::runtime::resolve_sampling(
                              defaults, ninfer::SamplingMode::Thinking, overrides);
                      }),
                      "top_k beyond the executable candidate domain was accepted");
    overrides.top_k = 0;

    overrides.temperature = std::numeric_limits<float>::quiet_NaN();
    failures += check(throws_invalid([&] {
                          (void)ninfer::runtime::resolve_sampling(
                              defaults, ninfer::SamplingMode::Thinking, overrides);
                      }),
                      "non-finite sampling override was accepted");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
