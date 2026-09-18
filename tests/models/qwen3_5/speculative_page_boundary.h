#pragma once

#include "ninfer/engine.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace ninfer::test {

// Exercise the same resource transition for DFlash's Full KV and DFlash2's cyclic-only state.
inline void speculative_page_boundary(Engine& engine) {
    const auto check = [](bool condition, const char* message) {
        if (!condition) { throw std::runtime_error(message); }
    };
    const auto request = [](bool reuse) {
        RequestOptions options;
        options.execution.requested_output_tokens = 16;
        options.execution.sampling.temperature    = 0.0F;
        options.execution.allow_prefix_reuse      = reuse;
        options.stop.include_model_defaults       = false;
        return options;
    };

    auto prompt = engine.tokenize_text("Count from one to twenty: one, two, three,");
    check(prompt.size() <= 63, "page-boundary prompt exceeds its fixed prefix");
    prompt.insert(prompt.begin(), 63 - prompt.size(), 198);
    const auto reference = engine.generate(engine.prepare_tokens(prompt), request(false));
    check(reference.generated_token_ids.size() == 16 &&
              reference.generated_token_ids[0] != reference.generated_token_ids[1],
          "page-boundary fixture must expose a distinct second output token");

    // Begin samples one token at E=63. Verify then maps past 64, but this stop commits only
    // one target column, ending at E=64. A full output budget keeps the verify window open.
    for (const bool reuse : {false, true}) {
        auto stopped_options = request(reuse);
        stopped_options.stop.token_ids.push_back(reference.generated_token_ids[1]);
        const auto stopped = engine.generate(engine.prepare_tokens(prompt), stopped_options);
        check(stopped.finish_reason == FinishReason::StopToken &&
                  stopped.generated_token_ids.size() == 2 && stopped.speculative.rounds == 1 &&
                  stopped.speculative.drafted_tokens > 0 &&
                  std::equal(stopped.generated_token_ids.begin(), stopped.generated_token_ids.end(),
                             reference.generated_token_ids.begin()),
              "cross-page speculative verify did not stop at the preceding page boundary");
    }

    auto followup = prompt;
    followup.insert(followup.end(), reference.generated_token_ids.begin(),
                    reference.generated_token_ids.begin() + 2);
    followup.push_back(198);
    const auto reused = engine.generate(engine.prepare_tokens(followup), request(true));
    const auto fresh  = engine.generate(engine.prepare_tokens(followup), request(false));
    check(reused.reused_prompt_tokens == 64,
          "page-boundary terminal did not retain its exact committed frontier");
    // The resource oracle is the exact retained frontier and successful subsequent execution.
    // Reuse and full prefill have different floating-point paths, so long generated text is not
    // an exact state oracle. Numerical and replay-state correctness have their own Op tests.
    check(reused.generated_token_ids.size() == 16 && fresh.generated_token_ids.size() == 16 &&
              reused.finish_reason == FinishReason::OutputLimit &&
              fresh.finish_reason == FinishReason::OutputLimit,
          "generation after page-boundary settlement did not complete");
}

} // namespace ninfer::test
