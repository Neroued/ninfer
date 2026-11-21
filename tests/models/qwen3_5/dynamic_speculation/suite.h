#pragma once

#include "models/qwen3_5/native_fixture.h"
#include "models/common/qwen/frontend/frontend.h"

namespace ninfer::test::qwen3_5::dynamic {
namespace q  = models::qwen3_5;
namespace sp = models::speculative;

inline void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

inline constexpr unsigned capacity = 256;
void exercise_rounds(NativeRankFixture&, models::qwen::Frontend&, const EngineOptions&);
} // namespace ninfer::test::qwen3_5::dynamic
