#pragma once

#include "ops/moe/native_fixture.h"

#include <filesystem>

namespace ninfer::test::native_moe {
WeightBanks read_artifact_weights(const std::filesystem::path&, Profile, int rank, int layer);
}
