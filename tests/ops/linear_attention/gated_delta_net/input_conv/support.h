#pragma once

#include "core/platform.h"

#include "ninfer/ops/gdn_input_proj.h"
#include "ops/input_projection_test_common.h"
#include "ops/suite.h"

#include <optional>

namespace ninfer::test::gdn_input_conv {

struct Profile {
    QType format;
    int hidden;
    const char* name;

    int qk_rows = 2048;

    int value_rows() const { return qk_rows == 1024 ? 3072 : (hidden == 2048 ? 4096 : 6144); }
};

inline constexpr Profile profiles[]{{QType::Q4_G64_FP16, 5120, "q4_q5"},
                                    {QType::Q8_G32_FP16, 2048, "q8"},
                                    {QType::Q8_G32_FP16, 2560, "q8"},
                                    {QType::NVFP4, 5120, "nvfp4"},
                                    {QType::FP8_E4M3FN_ROW_BF16, 5120, "fp8"}};

inline constexpr Profile local_profile{QType::Q8_G32_FP16, 2560, "q8_local", 1024};

const char* policy_name(ops::LinearPolicy policy);

struct Weights {
    Weights(Profile profile, bool record);
    ops::ProjectionWeights projection(ops::LinearPolicy policy) const;
    double project(int row, const float* input) const;
    int preserved() const;
    Profile profile;
    input_projection::DevicePackedWeight first;
    std::optional<input_projection::DevicePackedWeight> second;
};

struct SnapshotCase {
    int width;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
    int initial              = -1;
    bool graph               = false;
    bool full_reference      = false;
    int batch                = 0; // Zero selects the single-request state form.
    std::vector<int> valid;
    unsigned conv_seed = 0;
};

struct RecordCase {
    int width, batch;
    std::vector<int> valid;
    unsigned seed;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
};

int run_snapshot(Weights& weights, const SnapshotCase& c);
int snapshot_capacity_contract();
int run_record(Weights& weights, const RecordCase& c);

} // namespace ninfer::test::gdn_input_conv
