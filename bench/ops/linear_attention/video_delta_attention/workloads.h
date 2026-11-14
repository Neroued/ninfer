#pragma once

#include "op_measurement.cuh"
#include "ninfer/ops/video_delta_attention.h"

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::bench::vda {

struct Workload {
    std::string name;
    int video_frames, width, height;
    int frames, tokens_per_frame, text_tokens;
};

// OpenVDN/vdn-minimax-h3, e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b:
// docs/inference.md: 1344x768 canvas; output frames = 17*n+5, latent frames = 5*n+2.
// src/models/sequence_layout.py: 24x42 patched spatial grid.
// src/models/linear_attention/branch.py: prune both anchor frames, preserve offset=1.
// 345 is the published 14.4 s workload; 175 exercises a shorter 7.3 s request.
// 1024 is a representative synthetic prompt length, not a captured prompt trace.
// The documented six-reference example has 6948 text rows. Its 5664 separate
// conditioning rows belong to softmax attention and must not be added to this Op.
inline const std::array<Workload, 3> H3Workloads{{
    {"h3-175", 175, 1344, 768, 50, 1008, 1024},
    {"h3-345", 345, 1344, 768, 100, 1008, 1024},
    {"h3-ref6-345", 345, 1344, 768, 100, 1008, 6948},
}};

struct Options {
    measurement::Options timing;
    std::string op        = "gated";
    std::string case_name = "h3-345";
    std::vector<int> heads{56, 28, 14, 7};
    std::vector<int> custom_frames, custom_tokens;
    std::optional<int> text_override;
    int batch = 1;
    ops::VideoDeltaAttentionWindow window{5, 1, 1};
};

inline Options parse_options(int argc, char** argv) {
    Options options;
    bool named_case = false, custom_geometry = false, explicit_heads = false, explicit_tp = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help") {
            std::cout << "VDN public Op benchmark (synthetic values, H3 request geometry)\n"
                         "  --case h3-175|h3-345|h3-ref6-345|all (default h3-345)\n"
                         "  --tp 1,2,4,8 (default all; local heads=56/TP, no collectives)\n"
                         "  --op raw|gated|state|chain (default gated)\n"
                         "  --batch B --text-tokens L (override each case's prompt length)\n"
                         "  --frames F[,F...] --tokens-per-frame U[,U...] --heads H[,H...]\n"
                         "    Explicit geometry creates custom cases; do not combine with --case.\n"
                         "  --chunk C --radius R --offset O (custom geometry only)\n"
                         "  h3-175: 175 output frames, F=50, U=1008, L=1024\n"
                         "  h3-345: 345 output frames, F=100, U=1008, L=1024\n"
                         "  h3-ref6-345: same video geometry, documented reference prompt L=6948\n"
                         "  raw/gated exclude text init; chain includes init+gated each call.\n"
                         "  Prompt init is per layer/evaluation, not a cross-step cached state.\n";
            measurement::help();
            std::exit(0);
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--op")
            options.op = value;
        else if (flag == "--case") {
            options.case_name = value;
            named_case        = true;
        } else if (flag == "--tp") {
            explicit_tp = true;
            options.heads.clear();
            for (const int tp : measurement::integers(value, 1, 8)) {
                if (tp != 1 && tp != 2 && tp != 4 && tp != 8)
                    throw std::invalid_argument("VDN benchmark: TP must be 1, 2, 4 or 8");
                options.heads.push_back(56 / tp);
            }
        } else if (flag == "--heads") {
            explicit_heads = true;
            options.heads  = measurement::integers(value, 1, 56);
        } else if (flag == "--frames") {
            options.custom_frames = measurement::integers(value);
            custom_geometry       = true;
        } else if (flag == "--tokens-per-frame") {
            options.custom_tokens = measurement::integers(value);
            custom_geometry       = true;
        } else if (flag == "--batch")
            options.batch = measurement::integer(value);
        else if (flag == "--text-tokens")
            options.text_override = measurement::integer(value, 0);
        else if (flag == "--chunk") {
            options.window.chunk_frames = measurement::integer(value);
            custom_geometry             = true;
        } else if (flag == "--radius") {
            options.window.radius = measurement::integer(value, 0);
            custom_geometry       = true;
        } else if (flag == "--offset") {
            options.window.frame_offset = measurement::integer(value, 0);
            custom_geometry             = true;
        } else if (!measurement::parse(options.timing, flag, value)) {
            throw std::invalid_argument("VDN benchmark: unknown option " + std::string(flag));
        }
    }
    if (options.op != "raw" && options.op != "gated" && options.op != "state" &&
        options.op != "chain")
        throw std::invalid_argument("VDN benchmark: invalid op");
    if (explicit_heads && explicit_tp)
        throw std::invalid_argument("VDN benchmark: --heads and --tp are mutually exclusive");
    if (named_case && custom_geometry)
        throw std::invalid_argument(
            "VDN benchmark: named cases and custom geometry are mutually exclusive");
    if (custom_geometry) options.case_name = "custom";
    return options;
}

inline std::vector<Workload> select_workloads(const Options& options) {
    std::vector<Workload> result;
    if (options.case_name == "custom") {
        const auto frames =
            options.custom_frames.empty() ? std::vector<int>{100} : options.custom_frames;
        const auto tokens =
            options.custom_tokens.empty() ? std::vector<int>{1008} : options.custom_tokens;
        for (const int count : frames)
            for (const int per_frame : tokens)
                result.push_back({"custom", 0, 0, 0, count, per_frame, 1024});
    } else {
        for (const auto& workload : H3Workloads)
            if (options.case_name == "all" || options.case_name == workload.name)
                result.push_back(workload);
    }
    if (result.empty())
        throw std::invalid_argument("VDN benchmark: unknown case " + options.case_name);
    if (options.text_override)
        for (auto& workload : result) workload.text_tokens = *options.text_override;
    return result;
}

struct Cost {
    double io_bytes{}, contraction_flops{};
};

inline Cost useful_cost(const Options& options, const Workload& workload, int heads) {
    constexpr double dim = 128;
    const double groups  = double(heads) * options.batch;
    const double frames  = groups * workload.frames;
    const double rows    = frames * workload.tokens_per_frame;
    // One compulsory pass over Q,K,V,(gate),output, alpha, beta and initial state.
    // This is an IO lower bound, NOT actual DRAM traffic or a complete route roofline.
    // Contractions count A,B, two state updates and readout. Inversion and scalar
    // gates/norms are excluded explicitly; executed BF16 residual terms are not useful FLOPs.
    Cost cost{rows * (dim * (options.op == "raw" ? 8 : 10) + 4) + frames * dim * 4 +
                  groups * dim * dim * 4,
              6 * rows * dim * dim + 6 * frames * dim * dim * dim};
    const Cost init{groups * (workload.text_tokens * (dim * 4 + 4) + dim * dim * 4),
                    workload.text_tokens
                        ? groups * (4 * workload.text_tokens * dim * dim + 2 * dim * dim * dim)
                        : 0};
    if (options.op == "state") return init;
    if (options.op == "chain") {
        cost.io_bytes += init.io_bytes;
        cost.contraction_flops += init.contraction_flops;
    }
    return cost;
}

} // namespace ninfer::bench::vda
