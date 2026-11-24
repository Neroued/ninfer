#pragma once

#include "op_measurement.cuh"
#include "ninfer/ops/gdn_replay.h"

namespace ninfer::bench::gdn_replay {
struct Profile {
    int layers = 36, qk_heads = 16, value_heads = 48, conv_channels = 10240;
};

struct Options {
    measurement::Options timing;
    Profile profile;
    std::string op = "fold", commits = "dense", valid = "dense";
    std::string row_metadata = "host";
    std::vector<int> tokens{16};
    int batch = 1;
};

inline Options parse_options(int argc, char** argv) {
    Options o;
    bool explicit_tokens = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            std::cout
                << "GDN Replay benchmark\n"
                   "  --op fold|record --layers L --qk-heads Q --value-heads H --conv-channels C\n"
                   "  --tokens T[,T...] or --sweep START:END[:STEP] --batch B\n"
                   "  --commits dense|one|mixed|N[,N...] --valid dense|mixed|N[,N...]\n"
                   "  --row-metadata host|device (fold only, default host)\n"
                   "Head dimension=128, T=2..16, B=1..8.\n"
                   "Fold times all specified layers; record times one recurrence layer.\n";
            measurement::help();
            std::exit(0);
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--op")
            o.op = value;
        else if (flag == "--layers")
            o.profile.layers = measurement::integer(value);
        else if (flag == "--qk-heads")
            o.profile.qk_heads = measurement::integer(value);
        else if (flag == "--value-heads")
            o.profile.value_heads = measurement::integer(value);
        else if (flag == "--conv-channels")
            o.profile.conv_channels = measurement::integer(value);
        else if (flag == "--tokens") {
            if (explicit_tokens) throw std::invalid_argument("specify one token extent or sweep");
            o.tokens        = measurement::integers(value);
            explicit_tokens = true;
        } else if (flag == "--sweep") {
            if (explicit_tokens) throw std::invalid_argument("specify one token extent or sweep");
            o.tokens        = measurement::sweep(value);
            explicit_tokens = true;
        } else if (flag == "--batch")
            o.batch = measurement::integer(value);
        else if (flag == "--commits")
            o.commits = value;
        else if (flag == "--valid")
            o.valid = value;
        else if (flag == "--row-metadata") {
            if (value != "host" && value != "device")
                throw std::invalid_argument("row-metadata: host|device");
            o.row_metadata = value;
        } else if (!measurement::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option: " + std::string(flag));
    }
    if (o.op != "fold" && o.op != "record") throw std::invalid_argument("unknown Replay Op");
    if (o.op != "fold" && o.row_metadata != "host")
        throw std::invalid_argument("row-metadata applies to fold");
    if (o.batch > 8) throw std::invalid_argument("B must be <=8");
    for (int t : o.tokens)
        if (t < 2 || t > 16) throw std::invalid_argument("T must be in [2,16]");
    return o;
}

inline std::vector<int> extents(const std::string& selection, int width, int batch, bool commits) {
    std::vector<int> result;
    if (selection == "dense")
        result.assign(batch, width);
    else if (selection == "one" && commits)
        result.assign(batch, 1);
    else if (selection == "mixed") {
        constexpr int pattern[] = {0, 1, 2, 3, 16, 7, 12, 5};
        for (int b = 0; b < batch; ++b)
            result.push_back(commits ? std::min(width, pattern[b])
                                     : std::max(1, width - b % width));
    } else
        result = measurement::integers(selection, commits ? 0 : 1);
    if (result.size() != std::size_t(batch) ||
        *std::max_element(result.begin(), result.end()) > width)
        throw std::invalid_argument("prefix extents must match T/B");
    return result;
}

inline std::string csv_extent(std::string text) {
    std::replace(text.begin(), text.end(), ',', ':');
    return text;
}
} // namespace ninfer::bench::gdn_replay
