#include "ninfer/ops/context_bucket.h"
#include "context_bucket_workload.h"
#include "ops/common/host/graph_profile.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninfer::ops;
using namespace ninfer::bench::workload;

static_assert(select_context_bucket(0) == ContextBucket::K2);
static_assert(context_bucket_tokens(ContextBucket::M1) == 1048576U);
static_assert(context_bucket_name(ContextBucket::K32) == "32k");
static_assert([] {
    unsigned count = 0;
    ContextBucketRange::through(8193).for_each([&](ContextBucket) { ++count; });
    return count == 3;
}());

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

template <class Fn>
void expect_invalid(Fn&& fn, std::string_view message) {
    try {
        fn();
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(std::string(message));
}

void check_buckets() {
    constexpr std::array<std::uint32_t, 9> limits{
        2048, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
    };
    constexpr std::array<std::string_view, 9> names{
        "2k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1m",
    };
    require(kContextBucketTokens == limits, "bucket token table");
    for (std::size_t i = 0; i < limits.size(); ++i) {
        const auto bucket  = kAllContextBuckets.at(i);
        const auto minimum = i == 0 ? 0U : limits[i - 1] + 1;
        require(valid_context_bucket(bucket) && static_cast<std::size_t>(bucket) == i,
                "bucket ordinal and validity");
        require(context_bucket_tokens(bucket) == limits[i] &&
                    context_bucket_name(bucket) == names[i],
                "bucket tokens and name");
        require(parse_context_bucket(names[i]) == bucket, "bucket name round trip");
        require(select_context_bucket(minimum) == bucket &&
                    select_context_bucket(limits[i]) == bucket,
                "bucket selection at both inclusive boundaries");
    }
    for (const auto text : {"", "2K", "2048", "2k ", "4k", "1M"}) {
        expect_invalid([&] { (void)parse_context_bucket(text); }, "accepted a noncanonical bucket");
    }
    const auto invalid = static_cast<ContextBucket>(255);
    require(!valid_context_bucket(invalid), "invalid bucket ordinal");
    expect_invalid([&] { (void)context_bucket_tokens(invalid); }, "invalid bucket token lookup");
    expect_invalid([&] { (void)context_bucket_name(invalid); }, "invalid bucket name lookup");
    expect_invalid([] { (void)select_context_bucket(1048577); }, "context beyond 1m was accepted");

    const auto defaults = ContextBucketRange{};
    require(defaults.first == ContextBucket::K2 && defaults.last == ContextBucket::K2,
            "default range must be a singleton");
    const auto single = ContextBucketRange{ContextBucket::K64};
    require(single.first == ContextBucket::K64 && single.last == ContextBucket::K64,
            "explicit singleton");
    const auto through = ContextBucketRange::through(2049);
    require(through.first == ContextBucket::K2 && through.last == ContextBucket::K8,
            "range must cover every bucket through the ceiling");
    const ContextBucketRange range{ContextBucket::K32, ContextBucket::K128};
    range.validate(ContextBucket::K128);
    std::vector<ContextBucket> visited;
    range.for_each([&](ContextBucket bucket) { visited.push_back(bucket); });
    require(visited == std::vector{ContextBucket::K32, ContextBucket::K64, ContextBucket::K128},
            "inclusive range order");
    expect_invalid([&] { range.validate(ContextBucket::K64); }, "unsupported range was accepted");
    expect_invalid([&] { range.validate(invalid); }, "invalid support ceiling was accepted");
    for (auto invalid_range : {ContextBucketRange{invalid, ContextBucket::M1},
                               ContextBucketRange{ContextBucket::K2, invalid},
                               ContextBucketRange{ContextBucket::K8, ContextBucket::K2}}) {
        expect_invalid([&] { invalid_range.for_each([](ContextBucket) {}); },
                       "invalid range was visited");
    }
}

void check_profile() {
    using ninfer::ops::detail::make_op_graph_profile;
    std::array<std::uint64_t, 24> words{17, 23};
    const auto saved = make_op_graph_profile(words);
    require(saved == make_op_graph_profile(std::span{words}.first(2)),
            "profile pads omitted words with zeros");
    require(OpGraphProfile{} == make_op_graph_profile({}), "empty profile is zero initialized");
    words.back() = std::numeric_limits<std::uint64_t>::max();
    require(saved != make_op_graph_profile(words), "profile must copy and compare all 24 words");
    expect_invalid([] { (void)make_op_graph_profile(std::array<std::uint64_t, 25>{}); },
                   "oversize profile was accepted");
}

void check_workload_names() {
    require(parse_length_dist("uniform") == LengthDist::Uniform &&
                parse_length_dist("skewed-1long") == LengthDist::Skewed1Long &&
                parse_length_dist("staggered") == LengthDist::Staggered &&
                parse_length_dist("shared-prefix") == LengthDist::SharedPrefix,
            "length distribution parsing");
    require(parse_valid_dist("full") == ValidDist::Full &&
                parse_valid_dist("ragged") == ValidDist::Ragged &&
                parse_valid_dist("with-inert") == ValidDist::WithInert,
            "validity distribution parsing");
    for (auto dist : {LengthDist::Uniform, LengthDist::Skewed1Long, LengthDist::Staggered,
                      LengthDist::SharedPrefix}) {
        require(parse_length_dist(length_dist_name(dist)) == dist, "length distribution name");
    }
    for (auto dist : {ValidDist::Full, ValidDist::Ragged, ValidDist::WithInert}) {
        require(parse_valid_dist(valid_dist_name(dist)) == dist, "validity distribution name");
    }
    require(!parse_bucket_override("auto") && parse_bucket_override("32k") == ContextBucket::K32,
            "bucket override parsing");
    expect_invalid([] { (void)parse_length_dist("skewed"); }, "unknown length distribution");
    expect_invalid([] { (void)parse_valid_dist("FULL"); }, "unknown validity distribution");
    expect_invalid([] { (void)parse_bucket_override("Auto"); }, "unknown bucket override");
}

void check_workloads() {
    const auto uniform = generate_batch_workload(2049, 4, 3, LengthDist::Uniform, ValidDist::Full);
    require(uniform.valid_columns == std::vector{4, 4, 4} &&
                uniform.row_contexts == std::vector{2049, 2049, 2049} &&
                uniform.shared_prefix_tokens == 0 && uniform.max_visible_tokens == 2049 &&
                uniform.natural_bucket == ContextBucket::K8,
            "uniform workload must use the requested context and physical columns");

    const auto skewed =
        generate_batch_workload(4096, 8, 9, LengthDist::Skewed1Long, ValidDist::Ragged);
    require(skewed.row_contexts == std::vector{4096, 512, 1024, 384, 1536, 768, 256, 1280, 512},
            "skewed workload must cycle through the short contexts");
    const auto [min_valid, max_valid] =
        std::minmax_element(skewed.valid_columns.begin(), skewed.valid_columns.end());
    require(skewed.valid_columns.size() == 9 && skewed.valid_columns.front() == 8 &&
                *min_valid >= 1 && *min_valid < *max_valid && *max_valid == 8,
            "ragged workload must mix live lengths and keep row zero full");
    const auto repeated =
        generate_batch_workload(4096, 8, 9, LengthDist::Skewed1Long, ValidDist::Ragged);
    require(skewed.valid_columns == repeated.valid_columns &&
                skewed.row_contexts == repeated.row_contexts,
            "workloads must be deterministic");
    const auto clamped =
        generate_batch_workload(1000, 800, 4, LengthDist::Skewed1Long, ValidDist::Full);
    require(clamped.row_contexts == std::vector{1000, 800, 1000, 800},
            "short context must include valid columns and fit the target");

    const auto staggered =
        generate_batch_workload(1048576, 4, 4096, LengthDist::Staggered, ValidDist::WithInert);
    require(staggered.row_contexts.front() == 1048576 && staggered.row_contexts[2048] == 524288 &&
                staggered.row_contexts.back() == 0 && staggered.valid_columns.back() == 0 &&
                staggered.valid_columns[4094] > 0 && staggered.natural_bucket == ContextBucket::M1,
            "staggered contexts and inert row at a context/batch product above INT32_MAX");

    const auto shared =
        generate_batch_workload(1000, 4, 5, LengthDist::SharedPrefix, ValidDist::WithInert, 128);
    require(shared.shared_prefix_tokens == 640 && shared.row_contexts.front() == 1000 &&
                shared.row_contexts[1] < 1000 && shared.row_contexts[1] > shared.row_contexts[3] &&
                shared.row_contexts[3] >= 640 && shared.row_contexts.back() == 0 &&
                shared.max_visible_tokens == 1000,
            "shared prefix alignment and varying live suffixes");
    require(generate_batch_workload(127, 4, 1, LengthDist::SharedPrefix, ValidDist::WithInert)
                        .shared_prefix_tokens == 0 &&
                generate_batch_workload(128, 4, 1, LengthDist::SharedPrefix, ValidDist::WithInert)
                        .shared_prefix_tokens == 64,
            "a shared prefix requires context for at least two pages");
    const auto single =
        generate_batch_workload(32, 4, 1, LengthDist::Uniform, ValidDist::WithInert);
    require(single.valid_columns == std::vector{4} && single.row_contexts == std::vector{32},
            "a singleton with-inert workload must keep its row live");

    const auto empty = generate_batch_workload(0, 0, 2, LengthDist::Staggered, ValidDist::Ragged);
    require(empty.valid_columns == std::vector{0, 0} && empty.row_contexts == std::vector{0, 0} &&
                empty.max_visible_tokens == 0 && empty.natural_bucket == ContextBucket::K2,
            "zero-column workload");
    const auto no_rows = generate_batch_workload(8192, 4, 0, LengthDist::Uniform, ValidDist::Full);
    require(no_rows.valid_columns.empty() && no_rows.row_contexts.empty() &&
                no_rows.max_visible_tokens == 0 && no_rows.natural_bucket == ContextBucket::K2,
            "zero-row workload");
    expect_invalid(
        [] { (void)generate_batch_workload(2, 4, 1, LengthDist::Uniform, ValidDist::Full); },
        "target shorter than a full query row");
    expect_invalid(
        [] { (void)generate_batch_workload(1048577, 4, 1, LengthDist::Uniform, ValidDist::Full); },
        "target above largest bucket");
    expect_invalid(
        [] {
            (void)generate_batch_workload(128, 4, 1, LengthDist::SharedPrefix, ValidDist::Full, 0);
        },
        "zero page size");
}

} // namespace

int main() {
    try {
        check_buckets();
        check_profile();
        check_workload_names();
        check_workloads();
    } catch (const std::exception& error) {
        std::cerr << "context bucket test failed: " << error.what() << '\n';
        return 1;
    }
}
