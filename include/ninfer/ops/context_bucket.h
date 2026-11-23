#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::ops {

enum class ContextBucket : std::uint8_t {
    K2   = 0,
    K8   = 1,
    K16  = 2,
    K32  = 3,
    K64  = 4,
    K128 = 5,
    K256 = 6,
    K512 = 7,
    M1   = 8,
};

inline constexpr std::size_t kContextBucketCount = 9;
inline constexpr std::array<ContextBucket, kContextBucketCount> kAllContextBuckets{
    ContextBucket::K2,   ContextBucket::K8,   ContextBucket::K16,
    ContextBucket::K32,  ContextBucket::K64,  ContextBucket::K128,
    ContextBucket::K256, ContextBucket::K512, ContextBucket::M1,
};
inline constexpr std::array<std::uint32_t, kContextBucketCount> kContextBucketTokens{
    2048U, 8192U, 16384U, 32768U, 65536U, 131072U, 262144U, 524288U, 1048576U,
};

[[nodiscard]] constexpr bool valid_context_bucket(ContextBucket bucket) noexcept {
    return static_cast<std::size_t>(bucket) < kContextBucketCount;
}

[[nodiscard]] constexpr std::uint32_t context_bucket_tokens(ContextBucket bucket) {
    if (!valid_context_bucket(bucket)) throw std::invalid_argument("invalid context bucket");
    return kContextBucketTokens[static_cast<std::size_t>(bucket)];
}

[[nodiscard]] constexpr std::string_view context_bucket_name(ContextBucket bucket) {
    constexpr std::array<std::string_view, kContextBucketCount> names{
        "2k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1m",
    };
    if (!valid_context_bucket(bucket)) throw std::invalid_argument("invalid context bucket");
    return names[static_cast<std::size_t>(bucket)];
}

[[nodiscard]] inline ContextBucket parse_context_bucket(std::string_view text) {
    for (const auto bucket : kAllContextBuckets) {
        if (context_bucket_name(bucket) == text) return bucket;
    }
    throw std::invalid_argument("invalid context bucket: " + std::string(text));
}

// Device metadata gates inert queries. Zero visible tokens uses the smallest bucket.
[[nodiscard]] constexpr ContextBucket select_context_bucket(std::uint32_t visible_tokens) {
    for (const auto bucket : kAllContextBuckets) {
        if (context_bucket_tokens(bucket) >= visible_tokens) return bucket;
    }
    throw std::invalid_argument("visible tokens exceed the largest context bucket");
}

struct ContextBucketRange {
    ContextBucket first = ContextBucket::K2;
    ContextBucket last  = ContextBucket::K2;

    constexpr ContextBucketRange() = default;

    constexpr ContextBucketRange(ContextBucket single) noexcept : first(single), last(single) {}

    constexpr ContextBucketRange(ContextBucket lo, ContextBucket hi) noexcept
        : first(lo), last(hi) {}

    [[nodiscard]] static constexpr ContextBucketRange through(std::uint32_t max_visible_tokens) {
        return {ContextBucket::K2, select_context_bucket(max_visible_tokens)};
    }

    constexpr void validate(ContextBucket max_supported = ContextBucket::M1,
                            const char* op              = "context_bucket") const {
        if (!valid_context_bucket(first) || !valid_context_bucket(last) || first > last ||
            !valid_context_bucket(max_supported) || last > max_supported) {
            throw std::invalid_argument(std::string(op) +
                                        ": invalid or unsupported context bucket range");
        }
    }

    template <class Fn>
    constexpr void for_each(Fn&& fn) const {
        validate();
        for (auto i = static_cast<std::size_t>(first); i <= static_cast<std::size_t>(last); ++i) {
            fn(kAllContextBuckets[i]);
        }
    }
};

class OpGraphProfile;

namespace detail {
[[nodiscard]] inline OpGraphProfile make_op_graph_profile(std::span<const std::uint64_t> words);
} // namespace detail

// Each Op defines what equality guarantees for its captured launches.
class OpGraphProfile {
public:
    bool operator==(const OpGraphProfile&) const = default;

private:
    std::array<std::uint64_t, 24> words_{};
    friend OpGraphProfile detail::make_op_graph_profile(std::span<const std::uint64_t> words);
};

} // namespace ninfer::ops
