#pragma once

#include "targets/qwen3_6/impl/frontend/processor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>

namespace ninfer::targets::qwen3_6::frontend_internal {

struct MediaCacheKey {
    std::array<std::uint8_t, 32> digest{};
    Modality modality = Modality::Image;

    [[nodiscard]] bool operator==(const MediaCacheKey&) const noexcept = default;
};

struct PreparedMedia {
    VisionItem item;
    std::shared_ptr<const qwen3_6::PreparedMediaPayload> payload;
};

struct MediaCacheFlight;
struct MediaPreparationGate;

class MediaPreparationPermit {
public:
    MediaPreparationPermit() noexcept = default;
    ~MediaPreparationPermit();
    MediaPreparationPermit(MediaPreparationPermit&&) noexcept;
    MediaPreparationPermit& operator=(MediaPreparationPermit&&) noexcept;

    MediaPreparationPermit(const MediaPreparationPermit&)            = delete;
    MediaPreparationPermit& operator=(const MediaPreparationPermit&) = delete;

    void reset() noexcept { release(); }

private:
    explicit MediaPreparationPermit(std::shared_ptr<MediaPreparationGate> gate) noexcept;
    void release() noexcept;
    std::shared_ptr<MediaPreparationGate> gate_;

    friend class MediaPreprocessCache;
};

enum class MediaCacheDisposition : std::uint8_t {
    Hit,
    Producer,
    SingleflightWaiter,
};

struct PendingMedia {
    MediaCacheDisposition disposition = MediaCacheDisposition::Hit;
    PreparedMedia ready;
    std::shared_ptr<MediaCacheFlight> flight;
};

struct MediaCacheRequestStats {
    std::size_t hits               = 0;
    std::size_t misses             = 0;
    std::size_t singleflight_waits = 0;
    std::size_t built_patch_bytes  = 0;
    std::size_t reused_patch_bytes = 0;
    double build_seconds           = 0.0;
};

struct MediaCacheStats {
    std::size_t capacity_bytes       = 0;
    std::size_t live_capacity_bytes  = 0;
    std::size_t retained_bytes       = 0;
    std::size_t live_bytes           = 0;
    std::size_t entries              = 0;
    std::size_t inflight             = 0;
    std::size_t queued_tasks         = 0;
    std::size_t active_tasks         = 0;
    std::uint32_t preprocess_threads = 0;
    std::uint64_t hits               = 0;
    std::uint64_t misses             = 0;
    std::uint64_t singleflight_waits = 0;
    std::uint64_t evictions          = 0;
    std::uint64_t oversize_bypasses  = 0;
};

class MediaPreprocessCache {
public:
    using Builder = std::function<PreparedMedia()>;

    MediaPreprocessCache(std::size_t capacity_bytes, std::size_t live_capacity_bytes,
                         std::uint32_t preprocess_threads  = 0,
                         std::size_t maximum_request_bytes = 0);
    ~MediaPreprocessCache();

    MediaPreprocessCache(const MediaPreprocessCache&)            = delete;
    MediaPreprocessCache& operator=(const MediaPreprocessCache&) = delete;

    [[nodiscard]] std::shared_ptr<qwen3_6::PreparedMediaPayload>
    allocate_payload(std::size_t elements, const PreparationControl& control);

    [[nodiscard]] MediaPreparationPermit acquire_request(const PreparationControl& control) const;

    [[nodiscard]] PendingMedia begin_prepare(const MediaCacheKey& key,
                                             const PreparationControl& control, Builder builder);
    [[nodiscard]] PreparedMedia get_or_prepare(const MediaCacheKey& key,
                                               const PreparationControl& control, Builder builder,
                                               MediaCacheRequestStats& request_stats);
    [[nodiscard]] PreparedMedia await(const PendingMedia& pending,
                                      const PreparationControl& control,
                                      MediaCacheRequestStats& request_stats) const;

    [[nodiscard]] MediaCacheStats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

void check_preparation_control(const PreparationControl& control,
                               std::string_view stage = "preparation");

} // namespace ninfer::targets::qwen3_6::frontend_internal
