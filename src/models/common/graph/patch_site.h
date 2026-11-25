#pragma once

#include "models/common/graph.h"
#include "models/common/moe/schedule.h"
#include "ninfer/ops/context_bucket.h"

#include <array>

namespace ninfer::models {

class ContextBucketSite {
public:
    using Profile = std::function<ops::OpGraphProfile(ops::ContextBucket)>;
    using Capture = std::function<void(ops::ContextBucket, CudaExecutionView)>;

    ContextBucketSite(ops::ContextBucketRange, CudaExecutionView, const Profile&, const Capture&);

    // The inline body uses the same operand bindings as the standalone patch captures.
    void record(CudaExecutionView, const Capture&);
    void record(CudaExecutionView, const Capture&, const moe::MixerSliceRecorder&) const;

    void select(DecodeGraphExecutable&, ops::ContextBucket);
    void select(moe::LayerGraphExecutable&, std::size_t slice, ops::ContextBucket);

    [[nodiscard]] std::size_t variant_count() const noexcept { return patches_.size(); }

private:
    friend class GraphPatchCalibration;
    void install(DecodeGraphExecutable&, std::uint8_t);
    void install(moe::LayerGraphExecutable&, std::size_t slice, std::uint8_t);
    [[nodiscard]] std::uint8_t lookup(ops::ContextBucket) const;

    std::vector<DecodeGraphKernelPatch> patches_;
    DecodeGraphKernelSlice slice_;
    std::array<std::uint8_t, ops::kContextBucketCount> bucket_to_patch_;
    ops::ContextBucket first_;
    std::uint8_t installed_ = 0;
};

class SlotPatchSite {
public:
    using Capture = std::function<void(std::uint32_t, CudaExecutionView)>;

    SlotPatchSite(CudaExecutionView, const Capture&);
    void append(cudaStream_t);
    void select(DecodeGraphExecutable&, std::uint32_t);

    [[nodiscard]] bool appended() const noexcept { return child_.node != nullptr; }

private:
    std::array<DecodeGraphDefinition, 2> definitions_;
    DecodeGraphChild child_;
    std::uint32_t installed_slot_ = 0;
};

// One batch per executable, or mixer/steady pair. The initial capture is already
// uploaded. Exercise all remaining variants and restore the initial configuration.
class GraphPatchCalibration {
public:
    void add(DecodeGraphExecutable&, std::span<ContextBucketSite>);
    void add(DecodeGraphExecutable&, SlotPatchSite&);
    void add(moe::LayerGraphExecutable&, std::span<ContextBucketSite>);

    void run(GraphPreparation&, const std::function<void()>& upload);

private:
    struct Site {
        std::size_t variants;
        std::function<void(std::uint8_t)> install;
    };

    std::vector<Site> sites_;
};

struct ContextBucketBinding {
    ops::ContextBucketRange buckets;
    ContextBucketSite::Profile profile;
    ContextBucketSite::Capture capture;
    std::size_t bucket_offset = 0;
};

using ReadoutRecorder = std::function<void(const ContextBucketSite::Capture&)>;
using ReadoutBody     = std::function<void(const ReadoutRecorder&)>;

[[nodiscard]] std::vector<ContextBucketSite>
    prepare_readouts(CudaExecutionView, std::span<const ContextBucketBinding>);

void record_readouts(CudaExecutionView, std::span<ContextBucketSite>, const ReadoutBody&,
                     const moe::MixerSliceRecorder* = nullptr);

} // namespace ninfer::models
