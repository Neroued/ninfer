#include "models/common/graph/patch_site.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::models {
namespace {

constexpr auto missing_patch = std::numeric_limits<std::uint8_t>::max();
static_assert(ops::kContextBucketCount < missing_patch);

} // namespace

ContextBucketSite::ContextBucketSite(ops::ContextBucketRange buckets, CudaExecutionView cuda,
                                     const Profile& profile, const Capture& capture)
    : first_(buckets.first) {
    bucket_to_patch_.fill(missing_patch);
    std::vector<ops::OpGraphProfile> profiles;
    buckets.for_each([&](ops::ContextBucket bucket) {
        const auto key  = profile(bucket);
        const auto next = std::ranges::find(profiles, key) - profiles.begin();
        if (static_cast<std::size_t>(next) == profiles.size()) {
            auto& patch = patches_.emplace_back();
            patch.capture(cuda.stream(), [&] { capture(bucket, cuda); });
            if (patch.params().size() != patches_.front().params().size())
                throw std::logic_error("context bucket changed the kernel slice length");
            profiles.push_back(key);
        }
        bucket_to_patch_[static_cast<std::size_t>(bucket)] = static_cast<std::uint8_t>(next);
    });
}

void ContextBucketSite::record(CudaExecutionView cuda, const Capture& capture) {
    slice_.capture(cuda.stream(), [&] { capture(first_, cuda); });
    if (slice_.size() != patches_.front().params().size())
        throw std::logic_error("inline context bucket slice differs from its patch");
    installed_ = 0;
}

void ContextBucketSite::record(CudaExecutionView cuda, const Capture& capture,
                               const moe::MixerSliceRecorder& recorder) const {
    if (recorder([&] { capture(first_, cuda); }) != patches_.front().params().size())
        throw std::logic_error("MoE context bucket slice differs from its patch");
}

std::uint8_t ContextBucketSite::lookup(ops::ContextBucket bucket) const {
    const auto next = bucket_to_patch_.at(static_cast<std::size_t>(bucket));
    if (next == missing_patch) throw std::out_of_range("context bucket exceeds captured range");
    return next;
}

void ContextBucketSite::install(DecodeGraphExecutable& executable, std::uint8_t next) {
    if (next == installed_) return;
    executable.update_slice(slice_, patches_[next]);
    installed_ = next;
}

void ContextBucketSite::install(moe::LayerGraphExecutable& executable, std::size_t slice,
                                std::uint8_t next) {
    if (next == installed_) return;
    executable.update_mixer_slice(slice, patches_[next]);
    installed_ = next;
}

void ContextBucketSite::select(DecodeGraphExecutable& executable, ops::ContextBucket bucket) {
    install(executable, lookup(bucket));
}

void ContextBucketSite::select(moe::LayerGraphExecutable& executable, std::size_t slice,
                               ops::ContextBucket bucket) {
    install(executable, slice, lookup(bucket));
}

SlotPatchSite::SlotPatchSite(CudaExecutionView cuda, const Capture& capture) {
    for (std::uint32_t slot = 0; slot < definitions_.size(); ++slot)
        definitions_[slot].capture(cuda.stream(), [&] { capture(slot, cuda); });
}

void SlotPatchSite::append(cudaStream_t stream) {
    if (appended()) throw std::logic_error("slot patch site is already appended");
    child_ = definitions_.front().append_to(stream);
}

void SlotPatchSite::select(DecodeGraphExecutable& executable, std::uint32_t slot) {
    if (slot == installed_slot_) return;
    executable.update_child(child_, definitions_.at(slot));
    installed_slot_ = slot;
}

void GraphPatchCalibration::add(DecodeGraphExecutable& executable,
                                std::span<ContextBucketSite> sites) {
    for (auto& site : sites)
        if (site.variant_count() > 1)
            sites_.push_back({site.variant_count(),
                              [&executable, &site](auto next) { site.install(executable, next); }});
}

void GraphPatchCalibration::add(DecodeGraphExecutable& executable, SlotPatchSite& site) {
    sites_.push_back({2, [&executable, &site](auto next) { site.select(executable, next); }});
}

void GraphPatchCalibration::add(moe::LayerGraphExecutable& executable,
                                std::span<ContextBucketSite> sites) {
    for (std::size_t i = 0; i < sites.size(); ++i)
        if (sites[i].variant_count() > 1)
            sites_.push_back(
                {sites[i].variant_count(), [&executable, &site = sites[i], i](auto next) {
                     site.install(executable, i, next);
                 }});
}

void GraphPatchCalibration::run(GraphPreparation& preparation,
                                const std::function<void()>& upload) {
    if (!preparation.capturing() || sites_.empty()) return;
    const auto rounds = std::ranges::max_element(sites_, {}, &Site::variants)->variants;
    for (std::uint8_t next = 1; next < rounds; ++next) {
        for (const auto& site : sites_)
            if (next < site.variants) site.install(next);
        upload();
        preparation.checkpoint();
    }
    for (const auto& site : sites_) site.install(0);
    upload();
    preparation.checkpoint();
}

std::vector<ContextBucketSite> prepare_readouts(CudaExecutionView cuda,
                                                std::span<const ContextBucketBinding> bindings) {
    std::vector<ContextBucketSite> sites;
    sites.reserve(bindings.size());
    for (const auto& binding : bindings)
        sites.emplace_back(binding.buckets, cuda, binding.profile, binding.capture);
    return sites;
}

void record_readouts(CudaExecutionView cuda, std::span<ContextBucketSite> sites,
                     const ReadoutBody& body, const moe::MixerSliceRecorder* mixer) {
    std::size_t next = 0;
    body([&](const ContextBucketSite::Capture& capture) {
        if (next == sites.size()) throw std::logic_error("unbound graph readout");
        auto& site = sites[next++];
        if (mixer)
            site.record(cuda, capture, *mixer);
        else
            site.record(cuda, capture);
    });
    if (next != sites.size()) throw std::logic_error("graph readouts were not consumed");
}

} // namespace ninfer::models
