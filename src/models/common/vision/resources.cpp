#include "models/common/vision/resources.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::models::vision {

std::size_t Capacity::device_bytes() const {
    if (encoder.workspace_bytes > std::numeric_limits<std::size_t>::max() - output_bytes)
        throw std::overflow_error("Vision execution capacity overflows size_t");
    return encoder.workspace_bytes + output_bytes;
}

Capacity plan_capacity(std::size_t budget, std::uint32_t maximum, std::int32_t width,
                       const std::function<EncoderMemory(std::uint32_t)>& query) {
    if (!budget || !maximum || maximum > INT32_MAX || width <= 0)
        throw std::invalid_argument(
            "Vision budget, feature bound and output width must be positive");
    const auto candidate = [&](std::uint32_t n) {
        const auto bytes =
            Tensor::metadata(DType::BF16, {width, static_cast<std::int32_t>(n)}).bytes();
        return Capacity{budget, n, width, query(n), bytes};
    };
    auto selected = candidate(1);
    if (selected.device_bytes() > budget)
        throw std::invalid_argument(
            "Vision memory budget cannot hold one feature and its encoder workspace");
    // Bracket the budget from below. A large text context is not an encoder request;
    // querying its midpoint would probe projection extents the budget never permits.
    std::uint32_t last = 1;
    while (last < maximum) {
        last      = static_cast<std::uint32_t>(std::min<std::uint64_t>(2ULL * last, maximum));
        auto next = candidate(last);
        if (next.device_bytes() > budget) break;
        selected = next;
    }
    if (selected.max_features == maximum) return selected;
    std::uint32_t first = selected.max_features + 1;
    --last;
    while (first <= last) {
        const auto middle = first + (last - first) / 2;
        auto next         = candidate(middle);
        if (next.device_bytes() <= budget) {
            selected = next;
            first    = middle + 1;
        } else {
            last = middle - 1;
        }
    }
    return selected;
}

Storage::Storage(DeviceContext& device, Capacity capacity)
    : capacity_(capacity), output_(device.device(), capacity.output_bytes),
      staging_(capacity.encoder.host_staging_bytes), timer_(device) {}

bool Storage::begin(const WindowUse& use) {
    if (active_ && active_->id == use.id) {
        if (active_->owner.sequence != use.owner.sequence ||
            active_->owner.residency != use.owner.residency)
            throw std::logic_error("Vision window changed its execution owner");
        return false;
    }
    if (!use.produce || use.id <= last_id_ || !use.plan || !use.plan->features ||
        use.plan->features > capacity_.max_features)
        throw std::logic_error("Vision window has no matching bounded production");
    active_      = use;
    last_id_     = use.id;
    output_peak_ = std::max(output_peak_, active_bytes());
    timer_.start();
    return true;
}

void Storage::produced(std::size_t bytes) {
    if (!active_ || bytes > capacity_.encoder.workspace_bytes)
        throw std::logic_error("Vision production exceeds its prepared workspace");
    workspace_peak_ = std::max(workspace_peak_, bytes);
    timer_.record_stop();
    timing_pending_ = true;
}

void Storage::release(std::uint64_t id) noexcept {
    if (active_ && active_->id == id) active_.reset();
}

std::optional<std::uint64_t> Storage::take_timing() {
    if (!timing_pending_) return {};
    const auto elapsed = static_cast<std::uint64_t>(timer_.elapsed_ms() * 1.0e6 + 0.5);
    timing_pending_    = false;
    return elapsed;
}

std::size_t Storage::active_bytes() const noexcept {
    return active_ ? std::size_t(active_->plan->features) * capacity_.output_width * 2 : 0;
}

void Storage::reset_peaks() noexcept {
    output_peak_    = active_bytes();
    workspace_peak_ = 0;
}

Tensor Storage::output(std::uint32_t first, std::uint32_t count) {
    if (!count || first > capacity_.max_features || count > capacity_.max_features - first)
        throw std::out_of_range("Vision output range exceeds its fixed window");
    return Tensor(output_.span(), DType::BF16,
                  {capacity_.output_width, static_cast<std::int32_t>(capacity_.max_features)})
        .slice(1, first, count);
}

VisionMemorySummary Storage::observe() const noexcept {
    return {capacity_.budget_bytes,
            capacity_.max_features,
            capacity_.encoder.workspace_bytes,
            capacity_.output_bytes,
            capacity_.encoder.host_staging_bytes,
            active_bytes(),
            output_peak_};
}

} // namespace ninfer::models::vision
