#include "core/expert_cache.h"

#include "core/buffer.h"
#include "core/transfer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <stdexcept>

namespace ninfer {
namespace {

constexpr auto absent = kExpertAbsentOffset;

void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
}

void outside_capture(CudaExecutionView execution) {
    if (!execution.stream()) {
        throw std::invalid_argument("expert cache handoffs require an explicit CUDA stream");
    }
    cudaStreamCaptureStatus status;
    cuda_check(cudaStreamIsCapturing(execution.stream(), &status));
    if (status != cudaStreamCaptureStatusNone) {
        throw std::logic_error("expert cache handoffs run outside CUDA Graph capture");
    }
}


} // namespace

ExpertCacheRequirements describe_expert_cache(std::span<const ExpertBankCacheSpec> sources,
                                              const ExpertCacheConfig& config) {
    if (!config.max_batch_bytes || !config.max_batch_copies || !config.upload_slots ||
        !config.max_in_flight_uses || config.capacity_bytes % 256 || config.fixed_bytes % 256 ||
        config.fixed_bytes > config.capacity_bytes) {
        throw std::invalid_argument(
            "expert cache requires aligned capacity and bounded uploads/uses");
    }
    ExpertCacheRequirements out;
    out.config = config;
    LayoutBuilder offsets, divisors;
    for (const auto& bank : sources) {
        if (!bank.gate_up.region.count || bank.gate_up.region.count != bank.down.region.count) {
            throw std::invalid_argument("expert cache bank counts differ");
        }
        ExpertBankCacheLayout layout;
        layout.count    = bank.gate_up.region.count;
        layout.resident = bank.resident;
        LayoutBuilder entry;
        layout.gate_up     = plan_expert_matrix(entry, bank.gate_up);
        layout.down        = plan_expert_matrix(entry, bank.down);
        layout.entry_bytes = entry.finish(256, "expert entry");
        for (const auto* matrix : {&bank.gate_up, &bank.down}) {
            layout.payload_bytes += matrix->codes.expert_bytes + matrix->high.expert_bytes +
                                    matrix->scales.expert_bytes;
        }
        layout.first_entry = out.entry_count;
        if (!layout.resident) {
            layout.offsets = offsets.add(layout.count * sizeof(std::uint64_t), 8, "expert offsets");
            if (bank.gate_up.has_divisors) {
                layout.gate_up_divisors =
                    divisors.add(layout.count * sizeof(float), 4, "GU divisors");
            }
            if (bank.down.has_divisors) {
                layout.down_divisors =
                    divisors.add(layout.count * sizeof(float), 4, "down divisors");
            }
            out.entry_count += layout.count;
        }
        out.banks.push_back(layout);
    }
    out.offset_bytes  = offsets.finish();
    out.divisor_bytes = divisors.finish();
    if (out.entry_count) {
        LayoutBuilder ring;
        for (std::uint32_t i = 0; i < config.upload_slots; ++i) {
            (void)ring.add(out.offset_bytes, 1, "expert index snapshot");
        }
        out.pinned_bytes = ring.finish();
    }
    if (!out.entry_count && config.capacity_bytes) {
        throw std::invalid_argument("resident expert banks require no cache pool");
    }
    return out;
}

ExpertCacheRequirements describe_expert_cache(std::span<const ExpertBankSource> sources,
                                              const ExpertCacheConfig& config) {
    std::vector<ExpertBankCacheSpec> specs;
    specs.reserve(sources.size());
    for (const auto& bank : sources) {
        for (const auto* matrix : {&bank.gate_up, &bank.down}) {
            if (!matrix->parent) { throw std::invalid_argument("expert bank has no source"); }
            (void)bind_weight_bank(*matrix->parent, matrix->layout);
        }
        if (bank.gate_up.parent->device != bank.down.parent->device ||
            bank.gate_up.layout.region.count != bank.down.layout.region.count) {
            throw std::invalid_argument("expert matrices must share count and source device");
        }
        if (bank.gate_up.parent->device == Device::cpu()) {
            const auto validate_upload = [](const WeightBankView& matrix, CudaHostView upload) {
                validate_memory(upload.host);
                if (upload.host.data != matrix.parent->data ||
                    upload.host.bytes < matrix.parent->geometry.bytes || !upload.host.host_pinned ||
                    upload.host.device != Device::cpu() || !upload.device_address ||
                    upload.cuda_device.type() != DeviceType::CUDA) {
                    throw std::invalid_argument(
                        "Host expert source requires registered transfer backing");
                }
            };
            validate_upload(bank.gate_up, bank.gate_up_upload);
            validate_upload(bank.down, bank.down_upload);
            if (bank.gate_up_upload.cuda_device != bank.down_upload.cuda_device) {
                throw std::invalid_argument("expert upload mappings belong to different devices");
            }
        }
        const auto location = bank.gate_up.parent->device.type();
        if (location != DeviceType::CPU && location != DeviceType::CUDA) {
            throw std::invalid_argument("expert source requires bound CPU or CUDA storage");
        }
        specs.push_back({bank.gate_up.layout, bank.down.layout, location == DeviceType::CUDA});
    }
    return describe_expert_cache(specs, config);
}

struct ExpertCache::Impl {
    enum class State { Absent, Reserved, Published };

    struct Entry {
        ExpertKey key;
        State state          = State::Absent;
        std::uint64_t offset = absent, generation = 0;
        std::size_t copied = 0, ready_slot = 0;
        std::uint64_t ready_generation = 0;
        std::uint32_t pins = 0, open_uses = 0;
        bool dirty = false;
    };

    struct Slot {
        PinnedHostBuffer index_snapshot;
        CudaHostView index_source;
        CudaHostCopyBatch data_copies, index_copies;
        CudaCompletionEvent done;
        std::size_t payload_bytes = 0;
        bool pending              = false;
        std::uint64_t generation = 0, wait_epoch = 0;

        Slot(const ExpertCacheRequirements& req, const DeviceContext& ctx)
            : index_snapshot(req.offset_bytes),
              index_source(index_snapshot.cuda_view(ctx.device())),
              data_copies(ctx.device(), req.config.max_batch_copies),
              index_copies(ctx.device(), req.entry_count), done(ctx) {}
    };

    struct Use {
        struct Reference {
            std::size_t index;
            std::uint64_t generation;
        };

        CudaCompletionEvent done;
        std::vector<Reference> entries;
        std::optional<CudaExecutionView> execution;
        std::uint64_t generation = 0, wait_epoch = 0;
        bool released = false;

        Use(const DeviceContext& ctx, std::size_t count) : done(ctx) { entries.reserve(count); }
    };

    std::vector<ExpertBankSource> sources;
    ExpertCacheRequirements requirements;
    CudaExecutionView transfer;
    DeviceBuffer pool, offsets, divisors;
    std::vector<ExpertBankDeviceView> views;
    std::vector<Entry> entries;
    std::vector<std::unique_ptr<Slot>> slots;
    std::vector<std::unique_ptr<Use>> uses;
    std::vector<std::size_t> required, prefetch, dirty, completed, selection;
    std::uint64_t acquire_epoch = 0, writer_epoch = 0;
    std::size_t next_slot = 0;
    ExpertCacheUsage counters;
    bool failed = false;

    Impl(std::span<const ExpertBankSource> input, const ExpertCacheRequirements& req,
         const DeviceContext& ctx, CudaExecutionView upload, std::span<const ExpertPlacement> fixed)
        : sources(input.begin(), input.end()),
          requirements(describe_expert_cache(input, req.config)), transfer(upload),
          pool(ctx.device(), requirements.config.capacity_bytes),
          offsets(ctx.device(), requirements.offset_bytes),
          divisors(ctx.device(), requirements.divisor_bytes) {
        // The description is a capacity promise; execution recomputes it from the actual sources.
        if (req != requirements) {
            throw std::invalid_argument("expert cache requirements differ from sources");
        }
        if (transfer.device() != ctx.device()) {
            throw std::invalid_argument("expert upload device differs from its backing");
        }
        outside_capture(transfer);
        offsets.fill(0xff);
        entries.reserve(requirements.entry_count);
        for (auto* vector : {&required, &prefetch, &dirty, &completed, &selection}) {
            vector->reserve(requirements.entry_count);
        }
        for (std::size_t b = 0; b < sources.size(); ++b) {
            const auto& source = sources[b];
            const auto& layout = requirements.banks[b];
            if (!layout.resident && source.gate_up_upload.cuda_device != ctx.device()) {
                throw std::invalid_argument("expert upload mapping belongs to another GPU");
            }
            const auto bind = [&](const WeightBankView& matrix,
                                  const ExpertMatrixEntryLayout& image,
                                  const LayoutRegion& divisor_region) {
                if (layout.resident) {
                    if (matrix.parent->device != ctx.device()) {
                        throw std::invalid_argument("resident expert belongs to another GPU");
                    }
                    return device_weight_bank(matrix);
                }
                ExpertMatrixDeviceView view;
                const auto plane = [&](const LayoutRegion& region) -> const std::byte* {
                    return pool.data() && region.bytes
                               ? static_cast<const std::byte*>(pool.data()) + region.offset
                               : nullptr;
                };
                view.codes           = plane(image.codes);
                view.high            = plane(image.high);
                view.scales          = plane(image.scales);
                view.indexed_storage = pool.span();
                view.offsets =
                    static_cast<const std::uint64_t*>(layout.offsets.bind(offsets.span()).data);
                if (divisor_region.bytes) {
                    std::vector<float> values(layout.count);
                    for (std::uint32_t e = 0; e < layout.count; ++e) {
                        values[e] = matrix.divisor(e);
                    }
                    divisors.copy_from_host(values.data(), divisor_region.bytes,
                                            divisor_region.offset);
                    view.divisors =
                        static_cast<const float*>(divisor_region.bind(divisors.span()).data);
                    view.divisor_stride = 1;
                }
                return view;
            };
            views.push_back({bind(source.gate_up, layout.gate_up, layout.gate_up_divisors),
                             bind(source.down, layout.down, layout.down_divisors), layout.count,
                             ctx.device()});
            if (!layout.resident) {
                for (std::uint32_t e = 0; e < layout.count; ++e) {
                    entries.push_back({ExpertKey{ExpertBankId{static_cast<std::uint32_t>(b)}, e}});
                }
            }
        }
        if (requirements.entry_count) {
            for (std::uint32_t i = 0; i < requirements.config.upload_slots; ++i) {
                slots.push_back(std::make_unique<Slot>(requirements, ctx));
            }
        }
        for (std::uint32_t i = 0; i < requirements.config.max_in_flight_uses; ++i) {
            uses.push_back(std::make_unique<Use>(ctx, entries.size()));
        }
        replace({0, requirements.config.fixed_bytes, 256}, fixed, true);
    }

    ~Impl() {
        (void)cudaStreamSynchronize(transfer.stream());
        for (const auto& use : uses) {
            if (use->execution) (void)cudaStreamSynchronize(use->execution->stream());
        }
    }

    void healthy() const {
        if (failed) throw std::runtime_error("expert cache failed during CUDA submission");
    }

    std::optional<std::size_t> index(ExpertKey key) const {
        if (key.bank.index >= requirements.banks.size()) throw std::out_of_range("expert bank");
        const auto& bank = requirements.banks[key.bank.index];
        if (key.expert >= bank.count) throw std::out_of_range("logical expert");
        if (bank.resident) return {};
        return bank.first_entry + key.expert;
    }

    std::size_t size(std::size_t i) const {
        return requirements.banks[entries[i].key.bank.index].entry_bytes;
    }

    std::size_t payload_size(std::size_t i) const {
        return requirements.banks[entries[i].key.bank.index].payload_bytes;
    }

    void prepare(const ExpertCacheRequest& request) {
        const auto collect = [&](std::span<const ExpertKey> keys) {
            std::vector<std::size_t> result;
            result.reserve(keys.size());
            for (const auto key : keys) {
                if (auto i = index(key)) {
                    if (entries[*i].state == State::Absent)
                        throw std::invalid_argument("requested expert has no placement");
                    result.push_back(*i);
                }
            }
            std::sort(result.begin(), result.end());
            result.erase(std::unique(result.begin(), result.end()), result.end());
            return result;
        };
        auto next_required = collect(request.required);
        auto next_prefetch = collect(request.prefetch);
        std::erase_if(next_prefetch, [&](auto i) {
            return std::binary_search(next_required.begin(), next_required.end(), i);
        });
        required = std::move(next_required);
        prefetch = std::move(next_prefetch);
    }

    void reap() {
        for (auto& slot : slots) {
            if (slot->pending && slot->done.ready()) slot->pending = false;
        }
        for (auto& use : uses) {
            if (use->execution && use->released && use->done.ready()) {
                for (const auto ref : use->entries) {
                    if (entries[ref.index].generation == ref.generation) --entries[ref.index].pins;
                }
                use->entries.clear();
                use->execution.reset();
            }
        }
    }

    void mark_dirty(std::size_t i) {
        if (!entries[i].dirty) {
            entries[i].dirty = true;
            dirty.push_back(i);
        }
    }

    void wait_readers(std::size_t i) {
        const auto& entry = entries[i];
        if (entry.open_uses) throw std::logic_error("expert mutation retains an open reader");
        if (!entry.pins) return;
        for (auto& use : uses) {
            if (!use->execution || use->wait_epoch == writer_epoch) continue;
            const auto ref =
                std::lower_bound(use->entries.begin(), use->entries.end(), i,
                                 [](const auto& ref, auto i) { return ref.index < i; });
            if (ref != use->entries.end() && ref->index == i &&
                ref->generation == entry.generation) {
                use->done.wait(transfer.stream());
                use->wait_epoch = writer_epoch;
            }
        }
    }

    void replace(LayoutRegion region, std::span<const ExpertPlacement> placements,
                 bool initialize = false) {
        if (region.offset % 256 || region.bytes % 256 || region.offset > pool.size() ||
            region.bytes > pool.size() - region.offset ||
            (!initialize && region.offset < requirements.config.fixed_bytes)) {
            throw std::invalid_argument("expert replacement requires a mutable aligned region");
        }
        std::vector<std::pair<std::size_t, std::size_t>> targets;
        targets.reserve(placements.size());
        for (const auto& placement : placements) {
            const auto i = index(placement.key);
            if (!i || placement.offset % 256 || placement.offset > region.bytes ||
                size(*i) > region.bytes - placement.offset)
                throw std::invalid_argument("expert placement exceeds its region");
            targets.emplace_back(*i, region.offset + placement.offset);
        }
        std::sort(targets.begin(), targets.end());
        for (std::size_t n = 1; n < targets.size(); ++n) {
            if (targets[n - 1].first == targets[n].first)
                throw std::invalid_argument("expert placement contains a duplicate key");
        }
        std::sort(targets.begin(), targets.end(),
                  [](auto a, auto b) { return a.second < b.second; });
        for (std::size_t n = 1; n < targets.size(); ++n) {
            if (targets[n - 1].second + size(targets[n - 1].first) > targets[n].second)
                throw std::invalid_argument("expert placements overlap");
        }
        const auto end = region.offset + region.bytes;
        std::vector<std::size_t> retired;
        for (std::size_t i = 0; i < entries.size(); ++i) {
            const auto& entry = entries[i];
            if (entry.state == State::Absent || entry.offset >= end ||
                entry.offset + size(i) <= region.offset)
                continue;
            if (entry.offset < region.offset || entry.offset + size(i) > end)
                throw std::invalid_argument("replacement cuts through an expert placement");
            if (entry.open_uses)
                throw std::logic_error("expert replacement retains an open reader");
            retired.push_back(i);
        }
        for (const auto [i, offset] : targets) {
            const auto& entry = entries[i];
            if (entry.state != State::Absent &&
                !std::binary_search(retired.begin(), retired.end(), i))
                throw std::invalid_argument("expert already belongs to another region");
        }
        try {
            ++writer_epoch;
            for (const auto i : retired) wait_readers(i);
            for (const auto i : retired) {
                auto& entry = entries[i];
                counters.used_bytes -= size(i);
                entry.state  = State::Absent;
                entry.offset = absent;
                entry.copied = 0;
                entry.pins   = 0;
                ++entry.generation;
                mark_dirty(i);
            }
            for (const auto [i, offset] : targets) {
                auto& entry  = entries[i];
                entry.state  = State::Reserved;
                entry.offset = offset;
                ++entry.generation;
                counters.used_bytes += size(i);
            }
            // A region update cancels unsubmitted work for its old contents.
            const auto replaced = [&](auto i) {
                return std::binary_search(retired.begin(), retired.end(), i);
            };
            std::erase_if(required, replaced);
            std::erase_if(prefetch, replaced);
        } catch (...) {
            failed = true;
            throw;
        }
    }

    template <typename F>
    void planes(std::size_t i, F&& visit) const {
        const auto& entry  = entries[i];
        const auto& source = sources[entry.key.bank.index];
        const auto& layout = requirements.banks[entry.key.bank.index];
        const auto matrix  = [&](const WeightBankView& bank, CudaHostView upload,
                                const ExpertMatrixEntryLayout& image) {
            const std::array input{bank.layout.codes, bank.layout.high, bank.layout.scales};
            const std::array output{image.codes, image.high, image.scales};
            for (std::size_t p = 0; p < input.size(); ++p) {
                if (!input[p].expert_bytes) { continue; }
                const auto offset = input[p].offset + entry.key.expert * input[p].expert_stride;
                visit(upload, offset, input[p].expert_bytes, input[p].expert_stride,
                       output[p].offset);
            }
        };
        matrix(source.gate_up, source.gate_up_upload, layout.gate_up);
        matrix(source.down, source.down_upload, layout.down);
    }

    void append_copy(Slot& slot, std::size_t i, std::size_t limit) {
        auto& entry             = entries[i];
        std::size_t plane_begin = 0;
        planes(i, [&](CudaHostView source, std::size_t offset, std::size_t bytes, std::size_t,
                      std::size_t target) {
            const auto plane_end = plane_begin + bytes;
            if (entry.copied >= plane_begin && entry.copied < plane_end &&
                slot.payload_bytes < limit &&
                slot.data_copies.size() < slot.data_copies.capacity()) {
                const auto from  = entry.copied - plane_begin;
                const auto count = std::min(bytes - from, limit - slot.payload_bytes);
                slot.data_copies.append(pool.span().subspan(entry.offset + target + from, count),
                                        source.subspan(offset + from, count));
                entry.copied += count;
                slot.payload_bytes += count;
            }
            plane_begin = plane_end;
        });
    }

    bool affine_batch() const {
        if (completed.size() < 64) { return false; }
        const auto& first = entries[completed.front()];
        for (std::size_t n = 0; n < completed.size(); ++n) {
            const auto& entry = entries[completed[n]];
            if (entry.key.bank != first.key.bank || entry.key.expert != first.key.expert + n ||
                entry.offset != first.offset + n * size(completed.front())) {
                return false;
            }
        }
        return true;
    }

    void upload_affine() {
        const auto first = completed.front();
        const auto rows  = completed.size();
        planes(first, [&](CudaHostView source, std::size_t offset, std::size_t bytes,
                          std::size_t stride, std::size_t target) {
            const auto source_bytes      = (rows - 1) * stride + bytes;
            const auto destination_bytes = (rows - 1) * size(first) + bytes;
            copy_host_2d_async(
                pool.span().subspan(entries[first].offset + target, destination_bytes), size(first),
                source.subspan(offset, source_bytes), stride, bytes, rows, transfer);
            ++counters.data_copy_calls;
        });
    }

    void publish(Slot& slot) {
        if (dirty.empty()) { return; }
        std::sort(dirty.begin(), dirty.end());
        auto* words = static_cast<std::uint64_t*>(slot.index_snapshot.data());
        for (std::size_t j = 0; j < dirty.size(); ++j) {
            const auto& entry = entries[dirty[j]];
            words[j] = entry.state != State::Absent && entry.copied == payload_size(dirty[j])
                           ? entry.offset
                           : absent;
        }
        slot.index_copies.clear();
        for (std::size_t first = 0; first < dirty.size();) {
            auto end = first + 1;
            while (end < dirty.size() && dirty[end] == dirty[end - 1] + 1) { ++end; }
            const auto bytes = (end - first) * sizeof(std::uint64_t);
            slot.index_copies.append(
                offsets.span().subspan(dirty[first] * sizeof(std::uint64_t), bytes),
                slot.index_source.subspan(first * sizeof(std::uint64_t), bytes));
            counters.index_bytes += bytes;
            first = end;
        }
        slot.index_copies.submit(transfer);
        ++counters.index_copy_calls;
        for (const auto i : dirty) { entries[i].dirty = false; }
        dirty.clear();
    }

    bool ready(std::size_t i) const {
        const auto& entry = entries[i];
        if (entry.state != State::Published) return false;
        const auto& slot = *slots[entry.ready_slot];
        return entry.ready_generation != slot.generation || !slot.pending;
    }

    void finish_upload(Slot& slot) {
        for (const auto i : completed) mark_dirty(i);
        publish(slot);
        slot.done.record(transfer.stream());
        slot.pending = true;
        ++slot.generation;
        for (const auto i : completed) {
            auto& entry            = entries[i];
            entry.state            = State::Published;
            entry.ready_slot       = next_slot % slots.size();
            entry.ready_generation = slot.generation;
        }
        ++next_slot;
    }

    ExpertCacheProgress progress(ExpertUploadBudget budget) {
        reap();
        ExpertCacheProgress result;
        for (std::size_t batch = 0; !slots.empty() && batch < budget.max_batches &&
                                    result.submitted_bytes < budget.max_bytes;
             ++batch) {
            auto& slot = *slots[next_slot % slots.size()];
            if (slot.pending) {
                result.waiting_for_uploads = true;
                break;
            }
            const auto planning_start = std::chrono::steady_clock::now();
            completed.clear();
            const bool required_pending =
                std::any_of(required.begin(), required.end(),
                            [&](auto i) { return entries[i].state != State::Published; });
            const auto limit   = std::min(requirements.config.max_batch_bytes,
                                          budget.max_bytes - result.submitted_bytes);
            slot.payload_bytes = 0;
            slot.data_copies.clear();
            bool whole_experts = true;
            const auto append  = [&](std::size_t i, bool speculative) {
                auto& entry = entries[i];
                if (entry.state != State::Reserved || entry.copied == payload_size(i)) return true;
                if (slot.payload_bytes &&
                    payload_size(i) - entry.copied > limit - slot.payload_bytes)
                    return false;
                if (entry.copied) whole_experts = false;
                const auto before = slot.payload_bytes;
                append_copy(slot, i, limit);
                if (speculative) counters.prefetched_bytes += slot.payload_bytes - before;
                if (entry.copied == payload_size(i))
                    completed.push_back(i);
                else
                    whole_experts = false;
                return slot.payload_bytes < limit && entry.copied == payload_size(i) &&
                       slot.data_copies.size() < slot.data_copies.capacity();
            };
            bool room = true;
            for (const auto i : required) {
                if (!(room = append(i, false))) break;
            }
            if (room && !required_pending) {
                for (const auto i : prefetch) {
                    if (!append(i, true)) break;
                }
            }
            const bool affine = whole_experts && affine_batch();
            counters.planning_nanoseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                 std::chrono::steady_clock::now() - planning_start)
                                                 .count();
            if (!slot.payload_bytes && dirty.empty()) break;
            const auto submit_start = std::chrono::steady_clock::now();
            if (slot.payload_bytes) {
                if (affine)
                    upload_affine();
                else {
                    slot.data_copies.submit(transfer);
                    ++counters.data_copy_calls;
                }
                counters.data_bytes += slot.payload_bytes;
            }
            finish_upload(slot);
            counters.submit_nanoseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::steady_clock::now() - submit_start)
                                               .count();
            ++result.submitted_batches;
            result.submitted_bytes += slot.payload_bytes;
        }
        result.required_published = std::all_of(required.begin(), required.end(), [&](auto i) {
            return entries[i].state == State::Published;
        });
        result.prefetch_ready =
            std::all_of(prefetch.begin(), prefetch.end(), [&](auto i) { return ready(i); });
        return result;
    }
};

ExpertCache::ExpertCache(std::span<const ExpertBankSource> sources,
                         const ExpertCacheRequirements& requirements, const DeviceContext& device,
                         CudaExecutionView upload, std::span<const ExpertPlacement> fixed)
    : impl_(std::make_unique<Impl>(sources, requirements, device, upload, fixed)) {}

ExpertCache::~ExpertCache() = default;

void ExpertCache::replace(LayoutRegion region, std::span<const ExpertPlacement> placements) {
    impl_->healthy();
    outside_capture(impl_->transfer);
    impl_->replace(region, placements);
}

void ExpertCache::prepare(const ExpertCacheRequest& request) {
    impl_->healthy();
    outside_capture(impl_->transfer);
    impl_->prepare(request);
}

ExpertCacheProgress ExpertCache::progress(ExpertUploadBudget budget) {
    impl_->healthy();
    outside_capture(impl_->transfer);
    try {
        return impl_->progress(budget);
    } catch (...) {
        impl_->failed = true;
        throw;
    }
}

void ExpertCache::ready_keys(ExpertBankId bank, std::vector<ExpertKey>& keys) {
    auto& state = *impl_;
    state.healthy();
    state.reap();
    keys.clear();
    const auto& layout = state.requirements.banks.at(bank.index);
    for (std::uint32_t e = 0; e < layout.count; ++e) {
        if (layout.resident || state.ready(layout.first_entry + e)) keys.push_back({bank, e});
    }
}

void ExpertCache::wait_for_progress() {
    auto& state = *impl_;
    state.healthy();
    outside_capture(state.transfer);
    state.reap();
    if (state.slots.empty()) return;
    const auto& slot = *state.slots[state.next_slot % state.slots.size()];
    if (slot.pending)
        slot.done.synchronize();
    else {
        const auto use = std::find_if(state.uses.begin(), state.uses.end(), [](const auto& use) {
            return use->execution && use->released;
        });
        if (use != state.uses.end())
            (*use)->done.synchronize();
        else {
            const auto pending = std::find_if(state.slots.begin(), state.slots.end(),
                                              [](const auto& slot) { return slot->pending; });
            if (pending != state.slots.end()) (*pending)->done.synchronize();
        }
    }
    state.reap();
}

std::optional<ExpertCacheUse> ExpertCache::acquire(std::span<const ExpertKey> keys,
                                                   CudaExecutionView execution) {
    auto& state = *impl_;
    state.healthy();
    if (execution.device() != state.transfer.device())
        throw std::invalid_argument("expert consumer device differs");
    outside_capture(execution);
    state.reap();
    state.selection.clear();
    for (const auto key : keys) {
        if (auto i = state.index(key)) {
            if (state.entries[*i].state != Impl::State::Published) return {};
            state.selection.push_back(*i);
        }
    }
    std::sort(state.selection.begin(), state.selection.end());
    state.selection.erase(std::unique(state.selection.begin(), state.selection.end()),
                          state.selection.end());
    for (std::size_t u = 0; u < state.uses.size(); ++u) {
        auto& use = *state.uses[u];
        if (use.execution) continue;
        ++state.acquire_epoch;
        use.execution = execution;
        use.released  = false;
        ++use.generation;
        try {
            for (const auto i : state.selection) {
                const auto& entry = state.entries[i];
                auto& slot        = *state.slots[entry.ready_slot];
                if (entry.ready_generation == slot.generation &&
                    slot.wait_epoch != state.acquire_epoch) {
                    slot.done.wait(execution.stream());
                    slot.wait_epoch = state.acquire_epoch;
                }
            }
        } catch (...) {
            state.failed = true;
            throw;
        }
        for (const auto i : state.selection) {
            auto& entry = state.entries[i];
            ++entry.pins;
            ++entry.open_uses;
            use.entries.push_back({i, entry.generation});
        }
        return ExpertCacheUse{this, static_cast<std::uint32_t>(u), use.generation};
    }
    if (std::any_of(state.uses.begin(), state.uses.end(),
                    [](const auto& use) { return use->released; }))
        return {};
    throw std::logic_error("expert cache open use capacity exhausted");
}

void ExpertCache::release(ExpertCacheUse token, CudaExecutionView execution) {
    auto& state = *impl_;
    state.healthy();
    if (token.owner != this) throw std::invalid_argument("expert use belongs to another cache");
    auto& use = *state.uses.at(token.slot);
    if (!use.execution || use.generation != token.generation || use.released ||
        execution.device() != use.execution->device() ||
        execution.stream() != use.execution->stream())
        throw std::invalid_argument("expert use must be released once on its consumer stream");
    outside_capture(execution);
    try {
        use.done.record(execution.stream());
    } catch (...) {
        state.failed = true;
        throw;
    }
    for (const auto ref : use.entries) --state.entries[ref.index].open_uses;
    use.released = true;
}

const ExpertBankDeviceView& ExpertCache::view(ExpertBankId bank) const {
    return impl_->views.at(bank.index);
}

std::uint64_t ExpertCache::uploaded_bytes() const noexcept { return impl_->counters.data_bytes; }

ExpertCacheUsage ExpertCache::usage() const {
    impl_->reap();
    auto out = impl_->counters;
    out.ready_experts_per_bank.resize(impl_->requirements.banks.size());
    for (std::size_t b = 0; b < impl_->requirements.banks.size(); ++b) {
        const auto& bank = impl_->requirements.banks[b];
        if (bank.resident) out.ready_experts_per_bank[b] = bank.count;
    }
    for (const auto& slot : impl_->slots) {
        if (slot->pending) out.in_flight_bytes += slot->payload_bytes;
    }
    for (std::size_t i = 0; i < impl_->entries.size(); ++i) {
        out.published_experts += impl_->entries[i].state == Impl::State::Published;
        if (impl_->ready(i)) {
            out.ready_cache_bytes += impl_->size(i);
            ++out.ready_experts_per_bank[impl_->entries[i].key.bank.index];
        }
    }
    return out;
}

} // namespace ninfer
