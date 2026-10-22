#include "models/common/context/directory/allocation.h"


#include <iostream>
#include <array>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
namespace context = ninfer::models::context;
int failures      = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

void test_reserve_publish_and_rollback() {
    context::HostAllocator arena(0, 1025, 1);
    expect(arena.capacity_bytes() == 1025 && arena.occupied_bytes() == 0,
           "idle fixed backing remains fully charged");
    auto source = arena.allocate(257);
    expect(source && source->bytes() == 512 && arena.reserved_bytes() == 512,
           "allocation includes alignment padding immediately");
    const auto source_extent = source->extent();
    source->publish();
    source->publish();
    expect(arena.live_bytes() == 512 && arena.reserved_bytes() == 0,
           "publication charges one unique live allocation");
    {
        auto target = arena.allocate(257);
        expect(target && target->extent().offset != source->extent().offset &&
                   arena.occupied_bytes() == 1024 && arena.live_bytes() == 512 &&
                   arena.reserved_bytes() == 512,
               "whole destination is reserved while the source remains resident");
        expect(!arena.allocate(1) && !arena.can_allocate(1),
               "unused trailing byte cannot satisfy aligned allocation");
    }
    expect(arena.occupied_bytes() == 512 && arena.reserved_bytes() == 0 &&
               arena.peak_occupied_bytes() == 1024,
           "cancelled destination rolls back occupancy without losing peak");
    expect(source->extent() == source_extent, "failed destination preserves the source range");
    auto moved = std::move(*source);
    expect(!source->valid() && arena.allocation_count() == 1,
           "moving ownership neither duplicates nor releases bytes");
    expect(moved.release() && !moved.release() && arena.occupied_bytes() == 0 &&
               arena.capacity_bytes() == 1025,
           "last owner release frees reusable bytes without hiding fixed backing");
    expect(!arena.allocate(0) && !arena.allocate(std::numeric_limits<std::size_t>::max()),
           "zero and overflowing requests fail without changing accounting");
}

void test_fragmentation_and_split() {
    context::HostAllocator arena(0, 2048, 256);
    auto full = arena.allocate(2048);
    full->publish();
    std::vector<context::HostAllocation> pieces;
    pieces.reserve(8);
    while (full->bytes() > 256) {
        auto split = arena.split(std::move(*full), 256);
        pieces.push_back(std::move(split.first));
        *full = std::move(split.second);
    }
    pieces.push_back(std::move(*full));
    expect(arena.allocation_count() == 8 && arena.live_bytes() == 2048,
           "split reaches the smallest geometry without changing charged bytes");
    for (std::size_t index = 0; index < pieces.size(); index += 2) {
        (void)pieces[index].release();
    }
    expect(arena.free_bytes() == 1024 && !arena.can_allocate(512) && !arena.allocate(512),
           "actual contiguous extents, not total free bytes, govern availability");
    (void)pieces[1].release();
    auto merged = arena.allocate(768);
    expect(merged.has_value(), "adjacent released split pieces coalesce for another geometry");
    bool invalid_split_rejected = false;
    try {
        (void)arena.split(std::move(*merged), 1);
    } catch (const std::out_of_range&) { invalid_split_rejected = true; }
    expect(invalid_split_rejected && merged->valid() && merged->bytes() == 768,
           "invalid split leaves its source allocation owned");
    pieces.clear();
    merged.reset();
    expect(arena.occupied_bytes() == 0 && arena.allocation_count() == 0 && arena.can_allocate(2048),
           "all split pieces return to one complete reusable extent");
    context::HostAllocator disabled(0, 0, 256);
    expect(!disabled.allocate(256) && disabled.capacity_bytes() == 0,
           "Host zero has no backing or allocations");
}

void test_whole_items_and_projected_releases() {
    context::HostAllocator arena(0, 4096, 256);
    std::vector<context::HostAllocation> holes, separators;
    for (unsigned i = 0; i < 4; ++i) {
        holes.push_back(std::move(*arena.allocate(768)));
        separators.push_back(std::move(*arena.allocate(256)));
        separators.back().publish();
    }
    for (auto& hole : holes) hole.release();
    const std::array large{context::HostRequest{1024, 1}};
    expect(arena.free_bytes() == 3072 && !arena.plan(large),
           "aggregate bytes incorrectly admitted an indivisible item");
    const std::array pages{context::HostRequest{768, 4}};
    auto scattered = arena.plan(pages);
    expect(scattered && scattered->runs().size() == 4,
           "page placement required an unnecessary contiguous extent");
    {
        auto targets = arena.reserve(*scattered);
        expect(targets && targets->size() == 4 && arena.reserved_bytes() == 3072,
               "complete scattered reservation was not fulfilled");
    }
    expect(arena.free_bytes() == 3072 && arena.reserved_bytes() == 0,
           "scattered reservation rollback changed live storage");

    const std::array mixed{context::HostRequest{1024, 1}, context::HostRequest{768, 2}};
    const auto released = separators.front().extent();
    auto projected      = arena.plan(mixed, std::span(&released, 1));
    expect(projected && arena.free_bytes() == 3072 && !arena.reserve(*projected),
           "projected release mutated or prematurely reserved occupied storage");
    separators.front().release();
    auto targets = arena.reserve(*projected);
    expect(targets && arena.reserved_bytes() == 2560,
           "placement after the quoted release disagreed with reservation");
    std::size_t images = 0, kv = 0;
    for (const auto& run : *targets) {
        if (run.request == 0)
            images += run.count;
        else
            kv += run.count;
    }
    expect(images == 1 && kv == 2, "mixed reservation lost item identity");
    targets.reset();
    auto stale          = arena.plan(pages);
    auto other          = arena.allocate(256);
    const auto occupied = arena.occupied_bytes();
    expect(stale && other && !arena.reserve(*stale) && arena.occupied_bytes() == occupied,
           "stale reservation partially consumed remaining free ranges");
}
} // namespace

int main() {
    try {
        test_reserve_publish_and_rollback();
        test_fragmentation_and_split();
        test_whole_items_and_projected_releases();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
