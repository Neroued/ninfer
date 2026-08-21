// Transactional invariants of the content-addressed host KV store: dedup refcounts,
// shared-page pinning across an eviction inside an open save transaction (the failure mode
// behind "KV host cache segment references a missing page"), abort/rollback, seal
// validate-before-mutate, and the single-open-transaction contract.

#include "core/kv_host_cache.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

void check(bool condition, const char* label) {
    if (!condition) { throw std::runtime_error(std::string("check failed: ") + label); }
}

constexpr std::size_t kPageBytes   = 64 * 1024;
constexpr std::size_t kAnchorBytes = 4 * 1024;

struct DeviceScratch {
    void* data = nullptr;
    explicit DeviceScratch(std::size_t bytes) {
        if (cudaMalloc(&data, bytes) != cudaSuccess) {
            throw std::runtime_error("cudaMalloc scratch failed");
        }
    }
    ~DeviceScratch() { cudaFree(data); }
};

ninfer::KvHostCopySlice page_slice(const DeviceScratch& scratch) {
    return {scratch.data, kPageBytes};
}

ninfer::KvHostCopySlice anchor_slice(const DeviceScratch& scratch) {
    return {scratch.data, kAnchorBytes};
}

ninfer::KvHostCache::AnchorMeta meta_at(std::uint32_t frontier) {
    ninfer::KvHostCache::AnchorMeta meta{};
    meta.frontier = frontier;
    return meta;
}

// Stages the given keys (pinning already-stored ones), stages an anchor, and seals.
void save_segment(ninfer::KvHostCache& store, const DeviceScratch& scratch,
                  std::uint64_t anchor_key, std::uint32_t frontier,
                  const std::vector<std::uint64_t>& keys) {
    using PageKind = ninfer::KvHostCache::PageKind;
    for (const std::uint64_t key : keys) {
        if (store.pin_page(PageKind::Text, key)) { continue; }
        const auto slice = page_slice(scratch);
        if (!store.stage_page(PageKind::Text, key, {&slice, 1}, nullptr)) {
            store.abort_segment();
            throw std::runtime_error("stage_page refused with available budget");
        }
    }
    const auto state = anchor_slice(scratch);
    if (!store.stage_anchor(anchor_key, meta_at(frontier), {&state, 1}, nullptr)) {
        store.abort_segment();
        throw std::runtime_error("stage_anchor refused");
    }
    store.seal_segment(anchor_key, keys, {}, keys.back());
    cudaDeviceSynchronize();
}

void run() {
    using PageKind = ninfer::KvHostCache::PageKind;
    DeviceScratch scratch(kPageBytes);

    // Budget: exactly two pages plus one anchor per 256 MiB chunk carve is impractical at
    // real chunk size, so use a budget below one chunk to exercise bounded carving too.
    ninfer::KvHostCache::Config config;
    config.budget_bytes       = 3 * kPageBytes + 2 * kAnchorBytes;
    config.text_page_bytes    = kPageBytes;
    config.backend_page_bytes = 0;
    config.anchor_state_bytes = kAnchorBytes;
    ninfer::KvHostCache store(config);

    // 1. Basic save: pages and anchor become visible.
    save_segment(store, scratch, 0xA1, 128, {0x11, 0x12});
    check(store.has_page(PageKind::Text, 0x11), "page 11 stored");
    check(store.find_anchor(0xA1).has_value(), "anchor A1 stored");
    check(store.stats().stored_segments == 1, "one segment");

    // 2. Dedup: a sibling referencing a shared page stores only its own tail.
    save_segment(store, scratch, 0xA2, 128, {0x11, 0x13});
    check(store.stats().stored_pages == 3, "three distinct pages after dedup");

    // 3. Eviction inside an open transaction must not free pages the transaction pinned:
    //    staging a third segment sharing 0x11 needs a page slot; the budget is exhausted, so
    //    the store evicts the LRU segment (0xA1, an owner of 0x11) mid-transaction. The pin
    //    taken at the scan must keep 0x11 alive for the seal.
    save_segment(store, scratch, 0xA3, 128, {0x11, 0x14});
    check(store.has_page(PageKind::Text, 0x11), "shared page survived owner eviction");
    check(store.find_anchor(0xA3).has_value(), "segment sealed under eviction pressure");
    check(!store.find_anchor(0xA1).has_value() || !store.find_anchor(0xA2).has_value(),
          "an idle segment was evicted for budget");

    // 4. Abort rolls the store back: nothing new is visible afterwards.
    {
        const auto slice = page_slice(scratch);
        check(store.stage_page(PageKind::Text, 0x21, {&slice, 1}, nullptr), "stage for abort");
        const std::uint64_t stored_before = store.stats().stored_pages;
        store.abort_segment();
        cudaDeviceSynchronize();
        check(!store.has_page(PageKind::Text, 0x21), "aborted page not visible");
        check(store.stats().stored_pages == stored_before, "abort left counts unchanged");
    }

    // 5. Seal validates the whole reference set before mutating anything.
    {
        const auto slice = page_slice(scratch);
        check(store.stage_page(PageKind::Text, 0x31, {&slice, 1}, nullptr), "stage for seal");
        const auto state = anchor_slice(scratch);
        check(store.stage_anchor(0xB1, meta_at(64), {&state, 1}, nullptr), "anchor for seal");
        const std::uint64_t segments_before = store.stats().stored_segments;
        bool threw                          = false;
        try {
            const std::vector<std::uint64_t> keys{0x31, 0xDEAD};
            store.seal_segment(0xB1, keys, {}, 0x31);
        } catch (const std::logic_error&) { threw = true; }
        check(threw, "seal with a missing reference throws");
        check(store.stats().stored_segments == segments_before, "failed seal mutated nothing");
        check(!store.has_page(PageKind::Text, 0x31), "failed seal published no staged page");
        store.abort_segment();
        cudaDeviceSynchronize();

        // 6. A new transaction cannot open over an unfinished one.
        check(store.stage_anchor(0xB2, meta_at(64), {&state, 1}, nullptr), "reopen anchor");
        bool double_open = false;
        try {
            (void)store.stage_anchor(0xB3, meta_at(64), {&state, 1}, nullptr);
        } catch (const std::logic_error&) { double_open = true; }
        check(double_open, "second open transaction rejected");
        store.abort_segment();
        cudaDeviceSynchronize();
    }

    std::cout << "kv host cache invariants: OK\n";
}

} // namespace

int main() {
    int devices          = 0;
    const cudaError_t rc = cudaGetDeviceCount(&devices);
    if (cuda_unavailable(rc) || devices == 0) {
        std::cout << "SKIP: no CUDA device\n";
        return 0;
    }
    try {
        run();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
