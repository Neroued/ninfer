#include "runtime/engine/model_instance.h"
#include "models/qwen3_5/program/retrieval/window_capacity.h"

#include <iostream>
#include <stdexcept>

int main() {
    ninfer::EngineOptions valid;
    valid.max_context = 16384;
    valid.kvmem_window_pages = 64;
    (void)ninfer::runtime::normalize_engine_options(valid);
    const auto rejects = [](ninfer::EngineOptions options) {
        try {
            (void)ninfer::runtime::normalize_engine_options(options);
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    auto dual = valid;
    dual.max_concurrency = 2;
    (void)ninfer::runtime::normalize_engine_options(dual);
    namespace capacity = ninfer::models::qwen3_5::detail;
    // Both full working sets must fit while chunks grow, not just two descriptor slots.
    if (capacity::kvmem_pool_page_budget(16384, 1024, 64, 2) != 192 ||
        capacity::kvmem_pool_page_budget(16384, 1024, 64, 1) != 96 ||
        capacity::kvmem_pool_page_budget(512, 1024, 8, 2) != 16 ||
        capacity::kvmem_pool_page_budget(513, 2048, 8, 2) != 18) { return 1; }
    auto bad = valid;
    bad.max_concurrency = 3;
    if (!rejects(bad)) { return 1; }
    bad = valid;
    bad.kvmem_window_pages = 1;
    if (!rejects(bad)) { return 1; }
    bad.kvmem_window_pages = 0xffffffffU;
    if (!rejects(bad)) { return 1; }
    bad = valid;
    bad.speculative.backend = ninfer::SpeculativeBackend::DFlash;
    if (!rejects(bad)) { return 1; }
    bad = valid;
    bad.purpose = ninfer::EnginePurpose::CausalScoring;
    if (!rejects(bad)) { return 1; }
    bad = valid;
    bad.context_cache.enabled = false;
    if (!rejects(bad)) { return 1; }
    bad = valid;
    bad.context_cache.host_kv_capacity_bytes = 0;
    if (!rejects(bad)) { return 1; }
    for (const auto lanes : {1U, 2U}) {
        for (const auto backend : {ninfer::SpeculativeBackend::None, ninfer::SpeculativeBackend::Mtp,
                                  ninfer::SpeculativeBackend::DFlash2}) {
            auto vision = valid;
            vision.enable_vision = true;
            vision.max_concurrency = lanes;
            vision.speculative.backend = backend;
            (void)ninfer::runtime::normalize_engine_options(vision);
        }
    }
    // Capture storage follows configured chunk geometry, not an implicit 1024 cap.
    valid.prefill_chunk = 2048;
    (void)ninfer::runtime::normalize_engine_options(valid);
    std::cout << "ok\n";
}
