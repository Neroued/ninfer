#include "runtime/engine/context_cost.h"

#include "core/host_kv_arena.h"

#include <nlohmann/json.hpp>

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

#include <unistd.h>

namespace {

using Json = nlohmann::json;

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

template <class Function>
void expect_throw(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::exception&) { return; }
    expect(false, message);
}

std::array<ninfer::runtime::ContextTransferCost, 3> transfer(std::uint64_t batch_ns = 10) {
    std::array<ninfer::runtime::ContextTransferCost, 3> result;
    for (auto& direction : result) {
        direction = {.batch_ns        = batch_ns,
                     .operation_ns    = 3,
                     .ns_per_byte_q32 = ninfer::runtime::kContextCostQ32One / 2};
    }
    return result;
}

ninfer::runtime::ContextPrefillCost prefill(std::uint64_t chunk_ns = 3) {
    return {
        .chunk_ns              = chunk_ns,
        .token_ns_q32          = ninfer::runtime::kContextCostQ32One / 2,
        .attention_pair_ns_q32 = ninfer::runtime::kContextCostQ32One / 4,
        .vision_item_ns        = 5,
        .vision_patch_ns_q32   = 2 * ninfer::runtime::kContextCostQ32One,
    };
}

Json direction_json(const ninfer::runtime::ContextTransferCost& value) {
    return Json{{"batch_ns", value.batch_ns},
                {"operation_ns", value.operation_ns},
                {"ns_per_byte_q32", value.ns_per_byte_q32}};
}

Json transfer_json(const std::array<ninfer::runtime::ContextTransferCost, 3>& value) {
    return Json{{"d2h", direction_json(value[0])},
                {"h2d", direction_json(value[1])},
                {"d2d", direction_json(value[2])}};
}

Json prefill_json(const ninfer::runtime::ContextPrefillCost& value) {
    return Json{{"chunk_ns", value.chunk_ns},
                {"token_ns_q32", value.token_ns_q32},
                {"attention_pair_ns_q32", value.attention_pair_ns_q32},
                {"vision_item_ns", value.vision_item_ns},
                {"vision_patch_ns_q32", value.vision_patch_ns_q32}};
}

Json document(Json machines) {
    return Json{{"schema_version", 2},
                {"artifact_type", "ninfer_context_cost_presets"},
                {"machines", std::move(machines)}};
}

Json machine(std::string hardware, Json transfer_value, Json prefill_values) {
    return Json{{"hardware_class", std::move(hardware)},
                {"transfer", std::move(transfer_value)},
                {"prefill", std::move(prefill_values)}};
}

Json prefill_entry(std::string model, std::string weights,
                   const ninfer::runtime::ContextPrefillCost& value) {
    return Json{{"model_id", std::move(model)},
                {"weights_id", std::move(weights)},
                {"coefficients", prefill_json(value)}};
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void test_exact_evaluation() {
    expect(ninfer::runtime::context_cost_hardware_class("NVIDIA GeForce RTX 5090", 12, 0) ==
               "nvidia-geforce-rtx-5090-sm120",
           "hardware class duplicated or lost the NVIDIA vendor");
    expect(ninfer::runtime::context_cost_hardware_class("Example Accelerator", 9, 0) ==
               "nvidia-example-accelerator-sm90",
           "hardware class did not canonicalize a vendorless CUDA device name");

    ninfer::runtime::ContextMachineCostModel model{.transfer = transfer(), .prefill = prefill()};
    expect(model.transfer_ns(ninfer::runtime::ContextTransferDirection::DeviceToHost,
                             {.payload_bytes = 4, .copy_operations = 2}) == 16,
           "transfer operation roofline changed");
    expect(model.transfer_ns(ninfer::runtime::ContextTransferDirection::DeviceToHost,
                             {.payload_bytes = 100, .copy_operations = 1}) == 50,
           "transfer bandwidth roofline changed");
    expect(model.transfer_ns(ninfer::runtime::ContextTransferDirection::DeviceToHost, {}) == 0,
           "empty transfer has a nonzero cost");
    const std::array same_phase{
        ninfer::runtime::TransferBatchWork{
            .phase     = ninfer::runtime::MaterializationCopyPhase::PressureToHost,
            .direction = ninfer::runtime::ContextTransferDirection::DeviceToHost,
            .work      = {.payload_bytes = 4, .copy_operations = 1}},
        ninfer::runtime::TransferBatchWork{
            .phase     = ninfer::runtime::MaterializationCopyPhase::PressureToHost,
            .direction = ninfer::runtime::ContextTransferDirection::DeviceToHost,
            .work      = {.payload_bytes = 4, .copy_operations = 1}},
    };
    expect(model.transfer_batches_ns(same_phase) == 16,
           "same-phase transfers paid the batch intercept more than once");
    auto serial_phases     = same_phase;
    serial_phases[1].phase = ninfer::runtime::MaterializationCopyPhase::Candidate;
    expect(model.transfer_batches_ns(serial_phases) == 26,
           "serial transfer phases were incorrectly coalesced");
    auto independent_directions         = same_phase;
    independent_directions[1].direction = ninfer::runtime::ContextTransferDirection::HostToDevice;
    expect(model.transfer_batches_ns(independent_directions) == 26,
           "independent transfer directions were incorrectly coalesced");

    const ninfer::runtime::PrefillWork work = ninfer::runtime::make_prefill_work(100, 10, 2, 7, 8);
    expect(work.chunks == 2 && work.tokens == 10 && work.attention_pairs == 1055 &&
               work.vision_items == 2 && work.vision_patches == 7,
           "prefill feature construction changed");
    expect(ninfer::runtime::make_prefill_work(std::numeric_limits<std::uint64_t>::max(),
                                              std::numeric_limits<std::uint64_t>::max(), 0, 0, 1)
                   .attention_pairs == std::numeric_limits<std::uint64_t>::max(),
           "prefill attention work did not saturate");
    expect(model.prefill_ns(work) == 299, "prefill formula or Q32 rounding changed");
    model.prefill.chunk_ns = std::numeric_limits<std::uint64_t>::max();
    expect(model.prefill_ns({.chunks = 2}) == std::numeric_limits<std::uint64_t>::max(),
           "prefill cost did not saturate");
}

void test_kv_physical_work() {
    const ninfer::HostKVPageLayout page_major = ninfer::plan_host_kv_page_layout({
        .page_tokens        = 3,
        .device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
        .planes =
            {
                {ninfer::DType::BF16, 8, 4, 256},
                {ninfer::DType::I8, 16, 2, 256},
            },
    });
    const std::uint64_t page_payload =
        page_major.planes[0].page_payload_bytes + page_major.planes[1].page_payload_bytes;
    expect(page_payload < page_major.page_stride,
           "KV work fixture did not contain alignment padding");
    expect(ninfer::plan_host_kv_transfer_work(page_major, 3, 2) ==
               ninfer::TransferWork{.payload_bytes = 3 * page_payload, .copy_operations = 4},
           "PageMajor Host KV work counts padding or the wrong CUDA operations");
    expect(ninfer::plan_device_kv_copy_work(page_major, 3) ==
               ninfer::TransferWork{.payload_bytes = 3 * page_payload, .copy_operations = 6},
           "PageMajor Device KV work does not count one copy per plane and page");

    const ninfer::HostKVPageLayout head_major = ninfer::plan_host_kv_page_layout({
        .page_tokens        = 3,
        .device_plane_order = ninfer::PagedKVPlaneOrder::HeadMajor,
        .planes =
            {
                {ninfer::DType::I8, 8, 3, 256},
                {ninfer::DType::BF16, 4, 5, 256},
            },
    });
    const std::uint64_t head_payload =
        head_major.planes[0].page_payload_bytes + head_major.planes[1].page_payload_bytes;
    expect(ninfer::plan_host_kv_transfer_work(head_major, 7, 2) ==
               ninfer::TransferWork{.payload_bytes = 7 * head_payload, .copy_operations = 16},
           "HeadMajor Host KV work does not count one copy per head and run");
    expect(ninfer::plan_device_kv_copy_work(head_major, 7) ==
               ninfer::TransferWork{.payload_bytes = 7 * head_payload, .copy_operations = 14},
           "HeadMajor Device KV work should remain one copy per plane and page");
}

void test_schema_validation() {
    Json entries = Json::array();
    entries.push_back(prefill_entry("model", "weights", prefill()));
    const Json valid =
        document(Json::array({machine("machine", transfer_json(transfer()), entries)}));
    const auto parsed = ninfer::runtime::parse_context_cost_presets(valid.dump(), "test");
    expect(parsed.size() == 1 && parsed[0].transfer == transfer() &&
               parsed[0].prefill.size() == 1 && parsed[0].prefill[0].cost == prefill(),
           "valid layered preset did not parse");

    Json duplicate_machine = valid;
    duplicate_machine["machines"].push_back(duplicate_machine["machines"][0]);
    expect_throw(
        [&] {
            (void)ninfer::runtime::parse_context_cost_presets(duplicate_machine.dump(),
                                                              "duplicate-machine");
        },
        "duplicate hardware entry was accepted");

    Json duplicate_prefill = valid;
    duplicate_prefill["machines"][0]["prefill"].push_back(
        duplicate_prefill["machines"][0]["prefill"][0]);
    expect_throw(
        [&] {
            (void)ninfer::runtime::parse_context_cost_presets(duplicate_prefill.dump(),
                                                              "duplicate-prefill");
        },
        "duplicate model prefill entry was accepted");

    Json invalid                                              = valid;
    invalid["machines"][0]["transfer"]["d2h"]["operation_ns"] = -1;
    expect_throw(
        [&] { (void)ninfer::runtime::parse_context_cost_presets(invalid.dump(), "negative"); },
        "negative transfer coefficient was accepted");

    Json prefill_only                       = valid;
    prefill_only["machines"][0]["transfer"] = nullptr;
    expect(
        ninfer::runtime::parse_context_cost_presets(prefill_only.dump(), "prefill-only").size() ==
            1,
        "independent prefill-only machine entry was rejected");
    prefill_only["machines"][0]["prefill"] = Json::array();
    expect_throw(
        [&] { (void)ninfer::runtime::parse_context_cost_presets(prefill_only.dump(), "empty"); },
        "empty machine cost entry was accepted");
}

void test_resolution_and_atomic_upserts() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        ("ninfer-context-cost-test-" + std::to_string(static_cast<long long>(::getpid())));
    std::filesystem::create_directories(directory);
    const std::filesystem::path path = directory / "presets.json";
    try {
        const ninfer::runtime::ContextCostIdentity compiled_identity{
            .hardware_class = "nvidia-geforce-rtx-5090-sm120",
            .model_id       = "qwen3.6-27b",
            .weights_id     = "groupwise-int",
        };
        const auto compiled = ninfer::runtime::resolve_context_machine_cost(compiled_identity);
        expect(
            compiled.summary.transfer_source == ninfer::ContextCostPresetSource::CompiledDefault &&
                compiled.summary.prefill_source == ninfer::ContextCostPresetSource::CompiledDefault,
            "compiled transfer and prefill defaults did not resolve independently");

        const ninfer::runtime::ContextCostIdentity unknown{
            .hardware_class = "unmeasured-machine",
            .model_id       = "unmeasured-model",
            .weights_id     = "unmeasured-weights",
        };
        const auto generic = ninfer::runtime::resolve_context_machine_cost(unknown);
        expect(
            generic.summary.transfer_source == ninfer::ContextCostPresetSource::GenericDefault &&
                generic.summary.prefill_source == ninfer::ContextCostPresetSource::GenericDefault &&
                generic.model.transfer_ns(ninfer::runtime::ContextTransferDirection::DeviceToHost,
                                          {.payload_bytes = 1024, .copy_operations = 1}) > 0 &&
                generic.model.prefill_ns({.tokens = 1}) > 0,
            "unknown identity did not retain a numerical cost model");

        ninfer::runtime::upsert_context_transfer_cost_atomic(path, unknown.hardware_class,
                                                             transfer(77), R"({"run":1})");
        auto resolved = ninfer::runtime::resolve_context_machine_cost(unknown, path);
        expect(resolved.model.transfer[0].batch_ns == 77 &&
                   resolved.summary.transfer_source == ninfer::ContextCostPresetSource::External &&
                   resolved.summary.prefill_source ==
                       ninfer::ContextCostPresetSource::GenericDefault,
               "external transfer did not layer over generic prefill");

        ninfer::runtime::upsert_context_prefill_cost_atomic(path, unknown, prefill(91),
                                                            R"({"run":2})");
        resolved = ninfer::runtime::resolve_context_machine_cost(unknown, path);
        expect(resolved.model.transfer[0].batch_ns == 77 && resolved.model.prefill.chunk_ns == 91 &&
                   resolved.summary.transfer_source == ninfer::ContextCostPresetSource::External &&
                   resolved.summary.prefill_source == ninfer::ContextCostPresetSource::External,
               "independent external transfer/prefill entries did not compose");

        const ninfer::runtime::ContextCostIdentity other_model{
            .hardware_class = unknown.hardware_class,
            .model_id       = "other",
            .weights_id     = "other",
        };
        const auto layered_miss = ninfer::runtime::resolve_context_machine_cost(other_model, path);
        expect(layered_miss.summary.transfer_source == ninfer::ContextCostPresetSource::External &&
                   layered_miss.summary.prefill_source ==
                       ninfer::ContextCostPresetSource::GenericDefault,
               "external model miss incorrectly discarded the matching machine transfer");

        ninfer::runtime::upsert_context_transfer_cost_atomic(path, unknown.hardware_class,
                                                             transfer(123), R"({"run":3})");
        const auto parsed = ninfer::runtime::parse_context_cost_presets(read_file(path), "file");
        expect(parsed.size() == 1 && parsed[0].transfer->at(0).batch_ns == 123 &&
                   parsed[0].prefill.size() == 1 && parsed[0].prefill[0].cost.chunk_ns == 91,
               "transfer replacement did not preserve the model prefill list");

        const std::string before = read_file(path);
        expect_throw(
            [&] {
                ninfer::runtime::upsert_context_prefill_cost_atomic(path, unknown, prefill(),
                                                                    "not-json");
            },
            "invalid provenance was accepted");
        expect(read_file(path) == before, "failed upsert modified the preset file");
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }
    std::filesystem::remove_all(directory);
}

} // namespace

int main() {
    test_exact_evaluation();
    test_kv_physical_work();
    test_schema_validation();
    test_resolution_and_atomic_upserts();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
