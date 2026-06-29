#include "qus/core/arena.h"
#include "qus/core/device.h"
#include "qus/core/weight_store.h"
#include "qus/core/weight_store_parser.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int fail(std::string_view message) {
    std::cerr << message << '\n';
    return 1;
}

std::filesystem::path make_fixture() {
    const auto path = std::filesystem::temp_directory_path() / "qus_q5090_weight_store_fixture.qus";
    const std::filesystem::path script =
        std::filesystem::path(QUS_SOURCE_DIR) / "tests/fixtures/make_q5090_fixture.py";
    const std::string command =
        "python3 \"" + script.string() + "\" --out \"" + path.string() + "\"";
    if (std::system(command.c_str()) != 0) { throw std::runtime_error("fixture generator failed"); }
    return path;
}

std::vector<std::byte> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("failed to open fixture"); }
    const std::vector<char> chars{std::istreambuf_iterator<char>(in),
                                  std::istreambuf_iterator<char>()};
    std::vector<std::byte> bytes(chars.size());
    std::memcpy(bytes.data(), chars.data(), chars.size());
    return bytes;
}

qus::Q5090Expectations expectations() {
    qus::Q5090Expectations expected;
    expected.layer_count             = 64;
    expected.hidden_size             = 5120;
    expected.intermediate_size       = 17408;
    expected.vocab_size              = 248320;
    expected.num_attention_heads     = 24;
    expected.num_key_value_heads     = 4;
    expected.head_dim                = 256;
    expected.gdn_key_heads           = 16;
    expected.gdn_value_heads         = 48;
    expected.gdn_key_head_dim        = 128;
    expected.gdn_value_head_dim      = 128;
    expected.gdn_conv_width          = 4;
    expected.full_attention_interval = 4;
    expected.max_position_embeddings = 262144;
    return expected;
}

const qus::ParsedQ5090Tensor& find_tensor(const qus::ParsedQ5090File& parsed,
                                          std::string_view name) {
    for (const qus::ParsedQ5090Tensor& tensor : parsed.tensors) {
        if (tensor.name == name) { return tensor; }
    }
    throw std::runtime_error("tensor not found");
}

std::vector<std::byte> payload_bytes(const std::vector<std::byte>& file,
                                     const qus::ParsedQ5090Tensor& tensor) {
    const auto begin = file.begin() + static_cast<std::ptrdiff_t>(tensor.payload_offset);
    const auto end   = begin + static_cast<std::ptrdiff_t>(tensor.payload_bytes);
    return std::vector<std::byte>(begin, end);
}

int expect_device_bytes(const void* device, const std::vector<std::byte>& expected,
                        std::string_view label) {
    std::vector<std::byte> actual(expected.size());
    const cudaError_t err =
        cudaMemcpy(actual.data(), device, actual.size(), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " cudaMemcpy failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    if (actual != expected) {
        std::cerr << label << " payload mismatch\n";
        return 1;
    }
    return 0;
}

int expect_counts(const qus::WeightStore& store, std::uint64_t loaded_bytes) {
    int failures = 0;
    failures += store.tensor_count() == 10 ? 0 : fail("tensor_count mismatch");
    failures += store.quant_count() == 7 ? 0 : fail("quant_count mismatch");
    failures +=
        store.module_tensor_count(qus::ModuleKind::TextCore) == 6 ? 0 : fail("TEXT count mismatch");
    failures +=
        store.module_tensor_count(qus::ModuleKind::MtpDraft) == 2 ? 0 : fail("MTP count mismatch");
    failures += store.module_tensor_count(qus::ModuleKind::VisionEncoder) == 2
                    ? 0
                    : fail("VISION count mismatch");
    failures +=
        store.loaded_payload_bytes() == loaded_bytes ? 0 : fail("loaded payload bytes mismatch");
    return failures;
}

int expect_default_text_load(const qus::WeightStore& store, const qus::ParsedQ5090File& parsed,
                             const std::vector<std::byte>& file) {
    int failures = 0;
    failures += expect_counts(store, parsed.modules[0].payload_bytes);
    failures += store.module_loaded(qus::ModuleKind::TextCore) ? 0 : fail("TEXT not loaded");
    failures += !store.module_loaded(qus::ModuleKind::MtpDraft) ? 0 : fail("MTP loaded by default");
    failures +=
        !store.module_loaded(qus::ModuleKind::VisionEncoder) ? 0 : fail("VISION loaded by default");

    const auto& text_q = find_tensor(parsed, "layers.0.mlp.down_proj.weight");
    const qus::Weight* text_weight = store.qweight("layers.0.mlp.down_proj.weight");
    failures += text_weight != nullptr ? 0 : fail("missing text quant weight");
    if (text_weight != nullptr) {
        failures += text_weight->payload != nullptr ? 0 : fail("text quant payload is null");
        failures += text_weight->payload_bytes == text_q.payload_bytes
                        ? 0
                        : fail("text payload bytes mismatch");
        failures += text_weight->layout == qus::QuantLayout::RowSplit ? 0 : fail("text layout mismatch");
        failures += text_weight->scales != nullptr ? 0 : fail("text scales null");
        failures +=
            expect_device_bytes(text_weight->payload, payload_bytes(file, text_q), "text qweight");
    }

    const qus::Weight* gate = store.qweight("layers.0.mlp.gate_proj.weight");
    const qus::Weight* up   = store.qweight("layers.0.mlp.up_proj.weight");
    failures += gate != nullptr ? 0 : fail("missing fused gate segment");
    failures += up != nullptr ? 0 : fail("missing fused up segment");
    if (gate != nullptr && up != nullptr) {
        failures += gate->n == 5 && gate->k == 7 ? 0 : fail("gate segment shape mismatch");
        failures += up->n == 4 && up->k == 7 ? 0 : fail("up segment shape mismatch");
        failures += gate->payload == up->payload ? 0 : fail("fused segments should share payload");
        failures += gate->qdata != nullptr && gate->scales != nullptr ? 0 : fail("gate planes null");
        failures += up->qdata != nullptr && up->scales != nullptr ? 0 : fail("up planes null");
    }

    const auto& text_tensor_meta =
        find_tensor(parsed, "layers.0.input_layernorm.weight");
    const qus::Tensor* text_tensor = store.tensor("layers.0.input_layernorm.weight");
    failures += text_tensor != nullptr ? 0 : fail("missing text tensor");
    if (text_tensor != nullptr) {
        failures += text_tensor->data != nullptr ? 0 : fail("text tensor payload is null");
        failures += expect_device_bytes(text_tensor->data, payload_bytes(file, text_tensor_meta),
                                        "text tensor");
    }

    const qus::Weight* by_source = store.qweight(
        qus::ModuleKind::TextCore, static_cast<std::uint32_t>(qus::SourceKind::MlpDown), 0);
    failures += by_source == text_weight ? 0 : fail("source lookup mismatch");

    const qus::Weight* mtp = store.qweight("mtp.fc.weight");
    failures += mtp != nullptr ? 0 : fail("missing MTP metadata");
    if (mtp != nullptr) {
        failures += mtp->payload == nullptr ? 0 : fail("MTP payload loaded by default");
    }
    const qus::Weight* vision = store.qweight("model.visual.patch_embed.proj.weight");
    failures += vision != nullptr ? 0 : fail("missing VISION metadata");
    if (vision != nullptr) {
        failures += vision->payload == nullptr ? 0 : fail("VISION payload loaded by default");
    }
    return failures;
}

int expect_module_payload(const qus::WeightStore& store, const qus::ParsedQ5090File& parsed,
                          const std::vector<std::byte>& file, std::string_view name,
                          bool should_be_loaded) {
    const qus::ParsedQ5090Tensor& meta = find_tensor(parsed, name);
    const qus::Weight* weight          = store.qweight(name);
    if (weight == nullptr) { return fail("missing quant weight metadata"); }
    if (!should_be_loaded) {
        return weight->payload == nullptr ? 0 : fail("unexpected loaded payload");
    }
    int failures = 0;
    failures += weight->payload != nullptr ? 0 : fail("expected loaded payload");
    failures +=
        weight->payload_bytes == meta.payload_bytes ? 0 : fail("module payload size mismatch");
    if (weight->payload != nullptr) {
        failures += expect_device_bytes(weight->payload, payload_bytes(file, meta), name);
    }
    return failures;
}

template <typename Exception, typename Fn>
int expect_throws_without_arena_growth(qus::DeviceArena& arena, Fn&& fn, std::string_view label) {
    const std::size_t before_used = arena.used();
    const std::size_t before_peak = arena.peak_used();
    try {
        fn();
    } catch (const Exception&) {
        int failures = 0;
        failures += arena.used() == before_used ? 0 : fail("arena used grew after failed load");
        failures += arena.peak_used() == before_peak ? 0 : fail("arena peak grew after failed load");
        return failures;
    }
    std::cerr << label << " did not throw\n";
    return 1;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 0;
    }

    int failures                             = 0;
    const std::filesystem::path fixture_path = make_fixture();
    const std::vector<std::byte> file        = read_file(fixture_path);
    const qus::ParsedQ5090File parsed        = qus::parse_q5090_file(file, expectations());
    qus::DeviceContext ctx(0);

    qus::DeviceArena default_arena(65536);
    qus::WeightStore default_store(expectations());
    default_store.load(fixture_path.c_str(), default_arena, ctx);
    failures += expect_default_text_load(default_store, parsed, file);
    failures += default_arena.used() >= default_store.loaded_payload_bytes()
                    ? 0
                    : fail("default arena used is smaller than loaded payload bytes");
    failures += default_arena.peak_used() >= default_arena.used()
                    ? 0
                    : fail("default arena peak is smaller than used");
    default_arena.reset_peak();
    failures += default_arena.peak_used() == default_arena.used()
                    ? 0
                    : fail("default arena reset_peak did not reset to current used");
    default_store.clear();
    failures += default_store.tensor_count() == 0 ? 0 : fail("clear tensor_count mismatch");
    failures += default_store.quant_count() == 0 ? 0 : fail("clear quant_count mismatch");
    failures += default_store.loaded_payload_bytes() == 0 ? 0 : fail("clear loaded bytes mismatch");
    failures += !default_store.module_loaded(qus::ModuleKind::TextCore)
                    ? 0
                    : fail("clear TEXT module still loaded");

    qus::LoadOptions mtp_options;
    mtp_options.load_mtp = true;
    qus::DeviceArena mtp_arena(65536);
    qus::WeightStore mtp_store(expectations());
    mtp_store.load(fixture_path.c_str(), mtp_arena, ctx, mtp_options);
    failures += mtp_store.module_loaded(qus::ModuleKind::MtpDraft) ? 0 : fail("MTP not loaded");
    failures += !mtp_store.module_loaded(qus::ModuleKind::VisionEncoder)
                    ? 0
                    : fail("VISION loaded with MTP");
    failures += expect_module_payload(mtp_store, parsed, file, "mtp.fc.weight", true);
    failures += expect_module_payload(mtp_store, parsed, file,
                                      "model.visual.patch_embed.proj.weight", false);

    qus::LoadOptions vision_options;
    vision_options.load_vision = true;
    qus::DeviceArena vision_arena(65536);
    qus::WeightStore vision_store(expectations());
    vision_store.load(fixture_path.c_str(), vision_arena, ctx, vision_options);
    failures +=
        vision_store.module_loaded(qus::ModuleKind::VisionEncoder) ? 0 : fail("VISION not loaded");
    failures +=
        !vision_store.module_loaded(qus::ModuleKind::MtpDraft) ? 0 : fail("MTP loaded with VISION");
    failures += expect_module_payload(vision_store, parsed, file, "mtp.fc.weight", false);
    failures += expect_module_payload(vision_store, parsed, file,
                                      "model.visual.patch_embed.proj.weight", true);

    qus::LoadOptions all_options;
    all_options.load_mtp    = true;
    all_options.load_vision = true;
    qus::DeviceArena all_arena(65536);
    qus::WeightStore all_store(expectations());
    all_store.load(fixture_path.c_str(), all_arena, ctx, all_options);
    failures +=
        all_store.module_loaded(qus::ModuleKind::TextCore) ? 0 : fail("TEXT not loaded with all");
    failures +=
        all_store.module_loaded(qus::ModuleKind::MtpDraft) ? 0 : fail("MTP not loaded with all");
    failures += all_store.module_loaded(qus::ModuleKind::VisionEncoder)
                    ? 0
                    : fail("VISION not loaded with all");
    failures += expect_module_payload(all_store, parsed, file, "mtp.fc.weight", true);
    failures += expect_module_payload(all_store, parsed, file,
                                      "model.visual.patch_embed.proj.weight", true);

    qus::DeviceArena required_arena(65536);
    failures += expect_throws_without_arena_growth<std::runtime_error>(
        required_arena,
        [&] {
            qus::LoadOptions options;
            options.required_text_tensors.push_back("missing.text.tensor");
            qus::WeightStore store(expectations());
            store.load(fixture_path.c_str(), required_arena, ctx, options);
        },
        "missing required text tensor");

    qus::DeviceArena small_arena(1024);
    failures += expect_throws_without_arena_growth<std::bad_alloc>(
        small_arena,
        [&] {
            qus::WeightStore store(expectations());
            store.load(fixture_path.c_str(), small_arena, ctx);
        },
        "small arena");

    return failures == 0 ? 0 : fail("weight store q5090 test failed");
}
