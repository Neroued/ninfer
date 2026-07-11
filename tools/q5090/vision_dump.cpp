#include "qus/core/device.h"
#include "qus/core/weight_store.h"
#include "qus/model/processor.h"
#include "qus/model/vision.h"
#include "qus/text/chat_template.h"
#include "qus/text/tokenizer.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;

float bf16_to_f32(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

struct DumpContext {
    std::filesystem::path root;
    Json records = Json::array();
};

void dump_tensor(void* opaque, qus::model::VisionTapId id, int layer, const qus::Tensor& tensor,
                 cudaStream_t stream) {
    if (id == qus::model::VisionTapId::Block && layer != 0 && layer != 13 && layer != 26) {
        return;
    }
    auto& context = *static_cast<DumpContext*>(opaque);
    std::string name;
    if (id == qus::model::VisionTapId::PatchEmbed) {
        name = "vision/patch_embed";
    } else if (id == qus::model::VisionTapId::Block) {
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "vision/block_%02d", layer);
        name = buffer;
    } else {
        name = "vision/merger";
    }
    const std::string file_name = name.substr(name.find('/') + 1) + ".f32";
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(tensor.numel()));
    CUDA_CHECK(cudaMemcpy(bits.data(), tensor.data, bits.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));
    std::vector<float> values(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) { values[i] = bf16_to_f32(bits[i]); }
    std::ofstream file(context.root / file_name, std::ios::binary);
    if (!file) { throw std::runtime_error("failed to open Vision dump output"); }
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!file) { throw std::runtime_error("failed to write Vision dump output"); }
    context.records.push_back(Json{{"name", name},
                                   {"file", file_name},
                                   {"shape", {tensor.ne[1], tensor.ne[0]}},
                                   {"source_dtype", "bf16"},
                                   {"phase", "prefill"},
                                   {"step", 0},
                                   {"chunk", 0},
                                   {"position_begin", 0},
                                   {"position_end", tensor.ne[1]}});
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5 || (argc == 5 && std::string(argv[4]) != "--no-thinking")) {
        std::cerr << "usage: " << argv[0] << " WEIGHTS MESSAGES OUT_DIR [--no-thinking]\n";
        return 2;
    }
    try {
        const std::filesystem::path out_dir = argv[3];
        std::filesystem::create_directories(out_dir);
        qus::WeightStore weights;
        qus::LoadOptions load_options;
        load_options.load_vision = true;
        qus::DeviceContext context;
        weights.load(argv[1], context, load_options);
        qus::text::QwenTokenizer tokenizer(weights.take_tokenizer_bundle());
        qus::model::Processor processor(tokenizer);
        qus::text::ChatRenderOptions render_options;
        render_options.enable_thinking         = argc != 5;
        const auto messages                    = qus::text::read_messages_json(argv[2]);
        const qus::model::ProcessedInput input = processor.process(messages, render_options);
        qus::WorkspaceArena workspace(qus::model::Qwen3_6_Vision::workspace_bytes(input));
        qus::model::Qwen3_6_Vision vision(context, weights);
        DumpContext dump{out_dir};
        (void)vision.encode(input, workspace, &dump, dump_tensor);
        context.synchronize();
        Json manifest{{"format", "qus_activation_dump_v1"},
                      {"runtime", "cpp"},
                      {"raw_patches", input.stats.raw_patches},
                      {"vision_tokens", input.stats.vision_tokens},
                      {"tensors", std::move(dump.records)}};
        std::ofstream manifest_file(out_dir / "manifest.json");
        manifest_file << manifest.dump(2) << '\n';
        if (!manifest_file) { throw std::runtime_error("failed to write Vision dump manifest"); }
        std::cout << input.stats.summary() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
