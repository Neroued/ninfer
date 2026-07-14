#include "ninfer/model/processor.h"
#include "ninfer/runtime/engine.h"
#include "ninfer/text/chat_template.h"
#include "ninfer/text/tokenizer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

class TempDir {
public:
    TempDir() {
        path = std::filesystem::temp_directory_path() /
               ("ninfer_vision_e2e_" + std::to_string(::getpid()));
        std::filesystem::create_directories(path);
    }

    ~TempDir() { std::filesystem::remove_all(path); }

    std::filesystem::path path;
};

void write_color(const std::filesystem::path& path, int width, int height,
                 std::array<std::uint8_t, 3> rgb) {
    std::ofstream file(path, std::ios::binary);
    file << "P6\n" << width << ' ' << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        file.write(reinterpret_cast<const char*>(rgb.data()),
                   static_cast<std::streamsize>(rgb.size()));
    }
}

ninfer::model::ProcessedInput prepare(const ninfer::model::Processor& processor,
                                   const std::filesystem::path& image) {
    ninfer::text::ChatMessage message;
    message.role = "user";
    message.parts.push_back(ninfer::text::ChatPart::image(ninfer::media::Source{
        ninfer::media::SourceKind::Path, image.string(), "image/x-portable-pixmap"}));
    message.parts.push_back(ninfer::text::ChatPart::text_part("图片是什么颜色？只回答颜色。"));
    ninfer::text::ChatRenderOptions render;
    render.enable_thinking = false;
    return processor.process({message}, render);
}

std::vector<int> generate(ninfer::Engine& engine, ninfer::text::QwenTokenizer& tokenizer,
                          const ninfer::model::ProcessedInput& input) {
    std::vector<int> tokens;
    tokens.push_back(engine.prefill(input));
    const std::vector<int> stops = tokenizer.default_stop_token_ids();
    while (tokens.size() < 32 &&
           std::find(stops.begin(), stops.end(), tokens.back()) == stops.end()) {
        tokens.push_back(engine.decode_step());
    }
    return tokens;
}

struct SequenceResult {
    std::vector<std::vector<int>> tokens;
    std::string red;
    std::string blue;
    std::int32_t red_delta  = 0;
    std::int32_t blue_delta = 0;
};

SequenceResult run_sequence(const std::filesystem::path& weights,
                            const std::filesystem::path& red_path,
                            const std::filesystem::path& blue_path, bool use_cuda_graph) {
    ninfer::EngineOptions options;
    options.max_ctx          = 256;
    options.mtp_draft_tokens = 2;
    options.use_cuda_graph   = use_cuda_graph;
    ninfer::Engine engine(options);
    engine.load(weights.string());
    ninfer::text::QwenTokenizer tokenizer(engine.take_tokenizer_bundle());
    engine.set_stop_token_ids(tokenizer.default_stop_token_ids());
    ninfer::model::ProcessorOptions processor_options;
    processor_options.image_min_pixels = 64 * 64;
    processor_options.image_max_pixels = 128 * 64;
    ninfer::model::Processor processor(tokenizer, processor_options);
    const ninfer::model::ProcessedInput red_input  = prepare(processor, red_path);
    const ninfer::model::ProcessedInput blue_input = prepare(processor, blue_path);

    SequenceResult result;
    result.tokens.push_back(engine.generate(std::vector<int>{1}, 4));
    result.tokens.push_back(generate(engine, tokenizer, red_input));
    ninfer::model::ProcessedInput invalid = red_input;
    invalid.positions.pop_back();
    try {
        (void)engine.prefill(invalid);
        throw std::runtime_error("invalid multimodal prefill unexpectedly succeeded");
    } catch (const std::invalid_argument&) {
        if (engine.position() != 0 || engine.last_prefix_cache_hit() != 0) {
            throw std::runtime_error("failed multimodal prefill left a resident cache identity");
        }
        bool decode_rejected = false;
        try {
            (void)engine.decode_step();
        } catch (const std::runtime_error&) { decode_rejected = true; }
        if (!decode_rejected) {
            throw std::runtime_error("decode succeeded after failed multimodal prefill");
        }
    }
    result.tokens.push_back(generate(engine, tokenizer, blue_input));
    result.tokens.push_back(engine.generate(std::vector<int>{2}, 4));
    result.red        = tokenizer.decode(result.tokens[1]);
    result.blue       = tokenizer.decode(result.tokens[2]);
    result.red_delta  = red_input.rope_delta;
    result.blue_delta = blue_input.rope_delta;
    return result;
}

} // namespace

int main() {
    const std::filesystem::path weights =
        std::filesystem::path(NINFER_SOURCE_DIR) / "out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus";
    if (!std::filesystem::exists(weights)) {
        std::cout << "SKIP: real q5090 file not present\n";
        return 0;
    }
    std::size_t free_bytes   = 0;
    std::size_t total_bytes  = 0;
    const cudaError_t memory = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (memory == cudaErrorNoDevice || memory == cudaErrorInsufficientDriver) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    if (memory != cudaSuccess || free_bytes < 18ULL * 1024 * 1024 * 1024) {
        std::cout << "SKIP: insufficient free GPU memory for real Vision E2E\n";
        return 0;
    }

    TempDir temp;
    const auto red_path  = temp.path / "red.ppm";
    const auto blue_path = temp.path / "blue.ppm";
    write_color(red_path, 64, 64, {255, 0, 0});
    write_color(blue_path, 128, 64, {0, 0, 255});

    const SequenceResult eager = run_sequence(weights, red_path, blue_path, false);
    const SequenceResult graph = run_sequence(weights, red_path, blue_path, true);
    std::cout << "VISION_E2E red=" << graph.red << " blue=" << graph.blue << '\n';
    if (graph.red_delta == graph.blue_delta) {
        std::cerr << "Vision graph regression inputs did not produce distinct rope_delta values\n";
        return 1;
    }
    if (graph.tokens != eager.tokens) {
        std::cerr << "Vision/MTP CUDA Graph sequence diverged from eager execution\n";
        return 1;
    }
    if (graph.red.find("红") == std::string::npos || graph.blue.find("蓝") == std::string::npos ||
        graph.red == graph.blue) {
        std::cerr << "Vision E2E did not distinguish canonical red/blue images\n";
        return 1;
    }
    return 0;
}
