#include "ninfer/core/weight_store_parser.h"
#include "ninfer/model/processor.h"
#include "ninfer/text/chat_template.h"
#include "ninfer/text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::byte> read_prefix(const std::filesystem::path& path, std::size_t bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("failed to open q5090 file: " + path.string()); }
    std::vector<std::byte> out(bytes);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    if (in.gcount() != static_cast<std::streamsize>(bytes)) {
        throw std::runtime_error("q5090 file is truncated");
    }
    return out;
}

ninfer::Q5090TokenizerBundle load_tokenizer(const std::filesystem::path& path) {
    const std::uint64_t size                  = std::filesystem::file_size(path);
    const std::vector<std::byte> header_bytes = read_prefix(path, 4096);
    const ninfer::ParsedQ5090Header header       = ninfer::parse_q5090_header(header_bytes, size);
    const std::vector<std::byte> metadata     = read_prefix(path, header.payload_offset);
    ninfer::ParsedQ5090File parsed               = ninfer::parse_q5090_catalog(metadata, size);
    return std::move(parsed.tokenizer);
}

void write_binary(const std::filesystem::path& path, const std::vector<float>& values) {
    std::ofstream out(path, std::ios::binary);
    if (!out) { throw std::runtime_error("failed to open patch output: " + path.string()); }
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
}

std::uint64_t fnv1a(const std::vector<float>& values) {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    const auto* bytes  = reinterpret_cast<const unsigned char*>(values.data());
    for (std::size_t i = 0; i < values.size() * sizeof(float); ++i) {
        hash ^= bytes[i];
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 6) {
            std::cerr << "usage: " << argv[0]
                      << " WEIGHTS MESSAGES OUTPUT_JSON [PATCHES_F32] [--no-thinking]\n";
            return 2;
        }
        const std::filesystem::path weights       = argv[1];
        const std::filesystem::path messages_path = argv[2];
        const std::filesystem::path output_path   = argv[3];
        std::filesystem::path patches_path;
        bool thinking = true;
        for (int i = 4; i < argc; ++i) {
            if (std::string(argv[i]) == "--no-thinking") {
                thinking = false;
            } else if (patches_path.empty()) {
                patches_path = argv[i];
            } else {
                throw std::invalid_argument("unexpected preprocess argument");
            }
        }

        ninfer::text::QwenTokenizer tokenizer(load_tokenizer(weights));
        const std::vector<ninfer::text::ChatMessage> messages =
            ninfer::text::read_messages_json(messages_path);
        ninfer::model::Processor processor(tokenizer);
        ninfer::text::ChatRenderOptions render;
        render.enable_thinking           = thinking;
        ninfer::model::ProcessedInput input = processor.process(messages, render);

        nlohmann::json json;
        json["input_ids"]   = input.input_ids;
        json["token_types"] = input.token_types;
        json["positions"]   = nlohmann::json::array();
        for (int axis = 0; axis < 3; ++axis) {
            const auto values = input.position_axis(axis);
            json["positions"].push_back(std::vector<std::int32_t>(values.begin(), values.end()));
        }
        json["rope_delta"]    = input.rope_delta;
        json["patch_shape"]   = {input.stats.raw_patches, 1536};
        json["patch_fnv1a64"] = fnv1a(input.patches);
        json["stats"]         = {
            {"media_items", input.stats.media_items},
            {"raw_patches", input.stats.raw_patches},
            {"vision_tokens", input.stats.vision_tokens},
            {"attention_pairs", input.stats.attention_pairs},
            {"prompt_tokens", input.stats.prompt_tokens},
            {"patch_bytes", input.stats.patch_bytes},
        };
        json["vision_items"] = nlohmann::json::array();
        for (const ninfer::model::VisionItem& item : input.vision_items) {
            nlohmann::json spans = nlohmann::json::array();
            for (const ninfer::model::TokenSpan& span : item.token_spans) {
                spans.push_back({span.begin, span.count});
            }
            json["vision_items"].push_back({
                {"modality", item.modality == ninfer::model::Modality::Image ? "image" : "video"},
                {"grid", {item.grid.t, item.grid.h, item.grid.w}},
                {"patch_begin", item.patch_begin},
                {"patch_count", item.patch_count},
                {"timestamps", item.timestamps},
                {"token_spans", std::move(spans)},
            });
        }
        std::ofstream output(output_path);
        if (!output) { throw std::runtime_error("failed to open JSON output"); }
        output << json.dump(2) << '\n';
        if (!patches_path.empty()) { write_binary(patches_path, input.patches); }
        std::cerr << input.stats.summary() << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
