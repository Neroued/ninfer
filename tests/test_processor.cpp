#include "ninfer/model/processor.h"

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

struct TempDir {
    std::filesystem::path path;

    TempDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() / ("ninfer_processor_" + std::to_string(stamp));
        if (!std::filesystem::create_directory(path)) {
            throw std::runtime_error("failed to create processor temp directory");
        }
    }

    ~TempDir() { std::filesystem::remove_all(path); }
};

ninfer::Q5090TokenizerBundle tokenizer_bundle() {
    nlohmann::json vocab = {
        {"u", 0},  {"s", 1},  {"e", 2},  {"r", 3},  {"x", 4},  {"Ċ", 5},  {"Ġ", 6},  {"a", 7},
        {"i", 8},  {"t", 9},  {"n", 10}, {"k", 11}, {"v", 12}, {"d", 13}, {"o", 14}, {"c", 15},
        {".", 16}, {"0", 17}, {"1", 18}, {"2", 19}, {"3", 20}, {"4", 21}, {"5", 22}, {"6", 23},
        {"7", 24}, {"8", 25}, {"9", 26}, {"<", 27}, {">", 28}, {"/", 29}, {"h", 30}, {"m", 31},
        {"g", 32}, {"p", 33}, {"l", 34}, {"f", 35}, {"b", 36}, {"y", 37}, {"w", 38}, {"_", 39},
        {"|", 40}, {":", 41}, {"P", 42}, {"V", 43}, {"T", 44},
    };
    auto added = [](int id, const char* content) {
        return nlohmann::json{{"id", id},        {"content", content}, {"single_word", false},
                              {"lstrip", false}, {"rstrip", false},    {"normalized", false},
                              {"special", true}};
    };
    nlohmann::json tokens = nlohmann::json::array({
        added(248045, "<|im_start|>"),
        added(248046, "<|im_end|>"),
        added(248053, "<|vision_start|>"),
        added(248054, "<|vision_end|>"),
        added(248056, "<|image_pad|>"),
        added(248057, "<|video_pad|>"),
        added(248068, "<think>"),
        added(248069, "</think>"),
    });
    ninfer::Q5090TokenizerBundle bundle;
    bundle.tokenizer_json = nlohmann::json{
        {"model", {{"type", "BPE"}, {"vocab", vocab}}},
        {"added_tokens",
         tokens}}.dump();
    bundle.merges_txt             = "#version: 0.2\n";
    bundle.generation_config_json = R"({"eos_token_id":[248046]})";
    return bundle;
}

std::filesystem::path write_ppm(const std::filesystem::path& dir) {
    const std::filesystem::path path = dir / "gradient.ppm";
    std::ofstream out(path, std::ios::binary);
    out << "P6\n64 64\n255\n";
    for (int y = 0; y < 64; ++y) {
        for (int x = 0; x < 64; ++x) {
            const std::uint8_t pixel[3] = {static_cast<std::uint8_t>(x),
                                           static_cast<std::uint8_t>(y),
                                           static_cast<std::uint8_t>(x + y)};
            out.write(reinterpret_cast<const char*>(pixel), sizeof(pixel));
        }
    }
    return path;
}

ninfer::media::Source path_source(const std::filesystem::path& path) {
    return ninfer::media::Source{ninfer::media::SourceKind::Path, path.string(),
                              "image/x-portable-pixmap"};
}

std::string base64(const std::vector<std::uint8_t>& bytes) {
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((bytes.size() + 2) / 3 * 4);
    for (std::size_t i = 0; i < bytes.size(); i += 3) {
        const std::uint32_t value =
            static_cast<std::uint32_t>(bytes[i]) << 16U |
            (i + 1 < bytes.size() ? static_cast<std::uint32_t>(bytes[i + 1]) << 8U : 0U) |
            (i + 2 < bytes.size() ? bytes[i + 2] : 0U);
        out.push_back(alphabet[(value >> 18U) & 63U]);
        out.push_back(alphabet[(value >> 12U) & 63U]);
        out.push_back(i + 1 < bytes.size() ? alphabet[(value >> 6U) & 63U] : '=');
        out.push_back(i + 2 < bytes.size() ? alphabet[value & 63U] : '=');
    }
    return out;
}

ninfer::media::Source data_source(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    return ninfer::media::Source{ninfer::media::SourceKind::Data,
                              "data:image/x-portable-pixmap;base64," + base64(bytes),
                              "image/x-portable-pixmap"};
}

class HttpFixture {
public:
    explicit HttpFixture(const std::filesystem::path& path) {
        std::ifstream in(path, std::ios::binary);
        body_   = std::vector<char>((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
        socket_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (socket_ < 0) { throw std::runtime_error("failed to create HTTP fixture socket"); }
        sockaddr_in address{};
        address.sin_family      = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port        = 0;
        if (::bind(socket_, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0 ||
            ::listen(socket_, 1) != 0) {
            ::close(socket_);
            throw std::runtime_error("failed to bind HTTP fixture socket");
        }
        socklen_t size = sizeof(address);
        if (::getsockname(socket_, reinterpret_cast<sockaddr*>(&address), &size) != 0) {
            ::close(socket_);
            throw std::runtime_error("failed to inspect HTTP fixture socket");
        }
        port_   = ntohs(address.sin_port);
        worker_ = std::thread([this] { serve(); });
    }

    ~HttpFixture() {
        if (worker_.joinable()) { worker_.join(); }
        if (socket_ >= 0) { ::close(socket_); }
    }

    HttpFixture(const HttpFixture&)            = delete;
    HttpFixture& operator=(const HttpFixture&) = delete;

    [[nodiscard]] std::string url() const {
        return "http://127.0.0.1:" + std::to_string(port_) + "/image.ppm";
    }

private:
    static void send_all(int socket, const char* data, std::size_t size) {
        while (size != 0) {
            const ssize_t sent = ::send(socket, data, size, MSG_NOSIGNAL);
            if (sent <= 0) { return; }
            data += sent;
            size -= static_cast<std::size_t>(sent);
        }
    }

    void serve() {
        const int client = ::accept(socket_, nullptr, nullptr);
        if (client < 0) { return; }
        char request[1024];
        (void)::recv(client, request, sizeof(request), 0);
        const std::string header = "HTTP/1.1 200 OK\r\nContent-Type: image/x-portable-pixmap\r\n"
                                   "Content-Length: " +
                                   std::to_string(body_.size()) + "\r\nConnection: close\r\n\r\n";
        send_all(client, header.data(), header.size());
        send_all(client, body_.data(), body_.size());
        ::close(client);
        ::close(socket_);
        socket_ = -1;
    }

    int socket_         = -1;
    std::uint16_t port_ = 0;
    std::vector<char> body_;
    std::thread worker_;
};

int test_image_patch_layout_and_mrope(const std::filesystem::path& image_path) {
    ninfer::text::QwenTokenizer tokenizer(tokenizer_bundle());
    ninfer::model::ProcessorOptions options;
    options.image_min_pixels = 64 * 64;
    options.image_max_pixels = 64 * 64;
    ninfer::model::Processor processor(tokenizer, options);
    ninfer::text::ChatMessage message;
    message.role = "user";
    message.parts.push_back(ninfer::text::ChatPart::image(path_source(image_path)));
    message.parts.push_back(ninfer::text::ChatPart::text_part("x"));
    ninfer::text::ChatRenderOptions render;
    render.enable_thinking                 = false;
    const ninfer::model::ProcessedInput input = processor.process({message}, render);

    int failures = 0;
    failures += check(input.vision_items.size() == 1, "image item count mismatch");
    failures += check(input.vision_items[0].grid.t == 1 && input.vision_items[0].grid.h == 4 &&
                          input.vision_items[0].grid.w == 4,
                      "image grid mismatch");
    failures += check(input.stats.raw_patches == 16 && input.stats.vision_tokens == 4 &&
                          input.stats.attention_pairs == 256,
                      "image processor stats mismatch");
    failures += check(input.patches.size() == 16ULL * 1536, "image patch shape mismatch");
    failures += check(input.vision_items[0].token_spans.size() == 1 &&
                          input.vision_items[0].token_spans[0].count == 4,
                      "image placeholder span mismatch");
    const std::size_t span = input.vision_items[0].token_spans[0].begin;
    failures += check(input.position_axis(0)[span] == input.position_axis(1)[span] &&
                          input.position_axis(1)[span] == input.position_axis(2)[span],
                      "image first MRoPE position mismatch");
    failures += check(input.position_axis(1)[span + 2] == input.position_axis(1)[span] + 1 &&
                          input.position_axis(2)[span + 1] == input.position_axis(2)[span] + 1,
                      "image spatial MRoPE layout mismatch");

    const float p00 = -1.0f;
    const float p10 = 1.0f / 127.5f - 1.0f;
    failures += check(std::abs(input.patches[0] - p00) < 1e-7f, "first image patch pixel mismatch");
    failures += check(std::abs(input.patches[1] - p10) < 1e-7f, "image patch width order mismatch");
    failures +=
        check(input.patches[0] == input.patches[256], "image temporal duplication mismatch");
    // Patch 0,1,2,3 must be the 2x2 merger block: (0,0),(0,1),(1,0),(1,1).
    failures += check(std::abs(input.patches[1536] - (16.0f / 127.5f - 1.0f)) < 1e-7f,
                      "image merge-friendly patch order mismatch");
    return failures;
}

int test_video_prompt_and_temporal_duplication(const std::filesystem::path& image_path) {
    ninfer::text::QwenTokenizer tokenizer(tokenizer_bundle());
    ninfer::model::ProcessorOptions options;
    options.video_min_pixels = 64 * 64;
    options.video_max_pixels = 64 * 64 * 2;
    ninfer::model::Processor processor(tokenizer, options);
    ninfer::text::ChatMessage message;
    message.role = "user";
    message.parts.push_back(ninfer::text::ChatPart::video(path_source(image_path)));
    ninfer::text::ChatRenderOptions render;
    render.enable_thinking                 = false;
    const ninfer::model::ProcessedInput input = processor.process({message}, render);
    int failures                           = 0;
    failures += check(input.vision_items[0].grid.t == 1 &&
                          input.vision_items[0].timestamps == std::vector<double>{0.0},
                      "single-frame video metadata mismatch");
    failures += check(input.vision_items[0].token_spans.size() == 1 &&
                          input.vision_items[0].token_spans[0].count == 4,
                      "video placeholder span mismatch");
    failures += check(input.patches[0] == input.patches[256], "odd video frame padding mismatch");
    return failures;
}

int test_attention_budget_rejection(const std::filesystem::path& image_path) {
    ninfer::text::QwenTokenizer tokenizer(tokenizer_bundle());
    ninfer::model::ProcessorOptions options;
    options.image_min_pixels    = 64 * 64;
    options.image_max_pixels    = 64 * 64;
    options.max_attention_pairs = 255;
    ninfer::model::Processor processor(tokenizer, options);
    ninfer::text::ChatMessage message;
    message.role = "user";
    message.parts.push_back(ninfer::text::ChatPart::image(path_source(image_path)));
    try {
        (void)processor.process({message});
    } catch (const ninfer::model::ProcessorError& error) {
        return check(error.kind() == ninfer::model::ProcessorErrorKind::BudgetExceeded,
                     "vision attention budget used the wrong error category");
    }
    return check(false, "vision attention budget was not enforced");
}

int test_mixed_media_order(const std::filesystem::path& image_path) {
    ninfer::text::QwenTokenizer tokenizer(tokenizer_bundle());
    ninfer::model::ProcessorOptions options;
    options.image_min_pixels = 64 * 64;
    options.image_max_pixels = 64 * 64;
    options.video_min_pixels = 64 * 64;
    options.video_max_pixels = 64 * 64 * 2;
    ninfer::model::Processor processor(tokenizer, options);
    ninfer::text::ChatMessage message;
    message.role  = "user";
    message.parts = {ninfer::text::ChatPart::image(path_source(image_path)),
                     ninfer::text::ChatPart::text_part("x"),
                     ninfer::text::ChatPart::video(path_source(image_path)),
                     ninfer::text::ChatPart::image(path_source(image_path))};
    ninfer::text::ChatRenderOptions render;
    render.enable_thinking                 = false;
    const ninfer::model::ProcessedInput input = processor.process({message}, render);
    int failures                           = 0;
    failures += check(input.vision_items.size() == 3 &&
                          input.vision_items[0].modality == ninfer::model::Modality::Image &&
                          input.vision_items[1].modality == ninfer::model::Modality::Video &&
                          input.vision_items[2].modality == ninfer::model::Modality::Image,
                      "mixed media item order mismatch");
    failures +=
        check(input.vision_items[0].patch_begin == 0 && input.vision_items[1].patch_begin == 16 &&
                  input.vision_items[2].patch_begin == 32,
              "mixed media patch offsets mismatch");
    failures += check(
        input.vision_items[0].token_spans[0].begin < input.vision_items[1].token_spans[0].begin &&
            input.vision_items[1].token_spans[0].begin < input.vision_items[2].token_spans[0].begin,
        "mixed media token span order mismatch");
    return failures;
}

int test_data_uri_and_private_url(const std::filesystem::path& image_path) {
    ninfer::text::QwenTokenizer tokenizer(tokenizer_bundle());
    ninfer::model::ProcessorOptions options;
    options.image_min_pixels = 64 * 64;
    options.image_max_pixels = 64 * 64;
    ninfer::model::Processor processor(tokenizer, options);
    ninfer::text::ChatMessage data_message;
    data_message.role = "user";
    data_message.parts.push_back(ninfer::text::ChatPart::image(data_source(image_path)));
    const ninfer::model::ProcessedInput data = processor.process({data_message});
    int failures                          = 0;
    failures += check(data.stats.raw_patches == 16 && data.patches.front() == -1.0f,
                      "base64 data URI preprocessing mismatch");

    ninfer::text::ChatMessage private_message;
    private_message.role = "user";
    private_message.parts.push_back(ninfer::text::ChatPart::image(
        ninfer::media::Source{ninfer::media::SourceKind::Url, "http://127.0.0.1/image.png", {}}));
    try {
        (void)processor.process({private_message});
        failures += check(false, "private-network media URL was accepted");
    } catch (const std::invalid_argument&) {}
    return failures;
}

int test_http_media(const std::filesystem::path& image_path) {
    HttpFixture fixture(image_path);
    ninfer::text::QwenTokenizer tokenizer(tokenizer_bundle());
    ninfer::model::ProcessorOptions options;
    options.image_min_pixels      = 64 * 64;
    options.image_max_pixels      = 64 * 64;
    options.allow_private_network = true;
    ninfer::model::Processor processor(tokenizer, options);
    ninfer::text::ChatMessage message;
    message.role = "user";
    message.parts.push_back(ninfer::text::ChatPart::image(
        ninfer::media::Source{ninfer::media::SourceKind::Url, fixture.url(), "image/x-portable-pixmap"}));
    const ninfer::model::ProcessedInput input = processor.process({message});
    return check(input.stats.raw_patches == 16 && input.patches.front() == -1.0f,
                 "HTTP media preprocessing mismatch");
}

} // namespace

int main() {
    try {
        TempDir dir;
        const std::filesystem::path image = write_ppm(dir.path);
        int failures                      = 0;
        failures += test_image_patch_layout_and_mrope(image);
        failures += test_video_prompt_and_temporal_duplication(image);
        failures += test_mixed_media_order(image);
        failures += test_data_uri_and_private_url(image);
        failures += test_http_media(image);
        failures += test_attention_budget_rejection(image);
        if (failures == 0) { std::cout << "processor tests passed\n"; }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "processor test error: " << ex.what() << '\n';
        return 1;
    }
}
