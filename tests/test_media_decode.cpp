#include "media/decode/decode.h"

#include <array>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

// Neroued/ninfer#20: this 300x200 yuvj420p JPEG selected an swscale SIMD path that overwrote a
// width-tight RGB24 destination. Keep the exact public reproducer so the test does not depend on
// an encoder or external files.
constexpr std::string_view issue_20_jpeg_base64 = R"jpeg(
/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAx
NDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy
MjIyMjIyMjL/wAARCADIASwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUF
BAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVW
V1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi
4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAEC
AxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVm
Z2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq
8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDFooor7E+RCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAK
KKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKA
CiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiii
gAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA7TQtC0290W3uLi23yvu3NvYZwxHY+1aP/AAjGj/8APn/5
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
z/nP/IyxH+Of/pTP0rAf7rS/wx/JBRRRXmnWFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABXDeM/+QxD/
ANe6/wDoTV3NcN4z/wCQxD/17r/6E1fXcE/8jZf4ZHicQf7k/VHOUUUV+yHwoUUUUAFFFFABRRRQAUUUUAFFFFAHeeHdQsoNBto5
ry3jcbsq8oBHzHsTWr/aunf8/wDa/wDf5f8AGvLqK+HxfA+HxOIqV3Vac23surufQUeIatKlGmoLRJfceo/2rp3/AD/2v/f5f8aP
7V07/n/tf+/y/wCNeXUVz/8AEP8ADf8AP6X3I1/1lq/8+197PUf7V07/AJ/7X/v8v+NH9q6d/wA/9r/3+X/GvLqKP+If4b/n9L7k
H+stX/n2vvZ6j/aunf8AP/a/9/l/xo/tXTv+f+1/7/L/AI15dRR/xD/Df8/pfcg/1lq/8+197PUf7V07/n/tf+/y/wCNH9q6d/z/
ANr/AN/l/wAa8uoo/wCIf4b/AJ/S+5B/rLV/59r72eo/2rp3/P8A2v8A3+X/ABo/tXTv+f8Atf8Av8v+NeXUUf8AEP8ADf8AP6X3
IP8AWWr/AM+197PUf7V07/n/ALX/AL/L/jR/aunf8/8Aa/8Af5f8a8uoo/4h/hv+f0vuQf6y1f8An2vvZ6j/AGrp3/P/AGv/AH+X
/Gj+1dO/5/7X/v8AL/jXl1FH/EP8N/z+l9yD/WWr/wA+197PUf7V07/n/tf+/wAv+NH9q6d/z/2v/f5f8a8uoo/4h/hv+f0vuQf6
y1f+fa+9nqP9q6d/z/2v/f5f8aP7V07/AJ/7X/v8v+NeXUUf8Q/w3/P6X3IP9Zav/Ptfez1H+1dO/wCf+1/7/L/jR/aunf8AP/a/
9/l/xry6ij/iH+G/5/S+5B/rLV/59r72eo/2rp3/AD/2v/f5f8aP7V07/n/tf+/y/wCNeXUUf8Q/w3/P6X3IP9Zav/Ptfez1H+1d
O/5/7X/v8v8AjXGeLbiG51WJ4Jo5VEABaNgwzubjisGivUyfhOjlmJWJhUcnZqzS6nHjs6qYuj7KUUgooor6w8YKKKKACiiigAoo
ooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAK
KKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKA
CiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiii
gD//2Q==
)jpeg";

std::vector<std::uint8_t> decode_base64(std::string_view encoded) {
    std::array<int, 256> values;
    values.fill(-1);
    constexpr std::string_view alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (std::size_t i = 0; i < alphabet.size(); ++i) {
        values[static_cast<unsigned char>(alphabet[i])] = static_cast<int>(i);
    }

    std::vector<std::uint8_t> out;
    out.reserve(encoded.size() * 3 / 4);
    unsigned accumulator = 0;
    int bits             = 0;
    for (const unsigned char byte : encoded) {
        if (std::isspace(byte)) { continue; }
        if (byte == '=') { break; }
        const int value = values[byte];
        if (value < 0) { throw std::runtime_error("invalid base64 fixture"); }
        accumulator = (accumulator << 6U) | static_cast<unsigned>(value);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<std::uint8_t>(accumulator >> bits));
            accumulator &= (1U << bits) - 1U;
        }
    }
    return out;
}

void expect_pixel(const ninfer::media::decode::Image& image, int x, int y,
                  std::array<int, 3> expected, int tolerance) {
    const std::size_t offset = (static_cast<std::size_t>(y) * image.width + x) * 3;
    for (int channel = 0; channel < 3; ++channel) {
        const int actual = image.rgb[offset + static_cast<std::size_t>(channel)];
        if (actual < expected[static_cast<std::size_t>(channel)] - tolerance ||
            actual > expected[static_cast<std::size_t>(channel)] + tolerance) {
            throw std::runtime_error("decoded JPEG pixel mismatch");
        }
    }
}

void test_issue_20_unaligned_jpeg() {
    const std::vector<std::uint8_t> encoded  = decode_base64(issue_20_jpeg_base64);
    const ninfer::media::decode::Image image = ninfer::media::decode::decode_image(encoded, {});
    if (image.width != 300 || image.height != 200 || image.rgb.size() != 300U * 200U * 3U) {
        throw std::runtime_error("decoded JPEG dimensions mismatch");
    }

    constexpr std::array<int, 3> blue   = {29, 120, 199};
    constexpr std::array<int, 3> yellow = {255, 200, 1};
    expect_pixel(image, 0, 0, blue, 3);
    expect_pixel(image, 299, 0, blue, 3);
    expect_pixel(image, 0, 199, blue, 3);
    expect_pixel(image, 299, 199, blue, 3);
    expect_pixel(image, 150, 100, yellow, 4);
}

} // namespace

int main() {
    try {
        test_issue_20_unaligned_jpeg();
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
