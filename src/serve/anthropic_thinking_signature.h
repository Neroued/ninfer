#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace ninfer::serve {

// Process-local integrity authority for Anthropic Thinking blocks. The token binds the exact
// decoded UTF-8 text and its position in the Assistant content array without retaining responses.
class AnthropicThinkingSigner {
public:
    using Key = std::array<std::uint8_t, 32>;

    AnthropicThinkingSigner();
    explicit AnthropicThinkingSigner(Key key);

    [[nodiscard]] std::string sign(std::string_view thinking, std::size_t block_index) const;
    [[nodiscard]] bool verify(std::string_view thinking, std::size_t block_index,
                              std::string_view signature) const;

private:
    Key key_{};
};

} // namespace ninfer::serve
