#pragma once
#include <memory>
#include <string>
#include <string_view>

namespace ninfer::text {
// Request-owned incremental uniqueness for JSON string arrays. The immutable schema
// plan is shared; a copy owns independent parser, branch and seen-value state.
class UniqueStringState {
public:
    static void AnalyzeSchema(const std::string& normalized_schema);
    explicit UniqueStringState(const std::string& normalized_schema,
                               const std::string& reasoning_close = {});
    ~UniqueStringState();
    UniqueStringState(const UniqueStringState&);
    UniqueStringState& operator=(const UniqueStringState&);
    UniqueStringState(UniqueStringState&&) noexcept;
    UniqueStringState& operator=(UniqueStringState&&) noexcept;
    [[nodiscard]] bool accepts(std::string_view decoded_token) const;
    // False never changes the committed state.
    bool accept(std::string_view decoded_token);
    [[nodiscard]] bool has_constraints() const noexcept;
    [[nodiscard]] bool needs_check(std::string_view decoded_token) const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ninfer::text
