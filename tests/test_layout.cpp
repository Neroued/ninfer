#include "core/layout.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << message << '\n';
}

template <typename Exception, typename Operation>
void expect_throws(Operation operation, const char* message) {
    try {
        operation();
    } catch (const Exception&) { return; }
    expect(false, message);
}

void test_workspace_scopes() {
    ninfer::WorkspaceLayoutBuilder layout;
    expect(layout.peak_bytes() == 0, "empty workspace must need no storage");
    const auto tensor = layout.alloc(ninfer::DType::BF16, {3, 5});
    expect(tensor.data == nullptr && tensor.bytes() == 30 && tensor.ne[0] == 3 && tensor.ne[1] == 5,
           "dry-run tensor must retain shape and dtype without storage");
    expect(layout.peak_bytes(1) == 30, "first tensor needs 30 bytes");
    {
        auto outer       = layout.scope();
        const auto bytes = layout.alloc_bytes(17, 64);
        expect(bytes.data == nullptr && bytes.bytes == 17,
               "dry-run byte span must not own storage");
        expect(layout.peak_bytes(1) == 81, "aligned allocation ends at byte 81");
        {
            auto inner = layout.scope();
            auto moved = std::move(inner);
            (void)layout.alloc_bytes(11, 128);
            expect(layout.peak_bytes(1) == 139, "nested allocation ends at byte 139");
        }
        (void)layout.alloc_bytes(100, 1);
        expect(layout.peak_bytes(1) == 181, "moved inner scope must restore byte 81");
    }
    (void)layout.alloc_bytes(200, 1);
    expect(layout.peak_bytes(1) == 230, "outer scope must restore byte 30");
    expect(layout.peak_bytes() == 256, "final estimate rounds the peak to its requested alignment");

    expect_throws<std::runtime_error>(
        [&] {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(100, 1);
            throw std::runtime_error("unwind workspace scope");
        },
        "test exception must propagate through scope cleanup");
    expect(layout.peak_bytes(1) == 330, "scope unwinding must preserve the peak");
    (void)layout.alloc_bytes(101, 1);
    expect(layout.peak_bytes(1) == 331, "scope unwinding must restore the cursor");
}

void test_workspace_boundaries() {
    ninfer::WorkspaceLayoutBuilder layout;
    (void)layout.alloc_bytes(3, 1);
    const auto empty = layout.alloc_bytes(0, 3);
    expect(empty.data == nullptr && empty.bytes == 0 && layout.peak_bytes(1) == 3,
           "zero-byte scratch must remain a no-op, including its alignment");
    expect_throws<std::invalid_argument>([&] { (void)layout.alloc_bytes(1, 3); },
                                         "non-power-of-two alignment must fail");
    expect_throws<std::invalid_argument>([&] { (void)layout.peak_bytes(0); },
                                         "zero final alignment must fail");
    expect_throws<std::overflow_error>(
        [&] { (void)layout.alloc_bytes(std::numeric_limits<std::size_t>::max(), 8); },
        "allocation end overflow must fail");
    (void)layout.alloc_bytes(1, 1);
    expect(layout.peak_bytes(1) == 4, "failed allocation must not consume alignment padding");

    ninfer::WorkspaceLayoutBuilder tensor_path;
    (void)tensor_path.alloc_bytes(std::numeric_limits<std::size_t>::max() - 14, 1);
    expect_throws<std::overflow_error>(
        [&] { (void)tensor_path.alloc(ninfer::DType::BF16, {2, 2}, 8); },
        "tensor allocation end overflow must fail");
    (void)tensor_path.alloc_bytes(1, 1);
    expect(tensor_path.peak_bytes(1) == std::numeric_limits<std::size_t>::max() - 13,
           "failed tensor allocation must not consume alignment padding");

    ninfer::WorkspaceLayoutBuilder maximum;
    (void)maximum.alloc_bytes(std::numeric_limits<std::size_t>::max(), 1);
    expect(maximum.peak_bytes(1) == std::numeric_limits<std::size_t>::max(),
           "maximum representable unaligned size must remain representable");
    expect_throws<std::overflow_error>([&] { (void)maximum.peak_bytes(2); },
                                       "final alignment overflow must fail");
    expect_throws<std::overflow_error>([&] { (void)maximum.alloc_bytes(1, 2); },
                                       "cursor alignment overflow must fail");
}

} // namespace

int main() {
    test_workspace_scopes();
    test_workspace_boundaries();
    return failures == 0 ? 0 : 1;
}
