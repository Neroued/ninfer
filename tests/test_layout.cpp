#include "core/layout.h"

#include <iostream>
#include <limits>
#include <stdexcept>

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

void test_overflow_recovery() {
    ninfer::WorkspaceLayoutBuilder layout;
    expect(layout.peak_bytes(1) == 0, "empty workspace must need no storage");
    (void)layout.alloc_bytes(3, 1);
    expect_throws<std::overflow_error>(
        [&] { (void)layout.alloc_bytes(std::numeric_limits<std::size_t>::max(), 8); },
        "allocation end overflow must fail");
    (void)layout.alloc_bytes(1, 1);
    expect(layout.peak_bytes(1) == 4, "failed allocation must not consume alignment padding");

    ninfer::WorkspaceLayoutBuilder tensors;
    (void)tensors.alloc_bytes(std::numeric_limits<std::size_t>::max() - 14, 1);
    expect_throws<std::overflow_error>([&] { (void)tensors.alloc(ninfer::DType::BF16, {2, 2}, 8); },
                                       "tensor allocation end overflow must fail");
    (void)tensors.alloc_bytes(1, 1);
    expect(tensors.peak_bytes(1) == std::numeric_limits<std::size_t>::max() - 13,
           "failed tensor allocation must not consume alignment padding");
}

} // namespace

int main() {
    test_overflow_recovery();
    return failures == 0 ? 0 : 1;
}
