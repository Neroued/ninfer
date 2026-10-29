#include "core/arena.h"
#include "core/buffer.h"
#include "core/layout.h"
#include "core/transfer.h"

#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

template <class Exception, class Function>
void rejects(Function&& function) {
    try {
        function();
    } catch (const Exception&) { return; }
    throw std::runtime_error("expected invalid memory operation to fail");
}

} // namespace

int main() {
    using namespace ninfer;
    try {
        HostBuffer owner(4096);
        require(reinterpret_cast<std::uintptr_t>(owner.data()) % 256 == 0, "host alignment");
        MemoryArena arena(owner.span());
        auto matrix = arena.alloc(DType::FP32, {8, 4});
        auto* data  = static_cast<float*>(matrix.data);
        for (int i = 0; i < 32; ++i) { data[i] = static_cast<float>(i); }
        const auto strided = matrix.slice(0, 2, 3).permute({1, 0, 2, 3});
        require(strided.device == Device::cpu() && strided.nb[0] == 32, "strided CPU view");
        require(*static_cast<float*>(strided.data) == 2, "CPU slice origin");
        require(matrix.reshape({32}).device == Device::cpu(), "reshape residency");

        const auto used = arena.used();
        {
            auto scope         = arena.scope();
            const auto scratch = arena.alloc(DType::U8, {257});
            require(scratch.device == Device::cpu(), "scratch residency");
        }
        require(arena.used() == used && arena.peak_used() > used, "arena rewind and peak");
        rejects<std::bad_alloc>([&] { (void)arena.alloc_bytes(4096); });
        require(arena.used() == used, "failed allocation changes cursor");

        LayoutBuilder layout;
        const auto region  = layout.add_tensor(DType::FP32, {8, 4}, 256, "matrix");
        const auto rebound = region.bind(owner.span());
        require(rebound.data == matrix.data && rebound.device == matrix.device, "layout binding");
        const HostBuffer& readonly = owner;
        require(region.region.bind(readonly.span()).device == Device::cpu(), "const span binding");
        rejects<std::out_of_range>(
            [&] { (void)Tensor(owner.span().subspan(0, 1), DType::FP32, {1}); });
        rejects<std::invalid_argument>(
            [&] { (void)Tensor(owner.data(), Device{}, DType::U8, {1}); });

        HostBuffer copy(owner.size());
        std::memset(owner.data(), 0x5a, owner.size());
        copy_bytes(copy.span(), readonly.span());
        require(std::memcmp(owner.data(), copy.data(), owner.size()) == 0, "CPU copy");
        rejects<std::invalid_argument>(
            [&] { copy_bytes(copy.span().subspan(0, 1), owner.span()); });
        rejects<std::out_of_range>([&] { (void)owner.span().subspan(owner.size() + 1); });
        auto moved = std::move(owner);
        require(owner.data() == nullptr && owner.size() == 0, "move clears source");
        require(moved.data() == matrix.data, "move preserves borrowed views");

        WorkspaceLayoutBuilder planned;
        const auto metadata = planned.alloc(DType::FP32, {8, 4});
        require(!metadata.data && metadata.device == Device{}, "planning has no residency");
        require(planned.peak_bytes(1) == matrix.bytes(), "planning and allocation agree");

        const auto limit = std::numeric_limits<std::size_t>::max();
        WorkspaceLayoutBuilder bytes;
        (void)bytes.alloc_bytes(3, 1);
        rejects<std::overflow_error>([&] { (void)bytes.alloc_bytes(limit, 8); });
        require(bytes.peak_bytes(1) == 3, "failed byte layout changed the peak");
        (void)bytes.alloc_bytes(1, 1);
        require(bytes.peak_bytes(1) == 4, "failed byte layout retained alignment padding");
        {
            auto scope = bytes.scope();
            (void)bytes.alloc_bytes(3, 8);
            rejects<std::overflow_error>([&] { (void)bytes.alloc_bytes(limit, 16); });
            require(bytes.peak_bytes(1) == 11, "failed scoped layout changed the peak");
            (void)bytes.alloc_bytes(1, 1);
            require(bytes.peak_bytes(1) == 12, "failed scoped layout changed the cursor");
        }
        (void)bytes.alloc_bytes(9, 1);
        require(bytes.peak_bytes(1) == 13, "layout scope did not restore the saved cursor");

        WorkspaceLayoutBuilder tensors;
        (void)tensors.alloc_bytes(limit - 14, 1);
        rejects<std::overflow_error>([&] { (void)tensors.alloc(DType::BF16, {2, 2}, 8); });
        require(tensors.peak_bytes(1) == limit - 14, "failed tensor layout changed the peak");
        (void)tensors.alloc_bytes(1, 1);
        require(tensors.peak_bytes(1) == limit - 13,
                "failed tensor layout retained alignment padding");
        std::cout << "OK memory\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
