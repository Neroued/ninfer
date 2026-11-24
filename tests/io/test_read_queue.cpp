#include "core/io/file.h"
#include "core/io/read_queue.h"
#include "core/buffer.h"
#include "symbol_interpose.h"

#include <liburing.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
bool interrupt_submit = false;
bool short_submit     = false;
bool fail_submit      = false;
bool fail_read        = false;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template <class E = std::invalid_argument, class F>
void rejects(F fn) {
    try {
        fn();
    } catch (const E&) { return; }
    throw std::runtime_error("invalid I/O operation accepted");
}

std::byte pattern(std::size_t file, std::size_t offset) {
    return std::byte((offset * 17 + offset / 11 + file * 23) & 255);
}

struct Files {
    std::filesystem::path directory;
    static constexpr std::size_t bytes = 4096 + 123;

    Files() {
        std::string name = (std::filesystem::temp_directory_path() / "ninfer-io-XXXXXX").string();
        auto* path       = ::mkdtemp(name.data());
        if (!path) throw std::runtime_error("mkdtemp failed");
        directory = path;
        std::vector<std::byte> data(bytes);
        for (std::size_t file = 0; file < 2; ++file) {
            for (std::size_t i = 0; i < data.size(); ++i) data[i] = pattern(file, i);
            std::ofstream out(directory / std::to_string(file), std::ios::binary);
            out.write(reinterpret_cast<const char*>(data.data()), data.size());
            out.close();
            require(bool(out), "fixture write failed");
        }
    }

    ~Files() {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }
};

void run() {
    using namespace ninfer;
    using namespace ninfer::io;
    Files fixture;
    ReadOnlyFile buffered(fixture.directory / "0");
    std::array<std::byte, 90> small;
    buffered.read_exact(4090, small);
    for (std::size_t i = 0; i < small.size(); ++i)
        require(small[i] == pattern(0, 4090 + i), "buffered bytes differ");
    rejects<std::system_error>([&] { buffered.read_exact(Files::bytes, small); });

    std::array owners{ReadOnlyFile(fixture.directory / "0", ReadMode::Direct),
                      ReadOnlyFile(fixture.directory / "1", ReadMode::Direct)};
    const std::array files{owners[0].view(), owners[1].view()};
    HostBuffer slab(2 * 4096, 4096);
    std::array<MemorySpan, 2> buffers;
    for (std::size_t i = 0; i < buffers.size(); ++i)
        buffers[i] = slab.span().subspan(i * 4096, 4096);
    const auto final = direct_read_window(files[0], Files::bytes - 90, 90);
    require(final.offset + final.required == Files::bytes, "tail window lost required bytes");
    const auto received =
        owners[0].read_direct(final.offset, {static_cast<std::byte*>(slab.data()), final.bytes});
    require(received >= final.required && received < final.bytes, "short final read differs");
    rejects([&] { (void)direct_read_window(files[0], Files::bytes - 10, 11); });
    rejects([&] { (void)owners[0].read_direct(1, {static_cast<std::byte*>(slab.data()), 4096}); });
    rejects([&] { DirectReadQueue bad(files, buffers, 3); });
    auto overlap = buffers;
    overlap[1]   = buffers[0];
    rejects([&] { DirectReadQueue bad(files, overlap, 2); });

    DirectReadQueue queue(files, buffers, 2);

    struct ReadCase {
        std::uint32_t file;
        std::uint64_t offset;
    };

    // Fill both slots, then reuse each returned slot. Both files exercise a full page and EOF.
    constexpr std::array<ReadCase, 4> reads{{{0, 0}, {1, 4096}, {1, 0}, {0, 4096}}};
    std::array<bool, reads.size()> seen{};
    std::array<std::uint32_t, reads.size()> assigned_buffers{};
    std::size_t issued = 0, completed = 0;
    auto enqueue = [&](std::uint32_t buffer) {
        const auto read = reads[issued];
        require(queue.try_enqueue({read.file, buffer, read.offset, 4096, issued}) ==
                    ReadAdmission::Accepted,
                "read rejected");
        assigned_buffers[issued] = buffer;
        ++issued;
    };
    enqueue(0);
    rejects([&] { (void)queue.try_enqueue({1, 0, 0, 4096, 999}); });
    enqueue(1);
    require(queue.try_enqueue({0, 0, 0, 4096, 999}) == ReadAdmission::Full, "queue is not bounded");
    interrupt_submit = short_submit = true;
    queue.submit();
    std::array<ReadCompletion, 1> output;
    while (completed < seen.size()) {
        auto n = queue.poll(output);
        if (!n) {
            (void)queue.wait();
            continue;
        }
        for (std::size_t i = 0; i < n; ++i) {
            const auto c = output[i];
            require(c.tag < seen.size() && !seen[c.tag], "duplicate or invalid completion");
            require(c.buffer_index == assigned_buffers[c.tag], "completion returned wrong buffer");
            seen[c.tag]      = true;
            const auto read  = reads[c.tag];
            const auto bytes = std::min<std::uint64_t>(4096, Files::bytes - read.offset);
            require(!c.error && c.bytes == bytes, "read completion length/error differs");
            const auto* data = static_cast<const std::byte*>(buffers[c.buffer_index].data);
            for (std::size_t j = 0; j < bytes; ++j)
                require(data[j] == pattern(read.file, read.offset + j), "async bytes differ");
            ++completed;
            if (issued < seen.size()) enqueue(c.buffer_index);
        }
        queue.submit();
    }
    require(!queue.outstanding(), "completed reads retained capacity");
    std::jthread notifier([&] { queue.wake(); });
    require(queue.wait() == ReadWake::Signalled, "notification did not wake idle queue");
    notifier.join();

    require(queue.try_enqueue({0, 0, 0, 4096, 1234}) == ReadAdmission::Accepted,
            "error probe rejected");
    fail_submit = true;
    rejects<std::system_error>([&] { queue.submit(); });
    fail_read = true;
    queue.submit();
    while (!queue.poll(output)) (void)queue.wait();
    require(output[0].tag == 1234 && output[0].error == EBADF && !output[0].bytes,
            "kernel I/O failure was not delivered");

    require(queue.try_enqueue({0, 0, 4096, 4096, 9999}) == ReadAdmission::Accepted,
            "close probe rejected");
    queue.close();
    require(queue.try_enqueue({0, 1, 0, 4096, 1}) == ReadAdmission::Closed,
            "closed queue admitted read");
    queue.drain();
    require(!queue.outstanding(), "drain retained a writer");
    const auto* data = static_cast<const std::byte*>(buffers[0].data);
    for (std::size_t i = 0; i < 123; ++i)
        require(data[i] == pattern(0, 4096 + i), "close/drain did not finish the accepted read");
    std::fill_n(static_cast<std::byte*>(buffers[0].data), 4096, std::byte{});
    {
        DirectReadQueue closing(files, buffers, 2);
        require(closing.try_enqueue({1, 0, 0, 4096, 0}) == ReadAdmission::Accepted,
                "destructor probe");
    }
    for (std::size_t i = 0; i < 4096; ++i)
        require(data[i] == pattern(1, i), "destructor did not drain");
}
} // namespace

extern "C" int io_uring_submit(io_uring* ring) {
    static const auto real = ninfer::test::next_symbol<int (*)(io_uring*)>("io_uring_submit");
    if (interrupt_submit) {
        interrupt_submit = false;
        return -EINTR;
    }
    if (fail_submit) {
        fail_submit = false;
        return -EAGAIN;
    }
    if (fail_read) {
        fail_read                                                = false;
        ring->sq.sqes[ring->sq.sqe_head & ring->sq.ring_mask].fd = -1;
    }
    if (short_submit && ring->sq.sqe_tail - ring->sq.sqe_head > 1) {
        // Expose one SQE to liburing; its normal flush advances sqe_head. Restore the remaining
        // locally prepared tail so the next production submit must issue those requests.
        short_submit      = false;
        const auto tail   = ring->sq.sqe_tail;
        ring->sq.sqe_tail = ring->sq.sqe_head + 1;
        const auto result = real(ring);
        ring->sq.sqe_tail = tail;
        return result;
    }
    return real(ring);
}

int main() {
    try {
        run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
