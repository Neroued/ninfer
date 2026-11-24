#include "core/io/read_queue.h"

#include <liburing.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <exception>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <vector>

#include <poll.h>
#include <sys/eventfd.h>
#include <unistd.h>

namespace ninfer::io {
namespace {

void check(int result, const char* operation) {
    if (result < 0) throw std::system_error(-result, std::generic_category(), operation);
}

} // namespace

struct DirectReadQueue::Impl {
    struct Slot {
        DirectRead request{};
        bool busy = false;
    };

    io_uring ring{};
    bool initialized       = false;
    bool closed            = false;
    int notification       = -1;
    std::uint32_t capacity = 0;
    std::size_t pending    = 0;
    std::vector<FileView> files;
    std::vector<MemorySpan> buffers;
    std::vector<Slot> slots;

    ~Impl() {
        if (initialized) io_uring_queue_exit(&ring);
        if (notification >= 0) ::close(notification);
    }
};

DirectReadQueue::DirectReadQueue(std::span<const FileView> files,
                                 std::span<const MemorySpan> buffers, std::uint32_t capacity)
    : impl_(std::make_unique<Impl>()) {
    if (!capacity || capacity > buffers.size() || files.empty() ||
        files.size() > std::numeric_limits<int>::max() || buffers.size() > 65535)
        throw std::invalid_argument("invalid direct read queue capacity");
    auto& p    = *impl_;
    p.capacity = capacity;
    p.files.assign(files.begin(), files.end());
    p.buffers.assign(buffers.begin(), buffers.end());
    p.slots.resize(buffers.size());
    std::vector<int> descriptors;
    for (const auto file : files) {
        if (file.mode != ReadMode::Direct || file.descriptor < 0)
            throw std::invalid_argument("read queue requires direct files");
        descriptors.push_back(file.descriptor);
    }
    std::vector<iovec> vectors;
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        const auto buffer = buffers[i];
        validate_memory(buffer);
        if (buffer.device != Device::cpu() || !buffer.bytes ||
            buffer.bytes > std::numeric_limits<unsigned>::max())
            throw std::invalid_argument("read queue requires nonempty Host buffers");
        for (const auto file : files)
            validate_direct_read(file, 0, {static_cast<std::byte*>(buffer.data), buffer.bytes});
        const auto begin = reinterpret_cast<std::uintptr_t>(buffer.data);
        for (std::size_t j = 0; j < i; ++j) {
            const auto other = reinterpret_cast<std::uintptr_t>(buffers[j].data);
            if (begin < other + buffers[j].bytes && other < begin + buffer.bytes)
                throw std::invalid_argument("read queue buffers overlap");
        }
        vectors.push_back({buffer.data, buffer.bytes});
    }
    check(io_uring_queue_init(capacity, &p.ring, 0), "initialize direct read queue");
    p.initialized = true;
    check(io_uring_register_files(&p.ring, descriptors.data(), descriptors.size()),
          "register read files");
    check(io_uring_register_buffers(&p.ring, vectors.data(), vectors.size()),
          "register read buffers");
    p.notification = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (p.notification < 0) check(-errno, "create read queue notification");
}

DirectReadQueue::~DirectReadQueue() {
    try {
        close();
        drain();
    } catch (...) {
        // Releasing borrowed buffers with unknown kernel writers would violate ownership.
        std::terminate();
    }
}

ReadAdmission DirectReadQueue::try_enqueue(DirectRead request) {
    auto& p = *impl_;
    if (p.closed) return ReadAdmission::Closed;
    if (p.pending == p.capacity) return ReadAdmission::Full;
    if (request.file_index >= p.files.size() || request.buffer_index >= p.buffers.size() ||
        !request.bytes)
        throw std::invalid_argument("invalid direct read request");
    auto& slot        = p.slots[request.buffer_index];
    const auto buffer = p.buffers[request.buffer_index];
    if (slot.busy || request.bytes > buffer.bytes)
        throw std::invalid_argument("direct read buffer is busy or too small");
    validate_direct_read(p.files[request.file_index], request.file_offset,
                         {static_cast<std::byte*>(buffer.data), request.bytes});
    auto* sqe = io_uring_get_sqe(&p.ring);
    if (!sqe) return ReadAdmission::Full;
    io_uring_prep_read_fixed(sqe, request.file_index, buffer.data, request.bytes,
                             request.file_offset, request.buffer_index);
    sqe->flags |= IOSQE_FIXED_FILE;
    sqe->user_data = request.buffer_index;
    slot.request   = request;
    slot.busy      = true;
    ++p.pending;
    return ReadAdmission::Accepted;
}

void DirectReadQueue::submit() {
    auto& ring = impl_->ring;
    while (io_uring_sq_ready(&ring)) {
        const auto result = io_uring_submit(&ring);
        if (result == -EINTR) continue;
        check(result, "submit direct reads");
        if (!result && io_uring_sq_ready(&ring))
            throw std::runtime_error("direct read submission made no progress");
    }
}

std::size_t DirectReadQueue::poll(std::span<ReadCompletion> output) {
    auto& p           = *impl_;
    std::size_t count = 0;
    while (count < output.size()) {
        io_uring_cqe* cqe = nullptr;
        const auto result = io_uring_peek_cqe(&p.ring, &cqe);
        if (result == -EAGAIN) break;
        check(result, "poll direct read completion");
        const auto index = cqe->user_data;
        if (index >= p.slots.size() || !p.slots[index].busy)
            throw std::logic_error("unmatched direct read completion");
        auto& slot      = p.slots[index];
        output[count++] = {slot.request.tag, static_cast<std::uint32_t>(index),
                           cqe->res >= 0 ? static_cast<std::uint32_t>(cqe->res) : 0,
                           cqe->res < 0 ? -cqe->res : 0};
        slot.busy       = false;
        --p.pending;
        io_uring_cqe_seen(&p.ring, cqe);
    }
    return count;
}

ReadWake DirectReadQueue::wait() {
    auto& p = *impl_;
    std::array<pollfd, 2> descriptors{{{p.ring.ring_fd, POLLIN, 0}, {p.notification, POLLIN, 0}}};
    int result;
    do {
        result = ::poll(descriptors.data(), descriptors.size(), -1);
    } while (result < 0 && errno == EINTR);
    if (result < 0) check(-errno, "wait for direct reads");
    for (const auto& descriptor : descriptors)
        if (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL))
            throw std::runtime_error("direct read wait descriptor failed");
    if (descriptors[1].revents & POLLIN) {
        std::uint64_t value;
        ssize_t n;
        do { n = ::read(p.notification, &value, sizeof(value)); } while (n < 0 && errno == EINTR);
        if (n < 0 && errno != EAGAIN) check(-errno, "consume read queue notification");
        return ReadWake::Signalled;
    }
    return ReadWake::Completion;
}

void DirectReadQueue::wake() {
    const std::uint64_t value = 1;
    ssize_t n;
    do { n = ::write(impl_->notification, &value, sizeof(value)); } while (n < 0 && errno == EINTR);
    if (n < 0 && errno != EAGAIN) check(-errno, "wake direct read queue");
}

void DirectReadQueue::close() {
    impl_->closed = true;
    submit();
}

void DirectReadQueue::drain() {
    submit();
    std::array<ReadCompletion, 128> completions;
    while (impl_->pending)
        if (!poll(completions)) (void)wait();
}

std::size_t DirectReadQueue::outstanding() const noexcept { return impl_->pending; }

} // namespace ninfer::io
