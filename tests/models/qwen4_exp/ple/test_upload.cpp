#include "models/qwen4_exp/ple/upload.h"
#include "models/qwen4_exp/ple/parameters.h"
#include "core/buffer.h"
#include "core/decode_graph.h"
#include "core/transfer.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>

namespace {
using namespace ninfer;
using namespace ninfer::models::qwen4_exp::ple;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template <class F>
void rejects(F fn) {
    try {
        fn();
    } catch (const std::logic_error&) { return; }
    throw std::runtime_error("invalid upload operation accepted");
}

template <class F>
void until(F ready) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!ready()) {
        require(std::chrono::steady_clock::now() < end, "PLE upload timed out");
        std::this_thread::yield();
    }
}

void run() {
    DeviceContext context;
    UploadSlots uploads(context, 6, 2);
    DeviceBuffer copied(context.device(), 6 * kEmbeddingBytes);
    std::vector<std::uint16_t> actual(6 * kEmbeddingElements);
    auto input = [&](Ticket ticket, std::uint32_t tokens, std::uint32_t batch) {
        InputWindow window{};
        window.tokens = tokens;
        window.batch  = batch;
        std::fill_n(window.valid_columns.begin(), batch, tokens);
        const auto values = uploads.host_slots()[ticket.slot].subspan(
            0, std::size_t(tokens) * batch * kEmbeddingBytes);
        auto* words = static_cast<std::uint16_t*>(values.data);
        for (std::size_t i = 0; i < values.bytes / 2; ++i) {
            words[i] = std::uint16_t((i * 13 + ticket.slot * 97 + ticket.generation) & 0x7fff);
        }
        return HostEmbedding{ticket, values, window};
    };

    // A direct Host producer uses the transfer owner without any table or preparation service.
    const auto first       = input({0, 1}, 3, 2);
    auto wrong_slot        = first;
    wrong_slot.ticket.slot = 1;
    rejects([&] { uploads.upload(wrong_slot, context.transfer_view()); });
    uploads.upload(first, context.transfer_view());
    const auto second = input({1, 1}, 1, 1);
    uploads.upload(second, context.transfer_view());

    auto consume = [&](const HostEmbedding& host) {
        const auto binding = uploads.acquire(host.ticket, context.execution_view());
        binding.ready.wait(context.stream());
        const auto& e = binding.values;
        copy_bytes_async(copied.span().subspan(0, e.bytes()), {e.data, e.bytes(), e.device},
                         context.execution_view());
        require(!uploads.try_release(host.ticket), "acquired E released before its last reader");
        binding.consumed.record(context.stream());
        uploads.consumer_submitted(host.ticket, context.execution_view());
        until([&] { return uploads.try_release(host.ticket); });
        copied.copy_to_host(actual.data(), host.values.bytes);
        const auto* expected = static_cast<const std::uint16_t*>(host.values.data);
        require(std::equal(expected, expected + host.values.bytes / 2, actual.begin()),
                "upload changed BF16 words or E layout");
    };
    consume(first);
    consume(second);
    rejects([&] { uploads.upload(first, context.transfer_view()); });

    const auto next = input({0, 2}, 1, 3);
    uploads.upload(next, context.transfer_view());
    rejects([&] { (void)uploads.acquire(first.ticket, context.execution_view()); });
    rejects([&] { (void)uploads.try_release(first.ticket); });
    consume(next);

    // Discarding a prepared/uploaded input needs no artificial GPU consumer.
    const auto skipped = input({1, 2}, 1, 1);
    require(uploads.try_release(skipped.ticket), "never-uploaded input retained CUDA references");
    rejects([&] { uploads.upload(skipped, context.transfer_view()); });
    const auto discarded = input({1, 3}, 1, 2);
    uploads.upload(discarded, context.transfer_view());
    until([&] { return uploads.try_release(discarded.ticket); });

    DecodeGraphDefinition children[2], parent;
    for (std::uint32_t slot = 0; slot < 2; ++slot) {
        const auto binding = uploads.binding(slot, 3, 2);
        children[slot].capture(context.stream(), [&] {
            binding.ready.wait_external(context.stream());
            const auto& e = binding.values;
            copy_bytes_async(copied.span(), {e.data, e.bytes(), e.device},
                             context.execution_view());
            binding.consumed.record_external(context.stream());
        });
    }
    DecodeGraphChild child;
    parent.capture(context.stream(), [&] {
        CUDA_CHECK(cudaMemsetAsync(copied.data(), 0xff, copied.size(), context.stream()));
        child = children[0].append_to(context.stream());
    });
    DecodeGraphExecutable graph;
    graph.instantiate(parent);
    for (std::uint32_t iteration = 0; iteration < 32; ++iteration) {
        const auto slot = iteration % 2;
        const auto host = input({slot, iteration + 4}, 3, 2);
        uploads.upload(host, context.transfer_view());
        (void)uploads.acquire(host.ticket, context.execution_view());
        graph.update_child(child, children[slot]);
        graph.launch(context.stream());
        uploads.consumer_submitted(host.ticket, context.execution_view());
        until([&] { return uploads.try_release(host.ticket); });
        copied.copy_to_host(actual.data(), host.values.bytes);
        const auto* expected = static_cast<const std::uint16_t*>(host.values.data);
        require(std::equal(expected, expected + host.values.bytes / 2, actual.begin()),
                "graph child update consumed a stale PLE generation or slot");
    }
}
} // namespace

int main() {
    try {
        int count         = 0;
        const auto status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && !count)) {
            return 77;
        }
        CUDA_CHECK(status);
        run();
        std::cout << "PLE upload: exact E, independent slots and retirement passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
