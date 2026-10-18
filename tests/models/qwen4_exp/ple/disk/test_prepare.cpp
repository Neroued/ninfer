#include "models/qwen4_exp/ple/upload.h"
#include "models/qwen4_exp/ple/hash.h"
#include "models/qwen4_exp/ple/disk/prepare.h"
#include "core/buffer.h"
#include "core/decode_graph.h"
#include "core/transfer.h"
#include "symbol_interpose.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

namespace {
using namespace ninfer;
using namespace ninfer::models::qwen4_exp::ple;
bool fail_event_record = false;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template <class F>
void rejects(F fn) {
    try {
        fn();
    } catch (const std::logic_error&) { return; }
    throw std::runtime_error("invalid PLE lifecycle operation accepted");
}

template <class F>
void until(F ready) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!ready()) {
        require(std::chrono::steady_clock::now() < end, "PLE completion timed out");
        std::this_thread::yield();
    }
}

template <class T>
T read() {
    T value;
    std::cin.read(reinterpret_cast<char*>(&value), sizeof(value));
    require(bool(std::cin), "truncated test input");
    return value;
}

void emit(const void* data, std::size_t bytes) {
    std::cout.write(static_cast<const char*>(data), bytes);
}

struct Input {
    std::uint32_t tokens = read<std::uint32_t>(), batch = read<std::uint32_t>();
    std::int32_t eos = read<std::int32_t>();
    std::array<std::vector<std::int32_t>, 8> ids;
    std::array<InputRow, 8> rows;
    std::array<std::uint32_t, 8> valid{};

    Input() {
        require(batch && batch <= 8 && tokens && tokens <= 8, "invalid test shape");
        for (std::uint32_t b = 0; b < batch; ++b) {
            valid[b]         = read<std::uint32_t>();
            const auto older = read<std::int32_t>(), previous = read<std::int32_t>();
            ids[b].resize(tokens);
            for (auto& id : ids[b]) id = read<std::int32_t>();
            rows[b] = {{100 + b, 7, 31 + b, {older, previous}}, ids[b]};
        }
    }

    auto inputs() const { return std::span(rows).first(batch); }

    auto lengths() const { return std::span(valid).first(batch); }

    std::size_t bytes() const { return std::size_t(tokens) * batch * 5120; }
};

Ticket prepare(disk::PrepareService& service, const Input& input) {
    std::optional<Ticket> ticket;
    until([&] {
        ticket = service.prepare(input.tokens, input.inputs());
        return ticket.has_value();
    });
    return *ticket;
}

void ready(disk::PrepareService& service, Ticket ticket) {
    until([&] {
        const auto s = service.poll(ticket);
        if (s.error) std::rethrow_exception(s.error);
        return s.status == disk::PrepareStatus::HostReady && s.drained;
    });
}

void row_reads(const disk::Table& source) {
    std::array<std::uint16_t, 160 * 8> a, b{};
    a.fill(0xbeef);
    disk::RowReader reader(source, 2, 8, 4, 0);
    reader.begin(0, a, false);
    reader.begin(1, b, true);
    reader.request(0, 0, 44);
    reader.request(0, 1, 44);
    reader.request(1, 0, 44);
    reader.cancel(0, 0, 8);
    until([&] {
        reader.pump();
        return reader.drained(0) && reader.drained(1);
    });
    require(std::all_of(a.begin(), a.end(), [](auto v) { return v == 0xbeef; }),
            "cancelled row reader wrote output");
    require(!reader.error(1), "shared row failed");
    const auto reads = reader.stats().reads;
    reader.release(0);
    reader.request(1, 1, 44);
    require(reader.stats().reads == reads &&
                std::equal(b.begin(), b.begin() + 160, b.begin() + 160),
            "releasing a cancelled owner lost a shared row");
    emit(b.data(), 320);
    reader.release(1);

    disk::RowReader cancellation(source, 2, 8, 1, 0);
    cancellation.begin(0, a, false);
    cancellation.begin(1, b, true);
    cancellation.request(0, 0, 0);
    cancellation.request(1, 0, 46);
    cancellation.cancel(0, 0, 8);
    cancellation.pump();
    require(cancellation.drained(0), "queued cancellation waited for another owner's I/O");
    until([&] {
        cancellation.pump();
        return cancellation.drained(1);
    });
    cancellation.release(0);
    cancellation.release(1);

    disk::RowReader cached(source, 1, 8, 4, 512);
    const auto one = [&] {
        cached.begin(0, b, true);
        cached.request(0, 0, 44);
        until([&] {
            cached.pump();
            return cached.drained(0);
        });
        require(!cached.error(0), "cached row failed");
        cached.release(0);
    };
    one();
    const auto first = cached.stats().reads;
    one();
    require(cached.stats().reads == first, "row cache hit performed I/O");
    cached.begin(0, a, true);
    for (std::uint32_t i = 0; i < 5; ++i) cached.request(0, i, i);
    until([&] {
        cached.pump();
        return cached.drained(0);
    });
    cached.release(0);
    const auto before = cached.stats().reads;
    one();
    require(cached.stats().reads > before, "bounded cache did not evict");
    emit(b.data(), 320);
}

void hashes(const disk::Table& source, const Input& input) {
    for (const auto& row : input.inputs()) {
        std::vector<std::uint32_t> full(input.tokens * 16), split(full.size());
        hash_tokens(row.ids, row.identity.raw_tail, input.eos, source.parameters().hash, full);
        auto tail = row.identity.raw_tail;
        for (std::uint32_t t = 0; t < input.tokens; ++t) {
            hash_tokens(row.ids.subspan(t, 1), tail, input.eos, source.parameters().hash,
                        std::span(split).subspan(t * 16, 16));
            tail = {tail[1], row.ids[t]};
        }
        require(full == split, "chunked hash changed row IDs");
        emit(full.data(), full.size() * sizeof(full[0]));
    }
}

void host(const disk::Table& source, const Input& input, bool failure) {
    HostBuffer first(input.bytes()), second(input.bytes());
    const std::array outputs{first.span(), second.span()};
    disk::PrepareService service(
        source, input.eos,
        {.max_positions = input.tokens * input.batch, .io_depth = 4, .cache_bytes = 512}, outputs);
    if (!failure) {
        const std::array empty{InputRow{input.rows[0].identity, {}}};
        const auto cancelled = service.prepare(1, empty);
        require(cancelled.has_value(), "initial window admission failed");
        service.cancel(*cancelled);
        require(!service.prepare(1, empty, false), "prefetch consumed the last required slot");
        rejects([&] { service.prioritize(*cancelled); });
        until([&] { return service.poll(*cancelled).drained; });
        service.release(*cancelled);
    }
    const auto full = prepare(service, input);
    if (failure) {
        service.seal(full, input.lengths());
        until([&] {
            const auto s = service.poll(full);
            return s.status == disk::PrepareStatus::Failed && s.drained;
        });
        require(bool(service.poll(full).error), "failed window lost its error");
        rejects([&] { (void)service.embedding(full); });
        service.release(full);
        return;
    }
    auto incremental = input.rows;
    for (std::uint32_t b = 0; b < input.batch; ++b)
        incremental[b].ids = incremental[b].ids.first(1);
    std::optional<Ticket> next;
    until([&] {
        next = service.prepare(input.tokens, std::span(incremental).first(input.batch), false);
        return next.has_value();
    });
    require(!service.prepare(input.tokens, input.inputs()), "unbounded PLE window admission");
    rejects([&] { service.release(full); });
    rejects([&] { (void)service.embedding(full); });
    service.prioritize(*next);
    for (std::uint32_t t = 1; t < input.tokens; ++t) {
        std::array<Append, 8> append;
        for (std::uint32_t b = 0; b < input.batch; ++b)
            append[b] = {b, input.rows[b].ids.subspan(t, 1)};
        service.append(*next, std::span(append).first(input.batch));
    }
    require(service.poll(*next).status == disk::PrepareStatus::Preparing,
            "unsealed window published E");
    service.seal(full, input.lengths());
    service.seal(*next, input.lengths());
    rejects([&] { service.append(*next, {}); });
    ready(service, full);
    ready(service, *next);
    for (auto ticket : {full, *next}) {
        const auto e = service.embedding(ticket);
        require(e.window.tokens == input.tokens && e.window.batch == input.batch &&
                    e.window.identities[0].sequence == 100 && e.window.identities[0].frontier == 31,
                "window identity/shape changed");
        emit(e.values.data, e.values.bytes);
    }
    service.release(full);
    rejects([&] { (void)service.poll(full); });
    const auto cancelled = prepare(service, input);
    require(cancelled.slot == full.slot && cancelled.generation != full.generation,
            "slot reused without a new generation");
    service.cancel(cancelled);
    until([&] {
        const auto s = service.poll(cancelled);
        return s.status == disk::PrepareStatus::Cancelled && s.drained;
    });
    service.cancel(cancelled);
    require(service.poll(cancelled).drained, "repeated cancel restarted retirement");
    service.release(cancelled);
    service.release(*next);

    const auto reads = service.stats().reads;
    const std::array empty{InputRow{input.rows[0].identity, {}}};
    std::optional<Ticket> zero;
    until([&] {
        zero = service.prepare(1, empty);
        return zero.has_value();
    });
    service.seal(*zero, std::array<std::uint32_t, 1>{0});
    ready(service, *zero);
    const auto embedding = service.embedding(*zero);
    const auto* words    = static_cast<const std::uint16_t*>(embedding.values.data);
    require(embedding.values.bytes == 5120 &&
                std::all_of(words, words + 2560, [](auto w) { return w == 0; }),
            "empty reshaped window retained old E");
    require(service.stats().reads == reads, "empty window read the table");
    service.release(*zero);
}

class StreamGate {
public:
    explicit StreamGate(cudaStream_t stream) : stream_(stream) {
        CUDA_CHECK(cudaLaunchHostFunc(
            stream,
            [](void* argument) {
                auto& gate = *static_cast<StreamGate*>(argument);
                while (!gate.open_.load(std::memory_order_acquire)) std::this_thread::yield();
            },
            this));
    }

    ~StreamGate() {
        open();
        (void)cudaStreamSynchronize(stream_);
    }

    void open() { open_.store(true, std::memory_order_release); }
private:
    cudaStream_t stream_;
    std::atomic<bool> open_{false};
};

void gpu(const disk::Table& source, const Input& input) {
    DeviceContext context;
    UploadSlots uploads(context, input.tokens * input.batch, 2);
    disk::PrepareService service(
        source, input.eos,
        {.max_positions = input.tokens * input.batch, .io_depth = 4, .cache_bytes = 512},
        uploads.host_slots());
    const auto try_release = [&](Ticket ticket) {
        const auto state = service.poll(ticket);
        if (state.status == disk::PrepareStatus::Preparing || !state.drained ||
            !uploads.try_release(ticket)) {
            return false;
        }
        service.release(ticket);
        return true;
    };
    DeviceBuffer copied(context.device(), input.bytes());
    std::vector<std::uint16_t> actual(input.bytes() / 2);
    const auto ticket = prepare(service, input);
    service.seal(ticket, input.lengths());
    ready(service, ticket);
    {
        StreamGate transfer(context.transfer_stream()), consumer(context.stream());
        uploads.upload(service.embedding(ticket), context.transfer_view());
        const auto binding = uploads.acquire(ticket, context.execution_view());
        binding.ready.wait(context.stream());
        const auto& e = binding.values;
        copy_bytes_async(copied.span(), {e.data, e.bytes(), e.device}, context.execution_view());
        binding.consumed.record(context.stream());
        uploads.consumer_submitted(ticket, context.execution_view());
        service.cancel(ticket);
        until([&] {
            const auto s = service.poll(ticket);
            return s.status == disk::PrepareStatus::Cancelled && s.drained;
        });
        require(!try_release(ticket), "pending H2D released its pinned source");
        rejects([&] { (void)service.embedding(ticket); });
        transfer.open();
        until([&] { return uploads.upload_complete(ticket); });
        require(!try_release(ticket), "pending consumer released Device E");
        consumer.open();
        until([&] { return try_release(ticket); });
    }
    copied.copy_to_host(actual.data(), input.bytes());
    emit(actual.data(), input.bytes());

    for (bool consumer_failure : {false, true}) {
        const auto failed = prepare(service, input);
        service.seal(failed, input.lengths());
        ready(service, failed);
        bool rejected = false;
        try {
            fail_event_record = !consumer_failure;
            uploads.upload(service.embedding(failed), context.transfer_view());
            const auto binding = uploads.acquire(failed, context.execution_view());
            binding.ready.wait(context.stream());
            const auto& e = binding.values;
            copy_bytes_async(copied.span(), {e.data, e.bytes(), e.device},
                             context.execution_view());
            fail_event_record = true;
            binding.consumed.record(context.stream());
            uploads.consumer_submitted(failed, context.execution_view());
        } catch (const std::runtime_error&) {
            rejected = true;
            if (consumer_failure) {
                require(!uploads.try_release(failed), "failed record released a live consumer");
                // The submitter owns the failed consumer submission. Establish completion
                // before returning the slot, just as Program cancellation does.
                auto binding = uploads.binding(failed.slot, input.tokens, input.batch);
                binding.consumed.record(context.stream());
                uploads.consumer_submitted(failed, context.execution_view());
            }
            service.cancel(failed);
        }
        require(rejected && !fail_event_record, "CUDA event failure was not reported");
        until([&] { return try_release(failed); });
    }

    const auto again = prepare(service, input);
    service.seal(again, input.lengths());
    const auto captured = uploads.binding(again.slot, input.tokens, input.batch);
    const auto& fixed   = captured.values;
    DecodeGraphDefinition child, definition;
    child.capture(context.stream(), [&] {
        captured.ready.wait_external(context.stream());
        copy_bytes_async(copied.span(), {fixed.data, fixed.bytes(), fixed.device},
                         context.execution_view());
        captured.consumed.record_external(context.stream());
    });
    definition.capture(context.stream(), [&] { (void)child.append_to(context.stream()); });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    ready(service, again);
    uploads.upload(service.embedding(again), context.transfer_view());
    const auto binding = uploads.acquire(again, context.execution_view());
    const auto& e      = binding.values;
    require(e.data == fixed.data, "Device E changed its captured address");
    {
        StreamGate gate(context.stream());
        graph.launch(context.stream());
        uploads.consumer_submitted(again, context.execution_view());
        until([&] { return uploads.upload_complete(again); });
        require(!try_release(again), "child event retained the previous completed generation");
        gate.open();
    }
    until([&] { return try_release(again); });
    copied.copy_to_host(actual.data(), input.bytes());
    emit(actual.data(), input.bytes());
}
} // namespace

extern "C" cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    static const auto real =
        ninfer::test::next_symbol<cudaError_t (*)(cudaEvent_t, cudaStream_t)>("cudaEventRecord");
    if (fail_event_record) {
        fail_event_record = false;
        return cudaErrorInvalidResourceHandle;
    }
    return real(event, stream);
}

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("expected ARTIFACT --host|--gpu|--failure");
        const std::string_view mode = argv[2];
        if (mode == "--gpu") {
            int count         = 0;
            const auto status = cudaGetDeviceCount(&count);
            if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
                (status == cudaSuccess && !count))
                return 77;
            CUDA_CHECK(status);
        }
        Input input;
        auto source = [&] {
            artifact::Reader reader(argv[1]);
            return disk::Table(
                reader, load_lookup_parameters(
                            reader, reader.directory().bindings.at("text/layers/1/ple/table"),
                            reader.read_object(reader.directory().component("text").resources.at(
                                "ple_hash_controls")),
                            {3, 8}));
        }();
        if (mode == "--failure")
            host(source, input, true);
        else if (mode == "--host") {
            row_reads(source);
            hashes(source, input);
            host(source, input, false);
        } else if (mode == "--gpu")
            gpu(source, input);
        else
            throw std::invalid_argument("unknown test mode");
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
