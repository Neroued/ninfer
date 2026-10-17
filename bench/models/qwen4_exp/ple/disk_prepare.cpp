#include "models/qwen4_exp/ple/upload.h"
#include "models/qwen4_exp/ple/disk/prepare.h"
#include "artifact/materializer.h"
#include "core/buffer.h"
#include "core/decode_graph.h"
#include "ninfer/ops/linear.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

namespace {
using namespace ninfer;
using namespace ninfer::models::qwen4_exp::ple;
using Clock = std::chrono::steady_clock;

struct Options {
    std::string artifact, corpus, dump;
    std::uint32_t batch, tokens, iterations, cache_mib;
    std::uint32_t incremental_us = 0, work = 0;
    std::uint64_t budget = 512ULL * 1024 * 1024;
};

Options options(int argc, char** argv) {
    if (argc < 7)
        throw std::invalid_argument(
            "usage: ninfer_ple_disk_prepare_bench ARTIFACT CORPUS B T ITERATIONS CACHE_MIB "
            "[--read-budget-mib N] [--incremental-us N] [--overlap N] [--dump PATH]");
    Options o{argv[1],
              argv[2],
              {},
              std::uint32_t(std::stoul(argv[3])),
              std::uint32_t(std::stoul(argv[4])),
              std::uint32_t(std::stoul(argv[5])),
              std::uint32_t(std::stoul(argv[6]))};
    for (int i = 7; i < argc; i += 2) {
        if (i + 1 == argc) throw std::invalid_argument("missing option value");
        const std::string_view flag = argv[i];
        if (flag == "--read-budget-mib")
            o.budget = std::stoull(argv[i + 1]) * 1024 * 1024;
        else if (flag == "--incremental-us")
            o.incremental_us = std::stoul(argv[i + 1]);
        else if (flag == "--overlap")
            o.work = std::stoul(argv[i + 1]);
        else if (flag == "--dump")
            o.dump = argv[i + 1];
        else
            throw std::invalid_argument("unknown option");
    }
    if (!o.batch || o.batch > 8 || !o.tokens || !o.iterations ||
        std::uint64_t(o.batch) * o.tokens > std::uint32_t(INT32_MAX) / 2560)
        throw std::invalid_argument("invalid workload");
    return o;
}

template <class F>
void until(F ready) {
    const auto end = Clock::now() + std::chrono::seconds(30);
    while (!ready()) {
        if (Clock::now() >= end) throw std::runtime_error("PLE benchmark timed out");
        std::this_thread::yield();
    }
}

double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

template <class T>
void write(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void run(const Options& o) {
    artifact::Reader artifact(o.artifact);
    const auto& binding = artifact.directory().bindings.at("text/layers/1/ple/table");
    disk::Table table(artifact,
                      load_lookup_parameters(
                          artifact, binding,
                          artifact.directory().component("text").resources.at("ple_hash_controls"),
                          {3, 8}));
    const auto eos =
        artifact.directory().component("text").config.at("eos_token_id").get<std::int32_t>();
    std::uint32_t base_row = 0;
    for (const auto& part : binding.parts) {
        for (const auto& s : table.row(base_row).object.segments)
            if (s.file_offset % 4096 || s.destination_offset % 4096)
                throw std::invalid_argument("read budget requires page-aligned artifact volumes");
        base_row += artifact.geometry(part.object).shape[0];
    }
    for (auto file : table.files())
        if (file.offset_alignment > 4096 || file.memory_alignment > 4096)
            throw std::invalid_argument("benchmark budget requires 4 KiB direct reads");
    const auto window_bound = std::uint64_t(o.batch) * o.tokens * 16 * 4096;
    std::ifstream input(o.corpus, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() < 0 || std::size_t(input.tellg()) % 4)
        throw std::runtime_error("invalid raw U32 corpus");
    std::vector<std::int32_t> corpus(std::size_t(input.tellg()) / 4);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(corpus.data()), corpus.size() * 4);
    if (!input || std::uint64_t(2 + 100000) + (o.batch - 1) * 100003ULL +
                          std::uint64_t(o.iterations) * o.tokens >
                      corpus.size())
        throw std::invalid_argument("corpus does not cover workload");

    DeviceContext context;
    const auto positions = o.batch * o.tokens;
    UploadSlots uploads(context, positions, 2);
    disk::PrepareService service(table, eos,
                                 {.max_positions = positions,
                                  .io_depth      = 128,
                                  .cache_bytes   = std::size_t(o.cache_mib) * 1024 * 1024},
                                 uploads.host_slots());
    const auto memory = uploads.memory();
    std::cerr << "device=" << context.properties().name
              << " ordinary_host=" << memory.ordinary_host_bytes + service.memory_bytes()
              << " pinned_host=" << memory.pinned_host_bytes << " device_E=" << memory.device_bytes
              << " read_budget=" << o.budget << '\n';

    std::optional<artifact::MaterializedArtifact> weights;
    Weight projection;
    DeviceBuffer independent_input, projected;
    Tensor independent_x, y;
    DecodeGraphDefinition definition;
    DecodeGraphExecutable independent_work;
    if (o.work) {
        const auto handle =
            artifact.directory().bindings.at("text/layers/1/ple/key").parts.at(0).object;
        if (artifact.geometry(handle).shape != artifact::Shape{12800, 2560})
            throw std::runtime_error("unexpected merged PLE projection");
        const std::array requests{artifact::PlacementRequest{handle, context.device()}};
        weights.emplace(artifact::materialize(
            artifact, artifact::plan_materialization(artifact, requests), context.transfer_view()));
        const auto& parent = weights->device_parent(handle);
        projection = native_weight({{12800, 2560}, {{&parent, 0, parent.geometry.elements}}});
        independent_input = DeviceBuffer(context.device(), std::size_t(positions) * 5120);
        projected         = DeviceBuffer(context.device(), std::size_t(positions) * 25600);
        independent_input.fill();
        independent_x =
            Tensor(independent_input.span(), DType::BF16, {2560, std::int32_t(positions)});
        y = Tensor(projected.span(), DType::BF16, {12800, std::int32_t(positions)});
        ops::linear(independent_x, projection, y, context.execution_view());
        context.synchronize();
        definition.capture(context.stream(), [&] {
            for (std::uint32_t i = 0; i < o.work; ++i)
                ops::linear(independent_x, projection, y, context.execution_view());
        });
        independent_work.instantiate(definition);
        independent_work.upload(context.stream());
        context.synchronize();
    }
    CudaEventTimer work_time(context, context.stream()), exposed_time(context, context.stream());
    std::vector<std::uint16_t> downloaded(std::size_t(positions) * 2560);
    std::cout << "sample,batch,tokens,cache_mib,host_ms,device_ms,last_ids_to_ready_ms,reads,read_"
                 "MiB,cache_hits,shared_rows,work_ms,exposed_ms\n";
    for (std::uint32_t sample = 0; sample < o.iterations; ++sample) {
        if (service.stats().reads * 4096 + window_bound > o.budget) {
            std::cerr << "read budget reached after " << sample << " samples\n";
            break;
        }
        std::array<InputRow, 8> rows;
        std::array<std::span<const std::int32_t>, 8> ids;
        std::array<std::uint32_t, 8> lengths;
        for (std::uint32_t b = 0; b < o.batch; ++b) {
            const auto start = 2 + 100000 + b * 100003ULL + sample * o.tokens;
            ids[b]           = std::span(corpus).subspan(start, o.tokens);
            rows[b]          = {{b, sample, start, {corpus[start - 2], corpus[start - 1]}},
                       o.incremental_us ? ids[b].first(1) : ids[b]};
            lengths[b]       = o.tokens;
        }
        const auto before = service.stats();
        const auto start  = Clock::now();
        std::optional<Ticket> ticket;
        until([&] {
            ticket = service.prepare(o.tokens, std::span(rows).first(o.batch));
            return ticket.has_value();
        });
        if (o.work) {
            work_time.start();
            independent_work.launch(context.stream());
            work_time.record_stop();
            exposed_time.start();
        }
        for (std::uint32_t t = 1; o.incremental_us && t < o.tokens; ++t) {
            std::this_thread::sleep_for(std::chrono::microseconds(o.incremental_us));
            std::array<Append, 8> append;
            for (std::uint32_t b = 0; b < o.batch; ++b) append[b] = {b, ids[b].subspan(t, 1)};
            service.append(*ticket, std::span(append).first(o.batch));
        }
        const auto last_ids = o.incremental_us ? Clock::now() : start;
        service.seal(*ticket, std::span(lengths).first(o.batch));
        until([&] {
            const auto state = service.poll(*ticket);
            if (state.error) std::rethrow_exception(state.error);
            return state.status == disk::PrepareStatus::HostReady;
        });
        const auto host_ready = Clock::now();
        uploads.upload(service.embedding(*ticket), context.transfer_view());
        if (o.work) {
            const auto binding = uploads.acquire(*ticket, context.execution_view());
            binding.ready.wait(context.stream());
            const auto e = binding.values.reshape({2560, std::int32_t(positions)});
            exposed_time.record_stop();
            ops::linear(e, projection, y, context.execution_view());
            binding.consumed.record(context.stream());
            uploads.consumer_submitted(*ticket, context.execution_view());
        }
        until([&] { return uploads.upload_complete(*ticket); });
        const auto device_ready = Clock::now();
        const auto after        = service.stats();
        if (!o.dump.empty() && !sample) {
            const auto tensor = uploads.device_slot(ticket->slot, o.tokens, o.batch);
            CUDA_CHECK(
                cudaMemcpy(downloaded.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
            const auto host = service.embedding(*ticket);
            std::ofstream dump(o.dump, std::ios::binary);
            write(dump, o.tokens);
            write(dump, o.batch);
            write(dump, eos);
            for (std::uint32_t b = 0; b < o.batch; ++b) {
                write(dump, rows[b].identity.raw_tail);
                dump.write(reinterpret_cast<const char*>(ids[b].data()), o.tokens * 4);
            }
            dump.write(static_cast<const char*>(host.values.data), host.values.bytes);
            dump.write(reinterpret_cast<const char*>(downloaded.data()), host.values.bytes);
            if (o.work) {
                context.synchronize();
                std::vector<std::uint16_t> result(std::size_t(positions) * 12800);
                projected.copy_to_host(result.data(), result.size() * sizeof(result[0]));
                write(dump, std::uint32_t{12800});
                dump.write(reinterpret_cast<const char*>(result.data()),
                           result.size() * sizeof(result[0]));
            }
            dump.close();
            if (!dump) throw std::runtime_error("qualification dump failed");
        }
        until([&] { return service.poll(*ticket).drained && uploads.try_release(*ticket); });
        service.release(*ticket);
        std::cout << sample << ',' << o.batch << ',' << o.tokens << ',' << o.cache_mib << ','
                  << ms(start, host_ready) << ',' << ms(start, device_ready) << ','
                  << ms(last_ids, device_ready) << ',' << after.reads - before.reads << ','
                  << double(after.read_bytes - before.read_bytes) / (1024 * 1024) << ','
                  << after.cache_hits - before.cache_hits << ','
                  << after.shared_rows - before.shared_rows << ','
                  << (o.work ? work_time.elapsed_ms() : 0) << ','
                  << (o.work ? exposed_time.elapsed_ms() : 0) << '\n';
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout
            << "ninfer_ple_disk_prepare_bench ARTIFACT CORPUS B T ITERATIONS CACHE_MIB\n"
               "  --read-budget-mib N  Bound PLE requested bytes (default 512 MiB)\n"
               "  --incremental-us N   Append each next token column after this delay\n"
               "  --overlap N          Run N independent real K/V projections, then consume E\n"
               "  --dump PATH          Dump first input/Host E/Device E (and overlap projection)\n";
        return 0;
    }
    try {
        run(options(argc, argv));
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
