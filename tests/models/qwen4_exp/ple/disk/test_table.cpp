#include "models/qwen4_exp/ple/disk/table.h"
#include "core/buffer.h"
#include "core/io/read_queue.h"

#include <cstring>
#include <iostream>
#include <memory>

namespace {
template <class T>
void write(const T& value) {
    std::cout.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void run(const char* path) {
    using namespace ninfer;
    using namespace ninfer::artifact;
    using namespace models::qwen4_exp::ple;
    std::vector<std::uint32_t> starts;
    auto source = [&] {
        Reader reader(path);
        const auto& binding = reader.directory().bindings.at("text/layers/1/ple/table");
        std::uint32_t begin = 0;
        for (const auto& part : binding.parts) {
            starts.push_back(begin);
            begin += reader.geometry(part.object).shape[0];
        }
        return disk::Table(
            reader, load_lookup_parameters(
                        reader, binding,
                        reader.read_object(
                            reader.directory().component("text").resources.at("ple_hash_controls")),
                        {3, 8}));
    }();
    // Reader is gone. Preserve a borrowed row/LUT across an owner move.
    const auto first = source.row(0);
    auto table       = std::move(source);
    if (first.lut != table.row(0).lut) throw std::runtime_error("move invalidated PLE LUT");
    write(table.parameters().rows);
    const auto& controls = table.parameters().hash;
    for (const auto value : controls.multipliers) write(value);
    for (const auto value : controls.vocab_sizes) write(value);
    for (const auto value : controls.offsets) write(value);
    write(static_cast<std::uint32_t>(starts.size()));
    for (auto row : starts) write(*table.row(row).lut);

    std::vector<ReadSegment> segments(table.geometry().max_row_segments);
    std::vector<io::DirectReadWindow> windows(segments.size());
    HostBuffer slab(4096 * segments.size(), 4096);
    std::vector<MemorySpan> buffers;
    for (std::size_t i = 0; i < segments.size(); ++i)
        buffers.push_back(slab.span().subspan(i * 4096, 4096));
    io::DirectReadQueue queue(table.files(), buffers, buffers.size());
    std::vector<io::ReadCompletion> completions(buffers.size());
    std::uint32_t id;
    while (std::cin.read(reinterpret_cast<char*>(&id), sizeof(id))) {
        const auto row   = table.row(id);
        const auto count = map_file_range(row.object, row.offset, 90, segments);
        std::array<std::byte, 90> encoded{};
        for (std::size_t i = 0; i < count; ++i) {
            const auto& s = segments[i];
            windows[i] =
                io::direct_read_window(table.files()[s.file_index], s.file_offset, s.bytes);
            if (windows[i].bytes > buffers[i].bytes)
                throw std::runtime_error("test I/O slot too small");
            if (queue.try_enqueue({std::uint32_t(s.file_index), std::uint32_t(i), windows[i].offset,
                                   std::uint32_t(windows[i].bytes), i}) !=
                io::ReadAdmission::Accepted)
                throw std::runtime_error("row read rejected");
        }
        queue.submit();
        while (queue.outstanding()) {
            const auto n = queue.poll(completions);
            if (!n) {
                (void)queue.wait();
                continue;
            }
            for (std::size_t i = 0; i < n; ++i) {
                const auto c  = completions[i];
                const auto& w = windows[c.tag];
                const auto& s = segments[c.tag];
                if (c.error || c.bytes < w.required)
                    throw std::runtime_error("row read failed or truncated");
                std::memcpy(encoded.data() + s.destination_offset,
                            static_cast<const std::byte*>(buffers[c.buffer_index].data) + w.prefix,
                            s.bytes);
            }
        }
        std::array<std::uint16_t, 160> output;
        decode_nf4_row(encoded, *row.lut, output);
        write(output);
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    try {
        run(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
