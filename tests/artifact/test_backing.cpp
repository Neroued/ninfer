#include "artifact/backing.h"
#include "artifact/fixture.h"
#include "core/buffer.h"
#include "core/io/read_queue.h"

#include <cstring>
#include <iostream>

namespace {
using namespace ninfer;
using namespace ninfer::artifact;
using namespace ninfer::test::artifact_fixture;

void run() {
    Fixture fixture;
    fixture.root["objects"][3]["format"] = "bf16";
    fixture.root["objects"][3]["layout"] = "contiguous_le_v1";
    fixture.root["objects"][3]["shape"]  = {32};
    fixture.write(true);
    ObjectHandle tensor, resource;
    FileBacking backing;
    {
        Reader reader(fixture.entry);
        tensor   = reader.find("q5");
        resource = reader.find("asset");
        const std::array objects{tensor, resource, tensor};
        backing = prepare_file_backing(reader, objects);
        require(backing.files().size() == 2, "selected files were not deduplicated");
        rejects([&] { (void)prepare_file_backing(reader, std::array{reader.find("unused")}); },
                "unavailable selected object accepted");
    }
    const auto view       = backing.object(tensor);
    const auto file_views = backing.files();
    FileBacking owner     = std::move(backing);
    require(owner.geometry(tensor).bytes == 528, "geometry did not survive reader/move");
    rejects([&] { (void)owner.geometry(resource); }, "resource has weight geometry");
    rejects([&] { (void)owner.object(ObjectHandle{999}); }, "absent object bound");
    std::array<ReadSegment, 2> segments;
    const auto count = map_file_range(view, 0, view.bytes, segments);
    require(count == 2, "cross-file object mapping changed");
    require(map_file_range(view, view.bytes, 0, {}) == 0, "empty terminal range failed");
    rejects([&] { (void)map_file_range(view, view.bytes, 1, segments); },
            "range overflow accepted");
    rejects([&] { (void)map_file_range(view, 0, view.bytes, std::span(segments).first(1)); },
            "small output capacity accepted");

    HostBuffer slab(8192, 4096);
    const std::array buffers{slab.span().subspan(0, 4096), slab.span().subspan(4096, 4096)};
    io::DirectReadQueue queue(file_views, buffers, 2);
    std::array<io::DirectReadWindow, 2> windows;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& s = segments[i];
        windows[i]    = io::direct_read_window(file_views[s.file_index], s.file_offset, s.bytes);
        require(queue.try_enqueue({std::uint32_t(s.file_index), std::uint32_t(i), windows[i].offset,
                                   std::uint32_t(windows[i].bytes), i}) ==
                    io::ReadAdmission::Accepted,
                "backing read admission failed");
    }
    queue.submit();
    std::vector<std::byte> result(view.bytes);
    std::array<io::ReadCompletion, 2> done;
    while (queue.outstanding()) {
        const auto n = queue.poll(done);
        if (!n) {
            (void)queue.wait();
            continue;
        }
        for (std::size_t i = 0; i < n; ++i) {
            const auto c  = done[i];
            const auto& s = segments[c.tag];
            const auto& w = windows[c.tag];
            require(!c.error && c.bytes >= w.required, "backing short read lost required bytes");
            std::memcpy(result.data() + s.destination_offset,
                        static_cast<const std::byte*>(buffers[c.buffer_index].data) + w.prefix,
                        s.bytes);
        }
    }
    require(std::equal(result.begin(), result.end(), fixture.payload.begin() + 256),
            "cross-file bytes differ after Reader destruction");
}
} // namespace

int main() {
    try {
        run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
