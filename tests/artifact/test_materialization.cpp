#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/views.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <optional>

namespace ninfer::test {
void materialization_cuda_errors(DeviceContext& device);
}

namespace {

using namespace ninfer;
using namespace ninfer::artifact;
using namespace ninfer::test::artifact_fixture;

void materialization(DeviceContext& device) {
    Fixture fixture;
    fixture.write(true);
    std::optional<MaterializedArtifact> backing;
    ParameterReference row;
    ObjectHandle quantized;
    ObjectHandle divisors;
    ObjectHandle resource;
    const std::byte* original_host = nullptr;
    {
        Reader reader(fixture.entry);
        Binder binder(reader);
        (void)binder.parameter("matrix", {2, 130});
        row       = binder.parameter("row", {1, 130});
        quantized = reader.find("q5");
        divisors  = reader.find("divisors");
        (void)binder.parameter("values", {2}, Residency::Host, QType::FP32);
        binder.require_device(divisors);
        resource      = binder.resource("text", "tokenizer.json");
        original_host = binder.host_object(divisors).data();
        backing.emplace(materialize(reader, std::move(binder).finish(), device));
    }
    const auto& stats = backing->stats();
    require(stats.device_object_count == 2 && stats.h2d_bytes == 536 &&
                stats.device_capacity_bytes == 776 && stats.retained_host_bytes == 13,
            "resident parents were duplicated or file gaps were allocated");
    require(backing->host_bytes(divisors).data() == original_host,
            "retained Host bytes moved after creating borrowed resource views");
    require(backing->host_bytes(resource).size() == 5,
            "Host resource was lost after Reader destruction");
    const auto view   = bind_view(row, *backing);
    const auto native = native_weight(view);
    require(native.n == 1 && native.k == 130 &&
                native.payload == backing->device_parent(quantized).data,
            "row view no longer refers to its owning parent");
    std::vector<std::byte> downloaded(528);
    CUDA_CHECK(cudaMemcpy(downloaded.data(), backing->device_parent(quantized).data,
                          downloaded.size(), cudaMemcpyDeviceToHost));
    require(std::equal(downloaded.begin(), downloaded.end(), fixture.payload.begin() + 256),
            "cross-file upload changed encoded parent bytes");
    std::array<std::byte, 8> values{};
    CUDA_CHECK(cudaMemcpy(values.data(), backing->device_parent(divisors).data, values.size(),
                          cudaMemcpyDeviceToHost));
    require(std::equal(values.begin(), values.end(), backing->host_bytes(divisors).begin()),
            "Host/device demand did not retain identical bytes");
    std::array<std::byte, 1> code{};
    CUDA_CHECK(cudaMemcpy(code.data(), native.qdata, 1, cudaMemcpyDeviceToHost));
    require(code[0] == std::byte{0x52}, "native row pointer addressed a different row");
}

void failure_and_host_only(DeviceContext& device) {
    Fixture fixture;
    fixture.write();
    {
        std::fstream part(fixture.directory / "weights-second.bin",
                          std::ios::binary | std::ios::in | std::ios::out);
        part.seekp(16);
        part.put(0);
    }
    {
        Reader reader(fixture.entry);
        Binder binder(reader);
        (void)binder.parameter("matrix", {2, 130});
        rejects([&] { (void)materialize(reader, std::move(binder).finish(), device); },
                "invalid required continuation was uploaded");
    }
    fixture.write();
    Reader reader(fixture.entry);
    Binder binder(reader);
    const auto resource = binder.resource("text", "tokenizer.json");
    auto backing        = materialize(reader, std::move(binder).finish(), device);
    require(backing.stats().device_capacity_bytes == 0 && backing.stats().h2d_bytes == 0 &&
                backing.host_bytes(resource).size() == 5,
            "Host-only demand allocated device weights");
    materialization(device);
}

void file_roundtrip(DeviceContext& device, const std::filesystem::path& path, bool writer_fixture) {
    std::optional<MaterializedArtifact> storage;
    std::map<std::size_t, std::vector<std::byte>> expected;
    std::vector<ParameterReference> parameters;
    {
        Reader reader(path);
        require(reader.file_bytes() < 32ULL * 1024 * 1024,
                "this complete-byte fixture check is limited to small artifacts");
        Binder binder(reader);
        for (const auto& [name, binding] : reader.directory().bindings) {
            auto shape = binding.whole_object
                             ? reader.directory().tensor(binding.parts.front().object).shape
                             : Shape{binding.elements};
            parameters.push_back(binder.parameter(name, std::move(shape)));
        }
        for (const auto& [key, use] : reader.directory().uses) {
            for (const auto& [name, auxiliary] : use.auxiliaries) {
                (void)binder.values(auxiliary);
            }
        }
        for (const auto& [name, component] : reader.directory().components) {
            for (const auto& [role, object] : component.resources) {
                (void)binder.resource(name, role);
            }
        }
        auto plan = std::move(binder).finish();
        for (const auto& placement : plan.device_objects) {
            auto bytes = reader.read_object(placement.object);
            if (writer_fixture && reader.directory().tensor(placement.object).id == "matrix") {
                require(bytes.size() == 130 * 130 * 2, "Python writer changed tensor dimensions");
                for (std::size_t i = 0; i < bytes.size(); ++i) {
                    require(bytes[i] == std::byte((i * 37 + 11) % 251),
                            "C++ reading differs from known Python writer input bytes");
                }
            }
            expected.emplace(placement.object.index, std::move(bytes));
        }
        storage.emplace(materialize(reader, std::move(plan), device));
    }
    for (const auto& [index, bytes] : expected) {
        std::vector<std::byte> actual(bytes.size());
        CUDA_CHECK(cudaMemcpy(actual.data(), storage->device_parent({index}).data, actual.size(),
                              cudaMemcpyDeviceToHost));
        require(actual == bytes, "small artifact upload differs from source bytes");
    }
    for (const auto& parameter : parameters) {
        const auto view        = bind_view(parameter, *storage);
        std::uint64_t elements = 0;
        for (const auto& part : view.parts) {
            elements += part.end - part.begin;
            require(part.parent->data != nullptr, "resolved view lost its parent");
        }
        require(elements == weight_element_count(view.shape),
                "resolved view lost logical coverage");
    }
    std::cout << path.filename().string() << ": all bound parent bytes and logical views passed\n";
}

void staging_reuse(DeviceContext& device) {
    // More than one full staging ring, with distinct pages and a partial final block.
    constexpr std::size_t bytes = 5ULL * 64 * 1024 * 1024 + 1024;
    Fixture fixture;
    fixture.payload.clear();
    fixture.root    = {{"components", {{"text", {{"config", Json::object()}}}}},
                       {"objects", Json::array({{{"id", "large"},
                                                 {"kind", "tensor"},
                                                 {"shape", {bytes / 2}},
                                                 {"format", "bf16"},
                                                 {"layout", "contiguous_le_v1"},
                                                 {"offset", 0},
                                                 {"bytes", bytes}}})},
                       {"bindings", {{"large", {{"object", "large"}}}}},
                       {"uses", Json::array()},
                       {"files", Json::array({{{"path", nullptr}, {"payload_bytes", bytes}}})}};
    const auto text = fixture.root.dump();
    std::array<std::byte, 4096> header{};
    const std::array<unsigned char, 8> magic{'N', 'I', 'N', 'F', 'E', 'R', 0, 3};
    for (std::size_t i = 0; i < magic.size(); ++i) { header[i] = std::byte(magic[i]); }
    put_word(header, 8, text.size(), 8);
    std::memcpy(header.data() + 32, text.data(), text.size());
    std::ofstream file(fixture.entry, std::ios::binary | std::ios::trunc);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.write(reinterpret_cast<const char*>(header.data()), header.size());
    std::array<std::byte, 4096> page{};
    for (std::size_t offset = 0; offset < bytes; offset += page.size()) {
        for (std::size_t i = 0; i < page.size(); ++i) {
            page[i] = std::byte(((offset + i) * 17 + (offset / 4096) * 13) % 251);
        }
        file.write(reinterpret_cast<const char*>(page.data()),
                   std::min(page.size(), bytes - offset));
    }
    file.close();
    Reader reader(fixture.entry);
    Binder binder(reader);
    (void)binder.parameter("large", {bytes / 2});
    auto backing     = materialize(reader, std::move(binder).finish(), device);
    const auto* base = backing.device_parent(reader.find("large")).data;
    std::vector<std::byte> chunk(1024 * 1024);
    for (std::size_t offset = 0; offset < bytes; offset += chunk.size()) {
        const auto count = std::min(chunk.size(), bytes - offset);
        CUDA_CHECK(cudaMemcpy(chunk.data(), base + offset, count, cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < count; ++i) {
            const auto position = offset + i;
            require(chunk[i] == std::byte((position * 17 + (position / 4096) * 13) % 251),
                    "staging slot reuse overwrote an in-flight or later block");
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        int count         = 0;
        const auto result = cudaGetDeviceCount(&count);
        if (result == cudaErrorNoDevice || result == cudaErrorInsufficientDriver ||
            (result == cudaSuccess && count == 0)) {
            return 77;
        }
        CUDA_CHECK(result);
        DeviceContext device;
        if (argc == 3 && (std::string_view(argv[1]) == "--artifact" ||
                          std::string_view(argv[1]) == "--writer-fixture")) {
            file_roundtrip(device, argv[2], std::string_view(argv[1]) == "--writer-fixture");
            return 0;
        }
        if (argc != 1) {
            throw std::invalid_argument("expected [--artifact|--writer-fixture PATH]");
        }
        materialization(device);
        failure_and_host_only(device);
        ninfer::test::materialization_cuda_errors(device);
        staging_reuse(device);
        std::cout << "artifact materialization checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
