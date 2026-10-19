#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/views.h"
#include "core/device.h"
#include "core/transfer.h"
#include "ninfer/types.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <optional>

namespace ninfer::test {
void materialization_cuda_errors(DeviceContext& device);
void encoded_slice_checks(std::optional<CudaExecutionView>);
} // namespace ninfer::test

namespace {

using namespace ninfer;
using namespace ninfer::artifact;
using namespace ninfer::test::artifact_fixture;

void cpu_materialization() {
    Fixture fixture;
    fixture.write();
    std::optional<MaterializedArtifact> backing;
    ObjectHandle matrix, resource;
    const std::byte* resource_address;
    StartupEvent complete{};
    const StartupObserver observer{.callback = [&](const StartupEvent& event) {
        if (event.phase == StartupPhase::WeightsMaterialize &&
            event.status == StartupStatus::Complete) {
            complete = event;
        }
    }};
    {
        Reader reader(fixture.entry);
        Binder binder(reader);
        matrix           = reader.find("q5");
        resource         = binder.resource("text", "tokenizer.json");
        resource_address = binder.resource_bytes(resource).data();
        const std::array requests{PlacementRequest{matrix, Device::cpu()}};
        backing.emplace(
            materialize(reader, plan_materialization(reader, requests, std::move(binder).finish()),
                        {}, &observer));
    }
    const auto& parent = backing->host_parent(matrix);
    require(parent.device == Device::cpu() &&
                reinterpret_cast<std::uintptr_t>(parent.data) % 256 == 0,
            "CPU tensor backing is not aligned ordinary Host storage");
    require(std::equal(parent.data, parent.data + 528, fixture.payload.begin() + 256),
            "CPU-only materialization changed the encoded tensor");
    require(backing->host_bytes(resource).data() == resource_address &&
                backing->stats().device_capacity_bytes == 0 && backing->stats().h2d_bytes == 0,
            "CPU-only materialization moved resources or allocated GPU weights");
    require(complete.current == 528 && complete.total == 528,
            "CPU-only loading omitted Host weight progress");
}

void materialization(DeviceContext& device) {
    Fixture fixture;
    fixture.write(true);
    std::optional<MaterializedArtifact> backing;
    ParameterReference row;
    ObjectHandle quantized;
    ObjectHandle divisors;
    ObjectHandle resource;
    const std::byte* original_host = nullptr;
    StartupEvent complete{};
    const StartupObserver observer{.callback = [&](const StartupEvent& event) {
        if (event.phase == StartupPhase::WeightsMaterialize &&
            event.status == StartupStatus::Complete) {
            complete = event;
        }
    }};
    {
        Reader reader(fixture.entry);
        Binder binder(reader);
        (void)binder.parameter("matrix", {2, 130});
        row       = binder.parameter("row", {1, 130});
        quantized = reader.find("q5");
        divisors  = reader.find("divisors");
        (void)binder.parameter("values", {2}, QType::FP32);
        resource      = binder.resource("text", "tokenizer.json");
        original_host = binder.resource_bytes(resource).data();
        const std::array requests{PlacementRequest{quantized, device.device()},
                                  PlacementRequest{divisors, device.device()},
                                  PlacementRequest{divisors, Device::cpu()}};
        backing.emplace(
            materialize(reader, plan_materialization(reader, requests, std::move(binder).finish()),
                        device.transfer_view(), &observer));
    }
    const auto& stats = backing->stats();
    require(stats.device_object_count == 2 && stats.h2d_bytes == 536 &&
                stats.device_capacity_bytes == 776 && stats.retained_host_bytes == 13,
            "resident parents were duplicated or file gaps were allocated");
    require(complete.current == 544 && complete.total == 544,
            "dual-destination loading did not account for both copies");
    require(backing->host_bytes(resource).data() == original_host,
            "retained Host bytes moved after creating borrowed resource views");
    require(backing->host_bytes(resource).size() == 5,
            "Host resource was lost after Reader destruction");
    const auto view   = bind_view(row, *backing, device.device());
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

void registered_materialization(DeviceContext& device) {
    Fixture fixture;
    fixture.write();
    Reader reader(fixture.entry);
    const auto matrix = reader.find("q5");
    const auto values = reader.find("divisors");
    const std::array requests{PlacementRequest{matrix, Device::cpu()},
                              PlacementRequest{matrix, Device::cpu(), HostAccess::CudaTransfer},
                              PlacementRequest{values, Device::cpu()}};
    auto plan = plan_materialization(reader, requests);
    require(plan.ordinary_host_objects.size() == 1 && plan.registered_host_objects.size() == 1 &&
                plan.registered_host_bytes >= 528,
            "Host registration did not merge duplicate access demands");
    auto backing      = materialize(reader, std::move(plan), device.transfer_view());
    const auto source = backing.host_transfer_view(matrix);
    require(source.host.device == Device::cpu() && source.host.host_pinned &&
                source.host.data == backing.host_parent(matrix).data && source.device_address &&
                !backing.host_transfer_view(values).host.bytes,
            "registered and ordinary Host objects lost their access contract");
    auto replacement =
        materialize(reader, plan_materialization(reader, requests), device.transfer_view());
    replacement = std::move(backing);
    DeviceBuffer gpu(device.device(), source.host.bytes);
    CudaHostCopyBatch copies(device.device(), 2);
    copies.append(gpu.span().subspan(0, 17), source.subspan(0, 17));
    copies.append(gpu.span().subspan(17), source.subspan(17, source.host.bytes - 17));
    copies.submit(device.transfer_view());
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream()));
    std::vector<std::byte> actual(gpu.size());
    gpu.copy_to_host(actual.data(), actual.size());
    require(std::equal(actual.begin(), actual.end(), fixture.payload.begin() + 256) &&
                std::memcmp(source.host.data, actual.data(), actual.size()) == 0,
            "moving registered backing changed CPU or CUDA access");
    const std::array both{PlacementRequest{matrix, Device::cpu(), HostAccess::CudaTransfer},
                          PlacementRequest{values, Device::cpu(), HostAccess::CudaTransfer}};
    replacement = materialize(reader, plan_materialization(reader, both), device.transfer_view());
    for (const auto object : {matrix, values}) {
        const auto mapped   = replacement.host_transfer_view(object);
        const auto expected = reader.read_object(object);
        DeviceBuffer target(device.device(), expected.size());
        copies.clear();
        copies.append(target.span(), mapped);
        copies.submit(device.transfer_view());
        CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream()));
        std::vector<std::byte> received(expected.size());
        target.copy_to_host(received.data(), received.size());
        require(received == expected, "registered parent mapping used the wrong backing offset");
    }
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
        const std::array requests{PlacementRequest{reader.find("q5"), device.device()}};
        rejects(
            [&] {
                (void)materialize(
                    reader, plan_materialization(reader, requests, std::move(binder).finish()),
                    device.transfer_view());
            },
            "invalid required continuation was uploaded");
    }
    fixture.write();
    Reader reader(fixture.entry);
    Binder binder(reader);
    const auto resource = binder.resource("text", "tokenizer.json");
    auto backing =
        materialize(reader, plan_materialization(reader, std::span<const PlacementRequest>{},
                                                 std::move(binder).finish()));
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
        std::vector<PlacementRequest> requests;
        for (const auto& [name, binding] : reader.directory().bindings) {
            auto shape = binding.whole_object
                             ? reader.directory().tensor(binding.parts.front().object).shape
                             : Shape{binding.elements};
            parameters.push_back(binder.parameter(name, std::move(shape)));
            for (const auto& part : binding.parts) {
                requests.push_back({part.object, device.device()});
            }
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
        if (writer_fixture) { requests.push_back({reader.find("nvfp4"), Device::cpu()}); }
        auto plan = plan_materialization(reader, requests, std::move(binder).finish());
        for (const auto& placement : plan.device_objects) {
            const auto object = plan.tensors.at(placement.storage.index).whole_object.value();
            auto bytes        = reader.read_object(object);
            if (writer_fixture && reader.directory().tensor(object).id == "matrix") {
                require(bytes.size() == 130 * 130 * 2, "Python writer changed tensor dimensions");
                for (std::size_t i = 0; i < bytes.size(); ++i) {
                    require(bytes[i] == std::byte((i * 37 + 11) % 251),
                            "C++ reading differs from known Python writer input bytes");
                }
            }
            expected.emplace(object.index, std::move(bytes));
        }
        storage.emplace(materialize(reader, std::move(plan), device.transfer_view()));
        if (writer_fixture) {
            const auto& paged = storage->device_parent(reader.find("nf4"));
            require(paged.geometry.bytes == 8196 && paged.weight_divisors.size() == 1 &&
                        paged.weight_divisors[0] == 2.0F,
                    "NF4 materialization lost paged geometry or the FP32 divisor");
            const auto& bank = storage->device_parent(reader.find("nvfp4"));
            require(bank.weight_divisors.size() == 3 && bank.weight_divisors[0] == 2 &&
                        bank.weight_divisors[1] == 3 && bank.weight_divisors[2] == 5,
                    "NVFP4 materialization lost independent matrix divisors");
            require(bank.weight_divisors.data() ==
                        storage->host_parent(reader.find("nvfp4")).weight_divisors.data(),
                    "Host and CUDA parents do not share their owned divisor metadata");
            const auto slice = native_weight({{128, 128}, {{&bank, 384 * 128, 512 * 128}}});
            require(slice.weight_scale_divisor == 3 && slice.qdata == bank.data + 24576 &&
                        slice.scales == bank.data + 52224,
                    "materialized NVFP4 slice used a different expert");
        }
    }
    for (const auto& [index, bytes] : expected) {
        std::vector<std::byte> actual(bytes.size());
        CUDA_CHECK(cudaMemcpy(actual.data(), storage->device_parent({index}).data, actual.size(),
                              cudaMemcpyDeviceToHost));
        require(actual == bytes, "small artifact upload differs from source bytes");
    }
    for (const auto& parameter : parameters) {
        const auto view        = bind_view(parameter, *storage, device.device());
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
    const std::array requests{PlacementRequest{reader.find("large"), device.device()},
                              PlacementRequest{reader.find("large"), Device::cpu()}};
    auto backing =
        materialize(reader, plan_materialization(reader, requests, std::move(binder).finish()),
                    device.transfer_view());
    const auto* base = backing.device_parent(reader.find("large")).data;
    std::vector<std::byte> chunk(1024 * 1024);
    for (std::size_t offset = 0; offset < bytes; offset += chunk.size()) {
        const auto count = std::min(chunk.size(), bytes - offset);
        CUDA_CHECK(cudaMemcpy(chunk.data(), base + offset, count, cudaMemcpyDeviceToHost));
        require(std::memcmp(chunk.data(), backing.host_bytes(reader.find("large")).data() + offset,
                            count) == 0,
                "CPU/GPU fanout differs");
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
        if (argc == 2 && std::string_view(argv[1]) == "--cpu-only") {
            cpu_materialization();
            ninfer::test::encoded_slice_checks({});
            std::cout << "CPU-only tensor and resource materialization passed\n";
            return 0;
        }
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
        ninfer::test::encoded_slice_checks(device.transfer_view());
        registered_materialization(device);
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
