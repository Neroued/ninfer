#include "artifact/reader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <functional>
#include <limits>
#include <span>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::artifact {
namespace {

using Json = nlohmann::json;

constexpr std::array<std::byte, 8> kMagic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{1},
};
constexpr std::uint64_t kPrefixBytes      = 16;
constexpr std::uint64_t kPayloadAlignment = 4096;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a + b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, std::string_view label) {
    const auto biased = checked_add(value, alignment - 1, label);
    return biased / alignment * alignment;
}

std::uint64_t read_u64_le(const std::byte* data) noexcept {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= std::uint64_t(std::to_integer<unsigned char>(data[i])) << (i * 8);
    }
    return value;
}

template <std::size_t N>
void require_members(const Json& value, const std::array<const char*, N>& members,
                     std::string_view label) {
    if (!value.is_object() || value.size() != N) {
        throw ArtifactError(std::string(label) + " has missing or extra members");
    }
    for (const char* member : members) {
        if (!value.contains(member)) {
            throw ArtifactError(std::string(label) + " has missing or extra members");
        }
    }
}

const std::string& require_string(const Json& value, std::string_view label) {
    if (!value.is_string()) {
        throw ArtifactError(std::string(label) + " must be a nonempty string");
    }
    const auto& result = value.get_ref<const std::string&>();
    if (result.empty()) { throw ArtifactError(std::string(label) + " must be a nonempty string"); }
    return result;
}

std::uint64_t require_unsigned(const Json& value, std::string_view label, bool positive) {
    if (!value.is_number_unsigned()) {
        throw ArtifactError(std::string(label) + " must be an integer");
    }
    const auto result = value.get<std::uint64_t>();
    if (positive && result == 0) { throw ArtifactError(std::string(label) + " must be positive"); }
    return result;
}

NumericFormat parse_format(std::string_view name) {
    if (name == "BF16") { return NumericFormat::BF16; }
    if (name == "FP32") { return NumericFormat::FP32; }
    if (name == "I32") { return NumericFormat::I32; }
    if (name == "Q4G64_F16S") { return NumericFormat::Q4G64_F16S; }
    if (name == "Q5G64_F16S") { return NumericFormat::Q5G64_F16S; }
    if (name == "Q6G64_F16S") { return NumericFormat::Q6G64_F16S; }
    if (name == "W8G32_F16S") { return NumericFormat::W8G32_F16S; }
    throw ArtifactError("unknown tensor format: " + std::string(name));
}

StorageLayout parse_layout(std::string_view name) {
    if (name == "contiguous-le-v1") { return StorageLayout::ContiguousLeV1; }
    if (name == "row-split-k128-v1") { return StorageLayout::RowSplitK128V1; }
    throw ArtifactError("unknown tensor layout: " + std::string(name));
}

ResourceEncoding parse_encoding(std::string_view name) {
    if (name == "raw-bytes-v1") { return ResourceEncoding::RawBytesV1; }
    throw ArtifactError("unknown resource encoding: " + std::string(name));
}

TensorDescriptor parse_tensor(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "shape", "format", "layout", "offset", "bytes",
    };
    require_members(value, members, "tensor entry");

    const auto name        = require_string(value.at("name"), "tensor name");
    const auto format      = parse_format(require_string(value.at("format"), "tensor format"));
    const auto layout      = parse_layout(require_string(value.at("layout"), "tensor layout"));
    const auto offset      = require_unsigned(value.at("offset"), "tensor offset", false);
    const auto stored_size = require_unsigned(value.at("bytes"), "tensor bytes", true);

    const auto& raw_shape = value.at("shape");
    if (!raw_shape.is_array()) { throw ArtifactError("tensor shape must be an array"); }
    std::vector<std::uint64_t> shape;
    shape.reserve(raw_shape.size());
    for (const auto& dim : raw_shape) {
        shape.push_back(require_unsigned(dim, "shape dimension", true));
    }

    const auto expected_size = tensor_encoded_size(layout, format, shape);
    if (stored_size != expected_size) {
        throw ArtifactError("tensor " + name + " stores " + std::to_string(stored_size) +
                            " bytes; layout requires " + std::to_string(expected_size));
    }
    return {name, std::move(shape), format, layout, offset, stored_size};
}

ResourceDescriptor parse_resource(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "encoding", "offset", "bytes",
    };
    require_members(value, members, "resource entry");
    return {
        require_string(value.at("name"), "resource name"),
        parse_encoding(require_string(value.at("encoding"), "resource encoding")),
        require_unsigned(value.at("offset"), "resource offset", false),
        require_unsigned(value.at("bytes"), "resource bytes", true),
    };
}

ObjectDescriptor parse_object(const Json& value) {
    if (!value.is_object()) { throw ArtifactError("each object entry must be a JSON object"); }
    const auto it = value.find("kind");
    if (it == value.end() || !it->is_string()) {
        throw ArtifactError("object kind must be 'tensor' or 'resource'");
    }
    const auto& kind = it->get_ref<const std::string&>();
    if (kind == "tensor") { return parse_tensor(value); }
    if (kind == "resource") { return parse_resource(value); }
    throw ArtifactError("object kind must be 'tensor' or 'resource'");
}

struct TransparentStringHash {
    using is_transparent = void;

    std::size_t operator()(std::string_view value) const noexcept {
        return std::hash<std::string_view>{}(value);
    }

    std::size_t operator()(const std::string& value) const noexcept {
        return (*this)(std::string_view(value));
    }
};

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path) {
#ifdef _WIN32
        mapping_file_ = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (mapping_file_ == INVALID_HANDLE_VALUE) {
            throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                    "CreateFileW " + path.string());
        }

        LARGE_INTEGER file_size{};
        if (!::GetFileSizeEx(mapping_file_, &file_size)) {
            const auto error = ::GetLastError();
            ::CloseHandle(mapping_file_);
            mapping_file_ = INVALID_HANDLE_VALUE;
            throw std::system_error(static_cast<int>(error), std::system_category(),
                                    "GetFileSizeEx " + path.string());
        }
        if (file_size.QuadPart < 0 ||
            static_cast<std::uint64_t>(file_size.QuadPart) >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            ::CloseHandle(mapping_file_);
            mapping_file_ = INVALID_HANDLE_VALUE;
            throw ArtifactError("artifact size does not fit the process address space");
        }

        size_ = static_cast<std::size_t>(file_size.QuadPart);
        if (size_ != 0) {
            mapping_ = ::CreateFileMappingW(mapping_file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (mapping_ == nullptr) {
                const auto error = ::GetLastError();
                ::CloseHandle(mapping_file_);
                mapping_file_ = INVALID_HANDLE_VALUE;
                throw std::system_error(static_cast<int>(error), std::system_category(),
                                        "CreateFileMappingW " + path.string());
            }
            const void* view = ::MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
            if (view == nullptr) {
                const auto error = ::GetLastError();
                ::CloseHandle(mapping_);
                ::CloseHandle(mapping_file_);
                mapping_      = nullptr;
                mapping_file_ = INVALID_HANDLE_VALUE;
                throw std::system_error(static_cast<int>(error), std::system_category(),
                                        "MapViewOfFile " + path.string());
            }
            data_ = static_cast<const std::byte*>(view);
        }

        direct_file_ = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                     OPEN_EXISTING,
                                     FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING |
                                         FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,
                                     nullptr);
        if (direct_file_ == INVALID_HANDLE_VALUE) {
            const auto error = ::GetLastError();
            if (data_ != nullptr) { ::UnmapViewOfFile(data_); }
            if (mapping_ != nullptr) { ::CloseHandle(mapping_); }
            ::CloseHandle(mapping_file_);
            data_         = nullptr;
            mapping_      = nullptr;
            mapping_file_ = INVALID_HANDLE_VALUE;
            throw std::system_error(static_cast<int>(error), std::system_category(),
                                    "CreateFileW direct " + path.string());
        }
#else
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd < 0) {
            throw std::system_error(errno, std::generic_category(), "open " + path.string());
        }

        struct stat status {};

        if (::fstat(fd, &status) != 0) {
            const int error = errno;
            ::close(fd);
            throw std::system_error(error, std::generic_category(), "fstat " + path.string());
        }
        if (status.st_size < 0 ||
            static_cast<std::uintmax_t>(status.st_size) > std::numeric_limits<std::size_t>::max()) {
            ::close(fd);
            throw ArtifactError("artifact size does not fit the process address space");
        }

        const auto size = static_cast<std::size_t>(status.st_size);
        void* mapping   = nullptr;
        if (size != 0) {
            mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapping == MAP_FAILED) {
                const int error = errno;
                ::close(fd);
                throw std::system_error(error, std::generic_category(), "mmap " + path.string());
            }
        }
        fd_   = fd;
        data_ = static_cast<const std::byte*>(mapping);
        size_ = size;
#endif
    }

    ~MappedFile() {
#ifdef _WIN32
        if (data_ != nullptr) { ::UnmapViewOfFile(data_); }
        if (mapping_ != nullptr) { ::CloseHandle(mapping_); }
        if (direct_file_ != INVALID_HANDLE_VALUE) { ::CloseHandle(direct_file_); }
        if (mapping_file_ != INVALID_HANDLE_VALUE) { ::CloseHandle(mapping_file_); }
#else
        if (data_ != nullptr) { ::munmap(const_cast<std::byte*>(data_), size_); }
        if (fd_ >= 0) { ::close(fd_); }
#endif
    }

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const std::byte* data() const noexcept { return data_; }

    std::size_t size() const noexcept { return size_; }

    std::size_t read_direct(std::uint64_t absolute_offset, std::span<std::byte> destination) const {
        constexpr std::size_t alignment = Reader::direct_io_alignment;
        if (absolute_offset % alignment != 0 || destination.size() % alignment != 0 ||
            reinterpret_cast<std::uintptr_t>(destination.data()) % alignment != 0) {
            throw ArtifactError("direct artifact read is not 4096-byte aligned");
        }
#ifdef _WIN32
        std::size_t total = 0;
        while (total < destination.size()) {
            constexpr std::size_t max_read = 1ULL << 30;
            const auto amount = static_cast<DWORD>(std::min(max_read, destination.size() - total));
            const std::uint64_t offset = absolute_offset + total;
            OVERLAPPED operation{};
            operation.Offset     = static_cast<DWORD>(offset & 0xffffffffULL);
            operation.OffsetHigh = static_cast<DWORD>(offset >> 32U);

            DWORD bytes = 0;
            const BOOL started = ::ReadFile(direct_file_, destination.data() + total, amount,
                                            &bytes, &operation);
            if (!started) {
                const auto error = ::GetLastError();
                if (error == ERROR_HANDLE_EOF) { break; }
                if (error != ERROR_IO_PENDING ||
                    !::GetOverlappedResult(direct_file_, &operation, &bytes, TRUE)) {
                    const auto final_error = error == ERROR_IO_PENDING ? ::GetLastError() : error;
                    throw std::system_error(static_cast<int>(final_error), std::system_category(),
                                            "direct artifact read");
                }
            }
            total += bytes;
            if (bytes != amount) { break; }
        }
        return total;
#else
        if (absolute_offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
            destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
            throw ArtifactError("direct artifact read exceeds platform I/O limits");
        }

        ssize_t bytes = -1;
        do {
            bytes = ::pread(fd_, destination.data(), destination.size(),
                            static_cast<off_t>(absolute_offset));
        } while (bytes < 0 && errno == EINTR);
        if (bytes < 0) {
            throw std::system_error(errno, std::generic_category(), "direct artifact read");
        }
        return static_cast<std::size_t>(bytes);
#endif
    }

private:
#ifdef _WIN32
    HANDLE mapping_file_       = INVALID_HANDLE_VALUE;
    HANDLE direct_file_        = INVALID_HANDLE_VALUE;
    HANDLE mapping_            = nullptr;
#else
    int fd_                = -1;
#endif
    const std::byte* data_ = nullptr;
    std::size_t size_      = 0;
};

} // namespace

std::string_view object_name(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) -> std::string_view { return descriptor.name; },
                      object);
}

std::uint64_t object_offset(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.offset; }, object);
}

std::uint64_t object_bytes(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.bytes; }, object);
}

struct Reader::Impl {
    explicit Impl(const std::filesystem::path& path) : file(path) {
        if (file.size() < kPrefixBytes) {
            throw ArtifactError("artifact is shorter than the v1 prefix");
        }
        if (!std::equal(kMagic.begin(), kMagic.end(), file.data())) {
            throw ArtifactError("artifact magic is not NInfer v1");
        }

        const auto json_bytes = read_u64_le(file.data() + 8);
        if (json_bytes == 0) { throw ArtifactError("json_bytes must be positive"); }
        const auto metadata_end = checked_add(kPrefixBytes, json_bytes, "JSON range");
        payload_start           = align_up(metadata_end, kPayloadAlignment, "payload offset");
        if (metadata_end > file.size() || payload_start > file.size()) {
            throw ArtifactError("declared JSON or payload start extends beyond the file");
        }

        Json directory;
        try {
            const auto* begin = reinterpret_cast<const char*>(file.data() + kPrefixBytes);
            directory         = Json::parse(begin, begin + json_bytes);
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid JSON directory: ") + error.what());
        }

        static constexpr std::array root_members = {"model_id", "objects"};
        require_members(directory, root_members, "directory root");
        model = require_string(directory.at("model_id"), "model_id");

        const auto& raw_objects = directory.at("objects");
        if (!raw_objects.is_array() || raw_objects.empty()) {
            throw ArtifactError("objects must be a nonempty array");
        }
        entries.reserve(raw_objects.size());
        index.reserve(raw_objects.size());

        const auto payload_bytes = static_cast<std::uint64_t>(file.size()) - payload_start;
        std::uint64_t cursor     = 0;
        for (const auto& raw_object : raw_objects) {
            auto object          = parse_object(raw_object);
            const auto name      = object_name(object);
            const auto offset    = object_offset(object);
            const auto bytes     = object_bytes(object);
            const auto alignment = std::visit(
                [](const auto& descriptor) {
                    using Descriptor = std::decay_t<decltype(descriptor)>;
                    if constexpr (std::is_same_v<Descriptor, TensorDescriptor>) {
                        return tensor_alignment(descriptor.layout);
                    } else {
                        return resource_alignment(descriptor.encoding);
                    }
                },
                object);

            if (offset < cursor) {
                throw ArtifactError("object " + std::string(name) + " overlaps or is out of order");
            }
            if (offset % alignment != 0) {
                throw ArtifactError("object " + std::string(name) + " is not " +
                                    std::to_string(alignment) + "-byte aligned");
            }
            const auto end = checked_add(offset, bytes, "object payload range");
            if (end > payload_bytes) {
                throw ArtifactError("object " + std::string(name) + " extends beyond the file");
            }
            const auto object_index = entries.size();
            auto [_, inserted]      = index.emplace(std::string(name), object_index);
            if (!inserted) { throw ArtifactError("duplicate object name: " + std::string(name)); }
            entries.push_back(std::move(object));
            cursor = end;
        }
    }

    MappedFile file;
    std::string model;
    std::vector<ObjectDescriptor> entries;
    std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>> index;
    std::uint64_t payload_start = 0;
};

Reader::Reader(const std::filesystem::path& path) : impl_(std::make_unique<Impl>(path)) {}

Reader::~Reader()                            = default;
Reader::Reader(Reader&&) noexcept            = default;
Reader& Reader::operator=(Reader&&) noexcept = default;

const std::string& Reader::model_id() const noexcept { return impl_->model; }

const std::vector<ObjectDescriptor>& Reader::objects() const noexcept { return impl_->entries; }

const ObjectDescriptor* Reader::find(std::string_view name) const noexcept {
    const auto it = impl_->index.find(name);
    return it == impl_->index.end() ? nullptr : &impl_->entries[it->second];
}

std::uint64_t Reader::file_bytes() const noexcept { return impl_->file.size(); }

std::uint64_t Reader::payload_offset() const noexcept { return impl_->payload_start; }

PayloadSpan Reader::payload(const ObjectDescriptor& object) const {
    const auto absolute =
        checked_add(impl_->payload_start, object_offset(object), "absolute payload offset");
    const auto end = checked_add(absolute, object_bytes(object), "absolute payload range");
    if (end > impl_->file.size()) { throw ArtifactError("object payload extends beyond the file"); }
    return {
        absolute,
        std::span<const std::byte>(impl_->file.data() + absolute,
                                   static_cast<std::size_t>(object_bytes(object))),
    };
}

PayloadSpan Reader::payload(std::string_view name) const {
    const auto* object = find(name);
    if (object == nullptr) { throw ArtifactError("unknown artifact object: " + std::string(name)); }
    return payload(*object);
}

std::size_t Reader::read_direct(std::uint64_t absolute_offset,
                                std::span<std::byte> destination) const {
    return impl_->file.read_direct(absolute_offset, destination);
}

} // namespace ninfer::artifact
