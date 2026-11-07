#include "core/disk_kv_cache.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <unistd.h>

namespace ninfer {

// ---------------------------------------------------------------------------------------------
// XXH64

namespace {

constexpr std::uint64_t kPrime1 = 11400714785074694791ULL;
constexpr std::uint64_t kPrime2 = 14029467366897019727ULL;
constexpr std::uint64_t kPrime3 = 1609587929392839161ULL;
constexpr std::uint64_t kPrime4 = 9650029242287828579ULL;
constexpr std::uint64_t kPrime5 = 2870177450012600261ULL;

inline std::uint64_t rotl(std::uint64_t x, int r) noexcept { return (x << r) | (x >> (64 - r)); }

inline std::uint64_t read64(const unsigned char* p) noexcept {
    std::uint64_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline std::uint32_t read32(const unsigned char* p) noexcept {
    std::uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline std::uint64_t round(std::uint64_t acc, std::uint64_t input) noexcept {
    acc += input * kPrime2;
    acc = rotl(acc, 31);
    return acc * kPrime1;
}

inline std::uint64_t merge_round(std::uint64_t acc, std::uint64_t value) noexcept {
    value = round(0, value);
    acc ^= value;
    return acc * kPrime1 + kPrime4;
}

std::uint64_t finalize(std::uint64_t h, const unsigned char* p, std::size_t length) noexcept {
    while (length >= 8) {
        h ^= round(0, read64(p));
        h = rotl(h, 27) * kPrime1 + kPrime4;
        p += 8;
        length -= 8;
    }
    if (length >= 4) {
        h ^= static_cast<std::uint64_t>(read32(p)) * kPrime1;
        h = rotl(h, 23) * kPrime2 + kPrime3;
        p += 4;
        length -= 4;
    }
    while (length > 0) {
        h ^= static_cast<std::uint64_t>(*p) * kPrime5;
        h = rotl(h, 11) * kPrime1;
        ++p;
        --length;
    }
    h ^= h >> 33;
    h *= kPrime2;
    h ^= h >> 29;
    h *= kPrime3;
    h ^= h >> 32;
    return h;
}

} // namespace

std::uint64_t xxh64(const void* data, std::size_t length, std::uint64_t seed) {
    Xxh64Stream stream(seed);
    stream.update(data, length);
    return stream.digest();
}

void Xxh64Stream::reset(std::uint64_t seed) noexcept {
    seed_     = seed;
    v_[0]     = seed + kPrime1 + kPrime2;
    v_[1]     = seed + kPrime2;
    v_[2]     = seed;
    v_[3]     = seed - kPrime1;
    total_    = 0;
    buffered_ = 0;
}

void Xxh64Stream::update(const void* data, std::size_t length) noexcept {
    const auto* p = static_cast<const unsigned char*>(data);
    total_ += length;
    if (buffered_ + length < 32) {
        std::memcpy(buffer_ + buffered_, p, length);
        buffered_ += length;
        return;
    }
    if (buffered_ != 0) {
        const std::size_t fill = 32 - buffered_;
        std::memcpy(buffer_ + buffered_, p, fill);
        v_[0] = round(v_[0], read64(buffer_));
        v_[1] = round(v_[1], read64(buffer_ + 8));
        v_[2] = round(v_[2], read64(buffer_ + 16));
        v_[3] = round(v_[3], read64(buffer_ + 24));
        p += fill;
        length -= fill;
        buffered_ = 0;
    }
    std::uint64_t v0 = v_[0], v1 = v_[1], v2 = v_[2], v3 = v_[3];
    while (length >= 32) {
        v0 = round(v0, read64(p));
        v1 = round(v1, read64(p + 8));
        v2 = round(v2, read64(p + 16));
        v3 = round(v3, read64(p + 24));
        p += 32;
        length -= 32;
    }
    v_[0] = v0;
    v_[1] = v1;
    v_[2] = v2;
    v_[3] = v3;
    if (length != 0) {
        std::memcpy(buffer_, p, length);
        buffered_ = length;
    }
}

std::uint64_t Xxh64Stream::digest() const noexcept {
    std::uint64_t h;
    if (total_ >= 32) {
        h = rotl(v_[0], 1) + rotl(v_[1], 7) + rotl(v_[2], 12) + rotl(v_[3], 18);
        h = merge_round(h, v_[0]);
        h = merge_round(h, v_[1]);
        h = merge_round(h, v_[2]);
        h = merge_round(h, v_[3]);
    } else {
        h = seed_ + kPrime5;
    }
    h += total_;
    return finalize(h, buffer_, buffered_);
}

// ---------------------------------------------------------------------------------------------
// Header encoding (format 2)

namespace {

constexpr char kMagic[8] = {'N', 'I', 'D', 'K', 'V', '0', '0', '2'};
constexpr std::size_t kFixedHeaderBytes = 168;
constexpr std::size_t kCheckpointBytes  = 24;
constexpr std::size_t kSectionBytes     = 32;
// Dirty page-cache bytes one record may hold before its writer waits for the writeback.
constexpr std::uint64_t kDiskKVDirtyWindow = 128ULL << 20U;
constexpr std::uint64_t kGiB               = 1ULL << 30U;

constexpr std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) noexcept {
    return (value + alignment - 1U) / alignment * alignment;
}

std::uint64_t block_count(std::uint64_t bytes) noexcept {
    return (bytes + kDiskKVBlockBytes - 1U) / kDiskKVBlockBytes;
}

class ByteWriter {
public:
    explicit ByteWriter(std::vector<std::byte>& out) : out_(out) {}
    template <class T> void put(T value) {
        const auto* p = reinterpret_cast<const std::byte*>(&value);
        out_.insert(out_.end(), p, p + sizeof(T));
    }
    void put_bytes(const void* data, std::size_t length) {
        const auto* p = static_cast<const std::byte*>(data);
        out_.insert(out_.end(), p, p + length);
    }

private:
    std::vector<std::byte>& out_;
};

class ByteReader {
public:
    ByteReader(const std::byte* data, std::size_t length) : data_(data), length_(length) {}
    template <class T> bool get(T& value) {
        if (length_ - offset_ < sizeof(T)) { return false; }
        std::memcpy(&value, data_ + offset_, sizeof(T));
        offset_ += sizeof(T);
        return true;
    }
    bool get_bytes(void* out, std::size_t length) {
        if (length_ - offset_ < length) { return false; }
        std::memcpy(out, data_ + offset_, length);
        offset_ += length;
        return true;
    }
    [[nodiscard]] std::size_t offset() const noexcept { return offset_; }

private:
    const std::byte* data_;
    std::size_t length_;
    std::size_t offset_ = 0;
};

// Lays out the sections (tokens first) and returns the header size.
std::uint64_t layout_sections(std::vector<DiskKVSectionInfo>& sections, std::size_t checkpoints,
                              std::size_t key_bytes) {
    std::uint64_t blocks = 0;
    for (DiskKVSectionInfo& section : sections) {
        section.first_block = blocks;
        blocks += block_count(section.bytes);
    }
    const std::uint64_t header = align_up(kFixedHeaderBytes + key_bytes +
                                              checkpoints * kCheckpointBytes +
                                              sections.size() * kSectionBytes + blocks * 8U + 8U,
                                          kDiskKVAlignment);
    std::uint64_t offset = header;
    for (DiskKVSectionInfo& section : sections) {
        section.offset = offset;
        offset         = align_up(offset + section.bytes, kDiskKVAlignment);
    }
    return header;
}

std::uint64_t sections_end(const std::vector<DiskKVSectionInfo>& sections,
                           std::uint64_t header) noexcept {
    std::uint64_t end = header;
    for (const DiskKVSectionInfo& section : sections) {
        end = std::max(end, align_up(section.offset + section.bytes, kDiskKVAlignment));
    }
    return end;
}

std::vector<std::byte> encode_header(const DiskKVRecordHeader& header) {
    std::vector<std::byte> out;
    out.reserve(header.header_bytes);
    ByteWriter writer(out);
    writer.put_bytes(kMagic, sizeof(kMagic));
    writer.put<std::uint32_t>(kDiskKVFormatVersion);
    writer.put<std::uint32_t>(static_cast<std::uint32_t>(header.kind));
    writer.put<std::uint64_t>(header.header_bytes);
    writer.put<std::uint64_t>(header.file_bytes);
    writer.put<std::uint64_t>(header.id);
    writer.put<std::uint64_t>(header.blob.a);
    writer.put<std::uint64_t>(header.blob.b);
    writer.put<std::uint64_t>(header.key_hash);
    writer.put<std::uint64_t>(header.created_ns);
    writer.put<std::uint64_t>(header.full_digest);
    writer.put<std::uint32_t>(static_cast<std::uint32_t>(header.key.size()));
    writer.put<std::uint32_t>(header.token_count);
    writer.put<std::uint32_t>(static_cast<std::uint32_t>(header.checkpoints.size()));
    writer.put<std::uint32_t>(static_cast<std::uint32_t>(header.sections.size()));
    writer.put<std::uint64_t>(header.block_hashes.size());
    for (const std::uint64_t word : header.meta) { writer.put<std::uint64_t>(word); }
    writer.put_bytes(header.key.data(), header.key.size());
    for (const DiskKVCheckpoint& checkpoint : header.checkpoints) {
        writer.put<std::uint32_t>(checkpoint.frontier);
        writer.put<std::uint32_t>(checkpoint.kind);
        writer.put<std::uint32_t>(checkpoint.backend_frontier);
        writer.put<std::uint32_t>(checkpoint.state_section);
        writer.put<std::uint64_t>(checkpoint.digest);
    }
    for (const DiskKVSectionInfo& section : header.sections) {
        writer.put<std::uint32_t>(section.kind);
        writer.put<std::uint32_t>(0);
        writer.put<std::uint64_t>(section.bytes);
        writer.put<std::uint64_t>(section.offset);
        writer.put<std::uint64_t>(section.first_block);
    }
    for (const std::uint64_t hash : header.block_hashes) { writer.put<std::uint64_t>(hash); }
    writer.put<std::uint64_t>(xxh64(out.data(), out.size()));
    if (out.size() > header.header_bytes) {
        throw std::logic_error("disk KV header exceeds its reserved size");
    }
    out.resize(header.header_bytes, std::byte{0});
    return out;
}

bool pread_all(int fd, void* data, std::size_t length, std::uint64_t offset) {
    auto* p = static_cast<std::byte*>(data);
    while (length != 0) {
        const ssize_t got = ::pread(fd, p, length, static_cast<off_t>(offset));
        if (got < 0) {
            if (errno == EINTR) { continue; }
            return false;
        }
        if (got == 0) { return false; }
        p += got;
        length -= static_cast<std::size_t>(got);
        offset += static_cast<std::uint64_t>(got);
    }
    return true;
}

bool pwrite_all(int fd, const void* data, std::size_t length, std::uint64_t offset) {
    const auto* p = static_cast<const std::byte*>(data);
    while (length != 0) {
        const ssize_t put = ::pwrite(fd, p, length, static_cast<off_t>(offset));
        if (put < 0) {
            if (errno == EINTR) { continue; }
            return false;
        }
        p += put;
        length -= static_cast<std::size_t>(put);
        offset += static_cast<std::uint64_t>(put);
    }
    return true;
}

std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::int64_t mtime_ns(const struct stat& st) {
    return static_cast<std::int64_t>(st.st_mtim.tv_sec) * 1000000000LL + st.st_mtim.tv_nsec;
}

void fsync_directory(const std::filesystem::path& directory) {
    const int fd = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd >= 0) {
        (void)::fsync(fd);
        ::close(fd);
    }
}

std::string hex64(std::uint64_t id) {
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(id));
    return text;
}

std::optional<std::uint64_t> parse_hex64(const std::string& text) {
    if (text.size() != 16) { return std::nullopt; }
    std::uint64_t value = 0;
    for (const char c : text) {
        value <<= 4U;
        if (c >= '0' && c <= '9') {
            value |= static_cast<std::uint64_t>(c - '0');
        } else if (c >= 'a' && c <= 'f') {
            value |= static_cast<std::uint64_t>(c - 'a' + 10);
        } else {
            return std::nullopt;
        }
    }
    return value;
}

bool remove_temporary_if_dead(const std::filesystem::path& path) {
    const std::string name = path.filename().string();
    const std::size_t temp = name.find(".tmp.");
    if (temp == std::string::npos) { return false; }
    const long pid = std::strtol(name.c_str() + temp + 5, nullptr, 10);
    const bool dead = pid <= 0 || pid == static_cast<long>(::getpid()) ||
                      (::kill(static_cast<pid_t>(pid), 0) != 0 && errno == ESRCH);
    std::error_code error;
    return dead && std::filesystem::remove(path, error);
}

} // namespace

std::string DiskKVBlobId::hex() const { return hex64(a) + hex64(b); }

// ---------------------------------------------------------------------------------------------
// Header decoding

std::optional<DiskKVRecordHeader> DiskKVCache::read_header(int fd, std::uint64_t file_bytes) {
    if (file_bytes < kDiskKVAlignment) { return std::nullopt; }
    std::vector<std::byte> head(kFixedHeaderBytes);
    if (!pread_all(fd, head.data(), head.size(), 0)) { return std::nullopt; }
    if (std::memcmp(head.data(), kMagic, sizeof(kMagic)) != 0) { return std::nullopt; }
    ByteReader fixed(head.data(), head.size());
    char magic[8];
    std::uint32_t version = 0, kind = 0, key_len = 0, token_count = 0, checkpoints = 0,
                  sections = 0;
    std::uint64_t blocks = 0;
    DiskKVRecordHeader header;
    (void)fixed.get_bytes(magic, sizeof(magic));
    (void)fixed.get(version);
    (void)fixed.get(kind);
    (void)fixed.get(header.header_bytes);
    (void)fixed.get(header.file_bytes);
    (void)fixed.get(header.id);
    (void)fixed.get(header.blob.a);
    (void)fixed.get(header.blob.b);
    (void)fixed.get(header.key_hash);
    (void)fixed.get(header.created_ns);
    (void)fixed.get(header.full_digest);
    (void)fixed.get(key_len);
    (void)fixed.get(token_count);
    (void)fixed.get(checkpoints);
    (void)fixed.get(sections);
    (void)fixed.get(blocks);
    for (std::uint64_t& word : header.meta) { (void)fixed.get(word); }
    if (version != kDiskKVFormatVersion ||
        (kind != static_cast<std::uint32_t>(DiskKVRecordKind::Manifest) &&
         kind != static_cast<std::uint32_t>(DiskKVRecordKind::Blob)) ||
        header.file_bytes != file_bytes || header.header_bytes < kDiskKVAlignment ||
        header.header_bytes > (64ULL << 20U) || header.header_bytes % kDiskKVAlignment != 0 ||
        header.header_bytes > file_bytes || key_len > (1U << 20U) || checkpoints > 4096 ||
        sections > 4096 || sections == 0 || blocks > (header.header_bytes / 8U)) {
        return std::nullopt;
    }
    header.kind = static_cast<DiskKVRecordKind>(kind);
    std::vector<std::byte> all(header.header_bytes);
    if (!pread_all(fd, all.data(), all.size(), 0)) { return std::nullopt; }
    ByteReader reader(all.data(), all.size());
    std::vector<std::byte> skip(kFixedHeaderBytes);
    (void)reader.get_bytes(skip.data(), skip.size());
    header.key.resize(key_len);
    if (!reader.get_bytes(header.key.data(), key_len)) { return std::nullopt; }
    header.token_count = token_count;
    header.checkpoints.resize(checkpoints);
    for (DiskKVCheckpoint& checkpoint : header.checkpoints) {
        if (!reader.get(checkpoint.frontier) || !reader.get(checkpoint.kind) ||
            !reader.get(checkpoint.backend_frontier) || !reader.get(checkpoint.state_section) ||
            !reader.get(checkpoint.digest)) {
            return std::nullopt;
        }
    }
    header.sections.resize(sections);
    for (DiskKVSectionInfo& section : header.sections) {
        std::uint32_t pad = 0;
        if (!reader.get(section.kind) || !reader.get(pad) || !reader.get(section.bytes) ||
            !reader.get(section.offset) || !reader.get(section.first_block)) {
            return std::nullopt;
        }
    }
    header.block_hashes.resize(blocks);
    for (std::uint64_t& hash : header.block_hashes) {
        if (!reader.get(hash)) { return std::nullopt; }
    }
    const std::size_t hashed = reader.offset();
    std::uint64_t stored     = 0;
    if (!reader.get(stored) || stored != xxh64(all.data(), hashed)) { return std::nullopt; }
    if (header.sections.front().kind != static_cast<std::uint32_t>(DiskKVSectionKind::Tokens) ||
        header.sections.front().bytes != static_cast<std::uint64_t>(token_count) * 4U) {
        return std::nullopt;
    }
    std::uint64_t expected_block = 0;
    for (const DiskKVSectionInfo& section : header.sections) {
        if (section.offset < header.header_bytes || section.offset % kDiskKVAlignment != 0 ||
            section.offset + section.bytes > file_bytes || section.first_block != expected_block) {
            return std::nullopt;
        }
        expected_block += block_count(section.bytes);
    }
    if (expected_block != blocks) { return std::nullopt; }
    std::uint64_t refs = 0;
    if (header.kind == DiskKVRecordKind::Manifest) {
        if (header.sections.size() < 2 ||
            header.sections[1].kind != static_cast<std::uint32_t>(DiskKVSectionKind::BlobRefs) ||
            header.sections[1].bytes % 16U != 0) {
            return std::nullopt;
        }
        refs = header.sections[1].bytes / 16U;
    }
    for (const DiskKVCheckpoint& checkpoint : header.checkpoints) {
        if (checkpoint.frontier == 0 || checkpoint.frontier > token_count ||
            checkpoint.state_section >= refs) {
            return std::nullopt;
        }
    }
    return header;
}

// ---------------------------------------------------------------------------------------------
// Opened records

DiskKVRecordFile::DiskKVRecordFile(DiskKVRecordFile&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)), direct_fd_(std::exchange(other.direct_fd_, -1)),
      header_(std::move(other.header_)) {}

DiskKVRecordFile& DiskKVRecordFile::operator=(DiskKVRecordFile&& other) noexcept {
    if (this != &other) {
        if (fd_ >= 0) { ::close(fd_); }
        if (direct_fd_ >= 0) { ::close(direct_fd_); }
        fd_        = std::exchange(other.fd_, -1);
        direct_fd_ = std::exchange(other.direct_fd_, -1);
        header_    = std::move(other.header_);
    }
    return *this;
}

DiskKVRecordFile::~DiskKVRecordFile() {
    if (fd_ >= 0) { ::close(fd_); }
    if (direct_fd_ >= 0) { ::close(direct_fd_); }
}

std::uint32_t DiskKVRecordFile::section_of(DiskKVSectionKind kind) const noexcept {
    for (std::uint32_t index = 0; index < header_.sections.size(); ++index) {
        if (header_.sections[index].kind == static_cast<std::uint32_t>(kind)) { return index; }
    }
    return static_cast<std::uint32_t>(header_.sections.size());
}

namespace {

bool verify_section(const DiskKVRecordHeader& header, const DiskKVSectionInfo& section,
                    const std::byte* data, std::uint64_t from, std::uint64_t length) {
    // [from, from + length) of the section, from block aligned, length reaching a block end or
    // the section end.
    for (std::uint64_t begin = 0; begin < length; begin += kDiskKVBlockBytes) {
        const std::uint64_t part  = std::min<std::uint64_t>(kDiskKVBlockBytes, length - begin);
        const std::uint64_t block = section.first_block + (from + begin) / kDiskKVBlockBytes;
        if (block >= header.block_hashes.size() ||
            xxh64(data + begin, part) != header.block_hashes[block]) {
            return false;
        }
    }
    return true;
}

} // namespace

bool DiskKVRecordFile::read_prefix(
    std::uint32_t section_index, std::uint64_t bytes, std::span<std::byte> chunk,
    const std::function<void(std::span<const std::byte>, std::uint64_t)>& sink) {
    if (fd_ < 0 || section_index >= header_.sections.size() || chunk.empty() ||
        chunk.size() % kDiskKVBlockBytes != 0) {
        return false;
    }
    const DiskKVSectionInfo& section = header_.sections[section_index];
    if (bytes > section.bytes) { return false; }
    for (std::uint64_t position = 0; position < bytes; position += chunk.size()) {
        const std::uint64_t want = std::min<std::uint64_t>(chunk.size(), bytes - position);
        const std::uint64_t end =
            std::min<std::uint64_t>(align_up(position + want, kDiskKVBlockBytes), section.bytes);
        const std::uint64_t length = end - position;
        if (!pread_all(fd_, chunk.data(), length, section.offset + position)) { return false; }
        if (!verify_section(header_, section, chunk.data(), position, length)) { return false; }
        sink(std::span<const std::byte>(chunk.data(), want), position);
    }
    return true;
}

bool DiskKVRecordFile::read_prefix_pipelined(
    std::uint32_t section_index, std::uint64_t bytes, std::span<std::byte> first,
    std::span<std::byte> second,
    const std::function<void(std::span<const std::byte>, std::uint64_t)>& issue,
    const std::function<void()>& wait) {
    if (fd_ < 0 || section_index >= header_.sections.size() || first.empty() ||
        first.size() != second.size() || first.size() % kDiskKVBlockBytes != 0 ||
        reinterpret_cast<std::uintptr_t>(first.data()) % kDiskKVAlignment != 0 ||
        reinterpret_cast<std::uintptr_t>(second.data()) % kDiskKVAlignment != 0) {
        return false;
    }
    const DiskKVSectionInfo& section = header_.sections[section_index];
    if (bytes > section.bytes) { return false; }
    if (bytes == 0) { return true; }
    const std::uint64_t chunk = first.size();
    const auto load = [&](std::uint64_t position, std::byte* into) -> std::uint64_t {
        const std::uint64_t want = std::min<std::uint64_t>(chunk, bytes - position);
        const std::uint64_t end =
            std::min<std::uint64_t>(align_up(position + want, kDiskKVBlockBytes), section.bytes);
        const std::uint64_t length = end - position;
        bool ok                    = false;
        if (direct_fd_ >= 0) {
            const std::uint64_t padded = align_up(length, kDiskKVAlignment);
            ok = padded <= chunk && pread_all(direct_fd_, into, padded, section.offset + position);
        }
        if (!ok && !pread_all(fd_, into, length, section.offset + position)) { return 0; }
        return length;
    };
    std::byte* buffers[2] = {first.data(), second.data()};
    std::uint64_t loaded  = 0;
    std::thread loader;
    struct Joiner {
        std::thread& thread;
        ~Joiner() {
            if (thread.joinable()) { thread.join(); }
        }
    } joiner{loader};
    const std::uint64_t first_length = load(0, buffers[0]);
    for (std::uint64_t position = 0, index = 0; position < bytes; position += chunk, ++index) {
        if (loader.joinable()) { loader.join(); }
        const std::uint64_t length = position == 0 ? first_length : loaded;
        if (length == 0) { return false; }
        std::byte* data          = buffers[index % 2];
        const std::uint64_t want = std::min<std::uint64_t>(chunk, bytes - position);
        const std::uint64_t next = position + chunk;
        if (next < bytes) {
            loader = std::thread([&, next, index] { loaded = load(next, buffers[(index + 1) % 2]); });
        }
        issue(std::span<const std::byte>(data, want), position);
        const bool verified = verify_section(header_, section, data, position, length);
        wait();
        if (!verified) { return false; }
    }
    return true;
}

std::optional<std::uint64_t> DiskKVRecordFile::read_region(std::uint32_t first_section,
                                                          std::span<std::byte> into) {
    if (fd_ < 0 || first_section >= header_.sections.size() ||
        reinterpret_cast<std::uintptr_t>(into.data()) % kDiskKVAlignment != 0) {
        return std::nullopt;
    }
    const std::uint64_t start = header_.sections[first_section].offset;
    std::uint64_t end         = start;
    for (std::size_t index = first_section; index < header_.sections.size(); ++index) {
        end = std::max(end, header_.sections[index].offset + header_.sections[index].bytes);
    }
    const std::uint64_t length = end - start;
    const std::uint64_t padded = align_up(length, kDiskKVAlignment);
    if (padded > into.size()) { return std::nullopt; }
    bool ok = direct_fd_ >= 0 && pread_all(direct_fd_, into.data(), padded, start);
    if (!ok && !pread_all(fd_, into.data(), length, start)) { return std::nullopt; }
    for (std::size_t index = first_section; index < header_.sections.size(); ++index) {
        const DiskKVSectionInfo& section = header_.sections[index];
        if (!verify_section(header_, section, into.data() + (section.offset - start), 0,
                            section.bytes)) {
            return std::nullopt;
        }
    }
    return start;
}

// ---------------------------------------------------------------------------------------------
// Writer thread jobs

struct DiskKVCache::Job {
    std::uint64_t id = 0;
    int fd           = -1;
    std::filesystem::path temp;
    std::filesystem::path final_path;
    DiskKVRecordHeader header;
    std::uint64_t session = 0;
    std::vector<std::int32_t> tokens;
    std::vector<DiskKVBlobId> refs;
    std::uint64_t reserved = 0;
    bool started           = false;
    bool failed            = false;
    // Payload cursor (sections after the ones the cache writes itself).
    std::size_t section          = 1;
    std::uint64_t section_offset = 0;
    Xxh64Stream block_hash;
    std::uint64_t block_fill = 0;
    std::chrono::steady_clock::time_point first_write;
    std::uint64_t flushed = 0; // file offset below which the payload is on disk and dropped
};

std::filesystem::path DiskKVCache::manifest_path(std::uint64_t id) const {
    return manifests_dir_ / (hex64(id) + ".nidkv");
}

std::filesystem::path DiskKVCache::blob_path(DiskKVBlobId id) const {
    return blobs_dir_ / (id.hex() + ".nkvb");
}

std::filesystem::path DiskKVCache::temp_path(const std::filesystem::path& final_path) const {
    return std::filesystem::path(final_path.string() + ".tmp." +
                                 std::to_string(static_cast<long>(::getpid())));
}

DiskKVCache::DiskKVCache(Options options) : options_(std::move(options)) {
    if (options_.directory.empty()) { throw std::invalid_argument("disk KV cache needs a directory"); }
    if (options_.staging_buffers == 0 || options_.staging_bytes == 0 ||
        options_.staging_bytes % kDiskKVBlockBytes != 0) {
        throw std::invalid_argument("disk KV staging buffers must be whole 4 MiB blocks");
    }
    if (!options_.allocate || !options_.release) {
        options_.allocate = [](std::size_t bytes) { return std::aligned_alloc(kDiskKVAlignment, bytes); };
        options_.release  = [](void* data) { std::free(data); };
    }
    key_hash_             = xxh64(options_.key.data(), options_.key.size());
    stats_.capacity_bytes = options_.capacity_bytes;
    manifests_dir_        = options_.directory / "m";
    blobs_dir_            = options_.directory / "b";
    std::error_code error;
    std::filesystem::create_directories(manifests_dir_, error);
    std::filesystem::create_directories(blobs_dir_, error);
    if (!std::filesystem::is_directory(manifests_dir_) || !std::filesystem::is_directory(blobs_dir_)) {
        throw std::runtime_error("disk KV cache directory " + options_.directory.string() +
                                 " cannot be created");
    }
    // The manifests hold the conversations' token IDs, the blobs their KV: only the server's
    // user may read them, whatever the umask or the place of --disk-cache-dir.
    for (const auto* dir : {&manifests_dir_, &blobs_dir_}) {
        std::filesystem::permissions(*dir, std::filesystem::perms::owner_all,
                                     std::filesystem::perm_options::replace, error);
    }
    // One directory = one server: the accounting of the cap and of the files is per process.
    const std::filesystem::path lock = options_.directory / ".lock";
    lock_fd_ = ::open(lock.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (lock_fd_ < 0) {
        throw std::runtime_error("disk KV cache: cannot open " + lock.string() + ": " +
                                 std::strerror(errno));
    }
    if (::flock(lock_fd_, LOCK_EX | LOCK_NB) != 0) {
        char holder[32] = {};
        if (::pread(lock_fd_, holder, sizeof(holder) - 1, 0) < 0) { holder[0] = 0; }
        ::close(lock_fd_);
        lock_fd_ = -1;
        std::string pid(holder);
        while (!pid.empty() && (pid.back() == '\n' || pid.back() == ' ')) { pid.pop_back(); }
        throw std::runtime_error("disk cache directory " + options_.directory.string() +
                                 " is in use by another server (PID " +
                                 (pid.empty() ? std::string("unknown") : pid) +
                                 "): give each server its own --disk-cache-dir");
    }
    const std::string pid = std::to_string(static_cast<long>(::getpid())) + "\n";
    if (::ftruncate(lock_fd_, 0) != 0 || ::pwrite(lock_fd_, pid.data(), pid.size(), 0) < 0) {
        // The PID is informative only.
    }
    // Free-space margin: max(10 GiB, 5 % of the file system) unless given.
    if (options_.min_free_bytes == 0) {
        std::uint64_t total = 0;
        struct statvfs vfs {};
        if (!options_.free_bytes && ::statvfs(blobs_dir_.c_str(), &vfs) == 0) {
            total = static_cast<std::uint64_t>(vfs.f_blocks) * vfs.f_frsize;
        }
        options_.min_free_bytes = std::max<std::uint64_t>(10ULL * kGiB, total / 20U);
    }
    stats_.min_free_bytes = options_.min_free_bytes;
    std::random_device random;
    next_id_ = (static_cast<std::uint64_t>(random()) << 32U) ^ static_cast<std::uint64_t>(random()) ^
               static_cast<std::uint64_t>(now_ns());
    scan();
    buffers_.reserve(options_.staging_buffers);
    for (std::size_t index = 0; index < options_.staging_buffers; ++index) {
        void* data = options_.allocate(options_.staging_bytes);
        if (data == nullptr) {
            for (std::byte* buffer : buffers_) { options_.release(buffer); }
            ::close(lock_fd_);
            throw std::bad_alloc();
        }
        buffers_.push_back(static_cast<std::byte*>(data));
        free_buffers_.push_back(static_cast<std::uint32_t>(index));
    }
    thread_ = std::thread([this] { writer_loop(); });
}

DiskKVCache::~DiskKVCache() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    queue_cv_.notify_all();
    if (thread_.joinable()) { thread_.join(); }
    for (auto& job : jobs_) { abort_job(*job); }
    jobs_.clear();
    for (std::byte* buffer : buffers_) { options_.release(buffer); }
    if (lock_fd_ >= 0) { ::close(lock_fd_); }
}

std::optional<std::string> DiskKVCache::directory_holder(const std::filesystem::path& directory) {
    const std::filesystem::path lock = directory / ".lock";
    const int fd = ::open(lock.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) { return std::nullopt; }
    std::optional<std::string> holder;
    if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
        char text[32] = {};
        if (::pread(fd, text, sizeof(text) - 1, 0) < 0) { text[0] = 0; }
        std::string pid(text);
        while (!pid.empty() && (pid.back() == '\n' || pid.back() == ' ')) { pid.pop_back(); }
        holder = pid.empty() ? std::string("unknown") : pid;
    }
    ::close(fd); // releases the probe's lock, if it took it
    return holder;
}

std::uint64_t DiskKVCache::free_bytes() const {
    std::uint64_t bytes = 0;
    if (options_.free_bytes) {
        bytes = options_.free_bytes();
    } else {
        struct statvfs vfs {};
        if (::statvfs(blobs_dir_.c_str(), &vfs) == 0) {
            bytes = static_cast<std::uint64_t>(vfs.f_bavail) * vfs.f_frsize;
        }
    }
    stats_.fs_free_bytes = bytes;
    return bytes;
}

void DiskKVCache::scan() {
    std::error_code error;
    // Records of format 1 (one file per conversation in the directory itself): not readable.
    for (const auto& item : std::filesystem::directory_iterator(options_.directory, error)) {
        const std::string name = item.path().filename().string();
        if (remove_temporary_if_dead(item.path())) {
            ++stats_.removed_temp;
        } else if (item.path().extension() == ".nidkv" &&
                   std::filesystem::remove(item.path(), error)) {
            ++stats_.removed_broken;
        }
    }
    // Blobs: the name is the id, the size is enough to account for them; read headers on use.
    for (const auto& item : std::filesystem::directory_iterator(blobs_dir_, error)) {
        const std::filesystem::path path = item.path();
        if (remove_temporary_if_dead(path)) {
            ++stats_.removed_temp;
            continue;
        }
        const std::string stem = path.stem().string();
        const auto a           = stem.size() == 32 ? parse_hex64(stem.substr(0, 16)) : std::nullopt;
        const auto b           = stem.size() == 32 ? parse_hex64(stem.substr(16)) : std::nullopt;
        struct stat st {};
        if (path.extension() != ".nkvb" || !a || !b || ::stat(path.c_str(), &st) != 0) {
            if (std::filesystem::remove(path, error)) { ++stats_.removed_broken; }
            continue;
        }
        blobs_[DiskKVBlobId{*a, *b}] =
            Blob{.bytes = static_cast<std::uint64_t>(st.st_size), .refs = 0, .last_use_ns = mtime_ns(st)};
        used_bytes_ += static_cast<std::uint64_t>(st.st_size);
    }
    for (const auto& item : std::filesystem::directory_iterator(manifests_dir_, error)) {
        const std::filesystem::path path = item.path();
        if (remove_temporary_if_dead(path)) {
            ++stats_.removed_temp;
            continue;
        }
        if (path.extension() != ".nidkv") { continue; }
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) { continue; }
        struct stat st {};
        std::optional<DiskKVRecordHeader> header;
        if (::fstat(fd, &st) == 0) {
            header = read_header(fd, static_cast<std::uint64_t>(st.st_size));
        }
        std::vector<DiskKVBlobId> refs;
        bool ok = header && header->kind == DiskKVRecordKind::Manifest &&
                  hex64(header->id) + ".nidkv" == path.filename().string();
        if (ok) {
            const DiskKVSectionInfo& section = header->sections[1];
            std::vector<std::byte> bytes(section.bytes);
            ok = pread_all(fd, bytes.data(), bytes.size(), section.offset) &&
                 verify_section(*header, section, bytes.data(), 0, section.bytes);
            refs.resize(section.bytes / 16U);
            if (ok && !refs.empty()) { std::memcpy(refs.data(), bytes.data(), bytes.size()); }
            for (const DiskKVBlobId& id : refs) { ok = ok && blobs_.count(id) != 0; }
        }
        ::close(fd);
        if (!ok) {
            if (std::filesystem::remove(path, error)) { ++stats_.removed_broken; }
            continue;
        }
        for (const DiskKVBlobId& id : refs) { ++blobs_[id].refs; }
        Entry entry;
        entry.id          = header->id;
        entry.path        = path;
        entry.bytes       = header->file_bytes;
        entry.key_ok      = header->key == options_.key;
        entry.last_use_ns = mtime_ns(st);
        entry.token_count = header->token_count;
        entry.full_digest = header->full_digest;
        entry.checkpoints = std::move(header->checkpoints);
        entry.refs        = std::move(refs);
        used_bytes_ += entry.bytes;
        entries_.push_back(std::move(entry));
    }
    // Blobs no manifest references: leftovers of interrupted saves.
    for (auto it = blobs_.begin(); it != blobs_.end();) {
        if (it->second.refs == 0) {
            std::filesystem::remove(blob_path(it->first), error);
            used_bytes_ -= it->second.bytes;
            ++stats_.removed_orphans;
            it = blobs_.erase(it);
        } else {
            ++it;
        }
    }
    std::lock_guard lock(mutex_);
    (void)make_room(0);
}

void DiskKVCache::release_blob(DiskKVBlobId id) {
    const auto it = blobs_.find(id);
    if (it == blobs_.end()) { return; }
    if (it->second.refs != 0) { --it->second.refs; }
}

void DiskKVCache::remove_manifest(std::vector<Entry>::iterator it) {
    std::error_code error;
    std::filesystem::remove(it->path, error);
    used_bytes_ -= it->bytes;
    // Its blobs nobody else uses go with it.
    for (const DiskKVBlobId& id : it->refs) {
        const auto blob = blobs_.find(id);
        if (blob == blobs_.end()) { continue; }
        if (blob->second.refs != 0) { --blob->second.refs; }
        if (blob->second.refs == 0) {
            std::filesystem::remove(blob_path(id), error);
            used_bytes_ -= blob->second.bytes;
            ++stats_.evicted_blobs;
            blobs_.erase(blob);
        }
    }
    entries_.erase(it);
}

// Caller holds mutex_.
bool DiskKVCache::make_room(std::uint64_t bytes) {
    for (;;) {
        const bool over_cap = used_bytes_ + bytes > options_.capacity_bytes;
        const bool low_space =
            free_bytes() < bytes + reserved_bytes_ + options_.min_free_bytes;
        if (!over_cap && !low_space) { return true; }
        // Orphan blobs first (oldest), then the least recently used manifest.
        auto orphan = blobs_.end();
        for (auto it = blobs_.begin(); it != blobs_.end(); ++it) {
            if (it->second.refs == 0 &&
                (orphan == blobs_.end() || it->second.last_use_ns < orphan->second.last_use_ns)) {
                orphan = it;
            }
        }
        if (orphan != blobs_.end()) {
            std::error_code error;
            std::filesystem::remove(blob_path(orphan->first), error);
            used_bytes_ -= orphan->second.bytes;
            ++stats_.evicted_blobs;
            blobs_.erase(orphan);
            continue;
        }
        auto victim = entries_.end();
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (victim == entries_.end() || it->last_use_ns < victim->last_use_ns) { victim = it; }
        }
        if (victim == entries_.end()) { return false; }
        remove_manifest(victim);
        ++stats_.evicted;
    }
}

DiskKVCacheStats DiskKVCache::stats() const {
    std::lock_guard lock(mutex_);
    DiskKVCacheStats out = stats_;
    out.records          = 0;
    out.foreign_records  = 0;
    out.blobs            = blobs_.size();
    out.orphan_blobs     = 0;
    out.bytes            = 0;
    out.blob_bytes       = 0;
    for (const Entry& entry : entries_) {
        (entry.key_ok ? out.records : out.foreign_records) += 1;
        out.bytes += entry.bytes;
    }
    for (const auto& [id, blob] : blobs_) {
        out.blob_bytes += blob.bytes;
        if (blob.refs == 0) { ++out.orphan_blobs; }
    }
    out.bytes += out.blob_bytes;
    out.fs_free_bytes = free_bytes();
    return out;
}

std::uint64_t DiskKVCache::record_file_bytes(const DiskKVRecordSpec& spec, std::size_t key_bytes) {
    std::vector<DiskKVSectionInfo> sections;
    sections.push_back({.kind  = static_cast<std::uint32_t>(DiskKVSectionKind::Tokens),
                        .bytes = static_cast<std::uint64_t>(spec.tokens.size()) * 4U});
    if (spec.kind == DiskKVRecordKind::Manifest) {
        sections.push_back({.kind  = static_cast<std::uint32_t>(DiskKVSectionKind::BlobRefs),
                            .bytes = static_cast<std::uint64_t>(spec.blob_refs.size()) * 16U});
    }
    sections.insert(sections.end(), spec.sections.begin(), spec.sections.end());
    const std::uint64_t header = layout_sections(sections, spec.checkpoints.size(), key_bytes);
    return sections_end(sections, header);
}

std::uint64_t DiskKVCache::begin_session(std::span<const DiskKVBlobId> ids,
                                         std::vector<bool>& present) {
    std::lock_guard lock(mutex_);
    const std::uint64_t session = ++next_session_;
    std::vector<DiskKVBlobId>& pins = sessions_[session];
    present.assign(ids.size(), false);
    for (std::size_t index = 0; index < ids.size(); ++index) {
        const auto it = blobs_.find(ids[index]);
        if (it == blobs_.end()) { continue; }
        // The same id twice in one save: pinned once.
        if (std::find(pins.begin(), pins.end(), ids[index]) == pins.end()) {
            ++it->second.refs;
            it->second.last_use_ns = now_ns();
            pins.push_back(ids[index]);
            ++stats_.reused_blobs;
        }
        present[index] = true;
    }
    return session;
}

void DiskKVCache::abort_session(std::uint64_t session) noexcept {
    try {
        std::lock_guard lock(mutex_);
        queue_.push_back(Item{.kind = Item::Kind::Unpin, .id = session, .buffer = {}, .bytes = 0});
    } catch (...) {}
    queue_cv_.notify_one();
}

std::optional<DiskKVCache::Writer> DiskKVCache::begin_record(DiskKVRecordSpec spec) {
    const bool manifest = spec.kind == DiskKVRecordKind::Manifest;
    if (manifest && (spec.tokens.empty() || spec.checkpoints.empty())) { return std::nullopt; }
    auto job                   = std::make_unique<Job>();
    DiskKVRecordHeader& header = job->header;
    header.kind                = spec.kind;
    header.blob                = spec.blob;
    header.sections.push_back({.kind  = static_cast<std::uint32_t>(DiskKVSectionKind::Tokens),
                               .bytes = static_cast<std::uint64_t>(spec.tokens.size()) * 4U});
    if (manifest) {
        // One reference per blob.
        std::vector<DiskKVBlobId> unique;
        for (const DiskKVBlobId& id : spec.blob_refs) {
            if (std::find(unique.begin(), unique.end(), id) == unique.end()) { unique.push_back(id); }
        }
        for (DiskKVCheckpoint& checkpoint : spec.checkpoints) {
            if (checkpoint.state_section >= spec.blob_refs.size()) { return std::nullopt; }
            const DiskKVBlobId state = spec.blob_refs[checkpoint.state_section];
            checkpoint.state_section = static_cast<std::uint32_t>(
                std::find(unique.begin(), unique.end(), state) - unique.begin());
        }
        spec.blob_refs = std::move(unique);
        header.sections.push_back({.kind  = static_cast<std::uint32_t>(DiskKVSectionKind::BlobRefs),
                                   .bytes = static_cast<std::uint64_t>(spec.blob_refs.size()) * 16U});
        job->section = 2;
    }
    header.sections.insert(header.sections.end(), spec.sections.begin(), spec.sections.end());
    for (DiskKVCheckpoint& checkpoint : spec.checkpoints) {
        if (checkpoint.frontier == 0 || checkpoint.frontier > spec.tokens.size()) {
            return std::nullopt;
        }
        checkpoint.digest = xxh64(spec.tokens.data(), checkpoint.frontier * sizeof(std::int32_t));
    }
    header.header_bytes = layout_sections(header.sections, spec.checkpoints.size(), options_.key.size());
    header.file_bytes   = sections_end(header.sections, header.header_bytes);
    header.key          = options_.key;
    header.key_hash     = key_hash_;
    header.token_count  = static_cast<std::uint32_t>(spec.tokens.size());
    header.created_ns   = static_cast<std::uint64_t>(now_ns());
    header.full_digest  = xxh64(spec.tokens.data(), spec.tokens.size() * sizeof(std::int32_t));
    header.meta         = spec.meta;
    header.checkpoints  = spec.checkpoints;
    std::uint64_t blocks = 0;
    for (const DiskKVSectionInfo& section : header.sections) { blocks += block_count(section.bytes); }
    header.block_hashes.reserve(blocks);
    job->tokens  = std::move(spec.tokens);
    job->refs    = std::move(spec.blob_refs);
    job->session = spec.session;

    std::uint64_t expected = 0;
    for (std::size_t index = job->section; index < header.sections.size(); ++index) {
        expected += header.sections[index].bytes;
    }
    {
        std::lock_guard lock(mutex_);
        // Never evict for a record that cannot fit even in an empty cache.
        const bool hopeless = header.file_bytes > options_.capacity_bytes ||
                              free_bytes() + used_bytes_ - reserved_bytes_ <
                                  header.file_bytes + reserved_bytes_ + options_.min_free_bytes;
        if (stopping_ || hopeless || !make_room(header.file_bytes)) {
            ++stats_.space_refusals;
            return std::nullopt;
        }
        used_bytes_ += header.file_bytes;
        reserved_bytes_ += header.file_bytes;
        job->reserved = header.file_bytes;
        job->id       = ++next_id_;
        header.id     = manifest ? job->id : spec.blob.a;
    }
    job->final_path = manifest ? manifest_path(job->id) : blob_path(spec.blob);
    job->temp       = temp_path(job->final_path);
    job->fd = ::open(job->temp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (job->fd < 0) {
        std::lock_guard lock(mutex_);
        used_bytes_ -= job->reserved;
        reserved_bytes_ -= job->reserved;
        ++stats_.write_failures;
        return std::nullopt;
    }
    Writer writer;
    writer.cache_    = this;
    writer.id_       = job->id;
    writer.expected_ = expected;
    {
        std::lock_guard lock(mutex_);
        ++in_flight_;
        jobs_.push_back(std::move(job));
    }
    return writer;
}

// ---------------------------------------------------------------------------------------------
// Writer handle

DiskKVCache::Writer::Writer(Writer&& other) noexcept
    : cache_(std::exchange(other.cache_, nullptr)), id_(other.id_), submitted_(other.submitted_),
      expected_(other.expected_) {}

DiskKVCache::Writer& DiskKVCache::Writer::operator=(Writer&& other) noexcept {
    if (this != &other) {
        abort();
        cache_     = std::exchange(other.cache_, nullptr);
        id_        = other.id_;
        submitted_ = other.submitted_;
        expected_  = other.expected_;
    }
    return *this;
}

DiskKVCache::Writer::~Writer() { abort(); }

std::optional<DiskKVBuffer> DiskKVCache::Writer::acquire(const std::function<bool()>& abort) {
    if (cache_ == nullptr) { return std::nullopt; }
    std::unique_lock lock(cache_->mutex_);
    for (;;) {
        if (!cache_->free_buffers_.empty()) {
            const std::uint32_t index = cache_->free_buffers_.back();
            cache_->free_buffers_.pop_back();
            return DiskKVBuffer{.data  = cache_->buffers_[index],
                                .bytes = cache_->options_.staging_bytes,
                                .index = index};
        }
        if (cache_->stopping_) { return std::nullopt; }
        cache_->buffers_cv_.wait_for(lock, std::chrono::milliseconds(1));
        if (abort) {
            lock.unlock();
            const bool stop = abort();
            lock.lock();
            if (stop) { return std::nullopt; }
        }
    }
}

void DiskKVCache::Writer::submit(DiskKVBuffer buffer, std::size_t bytes) {
    if (cache_ == nullptr || bytes > buffer.bytes) {
        throw std::logic_error("disk KV buffer submitted to no record or overfilled");
    }
    submitted_ += bytes;
    {
        std::lock_guard lock(cache_->mutex_);
        cache_->queue_.push_back(
            Item{.kind = Item::Kind::Data, .id = id_, .buffer = buffer, .bytes = bytes});
    }
    cache_->queue_cv_.notify_one();
}

void DiskKVCache::Writer::give_back(DiskKVBuffer buffer) {
    if (cache_ == nullptr) { return; }
    cache_->return_buffer(buffer.index);
}

bool DiskKVCache::Writer::commit() {
    if (cache_ == nullptr) { return false; }
    if (submitted_ != expected_) {
        abort();
        return false;
    }
    {
        std::lock_guard lock(cache_->mutex_);
        cache_->queue_.push_back(Item{.kind = Item::Kind::Commit, .id = id_, .buffer = {}, .bytes = 0});
    }
    cache_->queue_cv_.notify_one();
    cache_ = nullptr;
    return true;
}

void DiskKVCache::Writer::abort() noexcept {
    if (cache_ == nullptr) { return; }
    try {
        std::lock_guard lock(cache_->mutex_);
        cache_->queue_.push_back(Item{.kind = Item::Kind::Abort, .id = id_, .buffer = {}, .bytes = 0});
    } catch (...) {}
    cache_->queue_cv_.notify_one();
    cache_ = nullptr;
}

void DiskKVCache::return_buffer(std::uint32_t index) noexcept {
    {
        std::lock_guard lock(mutex_);
        free_buffers_.push_back(index);
    }
    buffers_cv_.notify_all();
}

// ---------------------------------------------------------------------------------------------
// Writer thread

void DiskKVCache::writer_loop() {
    for (;;) {
        Item item;
        Job* job = nullptr;
        {
            std::unique_lock lock(mutex_);
            queue_cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
            if (queue_.empty()) {
                if (stopping_) { return; }
                continue;
            }
            item = queue_.front();
            queue_.pop_front();
            if (item.kind == Item::Kind::Unpin) {
                const auto session = sessions_.find(item.id);
                if (session != sessions_.end()) {
                    // Blobs of an interrupted save stay as orphans: the next attempt reuses them.
                    for (const DiskKVBlobId& id : session->second) { release_blob(id); }
                    sessions_.erase(session);
                }
                lock.unlock();
                idle_cv_.notify_all();
                continue;
            }
            for (auto& candidate : jobs_) {
                if (candidate->id == item.id) { job = candidate.get(); }
            }
        }
        if (job == nullptr) {
            if (item.kind == Item::Kind::Data) { return_buffer(item.buffer.index); }
            idle_cv_.notify_all();
            continue;
        }
        if (!job->started && !job->failed && item.kind != Item::Kind::Abort) {
            job->started     = true;
            job->first_write = std::chrono::steady_clock::now();
            const DiskKVSectionInfo& tokens = job->header.sections.front();
            bool ok = pwrite_all(job->fd, job->tokens.data(), tokens.bytes, tokens.offset);
            for (std::uint64_t begin = 0; ok && begin < tokens.bytes; begin += kDiskKVBlockBytes) {
                const std::uint64_t length =
                    std::min<std::uint64_t>(kDiskKVBlockBytes, tokens.bytes - begin);
                job->header.block_hashes.push_back(
                    xxh64(reinterpret_cast<const std::byte*>(job->tokens.data()) + begin, length));
            }
            if (ok && job->header.kind == DiskKVRecordKind::Manifest) {
                const DiskKVSectionInfo& refs = job->header.sections[1];
                ok = pwrite_all(job->fd, job->refs.data(), refs.bytes, refs.offset);
                for (std::uint64_t begin = 0; ok && begin < refs.bytes; begin += kDiskKVBlockBytes) {
                    const std::uint64_t length =
                        std::min<std::uint64_t>(kDiskKVBlockBytes, refs.bytes - begin);
                    job->header.block_hashes.push_back(
                        xxh64(reinterpret_cast<const std::byte*>(job->refs.data()) + begin, length));
                }
            }
            job->failed = !ok;
        }
        if (item.kind == Item::Kind::Data) {
            std::size_t consumed = 0;
            while (!job->failed && consumed < item.bytes) {
                while (job->section < job->header.sections.size() &&
                       job->header.sections[job->section].bytes == 0) {
                    ++job->section;
                }
                if (job->section >= job->header.sections.size()) {
                    job->failed = true;
                    break;
                }
                const DiskKVSectionInfo& section = job->header.sections[job->section];
                const std::uint64_t room         = section.bytes - job->section_offset;
                const std::uint64_t take = std::min<std::uint64_t>(room, item.bytes - consumed);
                const std::byte* data    = item.buffer.data + consumed;
                if (options_.before_write) { options_.before_write(); }
                const std::uint64_t at = section.offset + job->section_offset;
                if (!pwrite_all(job->fd, data, take, at)) {
                    job->failed = true;
                    break;
                }
                // Bounded dirty pages: start the writeback of what was just written and wait for
                // (then drop from the page cache) what lies more than kDiskKVDirtyWindow behind.
                // Without it a multi-GB record fills the page cache with dirty pages and the
                // kernel throttles every writer of the machine, the server's own log included
                // (measured: a short request stalled 4.6 s on its log lines).
                (void)::sync_file_range(job->fd, static_cast<off_t>(at), static_cast<off_t>(take),
                                        SYNC_FILE_RANGE_WRITE);
                if (at + take > job->flushed + kDiskKVDirtyWindow) {
                    const std::uint64_t until = at + take - kDiskKVDirtyWindow;
                    (void)::sync_file_range(job->fd, static_cast<off_t>(job->flushed),
                                            static_cast<off_t>(until - job->flushed),
                                            SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE |
                                                SYNC_FILE_RANGE_WAIT_AFTER);
                    (void)::posix_fadvise(job->fd, static_cast<off_t>(job->flushed),
                                          static_cast<off_t>(until - job->flushed),
                                          POSIX_FADV_DONTNEED);
                    job->flushed = until;
                }
                std::uint64_t hashed = 0;
                while (hashed < take) {
                    const std::uint64_t part =
                        std::min<std::uint64_t>(kDiskKVBlockBytes - job->block_fill, take - hashed);
                    job->block_hash.update(data + hashed, part);
                    job->block_fill += part;
                    hashed += part;
                    if (job->block_fill == kDiskKVBlockBytes) {
                        job->header.block_hashes.push_back(job->block_hash.digest());
                        job->block_hash.reset();
                        job->block_fill = 0;
                    }
                }
                consumed += take;
                job->section_offset += take;
                if (job->section_offset == section.bytes) {
                    if (job->block_fill != 0) {
                        job->header.block_hashes.push_back(job->block_hash.digest());
                        job->block_hash.reset();
                        job->block_fill = 0;
                    }
                    ++job->section;
                    job->section_offset = 0;
                }
            }
            return_buffer(item.buffer.index);
            continue;
        }
        if (item.kind == Item::Kind::Commit && !job->failed) {
            finish_job(*job);
        } else {
            abort_job(*job);
            std::lock_guard lock(mutex_);
            if (item.kind == Item::Kind::Abort) {
                ++stats_.aborted;
            } else {
                ++stats_.write_failures;
            }
            // A manifest that never landed hands its save's pins back.
            if (job->header.kind == DiskKVRecordKind::Manifest && item.kind == Item::Kind::Commit) {
                const auto session = sessions_.find(job->session);
                if (session != sessions_.end()) {
                    for (const DiskKVBlobId& id : session->second) { release_blob(id); }
                    sessions_.erase(session);
                }
            }
        }
        {
            std::lock_guard lock(mutex_);
            jobs_.erase(std::remove_if(jobs_.begin(), jobs_.end(),
                                       [&](const auto& candidate) { return candidate.get() == job; }),
                        jobs_.end());
            --in_flight_;
        }
        idle_cv_.notify_all();
    }
}

void DiskKVCache::abort_job(Job& job) noexcept {
    if (job.fd >= 0) {
        ::close(job.fd);
        job.fd = -1;
        ::unlink(job.temp.c_str());
    }
    std::lock_guard lock(mutex_);
    used_bytes_ -= job.reserved;
    reserved_bytes_ -= job.reserved;
    job.reserved = 0;
}

void DiskKVCache::finish_job(Job& job) {
    DiskKVRecordHeader& header = job.header;
    const bool manifest        = header.kind == DiskKVRecordKind::Manifest;
    while (job.section < header.sections.size() && header.sections[job.section].bytes == 0) {
        ++job.section;
    }
    std::uint64_t blocks = 0;
    for (const DiskKVSectionInfo& section : header.sections) { blocks += block_count(section.bytes); }
    bool ok = job.section == header.sections.size() && header.block_hashes.size() == blocks;
    if (ok && manifest) {
        // Every blob of the manifest must be pinned by its save (on disk, not evictable).
        std::lock_guard lock(mutex_);
        const auto session = sessions_.find(job.session);
        for (const DiskKVBlobId& id : job.refs) {
            ok = ok && session != sessions_.end() &&
                 std::find(session->second.begin(), session->second.end(), id) !=
                     session->second.end() &&
                 blobs_.count(id) != 0;
        }
    }
    if (ok) {
        const std::vector<std::byte> bytes = encode_header(header);
        ok = pwrite_all(job.fd, bytes.data(), bytes.size(), 0) &&
             ::ftruncate(job.fd, static_cast<off_t>(header.file_bytes)) == 0 &&
             ::fdatasync(job.fd) == 0;
    }
    if (ok) {
        (void)::posix_fadvise(job.fd, 0, 0, POSIX_FADV_DONTNEED);
        ::close(job.fd);
        job.fd = -1;
        // A manifest goes in only after its blobs' names are durable.
        if (manifest) { fsync_directory(blobs_dir_); }
        ok = ::rename(job.temp.c_str(), job.final_path.c_str()) == 0;
        if (!ok) { ::unlink(job.temp.c_str()); }
    }
    if (!ok) {
        abort_job(job);
        std::lock_guard lock(mutex_);
        ++stats_.write_failures;
        if (manifest) {
            const auto session = sessions_.find(job.session);
            if (session != sessions_.end()) {
                for (const DiskKVBlobId& id : session->second) { release_blob(id); }
                sessions_.erase(session);
            }
        }
        return;
    }
    if (manifest) { fsync_directory(manifests_dir_); }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - job.first_write).count();
    if (manifest || header.file_bytes >= (64ULL << 20U)) {
        std::fprintf(stderr,
                     "[disk-kv] writer: %s %s on disk, %.0f MiB in %.0f ms (%.1f GB/s)\n",
                     manifest ? "manifest" : "blob",
                     manifest ? hex64(header.id).c_str() : header.blob.hex().c_str(),
                     static_cast<double>(header.file_bytes) / (1024.0 * 1024.0), seconds * 1000.0,
                     seconds > 0.0 ? static_cast<double>(header.file_bytes) / seconds / 1e9 : 0.0);
    }
    std::lock_guard lock(mutex_);
    reserved_bytes_ -= job.reserved;
    ++stats_.written;
    stats_.written_bytes += header.file_bytes;
    if (!manifest) {
        ++stats_.written_blobs;
        const auto existing = blobs_.find(header.blob);
        if (existing != blobs_.end()) {
            // Written twice (an earlier interrupted save had it in flight): one copy counted.
            used_bytes_ -= job.reserved;
            ++existing->second.refs;
            existing->second.last_use_ns = now_ns();
        } else {
            blobs_[header.blob] = Blob{.bytes = header.file_bytes, .refs = 1, .last_use_ns = now_ns()};
        }
        job.reserved       = 0;
        const auto session = sessions_.find(job.session);
        if (session != sessions_.end()) {
            session->second.push_back(header.blob);
        } else {
            release_blob(header.blob); // the save is gone: an orphan
        }
        return;
    }
    job.reserved = 0;
    Entry entry;
    entry.id          = header.id;
    entry.path        = job.final_path;
    entry.bytes       = header.file_bytes;
    entry.key_ok      = true;
    entry.last_use_ns = now_ns();
    entry.token_count = header.token_count;
    entry.full_digest = header.full_digest;
    entry.checkpoints = header.checkpoints;
    entry.refs        = job.refs;
    // The save's pins become the manifest's references; pins it does not use are released.
    const auto session = sessions_.find(job.session);
    if (session != sessions_.end()) {
        std::vector<DiskKVBlobId> pins = std::move(session->second);
        sessions_.erase(session);
        for (const DiskKVBlobId& id : job.refs) {
            const auto pin = std::find(pins.begin(), pins.end(), id);
            if (pin != pins.end()) { pins.erase(pin); }
        }
        for (const DiskKVBlobId& id : pins) { release_blob(id); }
    }
    // Earlier saves of the same conversation: their tokens are a prefix of these.
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (it->key_ok && it->token_count <= header.token_count &&
            it->full_digest ==
                xxh64(job.tokens.data(), static_cast<std::size_t>(it->token_count) * 4U)) {
            const auto index = it - entries_.begin();
            remove_manifest(it);
            ++stats_.superseded;
            it = entries_.begin() + index;
        } else {
            ++it;
        }
    }
    entries_.push_back(std::move(entry));
}

bool DiskKVCache::drain(std::chrono::steady_clock::time_point deadline) {
    std::unique_lock lock(mutex_);
    const bool drained =
        idle_cv_.wait_until(lock, deadline, [&] { return in_flight_ == 0 && queue_.empty(); });
    if (!drained) {
        // Abort what is left: queued data is dropped, jobs are removed with their temporaries;
        // the saves' pins are still released in order.
        std::deque<Item> kept;
        for (Item& item : queue_) {
            if (item.kind == Item::Kind::Data) {
                free_buffers_.push_back(item.buffer.index);
            } else if (item.kind == Item::Kind::Unpin) {
                kept.push_back(item);
            }
        }
        queue_.clear();
        for (auto& job : jobs_) {
            queue_.push_back(Item{.kind = Item::Kind::Abort, .id = job->id, .buffer = {}, .bytes = 0});
        }
        for (Item& item : kept) { queue_.push_back(item); }
        lock.unlock();
        queue_cv_.notify_all();
        lock.lock();
        idle_cv_.wait_for(lock, std::chrono::seconds(5),
                          [&] { return in_flight_ == 0 && queue_.empty(); });
    }
    return drained;
}

// ---------------------------------------------------------------------------------------------
// Reading

std::optional<DiskKVRecordFile> DiskKVCache::open_blob(DiskKVBlobId id) const {
    const std::filesystem::path path = blob_path(id);
    DiskKVRecordFile file;
    file.fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (file.fd_ < 0) { return std::nullopt; }
    struct stat st {};
    if (::fstat(file.fd_, &st) != 0) { return std::nullopt; }
    std::optional<DiskKVRecordHeader> header =
        read_header(file.fd_, static_cast<std::uint64_t>(st.st_size));
    if (!header || header->kind != DiskKVRecordKind::Blob || !(header->blob == id) ||
        header->key != options_.key) {
        return std::nullopt;
    }
    file.header_    = std::move(*header);
    file.direct_fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    return file;
}

void DiskKVCache::discard(std::uint64_t manifest, std::optional<DiskKVBlobId> blob) noexcept {
    try {
        std::lock_guard lock(mutex_);
        std::error_code error;
        if (blob) {
            const auto it = blobs_.find(*blob);
            if (it != blobs_.end()) {
                std::filesystem::remove(blob_path(*blob), error);
                used_bytes_ -= it->second.bytes;
                blobs_.erase(it);
            }
        }
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->id == manifest) {
                remove_manifest(it);
                break;
            }
        }
        ++stats_.removed_broken;
        ++stats_.read_failures;
    } catch (...) {}
}

std::uint32_t DiskKVCache::longest_indexed_frontier(std::span<const std::int32_t> prompt,
                                                    std::uint32_t max_frontier) {
    std::lock_guard lock(mutex_);
    std::uint32_t best = 0;
    std::vector<std::pair<std::uint32_t, std::uint64_t>> digests;
    for (const Entry& entry : entries_) {
        if (!entry.key_ok) { continue; }
        for (const DiskKVCheckpoint& checkpoint : entry.checkpoints) {
            if (checkpoint.frontier <= best || checkpoint.frontier > max_frontier ||
                checkpoint.frontier > prompt.size()) {
                continue;
            }
            std::optional<std::uint64_t> digest;
            for (const auto& [f, d] : digests) {
                if (f == checkpoint.frontier) { digest = d; }
            }
            if (!digest) {
                digest = xxh64(prompt.data(), checkpoint.frontier * sizeof(std::int32_t));
                digests.emplace_back(checkpoint.frontier, *digest);
            }
            if (*digest == checkpoint.digest) { best = checkpoint.frontier; }
        }
    }
    return best;
}

std::optional<DiskKVCache::Match> DiskKVCache::find(std::span<const std::int32_t> prompt,
                                                    std::uint32_t max_frontier) {
    struct Candidate {
        std::uint64_t id;
        std::filesystem::path path;
        DiskKVCheckpoint checkpoint;
        std::vector<DiskKVBlobId> refs;
    };
    std::vector<Candidate> candidates;
    {
        std::lock_guard lock(mutex_);
        std::vector<std::pair<std::uint32_t, std::uint64_t>> digests;
        const auto digest_at = [&](std::uint32_t frontier) {
            for (const auto& [f, d] : digests) {
                if (f == frontier) { return d; }
            }
            const std::uint64_t d = xxh64(prompt.data(), frontier * sizeof(std::int32_t));
            digests.emplace_back(frontier, d);
            return d;
        };
        for (const Entry& entry : entries_) {
            if (!entry.key_ok) { continue; }
            for (const DiskKVCheckpoint& checkpoint : entry.checkpoints) {
                if (checkpoint.frontier == 0 || checkpoint.frontier > max_frontier ||
                    checkpoint.frontier > prompt.size()) {
                    continue;
                }
                if (digest_at(checkpoint.frontier) == checkpoint.digest) {
                    candidates.push_back({entry.id, entry.path, checkpoint, entry.refs});
                }
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.checkpoint.frontier > b.checkpoint.frontier;
    });
    std::vector<std::byte> chunk;
    for (Candidate& candidate : candidates) {
        DiskKVRecordFile file;
        file.fd_    = ::open(candidate.path.c_str(), O_RDONLY | O_CLOEXEC);
        bool broken = file.fd_ < 0;
        if (!broken) {
            struct stat st {};
            std::optional<DiskKVRecordHeader> header;
            if (::fstat(file.fd_, &st) == 0) {
                header = read_header(file.fd_, static_cast<std::uint64_t>(st.st_size));
            }
            broken = !header || header->kind != DiskKVRecordKind::Manifest ||
                     header->key != options_.key;
            if (!broken) { file.header_ = std::move(*header); }
        }
        bool match = false;
        if (!broken) {
            const std::uint64_t bytes = static_cast<std::uint64_t>(candidate.checkpoint.frontier) * 4U;
            if (chunk.empty()) { chunk.resize(kDiskKVBlockBytes); }
            match  = true;
            broken = !file.read_prefix(0, bytes, chunk, [&](std::span<const std::byte> data,
                                                            std::uint64_t offset) {
                if (std::memcmp(data.data(),
                                reinterpret_cast<const std::byte*>(prompt.data()) + offset,
                                data.size()) != 0) {
                    match = false;
                }
            });
        }
        if (broken) {
            discard(candidate.id, std::nullopt);
            continue;
        }
        if (!match) { continue; }
        file.direct_fd_ = ::open(candidate.path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        (void)::futimens(file.fd_, nullptr);
        {
            std::lock_guard lock(mutex_);
            for (Entry& entry : entries_) {
                if (entry.id == candidate.id) { entry.last_use_ns = now_ns(); }
            }
            ++stats_.hits;
        }
        return Match{.manifest   = std::move(file),
                     .checkpoint = candidate.checkpoint,
                     .blob_refs  = std::move(candidate.refs)};
    }
    return std::nullopt;
}

} // namespace ninfer
