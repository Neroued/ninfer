// Disk tier of the context cache (v0.7.2, format 2): content-addressed blobs shared between saves,
// manifests, reference counts, LRU under the cap, free-space margin, directory lock, atomicity.
#include "core/disk_kv_cache.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
using ninfer::DiskKVBlobId;
using ninfer::DiskKVBuffer;
using ninfer::DiskKVCache;
using ninfer::DiskKVRecordKind;
using ninfer::DiskKVRecordSpec;
using ninfer::DiskKVSectionKind;

namespace {

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (condition) { return; }
    std::cerr << "FAIL: " << label << '\n';
    ++failures;
}

constexpr std::size_t kMiB = 1ULL << 20U;

std::uint64_t directory_bytes(const fs::path& dir) {
    std::uint64_t bytes = 0;
    std::error_code error;
    for (const auto& item : fs::recursive_directory_iterator(dir, error)) {
        if (item.is_regular_file()) { bytes += item.file_size(); }
    }
    return bytes;
}

DiskKVCache::Options options(const fs::path& dir, std::uint64_t capacity,
                             const std::string& key = "model=a;kv=int8;tp=2") {
    DiskKVCache::Options out;
    out.directory       = dir;
    out.capacity_bytes  = capacity;
    out.min_free_bytes  = 1; // the test machine's free space is not the subject (see test_free)
    out.key             = key;
    out.staging_buffers = 2;
    out.staging_bytes   = 4 * kMiB;
    return out;
}

std::vector<std::int32_t> tokens(std::size_t count, std::int32_t base) {
    std::vector<std::int32_t> out(count);
    std::iota(out.begin(), out.end(), base);
    return out;
}

std::byte pattern(std::uint64_t seed, std::uint32_t section, std::uint64_t offset) {
    return static_cast<std::byte>((seed * 131U + section * 17U + offset * 7U + (offset >> 13U)) &
                                  0xffU);
}

// Streams `sizes` sections filled with pattern(seed, i + 1, offset) through the writer.
bool write_payload(DiskKVCache::Writer& writer, std::uint64_t seed,
                   const std::vector<std::uint64_t>& sizes, std::uint64_t stop_after = UINT64_MAX) {
    std::uint64_t written = 0;
    std::optional<DiskKVBuffer> buffer;
    std::size_t fill = 0;
    for (std::uint32_t section = 0; section < sizes.size(); ++section) {
        for (std::uint64_t offset = 0; offset < sizes[section]; ++offset) {
            if (!buffer) {
                buffer = writer.acquire();
                if (!buffer) { return false; }
                fill = 0;
            }
            buffer->data[fill++] = pattern(seed, section + 1, offset);
            ++written;
            if (fill == buffer->bytes) {
                writer.submit(*buffer, fill);
                buffer.reset();
            }
            if (written == stop_after) {
                if (buffer) { writer.submit(*buffer, fill); }
                return true;
            }
        }
    }
    if (buffer) { writer.submit(*buffer, fill); }
    return true;
}

DiskKVBlobId blob_id(std::uint64_t n) { return DiskKVBlobId{n * 0x1234567ULL + 1, n ^ 0xabcdefULL}; }

bool write_blob(DiskKVCache& cache, std::uint64_t session, DiskKVBlobId id, std::uint64_t seed,
                std::uint64_t bytes) {
    DiskKVRecordSpec spec;
    spec.kind    = DiskKVRecordKind::Blob;
    spec.blob    = id;
    spec.session = session;
    spec.tokens  = tokens(5, static_cast<std::int32_t>(seed));
    spec.sections.push_back({.kind = static_cast<std::uint32_t>(DiskKVSectionKind::TextKV), .bytes = bytes});
    auto writer = cache.begin_record(std::move(spec));
    return writer && write_payload(*writer, seed, {bytes}) && writer->commit();
}

// A conversation: `blocks` blobs of block_bytes (seed = id number), one state blob, a manifest
// with a tail section. Returns false if anything was refused.
struct Conversation {
    std::vector<std::int32_t> toks;
    std::vector<std::uint64_t> blocks; // blob numbers
    std::uint64_t state = 0;
    std::uint64_t tail_bytes = 0;
};

bool save(DiskKVCache& cache, const Conversation& conv, std::uint64_t block_bytes,
          std::uint64_t state_bytes, bool commit_manifest = true) {
    std::vector<DiskKVBlobId> ids;
    for (std::uint64_t n : conv.blocks) { ids.push_back(blob_id(n)); }
    ids.push_back(blob_id(conv.state));
    std::vector<bool> present;
    const std::uint64_t session = cache.begin_session(ids, present);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (present[i]) { continue; }
        const std::uint64_t n = i < conv.blocks.size() ? conv.blocks[i] : conv.state;
        if (!write_blob(cache, session, ids[i], n, i < conv.blocks.size() ? block_bytes : state_bytes)) {
            cache.abort_session(session);
            return false;
        }
    }
    if (!commit_manifest) {
        cache.abort_session(session);
        return true;
    }
    DiskKVRecordSpec spec;
    spec.kind      = DiskKVRecordKind::Manifest;
    spec.session   = session;
    spec.tokens    = conv.toks;
    spec.blob_refs = ids;
    spec.checkpoints.push_back({.frontier      = static_cast<std::uint32_t>(conv.toks.size()),
                                .kind          = 1,
                                .state_section = static_cast<std::uint32_t>(ids.size() - 1)});
    spec.sections.push_back(
        {.kind = static_cast<std::uint32_t>(DiskKVSectionKind::TextKV), .bytes = conv.tail_bytes});
    spec.meta[0] = 42;
    auto writer  = cache.begin_record(std::move(spec));
    if (!writer || !write_payload(*writer, conv.state + 1000, {conv.tail_bytes}) || !writer->commit()) {
        cache.abort_session(session);
        return false;
    }
    cache.drain(std::chrono::steady_clock::now() + std::chrono::seconds(30));
    return true;
}

bool check_blob(DiskKVCache& cache, std::uint64_t n, std::uint64_t bytes) {
    auto file = cache.open_blob(blob_id(n));
    if (!file) { return false; }
    std::vector<std::byte> chunk(4 * kMiB);
    bool same     = true;
    const bool ok = file->read_prefix(1, bytes, chunk,
                                      [&](std::span<const std::byte> data, std::uint64_t offset) {
                                          for (std::size_t i = 0; i < data.size(); ++i) {
                                              if (data[i] != pattern(n, 1, offset + i)) { same = false; }
                                          }
                                      });
    return ok && same;
}

bool check_blob_pipelined(DiskKVCache& cache, std::uint64_t n, std::uint64_t bytes) {
    auto file = cache.open_blob(blob_id(n));
    if (!file) { return false; }
    auto* a = static_cast<std::byte*>(std::aligned_alloc(4096, 4 * kMiB));
    auto* b = static_cast<std::byte*>(std::aligned_alloc(4096, 4 * kMiB));
    std::vector<std::byte> copy;
    std::uint64_t copy_offset = 0, seen = 0;
    bool same     = true;
    const bool ok = file->read_prefix_pipelined(
        1, bytes, std::span<std::byte>(a, 4 * kMiB), std::span<std::byte>(b, 4 * kMiB),
        [&](std::span<const std::byte> data, std::uint64_t offset) {
            copy.assign(data.begin(), data.end());
            copy_offset = offset;
        },
        [&] {
            for (std::size_t i = 0; i < copy.size(); ++i) {
                if (copy[i] != pattern(n, 1, copy_offset + i)) { same = false; }
            }
            seen += copy.size();
        });
    std::free(a);
    std::free(b);
    return ok && same && seen == bytes;
}

std::size_t count_files(const fs::path& dir, const std::string& needle) {
    std::size_t count = 0;
    std::error_code error;
    for (const auto& item : fs::recursive_directory_iterator(dir, error)) {
        if (item.path().filename().string().find(needle) != std::string::npos) { ++count; }
    }
    return count;
}

void test_xxh64() {
    expect(ninfer::xxh64("", 0) == 0xEF46DB3751D8E999ULL, "xxh64 empty");
    expect(ninfer::xxh64("a", 1) == 0xD24EC4F1A98C6E5BULL, "xxh64 a");
    expect(ninfer::xxh64("abc", 3) == 0x44BC2CF5AD770999ULL, "xxh64 abc");
    std::vector<unsigned char> data(1000);
    for (std::size_t i = 0; i < data.size(); ++i) { data[i] = static_cast<unsigned char>(i * 31U); }
    ninfer::Xxh64Stream stream;
    stream.update(data.data(), 7);
    stream.update(data.data() + 7, 100);
    stream.update(data.data() + 107, 893);
    expect(stream.digest() == ninfer::xxh64(data.data(), data.size()), "xxh64 streaming");
}

void test_roundtrip_and_sharing(const fs::path& dir) {
    fs::remove_all(dir);
    const std::uint64_t block = 3 * kMiB + 4096, state = 5 * kMiB + 123;
    {
        DiskKVCache cache(options(dir, 1024 * kMiB));
        const Conversation one{tokens(1000, 5), {1, 2, 3}, 100, 2 * kMiB + 7};
        expect(save(cache, one, block, state), "roundtrip: saved");
        const auto stats = cache.stats();
        expect(stats.records == 1 && stats.blobs == 4 && stats.written_blobs == 4,
               "roundtrip: one manifest, four blobs");
        // A second conversation with the same first two blocks (shared prefix): they are reused.
        std::vector<std::int32_t> other = tokens(1000, 5);
        other[900]                      = -7;
        const Conversation two{other, {1, 2, 4}, 101, kMiB};
        expect(save(cache, two, block, state), "sharing: saved");
        const auto after = cache.stats();
        expect(after.records == 2 && after.blobs == 6 && after.written_blobs == 6 &&
                   after.reused_blobs == 2,
               "sharing: the shared blocks are on disk once");
        // Turn 2 of the first conversation: supersedes it, keeps its blocks, adds one.
        std::vector<std::int32_t> longer = tokens(1500, 5);
        const Conversation three{longer, {1, 2, 3, 5}, 102, kMiB};
        expect(save(cache, three, block, state), "turn 2: saved");
        const auto turn2 = cache.stats();
        expect(turn2.records == 2 && turn2.superseded == 1, "turn 2: earlier save replaced");
        expect(!cache.open_blob(blob_id(100)).has_value(), "turn 2: the old state is gone");
        expect(cache.open_blob(blob_id(3)).has_value(), "turn 2: the old block stays");
        expect(turn2.bytes == directory_bytes(dir / "m") + directory_bytes(dir / "b"),
               "accounting: bytes match the files");
        // Manifests hold token IDs: directories 0700, files 0600 whatever the umask.
        const auto private_to_owner = [](const fs::path& path) {
            const auto bits = fs::status(path).permissions() & fs::perms::all;
            return (bits & (fs::perms::group_all | fs::perms::others_all)) == fs::perms::none;
        };
        bool all_private = private_to_owner(dir / "m") && private_to_owner(dir / "b") &&
                           private_to_owner(dir / ".lock");
        for (const auto* sub : {"m", "b"}) {
            for (const auto& item : fs::recursive_directory_iterator(dir / sub)) {
                all_private = all_private && private_to_owner(item.path());
            }
        }
        expect(all_private, "permissions: cache readable by its owner only");
    }
    expect(count_files(dir, ".tmp.") == 0, "roundtrip: no temporaries left");
    DiskKVCache cache(options(dir, 1024 * kMiB));
    expect(cache.stats().records == 2 && cache.stats().blobs == 7, "restart: index reloaded");
    std::vector<std::int32_t> prompt = tokens(1501, 5);
    expect(cache.longest_indexed_frontier(prompt, 1500) == 1500, "restart: index probe frontier");
    expect(cache.longest_indexed_frontier(prompt, 1499) < 1500, "restart: index probe bound");
    auto match                       = cache.find(prompt, 1500);
    expect(match.has_value(), "restart: found");
    if (match) {
        expect(match->checkpoint.frontier == 1500, "restart: frontier");
        expect(match->manifest.header().meta[0] == 42, "restart: meta");
        expect(match->blob_refs.size() == 5, "restart: blob references");
        expect(match->blob_refs[match->checkpoint.state_section] == blob_id(102),
               "restart: checkpoint state reference");
        auto* region = static_cast<std::byte*>(std::aligned_alloc(4096, 4 * kMiB));
        const auto start = match->manifest.read_region(
            match->manifest.section_of(DiskKVSectionKind::TextKV), std::span<std::byte>(region, 4 * kMiB));
        bool same = start.has_value();
        for (std::size_t i = 0; same && i < kMiB; ++i) { same = region[i] == pattern(1102, 1, i); }
        expect(same, "restart: tail region");
        std::free(region);
    }
    expect(check_blob(cache, 1, block) && check_blob(cache, 102, state), "restart: blob bytes");
    expect(check_blob_pipelined(cache, 5, block) && check_blob_pipelined(cache, 102, state),
           "restart: pipelined blob reads");
    std::vector<std::int32_t> miss = tokens(1501, 6);
    expect(!cache.find(miss, 1500).has_value(), "restart: other prompt misses");
    expect(cache.longest_indexed_frontier(miss, 1500) == 0, "restart: index probe misses");
}

void test_key(const fs::path& dir) {
    // Reuses the directory of test_roundtrip_and_sharing.
    DiskKVCache cache(options(dir, 1024 * kMiB, "model=a;kv=bf16;tp=2"));
    const auto stats = cache.stats();
    expect(stats.records == 0 && stats.foreign_records == 2 && stats.blobs == 7,
           "key: manifests of another key ignored, their blobs kept");
    std::vector<std::int32_t> prompt = tokens(1501, 5);
    expect(!cache.find(prompt, 1500).has_value(), "key: no hit across keys");
    expect(cache.longest_indexed_frontier(prompt, 1500) == 0, "key: index probe misses across keys");
}

void test_lock(const fs::path& dir) {
    fs::remove_all(dir);
    DiskKVCache first(options(dir, 64 * kMiB));
    bool refused = false;
    std::string message;
    try {
        DiskKVCache second(options(dir, 64 * kMiB));
    } catch (const std::exception& error) {
        refused = true;
        message = error.what();
    }
    expect(refused, "lock: a second server on the same directory is refused");
    const auto holder = DiskKVCache::directory_holder(dir);
    expect(holder && *holder == std::to_string(::getpid()), "lock: the probe names the holder");
    expect(!DiskKVCache::directory_holder(dir / "nothing").has_value(), "lock: free directory");
    expect(message.find(std::to_string(::getpid())) != std::string::npos &&
               message.find("--disk-cache-dir") != std::string::npos,
           "lock: the message names the holder and the remedy");
}

void test_cap_and_orphans(const fs::path& dir) {
    fs::remove_all(dir);
    const std::uint64_t block = 2 * kMiB, state = 3 * kMiB;
    // One conversation = 2 blocks + 1 state + manifest; room for about three.
    DiskKVCache sizing(options(dir / "sizing", 1024 * kMiB, "k"));
    expect(save(sizing, {tokens(300, 1), {11, 12}, 13, kMiB}, block, state), "cap: sizing save");
    const std::uint64_t one = sizing.stats().bytes;
    DiskKVCache cache(options(dir / "c", 3 * one + one / 2, "k"));
    for (int i = 0; i < 6; ++i) {
        const auto base = static_cast<std::uint64_t>(100 * (i + 1));
        expect(save(cache, {tokens(300, static_cast<std::int32_t>(base)), {base, base + 1}, base + 2, kMiB},
                    block, state),
               "cap: save " + std::to_string(i));
        expect(directory_bytes(dir / "c") <= cache.options().capacity_bytes,
               "cap: files under the cap after save " + std::to_string(i));
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
    const auto stats = cache.stats();
    expect(stats.records == 3 && stats.evicted == 3 && stats.blobs == 9,
           "cap: LRU manifests evicted with their blobs");
    // An interrupted save leaves its blobs as orphans; the next attempt reuses them.
    expect(save(cache, {tokens(300, 9000), {9000, 9001}, 9002, kMiB}, block, state, false),
           "orphans: interrupted save");
    cache.drain(std::chrono::steady_clock::now() + std::chrono::seconds(10));
    expect(cache.stats().orphan_blobs == 3, "orphans: kept");
    expect(save(cache, {tokens(300, 9000), {9000, 9001}, 9002, kMiB}, block, state),
           "orphans: second attempt");
    expect(cache.stats().reused_blobs >= 3 && cache.stats().orphan_blobs == 0, "orphans: reused");
    // A record larger than the cap is refused.
    DiskKVRecordSpec big;
    big.kind = DiskKVRecordKind::Blob;
    big.blob = blob_id(77777);
    big.tokens = tokens(5, 0);
    big.sections.push_back({.kind = 3, .bytes = 10 * one});
    expect(!cache.begin_record(std::move(big)).has_value(), "cap: record above the cap refused");
}

void test_orphans_removed_at_start(const fs::path& dir) {
    fs::remove_all(dir);
    {
        DiskKVCache cache(options(dir, 1024 * kMiB));
        save(cache, {tokens(300, 1), {1, 2}, 3, kMiB}, kMiB, kMiB, false);
        cache.drain(std::chrono::steady_clock::now() + std::chrono::seconds(10));
        expect(cache.stats().orphan_blobs == 3, "start: orphans before restart");
    }
    DiskKVCache cache(options(dir, 1024 * kMiB));
    expect(cache.stats().blobs == 0 && cache.stats().removed_orphans == 3,
           "start: orphans deleted at start");
}

void test_free(const fs::path& dir) {
    fs::remove_all(dir);
    // A file system of 40 MiB with a 10 MiB margin: the cache keeps 10 MiB free.
    DiskKVCache::Options opts = options(dir, 1024 * kMiB);
    opts.min_free_bytes       = 10 * kMiB;
    opts.free_bytes           = [dir] {
        const std::uint64_t used = directory_bytes(dir);
        return used >= 40 * kMiB ? 0 : 40 * kMiB - used;
    };
    DiskKVCache cache(opts);
    for (int i = 0; i < 5; ++i) {
        const auto base = static_cast<std::uint64_t>(100 * (i + 1));
        save(cache, {tokens(300, static_cast<std::int32_t>(base)), {base}, base + 1, kMiB}, 4 * kMiB,
             4 * kMiB);
        expect(directory_bytes(dir) <= 30 * kMiB + 64 * 1024,
               "free: margin kept after save " + std::to_string(i));
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
    expect(cache.stats().evicted >= 1, "free: older conversations evicted for the margin");
    // A record that cannot fit even in an empty cache is skipped.
    DiskKVRecordSpec big;
    big.kind   = DiskKVRecordKind::Blob;
    big.blob   = blob_id(55555);
    big.tokens = tokens(5, 0);
    big.sections.push_back({.kind = 3, .bytes = 35 * kMiB});
    expect(!cache.begin_record(std::move(big)).has_value() && cache.stats().space_refusals >= 1,
           "free: record above the free space refused");
}

void test_corruption(const fs::path& dir) {
    fs::remove_all(dir);
    {
        DiskKVCache cache(options(dir, 1024 * kMiB));
        expect(save(cache, {tokens(300, 1), {1, 2}, 3, kMiB}, 6 * kMiB, kMiB), "corruption: saved");
    }
    // Flip one payload byte of blob 1: its read fails, the others still read.
    const fs::path victim = dir / "b" / (blob_id(1).hex() + ".nkvb");
    {
        std::fstream stream(victim, std::ios::in | std::ios::out | std::ios::binary);
        stream.seekp(static_cast<std::streamoff>(fs::file_size(victim) - 100));
        char byte = 0;
        stream.write(&byte, 1);
    }
    {
        DiskKVCache cache(options(dir, 1024 * kMiB));
        expect(!check_blob(cache, 1, 6 * kMiB), "corruption: flipped byte detected");
        expect(!check_blob_pipelined(cache, 1, 6 * kMiB), "corruption: pipelined reader detects it");
        expect(check_blob(cache, 2, 6 * kMiB), "corruption: intact blob reads");
        auto match = cache.find(tokens(301, 1), 300);
        expect(match.has_value(), "corruption: manifest still found");
        if (match) { cache.discard(match->manifest.header().id, blob_id(1)); }
        expect(cache.stats().records == 0 && cache.stats().blobs == 0,
               "corruption: discard drops the manifest, the broken blob and the blobs only it used");
    }
    // A manifest whose blob is missing is deleted at start; a broken manifest header too.
    fs::remove_all(dir);
    {
        DiskKVCache cache(options(dir, 1024 * kMiB));
        save(cache, {tokens(300, 1), {1, 2}, 3, kMiB}, kMiB, kMiB);
        save(cache, {tokens(300, 50), {50}, 51, kMiB}, kMiB, kMiB);
    }
    fs::remove(dir / "b" / (blob_id(2).hex() + ".nkvb"));
    fs::path second;
    {
        DiskKVCache cache(options(dir, 1024 * kMiB));
        expect(cache.stats().records == 1 && cache.stats().removed_broken == 1,
               "start: a manifest with a missing blob is deleted");
    }
    for (const auto& item : fs::directory_iterator(dir / "m")) { second = item.path(); }
    {
        std::fstream stream(second, std::ios::in | std::ios::out | std::ios::binary);
        stream.seekp(200);
        char byte = 'x';
        stream.write(&byte, 1);
    }
    {
        DiskKVCache cache(options(dir, 1024 * kMiB));
        expect(cache.stats().records == 0 && cache.stats().removed_broken == 1 &&
                   cache.stats().blobs == 0,
               "start: broken manifest deleted, its blobs become orphans and go");
    }
}

void test_abort(const fs::path& dir) {
    fs::remove_all(dir);
    DiskKVCache cache(options(dir, 1024 * kMiB));
    {
        DiskKVRecordSpec spec;
        spec.kind   = DiskKVRecordKind::Blob;
        spec.blob   = blob_id(1);
        spec.tokens = tokens(5, 0);
        spec.sections.push_back({.kind = 3, .bytes = 6 * kMiB});
        auto writer = cache.begin_record(std::move(spec));
        expect(writer.has_value(), "abort: begun");
        if (writer) {
            write_payload(*writer, 1, {6 * kMiB}, 5 * kMiB);
            writer->abort();
        }
    }
    cache.drain(std::chrono::steady_clock::now() + std::chrono::seconds(10));
    expect(count_files(dir, ".nkvb") == 0 && cache.stats().aborted == 1, "abort: no file left");
    DiskKVRecordSpec spec;
    spec.kind   = DiskKVRecordKind::Blob;
    spec.blob   = blob_id(2);
    spec.tokens = tokens(5, 0);
    spec.sections.push_back({.kind = 3, .bytes = 6 * kMiB});
    auto writer = cache.begin_record(std::move(spec));
    if (writer) {
        write_payload(*writer, 2, {6 * kMiB}, 3 * kMiB);
        expect(!writer->commit(), "abort: incomplete payload refused");
    }
    cache.drain(std::chrono::steady_clock::now() + std::chrono::seconds(10));
    expect(count_files(dir, ".nkvb") == 0, "abort: refused payload removed");
}

// kill -9 in the middle of a write: the next start removes the temporary (and the lock is free).
void test_kill(const fs::path& dir) {
    fs::remove_all(dir);
    {
        DiskKVCache cache(options(dir, 1024 * kMiB));
        save(cache, {tokens(150, 50), {51}, 52, kMiB}, kMiB, kMiB);
    }
    std::cout.flush();
    std::cerr.flush();
    const pid_t child = ::fork();
    if (child == 0) {
        int writes                         = 0;
        DiskKVCache::Options child_options = options(dir, 1024 * kMiB);
        child_options.before_write         = [&] {
            if (++writes == 2) { ::raise(SIGKILL); }
        };
        DiskKVCache cache(child_options);
        DiskKVRecordSpec spec;
        spec.kind   = DiskKVRecordKind::Blob;
        spec.blob   = blob_id(60);
        spec.tokens = tokens(5, 0);
        spec.sections.push_back({.kind = 3, .bytes = 16 * kMiB});
        auto writer = cache.begin_record(std::move(spec));
        if (writer) { write_payload(*writer, 60, {16 * kMiB}); }
        std::this_thread::sleep_for(std::chrono::seconds(10));
        ::_exit(3);
    }
    int status = 0;
    ::waitpid(child, &status, 0);
    expect(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL, "kill: child killed mid-write");
    expect(count_files(dir, ".tmp.") == 1, "kill: temporary left by the dead writer");
    DiskKVCache cache(options(dir, 1024 * kMiB));
    expect(count_files(dir, ".tmp.") == 0, "kill: temporary removed at start");
    expect(cache.stats().removed_temp == 1 && cache.stats().records == 1,
           "kill: earlier save intact");
    expect(cache.find(tokens(151, 50), 150).has_value() && check_blob(cache, 51, kMiB),
           "kill: earlier save readable");
}

} // namespace

int main() {
    const fs::path root =
        fs::temp_directory_path() / ("ninfer_disk_kv_test_" + std::to_string(::getpid()));
    fs::remove_all(root);
    test_xxh64();
    test_roundtrip_and_sharing(root / "a");
    test_key(root / "a");
    test_lock(root / "l");
    test_cap_and_orphans(root / "c");
    test_orphans_removed_at_start(root / "o");
    test_free(root / "f");
    test_corruption(root / "b");
    test_abort(root / "e");
    test_kill(root / "k");
    fs::remove_all(root);
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "disk KV cache: all checks passed\n";
    return 0;
}
