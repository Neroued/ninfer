// webui_update.cpp — keeps a local copy of the prebuilt llama.cpp webui current.
//
// Upstream llama.cpp publishes the built webui (tools/ui static output) to the
// Hugging Face bucket ggml-org/llama-ui after each release, under both a
// release-tag folder and a rolling "latest" pointer. Layout:
//
//   https://huggingface.co/api/buckets/ggml-org/llama-ui/tree/latest
//       -> JSON file list: [{"type":"file","path":"latest/<rel>","size":N}, ...]
//   https://huggingface.co/buckets/ggml-org/llama-ui/resolve/latest/<rel>
//       -> 302 -> CDN byte stream for that file
//
// ensure_webui_available() compares the local marker file against
// latest/_app/version.json and, when the local copy is missing/stale/incomplete,
// downloads the full set into a staging directory and atomically swaps it in.
// Downloads use WinHTTP directly (the vendored httplib is built without TLS).

#include "serve/webui_update.h"

#include "serve/console_log.h"

#include <windows.h>
#include <winhttp.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#pragma comment(lib, "winhttp.lib")

namespace ninfer::serve {
namespace {

namespace fs = std::filesystem;

constexpr const char* kBucketApi    = "https://huggingface.co/api/buckets/ggml-org/llama-ui/tree/latest";
constexpr const char* kBucketBase   = "https://huggingface.co/buckets/ggml-org/llama-ui/resolve/latest/";
constexpr const char* kUserAgent    = "ninfer-serve";
constexpr const char* kMarkerFile   = ".ninfer-webui-version";
constexpr const char* kVersionPath  = "_app/version.json";
constexpr int kMaxAttempts          = 3;
constexpr DWORD kReadChunkBytes     = 1 << 20;

struct WebuiFile {
    std::string relative_path; // e.g. "index.html", "_app/immutable/..."
    uint64_t size             = 0;
};

// RAII closer for an HINTERNET handle. The vendored httplib has no TLS, so all
// webui downloads go through WinHTTP directly; a guard keeps every handle closed
// on both the success and every throw path.
struct HandleGuard {
    HINTERNET handle = nullptr;
    explicit HandleGuard(HINTERNET h) : handle(h) {}
    ~HandleGuard() {
        if (handle != nullptr) {
            ::WinHttpCloseHandle(handle);
            handle = nullptr;
        }
    }
    HandleGuard(const HandleGuard&) = delete;
    HandleGuard& operator=(const HandleGuard&) = delete;
};

std::string to_utf8(const fs::path& p) {
    const std::wstring w = p.wstring();
    if (w.empty()) { return std::string(); }
    const int len =
        ::WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<size_t>(len), '\0');
    ::WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), out.data(), len, nullptr, nullptr);
    return out;
}

std::wstring to_wide(const std::string& s) {
    if (s.empty()) { return std::wstring(); }
    const int len = ::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring out(static_cast<size_t>(len), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), len);
    return out;
}

// Splits "https://host/path..." into (host, path-and-query).
void split_url(const std::string& url, std::string& host, std::string& path) {
    const size_t scheme_end = url.find("://");
    // Host starts after the scheme so WinHttpConnect never sees "https://host".
    const size_t host_start = scheme_end == std::string::npos ? 0 : scheme_end + 3;
    const size_t path_start = url.find('/', host_start);
    host = url.substr(host_start, path_start == std::string::npos ? std::string::npos : path_start - host_start);
    path = path_start == std::string::npos ? "/" : url.substr(path_start);
}

// One WinHTTP GET to completion. Returns the body. Follows 301/302/303/307/308.
// Throws std::runtime_error on any failure.
std::string http_get(const std::string& url, const std::string& user_agent) {
    std::string host, path;
    split_url(url, host, path);

    const std::wstring w_agent = to_wide(user_agent);
    HINTERNET session = ::WinHttpOpen(w_agent.c_str(), WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, nullptr, nullptr, 0);
    if (session == nullptr) {
        throw std::runtime_error("WinHttpOpen failed: " + std::to_string(::GetLastError()));
    }
    HandleGuard session_guard(session);

    std::string body;
    for (int redirect = 0; redirect < 8; ++redirect) {
        const std::wstring w_host = to_wide(host);
        const std::wstring w_path = to_wide(path);

        // A fresh connect handle each pass: a redirect may cross to another host.
        HINTERNET connect = ::WinHttpConnect(session, w_host.c_str(), INTERNET_DEFAULT_HTTPS_PORT, 0);
        if (connect == nullptr) {
            throw std::runtime_error("WinHttpConnect(" + host + ") failed: " + std::to_string(::GetLastError()));
        }
        HandleGuard connect_guard(connect);

        HINTERNET request = ::WinHttpOpenRequest(connect, L"GET", w_path.c_str(), nullptr, nullptr, nullptr,
                                                  WINHTTP_FLAG_SECURE);
        if (request == nullptr) {
            throw std::runtime_error("WinHttpOpenRequest(" + url + ") failed: " +
                                     std::to_string(::GetLastError()));
        }
        HandleGuard request_guard(request);

        const std::wstring accept = L"Accept: application/json, text/html, */*";
        ::WinHttpAddRequestHeaders(request, accept.c_str(), static_cast<DWORD>(accept.size()),
                                   WINHTTP_ADDREQ_FLAG_ADD);

        if (!::WinHttpSendRequest(request, 0, 0, nullptr, 0, 0, 0) ||
            !::WinHttpReceiveResponse(request, nullptr)) {
            throw std::runtime_error("request to " + url + " failed: " + std::to_string(::GetLastError()));
        }

        DWORD status = 0;
        DWORD status_size = sizeof(status);
        ::WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, L"__WinHttpStatus",
                              &status, &status_size, nullptr);
        if (status >= 300 && status < 400) {
            DWORD location_size = 0;
            ::WinHttpQueryHeaders(request, WINHTTP_QUERY_LOCATION, nullptr, nullptr, &location_size, nullptr);
            if (location_size <= 1) {
                throw std::runtime_error("redirect from " + url + " had no Location header");
            }
            std::string location(location_size - 1, '\0');
            ::WinHttpQueryHeaders(request, WINHTTP_QUERY_LOCATION, nullptr, &location[0], &location_size, nullptr);
            split_url(location, host, path); // reconnect against the new host on the next pass
            continue;
        }
        if (status >= 400) {
            throw std::runtime_error("HTTP " + std::to_string(status) + " from " + url);
        }

        for (;;) {
            DWORD available = 0;
            if (!::WinHttpQueryDataAvailable(request, &available) || available == 0) { break; }
            std::vector<char> buffer(std::min<size_t>(available, kReadChunkBytes));
            DWORD read = 0;
            if (!::WinHttpReadData(request, buffer.data(), static_cast<DWORD>(buffer.size()), &read)) {
                throw std::runtime_error("read from " + url + " failed: " + std::to_string(::GetLastError()));
            }
            body.append(buffer.data(), read);
            if (read < available) { break; }
        }
        break;
    }
    return body;
}

std::string with_retry(const std::string& url, const std::string& what) {
    std::string last_error;
    for (int attempt = 1; attempt <= kMaxAttempts; ++attempt) {
        try {
            return http_get(url, kUserAgent);
        } catch (const std::exception& e) {
            last_error = e.what();
            if (attempt < kMaxAttempts) {
                ::Sleep(1000 * attempt);
            }
        }
    }
    throw std::runtime_error(what + ": " + last_error);
}

// File list for the "latest" pointer, with the "latest/" prefix stripped.
std::vector<WebuiFile> fetch_file_list() {
    const std::string body = with_retry(kBucketApi, "listing the webui bucket failed");
    const auto json = nlohmann::json::parse(body);
    std::vector<WebuiFile> files;
    for (const auto& entry : json) {
        if (entry.value("type", "") != "file") { continue; }
        const std::string path = entry.value("path", "");
        if (path.rfind("latest/", 0) != 0) { continue; } // "latest" pointer only
        WebuiFile file;
        file.relative_path = path.substr(7);
        file.size          = entry.value("size", 0ULL);
        if (file.relative_path.empty()) { continue; }
        files.push_back(std::move(file));
    }
    if (files.empty()) {
        throw std::runtime_error("webui bucket listing contained no files");
    }
    return files;
}

uint64_t file_size_or_zero(const fs::path& p) {
    std::error_code ec;
    const auto size = fs::file_size(p, ec);
    return ec ? 0 : size;
}

bool directory_exists(const fs::path& p) {
    std::error_code ec;
    return fs::is_directory(p, ec);
}

std::string read_file_text(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) { return std::string(); }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// Downloads one file to dest, verifying the final byte count against expected_size
// when it is non-zero.
void download_file(const std::string& relative_path, const fs::path& dest, uint64_t expected_size) {
    const std::string url = std::string(kBucketBase) + relative_path;
    std::string last_error;
    for (int attempt = 1; attempt <= kMaxAttempts; ++attempt) {
        try {
            const std::string body = http_get(url, kUserAgent);
            std::error_code ec;
            fs::create_directories(dest.parent_path(), ec);
            {
                std::ofstream out(dest, std::ios::binary | std::ios::trunc);
                if (!out) { throw std::runtime_error("cannot open " + to_utf8(dest)); }
                out.write(body.data(), static_cast<std::streamsize>(body.size()));
                out.flush();
                if (!out) { throw std::runtime_error("write failed for " + to_utf8(dest)); }
            }
            if (expected_size != 0 && file_size_or_zero(dest) != expected_size) {
                throw std::runtime_error("size mismatch for " + relative_path);
            }
            return;
        } catch (const std::exception& e) {
            last_error = e.what();
            if (attempt < kMaxAttempts) {
                ::Sleep(1000 * attempt);
            }
        }
    }
    throw std::runtime_error("downloading " + relative_path + " failed: " + last_error);
}

// The rolling "latest" pointer is only meaningful if it actually points at a
// published release folder; a 404 on version.json means the bucket is empty.
std::string fetch_latest_version() {
    const std::string body = with_retry(std::string(kBucketBase) + kVersionPath,
                                        "reading the webui version marker failed");
    const auto json = nlohmann::json::parse(body);
    const std::string version = json.value("version", "");
    if (version.empty()) {
        throw std::runtime_error("webui bucket has no version.json");
    }
    return version;
}

bool local_copy_is_current(const fs::path& webui_dir, const std::string& version) {
    if (!directory_exists(webui_dir)) { return false; }
    if (read_file_text(webui_dir / kMarkerFile) != version) { return false; }
    return file_size_or_zero(webui_dir / "index.html") != 0;
}

std::vector<fs::path> staged_stale_dirs(const fs::path& webui_dir) {
    // Staging dirs from a previously interrupted run: .ninfer-webui.<pid>.tmp
    std::vector<fs::path> stale;
    std::error_code ec;
    const fs::path parent = webui_dir.parent_path();
    const std::string prefix = ".ninfer-webui.";
    if (!directory_exists(parent)) { return stale; }
    for (const auto& entry : fs::directory_iterator(parent, ec)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind(prefix, 0) == 0 && name.size() > prefix.size() + 4 &&
            name.compare(name.size() - 4, 4, ".tmp") == 0) {
            stale.push_back(entry.path());
        }
    }
    return stale;
}

} // namespace

std::string resolve_webui_dir(const ServeOptions& options) {
    if (!options.webui_dir.empty()) { return options.webui_dir; }
    std::error_code ec;
    fs::path artifact(options.artifact_path);
    fs::path dir = artifact.parent_path();
    if (dir.empty()) { dir = fs::path("."); }
    return to_utf8((dir / "webui").lexically_normal());
}

std::string ensure_webui_available(const std::string& webui_dir) {
    const fs::path target(webui_dir);

    // Sweep staging dirs left behind by a previously interrupted download.
    for (const auto& stale : staged_stale_dirs(target)) {
        std::error_code ec;
        fs::remove_all(stale, ec);
        write_console_log(ConsoleLogLevel::Info, "removed stale webui staging directory " + to_utf8(stale));
    }

    const std::string version = fetch_latest_version();

    if (local_copy_is_current(target, version)) {
        write_console_log(ConsoleLogLevel::Info,
                          "webui up to date (version " + version + ") at " + webui_dir);
        return webui_dir;
    }

    write_console_log(ConsoleLogLevel::Info, "downloading latest webui (version " + version +
                                                  ") from ggml-org/llama-ui...");

    const std::vector<WebuiFile> files = fetch_file_list();

    // Stage in a sibling directory, then swap atomically: the served directory is
    // never half-written, and an interrupted run leaves the previous copy intact.
    const fs::path staging =
        target.parent_path() / (".ninfer-webui." + std::to_string(::GetCurrentProcessId()) + ".tmp");
    {
        std::error_code ec;
        fs::remove_all(staging, ec);
        fs::create_directories(staging, ec);
        if (ec) {
            throw std::runtime_error("cannot create staging directory " + to_utf8(staging) + ": " + ec.message());
        }
    }

    uint64_t total_bytes = 0;
    for (const auto& file : files) {
        const fs::path dest = staging / file.relative_path;
        download_file(file.relative_path, dest, file.size);
        total_bytes += file.size;
    }

    // version.json is downloaded with the rest; publish its value as the marker.
    const std::string staged_version = fetch_latest_version();
    {
        std::ofstream out(staging / kMarkerFile, std::ios::binary | std::ios::trunc);
        out << staged_version;
    }

    // Atomic swap: rename target aside, move staging in, drop the old copy.
    const fs::path old =
        target.parent_path() / (".ninfer-webui.old." + std::to_string(::GetCurrentProcessId()));
    std::error_code ec;
    fs::remove_all(old, ec);
    if (directory_exists(target)) {
        fs::rename(target, old, ec);
        if (ec) {
            throw std::runtime_error("cannot move aside " + to_utf8(target) + ": " + ec.message());
        }
    }
    ec.clear();
    fs::rename(staging, target, ec);
    if (ec) {
        // Roll the served copy back before failing.
        std::error_code ec2;
        if (directory_exists(old)) { fs::rename(old, target, ec2); }
        throw std::runtime_error("cannot install webui at " + to_utf8(target) + ": " + ec.message());
    }
    fs::remove_all(old, ec);

    const double mib = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
    write_console_log(ConsoleLogLevel::Info,
                      std::to_string(files.size()) + " webui files (" +
                          (mib >= 10 ? std::to_string(static_cast<int>(mib)) :
                                       std::to_string(static_cast<int>(mib * 10) / 10)) +
                          " MiB) installed at " + webui_dir);
    return webui_dir;
}

} // namespace ninfer::serve