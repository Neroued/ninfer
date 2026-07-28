#include "product/media_acquire/acquire.h"

#include <curl/curl.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::product::media_acquire {
namespace {

std::vector<std::uint8_t> decode_base64(std::string_view text) {
    static constexpr std::array<std::int8_t, 256> table = [] {
        std::array<std::int8_t, 256> out{};
        out.fill(-1);
        for (int i = 0; i < 26; ++i) {
            out[static_cast<std::size_t>('A' + i)] = static_cast<std::int8_t>(i);
            out[static_cast<std::size_t>('a' + i)] = static_cast<std::int8_t>(26 + i);
        }
        for (int i = 0; i < 10; ++i) {
            out[static_cast<std::size_t>('0' + i)] = static_cast<std::int8_t>(52 + i);
        }
        out[static_cast<std::size_t>('+')] = 62;
        out[static_cast<std::size_t>('/')] = 63;
        return out;
    }();

    std::vector<std::uint8_t> out;
    out.reserve(text.size() * 3 / 4);
    std::uint32_t bits = 0;
    int count          = 0;
    bool padded        = false;
    for (const unsigned char c : text) {
        if (c == '=') {
            padded = true;
            continue;
        }
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') { continue; }
        if (padded || table[c] < 0) { throw std::invalid_argument("malformed base64 media data"); }
        bits = (bits << 6U) | static_cast<std::uint32_t>(table[c]);
        count += 6;
        if (count >= 8) {
            count -= 8;
            out.push_back(static_cast<std::uint8_t>((bits >> count) & 0xffU));
        }
    }
    if (count >= 6) { throw std::invalid_argument("malformed base64 media padding"); }
    return out;
}

bool private_ipv4(std::uint32_t address) {
    const std::uint32_t a = ntohl(address);
    return (a >> 24U) == 0 || (a >> 24U) == 10 || (a >> 24U) == 127 || (a >> 16U) == 0xa9fe ||
           (a >> 20U) == 0xac1 || (a >> 16U) == 0xc0a8 || (a >> 22U) == 0x0191 ||
           (a >> 17U) == 0x633f || (a >> 24U) >= 224;
}

bool private_address(const sockaddr* address) {
    if (address->sa_family == AF_INET) {
        return private_ipv4(reinterpret_cast<const sockaddr_in*>(address)->sin_addr.s_addr);
    }
    if (address->sa_family != AF_INET6) { return true; }
    const in6_addr& a = reinterpret_cast<const sockaddr_in6*>(address)->sin6_addr;
    if (IN6_IS_ADDR_UNSPECIFIED(&a) || IN6_IS_ADDR_LOOPBACK(&a) || IN6_IS_ADDR_LINKLOCAL(&a) ||
        IN6_IS_ADDR_MULTICAST(&a) || (a.s6_addr[0] & 0xfeU) == 0xfcU) {
        return true;
    }
    if (IN6_IS_ADDR_V4MAPPED(&a)) {
        std::uint32_t v4 = 0;
        std::memcpy(&v4, &a.s6_addr[12], sizeof(v4));
        return private_ipv4(v4);
    }
    return false;
}

struct UrlParts {
    std::string scheme;
    std::string host;
    std::string port;
};

std::string curlu_part(CURLU* url, CURLUPart part, unsigned flags = 0) {
    char* raw            = nullptr;
    const CURLUcode code = curl_url_get(url, part, &raw, flags);
    if (code != CURLUE_OK || raw == nullptr) { return {}; }
    std::string out(raw);
    curl_free(raw);
    return out;
}

UrlParts parse_url(std::string_view value) {
    std::unique_ptr<CURLU, decltype(&curl_url_cleanup)> url(curl_url(), curl_url_cleanup);
    if (!url ||
        curl_url_set(url.get(), CURLUPART_URL, std::string(value).c_str(), 0) != CURLUE_OK) {
        throw std::invalid_argument("invalid media URL");
    }
    UrlParts out;
    out.scheme             = curlu_part(url.get(), CURLUPART_SCHEME);
    out.host               = curlu_part(url.get(), CURLUPART_HOST);
    out.port               = curlu_part(url.get(), CURLUPART_PORT, CURLU_DEFAULT_PORT);
    const std::string user = curlu_part(url.get(), CURLUPART_USER);
    if ((out.scheme != "http" && out.scheme != "https") || out.host.empty() || out.port.empty() ||
        !user.empty()) {
        throw std::invalid_argument("media URL must be credential-free HTTP(S)");
    }
    return out;
}

std::string resolve_public(const UrlParts& url, bool allow_private) {
#ifdef _WIN32
    static std::once_flag winsock_init;
    std::call_once(winsock_init, [] {
        WSADATA data{};
        const int result = ::WSAStartup(MAKEWORD(2, 2), &data);
        if (result != 0) {
            throw Error(ErrorKind::RemoteUnavailable,
                        "failed to initialize Winsock: " + std::to_string(result));
        }
    });
#endif
    addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* raw     = nullptr;
    const int rc      = getaddrinfo(url.host.c_str(), url.port.c_str(), &hints, &raw);
    if (rc != 0) {
        throw Error(ErrorKind::RemoteUnavailable,
#ifdef _WIN32
                    "failed to resolve media URL host: " + std::to_string(rc));
#else
                    "failed to resolve media URL host: " + std::string(gai_strerror(rc)));
#endif
    }
    std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> addresses(raw, freeaddrinfo);
    std::string selected;
    for (const addrinfo* it = raw; it != nullptr; it = it->ai_next) {
        if (!allow_private && private_address(it->ai_addr)) { continue; }
        std::array<char, INET6_ADDRSTRLEN> text{};
        const void* bytes =
            it->ai_family == AF_INET
                ? static_cast<const void*>(
                      &reinterpret_cast<const sockaddr_in*>(it->ai_addr)->sin_addr)
                : static_cast<const void*>(
                      &reinterpret_cast<const sockaddr_in6*>(it->ai_addr)->sin6_addr);
        if (inet_ntop(it->ai_family, bytes, text.data(), text.size()) != nullptr) {
            selected = text.data();
            if (it->ai_family == AF_INET) { break; }
        }
    }
    if (selected.empty()) {
        throw std::invalid_argument("media URL resolves only to disallowed network addresses");
    }
    return selected;
}

struct CurlBuffer {
    std::vector<std::uint8_t> bytes;
    std::size_t limit = 0;
};

std::size_t curl_write(char* data, std::size_t size, std::size_t count, void* opaque) {
    auto& out = *static_cast<CurlBuffer*>(opaque);
    if (size != 0 && count > std::numeric_limits<std::size_t>::max() / size) { return 0; }
    const std::size_t amount = size * count;
    if (amount > out.limit - std::min(out.limit, out.bytes.size())) { return 0; }
    out.bytes.insert(out.bytes.end(), reinterpret_cast<std::uint8_t*>(data),
                     reinterpret_cast<std::uint8_t*>(data) + amount);
    return amount;
}

std::vector<std::uint8_t> fetch_url(std::string url, const Policy& policy) {
    if (!policy.allow_remote) { throw std::invalid_argument("remote media URLs are disabled"); }
    static std::once_flag init;
    std::call_once(init, [] {
        if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
            throw std::runtime_error("failed to initialize libcurl");
        }
    });

    for (int redirect = 0; redirect <= policy.max_redirects; ++redirect) {
        const UrlParts parts = parse_url(url);
        const std::string ip = resolve_public(parts, policy.allow_private_network);
        std::string resolve  = parts.host + ":" + parts.port + ":";
        resolve += ip.find(':') == std::string::npos ? ip : "[" + ip + "]";
        curl_slist* resolve_list = curl_slist_append(nullptr, resolve.c_str());
        if (resolve_list == nullptr) { throw std::bad_alloc(); }
        std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> resolve_guard(
            resolve_list, curl_slist_free_all);
        std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(),
                                                                 curl_easy_cleanup);
        if (!curl) { throw std::runtime_error("failed to create libcurl handle"); }
        CurlBuffer buffer{.bytes = {}, .limit = policy.max_bytes};
        curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl.get(), CURLOPT_PROTOCOLS_STR, "http,https");
        curl_easy_setopt(curl.get(), CURLOPT_REDIR_PROTOCOLS_STR, "http,https");
        curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 0L);
        curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT_MS, policy.connect_timeout_ms);
        curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT_MS, policy.timeout_ms);
        curl_easy_setopt(curl.get(), CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl.get(), CURLOPT_SSL_VERIFYPEER, 1L);
        curl_easy_setopt(curl.get(), CURLOPT_SSL_VERIFYHOST, 2L);
        curl_easy_setopt(curl.get(), CURLOPT_PROXY, "");
        curl_easy_setopt(curl.get(), CURLOPT_RESOLVE, resolve_list);
        curl_easy_setopt(curl.get(), CURLOPT_MAXFILESIZE_LARGE,
                         static_cast<curl_off_t>(policy.max_bytes));
        curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, curl_write);
        curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &buffer);
        curl_easy_setopt(curl.get(), CURLOPT_USERAGENT, "ninfer/vision");
        const CURLcode code = curl_easy_perform(curl.get());
        if (code != CURLE_OK) {
            if (code == CURLE_OPERATION_TIMEDOUT) {
                throw Error(ErrorKind::RemoteTimeout,
                            "media URL fetch timed out: " + std::string(curl_easy_strerror(code)));
            }
            if (code == CURLE_FILESIZE_EXCEEDED || code == CURLE_WRITE_ERROR) {
                throw Error(ErrorKind::BudgetExceeded, "media URL exceeds byte limit");
            }
            throw Error(ErrorKind::RemoteUnavailable,
                        "failed to fetch media URL: " + std::string(curl_easy_strerror(code)));
        }
        long status = 0;
        curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &status);
        if (status >= 200 && status < 300) { return buffer.bytes; }
        if (status < 300 || status >= 400 || redirect == policy.max_redirects) {
            throw Error(ErrorKind::RemoteUnavailable,
                        "media URL returned HTTP " + std::to_string(status));
        }
        char* next = nullptr;
        curl_easy_getinfo(curl.get(), CURLINFO_REDIRECT_URL, &next);
        if (next == nullptr || *next == '\0') {
            throw Error(ErrorKind::RemoteUnavailable, "media URL redirect has no location");
        }
        url = next;
    }
    throw Error(ErrorKind::RemoteUnavailable, "too many media URL redirects");
}

std::vector<std::uint8_t> read_path(const Source& source, const Policy& policy) {
    std::error_code ec;
    std::filesystem::path path = std::filesystem::weakly_canonical(source.value, ec);
    if (ec || !std::filesystem::is_regular_file(path, ec)) {
        throw std::invalid_argument("media path is not a regular file: " + source.value);
    }
    if (!policy.media_root.empty()) {
        const std::filesystem::path root = std::filesystem::weakly_canonical(policy.media_root, ec);
        const auto relative              = std::filesystem::relative(path, root, ec);
        if (ec || relative.empty() || relative.generic_string().starts_with("..")) {
            throw std::invalid_argument("media path is outside configured media root");
        }
    }
    const std::uintmax_t size = std::filesystem::file_size(path, ec);
    if (ec) { throw std::invalid_argument("failed to inspect media file: " + source.value); }
    if (size > policy.max_bytes) {
        throw Error(ErrorKind::BudgetExceeded, "media file exceeds byte limit");
    }

    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::invalid_argument("failed to open media path: " + path.string()); }
    stream.seekg(0, std::ios::end);
    const std::streamoff end = stream.tellg();
    if (end < 0 || static_cast<std::uintmax_t>(end) > policy.max_bytes) {
        throw Error(ErrorKind::BudgetExceeded, "media file exceeds byte limit");
    }
    stream.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    if (!bytes.empty()) {
        stream.read(reinterpret_cast<char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()));
        if (!stream) { throw std::invalid_argument("failed to read media path: " + path.string()); }
    }
    return bytes;
}

} // namespace

std::vector<std::uint8_t> acquire_bytes(const Source& source, const Policy& policy) {
    if (policy.max_bytes == 0) { throw std::invalid_argument("media byte limit must be positive"); }
    if (source.kind == SourceKind::Bytes) {
        if (source.bytes.empty()) { throw std::invalid_argument("media source is empty"); }
        if (source.bytes.size() > policy.max_bytes) {
            throw Error(ErrorKind::BudgetExceeded, "media bytes exceed byte limit");
        }
        return source.bytes;
    }
    if (source.value.empty()) { throw std::invalid_argument("media source is empty"); }

    if (source.kind == SourceKind::Url) {
        std::vector<std::uint8_t> bytes = fetch_url(source.value, policy);
        if (bytes.empty()) { throw std::invalid_argument("media source contains no data"); }
        return bytes;
    }
    if (source.kind == SourceKind::Data) {
        const std::size_t comma = source.value.find(',');
        if (!source.value.starts_with("data:") || comma == std::string::npos ||
            source.value.substr(0, comma).find(";base64") == std::string::npos) {
            throw std::invalid_argument("media data source must be a base64 data URI");
        }
        const std::size_t encoded = source.value.size() - comma - 1;
        if (encoded > (policy.max_bytes / 3 + 1) * 4) {
            throw Error(ErrorKind::BudgetExceeded, "media data exceeds byte limit");
        }
        std::vector<std::uint8_t> bytes =
            decode_base64(std::string_view(source.value).substr(comma + 1));
        if (bytes.size() > policy.max_bytes) {
            throw Error(ErrorKind::BudgetExceeded, "media data exceeds byte limit");
        }
        if (bytes.empty()) { throw std::invalid_argument("media source contains no data"); }
        return bytes;
    }
    return read_path(source, policy);
}

} // namespace ninfer::product::media_acquire
