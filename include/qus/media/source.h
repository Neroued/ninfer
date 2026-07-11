#pragma once

#include <string>

namespace qus::media {

enum class SourceKind {
    Path,
    Url,
    Data,
};

struct Source {
    SourceKind kind = SourceKind::Path;
    std::string value;
    std::string media_type;
};

} // namespace qus::media
