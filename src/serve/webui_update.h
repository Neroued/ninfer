#pragma once

// Keep a local copy of the prebuilt llama.cpp webui (tools/ui static output)
// current. Upstream publishes it to the Hugging Face bucket ggml-org/llama-ui
// under both release-tag folders and a rolling "latest" pointer.

#include "serve/serve_options.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::serve {

// Resolves the webui directory for --webui auto mode: --webui-dir if set,
// otherwise <directory of the artifact path>/webui.
std::string resolve_webui_dir(const ServeOptions& options);

// Downloads the current prebuilt llama.cpp webui (ggml-org/llama-ui HF bucket,
// "latest" pointer) into webui_dir when the local copy is missing, stale, or
// incomplete, then returns webui_dir ready to serve. Throws std::runtime_error
// on an unrecoverable download failure. No-op when the local copy is current.
std::string ensure_webui_available(const std::string& webui_dir);

} // namespace ninfer::serve