# Vendored spdlog

NInfer vendors the compiled-library sources required from the upstream spdlog `v1.17.0` release:

- upstream: <https://github.com/gabime/spdlog>
- tag: `v1.17.0`
- commit: `79524ddd08a4ec981b7fea76afd08ee05f83755d`
- license: MIT; see `LICENSE`

The committed `include/` and `src/` directories are unchanged upstream files. Examples, tests,
benchmarks, installation support, and upstream build-system files are intentionally omitted. The
local `CMakeLists.txt` builds the same static compiled-library source set with bundled fmt. NInfer
does not discover a system package or fetch network content during configuration.
