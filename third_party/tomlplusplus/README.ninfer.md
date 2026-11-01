# Vendored toml++

NInfer vendors the upstream toml++ single-header distribution:

- upstream: <https://github.com/marzer/tomlplusplus>
- tag: `v3.4.0`
- commit: `30172438cee64926dc41fdd9c11fb3ba5b2ba9de`
- license: MIT; see [LICENSE](LICENSE)

`toml.hpp` and `LICENSE` are unchanged upstream files. The header supports TOML 1.0 and
requires C++17 or later. It uses the upstream header-only defaults without additional dependencies.

`cmake/Dependencies.cmake` provides `ninfer::toml` when product support is enabled. Link it
privately from the product configuration implementation and include `<toml.hpp>` there; keep
TOML types out of public configuration and Engine interfaces. Configuration does not download
sources or discover a system TOML package.
