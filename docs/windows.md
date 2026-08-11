# Windows development on RTX 5090

NInfer supports a native Windows build in addition to Linux. The runtime remains specialized for
one NVIDIA GeForce RTX 5090 and CUDA architecture `sm_120a`.

## Prerequisites

- 64-bit Windows 11;
- NVIDIA driver and CUDA Toolkit 13.1 or newer;
- Visual Studio 2022 or newer with **Desktop development with C++**;
- CMake 3.28 or newer;
- vcpkg with `VCPKG_ROOT` set to its installation directory;
- Python 3.11 when building the tests or running artifact tooling.

The repository contains a manifest that pins the curl and FFmpeg dependencies. vcpkg installs them
under the selected build directory during the first configuration and reuses them afterward.

## Configure and build

Run from a Visual Studio developer PowerShell in the repository root. Select the generator matching
the installed Visual Studio release; this example uses Visual Studio 2022:

```powershell
cmake -S . -B build-windows `
  -G "Visual Studio 17 2022" -A x64 `
  -T "cuda=$env:CUDA_PATH" `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows `
  -DCMAKE_CUDA_ARCHITECTURES=120a `
  -DNINFER_BUILD_APPS=ON `
  -DBUILD_TESTING=ON

cmake --build build-windows --config Release --parallel
```

The applications are produced under `build-windows/apps/Release/`.

## Validate

Run the native test suite from the build directory:

```powershell
ctest --test-dir build-windows -C Release --output-on-failure
```

Tests that require a real `.ninfer` artifact are skipped unless their documented environment
variables are set. The build uses the static CUDA runtime on Windows and the shared CUDA runtime on
Linux; all project targets consume that choice through the common `ninfer_cuda_runtime` target.
