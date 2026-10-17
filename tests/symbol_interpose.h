#pragma once

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>

namespace ninfer::test {

// Called from a test executable's exported replacement; cache the result per symbol.
template <class Function>
Function next_symbol(const char* name) {
    dlerror();
    void* address     = dlsym(RTLD_NEXT, name);
    const char* error = dlerror();
    if (error || !address) {
        std::fprintf(stderr, "cannot resolve interposed function %s: %s\n", name,
                     error ? error : "null function address");
        std::abort();
    }
    return reinterpret_cast<Function>(address);
}

} // namespace ninfer::test
