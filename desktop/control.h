#pragma once
#include "core.h"
#include <cstdint>

namespace ftcontrol {
// Loopback control plane for the LlamaCPP P100 desktop UI. Owns the llama-server
// child process, publishes every state change as JSON events over SSE, and serves
// the static frontend from the "ui" directory next to the executable.
class Host {
public:
    explicit Host(std::filesystem::path dataDir);
    ~Host();
    int run(uint16_t port, bool openBrowser);
    void shutdown();
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}
