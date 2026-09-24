#include "control.h"
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#ifdef _WIN32
#include <windows.h>
#endif

namespace {
std::filesystem::path default_data_dir() { return ft::data_directory() / "LlamaCppP100"; }
}

#ifdef _WIN32
static ftcontrol::Host* g_host = nullptr;
static BOOL WINAPI handle_ctrl(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_CLOSE_EVENT || type == CTRL_BREAK_EVENT) {
        if (g_host) g_host->shutdown();
        return FALSE;
    }
    return FALSE;
}
#endif

int main(int argc, char** argv) {
    uint16_t port = 18432;
    bool openBrowser = true;
    std::filesystem::path dataDir = default_data_dir();
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* err) -> std::string {
            if (i + 1 >= argc) { std::cerr << err << "\n"; exit(2); }
            return argv[++i];
        };
        if (arg == "--port") port = static_cast<uint16_t>(std::stoi(next("--port expects a number")));
        else if (arg == "--data-dir") dataDir = std::filesystem::u8path(next("--data-dir expects a path"));
        else if (arg == "--no-browser") openBrowser = false;
        else if (arg == "--help" || arg == "-h") {
            std::cout << "LlamaCPP P100 - local GUI host for llama-server\n"
                      << "  --port N        control-plane port (default 18432)\n"
                      << "  --data-dir DIR  settings and conversation storage\n"
                      << "  --no-browser    do not open the browser automatically\n";
            return 0;
        }
    }
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    try {
        std::filesystem::create_directories(dataDir);
        ftcontrol::Host host(dataDir);
#ifdef _WIN32
        g_host = &host;
        SetConsoleCtrlHandler(handle_ctrl, TRUE);
#endif
        return host.run(port, openBrowser);
    } catch (const std::exception& e) {
        std::cerr << "LlamaCPP P100 host: " << e.what() << "\n";
        return 1;
    }
}
