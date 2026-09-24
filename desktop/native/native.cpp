// LlamaCPP P100 — native Slint UI controller. Reuses the desktop-core engine glue
// (settings registry, GGUF inspection, recommender, process supervisor) directly in
// process; no HTTP layer. Shares settings.json / conversation.json with the web host.
#include "core.h"
#include <httplib.h>
#include <atomic>
#include <chrono>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#ifdef _WIN32
#include <windows.h>
#endif

#include "app.h"

namespace {

using ft::json;

// ---------------- formatting ----------------
std::string fmt_gib(uint64_t mib) {
    char buf[64];
    if (mib >= 1024) snprintf(buf, sizeof buf, "%.1f GiB", mib / 1024.0);
    else snprintf(buf, sizeof buf, "%d MiB", (int)mib);
    return buf;
}
std::string fmt_pair(uint64_t usedMib, uint64_t totalMib) {
    char buf[96];
    snprintf(buf, sizeof buf, "%.1f / %.1f GiB", usedMib / 1024.0, totalMib / 1024.0);
    return buf;
}
std::string fmt_size(uint64_t mib) {
    char buf[64];
    if (mib >= 1024) snprintf(buf, sizeof buf, "%.1f GiB", mib / 1024.0);
    else snprintf(buf, sizeof buf, "%d MiB", (int)mib);
    return buf;
}
std::string fmt_uptime(long long s) {
    char buf[64];
    if (s < 60) snprintf(buf, sizeof buf, "%llds", s);
    else if (s < 3600) snprintf(buf, sizeof buf, "%lldm %llds", s / 60, s % 60);
    else snprintf(buf, sizeof buf, "%lldh %lldm", s / 3600, (s / 60) % 60);
    return buf;
}
// native v1 renders chat as plain text: strip markdown markers, keep code fences visible
std::string strip_markdown(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    bool inFence = false;
    for (size_t i = 0; i < in.size();) {
        if (in.compare(i, 3, "```") == 0) { inFence = !inFence; i += 3;
            while (i < in.size() && in[i] != '\n' && !(in[i] == '\r')) ++i;  // drop language tag
            out += inFence ? "\n--- code ---\n" : "\n-----------\n";
            continue; }
        if (!inFence && in.compare(i, 2, "**") == 0) { i += 2; continue; }
        if (!inFence && in[i] == '`') { ++i; continue; }
        out += in[i]; ++i;
    }
    // strip heading hashes at line starts
    std::string clean;
    for (size_t i = 0; i < out.size();) {
        if (i == 0 || out[i - 1] == '\n') {
            size_t j = i;
            while (j < out.size() && out[j] == '#') ++j;
            if (j > i && j < out.size() && out[j] == ' ') { i = j + 1; continue; }
        }
        clean += out[i++];
    }
    return clean;
}

// ---------------- telemetry (same as control plane) ----------------
struct RamInfo { uint64_t total = 0, avail = 0; };
RamInfo ram_info() {
#ifdef _WIN32
    MEMORYSTATUSEX m{}; m.dwLength = sizeof(m);
    RamInfo r;
    if (GlobalMemoryStatusEx(&m)) { r.total = m.ullTotalPhys; r.avail = m.ullAvailPhys; }
    return r;
#else
    RamInfo r; std::ifstream in("/proc/meminfo"); std::string key; uint64_t n; std::string unit;
    while (in >> key >> n >> unit) {
        if (key == "MemTotal:") r.total = n * 1024;
        if (key == "MemAvailable:") r.avail = n * 1024;
    }
    return r;
#endif
}
struct VramInfo { uint64_t total = 0, freeB = 0; bool known = false; std::string name; };
struct NvmlApi {
    bool ok = false;
#ifdef _WIN32
    HMODULE lib = nullptr;
    void* device = nullptr;
    int (*mem)(void*, void*) = nullptr;
    struct NvmlMemory { unsigned long long total, freeB, used; };
    NvmlApi() {
        lib = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!lib) return;
        using Init = int(*)(); using Handle = int(*)(unsigned, void**);
        auto init = reinterpret_cast<Init>(GetProcAddress(lib, "nvmlInit_v2"));
        auto get = reinterpret_cast<Handle>(GetProcAddress(lib, "nvmlDeviceGetHandleByIndex_v2"));
        mem = reinterpret_cast<int (*)(void*, void*)>(GetProcAddress(lib, "nvmlDeviceGetMemoryInfo"));
        if (init && get && mem && init() == 0 && get(0, &device) == 0) ok = true;
    }
#endif
};
VramInfo vram_info() {
    VramInfo v;
#ifdef _WIN32
    static NvmlApi api;
    if (!api.ok) return v;
    NvmlApi::NvmlMemory m{};
    if (api.mem(api.device, &m) == 0) { v.total = m.total; v.freeB = m.freeB; v.known = true; }
#endif
    return v;
}
void trim_memory() {
#ifdef _WIN32
    SetProcessWorkingSetSizeEx(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1, 0);
#endif
}

bool contains(const std::string& hay, const char* needle) { return hay.find(needle) != std::string::npos; }
slint::SharedString SS(const std::string& s) { return slint::SharedString(s.c_str()); }
std::string STR(const slint::SharedString& s) { return std::string(s.data(), s.size()); }

// ---------------- controller ----------------
struct Controller {
    std::filesystem::path data;
    json config = ft::defaults();
    ft::Hardware hardware;
    std::vector<ft::ModelInfo> library;
    int selectedIndex = -1;

    enum class EngineState { Stopped, Loading, Ready, Error };
    std::atomic<int> engine{0};          // 0 stopped 1 loading 2 ready 3 error
    std::string detail, loadStage;
    std::atomic<int> loadPercent{-1};
    std::chrono::steady_clock::time_point loadStarted, readyAt;
    std::atomic<bool> pendingReload{false}, generating{false}, cancelled{false}, closing{false}, scanning{false};
    int activePort = 0;
    std::string activeModel;
    ft::Process process;
    std::thread engineWorker, chatWorker, scanner;

    json conversation = json::array();
    // live streaming buffer
    std::mutex streamMutex;
    std::string streamContent, streamReasoning;
    double liveTokSec = 0;
    long long liveTtftMs = 0;
    std::string lastStatsLine;
    bool streamDirty = false;

    // logs
    std::mutex logMutex;
    std::deque<std::string> logLines;   // level folded into prefix flags via levelOf
    bool logsDirty = false;

    // stats
    std::atomic<double> lastTokSec{0};
    std::atomic<long long> lastTtftMs{0};
    std::atomic<int> requestsTotal{0};
    uint64_t tokensIn = 0, tokensOut = 0;
    int ctxUsed = 0, ctxTotal = 0;
    uint64_t ramTotal = 0, ramFree = 0, vramTotal = 0, vramFree = 0;
    bool vramKnown = false;
    std::string statusText = "Starting…";
    int statusLevel = 0;

    std::shared_ptr<httplib::Client> activeClient;
    std::mutex clientMutex;

    std::thread telemetry;
    AppWindow* ui = nullptr;   // owned by main; workers reach it only via the event loop

    uint64_t mib(uint64_t b) const { return b / (1024 * 1024); }

    // ---------- bootstrap (same conventions as the web host) ----------
    void bootstrap() {
#ifdef _WIN32
        {
            wchar_t* local = nullptr; size_t n = 0;
            _wdupenv_s(&local, &n, L"LOCALAPPDATA");
            if (local) {
                std::filesystem::path legacy = std::filesystem::path(local) / "FreeTokenDesktop";
                free(local);
                if (!std::filesystem::exists(data / "settings.json") && std::filesystem::exists(legacy / "settings.json")) {
                    std::error_code ec;
                    std::filesystem::copy_file(legacy / "settings.json", data / "settings.json", ec);
                    if (std::filesystem::exists(legacy / "conversation.json"))
                        std::filesystem::copy_file(legacy / "conversation.json", data / "conversation.json", ec);
                }
            }
        }
#endif
        hardware = ft::detect_hardware();
        try {
            auto saved = ft::read_json(data / "settings.json", json::object());
            for (auto it = saved.begin(); it != saved.end(); ++it)
                if (config.contains(it.key()) && it.value().type() == config[it.key()].type()) config[it.key()] = it.value();
        } catch (...) {}
        try { conversation = ft::read_json(data / "conversation.json", json::array()); } catch (...) { conversation = json::array(); }
        if (!conversation.is_array()) conversation = json::array();

        std::filesystem::path cwd = std::filesystem::current_path();
        std::filesystem::path exeDir = cwd;
#ifdef _WIN32
        wchar_t exe[32768]{};
        auto n = GetModuleFileNameW(nullptr, exe, 32768);
        if (n && n < 32768) exeDir = std::filesystem::path(exe).parent_path();
#endif
        std::vector<std::filesystem::path> roots = {cwd, exeDir.parent_path().parent_path(), exeDir.parent_path().parent_path().parent_path()};
#ifdef _WIN32
        std::filesystem::path serverRel = "build/ft-windows-cuda-dev/bin/llama-server.exe";
#else
        std::filesystem::path serverRel = "build/bin/llama-server";
#endif
        if (config.at("server") == "")
            for (auto& r : roots)
                if (std::filesystem::exists(r / serverRel)) { config["server"] = (r / serverRel).string(); break; }
        if (config["model_folder"] == "")
            for (auto& r : roots)
                if (std::filesystem::is_directory(r / "models")) { config["model_folder"] = (r / "models").string(); break; }
    }

    void save_config() { try { ft::save_json(data / "settings.json", config); } catch (...) {} }
    void save_conversation() { try { ft::save_json(data / "conversation.json", conversation); } catch (...) {} }

    // ---------- status / state push ----------
    const char* engine_state_name() const {
        return engine == 1 ? "loading" : engine == 2 ? "ready" : engine == 3 ? "error" : "stopped";
    }
    void set_status(const std::string& t, int level) {
        statusText = t; statusLevel = level;
        push_ui([=](AppWindow& w) { w.global<AppState>().set_status_text(SS(t)); w.global<AppState>().set_status_level(level); });
    }

    // Workers never touch the UI thread directly; all UI access goes through the event loop.
    // The raw pointer is safe: workers are joined before the window is destroyed, and queued
    // callbacks that never run simply never dereference it.
    template <typename F>
    void push_ui(F&& fn) {
        AppWindow* w = ui;
        slint::invoke_from_event_loop([w, fn]() mutable {
            if (w) fn(*w);
        });
    }

    // ---------- models ----------
    std::string caps_of(const ft::ModelInfo& m) const {
        std::string caps = "text";
        if (!m.projector.empty()) caps += " · vision";
        if (m.experts) caps += " · moe";
        return caps;
    }
    void refresh_models() {
        std::vector<ModelRow> rows;
        std::string details, detailsName;
        for (size_t i = 0; i < library.size(); ++i) {
            auto& m = library[i];
            std::string size = fmt_size(mib(m.bytes));
            std::string ctxLabel = m.context >= 1024 ? std::to_string(m.context / 1024) + "K" : std::to_string(m.context);
            rows.push_back({SS(m.name), SS(size), SS(caps_of(m)), config.value("model", std::string()) == m.path.string()});
            if ((int)i == selectedIndex) {
                detailsName = m.name;
                std::ostringstream s;
                s << "Family: " << (m.architecture.empty() ? "?" : m.architecture) << "\n"
                  << "Size on disk: " << size << "\n"
                  << "Layers: " << (m.layers ? std::to_string(m.layers) : "?") << "\n"
                  << "Context length: " << (m.context ? ctxLabel + " tokens" : "?") << "\n"
                  << "Experts: " << (m.experts ? std::to_string(m.experts) : std::string("dense")) << "\n"
                  << "Capabilities: " << caps_of(m) << "\n"
                  << "Projector: " << (m.projector.empty() ? std::string("none") : m.projector.filename().string());
                if (!m.valid) s << "\nError: " << m.error;
                details = s.str();
            }
        }
        push_ui([=](AppWindow& w) {
            w.set_models(std::make_shared<slint::VectorModel<ModelRow>>(rows));
            w.set_details_name(SS(detailsName));
            w.set_details(SS(details));
            w.global<AppState>().set_model_count((int)library.size());
        });
    }

    void scan_folder(const std::string& folder) {
        if (scanning) return;
        if (folder.empty()) { set_status("Choose a model folder first.", 3); return; }
        if (scanner.joinable()) scanner.join();
        scanning = true;
        set_status("Scanning model folder…", 2);
        scanner = std::thread([this, folder] {
            try {
                auto result = ft::scan_models(std::filesystem::u8path(folder));
                library = std::move(result);
                scanning = false;
                refresh_models();
                save_config();
                set_status(std::to_string(library.size()) + " models found. Select one to load it.", 1);
            } catch (const std::exception& e) {
                scanning = false;
                set_status(e.what(), 4);
            }
        });
    }

    void select_model(int i) {
        if (i < 0 || i >= (int)library.size()) { pair_projector(i, ""); return; }
        auto& m = library[i];
        if (!m.valid) { set_status(m.error, 4); return; }
        std::string prev = config.value("model", std::string());
        bool changed = prev != m.path.string();
        config["model"] = m.path.string();
        if (changed || config.value("mmproj", std::string()).empty()) config["mmproj"] = m.projector.string();
        if (changed || config.value("recommended", json::object()).empty()) {
            config["recommended"] = ft::recommend(m, hardware);
            config["recommended_model"] = config["model"];
            if (config.value("auto_settings", std::string("1")) == "1")
                for (auto it = config["recommended"].begin(); it != config["recommended"].end(); ++it)
                    config[it.key()] = it.value();
        }
        if (engine == 2 && changed) pendingReload = true;
        save_config();
        push_settings_to_ui();
        refresh_models();
        refresh_budget();
        set_status("Model selected. Review Tune or load when ready.", 1);
    }

    void pair_projector(int i, const std::string& path) {
        int idx = i >= 0 ? i : selectedIndex;
        if (idx < 0 || idx >= (int)library.size()) { set_status("Select a model first.", 3); return; }
        if (path.empty()) return;
        auto p = ft::inspect_model(std::filesystem::u8path(path));
        if (!p.valid || !p.is_projector) { set_status("Choose a readable vision projector GGUF.", 4); return; }
        library[idx].projector = p.path;
        if (config.value("model", std::string()) == library[idx].path.string()) config["mmproj"] = p.path.string();
        save_config();
        refresh_models();
        set_status("Projector paired. Reload the model to enable vision.", 1);
    }

    void add_model(const std::string& path) {
        if (path.empty()) return;
        auto m = ft::inspect_model(std::filesystem::u8path(path));
        if (!m.valid || m.is_projector) { set_status("Choose a valid language-model GGUF, not a projector.", 4); return; }
        for (auto& e : library)
            if (e.path.string() == m.path.string()) { set_status("Already in the library.", 3); return; }
        library.push_back(m);
        save_config();
        refresh_models();
        set_status("Model added.", 1);
    }

    // ---------- startup review ----------
    struct BudgetRowData { std::string label, value; int kind; };
    std::vector<BudgetRowData> budget_rows;
    std::vector<std::string> budget_warnings;
    void refresh_budget() {
        budget_rows.clear();
        budget_warnings.clear();
        try {
            auto modelPath = std::filesystem::u8path(config.value("model", std::string()));
            if (!std::filesystem::is_regular_file(modelPath)) {
                budget_warnings.push_back("Select a model in Models to see a memory plan.");
            } else {
                auto m = ft::inspect_model(modelPath);
                long long ngl = std::stoll(config.value("gpu_layers", std::string("0")));
                long long layers = std::max<long long>(1, (long long)m.layers);
                long long gpuLayers = std::clamp(ngl >= 999 ? layers : ngl, 0LL, layers);
                double gpuFraction = (double)gpuLayers / layers;
                auto value = [&](const char* key, uint64_t fallback) {
                    try { return (uint64_t)std::stoull(config.value(key, std::string())); }
                    catch (...) { return fallback; }
                };
                uint64_t gpuCache = value("gpu_cache", 0), cpuCache = value("cpu_cache", 0), reserve = value("reserve", 512);
                bool expertOn = config.value("expert_enabled", std::string("1")) == "1";
                double bytesPerKv = 4.0;
                if (config.value("cache_k", std::string("f16")) != "f16" || config.value("cache_v", std::string("f16")) != "f16") bytesPerKv = 2.2;
                double headDim = m.heads ? (double)m.embedding / (double)m.heads : 128.0;
                double kvPerToken = (double)m.layers * (m.kv_heads ? m.kv_heads : m.heads) * headDim * bytesPerKv;
                long long ctx = 0;
                try { ctx = std::stoll(config.value("ctx", std::string("4096"))); } catch (...) {}
                uint64_t kvTotal = (uint64_t)(kvPerToken * (double)ctx);
                uint64_t weightsGpu = (uint64_t)((double)m.bytes * gpuFraction);
                uint64_t weightsCpu = m.bytes - weightsGpu;
                bool kvOnGpu = gpuLayers > 0 && config.value("no_kv", std::string("0")) != "1";
                auto add = [&](const char* label, uint64_t b, int kind) {
                    if (mib(b) == 0) return;
                    budget_rows.push_back({label, fmt_gib(mib(b)), kind});
                };
                add("Model weights on GPU", weightsGpu, 0);
                if (expertOn && gpuCache) add("Expert cache on GPU", gpuCache * 1024 * 1024, 1);
                add(kvOnGpu ? "KV cache on GPU" : "KV cache on RAM", kvTotal, 2);
                if (gpuLayers > 0) add("Scratch and reserve", reserve * 1024 * 1024, 3);
                add("Model weights on RAM", weightsCpu, 0);
                if (expertOn && cpuCache) add("Expert cache on RAM", cpuCache * 1024 * 1024, 1);
                budget_warnings.push_back("Estimates only - not calibrated. Actual usage depends on backend scratch, CUDA graphs and other workloads.");
                if (config.value("graphs", std::string("auto")) == "auto" && contains(hardware.gpu, "P100"))
                    budget_warnings.push_back("P100/SM60: the current backend guard disables CUDA graph replay.");
            }
        } catch (const std::exception& e) {
            budget_warnings.push_back(e.what());
        }
        std::vector<BudgetRow> rows;
        for (auto& r : budget_rows) rows.push_back({SS(r.label), SS(r.value), r.kind});
        std::vector<slint::SharedString> warns;
        for (auto& w : budget_warnings) warns.push_back(SS(w));
        push_ui([=](AppWindow& w) {
            w.set_budget(std::make_shared<slint::VectorModel<BudgetRow>>(rows));
            w.set_warning_list(std::make_shared<slint::VectorModel<slint::SharedString>>(warns));
        });
    }

    // ---------- engine ----------
    void set_engine(int s, const std::string& stage, int percent) {
        engine = s;
        loadStage = stage;
        loadPercent = percent;
        push_status_from_state();
    }
    void push_status_from_state() {
        std::string text;
        int level = 0;
        int e = engine;
        if (e == 1) { text = "Loading model — " + loadStage + " (" + std::to_string(std::max(0, (int)loadPercent)) + "%)"; level = 2; }
        else if (e == 2) text = pendingReload ? "Ready — settings changed, reload to apply" : "Local inference ready";
        else if (e == 3) { text = "Engine error — " + detail; level = 4; }
        else text = "Model not loaded";
        if (e == 2) level = pendingReload ? 3 : 1;
        set_status(text, level);
        int s = e;
        std::string stage = loadStage;
        int pct = loadPercent;
        bool pr = pendingReload;
        push_ui([=](AppWindow& w) {
            auto& st = w.global<AppState>();
            st.set_engine_state(s);
            st.set_load_stage(SS(stage));
            st.set_load_percent(std::max(0, pct));
            st.set_pending_reload(pr);
        });
    }

    void on_server_line(const std::string& line) {
        {
            std::lock_guard<std::mutex> g(logMutex);
            logLines.push_back(line);
            if (logLines.size() > 800) logLines.pop_front();
            logsDirty = true;
        }
        const char* stage = nullptr;
        int pct = -1;
        if (contains(line, "llama_model_loader:")) { stage = "Reading model file"; pct = 5; }
        if (contains(line, "load_tensors: loading model tensors")) { stage = "Loading tensors"; pct = 15; }
        if (contains(line, "load_tensors: offloaded")) { stage = "Placing layers on GPU"; pct = 35; }
        if (contains(line, "llama_context:") || contains(line, "warmup")) { stage = "Allocating context and warmup"; pct = 70; }
        if (contains(line, "server is listening") || contains(line, "listening on")) { stage = "Starting API"; pct = 95; }
        if (stage) set_engine(1, stage, pct);
    }

    void start_engine() {
        if (generating) { set_status("Stop the current response before reloading.", 3); return; }
        if (engineWorker.joinable()) engineWorker.join();
        if (config.value("model", std::string()).empty()) { set_status("Select a model first (Models page).", 3); return; }
        try { ft::validate(config); } catch (const std::exception& e) { set_status(e.what(), 4); return; }
        json c = config;
        cancelled = false;
        loadStarted = std::chrono::steady_clock::now();
        set_engine(1, "Preparing", 0);
        engineWorker = std::thread([this, c] {
            try {
                auto args = ft::arguments(c);
                args.push_back("--metrics");
                args.push_back("--slots");
                int port = std::stoi(c.at("port").get<std::string>());
                httplib::Server probe;
                if (!probe.bind_to_port("127.0.0.1", port)) throw std::runtime_error("Port is already in use. Change it in the web UI settings or free the port.");
                probe.stop();
                activeModel = std::filesystem::u8path(c.value("model", std::string())).filename().string();
                process.start(args, data / "server.log", c.value("graphs", std::string("auto")) == "off",
                              [this](const std::string& line) { on_server_line(line); });
                auto client = std::make_shared<httplib::Client>("127.0.0.1", port);
                client->set_connection_timeout(1, 0);
                client->set_read_timeout(1, 0);
                { std::lock_guard<std::mutex> g(clientMutex); activeClient = client; }
                bool healthy = false;
                auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(5);
                while (!cancelled && std::chrono::steady_clock::now() < deadline) {
                    if (!process.running()) throw std::runtime_error("llama-server exited during startup. Check Monitor for the log.");
                    auto res = client->Get("/health");
                    if (res && res->status == 200) { healthy = true; break; }
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                }
                if (cancelled || !healthy) {
                    process.stop();
                    activePort = 0;
                    detail = cancelled ? "Startup cancelled." : "Startup timed out. Check Monitor for the log.";
                    set_engine(3, "", -1);
                    return;
                }
                activePort = port;
                readyAt = std::chrono::steady_clock::now();
                pendingReload = false;
                set_engine(2, "", 100);
                trim_memory();
                set_status("Model loaded on port " + std::to_string(port) + ".", 1);
            } catch (const std::exception& e) {
                process.stop();
                activePort = 0;
                detail = e.what();
                set_engine(3, "", -1);
            }
        });
    }

    void unload_engine() {
        if (generating) cancel_chat();
        if (engineWorker.joinable()) engineWorker.join();
        process.stop();
        activePort = 0;
        set_engine(0, "", -1);
        set_status("Model unloaded. Conversation kept.", 0);
    }

    // ---------- chat ----------
    void rebuild_messages_model() {
        std::vector<MsgRow> rows;
        for (auto& m : conversation) {
            try {
                std::string role = m.at("role").get<std::string>();
                std::string content = strip_markdown(m.value("content", std::string()));
                std::string reasoning = m.value("reasoning", std::string());
                rows.push_back({role == "user" ? 0 : 1, SS(content), SS(reasoning), ""});
            } catch (...) {}
        }
        if (generating) {
            std::lock_guard<std::mutex> g(streamMutex);
            std::string stats = lastStatsLine;
            rows.push_back({1, SS(strip_markdown(streamContent)), SS(streamReasoning), SS(stats)});
        }
        push_ui([=](AppWindow& w) { w.set_messages(std::make_shared<slint::VectorModel<MsgRow>>(rows)); });
    }

    void run_chat(const std::string& user) {
        if (chatWorker.joinable()) chatWorker.join();
        if (engine != 2) { set_status("Load a model first (Models page).", 3); return; }
        if (pendingReload) { set_status("Settings changed — reload the model first.", 3); return; }
        if (generating) { set_status("Still responding — stop first.", 3); return; }
        json requestMessages = conversation;
        {
            std::lock_guard<std::mutex> g(configMutex);
            if (config.value("system", std::string()) != "")
                requestMessages.insert(requestMessages.begin(), json{{"role", "system"}, {"content", config.value("system", std::string())}});
        }
        double temperature = 0.7, topP = 0.95, minP = 0.05, repeatPenalty = 1.0;
        int maxTokens = 512, topK = 20;
        std::string thinking = "0";
        {
            std::lock_guard<std::mutex> g(configMutex);
            auto d = [](const json& j, const char* k, double f) { try { return std::stod(j.at(k).get<std::string>()); } catch (...) { return f; } };
            auto i = [](const json& j, const char* k, int f) { try { return std::stoi(j.at(k).get<std::string>()); } catch (...) { return f; } };
            temperature = d(config, "temperature", 0.7); topP = d(config, "top_p", 0.95); minP = d(config, "min_p", 0.05);
            repeatPenalty = d(config, "repeat_penalty", 1.0); maxTokens = i(config, "max_tokens", 512); topK = i(config, "top_k", 20);
            thinking = config.value("thinking", std::string("0"));
        }
        conversation.push_back(json{{"role", "user"}, {"content", user}});
        save_conversation();
        cancelled = false;
        generating = true;
        {
            std::lock_guard<std::mutex> g(streamMutex);
            streamContent.clear(); streamReasoning.clear();
            lastStatsLine.clear();
            streamDirty = true;
        }
        push_status_from_state();
        rebuild_messages_model();
        json body = {{"model", "local"}, {"messages", requestMessages}, {"stream", true},
                     {"temperature", temperature}, {"max_tokens", maxTokens},
                     {"chat_template_kwargs", json{{"enable_thinking", thinking == "1"}}},
                     {"top_k", topK}, {"top_p", topP}, {"min_p", minP}, {"repeat_penalty", repeatPenalty},
                     {"timings_per_token", true}, {"stream_options", json{{"include_usage", true}}}};
        int port = activePort;
        chatWorker = std::thread([this, body, port] {
            auto started = std::chrono::steady_clock::now();
            std::string answer, reasoning, stopReason = "eos";
            bool firstToken = true;
            long long ttftMs = 0;
            json usage;
            try {
                auto client = std::make_shared<httplib::Client>("127.0.0.1", port);
                client->set_connection_timeout(2, 0);
                client->set_read_timeout(300, 0);
                { std::lock_guard<std::mutex> g(clientMutex); activeClient = client; }
                size_t total = 0;
                ft::Events events([&](const json& event) {
                    if (event.contains("error")) throw std::runtime_error(event["error"].dump());
                    if (cancelled) throw std::runtime_error("__cancelled__");
                    if (event.contains("usage") && event["usage"].is_object()) usage = event["usage"];
                    if (!event.contains("choices") || !event["choices"].is_array() || event["choices"].empty()) {
                        if (event.contains("timings")) {
                            auto& t = event["timings"];
                            double tokSec = t.value("predicted_per_second", 0.0);
                            lastTokSec = tokSec;
                            std::lock_guard<std::mutex> g(streamMutex);
                            liveTokSec = tokSec;
                        }
                        return;
                    }
                    auto delta = event["choices"][0].value("delta", json::object());
                    if (firstToken) {
                        firstToken = false;
                        ttftMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - started).count();
                        lastTtftMs = ttftMs;
                    }
                    if (delta.contains("reasoning_content") && delta["reasoning_content"].is_string()) {
                        auto text = delta["reasoning_content"].get<std::string>();
                        total += text.size(); reasoning += text;
                        if (total > 1024 * 1024) throw std::runtime_error("Response limit reached (1 MiB).");
                        std::lock_guard<std::mutex> g(streamMutex);
                        streamReasoning += text;
                        streamDirty = true;
                    }
                    if (delta.contains("content") && delta["content"].is_string()) {
                        auto text = delta["content"].get<std::string>();
                        total += text.size(); answer += text;
                        if (total > 1024 * 1024) throw std::runtime_error("Response limit reached (1 MiB).");
                        std::lock_guard<std::mutex> g(streamMutex);
                        streamContent += text;
                        streamDirty = true;
                    }
                    if (event.contains("timings")) {
                        auto& t = event["timings"];
                        std::lock_guard<std::mutex> g(streamMutex);
                        liveTokSec = t.value("predicted_per_second", 0.0);
                        char buf[64];
                        snprintf(buf, sizeof buf, "%.1f tok/s · TTFT %lld ms · %d tokens", liveTokSec, ttftMs, (int)t.value("predicted_n", 0));
                        lastStatsLine = buf;
                    }
                });
                httplib::Request request;
                request.method = "POST";
                request.path = "/v1/chat/completions";
                request.body = body.dump();
                request.set_header("Content-Type", "application/json");
                int httpStatus = 0;
                std::string errorBody;
                request.response_handler = [&](const httplib::Response& r) { httpStatus = r.status; return true; };
                request.content_receiver = [&](const char* bytes, size_t n, uint64_t, uint64_t) {
                    if (cancelled) return false;
                    if (httpStatus != 200) { if (errorBody.size() < 8192) errorBody.append(bytes, std::min(n, size_t(8192 - errorBody.size()))); }
                    else events.feed(bytes, n);
                    return true;
                };
                auto response = client->send(request);
                if (cancelled) stopReason = "cancelled";
                else if (!response) { stopReason = "error"; throw std::runtime_error("Request failed: " + httplib::to_string(response.error())); }
                else if (httpStatus != 200) { stopReason = "error"; throw std::runtime_error("HTTP " + std::to_string(httpStatus) + ": " + errorBody); }
                else if (!events.done) stopReason = "truncated";
                if (usage.is_object()) {
                    tokensIn += usage.value("prompt_tokens", 0);
                    tokensOut += usage.value("completion_tokens", 0);
                    requestsTotal += 1;
                }
            } catch (const std::exception& e) {
                if (std::string(e.what()) == "__cancelled__") stopReason = "cancelled";
                else stopReason = "error";
                set_status(std::string("Response stopped: ") + (stopReason == "error" ? e.what() : "you pressed Stop."), stopReason == "error" ? 4 : 0);
            }
            {
                std::lock_guard<std::mutex> g(streamMutex);
                streamContent = answer;
                streamReasoning = reasoning;
                char buf[96];
                snprintf(buf, sizeof buf, "%.1f tok/s · TTFT %lld ms · stopped by you", lastTokSec.load(), ttftMs);
                lastStatsLine = stopReason == "cancelled" ? buf : lastStatsLine;
                streamDirty = true;
            }
            json msg{{"role", "assistant"}, {"content", answer}};
            if (!reasoning.empty()) msg["reasoning"] = reasoning;
            conversation.push_back(msg);
            save_conversation();
            generating = false;
            push_status_from_state();
        });
    }

    void cancel_chat() {
        cancelled = true;
        std::lock_guard<std::mutex> g(clientMutex);
        if (activeClient) activeClient->stop();
    }

    void new_chat() {
        cancel_chat();
        if (chatWorker.joinable()) chatWorker.join();
        conversation = json::array();
        save_conversation();
        rebuild_messages_model();
        set_status("New conversation.", 0);
    }

    // ---------- settings ----------
    std::mutex configMutex;
    std::string pendingKey;
    std::string pendingValue;
    std::chrono::steady_clock::time_point pendingAt;

    void queue_setting(const std::string& key, const std::string& value) {
        std::lock_guard<std::mutex> g(configMutex);
        pendingKey = key;
        pendingValue = value;
        pendingAt = std::chrono::steady_clock::now();
    }
    void flush_settings() {
        std::string key, value;
        {
            std::lock_guard<std::mutex> g(configMutex);
            if (pendingKey.empty()) return;
            if (std::chrono::steady_clock::now() - pendingAt < std::chrono::milliseconds(600)) return;
            key = pendingKey;
            value = pendingValue;
            pendingKey.clear();
        }
        json next = config;
        if (next.contains(key) && next[key].is_string()) next[key] = value;
        try {
            ft::validate(next);
            bool wasPending = pendingReload;
            config = next;
            save_config();
            bool needsReload = false;
            for (auto& f : ft::settings())
                if (f.key == key) needsReload = true;
            if (key == "model" || key == "mmproj" || key == "graphs") needsReload = true;
            if (engine == 2 && needsReload) pendingReload = true;
            if (needsReload != wasPending) push_status_from_state();
            if (needsReload) set_status("Saved. Reload the model to apply.", 3);
            refresh_budget();
        } catch (const std::exception& e) {
            set_status(key + ": " + e.what(), 4);
            push_settings_to_ui();  // revert the edit box to the last valid value
        }
    }

    // ---------- push settings values into the UI ----------
    void push_settings_to_ui() {
        std::lock_guard<std::mutex> g(configMutex);
        json c = config;
        auto idx = [](const json& j, const char* k, const std::vector<std::string>& opts) {
            std::string v = j.value(k, std::string(opts.empty() ? "" : opts[0]));
            for (size_t i = 0; i < opts.size(); ++i)
                if (opts[i] == v) return (int)i;
            return 0;
        };
        push_ui([=](AppWindow& w) {
            w.set_v_ctx(SS(c.value("ctx", "4096")));
            w.set_v_gpu(SS(c.value("gpu_layers", "0")));
            w.set_v_threads(SS(c.value("threads", "4")));
            w.set_v_tbatch(SS(c.value("threads_batch", "4")));
            w.set_v_batch(SS(c.value("batch", "512")));
            w.set_v_ubatch(SS(c.value("ubatch", "128")));
            w.set_i_flash(idx(c, "flash", {"auto", "on", "off"}));
            w.set_i_ck(idx(c, "cache_k", {"f16", "q8_0", "q4_0"}));
            w.set_i_cv(idx(c, "cache_v", {"f16", "q8_0", "q4_0"}));
            w.set_v_temp(SS(c.value("temperature", "0.7")));
            w.set_v_maxtok(SS(c.value("max_tokens", "512")));
            w.set_v_topk(SS(c.value("top_k", "20")));
            w.set_v_topp(SS(c.value("top_p", "0.95")));
            w.set_v_minp(SS(c.value("min_p", "0.05")));
            w.set_v_rep(SS(c.value("repeat_penalty", "1")));
            w.set_v_system(SS(c.value("system", "")));
            w.set_b_think(c.value("thinking", "0") == "1");
            w.set_b_auto(c.value("auto_settings", "1") == "1");
            w.set_v_gpucache(SS(c.value("gpu_cache", "")));
            w.set_v_cpucache(SS(c.value("cpu_cache", "")));
            w.set_v_reserve(SS(c.value("reserve", "512")));
        });
    }

    // ---------- restore profile ----------
    void restore_profile(bool detect) {
        std::string modelId = config.value("model", std::string());
        if (modelId.empty()) { set_status("Select a model before detecting settings.", 3); return; }
        auto m = ft::inspect_model(std::filesystem::u8path(modelId));
        if (!m.valid) { set_status("The selected model file could not be read.", 4); return; }
        if (detect) hardware = ft::detect_hardware();
        config["recommended"] = ft::recommend(m, hardware);
        config["recommended_model"] = modelId;
        for (auto it = config["recommended"].begin(); it != config["recommended"].end(); ++it)
            config[it.key()] = it.value();
        save_config();
        push_settings_to_ui();
        refresh_budget();
        set_status(detect ? "Hardware detected — recommended profile applied." : "Recommended profile restored.", 1);
        if (engine == 2) { pendingReload = true; push_status_from_state(); }
    }

    // ---------- telemetry + throttled UI flush loop ----------
    void telemetry_loop() {
        std::shared_ptr<httplib::Client> slotsClient;
        int cachedPort = 0;
        int tick = 0;
        while (!closing) {
            ++tick;
            // flush dirty chat stream every tick (250 ms)
            bool dirty = false;
            { std::lock_guard<std::mutex> g(streamMutex); dirty = streamDirty; if (dirty) streamDirty = false; }
            if (dirty) rebuild_messages_model();
            // flush logs every 2 ticks (500 ms)
            if (tick % 2 == 0) {
                bool ld = false;
                { std::lock_guard<std::mutex> g(logMutex); ld = logsDirty; if (ld) logsDirty = false; }
                if (ld) {
                    std::vector<LogRow> rows;
                    {
                        std::lock_guard<std::mutex> g(logMutex);
                        for (auto& l : logLines) {
                            int level = contains(l, "ERR") || contains(l, "error") ? 2 : contains(l, "WRN") || contains(l, "warn") ? 1 : 0;
                            rows.push_back({SS(l), level});
                        }
                    }
                    push_ui([=](AppWindow& w) { w.set_logs(std::make_shared<slint::VectorModel<LogRow>>(rows)); });
                }
            }
            // telemetry every 4 ticks (1 s); 16 ticks (4 s) when idle and engine stopped
            bool busy = generating || engine != 0;
            if (tick % (busy ? 4 : 16) == 0) {
                auto ram = ram_info();
                auto vram = vram_info();
                ramTotal = mib(ram.total); ramFree = mib(ram.avail);
                if (vram.known) { vramTotal = mib(vram.total); vramFree = mib(vram.freeB); vramKnown = true; }
                if (engine == 2 && activePort) {
                    if (!slotsClient || cachedPort != activePort) {
                        slotsClient = std::make_shared<httplib::Client>("127.0.0.1", activePort);
                        slotsClient->set_connection_timeout(0, 300000);
                        slotsClient->set_read_timeout(0, 300000);
                        cachedPort = activePort;
                    }
                    auto res = slotsClient->Get("/slots");
                    if (res && res->status == 200) {
                        try {
                            auto slots = json::parse(res->body);
                            if (slots.is_array() && !slots.empty()) {
                                ctxUsed = slots[0].value("n_prompt_tokens", 0);
                                ctxTotal = slots[0].value("n_ctx", 0);
                            }
                        } catch (...) {}
                    }
                }
                long long up = engine == 2 ? std::chrono::duration_cast<std::chrono::seconds>(
                                                  std::chrono::steady_clock::now() - readyAt).count() : 0;
                uint64_t vu = vramKnown ? vramTotal - vramFree : 0;
                uint64_t ru = ramTotal - ramFree;
                char tokBuf[32] = "-", ttftBuf[32] = "-", ctxBuf[64] = "-";
                if (lastTokSec > 0) snprintf(tokBuf, sizeof tokBuf, "%.1f", lastTokSec.load());
                if (lastTtftMs > 0) snprintf(ttftBuf, sizeof ttftBuf, "%lld ms", lastTtftMs.load());
                if (ctxTotal > 0) snprintf(ctxBuf, sizeof ctxBuf, "%d / %d", ctxUsed, ctxTotal);
                std::string chipVram = vramKnown ? fmt_pair(vu, vramTotal) : "n/a";
                std::string chipRam = fmt_pair(ru, ramTotal);
                std::string tokText = tokBuf, ttftText = ttftBuf, upText = fmt_uptime(up);
                std::string ctxText = ctxBuf;
                std::string am = activeModel;
                int req = requestsTotal;
                int vp = vramKnown && vramTotal ? (int)(vu * 100 / vramTotal) : 0;
                int rp = ramTotal ? (int)(ru * 100 / ramTotal) : 0;
                push_ui([=](AppWindow& w) {
                    auto& st = w.global<AppState>();
                    st.set_chip_vram(SS(chipVram));
                    st.set_chip_ram(SS(chipRam));
                    st.set_vram_pct(vp);
                    st.set_ram_pct(rp);
                    st.set_tok_sec_text(SS(tokText));
                    st.set_ttft_text(SS(ttftText));
                    st.set_ctx_text(SS(ctxText));
                    st.set_requests(req);
                    st.set_uptime_text(SS(upText));
                    st.set_active_model(SS(am));
                });
            }
            flush_settings();
            for (int i = 0; i < 25 && !closing; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    // ---------- wire UI ----------
    void connect(AppWindow& w) {
        ui = &w;
        
        w.global<AppState>().set_page(0);
        w.on_send([this](slint::SharedString t) {
            std::string text = STR(t);
            if (text.empty()) return;
            w_set_draft("");
            run_chat(text);
        });
        w.on_cancel_chat([this] { cancel_chat(); });
        w.on_new_chat([this] { new_chat(); });
        w.on_load_engine([this] { start_engine(); });
        w.on_unload_engine([this] { unload_engine(); });
        w.on_rescan([this](slint::SharedString f) {
            std::string folder = STR(f);
            if (!folder.empty()) config["model_folder"] = folder;
            save_config();
            scan_folder(folder);
        });
        w.on_use_model([this](int i) { select_model(i); });
        w.on_pair_projector([this](int i, slint::SharedString p) { pair_projector(i, STR(p)); });
        w.on_add_model([this](slint::SharedString p) { add_model(STR(p)); });
        w.on_set_key([this](slint::SharedString k, slint::SharedString v) { queue_setting(STR(k), STR(v)); });
        w.on_restore_profile([this](bool detect) { restore_profile(detect); });
        w.on_goto_page([this](int p) {
            ui->global<AppState>().set_page(p);

        });
    }
    void w_set_draft(const std::string& s) {
        push_ui([=](AppWindow& w) { w.set_draft(SS(s)); });
    }
};

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    _putenv("SLINT_STYLE=fluent-dark");
#endif
    std::filesystem::path dataDir = ft::data_directory() / "LlamaCppP100";
    std::filesystem::create_directories(dataDir);

    auto ui = AppWindow::create();
    

    Controller c;
    c.data = dataDir;
    c.bootstrap();
    c.connect(*ui);
    c.push_settings_to_ui();
    c.refresh_models();
    c.refresh_budget();
    c.rebuild_messages_model();
    c.push_status_from_state();
    c.telemetry = std::thread([&c] { c.telemetry_loop(); });
    if (!c.config.value("model_folder", std::string()).empty()) c.scan_folder(c.config.value("model_folder", std::string()));
    trim_memory();

    std::cout << "LlamaCPP P100 native UI (Slint). Data: " << dataDir.string() << "\n";
    ui->run();

    c.closing = true;
    c.cancel_chat();
    if (c.chatWorker.joinable()) c.chatWorker.join();
    if (c.engineWorker.joinable()) c.engineWorker.join();
    if (c.scanner.joinable()) c.scanner.join();
    if (c.telemetry.joinable()) c.telemetry.join();
    c.process.stop();
    c.save_config();
    return 0;
}
