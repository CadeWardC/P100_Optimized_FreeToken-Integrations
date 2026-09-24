#include "control.h"
#include <httplib.h>
#include <algorithm>
#include <condition_variable>
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#include <shobjidl.h>
#include <psapi.h>
#else
#include <unistd.h>
#endif

namespace ftcontrol {
using ft::json;

namespace {
constexpr size_t kLogRingMax = 6000;
constexpr size_t kQueueMax = 512;

// Each HTTP worker initializes its own apartment for the native folder dialog.
std::string browse_model_folder(const std::string& initial) {
#ifdef _WIN32
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    if (FAILED(hr)) throw std::runtime_error("Could not initialize the folder picker.");
    struct Apartment { ~Apartment() { CoUninitialize(); } } apartment;
    IFileOpenDialog* raw = nullptr;
    hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&raw));
    if (FAILED(hr)) throw std::runtime_error("Could not open the folder picker.");
    auto releaseDialog = [](IFileOpenDialog* p) { p->Release(); };
    std::unique_ptr<IFileOpenDialog, decltype(releaseDialog)> dialog(raw, releaseDialog);
    DWORD options = 0;
    hr = dialog->GetOptions(&options);
    if (SUCCEEDED(hr)) hr = dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST | FOS_NOCHANGEDIR);
    if (FAILED(hr)) throw std::runtime_error("Could not configure the folder picker.");
    dialog->SetTitle(L"Choose your model folder");
    if (!initial.empty()) {
        IShellItem* start = nullptr;
        auto path = std::filesystem::absolute(std::filesystem::u8path(initial)).wstring();
        if (SUCCEEDED(SHCreateItemFromParsingName(path.c_str(), nullptr, IID_PPV_ARGS(&start)))) {
            dialog->SetFolder(start);
            start->Release();
        }
    }
    hr = dialog->Show(nullptr);
    if (hr == HRESULT_FROM_WIN32(ERROR_CANCELLED)) return {};
    if (FAILED(hr)) throw std::runtime_error("Could not display the folder picker.");
    IShellItem* item = nullptr;
    hr = dialog->GetResult(&item);
    if (FAILED(hr)) throw std::runtime_error("Could not read the selected folder.");
    PWSTR path = nullptr;
    hr = item->GetDisplayName(SIGDN_FILESYSPATH, &path);
    item->Release();
    if (FAILED(hr)) throw std::runtime_error("Choose a local filesystem folder.");
    auto freePath = [](wchar_t* p) { CoTaskMemFree(p); };
    std::unique_ptr<wchar_t, decltype(freePath)> selected(path, freePath);
    return std::filesystem::path(path).u8string();
#else
    throw std::runtime_error("Folder browsing is available on Windows. Paste a folder path instead.");
#endif
}

uint64_t mib(uint64_t bytes) { return bytes / (1024 * 1024); }

struct ProcMem { uint64_t workingSetMib = 0, privateMib = 0; };
ProcMem current_proc_mem() {
    ProcMem p;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        p.workingSetMib = mib(pmc.WorkingSetSize);
        p.privateMib = mib(pmc.PrivateUsage);
    }
#endif
    return p;
}

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
// NVML is loaded once and kept for the process lifetime; per-second load/free cycles
// churn the loader and the NVML client's own allocations.
VramInfo vram_info() {
    VramInfo v;
#ifdef _WIN32
    struct NvmlApi {
        bool ok = false;
        HMODULE lib = nullptr;
        void* device = nullptr;
        int (*mem)(void*, void*) = nullptr;
        int (*name)(void*, char*, unsigned) = nullptr;
        struct NvmlMemory { unsigned long long total, freeB, used; };
        NvmlApi() {
            lib = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
            if (!lib) return;
            using Init = int(*)(); using Handle = int(*)(unsigned, void**);
            auto init = reinterpret_cast<Init>(GetProcAddress(lib, "nvmlInit_v2"));
            auto get = reinterpret_cast<Handle>(GetProcAddress(lib, "nvmlDeviceGetHandleByIndex_v2"));
            mem = reinterpret_cast<int (*)(void*, void*)>(GetProcAddress(lib, "nvmlDeviceGetMemoryInfo"));
            name = reinterpret_cast<int (*)(void*, char*, unsigned)>(GetProcAddress(lib, "nvmlDeviceGetName"));
            if (init && get && mem && init() == 0 && get(0, &device) == 0) ok = true;
        }
    };
    static NvmlApi api;
    if (!api.ok) return v;
    NvmlApi::NvmlMemory m{};
    if (api.mem(api.device, &m) == 0) { v.total = m.total; v.freeB = m.freeB; v.known = true; }
    if (api.name) {
        char buffer[96]{};
        if (api.name(api.device, buffer, sizeof buffer) == 0) v.name = buffer;
    }
#endif
    return v;
}

void trim_memory() {
#ifdef _WIN32
    // tell Windows to move everything not currently touched out of the working set
    SetProcessWorkingSetSizeEx(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1, 0);
#endif
}

bool contains(const std::string& hay, const char* needle) { return hay.find(needle) != std::string::npos; }
}  // namespace

struct Host::Impl {
    std::filesystem::path data;
    json config = ft::defaults();
    std::mutex cfgMutex;
    ft::Hardware hardware;
    std::vector<ft::ModelInfo> library;
    std::thread scanner;
    std::vector<ft::ModelInfo> scanned;
    bool scanning = false;

    enum class EngineState { Stopped, Loading, Ready, Error };
    std::atomic<EngineState> engine{EngineState::Stopped};
    std::string detail, loadStage;
    std::chrono::steady_clock::time_point loadStarted;
    std::atomic<long long> loadPercent{-1};
    int activePort = 0;
    std::string activeModel, activeMmproj;
    std::chrono::steady_clock::time_point readyAt;
    bool pendingReload = false;
    std::mutex stateMutex;

    ft::Process process;
    std::thread engineWorker, chatWorker;
    std::mutex engineLifecycleMutex;
    std::atomic<bool> cancelled{false}, closing{false}, generating{false};
    std::mutex clientMutex;
    std::shared_ptr<httplib::Client> activeClient;

    json conversation = json::array();
    json chats = json::array();
    std::string activeChat = "initial";
    std::mutex chatOperationMutex;

    // log ring
    std::mutex logMutex;
    std::deque<json> logRing;
    uint64_t logSeq = 0;

    // live stats
    std::mutex statsMutex;
    uint64_t tokensIn = 0, tokensOut = 0, requestsTotal = 0;
    double lastTokSec = 0, lastTtftMs = 0;
    int ctxUsed = 0, ctxTotal = 0;
    uint64_t ramTotalMib = 0, ramFreeMib = 0, vramTotalMib = 0, vramFreeMib = 0;
    std::thread telemetry;

    // event bus
    struct Sub {
        std::deque<json> q;
        std::mutex m;
    };
    std::mutex busMutex;
    std::map<int, std::shared_ptr<Sub>> subs;
    int nextSub = 1;

    httplib::Server server;

    void publish(json ev) {
        std::lock_guard<std::mutex> guard(busMutex);
        for (auto& [id, sub] : subs) {
            std::lock_guard<std::mutex> g(sub->m);
            if (sub->q.size() >= kQueueMax) sub->q.pop_front();
            sub->q.push_back(ev);
        }
    }

    void log_line(const std::string& line) {
        json rec{{"type", "log"}, {"seq", ++logSeq}, {"text", line}};
        {
            std::lock_guard<std::mutex> g(logMutex);
            logRing.push_back(rec);
            while (logRing.size() > kLogRingMax) logRing.pop_front();
        }
        publish(rec);
    }

    // ---------- state ----------
    json engine_json() {
        std::lock_guard<std::mutex> g(stateMutex);
        const char* s = engine == EngineState::Stopped ? "stopped" : engine == EngineState::Loading ? "loading"
                        : engine == EngineState::Ready ? "ready" : "error";
        json j{{"state", s}, {"detail", detail}, {"stage", loadStage}, {"percent", loadPercent.load()},
               {"model", activeModel}, {"mmproj", activeMmproj}, {"port", activePort},
               {"pendingReload", pendingReload}, {"generating", generating.load()}};
        if (hardware.free_vram_known) j["vramFreeKnown"] = true;
        if (engine == EngineState::Ready)
            j["uptimeS"] = (long long)std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - readyAt).count();
        return j;
    }

    json model_json(const ft::ModelInfo& m, int index) {
        json j{{"id", m.path.u8string()}, {"name", m.name}, {"valid", m.valid},
               {"architecture", m.architecture}, {"sizeMib", mib(m.bytes)},
               {"layers", m.layers}, {"context", m.context}, {"embedding", m.embedding},
               {"heads", m.heads}, {"kvHeads", m.kv_heads}, {"experts", m.experts},
               {"projector", m.projector.u8string()}, {"error", m.error}, {"index", index}};
        if (m.context) j["contextLabel"] = m.context >= 1024 ? (std::to_string(m.context / 1024) + "K") : std::to_string(m.context);
        json caps = json::array();
        caps.push_back("text");
        if (!m.projector.empty()) caps.push_back("vision");
        if (m.experts) caps.push_back("moe");
        j["capabilities"] = caps;
        return j;
    }

    json stats_json() {
        std::lock_guard<std::mutex> g(statsMutex);
        auto pmem = current_proc_mem();
        return json{{"requestsTotal", requestsTotal}, {"tokensIn", tokensIn}, {"tokensOut", tokensOut},
                    {"lastTokSec", lastTokSec}, {"lastTtftMs", lastTtftMs},
                    {"ctxUsed", ctxUsed}, {"ctxTotal", ctxTotal},
                    {"ramTotalMib", ramTotalMib}, {"ramFreeMib", ramFreeMib},
                    {"vramTotalMib", vramTotalMib}, {"vramFreeMib", vramFreeMib},
                    {"guiWorkingSetMib", pmem.workingSetMib},
                    {"guiPrivateMib", pmem.privateMib}};
    }

    json state_json() {
        json models = json::array();
        for (size_t i = 0; i < library.size(); ++i) models.push_back(model_json(library[i], (int)i));
        json j{{"type", "state"},
               {"app", json{{"name", "LlamaCPP P100"}, {"version", "0.1.0"}, {"dataDir", data.u8string()}}},
               {"engine", engine_json()},
               {"settings", config},
               {"hardware", json{{"gpu", hardware.gpu}, {"threads", hardware.threads},
                                  {"ramTotalMib", mib(hardware.ram)}, {"vramTotalMib", mib(hardware.vram)},
                                  {"vramFreeMib", hardware.free_vram_known ? json(mib(hardware.free_vram)) : json()}}},
               {"library", json{{"folder", config.value("model_folder", std::string())}, {"models", models}}},
               {"conversation", conversation}, {"chats", chat_summaries()}, {"activeChat", activeChat},
               {"stats", stats_json()},
               {"review", review_json()}};
        return j;
    }

    // ---------- startup review (structured, honest estimates) ----------
    json review_json() {
        json budget = json::array(), warnings = json::array();
        try {
            auto modelPath = std::filesystem::u8path(config.value("model", std::string()));
            if (!std::filesystem::is_regular_file(modelPath)) {
                warnings.push_back("Select a model in Models to see a memory plan.");
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
                long long ctx = 0;
                try { ctx = std::stoll(config.value("ctx", std::string("4096"))); } catch (...) {}
                uint64_t kvTotal = (uint64_t)ft::kv_cache_bytes(m, ctx, config.at("cache_k"), config.at("cache_v"));
                uint64_t weightsGpu = (uint64_t)((double)m.bytes * gpuFraction);
                uint64_t weightsCpu = m.bytes - weightsGpu;

                bool kvOnGpu = gpuLayers > 0 && config.value("no_kv", std::string("0")) != "1";
                auto seg = [&](const char* label, const char* kind, uint64_t b, const char* note, const char* device) {
                    return json{{"label", label}, {"kind", kind}, {"mib", mib(b)}, {"note", note}, {"device", device}};
                };
                if (weightsGpu) budget.push_back(seg("Model weights on GPU", "weights", weightsGpu,
                    std::to_string(gpuLayers).append(" of ").append(std::to_string(m.layers)).append(" layers offloaded").c_str(), "vram"));
                if (expertOn && gpuCache) budget.push_back(seg("Expert cache on GPU", "cache", gpuCache * 1024 * 1024, "--moe-gpu-cache-mib", "vram"));
                if (kvTotal) budget.push_back(seg(kvOnGpu ? "KV cache on GPU" : "KV cache on RAM", "kv", kvTotal,
                    std::to_string(ctx).append(" tokens x ").append(config.value("cache_k", std::string("f16"))).c_str(), kvOnGpu ? "vram" : "ram"));
                if (gpuLayers > 0) budget.push_back(seg("Scratch and reserve", "reserve", reserve * 1024 * 1024, "backend scratch is not tracked", "vram"));
                if (weightsCpu) budget.push_back(seg("Model weights on RAM", "weights", weightsCpu,
                    std::to_string(layers - gpuLayers).append(" layers on CPU").c_str(), "ram"));
                if (expertOn && cpuCache) budget.push_back(seg("Expert cache on RAM", "cache", cpuCache * 1024 * 1024, "--moe-cpu-cache-mib", "ram"));

                warnings.push_back("Estimates only - not calibrated. Actual usage depends on backend scratch, CUDA graphs and other workloads.");
                if (config.value("graphs", std::string("auto")) == "auto" && contains(hardware.gpu, "P100"))
                    warnings.push_back("P100/SM60: the current backend guard disables CUDA graph replay.");
                if (expertOn && (cpuCache || value("cpu_tile", 0))) {
                    if (ngl != 0) warnings.push_back("CPU expert cache/tile requires CPU placement (GPU layers 0) - validate will reject other combinations.");
                }
            }
        } catch (const std::exception& e) {
            warnings.push_back(e.what());
        }
        std::ostringstream s;
        s << "Startup review - preview, not calibrated. Available RAM: " << ft::available_memory() / 1024 / 1024
          << " MiB. Logical CPU threads: " << std::thread::hardware_concurrency() << ".";
        return json{{"budget", budget}, {"warnings", warnings}, {"text", s.str()}};
    }

    // ---------- engine lifecycle ----------
    void set_engine(EngineState s, const std::string& d, const std::string& stage = "", long long percent = -1) {
        {
            std::lock_guard<std::mutex> g(stateMutex);
            engine = s; detail = d; loadStage = stage; loadPercent = percent;
        }
        json ev{{"type", "state"}, {"engine", engine_json()}};
        publish(ev);
    }

    void finish_scan(bool ok) {
        {
            std::lock_guard<std::mutex> g(stateMutex);
            if (ok) library = std::move(scanned);
            scanning = false;
        }
        save_config();
        publish(json{{"type", "scanning"}, {"active", false}});
        publish(json{{"type", "library"}});
        if (ok) {
            int n = 0;
            { std::lock_guard<std::mutex> g(stateMutex); n = (int)library.size(); }
            publish(json{{"type", "toast"}, {"level", "success"}, {"text", std::to_string(n) + " models found."}});
        }
    }

    void scan_folder(const std::string& folder) {
        if (scanning) return;
        if (folder.empty()) { publish(json{{"type", "toast"}, {"level", "error"}, {"text", "Choose a model folder first."}}); return; }
        if (scanner.joinable()) scanner.join();
        scanning = true;
        publish(json{{"type", "scanning"}, {"active", true}});
        scanner = std::thread([this, folder] {
            try {
                auto result = ft::scan_models(std::filesystem::u8path(folder));
                {
                    std::lock_guard<std::mutex> g(stateMutex);
                    scanned = std::move(result);
                }
                finish_scan(true);
            } catch (const std::exception& e) {
                publish(json{{"type", "toast"}, {"level", "error"}, {"text", e.what()}});
                finish_scan(false);
            }
        });
    }

    ft::Hardware profile_hardware(const json& c, bool fresh=true) {
        auto detected=fresh?ft::detect_hardware():hardware;
        detected.extra_gpu_bytes=0;
        auto projector=c.value("mmproj",std::string());
        if(!projector.empty()) {
            std::error_code ec;auto bytes=std::filesystem::file_size(std::filesystem::u8path(projector),ec);
            if(!ec)detected.extra_gpu_bytes=bytes+128ull*1024*1024;
        }
        return detected;
    }

    void start_engine() {
        std::lock_guard<std::mutex> lifecycle(engineLifecycleMutex);
        if (generating) throw std::runtime_error("Stop the current response before reloading.");
        if (engine == EngineState::Loading) throw std::runtime_error("A model is already loading.");
        if (engineWorker.joinable()) engineWorker.join();
        save_config();
        json c;
        {
            std::lock_guard<std::mutex> g(cfgMutex);
            if (config.value("model", std::string()).empty()) {
                throw std::runtime_error("Select a model first.");
            }
            c = config;
        }
        cancelled = false;
        set_engine(EngineState::Loading, "Preparing to launch llama-server", "Preparing", 0);
        {
            std::lock_guard<std::mutex> g(stateMutex);
            loadStarted = std::chrono::steady_clock::now();
        }
        engineWorker = std::thread([this, c]() mutable {
            try {
                ft::arguments(c); // Validate paths/settings before stopping the old engine.
                int port = std::stoi(c.at("port").get<std::string>());
                // Release our previous server's port before checking for another owner.
                process.stop();
                // Rebudget an untouched automatic profile after releasing this
                // engine's allocations. Hand-edited load settings remain manual.
                auto baseline=c.value("freetoken_baseline",json::object());
                bool automatic=!baseline.empty() && c.value("auto_settings",std::string("1"))=="1";
                for(auto it=baseline.begin();it!=baseline.end();++it)
                    if(it.key()!="freetoken_note" && (!c.contains(it.key()) || c[it.key()]!=it.value()))automatic=false;
                hardware=profile_hardware(c);
                if(automatic) {
                    auto profile=ft::freetoken_profile(ft::inspect_model(std::filesystem::u8path(c.at("model").get<std::string>())),hardware,std::stoull(c.at("ctx").get<std::string>()),c.at("cache_k"),c.at("cache_v"));
                    c.update(profile);c["freetoken_baseline"]=profile;c["freetoken_model"]=c["model"];
                    std::lock_guard<std::mutex> g(cfgMutex);config=c;save_config();
                }
                auto args=ft::arguments(c);args.push_back("--metrics");args.push_back("--slots");
                {
                    std::lock_guard<std::mutex> g(stateMutex);
                    activePort = 0;
                }
                httplib::Server probe;
                if (!probe.bind_to_port("127.0.0.1", port)) throw std::runtime_error("Port is already in use. Choose another port in Tune.");
                probe.stop();
                {
                    std::lock_guard<std::mutex> g(cfgMutex);
                    activeMmproj = c.value("mmproj", std::string());
                    activeModel = c.value("model", std::string());
                }
                process.start(args, data / "server.log", c.value("graphs", std::string("auto")) == "off",
                              [this](const std::string& line) {
                                  on_server_line(line);
                              });
                auto client = std::make_shared<httplib::Client>("127.0.0.1", port);
                client->set_connection_timeout(1, 0);
                client->set_read_timeout(1, 0);
                { std::lock_guard<std::mutex> g(clientMutex); activeClient = client; }
                bool healthy = false;
                auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(5);
                long long elapsedMs = 0;
                while (!cancelled && std::chrono::steady_clock::now() < deadline) {
                    if (!process.running()) throw std::runtime_error("llama-server exited during startup. Open Monitor for the log.");
                    auto res = client->Get("/health");
                    auto now = std::chrono::steady_clock::now();
                    elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - loadStarted).count();
                    if (res && res->status == 200) { healthy = true; break; }
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                }
                if (cancelled || !healthy) {
                    process.stop();
                    set_engine(EngineState::Error, cancelled ? "Startup cancelled." : "Startup timed out after 5 minutes. Open Monitor for the log.");
                    return;
                }
                {
                    std::lock_guard<std::mutex> g(stateMutex);
                    activePort = port;
                    readyAt = std::chrono::steady_clock::now();
                    pendingReload = false;
                }
                set_engine(EngineState::Ready, "", "", 100);
                trim_memory();
                publish(json{{"type", "toast"}, {"level", "success"},
                             {"text", "Model loaded on port " + std::to_string(port) + " in " + std::to_string(elapsedMs / 1000) + " s."}});
            } catch (const std::exception& e) {
                process.stop();
                set_engine(EngineState::Error, e.what());
            }
        });
    }

    void on_server_line(const std::string& line) {
        log_line(line);
        const char* stage = nullptr;
        long long percent = -1;
        if (contains(line, "llama_model_loader:")) { stage = "Reading model file"; percent = 5; }
        if (contains(line, "load_tensors: loading model tensors")) { stage = "Loading tensors"; percent = 15; }
        if (contains(line, "load_tensors: offloaded")) { stage = "Placing layers on GPU"; percent = 35; }
        if (contains(line, "llama_context:") || contains(line, "warmup")) { stage = "Allocating context and warmup"; percent = 70; }
        if (contains(line, "server is listening") || contains(line, "listening on")) { stage = "Starting API"; percent = 95; }
        if (stage) set_engine(EngineState::Loading, "", stage, percent);
    }

    void unload_engine() {
        std::lock_guard<std::mutex> lifecycle(engineLifecycleMutex);
        if (generating) cancel_chat();
        if (engineWorker.joinable()) engineWorker.join();
        process.stop();
        {
            std::lock_guard<std::mutex> g(stateMutex);
            activePort = 0;
        }
        set_engine(EngineState::Stopped, "");
        publish(json{{"type", "toast"}, {"level", "info"}, {"text", "Model unloaded. Conversation kept."}});
    }

    // ---------- chat ----------
    json chat_summaries() {
        json result=json::array();
        for(auto& chat:chats) result.push_back({{"id",chat.at("id")},{"title",chat.value("title",std::string("New chat"))}});
        return result;
    }
    void save_conversation() {
        json entry{{"id",activeChat},{"title",conversation.empty()?"New chat":conversation[0].value("content",std::string("New chat")).substr(0,80)},{"messages",conversation}};
        bool found=false;
        for(auto& chat:chats)if(chat.at("id")==activeChat){chat=entry;found=true;break;}
        if(!found)chats.push_back(entry);
        ft::save_json(data / "chats.json", json{{"active",activeChat},{"chats",chats}});
        ft::save_json(data / "conversation.json", conversation);
    }

    void cancel_chat() {
        cancelled = true;
        std::lock_guard<std::mutex> g(clientMutex);
        if (activeClient) activeClient->stop();
    }

    void run_chat(const std::string& user) {
        if (chatWorker.joinable()) chatWorker.join();
        json requestMessages = conversation;
        std::string system;
        double temperature = 0.7, topP = 0.95, minP = 0.05, repeatPenalty = 1.0;
        int maxTokens = 512, topK = 20;
        std::string thinking = "0";
        json extraOptions;
        {
            std::lock_guard<std::mutex> g(cfgMutex);
            requestMessages = conversation;
            if (config.value("system", std::string()) != "")
                requestMessages.insert(requestMessages.begin(), json{{"role", "system"}, {"content", config.value("system", std::string())}});
            auto d = [](const json& j, const char* k, double f) { try { return std::stod(j.at(k).get<std::string>()); } catch (...) { return f; } };
            auto i = [](const json& j, const char* k, int f) { try { return std::stoi(j.at(k).get<std::string>()); } catch (...) { return f; } };
            temperature = d(config, "temperature", 0.7); topP = d(config, "top_p", 0.95); minP = d(config, "min_p", 0.05);
            repeatPenalty = d(config, "repeat_penalty", 1.0); maxTokens = i(config, "max_tokens", 512); topK = i(config, "top_k", 20);
            thinking = config.value("thinking", std::string("0"));
            extraOptions = ft::inference_options(config);
        }
        cancelled = false;
        generating = true;
        set_engine(engine.load(), "");  // refresh generating flag
        json body = {{"model", "local"}, {"messages", requestMessages}, {"stream", true},
                     {"temperature", temperature}, {"max_tokens", maxTokens},
                     {"chat_template_kwargs", json{{"enable_thinking", thinking == "1"}}},
                     {"top_k", topK}, {"top_p", topP}, {"min_p", minP}, {"repeat_penalty", repeatPenalty},
                     {"cache_prompt", true}, {"timings_per_token", true}, {"stream_options", json{{"include_usage", true}}}};
        int port = activePort;
        body.update(extraOptions);
        publish(json{{"type", "chat.begin"}});
        chatWorker = std::thread([this, body, port] {
            auto started = std::chrono::steady_clock::now();
            std::string answer, reasoning;
            bool firstToken = true;
            long long ttftMs = 0;
            json usage;
            double responseTokSec = 0;
            std::string stopReason = "eos";
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
                    if(event.contains("timings")) responseTokSec=event["timings"].value("predicted_per_second",responseTokSec);
                    if (!event.contains("choices") || event["choices"].is_array() == false) return;
                    if (event["choices"].empty()) {
                        if (event.contains("timings")) {
                            auto& t = event["timings"];
                            double tokSec = t.value("predicted_per_second", 0.0);
                            std::lock_guard<std::mutex> g(statsMutex);
                            lastTokSec = tokSec;
                        }
                        return;
                    }
                    auto delta = event["choices"][0].value("delta", json::object());
                    bool hasText=(delta.contains("content") && delta["content"].is_string() && !delta["content"].get<std::string>().empty()) ||
                        (delta.contains("reasoning_content") && delta["reasoning_content"].is_string() && !delta["reasoning_content"].get<std::string>().empty());
                    if (firstToken && hasText) {
                        firstToken = false;
                        ttftMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - started).count();
                    }
                    if (delta.contains("reasoning_content") && delta["reasoning_content"].is_string()) {
                        auto text = delta["reasoning_content"].get<std::string>();
                        total += text.size(); reasoning += text;
                        publish(json{{"type", "chat.delta"}, {"kind", "reasoning"}, {"text", text}});
                    }
                    if (delta.contains("content") && delta["content"].is_string()) {
                        auto text = delta["content"].get<std::string>();
                        total += text.size(); answer += text;
                        publish(json{{"type", "chat.delta"}, {"kind", "content"}, {"text", text}});
                    }
                    if (event.contains("timings")) {
                        auto& t = event["timings"];
                        json metrics{{"type", "chat.metrics"},
                                     {"tokSec", t.value("predicted_per_second", 0.0)},
                                     {"predN", t.value("predicted_n", 0)},
                                     {"promptTokSec", t.value("prompt_per_second", 0.0)},
                                     {"ttftMs", ttftMs}};
                        publish(metrics);
                        std::lock_guard<std::mutex> g(statsMutex);
                        lastTokSec = t.value("predicted_per_second", 0.0);
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
                    if (httpStatus != 200) { if (errorBody.size() < 8192) error_body_append(errorBody, bytes, n); }
                    else events.feed(bytes, n);
                    return true;
                };
                auto response = client->send(request);
                if (cancelled) { stopReason = "cancelled"; }
                else if (!response) { stopReason = "error"; throw std::runtime_error("Request failed: " + httplib::to_string(response.error())); }
                else if (httpStatus != 200) { stopReason = "error"; throw std::runtime_error("HTTP " + std::to_string(httpStatus) + ": " + errorBody); }
                else if (!events.done) { stopReason = "truncated"; }
                if (usage.is_object()) {
                    try {
                        std::lock_guard<std::mutex> g(statsMutex);
                        tokensIn += usage.value("prompt_tokens", 0);
                        tokensOut += usage.value("completion_tokens", 0);
                        requestsTotal += 1;
                        lastTtftMs = (double)ttftMs;
                    } catch (...) {}
                }
            } catch (const std::exception& e) {
                if (std::string(e.what()) == "__cancelled__") stopReason = "cancelled";
                else { stopReason = "error"; }
                log_line(std::string("Chat: ") + e.what());
                publish(json{{"type", "toast"}, {"level", "error"}, {"text", std::string(stopReason == "error" ? e.what() : "Response stopped.")}});
            }
            {
                std::lock_guard<std::mutex> g(cfgMutex);
                if (!answer.empty() || !reasoning.empty()) {
                    json msg{{"role", "assistant"}, {"content", answer}};
                    if (!reasoning.empty()) msg["reasoning"] = reasoning;
                    msg["stats"]={{"genTokens",usage.is_object()?usage.value("completion_tokens",0):0},
                        {"promptTokens",usage.is_object()?usage.value("prompt_tokens",0):0},
                        {"ttftMs",ttftMs},{"tokSec",responseTokSec},{"stopReason",stopReason}};
                    conversation.push_back(msg);
                    try { save_conversation(); } catch(const std::exception& e) { log_line(std::string("Could not save chat: ")+e.what()); }
                }
            }
            generating = false;
            json done{{"type", "chat.done"}, {"stopReason", stopReason},
                      {"content", answer}, {"reasoning", reasoning},
                      {"ttftMs", ttftMs}, {"tokSec", responseTokSec},
                      {"usage", usage.is_object() ? usage : json::object()}};
            publish(done);
            set_engine(engine.load(), "");
        });
    }
    static void error_body_append(std::string& body, const char* bytes, size_t n) {
        body.append(bytes, std::min(n, size_t(8192 - body.size())));
    }

    // ---------- telemetry ----------
    size_t subscriber_count() {
        std::lock_guard<std::mutex> g(busMutex);
        return subs.size();
    }

    void telemetry_loop() {
        std::shared_ptr<httplib::Client> slotsClient;
        int cachedPort = 0;
        auto tick = [&](bool busy) {
            auto ram = ram_info();
            auto vram = vram_info();
            auto pmem = current_proc_mem();
            {
                std::lock_guard<std::mutex> g(statsMutex);
                ramTotalMib = mib(ram.total); ramFreeMib = mib(ram.avail);
                if (vram.known) { vramTotalMib = mib(vram.total); vramFreeMib = mib(vram.freeB); }
            }
            json ev{{"type", "telemetry"},
                    {"ramTotalMib", mib(ram.total)}, {"ramFreeMib", mib(ram.avail)},
                    {"vramTotalMib", vram.known ? json(mib(vram.total)) : json()},
                    {"vramFreeMib", vram.known ? json(mib(vram.freeB)) : json()},
                    {"gpuName", vram.name.empty() ? json() : json(vram.name)},
                    {"guiWorkingSetMib", pmem.workingSetMib},
                    {"guiPrivateMib", pmem.privateMib}};
            if (engine == EngineState::Ready && activePort) {
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
                            auto& slot = slots[0];
                            ev["ctxUsed"] = slot.value("n_prompt_tokens", 0);
                            ev["ctxTotal"] = slot.value("n_ctx", 0);
                            ev["processing"] = slot.value("is_processing", false);
                            std::lock_guard<std::mutex> g(statsMutex);
                            ctxUsed = slot.value("n_prompt_tokens", 0);
                            ctxTotal = slot.value("n_ctx", 0);
                        }
                    } catch (...) {}
                }
            } else {
                slotsClient.reset();
            }
            publish(ev);
        };
        while (!closing) {
            // Fast cadence while a UI is watching or the engine is working; long sleep when
            // nothing needs us, so idle memory and wakeups stay minimal.
            bool busy = generating || engine != EngineState::Stopped;
            bool watched = subscriber_count() > 0;
            tick(busy || watched);
            static std::once_flag firstTick;
            std::call_once(firstTick, [] { trim_memory(); });  // NVML allocations landed on this tick
            int ticks = (busy || watched) ? 10 : 60;  // 1 s active, 6 s idle
            for (int i = 0; i < ticks && !closing; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    // ---------- config ----------
    void save_config() {
        ft::save_json(data / "settings.json", config);
    }

    void bootstrap() {
        // one-time migration from the legacy FLTK app data folder
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
        auto history=ft::read_json(data / "chats.json",json::object());
        if(history.contains("chats") && history["chats"].is_array()) {
            chats=history["chats"];activeChat=history.value("active",std::string("initial"));
            for(auto& chat:chats)if(chat.at("id")==activeChat)conversation=chat.at("messages");
        }
        save_conversation();

        std::filesystem::path cwd = std::filesystem::current_path();
        std::filesystem::path exeDir = cwd;
#ifdef _WIN32
        wchar_t exe[32768]{};
        auto n = GetModuleFileNameW(nullptr, exe, 32768);
        if (n && n < 32768) exeDir = std::filesystem::path(exe).parent_path();
#elif defined(__linux__)
        std::error_code error;
        auto exe = std::filesystem::read_symlink("/proc/self/exe", error);
        if (!error) exeDir = exe.parent_path();
#endif
        if (ui_dir.empty()) ui_dir = (exeDir / "ui").u8string();
#ifdef _WIN32
        std::filesystem::path serverRel = "build/ft-windows-cuda-dev/bin/llama-server.exe";
#else
        std::filesystem::path serverRel = "build/ft-linux-p100/bin/llama-server";
#endif
        std::vector<std::filesystem::path> roots = {cwd, exeDir.parent_path().parent_path(), exeDir.parent_path().parent_path().parent_path()};
        if (config.at("server") == "" && std::filesystem::is_regular_file(exeDir / "llama-server"))
            config["server"] = (exeDir / "llama-server").u8string();
        if (config.at("server") == "")
            for (auto& r : roots)
                if (std::filesystem::exists(r / serverRel)) { config["server"] = (r / serverRel).u8string(); break; }
        if (config["model_folder"] == "")
            for (auto& r : roots)
                if (std::filesystem::is_directory(r / "models")) { config["model_folder"] = (r / "models").u8string(); break; }
    }

    std::string ui_dir;

    // ---------- routes ----------
    void apply_settings_patch(const json& patch, json& response) {
        json next;
        {
            std::lock_guard<std::mutex> g(cfgMutex);
            next = config;
        }
        for (auto it = patch.begin(); it != patch.end(); ++it) {
            if (!next.contains(it.key()) || !next[it.key()].is_string() || !it.value().is_string())
                throw std::runtime_error("Unknown or invalid setting: " + it.key());
            next[it.key()] = it.value();
        }
        if (patch.contains("cache_k") || patch.contains("cache_v")) {
            if(generating || engine==EngineState::Loading) throw std::runtime_error("Wait for the current operation before changing KV cache.");
            if(next.at("auto_settings")=="1" && next.at("model")!="") {
                auto profile=ft::freetoken_profile(ft::inspect_model(std::filesystem::u8path(next.at("model").get<std::string>())),
                    profile_hardware(next,engine!=EngineState::Ready),std::stoull(next.at("ctx").get<std::string>()),next.at("cache_k"),next.at("cache_v"));
                next.update(profile);
                next["freetoken_baseline"]=profile;next["freetoken_model"]=next.at("model");
            }
            auto v=next.at("cache_v").get<std::string>();
            if(v!="f16" && v!="f32" && v!="bf16")next["flash"]="on";
        }
        ft::validate(next);
        // which keys need a reload vs apply on the next request
        json reloadKeys = json::array();
        for (auto& f : ft::settings()) {
            if (next.contains(f.key) && next[f.key] != config_at(f.key)) reloadKeys.push_back(f.key);
        }
        for (const char* k : {"model", "mmproj", "graphs", "expert_enabled", "server", "port"}) {
            if (next.contains(k) && next[k] != config_at(k)) reloadKeys.push_back(k);
        }
        bool needsReload = !reloadKeys.empty();
        {
            std::lock_guard<std::mutex> g(cfgMutex);
            config = next;
            std::lock_guard<std::mutex> g2(stateMutex);
            if (engine == EngineState::Ready) pendingReload = pendingReload || needsReload;
            save_config();
        }
        response = json{{"ok", true}, {"pendingReload", needsReload}, {"reloadKeys", reloadKeys}, {"engine", engine_json()}};
        publish(json{{"type", "settings"}, {"engine", engine_json()}});
    }
    json config_at(const std::string& key) {
        std::lock_guard<std::mutex> g(cfgMutex);
        return config.contains(key) ? config[key] : json();
    }

    void routes() {
        server.Get("/api/settings/schema", [](const httplib::Request&, httplib::Response& res) {
            json fields=json::array();
            for(auto& f:ft::settings()) fields.push_back({{"key",f.key},{"label",f.label},{"flag",f.flag},{"initial",f.initial},{"help",f.help},{"toggle",f.toggle}});
            res.set_content(fields.dump(), "application/json");
        });
        server.Post("/api/freetoken/configure", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                if(generating || engine==EngineState::Loading) throw std::runtime_error("Wait for the current operation to finish before applying a profile.");
                auto action=json::parse(req.body).value("action",std::string("auto"));
                json next;
                { std::lock_guard<std::mutex> g(cfgMutex); next=config; }
                auto model=next.value("model",std::string());
                if(action=="auto") {
                    auto detected=profile_hardware(next,engine!=EngineState::Ready);
                    auto profile=ft::freetoken_profile(ft::inspect_model(std::filesystem::u8path(model)),detected,std::stoull(next.at("ctx").get<std::string>()),next.at("cache_k"),next.at("cache_v"));
                    next.update(profile);
                    next["auto_settings"]="1";
                    next["freetoken_baseline"]=profile; next["freetoken_model"]=model;
                    hardware=detected;
                } else if(action=="restore") {
                    if(next["freetoken_model"]!=model || next["freetoken_baseline"].empty()) throw std::runtime_error("Auto-configure FreeToken for this model before restoring its profile.");
                    auto baseline=next["freetoken_baseline"];
                    next.update(baseline);
                } else if(action=="disable") next["expert_enabled"]="0";
                else throw std::runtime_error("Unknown FreeToken profile action");
                ft::validate(next);
                { std::lock_guard<std::mutex> g(cfgMutex); config=next; save_config(); }
                { std::lock_guard<std::mutex> g(stateMutex); if(engine==EngineState::Ready) pendingReload=true; }
                res.set_content(json{{"ok",true},{"engine",engine_json()}}.dump(), "application/json");
                publish(json{{"type","settings"}});
            } catch(const std::exception& e) {
                res.status=400; res.set_content(json{{"error",e.what()}}.dump(), "application/json");
            }
        });
        // Each SSE subscriber occupies one pool worker for its lifetime; 6 covers a couple of
        // open UI tabs plus API calls while keeping thread stacks (1 MB VA each) minimal.
        server.new_task_queue = [] { return new httplib::ThreadPool(6); };
        server.Get("/api/healthz", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("{\"ok\":true}", "application/json");
        });
        server.Get("/api/events", [this](const httplib::Request&, httplib::Response& res) {
            auto sub = std::make_shared<Sub>();
            {
                std::lock_guard<std::mutex> g(busMutex);
                int id = nextSub++;
                subs[id] = sub;
                res.set_header("X-Sub-Id", std::to_string(id));
            }
            sub->m.lock();
            sub->q.push_back(json{{"type", "hello"}});
            sub->m.unlock();
            res.set_header("Cache-Control", "no-cache");
            res.set_chunked_content_provider(
                "text/event-stream",
                [this, sub](size_t, httplib::DataSink& sink) {
                    json ev;
                    bool got = false;
                    for (int waited = 0; waited < 150 && !got && !closing; ++waited) {
                        {
                            std::lock_guard<std::mutex> g(sub->m);
                            if (!sub->q.empty()) { ev = sub->q.front(); sub->q.pop_front(); got = true; }
                        }
                        if (!got) std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                    auto frame = [&](const std::string& payload) {
                        std::string s = "data: " + payload + "\n\n";
                        return sink.write(s.data(), s.size());
                    };
                    if (got) return frame(ev.dump());
                    if (closing) return false;
                    return sink.write(": ping\n\n", 8);  // heartbeat keeps intermediaries from closing the stream
                },
                [this, sub](bool) {
                    std::lock_guard<std::mutex> g(busMutex);
                    for (auto it = subs.begin(); it != subs.end(); ++it)
                        if (it->second == sub) { subs.erase(it); break; }
                });
        });
        server.Get("/api/state", [this](const httplib::Request&, httplib::Response& res) {
            res.set_content(state_json().dump(), "application/json");
        });
        server.Get("/api/logs", [this](const httplib::Request& req, httplib::Response& res) {
            uint64_t since = 0;
            try { since = std::stoull(req.get_param_value("since")); } catch (...) {}
            json lines = json::array();
            uint64_t last = since;
            {
                std::lock_guard<std::mutex> g(logMutex);
                for (auto& rec : logRing)
                    if (rec["seq"].get<uint64_t>() > since) { lines.push_back(rec); last = rec["seq"].get<uint64_t>(); }
            }
            res.set_content(json{{"lines", lines}, {"lastSeq", last}}.dump(), "application/json");
        });
        server.Get("/api/review", [this](const httplib::Request&, httplib::Response& res) {
            res.set_content(review_json().dump(), "application/json");
        });
        server.Post("/api/models/browse", [](const httplib::Request& req, httplib::Response& res) {
            static std::mutex pickerMutex;
            std::unique_lock<std::mutex> lock(pickerMutex, std::try_to_lock);
            try {
                if (!lock.owns_lock()) throw std::runtime_error("A folder picker is already open.");
                auto body = json::parse(req.body);
                auto folder = browse_model_folder(body.value("folder", std::string()));
                res.set_content(json{{"folder", folder}, {"cancelled", folder.empty()}}.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            }
        });
        server.Post("/api/models/scan", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                auto body = json::parse(req.body);
                std::string folder = body.value("folder", std::string());
                if (scanning) throw std::runtime_error("Wait for the current model scan to finish.");
                {
                    std::lock_guard<std::mutex> g(cfgMutex);
                    if (folder.empty()) folder = config.value("model_folder", std::string());
                    if (folder.empty()) throw std::runtime_error("Enter a model folder first.");
                    std::error_code ec;
                    if (!std::filesystem::is_directory(std::filesystem::u8path(folder), ec) || ec)
                        throw std::runtime_error("Model folder does not exist or cannot be accessed.");
                    config["model_folder"] = folder;
                }
                scan_folder(folder);
                res.set_content("{\"ok\":true}", "application/json");
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            }
        });
        server.Post("/api/models/select", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                auto body = json::parse(req.body);
                std::string id = body.value("id", std::string());
                const ft::ModelInfo* found = nullptr;
                {
                    std::lock_guard<std::mutex> g(stateMutex);
                    for (auto& m : library)
                        if (m.path.u8string() == id) { found = &m; break; }
                }
                if (!found) throw std::runtime_error("Model not found in library.");
                if (!found->valid) throw std::runtime_error(found->error);
                ft::ModelInfo m = *found;
                std::string projector = body.value("projector", std::string());
                bool changed;
                {
                    std::lock_guard<std::mutex> g(cfgMutex);
                    changed = config["model"] != m.path.u8string();
                    config["model"] = m.path.u8string();
                    if (!projector.empty()) config["mmproj"] = projector;
                    else if (changed || config.value("mmproj", std::string()).empty()) config["mmproj"] = m.projector.u8string();
                    if (changed || config.value("recommended", json::object()).empty()) {
                        config["recommended"] = ft::recommend(m, profile_hardware(config,engine!=EngineState::Ready));
                        config["recommended_model"] = config["model"];
                        if (config.value("auto_settings", std::string("1")) == "1")
                        {
                            for (auto it = config["recommended"].begin(); it != config["recommended"].end(); ++it)
                                if(it.key()!="max_tokens") config[it.key()] = it.value();
                            config["freetoken_baseline"]=config["recommended"];config["freetoken_model"]=config["model"];
                        }
                    }
                    save_config();
                }
                {
                    std::lock_guard<std::mutex> g(stateMutex);
                    if (engine == EngineState::Ready && changed) pendingReload = true;
                }
                res.set_content(json{{"ok", true}, {"engine", engine_json()}}.dump(), "application/json");
                publish(json{{"type", "settings"}, {"engine", engine_json()}});
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            }
        });
        server.Post("/api/models/add", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                auto body = json::parse(req.body);
                auto path = std::filesystem::u8path(body.value("path", std::string()));
                auto m = ft::inspect_model(path);
                if (!m.valid || m.is_projector) throw std::runtime_error("Choose a valid language-model GGUF, not a projector.");
                {
                    std::lock_guard<std::mutex> g(stateMutex);
                    for (auto& existing : library)
                        if (existing.path.u8string() == m.path.u8string()) throw std::runtime_error("Model is already in the library.");
                    library.push_back(m);
                }
                publish(json{{"type", "library"}});
                res.set_content(json{{"ok", true}, {"id", m.path.u8string()}}.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            }
        });
        server.Post("/api/profile/restore", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                bool detect = json::parse(req.body).value("detect", false);
                json recommended;
                std::string modelId;
                {
                    std::lock_guard<std::mutex> g(cfgMutex);
                    modelId = config.value("model", std::string());
                }
                if (modelId.empty()) throw std::runtime_error("Select a readable GGUF model before detecting settings.");
                auto m = ft::inspect_model(std::filesystem::u8path(modelId));
                if (!m.valid) throw std::runtime_error("The selected model file could not be read.");
                json current;{std::lock_guard<std::mutex> g(cfgMutex);current=config;}
                if (detect && engine!=EngineState::Ready) hardware = ft::detect_hardware();
                recommended = ft::recommend(m, profile_hardware(current,engine!=EngineState::Ready));
                {
                    std::lock_guard<std::mutex> g(cfgMutex);
                    config["recommended"] = recommended;
                    config["recommended_model"] = modelId;
                    config["freetoken_baseline"]=recommended;config["freetoken_model"]=modelId;
                    for (auto it = recommended.begin(); it != recommended.end(); ++it) config[it.key()] = it.value();
                    save_config();
                }
                {
                    std::lock_guard<std::mutex> g(stateMutex);
                    if (engine == EngineState::Ready) pendingReload = true;
                }
                publish(json{{"type", "settings"}, {"engine", engine_json()}});
                res.set_content(json{{"ok", true}, {"engine", engine_json()}}.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            }
        });
        server.Post("/api/engine/load", [this](const httplib::Request&, httplib::Response& res) {
            try {
                start_engine();
                res.set_content(json{{"ok", true}, {"engine", engine_json()}}.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(json{{"error", e.what()}, {"engine", engine_json()}}.dump(), "application/json");
            }
        });
        server.Post("/api/engine/unload", [this](const httplib::Request&, httplib::Response& res) {
            unload_engine();
            res.set_content(json{{"ok", true}, {"engine", engine_json()}}.dump(), "application/json");
        });
        server.Post("/api/settings", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                auto patch = json::parse(req.body);
                json response;
                apply_settings_patch(patch, response);
                res.set_content(response.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            }
        });
        server.Post("/api/chat", [this](const httplib::Request& req, httplib::Response& res) {
            std::lock_guard<std::mutex> operation(chatOperationMutex);
            try {
                if (engine != EngineState::Ready) throw std::runtime_error("Load a model first.");
                if (pendingReload) throw std::runtime_error("Settings changed - reload the model first.");
                if (generating) throw std::runtime_error("Still responding - stop first.");
                auto body = json::parse(req.body);
                std::string content = body.value("content", std::string());
                if (content.empty()) throw std::runtime_error("Message is empty.");
                if (content.size() > 200000 || conversation.dump().size() > 2 * 1024 * 1024)
                    throw std::runtime_error("Conversation limit reached. Start a new conversation.");
                {
                    std::lock_guard<std::mutex> g(cfgMutex);
                    conversation.push_back(json{{"role", "user"}, {"content", content}});
                    save_conversation();
                }
                run_chat(content);
                res.set_content("{\"ok\":true}", "application/json");
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            }
        });
        server.Post("/api/chat/cancel", [this](const httplib::Request&, httplib::Response& res) {
            cancel_chat();
            res.set_content("{\"ok\":true}", "application/json");
        });
        server.Post("/api/conversation/new", [this](const httplib::Request&, httplib::Response& res) {
            std::lock_guard<std::mutex> operation(chatOperationMutex);
            if (generating) {res.status=409;res.set_content("{\"error\":\"Stop the current response before starting a new chat.\"}","application/json");return;}
            {
                std::lock_guard<std::mutex> g(cfgMutex);
                save_conversation();
                activeChat=std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
                conversation = json::array();
                save_conversation();
            }
            publish(json{{"type", "conversation"}});
            res.set_content("{\"ok\":true}", "application/json");
        });
        server.Post("/api/conversation/select", [this](const httplib::Request& req, httplib::Response& res) {
            std::lock_guard<std::mutex> operation(chatOperationMutex);
            try {
                if(generating)throw std::runtime_error("Stop the current response before switching chats.");
                auto id=json::parse(req.body).at("id").get<std::string>();
                {std::lock_guard<std::mutex> g(cfgMutex);
                    auto it=std::find_if(chats.begin(),chats.end(),[&](const json& chat){return chat.at("id")==id;});
                    if(it==chats.end())throw std::runtime_error("Chat not found.");
                    activeChat=id;conversation=it->at("messages");save_conversation();
                }
                publish(json{{"type","conversation"}});res.set_content("{\"ok\":true}","application/json");
            } catch(const std::exception& e){res.status=400;res.set_content(json{{"error",e.what()}}.dump(),"application/json");}
        });

        // static UI
        server.Get("/", [this](const httplib::Request&, httplib::Response& res) {
            std::ifstream in(std::filesystem::u8path(ui_dir) / "index.html", std::ios::binary);
            if (!in) { res.status = 404; res.set_content("UI assets not found next to the executable (expected ./ui/index.html).", "text/plain"); return; }
            std::string body((std::istreambuf_iterator<char>(in)), {});
            res.set_content(body, "text/html; charset=utf-8");
        });
        server.set_mount_point("/assets", (std::filesystem::u8path(ui_dir) / "assets").u8string());
    }

    int run(uint16_t port, bool openBrowser) {
        bootstrap();
        routes();
        trim_memory();
        telemetry = std::thread([this] { telemetry_loop(); });
        if (!config.value("model_folder", std::string()).empty()) scan_folder(config.value("model_folder", std::string()));
        publish(json{{"type", "state"}, {"engine", engine_json()}});
        std::string url = "http://127.0.0.1:" + std::to_string(port) + "/";
#ifdef _WIN32
        if (openBrowser) {
            std::filesystem::path edgePath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
            if (!std::filesystem::exists(edgePath))
                edgePath = "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe";
            bool launchedApp = false;
            if (std::filesystem::exists(edgePath)) {
                std::wstring edgeArgs = L"--app=\"" + std::wstring(url.begin(), url.end()) + L"\" --window-size=1280,840";
                HINSTANCE hInst = ShellExecuteW(nullptr, L"open", edgePath.c_str(), edgeArgs.c_str(), nullptr, SW_SHOWNORMAL);
                launchedApp = (reinterpret_cast<INT_PTR>(hInst) > 32);
            }
            if (!launchedApp) {
                ShellExecuteW(nullptr, L"open", std::wstring(url.begin(), url.end()).c_str(), nullptr, nullptr, SW_SHOWNOACTIVATE);
            }
        }
#else
        if (openBrowser) { std::string cmd = "xdg-open " + url; if (system(cmd.c_str()) != 0) {} }
#endif
        std::cout << "LlamaCPP P100 UI: " << url << "  (data: " << data.u8string() << ")\n";
        if (!server.listen("127.0.0.1", port)) return 1;
        return 0;
    }

    void shutdown() {
        if (closing.exchange(true)) return;
        cancel_chat();
        if (chatWorker.joinable()) chatWorker.join();
        if (engineWorker.joinable()) engineWorker.join();
        if (scanner.joinable()) scanner.join();
        if (telemetry.joinable()) telemetry.join();
        process.stop();
        try { save_config(); } catch (...) {}
        server.stop();
    }
};

Host::Host(std::filesystem::path dataDir) : impl(new Impl{std::move(dataDir)}) {}
Host::~Host() { shutdown(); }
int Host::run(uint16_t port, bool openBrowser) { return impl->run(port, openBrowser); }
void Host::shutdown() { impl->shutdown(); }
}
