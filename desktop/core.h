#pragma once
#include <nlohmann/json.hpp>
#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ft {
using json = nlohmann::ordered_json;
struct Setting { std::string key, label, flag, initial, help; bool toggle = false; };
const std::vector<Setting>& settings();
json defaults();
json inference_defaults();
void validate_extended(const json& config);
json inference_options(const json& config);
std::vector<std::string> arguments(const json& config);
void validate(const json& config);
std::filesystem::path data_directory();
void save_json(const std::filesystem::path& file, const json& value);
json read_json(const std::filesystem::path& file, const json& fallback);
std::string quote_windows(const std::string& value);
std::string describe_plan(const json& config);
uint64_t available_memory();
struct GpuMem { uint64_t total = 0, freeB = 0; bool known = false; std::string name; };
GpuMem gpu_memory();   // cached NVML query, safe to poll at 1 Hz
struct RamMem { uint64_t total = 0, avail = 0; };
RamMem ram_memory();
struct ModelInfo {
    std::filesystem::path path, projector;
    std::string name, architecture, error;
    uint64_t bytes=0, context=0, layers=0, embedding=0, heads=0, kv_heads=0, experts=0, loops=1;
    uint64_t expert_bytes=0, key_length=0, value_length=0, kv_bytes_per_token=0;
    bool tensor_sizes_known=false;
    bool valid=false, is_projector=false;
};
struct Hardware { unsigned threads=1, physical_cores=0; uint64_t ram=0, vram=0, free_vram=0, extra_gpu_bytes=0; bool free_vram_known=false; int compute_major=0, compute_minor=0; std::string gpu="CPU only / GPU unavailable"; };
ModelInfo inspect_model(const std::filesystem::path& path);
std::vector<ModelInfo> scan_models(const std::filesystem::path& directory);
Hardware detect_hardware();
double kv_cache_bytes(const ModelInfo& model, uint64_t tokens, const std::string& key, const std::string& value);
json recommend(const ModelInfo& model,const Hardware& hardware,uint64_t preferred_context=0, const std::string& cache_k="f16", const std::string& cache_v="f16");
json freetoken_profile(const ModelInfo& model,const Hardware& hardware,uint64_t preferred_context=0, const std::string& cache_k="f16", const std::string& cache_v="f16");

class Process {
public:
    Process(); ~Process();
    Process(const Process&) = delete;
    Process& operator=(const Process&) = delete;
    // Redirects child stdout/stderr to `log`; when `on_line` is set the output is also
    // piped and dispatched line-by-line (without trailing newline) as the child writes it.
    void start(const std::vector<std::string>& args, const std::filesystem::path& log, bool graphs_off,
               const std::function<void(const std::string&)>& on_line = {});
    void stop();
    bool running();
private:
    struct Impl; std::unique_ptr<Impl> impl;
};

class Events {
public:
    explicit Events(std::function<void(const json&)> callback) : callback(std::move(callback)) {}
    void feed(const char* data, size_t size);
    bool done = false;
private:
    std::string pending;
    std::function<void(const json&)> callback;
};
}
