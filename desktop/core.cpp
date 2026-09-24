#include "core.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <cmath>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace ft {
const std::vector<Setting>& settings() {
    static const std::vector<Setting> fields = {
        {"fit", "Automatic memory fit", "--fit", "off", "Let the engine adjust placement to available memory: on or off."},
        {"parallel", "Concurrent server sessions", "--parallel", "1", "FreeToken expert caching requires one session."},
        {"main_gpu", "Main GPU index", "--main-gpu", "0", "GPU index used for primary computation."},
        {"split_mode", "GPU split mode", "--split-mode", "layer", "none, layer or row."},
        {"tensor_split", "GPU allocation ratios", "--tensor-split", "", "Comma-separated ratios, for example 3,1. Empty uses engine defaults."},
        {"rope_base", "RoPE frequency base", "--rope-freq-base", "", "Empty uses the model default."},
        {"rope_scale", "RoPE frequency scale", "--rope-freq-scale", "", "Empty uses the model default."},
        {"mlock", "Keep model in memory", "--mlock", "0", "Lock model memory to reduce paging; requires enough RAM.", true},
        {"cpu_moe", "Force expert weights onto CPU", "--cpu-moe", "0", "Standard MoE offload, separate from FreeToken custom caching.", true},
        {"cpu_moe_layers", "CPU expert layer count", "--n-cpu-moe", "0", "Number of layers whose expert weights stay on CPU."},
        {"kv_unified", "Unified KV cache", "--kv-unified", "0", "Share KV capacity between concurrent sessions.", true},
        {"ctx_checkpoints", "Context checkpoints per slot", "--ctx-checkpoints", "8", "0 disables context checkpoints."},
        {"no_shift", "Stop at context limit", "--no-context-shift", "1", "When off, shift context as it fills. Model support varies.", true},
        {"chat_template", "Chat template", "--chat-template", "", "Optional Jinja template; empty uses the model template."},
        {"reasoning_format", "Reasoning parsing", "--reasoning-format", "deepseek", "none, deepseek or deepseek-legacy."},
        {"draft_model", "Draft model path", "--model-draft", "", "Optional compatible GGUF for speculative decoding."},
        {"draft_gpu_layers", "Draft GPU layers", "--gpu-layers-draft", "0", "Placement for the draft model."},
        {"draft_max", "Maximum draft tokens", "--draft-max", "16", "Upper bound on speculative drafts."},
        {"draft_min", "Minimum draft tokens", "--draft-min", "0", "Discard shorter drafts."},
        {"draft_p_min", "Draft probability cutoff", "--draft-p-min", "0.75", "Continue drafting above this probability."},
        {"ctx", "Context tokens", "--ctx-size", "4096", "Requires reload. Total context capacity; custom MoE path uses one slot."},
        {"threads", "Decode threads", "--threads", "4", "CPU worker threads. Start near physical core count."},
        {"threads_batch", "Prefill threads", "--threads-batch", "4", "Threads for prompt processing."},
        {"gpu_layers", "GPU layers", "--gpu-layers", "0", "Attention/shared layer placement. 0 keeps these on CPU."},
        {"batch", "Batch tokens", "--batch-size", "512", "Logical maximum prompt batch."},
        {"ubatch", "Microbatch tokens", "--ubatch-size", "128", "Physical batch; cannot exceed logical batch."},
        {"flash", "Flash attention", "--flash-attn", "auto", "auto, on or off; backend/model must support it."},
        {"cache_k", "KV key type", "--cache-type-k", "f16", "Backend validates supported cache types."},
        {"cache_v", "KV value type", "--cache-type-v", "f16", "Quantized value cache may require flash attention."},
        {"gpu_cache", "GPU expert cache MiB", "--moe-gpu-cache-mib", "", "Optional. Custom offload requires a single sequence."},
        {"cpu_cache", "CPU expert cache MiB", "--moe-cpu-cache-mib", "", "Optional CPU-only custom path. Cannot combine with GPU expert cache."},
        {"cpu_tile", "CPU expert tile MiB", "--moe-cpu-tile-mib", "", "Optional CPU-only tile path."},
        {"reserve", "GPU reserve MiB", "--moe-gpu-reserve-mib", "512", "Headroom; backend scratch and graph allocations also consume VRAM."},
        {"staging", "Transfer buffer MiB", "--moe-staging-mib", "1", "Host transfer staging buffer."},
        {"compute", "Expert arena MiB", "--moe-compute-mib", "64", "Excludes backend scratch and graph capture objects."},
        {"miss", "CPU miss share %", "--moe-cpu-miss-percent", "0", "0 GPU only; -1 calibrated; 1..100 fixed CPU share."},
        {"device", "Expert device", "--moe-device", "CUDA0", "Backend device name."},
        {"pipeline", "Pipeline expert transfers", "--moe-pipeline", "0", "Experimental copy/compute overlap.", true},
        {"fast", "Native numerical mode", "--moe-fast", "0", "Opt-in: may change numerical results and generated tokens.", true},
        {"semantic", "Semantic checkpoints", "--semantic-checkpoints", "0", "Requires compatible kernels and CPU attention; no draft model.", true},
        {"checkpoint", "Checkpoint budget MiB", "--semantic-checkpoint-mib", "512", "Bounded full-state checkpoints per slot."},
        {"no_mmap", "Disable memory mapping", "--no-mmap", "0", "May increase RAM usage during model load.", true},
        {"no_repack", "Disable weight repacking", "--no-repack", "0", "Keep original weight representation.", true},
        {"no_warmup", "Skip initial warmup", "--no-warmup", "0", "Defers warmup cost to the first request.", true},
        {"no_kv", "Keep KV cache on CPU", "--no-kv-offload", "0", "Automatically enabled for custom CPU cache/tile mode.", true},
        {"no_op", "Disable operation offload", "--no-op-offload", "0", "Automatically enabled for custom CPU cache/tile mode.", true}
    };
    return fields;
}
json defaults() {
    json j = {{"version",1},{"server",""},{"model",""},{"port","18080"},
        {"temperature","0.7"},{"max_tokens","512"},{"system","You are a helpful assistant."},{"graphs","auto"},{"thinking","0"},
        {"model_folder",""},{"mmproj",""},{"auto_settings","1"},{"expert_enabled","1"},{"recommended",json::object()},{"recommended_model",""},
        {"top_k","20"},{"top_p","0.95"},{"min_p","0.05"},{"repeat_penalty","1.0"}};
    for (auto& f : settings()) j[f.key] = f.initial;
    j.update(inference_defaults());
    j["freetoken_baseline"] = json::object();
    j["freetoken_model"] = "";
    j["freetoken_note"] = "";
    return j;
}
static int number(const json& j, const char* key, int lo, int hi) {
    auto s = j.at(key).get<std::string>();
    size_t used = 0; long long n;
    try { n = std::stoll(s, &used); } catch (...) { throw std::runtime_error(std::string(key)+": enter a whole number"); }
    if (used != s.size() || n < lo || n > hi) throw std::runtime_error(std::string(key)+": outside supported range");
    return static_cast<int>(n);
}
void validate(const json& c) {
    number(c,"port",1024,65535); number(c,"ctx",128,1048576);
    number(c,"threads",1,4096); number(c,"threads_batch",1,4096);
    number(c,"gpu_layers",0,999);
    if (number(c,"max_tokens",-1,1048576)==0) throw std::runtime_error("Output limit must be positive or -1 for no limit");
    if (number(c,"ubatch",1,1048576)>number(c,"batch",1,1048576)) throw std::runtime_error("Microbatch cannot exceed batch size");
    for (auto key : {"gpu_cache","cpu_cache","cpu_tile","staging","compute","checkpoint"})
        if (!c.at(key).get<std::string>().empty()) number(c,key,1,1048576);
    number(c,"reserve",0,1048576); number(c,"miss",-1,100);
    size_t used; double temp;
    auto t = c.at("temperature").get<std::string>();
    try { temp = std::stod(t,&used); } catch (...) { throw std::runtime_error("Invalid temperature"); }
    if (used != t.size() || !std::isfinite(temp) || temp < 0 || temp > 10) throw std::runtime_error("Temperature must be between 0 and 10");
    auto flash = c.at("flash").get<std::string>();
    if (flash!="auto" && flash!="on" && flash!="off") throw std::runtime_error("Flash attention: use auto, on or off");
    bool cpu = c.value("expert_enabled",std::string("1"))=="1" && (c.at("cpu_cache")!="" || c.at("cpu_tile")!="");
    if (cpu && (c.at("gpu_cache")!="" || c.at("gpu_layers")!="0")) throw std::runtime_error("CPU expert cache/tile requires CPU placement and no GPU expert cache");
    if (c.at("expert_enabled")=="1" && c.at("semantic")=="1" && (c.at("fast")=="1" || c.at("gpu_layers")!="0")) throw std::runtime_error("Semantic checkpoints require CPU attention and compatible numerical mode");
    if (c.at("graphs")!="auto" && c.at("graphs")!="off") throw std::runtime_error("Graphs: only backend Auto and Off are implemented");
    number(c,"top_k",0,10000);
    for(auto key:{"top_p","min_p","repeat_penalty"}) {auto s=c.at(key).get<std::string>();size_t n=0;double v=std::stod(s,&n);if(n!=s.size()||!std::isfinite(v)||v<0||v>(std::string(key)=="repeat_penalty"?10:1))throw std::runtime_error(std::string("Invalid ")+key);}
    validate_extended(c);
}
std::vector<std::string> arguments(const json& c) {
    validate(c);
    auto exe = c.at("server").get<std::string>(); auto model = c.at("model").get<std::string>();
    if (!std::filesystem::is_regular_file(std::filesystem::u8path(exe))) throw std::runtime_error("Select an existing llama-server executable");
    if (!std::filesystem::is_regular_file(std::filesystem::u8path(model))) throw std::runtime_error("Select an existing GGUF model");
    std::vector<std::string> a={exe,"--model",model,"--host","127.0.0.1","--port",c.at("port")};
    if(c.value("mmproj",std::string())!="") {
        auto projector=c.at("mmproj").get<std::string>();if(!std::filesystem::is_regular_file(std::filesystem::u8path(projector)))throw std::runtime_error("Selected vision projector is missing");
        a.push_back("--mmproj");a.push_back(projector);
    }
    bool custom = c.value("expert_enabled",std::string("1"))=="1" && (c.at("gpu_cache")!="" || c.at("cpu_cache")!="" || c.at("cpu_tile")!="");
    for (auto& f : settings()) {
        auto value = c.at(f.key).get<std::string>();
        if(f.key=="mlock" || f.key=="no_mmap") continue;
        if (!custom && (f.flag.rfind("--moe-",0)==0)) continue;
        if (c.at("expert_enabled")=="0" && (f.key=="semantic" || f.key=="checkpoint")) continue;
        if (c.at("draft_model")=="" && f.key.rfind("draft_",0)==0) continue;
        if (value.empty() || (f.toggle && value!="1")) continue;
        a.push_back(f.flag); if (!f.toggle) a.push_back(value);
    }
    if(c.at("no_shift")=="0") a.push_back("--context-shift");
    if(c.at("mlock")=="1" || c.at("no_mmap")=="1") {
        a.push_back("--load-mode");
        a.push_back(c.at("mlock")=="1" ? (c.at("no_mmap")=="1" ? "mlock" : "mmap+mlock") : "none");
    }
    if (custom && (c.at("cpu_cache")!="" || c.at("cpu_tile")!="")) {
        if(c.at("no_kv")!="1") a.push_back("--no-kv-offload");
        if(c.at("no_op")!="1") a.push_back("--no-op-offload");
    }
    return a;
}
std::string quote_windows(const std::string& s) {
    std::string out="\""; size_t slashes=0;
    for(char c:s) { if(c=='\\') {++slashes; continue;} if(c=='\"') out.append(slashes*2+1,'\\'); else out.append(slashes,'\\'); slashes=0; out+=c; }
    out.append(slashes*2,'\\'); return out+'\"';
}
std::filesystem::path data_directory() {
#ifdef _WIN32
    wchar_t* root=nullptr; size_t n=0; _wdupenv_s(&root,&n,L"LOCALAPPDATA");
    auto path=root?std::filesystem::path(root):std::filesystem::temp_directory_path(); free(root);
#else
    auto xdg=std::getenv("XDG_CONFIG_HOME"), home=std::getenv("HOME");
    auto path=xdg?std::filesystem::path(xdg):(home?std::filesystem::path(home)/".config":std::filesystem::temp_directory_path());
#endif
    return path/"FreeTokenDesktop";
}
void save_json(const std::filesystem::path& file,const json& j) {
    std::filesystem::create_directories(file.parent_path()); auto tmp=file; tmp+=".tmp";
    { std::ofstream out(tmp,std::ios::binary|std::ios::trunc); out << j.dump(2); out.close(); if(!out) throw std::runtime_error("Could not save "+file.u8string()); }
#ifdef _WIN32
    if(!MoveFileExW(tmp.c_str(),file.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)) throw std::runtime_error("Could not replace settings file");
#else
    std::filesystem::rename(tmp,file);
#endif
}
json read_json(const std::filesystem::path& file,const json& fallback) {
    if(!std::filesystem::exists(file)) return fallback;
    if(std::filesystem::file_size(file)>8*1024*1024) throw std::runtime_error("Saved file exceeds 8 MiB limit");
    std::ifstream in(file); return json::parse(in);
}
uint64_t available_memory() {
#ifdef _WIN32
    MEMORYSTATUSEX m{}; m.dwLength=sizeof(m); return GlobalMemoryStatusEx(&m)?m.ullAvailPhys:0;
#else
    std::ifstream in("/proc/meminfo"); std::string line;
    while(std::getline(in,line)) if(line.rfind("MemAvailable:",0)==0) {std::istringstream s(line.substr(13)); uint64_t n; if(s>>n) return n*1024;}
    return 0;
#endif
}
std::string describe_plan(const json& c) {
    std::ostringstream s;
    s << "Startup review - preview, not calibrated\n\nAvailable system RAM: " << available_memory()/1024/1024 << " MiB\nLogical CPU threads: " << std::thread::hardware_concurrency();
    auto p=std::filesystem::u8path(c.at("model").get<std::string>());
    if(std::filesystem::is_regular_file(p)) s << "\nModel file: " << std::filesystem::file_size(p)/1024/1024 << " MiB (not a RAM/VRAM estimate)";
    s << "\n\nOne active model, one server slot.\nContext: " << c.at("ctx").get<std::string>() << " tokens.\n";
    s << "\nGraphs: " << c.at("graphs").get<std::string>() << ". Auto defers to the backend.\nP100/SM60: current backend guard disables replay.\nThe SM61 patch is not a verified P100 fix.\nExperimental SM60 capture must retain operation checks,\nprove replay and pass numerical/memory tests on hardware.\n";
    s << "\nCache changes: Apply & reload restarts the managed server.\nConversation text is kept; KV state is reprocessed.\nCoordinated HTTP resizing is not implemented yet.\n";
    s << "\nRecommendations use GGUF dimensions, CPU threads and CUDA device capacity.\nThey reserve headroom but are not calibrated speed benchmarks.\nOther GPU workloads and model-specific buffers can change actual memory use.\n";
    return s.str();
}
void Events::feed(const char* data,size_t size) {
    pending.append(data,size);
    if(pending.size()>1024*1024) throw std::runtime_error("Stream event exceeds 1 MiB");
    size_t p;
    while((p=pending.find('\n'))!=std::string::npos) {
        auto line=pending.substr(0,p); pending.erase(0,p+1);
        if(!line.empty() && line.back()=='\r') line.pop_back();
        if(line.rfind("data:",0)!=0) continue;
        auto payload=line.substr(5); auto start=payload.find_first_not_of(' ');
        if(start==std::string::npos) continue; payload.erase(0,start);
        if(payload=="[DONE]") {done=true;continue;}
        callback(json::parse(payload));
    }
}
}
