#include "core.h"
#include <fstream>
#include <algorithm>
#include <map>
#include <regex>
#include <thread>
#include <stdexcept>
#include <iomanip>
#include <sstream>
#ifdef _WIN32
#include <windows.h>
#endif

namespace ft {
namespace {
std::string lower(std::string s) {for(auto& c:s)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));return s;}
struct Reader {
    std::ifstream in; uint64_t remaining;
    explicit Reader(const std::filesystem::path& p):in(p,std::ios::binary),remaining(std::min<uint64_t>(std::filesystem::file_size(p),64*1024*1024)) {}
    void read(char* p,uint64_t n) {if(n>remaining)throw std::runtime_error("GGUF metadata exceeds inspection limit");in.read(p,n);if(!in)throw std::runtime_error("Incomplete GGUF header");remaining-=n;}
    template<class T>T value(){T v{};read(reinterpret_cast<char*>(&v),sizeof(v));return v;}
    void skip(uint64_t n){if(n>remaining)throw std::runtime_error("Invalid GGUF metadata length");in.seekg(n,std::ios::cur);if(!in)throw std::runtime_error("Invalid GGUF offset");remaining-=n;}
    std::string string(){auto n=value<uint64_t>();if(n>1024*1024)throw std::runtime_error("GGUF string too large");std::string s(n,'\0');read(s.data(),n);return s;}
    json item(uint32_t t,int depth=0) {
        switch(t) {
        case 0:return value<uint8_t>(); case 1:return value<int8_t>();case 2:return value<uint16_t>();case 3:return value<int16_t>();
        case 4:return value<uint32_t>();case 5:return value<int32_t>();case 6:return value<float>();case 7:return value<uint8_t>()!=0;
        case 8:return string();case 10:return value<uint64_t>();case 11:return value<int64_t>();case 12:return value<double>();
        case 9:{if(depth)throw std::runtime_error("Nested GGUF arrays unsupported");auto type=value<uint32_t>();auto n=value<uint64_t>();if(n>2000000)throw std::runtime_error("GGUF array too large");
            static const int sizes[]={1,1,2,2,4,4,4,1,0,0,8,8,8};
            if(type>12 || type==9)throw std::runtime_error("Invalid GGUF array type");
            if(sizes[type] && n<=10000){auto values=json::array();for(uint64_t i=0;i<n;++i)values.push_back(item(type,depth+1));return values;}
            if(sizes[type])skip(n*sizes[type]);else for(uint64_t i=0;i<n;++i){auto len=value<uint64_t>();skip(len);}return nullptr;}
        default:throw std::runtime_error("Unknown GGUF metadata type");
        }
    }
};
}
ModelInfo inspect_model(const std::filesystem::path& path) {
    ModelInfo m;m.path=path;m.name=path.stem().u8string();
    try {
        m.bytes=std::filesystem::file_size(path);Reader r(path);
        if(r.value<uint32_t>()!=0x46554747)throw std::runtime_error("Not a GGUF file");
        auto version=r.value<uint32_t>();if(version<2||version>3)throw std::runtime_error("Unsupported GGUF version");
        auto tensors=r.value<uint64_t>();auto count=r.value<uint64_t>();if(count>100000 || tensors>1000000)throw std::runtime_error("Too many GGUF fields");
        std::map<std::string,json> meta;
        for(uint64_t i=0;i<count;++i){auto key=r.string();auto v=r.item(r.value<uint32_t>());if(!v.is_null())meta[key]=std::move(v);}
        if(meta["general.architecture"].is_string())m.architecture=meta["general.architecture"].get<std::string>();
        auto number=[&](const std::string& k,uint64_t fallback=0){auto v=meta[m.architecture+"."+k];return v.is_number_unsigned()?v.get<uint64_t>():fallback;};
        m.context=number("context_length");m.layers=number("block_count");m.embedding=number("embedding_length");m.heads=number("attention.head_count");m.kv_heads=number("attention.head_count_kv",m.heads);m.experts=number("expert_count");m.loops=number("num_loops",1);
        m.key_length=number("attention.key_length",m.heads?m.embedding/m.heads:0);
        m.value_length=number("attention.value_length",m.key_length);
        auto kvHeads=meta[m.architecture+".attention.head_count_kv"];
        auto swa=meta[m.architecture+".attention.sliding_window_pattern"];
        if(kvHeads.is_array() && kvHeads.size()==m.layers && m.layers<10000) {
            uint64_t maxValue=0,keyTotal=0;m.kv_heads=0;bool valid=true;
            for(size_t i=0;i<m.layers;++i) {
                if(!kvHeads[i].is_number_integer() || kvHeads[i].get<int64_t>()<0 || kvHeads[i].get<int64_t>()>100000){valid=false;break;}
                auto heads=kvHeads[i].get<uint64_t>();m.kv_heads=std::max(m.kv_heads,heads);
                bool sliding=swa.is_array() && swa.size()==m.layers && swa[i].is_boolean() && swa[i].get<bool>();
                auto key=sliding?number("attention.key_length_swa",m.key_length):m.key_length;
                auto value=sliding?number("attention.value_length_swa",m.value_length):m.value_length;
                keyTotal+=heads*key;maxValue=std::max(maxValue,heads*value);
            }
            // Non-flash attention pads V to the largest layer. Reserve full
            // context even for sliding layers, rather than underbudgeting them.
            if(valid)m.kv_bytes_per_token=2*(keyTotal+maxValue*m.layers);
        }
        // Tensor offsets account for quantization and padding without loading weights.
        std::vector<std::pair<uint64_t,bool>> offsets;
        for(uint64_t i=0;i<tensors;++i) {
            auto name=r.string();auto dims=r.value<uint32_t>();
            if(dims<1 || dims>4)throw std::runtime_error("Invalid tensor dimensions");
            r.skip(dims*8);r.value<uint32_t>();auto offset=r.value<uint64_t>();
            offsets.emplace_back(offset,name.find("_exps.")!=std::string::npos);
        }
        uint64_t alignment=32;
        if(meta["general.alignment"].is_number_unsigned())alignment=meta["general.alignment"].get<uint64_t>();
        if(!alignment || alignment>1048576)throw std::runtime_error("Invalid tensor alignment");
        uint64_t start=static_cast<uint64_t>(r.in.tellg());start=(start+alignment-1)/alignment*alignment;
        std::sort(offsets.begin(),offsets.end());
        if(!offsets.empty()) {
            if(start>m.bytes)throw std::runtime_error("Missing tensor data");
            for(size_t i=0;i<offsets.size();++i) {
                auto end=i+1<offsets.size()?offsets[i+1].first:m.bytes-start;
                if(end<offsets[i].first || end>m.bytes-start)throw std::runtime_error("Invalid tensor offset");
                if(offsets[i].second)m.expert_bytes+=end-offsets[i].first;
            }
            m.tensor_sizes_known=true;
        }
        m.is_projector=m.architecture=="clip" || lower(m.name).find("mmproj")!=std::string::npos;
        m.valid=!m.architecture.empty();if(!m.valid)m.error="Missing architecture metadata";
        std::smatch parts;auto filename=path.filename().u8string();
        if(std::regex_match(filename,parts,std::regex("(.*)-00001-of-([0-9]{5})(\\.[gG][gG][uU][fF])"))) {
            auto count=std::stoi(parts[2]);if(count<1||count>4096)throw std::runtime_error("Invalid GGUF shard count");m.bytes=0;m.tensor_sizes_known=false;
            for(int i=1;i<=count;++i){std::ostringstream n;n<<parts[1].str()<<'-'<<std::setw(5)<<std::setfill('0')<<i<<"-of-"<<parts[2].str()<<parts[3].str();m.bytes+=std::filesystem::file_size(path.parent_path()/std::filesystem::u8path(n.str()));}
        }
    }catch(const std::exception& e){m.valid=false;m.error=e.what();}
    return m;
}
std::vector<ModelInfo> scan_models(const std::filesystem::path& directory) {
    if(!std::filesystem::is_directory(directory))throw std::runtime_error("Choose an existing model folder");
    std::vector<ModelInfo> models,projectors;std::error_code ec;size_t visited=0;
    for(auto it=std::filesystem::recursive_directory_iterator(directory,std::filesystem::directory_options::skip_permission_denied);it!=std::filesystem::recursive_directory_iterator();it.increment(ec)) {
        if(ec){ec.clear();continue;}if(++visited>100000)throw std::runtime_error("Folder contains too many entries; choose a smaller folder");
        if(!it->is_regular_file(ec)||lower(it->path().extension().u8string())!=".gguf")continue;
        auto name=lower(it->path().filename().u8string());
        // Only the first shard is selectable; llama.cpp resolves its siblings.
        if(std::regex_search(name,std::regex("-[0-9]{5}-of-[0-9]{5}\\.gguf$")) && name.find("-00001-of-")==std::string::npos)continue;
        auto m=inspect_model(it->path());
        if(m.is_projector||name.find("mmproj")!=std::string::npos)projectors.push_back(m);else models.push_back(m);
    }
    for(auto& m:models) {
        std::vector<ModelInfo*> local;
        for(auto& p:projectors)if(p.valid&&p.path.parent_path()==m.path.parent_path())local.push_back(&p);
        auto siblings=std::count_if(models.begin(),models.end(),[&](const ModelInfo& n){return n.path.parent_path()==m.path.parent_path();});
        // A projector next to one model is a strong association. Shared folders
        // with multiple models are left unpaired rather than claiming vision.
        if(local.size()==1 && siblings==1)m.projector=local[0]->path;
        else for(auto* p:local) {
            auto stem=lower(m.name),proj=lower(p->name);
            auto quant=stem.find_last_of('-');if(quant!=std::string::npos && (stem.substr(quant+1,1)=="q"||stem.substr(quant+1,2)=="iq"||stem.substr(quant+1,1)=="f"))stem.resize(quant);
            if(stem.size()>5&&proj.find(stem)!=std::string::npos){if(!m.projector.empty()){m.projector.clear();break;}m.projector=p->path;}
        }
    }
    std::sort(models.begin(),models.end(),[](const ModelInfo& a,const ModelInfo& b){return a.path<b.path;});return models;
}
Hardware detect_hardware() {
    Hardware h;h.threads=std::max(1u,std::thread::hardware_concurrency());h.ram=available_memory();
#ifdef _WIN32
    DWORD coreBytes=0;GetLogicalProcessorInformation(nullptr,&coreBytes);
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> cores(coreBytes/sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if(coreBytes && GetLogicalProcessorInformation(cores.data(),&coreBytes))
        for(const auto& core:cores)if(core.Relationship==RelationProcessorCore)++h.physical_cores;
    // Driver API reports the same first CUDA device used by the server. No GPU
    // context is created and no user workload is disturbed.
    auto dll=LoadLibraryExW(L"nvcuda.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
    if(dll) {
        using Init=int(__stdcall*)(unsigned);using Get=int(__stdcall*)(int*,int);using Name=int(__stdcall*)(char*,int,int);using Mem=int(__stdcall*)(size_t*,int);
        auto init=reinterpret_cast<Init>(GetProcAddress(dll,"cuInit"));auto get=reinterpret_cast<Get>(GetProcAddress(dll,"cuDeviceGet"));auto name=reinterpret_cast<Name>(GetProcAddress(dll,"cuDeviceGetName"));auto mem=reinterpret_cast<Mem>(GetProcAddress(dll,"cuDeviceTotalMem_v2"));
        int device;char label[256]{};size_t bytes=0;
        if(init&&get&&name&&mem&&init(0)==0&&get(&device,0)==0&&name(label,256,device)==0&&mem(&bytes,device)==0){
            h.gpu=label;h.vram=bytes;
            using Attribute=int(__stdcall*)(int*,int,int);
            auto attribute=reinterpret_cast<Attribute>(GetProcAddress(dll,"cuDeviceGetAttribute"));
            int major=0,minor=0;
            if(attribute && attribute(&major,75,device)==0 && attribute(&minor,76,device)==0){h.compute_major=major;h.compute_minor=minor;}
        }
        FreeLibrary(dll);
    }
    auto nvml=LoadLibraryExW(L"nvml.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
    if(nvml) {
        using Init=int(*)();using Handle=int(*)(unsigned,void**);struct Memory {unsigned long long total,free,used;};using Info=int(*)(void*,Memory*);
        auto init=reinterpret_cast<Init>(GetProcAddress(nvml,"nvmlInit_v2"));auto stop=reinterpret_cast<Init>(GetProcAddress(nvml,"nvmlShutdown"));
        auto get=reinterpret_cast<Handle>(GetProcAddress(nvml,"nvmlDeviceGetHandleByIndex_v2"));auto info=reinterpret_cast<Info>(GetProcAddress(nvml,"nvmlDeviceGetMemoryInfo"));
        // Use free-memory telemetry only when there is one visible NVML GPU;
        // otherwise its index need not match CUDA's first device.
        using Count=int(*)(unsigned*);auto count=reinterpret_cast<Count>(GetProcAddress(nvml,"nvmlDeviceGetCount_v2"));
        if(init&&stop&&get&&info&&count&&init()==0){void* device=nullptr;Memory memory{};unsigned n=0;if(h.vram&&count(&n)==0&&n==1&&get(0,&device)==0&&info(device,&memory)==0&&memory.free<=memory.total){h.free_vram=memory.free;h.free_vram_known=true;}stop();}
        FreeLibrary(nvml);
    }
#endif
    return h;
}
static double cache_bytes(const std::string& type) {
    if(type=="q4_0" || type=="iq4_nl")return 18.0/32;
    if(type=="q4_1")return 20.0/32;
    if(type=="q5_0")return 22.0/32;
    if(type=="q5_1")return 24.0/32;
    if(type=="q8_0")return 34.0/32;
    return type=="f32"?4.0:2.0;
}
double kv_cache_bytes(const ModelInfo& m,uint64_t tokens,const std::string& k,const std::string& v) {
    double key=m.key_length?m.key_length:(m.heads?double(m.embedding)/m.heads:128.0);
    double value=m.value_length?m.value_length:key;
    // Aggregate metadata includes per-layer padding. Mixed types use the larger
    // element size as a conservative bound when per-layer K/V totals are absent.
    double perToken=m.kv_bytes_per_token?m.kv_bytes_per_token*std::max(cache_bytes(k),cache_bytes(v))/2:
        m.layers*m.kv_heads*(key*cache_bytes(k)+value*cache_bytes(v));
    return tokens*std::max<uint64_t>(1,m.loops)*perToken;
}
json recommend(const ModelInfo& m,const Hardware& h,uint64_t preferred_context,const std::string& cache_k,const std::string& cache_v) {
    json p;auto d=defaults();
    for(const auto& f:settings())p[f.key]=d[f.key];
    // Hardware recommendations never change the user's inference preferences.
    p["expert_enabled"]="1";
    p["cache_k"]=cache_k;p["cache_v"]=cache_v;
    auto workers=std::max(1u,std::min(h.threads,h.physical_cores?h.physical_cores:(h.threads<=4?h.threads:h.threads/2)));
    bool pascalGpu=h.compute_major==6 || lower(h.gpu).find("p100")!=std::string::npos;
    bool p100=(h.compute_major==6 && h.compute_minor==0) || lower(h.gpu).find("p100")!=std::string::npos;
    p["graphs"]="auto";p["threads"]=std::to_string(workers);p["threads_batch"]=std::to_string(workers);
    p["ctx"]=std::to_string(m.context?std::max<uint64_t>(128,std::min<uint64_t>(4096,m.context)):2048);
    if(preferred_context)p["ctx"]=std::to_string(std::clamp<uint64_t>(preferred_context,128,std::max<uint64_t>(128,m.context?std::min<uint64_t>(m.context,1048576):1048576)));
    p["batch"]="512";p["ubatch"]=workers<=4?"64":"128";
    p["semantic"]="0";p["ctx_checkpoints"]="0";
    p["freetoken_note"]="CPU placement: no usable GPU capacity was detected.";
    constexpr double MiB=1024.0*1024;
    double cpuWeightBytes=m.bytes;
    // Reserve explicit allocations instead of discarding 30% of every GPU.
    if(m.valid&&m.layers&&m.layers<10000&&m.embedding&&m.heads&&m.kv_heads&&m.loops<100&&h.vram) {
        double ctx=std::stod(p["ctx"].get<std::string>());
        double key=m.key_length?m.key_length:double(m.embedding)/m.heads;
        double value=m.value_length?m.value_length:key;
        auto kvFor=[&](double tokens){return kv_cache_bytes(m,static_cast<uint64_t>(tokens),cache_k,cache_v);};
        double available=h.free_vram_known?double(std::min(h.vram,h.free_vram)):h.vram*0.85;
        double reserve=(p100?512.0:640.0)*MiB;
        double scratch=512.0*MiB;
        auto weightBudget=[&](double tokens){return available-reserve-scratch-h.extra_gpu_bytes-kvFor(tokens);};
        // If shortening the default context fits the entire model, avoid CPU
        // execution or expert transfers before selecting a partial placement.
        if(!preferred_context && ctx>2048 && weightBudget(ctx)<m.bytes && weightBudget(2048)>=m.bytes)ctx=2048;
        double budget=weightBudget(ctx);
        if(budget>0 && m.bytes>0) {
            int layers=budget>=m.bytes?999:static_cast<int>(m.layers*std::max(0.0,budget-m.bytes*0.15)/m.bytes);
            p["gpu_layers"]=std::to_string(std::clamp(layers,0,999));
            p["ctx"]=std::to_string(static_cast<uint64_t>(ctx));
            cpuWeightBytes=layers==999?0:m.bytes*(1.0-double(layers)/m.layers);
            p["freetoken_note"]=layers==999?"Model fits in GPU memory; using normal GPU execution without an extra expert cache.":"Partial GPU placement, with memory reserved for context and other applications.";
        }
        bool supported=m.architecture=="llama" || m.architecture=="qwen35moe" || m.architecture=="gemma4";
        if(p["gpu_layers"]!="999" && supported && m.experts && m.tensor_sizes_known && m.expert_bytes && m.expert_bytes<m.bytes) {
            double shared=(m.bytes-m.expert_bytes)*1.03+h.extra_gpu_bytes;
            double arena=64.0*MiB;
            double expertScratch=(h.compute_major>=7?256.0:512.0)*MiB;
            // Reduce context only when needed to retain a useful GPU expert cache.
            auto cacheFor=[&](double tokens){return available-reserve-shared-kvFor(tokens)-expertScratch-arena;};
            double cacheTarget=std::min(double(m.expert_bytes),available*0.40);
            while(!preferred_context && ctx>2048 && cacheFor(ctx)<cacheTarget)ctx=std::max(2048.0,ctx/2);
            // Keep room for runtime allocations absent from GGUF tensor sizes.
            double runtimeCap=h.vram<=8ull*1024*MiB?(available-h.extra_gpu_bytes)*0.50:available;
            double cache=std::min({double(m.expert_bytes),cacheFor(ctx),runtimeCap});
            if(cache>=512*MiB) {
                p["ctx"]=std::to_string(static_cast<uint64_t>(ctx));
                p["gpu_layers"]="999";p["gpu_cache"]=std::to_string(static_cast<uint64_t>(cache/MiB)/64*64);
                p["reserve"]=std::to_string(static_cast<uint64_t>(reserve/MiB));
                p["compute"]="64";p["staging"]="1";p["pipeline"]="1";p["miss"]="0";
                p["fast"]="1";p["no_repack"]="1";p["flash"]="off";
                p["batch"]="512";p["ubatch"]=available>=6*1024*MiB?"128":"32";
                cpuWeightBytes=std::max(0.0,double(m.expert_bytes)-std::stoull(p["gpu_cache"].get<std::string>())*MiB);
                p["freetoken_note"]="GPU attention and shared weights, "+p["gpu_cache"].get<std::string>()+" MiB expert cache, GPU-only expert misses and transfer pipeline. Native kernels may change generated text.";
            }
        }
        p["reserve"]=std::to_string(static_cast<uint64_t>(reserve/MiB));
    }
    if(h.ram && cpuWeightBytes+1024*MiB>h.ram) p["freetoken_note"]=p["freetoken_note"].get<std::string>()+" Limited RAM remains for weights used by the CPU or transferred to the GPU; paging may limit speed.";
    bool quantV=cache_v!="f16" && cache_v!="f32" && cache_v!="bf16";
    if(pascalGpu && quantV) throw std::runtime_error("This Pascal profile requires flash attention off. Use Q4 keys with F16 values, or select a supported attention backend manually.");
    if(quantV)p["flash"]="on";
    if(pascalGpu)p["flash"]="off";
    if(p100) {
        p["graphs"]="off";
        p["freetoken_note"]=p["freetoken_note"].get<std::string>()+" P100 compatibility: CUDA replay and flash attention are off.";
    }
    if(workers<=4 && p["gpu_layers"]!="0")p["freetoken_note"]=p["freetoken_note"].get<std::string>()+" GPU-first placement avoids extra expert work on the "+std::to_string(workers)+"-core CPU.";
    return p;
}
}
namespace ft {
GpuMem gpu_memory() {
    GpuMem g;
#ifdef _WIN32
    struct Mem { unsigned long long total, freeB, used; };
    struct Api {
        bool ok = false;
        HMODULE lib = nullptr;
        void* dev = nullptr;
        int (*mem)(void*, void*) = nullptr;
        Api() {
            lib = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
            if (!lib) return;
            using Init = int(*)(); using Handle = int(*)(unsigned, void**);
            auto init = reinterpret_cast<Init>(GetProcAddress(lib, "nvmlInit_v2"));
            auto get = reinterpret_cast<Handle>(GetProcAddress(lib, "nvmlDeviceGetHandleByIndex_v2"));
            mem = reinterpret_cast<int (*)(void*, void*)>(GetProcAddress(lib, "nvmlDeviceGetMemoryInfo"));
            if (init && get && mem && init() == 0 && get(0, &dev) == 0) ok = true;
        }
    };
    static Api api;
    if (!api.ok) return g;
    Mem m{};
    if (api.mem(api.dev, &m) == 0) { g.total = m.total; g.freeB = m.freeB; g.known = true; }
#endif
    return g;
}
RamMem ram_memory() {
    RamMem r;
#ifdef _WIN32
    MEMORYSTATUSEX m{}; m.dwLength = sizeof(m);
    if (GlobalMemoryStatusEx(&m)) { r.total = m.ullTotalPhys; r.avail = m.ullAvailPhys; }
#else
    std::ifstream in("/proc/meminfo"); std::string key; uint64_t n; std::string unit;
    while (in >> key >> n >> unit) {
        if (key == "MemTotal:") r.total = n * 1024;
        if (key == "MemAvailable:") r.avail = n * 1024;
    }
#endif
    return r;
}
}
