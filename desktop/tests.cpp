#include "core.h"
#include <httplib.h>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <fstream>
#include <algorithm>
#include <atomic>
static void check(bool pass,const char* message) {if(!pass) throw std::runtime_error(message);}
static void fixture(const std::filesystem::path& path,const std::string& arch="llama",bool tensorDetails=false) {
    std::filesystem::create_directories(path.parent_path());std::ofstream out(path,std::ios::binary);
    auto u32=[&](uint32_t n){out.write(reinterpret_cast<char*>(&n),4);};auto u64=[&](uint64_t n){out.write(reinterpret_cast<char*>(&n),8);};
    auto str=[&](const std::string& s){u64(s.size());out.write(s.data(),s.size());};
    u32(0x46554747);u32(3);u64(tensorDetails?3:0);u64(tensorDetails?9:7);
    str("general.architecture");u32(8);str(arch);
    for(auto pair:std::vector<std::pair<std::string,uint32_t>>{{"context_length",8192},{"block_count",32},{"embedding_length",4096},{"attention.head_count",32},{"attention.head_count_kv",8},{"num_loops",1}}){str(arch+"."+pair.first);if(tensorDetails&&pair.first=="attention.head_count_kv"){u32(9);u32(5);u64(32);for(int i=0;i<32;++i)u32(4);}else{u32(4);u32(pair.second);}}
    if(tensorDetails) {
        str(arch+".attention.key_length");u32(4);u32(512);
        str(arch+".attention.value_length");u32(4);u32(256);
        for(auto tensor:std::vector<std::pair<std::string,uint64_t>>{{"token_embd.weight",0},{"blk.0.ffn_up_exps.weight",64},{"output.weight",320}}){str(tensor.first);u32(1);u64(16);u32(0);u64(tensor.second);}
        while(static_cast<uint64_t>(out.tellp())%32)out.put(0);
        for(int i=0;i<384;++i)out.put(0);
    }
}
int main(int argc,char** argv) {
    if(argc>2 && std::string(argv[1])=="--inspect-profile") {
        auto m=ft::inspect_model(std::filesystem::u8path(argv[2]));
        if(!m.valid){std::cerr<<m.error;return 1;}
        auto h=ft::detect_hardware();
        bool simulated=argc>3 && (std::string(argv[3])=="p100-12" || std::string(argv[3])=="p100-16");
        if(simulated){h.threads=4;h.physical_cores=4;h.ram=12ull<<30;h.vram=(std::string(argv[3])=="p100-12"?12ull:16ull)<<30;h.free_vram=h.vram;h.free_vram_known=true;h.compute_major=6;h.compute_minor=0;h.gpu="Tesla P100";}
        if(argc>4)h.extra_gpu_bytes=std::filesystem::file_size(std::filesystem::u8path(argv[4]))+128ull*1024*1024;
        std::cout<<ft::json{{"simulated_hardware",simulated},{"gpu",h.gpu},{"model_bytes",m.bytes},{"expert_bytes",m.expert_bytes},{"kv_bytes_per_token",m.kv_bytes_per_token},{"tensor_sizes_known",m.tensor_sizes_known},{"available_ram",h.ram},{"free_vram",h.free_vram},{"profile",ft::recommend(m,h)}}.dump(2);
        return 0;
    }
    if(argc>1 && std::string(argv[1])=="--model") {
        int port=0;for(int i=1;i+1<argc;++i) if(std::string(argv[i])=="--port") port=std::stoi(argv[i+1]);
        httplib::Server mock;
        mock.Get("/health",[](const httplib::Request&,httplib::Response& r){r.set_content("{\"status\":\"ok\"}","application/json");});
        mock.Post("/v1/chat/completions",[](const httplib::Request& req,httplib::Response& r){
            auto j=ft::json::parse(req.body);
            if(!j.contains("messages") || j["messages"].empty()) {r.status=400;r.set_content("missing messages","text/plain");return;}
            for(auto& message:j["messages"]) if(!message.is_object() || !message.contains("role") || !message.contains("content")) {
                r.status=400;r.set_content("malformed message object","text/plain");return;
            }
            bool slow=j["messages"].back().value("content",std::string())=="Cancel this streaming request.";
            r.set_chunked_content_provider("text/event-stream",[slow](size_t offset,httplib::DataSink& sink){
                if(slow) std::this_thread::sleep_for(std::chrono::milliseconds(20));
                std::string token=slow?"partial ":(offset==0?"Native ":"stream verified.");
                auto event=ft::json{{"choices",ft::json::array({{{"delta",{{"content",token}}}}})}};
                event["timings"]={{"predicted_per_second",25.0},{"predicted_n",2}};
                auto wire="data: "+event.dump()+"\n\n";sink.write(wire.data(),wire.size());
                if(!slow && offset!=0) {std::string done="data: [DONE]\n\n";sink.write(done.data(),done.size());sink.done();}
                return true;
            });
        });
        return mock.listen("127.0.0.1",port)?0:1;
    }
    if(argc>1 && std::string(argv[1])=="--child") {std::cout<<"child started"<<std::endl;std::this_thread::sleep_for(std::chrono::seconds(30));return 0;}
    try {
        int count=0;ft::Events events([&](const ft::json& j) {check(j.at("text")=="hello","SSE payload");++count;});
        std::string wire=": heartbeat\r\ndata: {\"text\":\"hello\"}\r\n\r\ndata: [DONE]\n\n";
        for(char c:wire) events.feed(&c,1);
        check(count==1 && events.done,"Fragmented SSE framing");
        auto c=ft::defaults();ft::validate(c);
        c["max_tokens"]="-1";ft::validate(c);
        c["seed"]="123";c["frequency_penalty"]="0.4";c["stop_strings"]="END\nSTOP";
        c["json_schema"]="{\"type\":\"object\"}";
        auto inference=ft::inference_options(c);
        check(inference["seed"]==123 && inference["frequency_penalty"]==0.4 && inference["stop"].size()==2,"Inference options preserve numeric types and stop strings");
        check(inference["response_format"]["json_schema"]["schema"]["type"]=="object","Structured output reaches engine request");
        check(inference["dry_penalty_last_n"]==4096,"Full-context DRY window translates to nonnegative engine value");
        c=ft::defaults();c["max_tokens"]="0";bool badLimit=false;try{ft::validate(c);}catch(...){badLimit=true;}check(badLimit,"Zero output limit rejected; unlimited uses -1");
        c=ft::defaults();
        c["cpu_cache"]="512";c["gpu_cache"]="512";bool rejected=false;
        try {ft::validate(c);} catch(...) {rejected=true;} check(rejected,"Reject incompatible CPU/GPU cache");
        c=ft::defaults();c["temperature"]="nan";rejected=false;
        try {ft::validate(c);} catch(...) {rejected=true;}check(rejected,"Reject nonfinite temperature");
        check(ft::quote_windows("C:\\space dir\\")=="\"C:\\space dir\\\\\"","Trailing slash Windows quoting");
        auto dir=std::filesystem::temp_directory_path()/ ("ft-desktop-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        auto file=dir/"config.json";ft::save_json(file,ft::defaults());check(ft::read_json(file,{})==ft::defaults(),"Settings round trip");
        auto models=dir/"models";
        fixture(models/"vision"/"model-Q4_0.gguf");fixture(models/"vision"/"mmproj.gguf","clip");
        fixture(models/"shared"/"alpha.gguf");fixture(models/"shared"/"beta.gguf");fixture(models/"shared"/"mmproj.gguf","clip");
        fixture(models/"split-00001-of-00002.gguf");fixture(models/"split-00002-of-00002.gguf");
        std::ofstream(models/"broken.gguf")<<"GGUF";
        auto found=ft::scan_models(models);check(found.size()==5,"Scanner excludes projector files and later shards");
        int vision=0,invalid=0;for(auto& m:found){if(!m.projector.empty())++vision;if(!m.valid)++invalid;}
        check(vision==1&&invalid==1,"Projector association avoids ambiguous folders and reports invalid GGUF");
        fixture(models/"tensor-layout.gguf","gemma4",true);
        auto layout=ft::inspect_model(models/"tensor-layout.gguf");
        check(layout.valid && layout.tensor_sizes_known && layout.expert_bytes==256,"Tensor offsets identify expert weight bytes");
        check(layout.kv_heads==4 && layout.kv_bytes_per_token==196608,"Per-layer KV head arrays do not fall back to query head counts");
        auto model=ft::inspect_model(models/"vision"/"model-Q4_0.gguf");check(model.valid&&model.context==8192&&model.layers==32,"GGUF dimensions");
        ft::Hardware hardware;hardware.threads=16;hardware.ram=32ull<<30;hardware.vram=8ull<<30;
        model.bytes=2ull<<30;auto recommended=ft::recommend(model,hardware);check(recommended["gpu_layers"]=="999"&&recommended["ctx"]=="4096","Fitting model recommendation");
        auto denseProfile=ft::freetoken_profile(model,hardware);
        check(denseProfile["gpu_cache"]=="" && !denseProfile.contains("max_tokens"),"Dense model profile skips experts and preserves output preference");
        auto moe=model;moe.experts=8;moe.bytes=14ull<<30;moe.expert_bytes=12ull<<30;moe.tensor_sizes_known=true;
        auto gpuProfile=ft::freetoken_profile(moe,hardware);auto profileConfig=ft::defaults();profileConfig.update(gpuProfile);ft::validate(profileConfig);
        check(gpuProfile["pipeline"]=="1" && gpuProfile["semantic"]=="0" && gpuProfile["fast"]=="1" && gpuProfile["miss"]=="0" && gpuProfile["gpu_layers"]=="999" && gpuProfile["no_kv"]=="0","Oversized MoE uses GPU-resident native execution, not CPU checkpoints");
        check(ft::recommend(moe,hardware)==gpuProfile,"Both automatic entry points use the same policy");
        auto busy=hardware;busy.free_vram_known=true;busy.free_vram=5ull<<30;
        auto busyProfile=ft::recommend(moe,busy);
        check(std::stoi(busyProfile["gpu_cache"].get<std::string>())<std::stoi(gpuProfile["gpu_cache"].get<std::string>()),"Other GPU workloads reduce the expert budget");
        auto large=hardware;large.vram=32ull<<30;auto largeProfile=ft::recommend(moe,large);
        check(largeProfile["gpu_layers"]=="999" && largeProfile["gpu_cache"]=="","Fitting MoE avoids redundant expert caching");
        auto p100=hardware;p100.gpu="Tesla P100";check(ft::recommend(moe,p100)["graphs"]=="off","P100 retains replay guard");
        auto target=moe;target.architecture="gemma4";target.bytes=14439363584ull;target.expert_bytes=12846382080ull;target.kv_bytes_per_token=245760;
        p100.threads=4;p100.physical_cores=4;p100.vram=16ull<<30;p100.free_vram=p100.vram;p100.free_vram_known=true;p100.compute_major=6;p100.compute_minor=0;
        auto target16=ft::recommend(target,p100);
        check(target16["gpu_layers"]=="999" && target16["gpu_cache"]=="","16 GiB P100 keeps fitting Gemma entirely on GPU");
        check(target16["threads"]=="4" && target16["threads_batch"]=="4" && target16["flash"]=="off" && target16["graphs"]=="off","N150 workers and P100 compatibility are preserved on the normal GPU path");
        p100.extra_gpu_bytes=1194828160ull+128ull*1024*1024;
        auto targetVision=ft::recommend(target,p100);
        check(targetVision["gpu_cache"]!="" && std::stoi(targetVision["gpu_cache"].get<std::string>())>11000 && targetVision["miss"]=="0","16 GiB P100 with vision prioritizes a large expert cache over N150 execution");
        p100.vram=12ull<<30;p100.free_vram=p100.vram;
        auto target12=ft::recommend(target,p100);
        check(std::stoi(target12["gpu_cache"].get<std::string>())<std::stoi(targetVision["gpu_cache"].get<std::string>()) && target12["gpu_layers"]=="999","12 GiB P100 uses a smaller cache with GPU attention");
        auto n150=p100;n150.physical_cores=0;check(ft::recommend(target,n150)["threads"]=="4","Four non-SMT threads are not halved when topology is unavailable");
        auto renamed=p100;renamed.gpu="CUDA device";check(ft::recommend(target,renamed)["graphs"]=="off","SM60 guard does not depend on the marketing name");
        auto laptop=hardware;laptop.compute_major=8;laptop.compute_minor=9;laptop.free_vram_known=true;laptop.free_vram=(8ull<<30)-(256ull<<20);laptop.extra_gpu_bytes=p100.extra_gpu_bytes;
        auto laptopProfile=ft::recommend(target,laptop,2048);
        check(laptopProfile["graphs"]=="auto" && laptopProfile["ubatch"]=="128" && std::stoi(laptopProfile["gpu_cache"].get<std::string>())>=3000 && std::stoull(laptopProfile["gpu_cache"].get<std::string>())*1024*1024<=(laptop.free_vram-laptop.extra_gpu_bytes)/2,"RTX 4060 reserves runtime headroom without inheriting P100 restrictions");
        auto longContext=ft::recommend(target,laptop,8192);
        check(longContext["ctx"]=="8192","Explicit context preference is not shortened to inflate expert cache");
        auto incomplete=moe;incomplete.tensor_sizes_known=false;check(ft::recommend(incomplete,hardware)["gpu_cache"]=="","Unknown tensor layout uses conservative placement");
        check(!gpuProfile.contains("thinking") && !gpuProfile.contains("temperature"),"Tuning preserves inference preferences");
        auto visionHardware=hardware;visionHardware.extra_gpu_bytes=1ull<<30;
        auto visionProfile=ft::recommend(moe,visionHardware);
        check(std::stoi(visionProfile["gpu_cache"].get<std::string>())<std::stoi(gpuProfile["gpu_cache"].get<std::string>()),"Vision component reduces expert cache budget");
        auto noGpu=hardware;noGpu.vram=0;
        auto cpuProfile=ft::freetoken_profile(moe,noGpu);profileConfig=ft::defaults();profileConfig.update(cpuProfile);ft::validate(profileConfig);
        check(cpuProfile["cpu_cache"]=="" && cpuProfile["pipeline"]=="0" && cpuProfile["gpu_layers"]=="0" && cpuProfile["semantic"]=="0","CPU fallback avoids extra caches and checkpoint copies");
        for(uint64_t gib:{0,2,4,8,16,24,48})for(int freePercent:{0,25,75,100}) {
            auto device=hardware;device.vram=gib<<30;device.free_vram_known=true;device.free_vram=device.vram*freePercent/100;
            auto profile=ft::recommend(moe,device);auto config=ft::defaults();config.update(profile);ft::validate(config);
            if(profile["gpu_cache"]!="")check(std::stoull(profile["gpu_cache"].get<std::string>())*1024*1024<device.free_vram,"Expert cache respects available GPU memory");
            check(profile["semantic"]=="0" && profile["cpu_cache"]=="","Hardware matrix never enables checkpoint copies or duplicate CPU caches");
        }
        auto q4=ft::recommend(moe,hardware,4096,"q4_0","q4_0");
        check(q4["cache_k"]=="q4_0" && q4["cache_v"]=="q4_0" && q4["flash"]=="on","Q4 profile preserves types and enables required attention");
        check(ft::kv_cache_bytes(moe,4096,"q4_0","q4_0")<ft::kv_cache_bytes(moe,4096,"f16","f16"),"Q4 reduces KV estimate");
        check(ft::kv_cache_bytes(moe,4096,"q4_0","f16")>=ft::kv_cache_bytes(moe,4096,"q4_0","q4_0"),"Mixed cache cannot be counted as all Q4");
        auto pascalHardware=hardware;pascalHardware.compute_major=6;pascalHardware.gpu="Tesla P100";
        bool badKv=false;try{ft::recommend(moe,pascalHardware,4096,"q4_0","q4_0");}catch(...){badKv=true;}
        check(badKv,"Pascal auto rejects quantized V with unsupported attention profile");
        check(ft::recommend(moe,pascalHardware,4096,"q4_0","f16")["flash"]=="off","Pascal allows Q4 keys with F16 values");
        hardware.vram=0;auto cpu=ft::recommend(model,hardware);check(cpu["gpu_layers"]=="0","Unknown GPU uses CPU baseline");
        auto unknown=model;unknown.heads=0;hardware.vram=8ull<<30;check(ft::recommend(unknown,hardware)["gpu_layers"]=="0","Incomplete model metadata prevents speculative GPU fit");
        hardware.free_vram_known=true;hardware.free_vram=256ull<<20;check(ft::recommend(model,hardware)["gpu_layers"]=="0","Busy GPU memory reduces automatic offload");
        c=ft::defaults();c["recommended"]=recommended;c["recommended_model"]=model.path.u8string();c["ctx"]="1024";ft::save_json(file,c);
        auto restored=ft::read_json(file,{});check(restored["ctx"]=="1024"&&restored["recommended"]["ctx"]=="4096","Overrides retain original recommendation after restart");
        c["server"]=std::filesystem::absolute(argv[0]).u8string();c["model"]=model.path.u8string();c["mmproj"]=(models/"vision"/"mmproj.gguf").u8string();
        auto args=ft::arguments(c);check(std::find(args.begin(),args.end(),"--mmproj")!=args.end(),"Projector is passed to server");
        c["expert_enabled"]="0";c["gpu_cache"]="1024";c["cpu_cache"]="1024";args=ft::arguments(c);
        check(std::none_of(args.begin(),args.end(),[](const std::string& s){return s.rfind("--moe-",0)==0;}),"FreeToken off suppresses expert options while retaining budgets");
        c=ft::defaults();c["ctx"]="8192";ft::save_json(file,c);check(ft::read_json(file,{}).at("ctx")=="8192","Atomic replacement");
        std::atomic<bool> child_output{false};
        ft::Process child;child.start({std::filesystem::absolute(argv[0]).u8string(),"--child"},dir/"child.log",true,
            [&](const std::string& line){if(line=="child started") child_output=true;});
        check(child.running(),"Child launch");
        for(int i=0;i<100 && !child_output;++i) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        child.stop();check(!child.running(),"Owned child cleanup");
        check(child_output,"Child output reaches live log callback");
        httplib::Server server;server.Get("/health",[](const httplib::Request&,httplib::Response& r){r.set_content("{\"status\":\"ok\"}","application/json");});
        int port=server.bind_to_any_port("127.0.0.1");check(port>0,"Mock port binding");
        std::thread thread([&]{server.listen_after_bind();});server.wait_until_ready();
        httplib::Client client("127.0.0.1",port);auto response=client.Get("/health");server.stop();thread.join();
        check(response && response->status==200,"Loopback HTTP");
        std::filesystem::remove_all(models);std::filesystem::remove(file);std::filesystem::remove(dir/"child.log");std::filesystem::remove(dir);
        std::cout<<"PASS: SSE, settings constraints, argument quoting, persistence, process lifecycle, HTTP\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

