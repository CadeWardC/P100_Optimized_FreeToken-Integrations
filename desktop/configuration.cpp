#include "core.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace ft {
json inference_defaults() {
    return {{"seed","-1"},{"frequency_penalty","0"},{"presence_penalty","0"},
        {"repeat_last_n","64"},{"typical_p","1"},{"xtc_probability","0"},{"xtc_threshold","0.1"},
        {"mirostat","0"},{"mirostat_tau","5"},{"mirostat_eta","0.1"},
        {"dynatemp_range","0"},{"dynatemp_exponent","1"},{"dry_multiplier","0"},
        {"dry_base","1.75"},{"dry_allowed_length","2"},{"dry_penalty_last_n","-1"},
        {"reasoning_budget","-1"},{"reasoning_message",""},{"stop_strings",""},
        {"json_schema",""},{"grammar",""},{"tools_json",""},{"show_output_tokens","1"}};
}
static double numeric(const json& c, const char* key, double lo, double hi, bool integer=false, bool optional=false) {
    auto s=c.at(key).get<std::string>();
    if(optional && s.empty()) return 0;
    size_t n=0; double v;
    try { v=std::stod(s,&n); } catch(...) { throw std::runtime_error(std::string(key)+": enter a number"); }
    if(n!=s.size() || !std::isfinite(v) || v<lo || v>hi || (integer && v!=std::floor(v)))
        throw std::runtime_error(std::string(key)+": outside supported range");
    return v;
}
void validate_extended(const json& c) {
    for(auto key:{"main_gpu","cpu_moe_layers","ctx_checkpoints","draft_gpu_layers","draft_min","draft_max"}) numeric(c,key,0,1048576,true);
    numeric(c,"parallel",1,64,true);
    if(std::stoi(c.at("draft_min").get<std::string>())>std::stoi(c.at("draft_max").get<std::string>())) throw std::runtime_error("Minimum draft tokens cannot exceed maximum");
    for(auto key:{"rope_base","rope_scale"}) numeric(c,key,0.000001,1e12,false,true);
    numeric(c,"seed",-1,4294967295.0,true);
    for(auto key:{"repeat_last_n","dry_penalty_last_n","reasoning_budget"}) numeric(c,key,-1,1048576,true);
    numeric(c,"dry_allowed_length",0,1048576,true);
    numeric(c,"mirostat",0,2,true);
    for(auto key:{"typical_p","xtc_probability","xtc_threshold","draft_p_min"}) numeric(c,key,0,1);
    for(auto key:{"presence_penalty","frequency_penalty"}) numeric(c,key,-2,2);
    for(auto key:{"mirostat_tau","mirostat_eta","dynatemp_range","dynatemp_exponent","dry_multiplier","dry_base"}) numeric(c,key,0,100);
    for(auto pair:std::vector<std::pair<const char*,std::vector<std::string>>>{
        {"fit",{"on","off"}},{"split_mode",{"none","layer","row"}},
        {"reasoning_format",{"none","deepseek","deepseek-legacy"}},
        {"cache_k",{"f32","f16","bf16","q8_0","q4_0","q4_1","q5_0","q5_1","iq4_nl"}},
        {"cache_v",{"f32","f16","bf16","q8_0","q4_0","q4_1","q5_0","q5_1","iq4_nl"}}}) {
        auto value=c.at(pair.first).get<std::string>();
        if(std::find(pair.second.begin(),pair.second.end(),value)==pair.second.end()) throw std::runtime_error(std::string("Invalid ")+pair.first);
    }
    for(auto& f:settings()) if(f.toggle && c.at(f.key)!="0" && c.at(f.key)!="1") throw std::runtime_error(f.label+": use on or off");
    for(auto key:{"thinking","expert_enabled","auto_settings","show_output_tokens"})
        if(c.at(key)!="0" && c.at(key)!="1") throw std::runtime_error(std::string("Invalid ")+key);
    auto ratios=c.at("tensor_split").get<std::string>();
    if(!ratios.empty()) {
        std::istringstream in(ratios); std::string part; double total=0;
        if(ratios.back()==',') throw std::runtime_error("GPU allocation ratios: trailing comma");
        while(std::getline(in,part,',')) { json temp{{"ratio",part}}; total+=numeric(temp,"ratio",0,1000000); }
        if(total<=0) throw std::runtime_error("GPU allocation ratios must include a positive value");
    }
    for(auto key:{"json_schema","tools_json"}) if(c.at(key)!="") {
        auto parsed=json::parse(c.at(key).get<std::string>());
        if(std::string(key)=="json_schema" ? !parsed.is_object() : !parsed.is_array()) throw std::runtime_error(std::string(key)+": invalid JSON type");
    }
    if(c.at("json_schema")!="" && c.at("grammar")!="") throw std::runtime_error("Choose either JSON schema or grammar, not both");
    if(c.at("draft_model")!="" && !std::filesystem::is_regular_file(std::filesystem::u8path(c.at("draft_model").get<std::string>()))) throw std::runtime_error("Draft model file does not exist");
    bool enabled=c.at("expert_enabled")=="1";
    bool custom=enabled && (c.at("gpu_cache")!="" || c.at("cpu_cache")!="" || c.at("cpu_tile")!="");
    if(custom && (c.at("parallel")!="1" || c.at("fit")!="off" || c.at("draft_model")!="" || c.at("cpu_moe")=="1" || c.at("cpu_moe_layers")!="0"))
        throw std::runtime_error("FreeToken caching requires one session, memory fit off, no draft model and standard CPU expert offload disabled");
    if(enabled && c.at("gpu_cache")=="" && (c.at("fast")=="1" || c.at("pipeline")=="1" || c.at("miss")!="0")) throw std::runtime_error("FreeToken pipeline, fast mode and CPU miss sharing require a GPU expert cache");
    if(enabled && c.at("semantic")=="1" && (!custom || c.at("ctx_checkpoints")=="0" || c.at("draft_model")!="" || c.at("no_kv")!="1" || c.at("no_op")!="1")) throw std::runtime_error("Semantic checkpoints require expert caching, a positive checkpoint count, CPU KV/operations and no draft model");
}
json inference_options(const json& c) {
    json result;
    for(auto key:{"seed","repeat_last_n","dry_allowed_length","dry_penalty_last_n","mirostat"}) result[key]=std::stoll(c.at(key).get<std::string>());
    for(auto key:{"repeat_last_n","dry_penalty_last_n"}) if(result[key]==-1) result[key]=std::stoi(c.at("ctx").get<std::string>());
    for(auto key:{"frequency_penalty","presence_penalty","typical_p","xtc_probability","xtc_threshold","mirostat_tau","mirostat_eta","dynatemp_range","dynatemp_exponent","dry_multiplier","dry_base"}) result[key]=std::stod(c.at(key).get<std::string>());
    result["reasoning_budget_tokens"]=std::stoi(c.at("reasoning_budget").get<std::string>());
    result["reasoning_budget_message"]=c.at("reasoning_message");
    json stops=json::array(); std::istringstream lines(c.at("stop_strings").get<std::string>()); std::string line;
    while(std::getline(lines,line)) { if(!line.empty() && line.back()=='\r') line.pop_back(); if(!line.empty()) stops.push_back(line); }
    if(!stops.empty()) result["stop"]=stops;
    if(c.at("json_schema")!="") result["response_format"]={{"type","json_schema"},{"json_schema",{{"name","response"},{"schema",json::parse(c.at("json_schema").get<std::string>())}}}};
    if(c.at("grammar")!="") result["grammar"]=c.at("grammar");
    if(c.at("tools_json")!="") result["tools"]=json::parse(c.at("tools_json").get<std::string>());
    return result;
}
json freetoken_profile(const ModelInfo& m,const Hardware& h,uint64_t preferred_context,const std::string& cache_k,const std::string& cache_v) {
    if(!m.valid) throw std::runtime_error("Select a readable model first");
    auto p=recommend(m,h,preferred_context,cache_k,cache_v);
    // Profile owns load settings, not the user's sampling or response-length choices.
    for(auto key:{"temperature","max_tokens","thinking","top_k","top_p","min_p","repeat_penalty"}) p.erase(key);
    return p;
}
}
