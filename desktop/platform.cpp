#include "core.h"
#include <stdexcept>
#include <chrono>
#include <thread>
#ifdef _WIN32
#include <windows.h>
#else
#include <spawn.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
extern char** environ;
#endif

namespace ft {
struct Process::Impl {
#ifdef _WIN32
    HANDLE process=nullptr, job=nullptr;
#else
    pid_t pid=-1;
#endif
    std::thread reader;
};
Process::Process():impl(new Impl) {}
Process::~Process() {stop();}
#ifdef _WIN32
static std::wstring wide(const std::string& s) {
    int n=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,s.data(),static_cast<int>(s.size()),nullptr,0);
    if(n==0 && !s.empty()) throw std::runtime_error("Invalid UTF-8 path");
    std::wstring out(n,0); MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,s.data(),static_cast<int>(s.size()),&out[0],n); return out;
}
#endif
static void dispatch_lines(const std::string& chunk,std::string& pending,const std::function<void(const std::string&)>& on_line) {
    if(!on_line) return;
    pending.append(chunk);
    size_t p;
    while((p=pending.find('\n'))!=std::string::npos) {
        auto line=pending.substr(0,p); pending.erase(0,p+1);
        if(!line.empty() && line.back()=='\r') line.pop_back();
        if(!line.empty()) on_line(line);
    }
}
void Process::start(const std::vector<std::string>& args,const std::filesystem::path& log,bool graphs_off,
                    const std::function<void(const std::string&)>& on_line) {
    stop(); if(args.empty()) throw std::runtime_error("Empty process arguments");
    std::filesystem::create_directories(log.parent_path());
#ifdef _WIN32
    SECURITY_ATTRIBUTES sa{sizeof(sa),nullptr,TRUE};
    HANDLE out=CreateFileW(log.c_str(),GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE,&sa,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(out==INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot open server log");
    HANDLE input=CreateFileW(L"NUL",GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,&sa,OPEN_EXISTING,0,nullptr);
    if(input==INVALID_HANDLE_VALUE) {CloseHandle(out);throw std::runtime_error("Cannot open null input");}
    HANDLE pipe_read=nullptr,pipe_write=nullptr;
    if(on_line && !CreatePipe(&pipe_read,&pipe_write,&sa,1<<16)) {CloseHandle(out);CloseHandle(input);throw std::runtime_error("Cannot create log pipe");}
    HANDLE child_out=pipe_write?pipe_write:out;
    std::string cmd; for(auto& arg:args) {if(!cmd.empty()) cmd+=' ';cmd+=quote_windows(arg);}
    auto command=wide(cmd); auto exe=wide(args[0]);
    std::vector<wchar_t> env; auto raw=GetEnvironmentStringsW();
    if(!raw) {CloseHandle(out);CloseHandle(input);if(pipe_write)CloseHandle(pipe_write);if(pipe_read)CloseHandle(pipe_read);throw std::runtime_error("Cannot read environment");}
    for(auto p=raw;*p;p+=wcslen(p)+1) {
        if(_wcsnicmp(p,L"GGML_CUDA_DISABLE_GRAPHS=",25)==0) continue;
        env.insert(env.end(),p,p+wcslen(p)+1);
    }
    FreeEnvironmentStringsW(raw);
    if(graphs_off) {std::wstring flag=L"GGML_CUDA_DISABLE_GRAPHS=1";env.insert(env.end(),flag.begin(),flag.end());env.push_back(0);}
    env.push_back(0);
    auto job=CreateJobObjectW(nullptr,nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{}; limits.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if(!job || !SetInformationJobObject(job,JobObjectExtendedLimitInformation,&limits,sizeof(limits))) {
        if(job) CloseHandle(job);CloseHandle(out);CloseHandle(input);
        if(pipe_write)CloseHandle(pipe_write); if(pipe_read)CloseHandle(pipe_read);
        throw std::runtime_error("Cannot create server job");
    }
    STARTUPINFOW si{}; si.cb=sizeof(si);si.dwFlags=STARTF_USESTDHANDLES;si.hStdOutput=child_out;si.hStdError=child_out;si.hStdInput=input;
    PROCESS_INFORMATION pi{};
    BOOL ok=CreateProcessW(exe.c_str(),command.data(),nullptr,nullptr,TRUE,CREATE_NO_WINDOW|CREATE_SUSPENDED|CREATE_UNICODE_ENVIRONMENT,env.data(),nullptr,&si,&pi);
    CloseHandle(input);
    if(pipe_write) CloseHandle(pipe_write); // the child owns its copy now; EOF requires ours gone
    if(!ok) {auto err=GetLastError();CloseHandle(out);CloseHandle(job);if(pipe_read)CloseHandle(pipe_read);
        throw std::runtime_error("Server launch failed, Windows error "+std::to_string(err));}
    if(!AssignProcessToJobObject(job,pi.hProcess)) {TerminateProcess(pi.hProcess,1);CloseHandle(pi.hProcess);CloseHandle(pi.hThread);CloseHandle(job);CloseHandle(out);
        if(pipe_read)CloseHandle(pipe_read);
        throw std::runtime_error("Cannot attach server to process job");}
    impl->process=pi.hProcess;impl->job=job;ResumeThread(pi.hThread);CloseHandle(pi.hThread);
    if(pipe_read) {
        impl->reader=std::thread([pipe_read,out,on_line] {
            char buf[8192];std::string pending;DWORD n=0,written=0;
            while(ReadFile(pipe_read,buf,sizeof buf,&n,nullptr) && n) {
                std::string chunk(buf,n);
                if(!WriteFile(out,buf,n,&written,nullptr)) break;
                try {dispatch_lines(chunk,pending,on_line);} catch(...) {}
            }
            if(!pending.empty()) {WriteFile(out,pending.data(),(DWORD)pending.size(),&written,nullptr);if(on_line)on_line(pending);}
            CloseHandle(pipe_read);CloseHandle(out);
        });
    } else CloseHandle(out);
#else
    int fds[2]={-1,-1};
    if(on_line && pipe(fds)!=0) throw std::runtime_error("Cannot create log pipe");
    int out_fd=::open(log.c_str(),O_WRONLY|O_CREAT|O_TRUNC,0600);
    if(out_fd<0) {if(fds[0]>=0){close(fds[0]);close(fds[1]);}throw std::runtime_error("Cannot open server log");}
    int child_out=fds[1]>=0?fds[1]:out_fd;
    std::vector<char*> argv; for(auto& a:args) argv.push_back(const_cast<char*>(a.c_str()));argv.push_back(nullptr);
    std::vector<std::string> env;
    for(char** e=environ;*e;++e) if(std::string(*e).rfind("GGML_CUDA_DISABLE_GRAPHS=",0)!=0) env.emplace_back(*e);
    if(graphs_off) env.emplace_back("GGML_CUDA_DISABLE_GRAPHS=1");
    std::vector<char*> envp;for(auto& e:env) envp.push_back(e.data());envp.push_back(nullptr);
    posix_spawn_file_actions_t actions;posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions,STDIN_FILENO,"/dev/null",O_RDONLY,0);
    posix_spawn_file_actions_adddup2(&actions,child_out,STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions,child_out,STDERR_FILENO);
    if(fds[0]>=0) posix_spawn_file_actions_addclose(&actions,fds[0]);
    if(fds[1]>STDERR_FILENO) posix_spawn_file_actions_addclose(&actions,fds[1]);
    if(out_fd>STDERR_FILENO && out_fd!=fds[1]) posix_spawn_file_actions_addclose(&actions,out_fd);
    posix_spawnattr_t attr;posix_spawnattr_init(&attr);
    posix_spawnattr_setflags(&attr,POSIX_SPAWN_SETPGROUP);posix_spawnattr_setpgroup(&attr,0);
    int error=posix_spawn(&impl->pid,args[0].c_str(),&actions,&attr,argv.data(),envp.data());
    posix_spawnattr_destroy(&attr);posix_spawn_file_actions_destroy(&actions);
    if(fds[1]>=0) close(fds[1]);
    if(error) {close(out_fd);if(fds[0]>=0)close(fds[0]);impl->pid=-1;throw std::runtime_error("Server launch failed: "+std::to_string(error));}
    if(fds[0]>=0) {
        impl->reader=std::thread([fds0=fds[0],out_fd,on_line] {
            char buf[8192];std::string pending;ssize_t n=0;
            while((n=::read(fds0,buf,sizeof buf))>0) {
                std::string chunk(buf,(size_t)n);
                if(::write(out_fd,buf,(size_t)n)<0) break;
                try {dispatch_lines(chunk,pending,on_line);} catch(...) {}
            }
            if(!pending.empty()) {if(::write(out_fd,pending.data(),pending.size())>=0 && on_line)on_line(pending);}
            close(fds0);close(out_fd);
        });
    } else close(out_fd);
#endif
}
bool Process::running() {
#ifdef _WIN32
    return impl->process && WaitForSingleObject(impl->process,0)==WAIT_TIMEOUT;
#else
    if(impl->pid<0) return false;
    int status;auto r=waitpid(impl->pid,&status,WNOHANG);if(r==impl->pid) {impl->pid=-1;return false;}return r==0;
#endif
}
void Process::stop() {
#ifdef _WIN32
    if(impl->job) {CloseHandle(impl->job);impl->job=nullptr;}
    if(impl->process) {WaitForSingleObject(impl->process,2000);CloseHandle(impl->process);impl->process=nullptr;}
#else
    if(impl->pid>0) {
        kill(-impl->pid,SIGTERM);
        for(int i=0;i<20 && running();++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if(impl->pid>0) {kill(-impl->pid,SIGKILL);waitpid(impl->pid,nullptr,0);impl->pid=-1;}
    }
#endif
    if(impl->reader.joinable()) impl->reader.join();
}
}
