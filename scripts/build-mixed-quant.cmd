@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cmake --build "%~dp0..\build\ft-windows-cuda-dev" --target test-moe-cache test-moe-offload test-moe-integration llama-server --parallel 4
