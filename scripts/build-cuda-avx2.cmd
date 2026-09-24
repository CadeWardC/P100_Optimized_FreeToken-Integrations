@echo off
setlocal
if not defined FT_VS_ROOT set "FT_VS_ROOT=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "%FT_VS_ROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set "FT_BUILD=%~dp0..\build\ft-windows-cuda-dev"
set "FT_BIN=%FT_BUILD%\bin"
rem Retain the portable server. The desktop selects AVX2 only on compatible CPUs.
if not exist "%FT_BIN%\llama-server-portable.exe" copy /y "%FT_BIN%\llama-server.exe" "%FT_BIN%\llama-server-portable.exe" >nul
if errorlevel 1 exit /b 1
cmake -S "%~dp0..\llama.cpp" -B "%FT_BUILD%" -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_BMI2=ON -DGGML_CUDA_GRAPHS=ON
if errorlevel 1 exit /b 1
cmake --build "%FT_BUILD%" --target llama-server test-moe-cache test-moe-offload test-moe-integration test-backend-ops --parallel 4
if errorlevel 1 goto restore_failure
copy /y "%FT_BIN%\llama-server.exe" "%FT_BIN%\llama-server-avx2.exe" >nul
if errorlevel 1 goto restore_failure
copy /y "%FT_BIN%\llama-server-portable.exe" "%FT_BIN%\llama-server.exe" >nul
exit /b %errorlevel%
:restore_failure
copy /y "%FT_BIN%\llama-server-portable.exe" "%FT_BIN%\llama-server.exe" >nul
exit /b 1
