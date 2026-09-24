@echo off
setlocal
if not defined FT_CUDA_GRAPHS set "FT_CUDA_GRAPHS=OFF"
if not defined FT_VS_ROOT set "FT_VS_ROOT=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
if not defined FT_CUDA_ROOT set "FT_CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
call "%FT_VS_ROOT%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
set "PATH=%FT_VS_ROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
cmake -S "%~dp0..\llama.cpp" -B "%~dp0..\build\ft-windows-cuda-dev" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_BMI2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=%FT_CUDA_GRAPHS% "-DCMAKE_CUDA_ARCHITECTURES=60;89" "-DCMAKE_CUDA_COMPILER=%FT_CUDA_ROOT%/bin/nvcc.exe" "-DCUDAToolkit_ROOT=%FT_CUDA_ROOT%" -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF
if errorlevel 1 exit /b 1
cmake --build "%~dp0..\build\ft-windows-cuda-dev" --target test-moe-cache test-moe-offload test-moe-integration test-backend-ops llama-cli llama-server --parallel 4
exit /b %errorlevel%
