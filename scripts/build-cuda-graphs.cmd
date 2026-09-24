@echo off
setlocal
set "FT_CUDA_GRAPHS=ON"
call "%~dp0build-cuda-dev.cmd"
exit /b %errorlevel%
