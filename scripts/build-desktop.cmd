@echo off
setlocal
if not defined FT_VS_ROOT set "FT_VS_ROOT=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "%FT_VS_ROOT%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
set "PATH=%FT_VS_ROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
cmake -S "%~dp0..\desktop" -B "%~dp0..\build\desktop" -G Ninja -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
cmake --build "%~dp0..\build\desktop" --parallel 4
if errorlevel 1 exit /b 1
ctest --test-dir "%~dp0..\build\desktop" --output-on-failure
exit /b %errorlevel%
