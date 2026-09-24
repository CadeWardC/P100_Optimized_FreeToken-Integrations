@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cmake --build build/desktop --target freetoken-host desktop-tests freetoken-desktop --parallel 4
