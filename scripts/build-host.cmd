@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (echo VCVARS FAILED & exit /b 1)
cmake --build "C:\Users\cadel\Documents\Coding_Projects\p100 FreeToken\build\desktop" --target freetoken-host --parallel 4
exit /b %errorlevel%
