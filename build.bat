@echo off
rmdir /s /q build
rmdir /s /q .vs
mkdir build
cd build
cmake ../ -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --parallel
cd ..
xcopy "bin\*" "build\Release\bin\" /E /Y /I
copy "requirements.txt" "build\Release\requirements.txt"
pause