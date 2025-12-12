cmake_minimum_required(VERSION 3.20)
project(RSAI)

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SRC_FILES
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

add_executable(RSAI ${SRC_FILES})

target_include_directories(RSAI PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
)

# Macro OS definition
target_compile_definitions(RSAI PRIVATE
    $<$<PLATFORM_ID:Windows>:WINDOWS>
    $<$<PLATFORM_ID:Linux>:LINUX>
    $<$<PLATFORM_ID:Darwin>:MACOS>
    $<$<PLATFORM_ID:Android>:ANDROID>
)