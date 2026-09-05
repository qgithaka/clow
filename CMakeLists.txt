cmake_minimum_required(VERSION 3.25)
project(clow VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for language servers
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization and Warning Flags
if(MSVC)
    add_compile_options(/W4 /permissive- /utf-8 /EHsc)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wconversion)
endif()

# Qt 6 Configuration
find_package(Qt6 COMPONENTS Core Gui Widgets OpenGLWidgets Network QUIET)
if(Qt6_FOUND)
    message(STATUS "Qt6 found version: ${Qt6_VERSION}")
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)
else()
    message(WARNING "Qt6 not found in standard paths. Headless / Core build will be active.")
endif()

# Include Directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/config
)

# Core Library Sources (Non-GUI Engine)
file(GLOB_RECURSE CLOW_CORE_SOURCES
    "src/core/*.cpp"
    "src/broker/*.cpp"
    "src/ai/*.cpp"
    "src/data/*.cpp"
)

file(GLOB_RECURSE CLOW_CORE_HEADERS
    "src/core/*.h"
    "src/broker/*.h"
    "src/ai/*.h"
    "src/data/*.h"
)

# Build Static Core Engine Library
add_library(clow_core STATIC
    ${CLOW_CORE_SOURCES}
    ${CLOW_CORE_HEADERS}
)

if(Qt6_FOUND)
    target_link_libraries(clow_core PUBLIC
        Qt6::Core
        Qt6::Network
    )
endif()

# GUI Terminal Executable Sources
file(GLOB_RECURSE CLOW_UI_SOURCES
    "src/ui/*.cpp"
)

file(GLOB_RECURSE CLOW_UI_HEADERS
    "src/ui/*.h"
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
    add_executable(clow_terminal
        src/main.cpp
        ${CLOW_UI_SOURCES}
        ${CLOW_UI_HEADERS}
    )

    target_link_libraries(clow_terminal PRIVATE
        clow_core
    )

    if(Qt6_FOUND)
        target_link_libraries(clow_terminal PRIVATE
            Qt6::Core
            Qt6::Gui
            Qt6::Widgets
            Qt6::OpenGLWidgets
        )
    endif()
endif()

# C++ Unit Tests
enable_testing()
file(GLOB_RECURSE TEST_CPP_SOURCES "tests/cpp/*.cpp")
if(TEST_CPP_SOURCES)
    add_executable(clow_cpp_tests
        ${TEST_CPP_SOURCES}
    )
    target_link_libraries(clow_cpp_tests PRIVATE
        clow_core
    )
    add_test(NAME ClowCppTests COMMAND clow_cpp_tests)
endif()
