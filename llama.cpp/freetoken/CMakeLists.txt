cmake_minimum_required(VERSION 3.14)
project(freetoken VERSION 0.1.0 LANGUAGES C CXX)

set(FREETOKEN_GGML_SOURCE_DIR "" CACHE PATH "GGML source directory, if the host has not added GGML")
option(FREETOKEN_PATCHED_GGML "Enable this fork's workspace telemetry and precise CUDA extensions" OFF)
option(FREETOKEN_BUILD_EXAMPLES "Build the standalone FreeToken smoke example" OFF)

if (NOT TARGET ggml)
    if (NOT FREETOKEN_GGML_SOURCE_DIR)
        message(FATAL_ERROR "Add GGML before FreeToken or set FREETOKEN_GGML_SOURCE_DIR to a GGML source directory")
    endif()
    add_subdirectory("${FREETOKEN_GGML_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/ggml")
endif()
if (NOT TARGET ggml-cpu)
    message(FATAL_ERROR "FreeToken requires the GGML CPU backend (GGML_CPU=ON)")
endif()

find_package(Threads REQUIRED)
add_library(freetoken-runtime STATIC src/moe.cpp src/cache.cpp)
add_library(freetoken::runtime ALIAS freetoken-runtime)
set_target_properties(freetoken-runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(freetoken-runtime PUBLIC cxx_std_17)
target_include_directories(freetoken-runtime PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(freetoken-runtime PUBLIC
    $<BUILD_INTERFACE:ggml> $<BUILD_INTERFACE:ggml-cpu>
    $<INSTALL_INTERFACE:ggml::ggml> $<INSTALL_INTERFACE:ggml::ggml-cpu>
    PRIVATE Threads::Threads)
if (FREETOKEN_PATCHED_GGML)
    target_compile_definitions(freetoken-runtime PRIVATE FREETOKEN_PATCHED_GGML)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS freetoken-runtime ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/freetoken DESTINATION include)
set(FREETOKEN_INCLUDE_INSTALL_DIR include)
configure_package_config_file(cmake/freetoken-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/freetoken-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/freetoken
    PATH_VARS CMAKE_INSTALL_LIBDIR FREETOKEN_INCLUDE_INSTALL_DIR)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/freetoken-config-version.cmake
    VERSION 0.1.0 COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/freetoken-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/freetoken-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/freetoken)
if (MSVC)
    target_compile_options(freetoken-runtime PRIVATE /EHsc)
endif()

if (FREETOKEN_BUILD_EXAMPLES)
    enable_testing()
    add_executable(freetoken-smoke examples/smoke.cpp)
    target_link_libraries(freetoken-smoke PRIVATE freetoken::runtime)
    if (MSVC)
        target_compile_options(freetoken-smoke PRIVATE /EHsc)
    endif()
    if (WIN32)
        foreach(backend ggml ggml-base ggml-cpu)
            get_target_property(backend_type ${backend} TYPE)
            if (backend_type STREQUAL "SHARED_LIBRARY")
                add_custom_command(TARGET freetoken-smoke POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        $<TARGET_FILE:${backend}> $<TARGET_FILE_DIR:freetoken-smoke>)
            endif()
        endforeach()
    endif()
    add_test(NAME freetoken-smoke COMMAND freetoken-smoke)
endif()
