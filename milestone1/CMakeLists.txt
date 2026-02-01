cmake_minimum_required(VERSION 3.20)
project(rocket_sim_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_TESTS "Build test and benchmark targets" ON)

# Enable all warnings for host and device code
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()
if (CMAKE_CUDA_COMPILER_ID)
    # Keep strong warnings on CUDA translation units without pedantic line-directive noise
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

set(SRC_FILES
    src/ball_physics.cu
    src/batch_manager.cu
    src/rocketsim_gpu_api.cu
)

add_library(rocketsim_gpu STATIC ${SRC_FILES})
target_include_directories(rocketsim_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Restrict to at least SM70 (Volta) by default; users can override
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

if (BUILD_TESTS)
    enable_testing()
    add_executable(test_ball_physics tests/test_ball_physics.cu)
    target_link_libraries(test_ball_physics PRIVATE rocketsim_gpu)
    add_test(NAME ball_physics COMMAND test_ball_physics)

    add_executable(benchmark tests/benchmark.cu)
    target_link_libraries(benchmark PRIVATE rocketsim_gpu)
endif()
