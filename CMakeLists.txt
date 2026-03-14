cmake_minimum_required(VERSION 3.16)

project(GraphPropagatedLatentSimilarity
    VERSION 1.0
    DESCRIPTION "Simulation code for Graph-Propagated Latent Similarity in Agent-Based Segregation Dynamics"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -O3)
endif()

add_executable(sgnn_runner
    cpp/sgnn_runner.cpp
)
