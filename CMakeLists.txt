cmake_minimum_required(VERSION 3.5)
project(cudaCourse LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Python3 COMPONENTS Development NumPy REQUIRED)

include_directories(
    include 
    ${EIGEN3_INCLUDE_DIRS} 
    ${BOOST_INCLUDE_DIRS}  
    ${CMAKE_SOURCE_DIR}/src 
    ${CUDA_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)

set(SOURCE_FILES
    # Add other source files as needed
    src/workspace/Workspace.cpp
    src/obstacle/Obstacle.cpp
    src/agent/Agent.cpp
    src/planners/RRT.cu
    src/planners/Planner.cu
)

# Add a library target for your non-CUDA code
add_library(cudaCourse_lib ${SOURCE_FILES})

# Add your CUDA executable
add_executable(main vectorAdd.cu)

# Link the library and other dependencies with the CUDA executable
target_link_libraries(main
    cudaCourse_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

# Ensure CUDA separable compilation
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)