cmake_minimum_required(VERSION 3.5)
project(cudaCourse LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Python3 COMPONENTS Development NumPy REQUIRED)
find_package(OpenCV REQUIRED)

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
target_link_libraries(
    main
    cudaCourse_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    ${OpenCV_LIBS}
)

# Ensure CUDA separable compilation
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(imageGray imageGray.cu)

target_link_libraries(
    imageGray
    cudaCourse_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    ${OpenCV_LIBS}
)

# Ensure CUDA separable compilation
set_target_properties(imageGray PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(matrixMult matrixMult.cu)

target_link_libraries(
    matrixMult
    cudaCourse_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    ${OpenCV_LIBS}
)

# Ensure CUDA separable compilation
set_target_properties(matrixMult PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


# Ch 7 Convolution:
add_executable(convolutionBasic ch7_convolution/convolutionBasic.cu)

target_link_libraries(
    convolutionBasic
    cudaCourse_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    ${OpenCV_LIBS}
)

# Ensure CUDA separable compilation
set_target_properties(convolutionBasic PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(convolutionConstantMemory ch7_convolution/convolutionConstantMemory.cu)

target_link_libraries(
    convolutionConstantMemory
    cudaCourse_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    ${OpenCV_LIBS}
)

# Ensure CUDA separable compilation
set_target_properties(convolutionConstantMemory PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_executable(convolutionTiled ch7_convolution/convolutionTiled.cu)

target_link_libraries(
    convolutionTiled
    cudaCourse_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    ${OpenCV_LIBS}
)

# Ensure CUDA separable compilation
set_target_properties(convolutionTiled PROPERTIES CUDA_SEPARABLE_COMPILATION ON)