#include "planners/Planner.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>
#include <Eigen/Core>
#include "obstacle/Obstacle.h"
#include "workspace/Workspace.h"
#include "agent/Agent.h"

// Global: called from the CPU to the GPU
__global__ void generateRandomSampleKernel(Eigen::Vector2d* samples, int numSamples){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numSamples) {
        samples[idx] = Planner::generateRandomSampleDevice();
    }
}

// Host function to manage GPU memory and launch kernel
std::vector<Eigen::Vector2d> Planner::generateRandomSamples() {
    const int numSamples = 100; // Number of samples to generate
    Eigen::Vector2d* d_samples; // Device memory pointer (GPU)
    std::vector<Eigen::Vector2d> h_samples(numSamples); // Host memory vector (CPU)

    // Allocate device memory
    cudaMalloc(&d_samples, numSamples * sizeof(Eigen::Vector2d)); // Allocate memory on the GPU

    // Launch kernel
    int threadsPerBlock = 256; // Number of threads per block
    int blocksPerGrid = (numSamples + threadsPerBlock - 1) / threadsPerBlock; // Number of blocks per grid
    generateRandomSampleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_samples, numSamples); // Launch kernel
    cudaDeviceSynchronize(); // Wait for the GPU to finish

    // Copy result back to host
    cudaMemcpy(h_samples.data(), d_samples, numSamples * sizeof(Eigen::Vector2d), cudaMemcpyDeviceToHost); // Copy memory from GPU to CPU

    // Free device memory
    cudaFree(d_samples); // Free memory on the GPU

    return h_samples; // Return the samples
}

// Device: called from the GPU to the GPU
__device__ Eigen::Vector2d Planner::generateRandomSampleDevice(){
    return Eigen::Vector2d(0, 0);
}

