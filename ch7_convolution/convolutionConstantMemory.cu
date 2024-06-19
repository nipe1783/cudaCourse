// Convolution where F is stored in F constant memory off chip. This slightly improves performance. OP/B = 0.5.

#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>
#include <ctime>

// assign F to constant memory:
__constant__ float F[3 * 3]; // F is stored in constant memory off chip


// Kernel function. OP/B = 0.25. Horrible performance.
__global__ void convolution_2D_basic_kernel(const float *N, float *P, int r, int width, int height) {
    // Determine the row and column index of the output matrix.
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize the output value to 0.
    float Pvalue = 0.0f;
    for(int fRow = 0; fRow < 2 * r + 1; fRow++) {
        for(int fCol = 0; fCol < 2 * r + 1; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if(inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                int N_index = inRow * width + inCol;
                int F_index = fRow * (2 * r + 1) + fCol;
                Pvalue += N[N_index] * F[F_index];
            }
        }
    }
    int P_index = outRow * width + outCol;
    P[P_index] = Pvalue;
    printf("Thread (%d, %d) -> P[%d] = %f\n", outRow, outCol, P_index, P[P_index]);
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Host function to perform convolution and compare results
void check_result(const float *h_N, const float *h_F, const float *h_P, int rows_N, int cols_N, int rows_f, int cols_f, int r) {
    bool success = true;
    for (int outRow = 0; outRow < rows_N; ++outRow) {
        for (int outCol = 0; outCol < cols_N; ++outCol) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < rows_f; ++fRow) {
                for (int fCol = 0; fCol < cols_f; ++fCol) {
                    int inRow = outRow - r + fRow;
                    int inCol = outCol - r + fCol;
                    if (inRow >= 0 && inRow < rows_N && inCol >= 0 && inCol < cols_N) {
                        Pvalue += h_N[inRow * cols_N + inCol] * h_F[fRow * cols_f + fCol];
                    }
                }
            }
            if (fabs(h_P[outRow * cols_N + outCol] - Pvalue) > 1e-5) {
                success = false;
                printf("Mismatch at (%d, %d): GPU = %f, CPU = %f\n", outRow, outCol, h_P[outRow * cols_N + outCol], Pvalue);
            }
        }
    }
    if (success) {
        printf("Results match!\n");
    } else {
        printf("Results do not match!\n");
    }
}

int main(void) {
    const int rows_N = 16;
    const int cols_N = 16;
    const int rows_f = 3;
    const int cols_f = 3;
    const int r = 1;
    float h_F[rows_f][cols_f] = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };

    float h_N[rows_N][cols_N];
    std::srand(std::time(0));
    for (int i = 0; i < rows_N; ++i) {
        for (int j = 0; j < cols_N; ++j) {
            h_N[i][j] = std::rand() % 5 + 1;
        }
    }

    int size_N = rows_N * cols_N * sizeof(float);
    float *d_N, *d_P;
    checkCudaError(cudaMalloc((void**)&d_N, size_N), "Failed to allocate device memory for d_N");
    checkCudaError(cudaMalloc((void**)&d_P, size_N), "Failed to allocate device memory for d_P");
    checkCudaError(cudaMemcpy(d_N, h_N, size_N, cudaMemcpyHostToDevice), "Failed to copy data from host to device for d_N");

    // Copy F to constant memory:
    checkCudaError(cudaMemcpyToSymbol(F, h_F, rows_f * cols_f * sizeof(float)), "Failed to copy data to constant memory for F");

    dim3 dimGrid(rows_N / 4, cols_N / 4, 1);
    dim3 dimBlock(4, 4, 1);
    convolution_2D_basic_kernel<<<dimGrid, dimBlock>>>(d_N, d_P, r, rows_N, cols_N);
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution failed");

    float h_P[rows_N * cols_N];
    checkCudaError(cudaMemcpy(h_P, d_P, size_N, cudaMemcpyDeviceToHost), "Failed to copy data from device to host for h_P");

    check_result((float*)h_N, (float*)h_F, h_P, rows_N, cols_N, rows_f, cols_f, r);

    checkCudaError(cudaFree(d_N), "Failed to free device memory for d_N");
    checkCudaError(cudaFree(d_P), "Failed to free device memory for d_P");

    return 0;
}
