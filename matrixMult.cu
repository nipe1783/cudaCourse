#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>

// assumes square matrices. Each thread computes one element of P.
__global__ 
void matrixMultP(const float* M, const float* N, float* P, int width){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width && col < width){
        float pValue = 0;
        int pIndex = row * width + col; // Flattened index of matrix P
        for (int k = 0; k < width; k++){
            int MIndex = row * width + k; // Flattened index of matrix M
            int NIndex = k * width + col; // Flattened index of matrix N
            pValue += M[MIndex] * N[NIndex];
        }
        P[pIndex] = pValue;
    }
    printf("Thread (%d, %d) computed element (%d, %d)\n", threadIdx.x, threadIdx.y, row, col);
}

//assumes square matrices. Each thread computes one row of P.
__global__
void matrixMultRow(const float* M, const float* N, float* P, int width){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width){
        for (int col = 0; col < width; col++){
            float pValue = 0;
            int pIndex = row * width + col; // Flattened index of matrix P
            for (int k = 0; k < width; k++){
                int MIndex = row * width + k; // Flattened index of matrix M
                int NIndex = k * width + col; // Flattened index of matrix N
                pValue += M[MIndex] * N[NIndex];
            }
            P[pIndex] = pValue;
        }
    }
    printf("Thread (%d, %d) computed row %d\n", threadIdx.x, threadIdx.y, row);
}

int main(void){

   // initialize host matrices
    int width = 4;
    int size = width * width * sizeof(float);
    float *h_M = (float*)malloc(size);
    float *h_N = (float*)malloc(size);
    float *h_P = (float*)malloc(size);
    
    // populate matrices with random floats:
    for (int i = 0; i < width; i++){
        for (int j = 0; j < width; j++){
            h_M[i * width + j] = (float)rand() / RAND_MAX;
            h_N[i * width + j] = (float)rand() / RAND_MAX;
        }
    }


    // initialize device matrices
    float *d_M, *d_N, *d_P;
    cudaMalloc((void**)&d_M, size);
    cudaMalloc((void**)&d_N, size);
    cudaMalloc((void**)&d_P, size);

    // copy host matrices to device
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    // set dimension of grid (number of blocks)
    dim3 dimGrid(width/2, width/2, 1); // Splits matrix into 4 sections
    dim3 dimBlock(2, 2, 1); // each thread computes 1 element of P

    // launch kernel
    matrixMultP<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, width);

    // copy result back to host
    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    // verify result
    for (int i = 0; i < width; i++){
        for (int j = 0; j < width; j++){
            float pValue = 0;
            for (int k = 0; k < width; k++){
                pValue += h_M[i * width + k] * h_N[k * width + j];
            }
            if (fabs(pValue - h_P[i * width + j]) > 1e-5){
                fprintf(stderr, "Result verification failed at element (%d, %d)!\n", i, j);
                exit(EXIT_FAILURE);
            }
        }
    }

    printf("Test PASSED\n");

    // cleanup
    


    // each thread computes one row of P
    // set dimension of grid (number of blocks)
    dim3 dimGridR(2, 1, 1); // Splits matrix into 2 sections
    dim3 dimBlockR(2, 1, 1); // each thread computes 1 row of P

    // launch kernel
    matrixMultRow<<<dimGridR, dimBlockR>>>(d_M, d_N, d_P, width);

    // copy result back to host
    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    // verify result
    for (int i = 0; i < width; i++){
        for (int j = 0; j < width; j++){
            float pValue = 0;
            for (int k = 0; k < width; k++){
                pValue += h_M[i * width + k] * h_N[k * width + j];
            }
            if (fabs(pValue - h_P[i * width + j]) > 1e-5){
                fprintf(stderr, "Result verification failed at element (%d, %d)!\n", i, j);
                exit(EXIT_FAILURE);
            }
        }
    }

    printf("Test PASSED\n");


    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    free(h_M);
    free(h_N);
    free(h_P);


}

