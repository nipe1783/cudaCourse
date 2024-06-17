#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        C[i] = A[i] + B[i];
    }
}

int main(void){

    float *h_A, *h_B, *h_C; // host vectors
    float *d_A, *d_B, *d_C; // device vectors

    int N = 1000000; // vector size
    int size = N * sizeof(float);

    // generate random arrays on host:
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    // populate arrays
    for (int i = 0; i < N; i++){
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate memory for each vector on device
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // copy host vectors to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // launch kernel
    vectorAdd<<<ceil(N/256.0), 256>>>(d_A, d_B, d_C, N);

    // copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // verify result
    for (int i = 0; i < N; i++){
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5){
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;   

}