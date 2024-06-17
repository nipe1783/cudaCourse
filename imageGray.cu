#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>

__global__ 
void colorToGrayScaleConversion(unsigned char *Pout, unsigned char *Pin, int width, int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x; // col index in image
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row index in image
    if (col < width && row < height)
    {
        int grayOffset = row * width + col; // Flattened index of pixel
        int rgbOffset = grayOffset * 3; // 3 channels per pixel
        unsigned char r = Pin[rgbOffset]; // red value
        unsigned char g = Pin[rgbOffset + 1]; // green value
        unsigned char b = Pin[rgbOffset + 2]; // blue value
        Pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b; // grayscale value
    }
    
}

int main(void){

    // import image using opencv
    cv::Mat image = cv::imread("/home/nic/dev/research/cudaCourse/image.png");

    int m = image.rows;
    int n = image.cols;
    int N = m * n;
    int size = N * sizeof(float);
    
    // set dimension of grid (number of blocks)
    dim3 dimGrid(ceil(m/16.0), ceil(n/16.0), 1);  // Number of blocks depends on image size
    dim3 dimBlock(16, 16, 1); // 16x16 threads per block

    printf("Number of blocks: %d\n", dimGrid.x * dimGrid.y);
    printf("Dimension of grid: %d x %d\n", dimGrid.x, dimGrid.y);
    printf("Number of threads per block: %d\n", dimBlock.x * dimBlock.y);
    printf("Dimension of block: %d x %d\n", dimBlock.x, dimBlock.y);
    printf("Number of threads: %d\n", dimGrid.x * dimGrid.y * dimBlock.x * dimBlock.y);
    printf("Number of pixels: %d\n", N);
    

    // Allocate memory for each vector on device
    unsigned char *d_image, *d_grayImage;
    cudaMalloc((void**)&d_image, N * 3 * sizeof(unsigned char));
    cudaMalloc((void**)&d_grayImage, N * sizeof(unsigned char));

    // copy host vectors to device
    cudaMemcpy(d_image, image.data, N * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // launch kernel
    colorToGrayScaleConversion<<<dimGrid, dimBlock>>>(d_grayImage, d_image, m, n);

    // copy result back to host
    unsigned char *grayImage = (unsigned char*)malloc(N * sizeof(unsigned char));
    cudaMemcpy(grayImage, d_grayImage, N * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // create new image
    cv::Mat gray(m, n, CV_8UC1, grayImage);
    cv::imwrite("/home/nic/dev/research/cudaCourse/grayImage.png", gray);

    // cleanup
    free(grayImage);
    cudaFree(d_image);
    cudaFree(d_grayImage);


    return 0;   

}