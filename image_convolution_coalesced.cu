#include <cuda.h>
#include <stdio.h>

#define FILTER_WIDTH 3
#define TILE_WIDTH 16

__constant__ float d_Filter[FILTER_WIDTH * FILTER_WIDTH];


__global__ void convolution2DCoalesced(float *d_Input, float *d_Output, int width, int height) {
    __shared__ float sharedMem[TILE_WIDTH + FILTER_WIDTH - 1][TILE_WIDTH + FILTER_WIDTH - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;
    int row_i = row_o - FILTER_WIDTH / 2;
    int col_i = col_o - FILTER_WIDTH / 2;


    if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width))
        sharedMem[ty][tx] = d_Input[row_i * width + col_i];
    else
        sharedMem[ty][tx] = 0.0f;

    __syncthreads();

    float outputValue = 0.0f;
    if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
        for (int i = 0; i < FILTER_WIDTH; ++i)
            for (int j = 0; j < FILTER_WIDTH; ++j)
                outputValue += d_Filter[i * FILTER_WIDTH + j] * sharedMem[ty + i][tx + j];

        if (row_o < height && col_o < width)
            d_Output[row_o * width + col_o] = outputValue;
    }
}

int main() {
    int width = 1024;
    int height = 1024;
    size_t size = width * height * sizeof(float);

    float *h_Input = (float *)malloc(size);
    float *h_Output = (float *)malloc(size);
    float h_Filter[FILTER_WIDTH * FILTER_WIDTH] = {
        0, -1, 0,
       -1,  5, -1,
        0, -1, 0
    };

    for (int i = 0; i < width * height; ++i)
        h_Input[i] = rand() / (float)RAND_MAX;

    float *d_Input, *d_Output;
    cudaMalloc((void **)&d_Input, size);
    cudaMalloc((void **)&d_Output, size);

    cudaMemcpy(d_Input, h_Input, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_Filter, h_Filter, FILTER_WIDTH * FILTER_WIDTH * sizeof(float));

    dim3 dimBlock(TILE_WIDTH + FILTER_WIDTH - 1, TILE_WIDTH + FILTER_WIDTH - 1);
    dim3 dimGrid(ceil(width / (float)TILE_WIDTH), ceil(height / (float)TILE_WIDTH));

    convolution2DCoalesced<<<dimGrid, dimBlock>>>(d_Input, d_Output, width, height);

    cudaMemcpy(h_Output, d_Output, size, cudaMemcpyDeviceToHost);

    free(h_Input); free(h_Output);
    cudaFree(d_Input); cudaFree(d_Output);

    return 0;
}
