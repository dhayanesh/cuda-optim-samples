#include <cuda.h>
#include <stdio.h>

#define N 1048576
#define STREAM_COUNT 4

__global__ void vectorAdd(float *A, float *B, float *C, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

int main() {
    size_t size = N * sizeof(float);
    float *h_A, *h_B, *h_C;
    float *d_A[STREAM_COUNT], *d_B[STREAM_COUNT], *d_C[STREAM_COUNT];

    cudaHostAlloc((void **)&h_A, size, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_B, size, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_C, size, cudaHostAllocDefault);

    for (int i = 0; i < N; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    cudaStream_t streams[STREAM_COUNT];
    for (int i = 0; i < STREAM_COUNT; ++i)
        cudaStreamCreate(&streams[i]);

    size_t streamSize = N / STREAM_COUNT;
    size_t streamBytes = streamSize * sizeof(float);

    for (int i = 0; i < STREAM_COUNT; ++i) {
        cudaMalloc((void **)&d_A[i], streamBytes);
        cudaMalloc((void **)&d_B[i], streamBytes);
        cudaMalloc((void **)&d_C[i], streamBytes);
    }

    for (int i = 0; i < STREAM_COUNT; ++i) {
        int offset = i * streamSize;

        cudaMemcpyAsync(d_A[i], h_A + offset, streamBytes, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_B[i], h_B + offset, streamBytes, cudaMemcpyHostToDevice, streams[i]);

        int threadsPerBlock = 256;
        int blocksPerGrid = (streamSize + threadsPerBlock - 1) / threadsPerBlock;
        vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(d_A[i], d_B[i], d_C[i], streamSize);

        cudaMemcpyAsync(h_C + offset, d_C[i], streamBytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < STREAM_COUNT; ++i)
        cudaStreamSynchronize(streams[i]);

    for (int i = 0; i < STREAM_COUNT; ++i) {
        cudaFree(d_A[i]); cudaFree(d_B[i]); cudaFree(d_C[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);

    return 0;
}
