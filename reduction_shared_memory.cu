#include <cuda.h>
#include <stdio.h>

__global__ void reduceSharedMemory(float *d_in, float *d_out, int N) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float mySum = 0.0f;
    if (idx < N)
        mySum = d_in[idx];
    if (idx + blockDim.x < N)
        mySum += d_in[idx + blockDim.x];

    sdata[tid] = mySum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        d_out[blockIdx.x] = sdata[0];
}

int main() {
    int N = 1 << 24;
    size_t size = N * sizeof(float);

    float *h_in = (float *)malloc(size);
    float h_result = 0.0f;

    for (int i = 0; i < N; ++i)
        h_in[i] = 1.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, sizeof(float) * (N / (1024 * 2)));

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    int blocksPerGrid = N / (threadsPerBlock * 2);
    size_t sharedMemSize = threadsPerBlock * sizeof(float);
    reduceSharedMemory<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_in, d_out, N);

    float *h_partialSums = (float *)malloc(blocksPerGrid * sizeof(float));
    cudaMemcpy(h_partialSums, d_out, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < blocksPerGrid; ++i)
        h_result += h_partialSums[i];

    printf("Reduced sum: %f\n", h_result);

    free(h_in); free(h_partialSums);
    cudaFree(d_in); cudaFree(d_out);

    return 0;
}
