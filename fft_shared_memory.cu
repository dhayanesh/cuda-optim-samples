#include <cuda.h>
#include <stdio.h>
#include <math.h>

#define N 1024
#define BLOCK_SIZE 256

__global__ void fftKernel(float2 *d_signal, float2 *d_result, int n) {
    __shared__ float2 s_data[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n)
        s_data[tid] = d_signal[idx];
    else
        s_data[tid] = make_float2(0.0f, 0.0f);

    __syncthreads();

    for (unsigned int stride = 1; stride <= BLOCK_SIZE / 2; stride *= 2) {
        __syncthreads();
        int index = 2 * stride * tid;
        if (index < BLOCK_SIZE) {
            float2 temp = s_data[index + stride];
            float angle = -2.0f * M_PI * (tid % stride) / (float)(2 * stride);
            float2 W = make_float2(cosf(angle), sinf(angle));
            float2 t = make_float2(
                W.x * temp.x - W.y * temp.y,
                W.x * temp.y + W.y * temp.x
            );
            s_data[index + stride] = make_float2(
                s_data[index].x - t.x,
                s_data[index].y - t.y
            );
            s_data[index] = make_float2(
                s_data[index].x + t.x,
                s_data[index].y + t.y
            );
        }
    }

    __syncthreads();

    if (idx < n)
        d_result[idx] = s_data[tid];
}

int main() {
    size_t size = N * sizeof(float2);

    float2 *h_signal = (float2 *)malloc(size);
    float2 *h_result = (float2 *)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_signal[i].x = sinf(2.0f * M_PI * i / N);
        h_signal[i].y = 0.0f;
    }

    float2 *d_signal, *d_result;
    cudaMalloc((void **)&d_signal, size);
    cudaMalloc((void **)&d_result, size);

    cudaMemcpy(d_signal, h_signal, size, cudaMemcpyHostToDevice);

    int blocksPerGrid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fftKernel<<<blocksPerGrid, BLOCK_SIZE>>>(d_signal, d_result, N);

    cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);

    free(h_signal); free(h_result);
    cudaFree(d_signal); cudaFree(d_result);

    return 0;
}
