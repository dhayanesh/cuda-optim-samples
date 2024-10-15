#include <mpi.h>
#include <cuda.h>
#include <stdio.h>

#define N 1048576 

__global__ void vectorScale(float *d_data, float scalar, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
        d_data[idx] *= scalar;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    if (worldSize < 2) {
        printf("This example requires at least two processes.\n");
        MPI_Finalize();
        return 0;
    }

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaSetDevice(worldRank % deviceCount);

    size_t size = N * sizeof(float);
    float *h_data;
    float *d_data;

    cudaHostAlloc((void **)&h_data, size, cudaHostAllocDefault);
    cudaMalloc((void **)&d_data, size);

    if (worldRank == 0) {
        for (int i = 0; i < N; ++i)
            h_data[i] = rand() / (float)RAND_MAX;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    if (worldRank == 0) {
        cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);

        cudaStreamSynchronize(stream);

        MPI_Send(d_data, N, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    } else if (worldRank == 1) {
        MPI_Recv(d_data, N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        vectorScale<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, 2.0f, N);

        cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

        printf("Data received and processed on rank 1.\n");
    }

    cudaFreeHost(h_data);
    cudaFree(d_data);
    cudaStreamDestroy(stream);

    MPI_Finalize();
    return 0;
}
