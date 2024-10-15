  # Advanced CUDA Programs for High-Performance Computing

This repository contains CUDA programs that utilize advanced optimization techniques, multi-GPU communication, and efficient data handling for high-performance computing applications.

## Contents

1. **Optimized Image Filtering with Memory Coalescing**  
   **Filename**: `image_convolution_coalesced.cu`  
   **Description**:  
   This CUDA program performs 2D image convolution using shared memory and memory coalescing techniques to optimize performance. It demonstrates efficient use of shared memory and coalesced memory access patterns.

2. **Parallel Reduction Using Shared Memory and Strided Indexing**  
   **Filename**: `reduction_shared_memory.cu`  
   **Description**:  
   This CUDA program implements an efficient reduction operation using shared memory and strided indexing. It reduces global memory accesses and optimizes memory bandwidth utilization.

3. **Asynchronous Data Transfer with CUDA Streams**  
   **Filename**: `async_data_transfer_streams.cu`  
   **Description**:  
   This CUDA program demonstrates how to use CUDA streams to overlap data transfer and kernel execution. By utilizing multiple streams, it achieves concurrent execution and improved performance.

4. **Implementing a Fast Fourier Transform (FFT) with Shared Memory**  
   **Filename**: `fft_shared_memory.cu`  
   **Description**:  
   This CUDA program implements a simplified version of the Fast Fourier Transform (FFT) using shared memory for performance optimization. It is useful for applications requiring frequency domain analysis.

5. **Multi-GPU Communication with MPI and CUDA Streams**  
   **Filename**: `multi_gpu_mpi_streams.cu`  
   **Description**:  
   This program demonstrates multi-GPU communication using MPI (Message Passing Interface) combined with CUDA streams. It shows how to integrate inter-process communication with GPU computations using CUDA.

## Compilation and Execution

### Requirements:
- **System with multiple NVIDIA GPUs**
- **CUDA Toolkit installed**
- **MPI library installed** (e.g., OpenMPI)

