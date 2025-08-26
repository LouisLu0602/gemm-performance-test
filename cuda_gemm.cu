// gemm_cuda_window.cu
// CUDA GEMM with a Windows "result window" (MessageBoxA) showing the timing/GFLOP/s.
// - Prompts for M N K via console OR accepts command-line args.
// - After run, shows a Windows message box with key results.
//
// Build (Windows, kernel only):
//   nvcc -O3 -arch=sm_80 -o gemm_cuda_win gemm_cuda_window.cu
//
// Build with cuBLAS (optional):
//   nvcc -O3 -arch=sm_80 -DUSE_CUBLAS -lcublas -o gemm_cuda_win gemm_cuda_window.cu
//
// Notes:
// * Uses Win32 API (MessageBoxA) to display results.
// * Keep a console for input convenience. If you want a fully GUI input dialog,
//   that requires a Win32 dialog resource or a small GUI framework (not included here).
//
// If you're not on Windows, this will compile but the MessageBox won't be used.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <cassert>
#include <algorithm>

#include <cuda_runtime.h>
#ifdef USE_CUBLAS
#include <cublas_v2.h>
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#ifndef TILE
#define TILE 32
#endif

#define CUDA_CHECK(expr) do {                                   \
    cudaError_t _err = (expr);                                  \
    if (_err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",       \
                #expr, __FILE__, __LINE__, cudaGetErrorString(_err)); \
        std::exit(1);                                           \
    }                                                           \
} while (0)

#ifdef USE_CUBLAS
#define CUBLAS_CHECK(expr) do {                                 \
    cublasStatus_t _st = (expr);                                \
    if (_st != CUBLAS_STATUS_SUCCESS) {                         \
        fprintf(stderr, "cuBLAS error %d at %s:%d: %s\n",       \
                (int)_st, __FILE__, __LINE__, #expr);           \
        std::exit(1);                                           \
    }                                                           \
} while (0)
#endif

// Row-major index helper
__host__ __device__ inline int idxRM(int row, int col, int ld) {
    return row * ld + col;
}

// Tiled shared-memory SGEMM (row-major)
__global__ void sgemm_tiled_kernel(int M, int N, int K,
                                   const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * blockDim.y + threadIdx.y; // 0..M-1
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 0..N-1

    float acc = 0.0f;

    // Iterate over tiles of K dimension
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x; // along K
        int b_row = t * TILE + threadIdx.y; // along K

        // Load A tile (row, a_col)
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[idxRM(row, a_col, K)];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile (b_row, col)
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[idxRM(b_row, col, N)];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute this tile product
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc = fmaf(As[threadIdx.y][k], Bs[k][threadIdx.x], acc);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[idxRM(row, col, N)] = acc;
    }
}

static void init_matrix(std::vector<float>& h, int rows, int cols, unsigned seed) {
    // Deterministic-ish init without <random> for portability
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            unsigned v = (unsigned)(r * 1315423911u + c * 2654435761u + seed);
            // scale to [0,1)
            float x = (float)(v & 0x00FFFFFF) / (float)0x01000000;
            h[idxRM(r, c, cols)] = x - 0.5f; // roughly [-0.5, 0.5)
        }
    }
}

int main(int argc, char** argv) {
    int M = -1, N = -1, K = -1;
    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    } else {
        std::printf("Enter M N K: ");
        if (std::scanf("%d %d %d", &M, &N, &K) != 3) {
            std::fprintf(stderr, "Failed to read M N K. Using defaults 10000 10000 10000.\n");
            M = 10000; N = 10000; K = 10000;
        }
    }

    if (M <= 0 || N <= 0 || K <= 0) {
        std::fprintf(stderr, "Invalid sizes. All dimensions must be positive.\n");
        return 1;
    }

    std::printf("GEMM: C[MxN] = A[MxK] * B[KxN], float32, row-major\n");
    std::printf("Dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    std::printf("Block: %dx%d  (TILE=%d)\n", TILE, TILE, TILE);

    size_t bytesA = (size_t)M * (size_t)K * sizeof(float);
    size_t bytesB = (size_t)K * (size_t)N * sizeof(float);
    size_t bytesC = (size_t)M * (size_t)N * sizeof(float);
    double totalGiB = (bytesA + bytesB + bytesC) / (1024.0 * 1024.0 * 1024.0);
    std::printf("Memory (A+B+C): %.2f GiB\n", totalGiB);

    // Host buffers
    std::vector<float> hA((size_t)M * K);
    std::vector<float> hB((size_t)K * N);
    std::vector<float> hC((size_t)M * N, 0.f);

    init_matrix(hA, M, K, 123u);
    init_matrix(hB, K, N, 456u);

    // Device buffers
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dA, bytesA));
    CUDA_CHECK(cudaMalloc((void**)&dB, bytesB));
    CUDA_CHECK(cudaMalloc((void**)&dC, bytesC));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, bytesC));

    dim3 block(TILE, TILE);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    // Warm-up
    sgemm_tiled_kernel<<<grid, block>>>(M, N, K, dA, dB, dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Time kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    sgemm_tiled_kernel<<<grid, block>>>(M, N, K, dA, dB, dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float msKernel = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&msKernel, start, stop));

    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = flops / (msKernel * 1.0e6);

    std::printf("Kernel time: %.3f ms,  GFLOP/s: %.2f\n", msKernel, gflops);

#ifdef USE_CUBLAS
    // cuBLAS SGEMM for comparison
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;

    CUDA_CHECK(cudaMemset(dC, 0, bytesC));

    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             dB, N,
                             dA, K,
                             &beta,
                             dC, N));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float msBLAS = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&msBLAS, start, stop));
    double gflopsBLAS = flops / (msBLAS * 1.0e6);
    std::printf("[cuBLAS] time: %.3f ms, GFLOP/s: %.2f\n", msBLAS, gflopsBLAS);
    CUBLAS_CHECK(cublasDestroy(handle));
#endif

#ifdef _WIN32
    // Build a result string and show in a window
    char msg[512];
    std::snprintf(msg, sizeof(msg),
                  "CUDA GEMM (float32)\n"
                  "M=%d, N=%d, K=%d\n"
                  "Block %dx%d (TILE=%d)\n"
                  "Memory (A+B+C): %.2f GiB\n"
                  "Kernel time: %.3f ms\n"
                  "GFLOP/s: %.2f\n",
                  M, N, K, TILE, TILE, TILE, totalGiB, msKernel, gflops);
    MessageBoxA(nullptr, msg, "GEMM Results", MB_OK | MB_ICONINFORMATION);
#endif

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return 0;
}
