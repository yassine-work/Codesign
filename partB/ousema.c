#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// --- SET THESE TO YOUR ACTUAL BENCHMARK RESULTS FOR PERFECT LOAD BALANCING ---
#define NVIDIA_GFLOPS_GUESS 491.0 
#define INTEL_GFLOPS_GUESS  200.0 

#define WG_X 16
#define WG_Y 16
#define TM   4
#define TN   4
#define TILE_M (WG_Y * TM)   // 64
#define TILE_N (WG_X * TN)   // 64
#define TILE_K  16

#define CHECK_CL(err, msg) do { \
    if ((err) != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error %d at %s\n", (err), (msg)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// ---------------- KERNEL 1: UNCOALESCED (For NVIDIA) ----------------
const char *kernel_uncoalesced =
"__kernel void sgemm_uncoalesced(\n"
"    const int M, const int N, const int K,\n"
"    __global const float* A,\n"
"    __global const float* B,\n"
"    __global float* C)\n"
"{\n"
"    const int globalRow = get_global_id(1);\n"
"    const int globalCol = get_global_id(0);\n"
"    if (globalRow < M && globalCol < N) {\n"
"        float acc = 0.0f;\n"
"        for (int k = 0; k < K; k++) {\n"
"            acc += A[globalRow * K + k] * B[k * N + globalCol];\n"
"        }\n"
"        C[globalRow * N + globalCol] = acc;\n"
"    }\n"
"}\n";

// ---------------- KERNEL 2: TILED OPTIMIZED (For INTEL) ----------------
const char *kernel_tiled =
"__attribute__((reqd_work_group_size(16,16,1)))\n"
"__kernel void sgemm_tiled(\n"
"    const int M, const int N, const int K,\n"
"    __global const float* restrict A,\n"
"    __global const float* restrict B,\n"
"    __global float* restrict C,\n"
"    __local float* As,\n"
"    __local float* Bs)\n"
"{\n"
"    // [Insert your exact Part A kernel body here]\n"
"    const int lx = get_local_id(0);\n"
"    const int ly = get_local_id(1);\n"
"    const int gx = get_group_id(0);\n"
"    const int gy = get_group_id(1);\n"
"    const int rowBase = gy * 64 + ly * 4;\n"
"    const int colBase = gx * 64 + lx * 4;\n"
"    const int tid = ly * 16 + lx;\n"
"    float acc[4][4] = {0};\n"
"    for (int k0 = 0; k0 < K; k0 += 16) {\n"
"        #pragma unroll\n"
"        for (int t = 0; t < 4; ++t) {\n"
"            int idx = tid + t * 256;\n"
"            int aRow = idx / 16, aCol = idx - aRow * 16;\n"
"            int gAR = gy * 64 + aRow, gAC = k0 + aCol;\n"
"            As[aRow * 16 + aCol] = (gAR < M && gAC < K) ? A[gAR * K + gAC] : 0.0f;\n"
"            int bRow = idx / 64, bCol = idx - bRow * 64;\n"
"            int gBR = k0 + bRow, gBC = gx * 64 + bCol;\n"
"            Bs[bRow * 64 + bCol] = (gBR < K && gBC < N) ? B[gBR * N + gBC] : 0.0f;\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        #pragma unroll\n"
"        for (int kk = 0; kk < 16; ++kk) {\n"
"            float a[4], b[4];\n"
"            #pragma unroll\n"
"            for (int i = 0; i < 4; ++i) a[i] = As[(ly * 4 + i) * 16 + kk];\n"
"            #pragma unroll\n"
"            for (int j = 0; j < 4; ++j) b[j] = Bs[kk * 64 + (lx * 4 + j)];\n"
"            #pragma unroll\n"
"            for (int i = 0; i < 4; ++i)\n"
"                #pragma unroll\n"
"                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    #pragma unroll\n"
"    for (int i = 0; i < 4; ++i) {\n"
"        int r = rowBase + i;\n"
"        if (r < M) {\n"
"            #pragma unroll\n"
"            for (int j = 0; j < 4; ++j) {\n"
"                int c = colBase + j;\n"
"                if (c < N) C[r * N + c] = acc[i][j];\n"
"            }\n"
"        }\n"
"    }\n"
"}\n";

// Helper to fetch specific platform device
cl_device_id get_device_from_platform(int platform_index) {
    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, NULL, &num_platforms);
    cl_platform_id *platforms = malloc(sizeof(cl_platform_id) * num_platforms);
    clGetPlatformIDs(num_platforms, platforms, NULL);
    
    cl_device_id device;
    clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    free(platforms);
    return device;
}

int main(void) {
    const int SIZE = 8192;
    const int M = SIZE, N = SIZE, K = SIZE;

    // 1. Calculate the Workload Split
    double total_power = NVIDIA_GFLOPS_GUESS + INTEL_GFLOPS_GUESS;
    double nv_ratio = NVIDIA_GFLOPS_GUESS / total_power;
    
    int M_NV = (int)(M * nv_ratio);
    // Round M_NV to nearest multiple of TILE_M (64) so Intel workgroups align perfectly
    M_NV = (M_NV / TILE_M) * TILE_M; 
    int M_IG = M - M_NV;

    printf("Total Rows: %d | NVIDIA Rows: %d | Intel Rows: %d\n", M, M_NV, M_IG);

    // Host memory
    float *h_A = (float*)malloc((size_t)M * K * sizeof(float));
    float *h_B = (float*)malloc((size_t)K * N * sizeof(float));
    float *h_C = (float*)malloc((size_t)M * N * sizeof(float));
    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 2.0f;
    memset(h_C, 0, (size_t)M * N * sizeof(float));

    cl_int err;
    
    // 2. Setup Devices (Platform 0 = NVIDIA, Platform 2 = Intel based on your clinfo)
    cl_device_id dev_nv = get_device_from_platform(0);
    cl_device_id dev_ig = get_device_from_platform(2);

    cl_context ctx_nv = clCreateContext(NULL, 1, &dev_nv, NULL, NULL, &err);
    cl_context ctx_ig = clCreateContext(NULL, 1, &dev_ig, NULL, NULL, &err);
    
    // Create queues OUT-OF-ORDER if supported, or standard if not.
    cl_command_queue q_nv = clCreateCommandQueue(ctx_nv, dev_nv, 0, &err);
    cl_command_queue q_ig = clCreateCommandQueue(ctx_ig, dev_ig, 0, &err);

    // 3. Allocate Buffers
    // NVIDIA buffers (Top part of A, all of B, Top part of C)
    cl_mem d_A_nv = clCreateBuffer(ctx_nv, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)M_NV * K * sizeof(float), h_A, &err);
    cl_mem d_B_nv = clCreateBuffer(ctx_nv, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)K * N * sizeof(float), h_B, &err);
    cl_mem d_C_nv = clCreateBuffer(ctx_nv, CL_MEM_WRITE_ONLY, (size_t)M_NV * N * sizeof(float), NULL, &err);

    // Intel buffers (Bottom part of A, all of B, Bottom part of C)
    float *h_A_offset = h_A + (M_NV * K); // Shift pointer to where Intel starts
    cl_mem d_A_ig = clCreateBuffer(ctx_ig, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)M_IG * K * sizeof(float), h_A_offset, &err);
    cl_mem d_B_ig = clCreateBuffer(ctx_ig, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)K * N * sizeof(float), h_B, &err);
    cl_mem d_C_ig = clCreateBuffer(ctx_ig, CL_MEM_WRITE_ONLY, (size_t)M_IG * N * sizeof(float), NULL, &err);

    // 4. Build Programs
    const char *src_nv[] = { kernel_uncoalesced };
    cl_program prog_nv = clCreateProgramWithSource(ctx_nv, 1, src_nv, NULL, &err);
    clBuildProgram(prog_nv, 1, &dev_nv, NULL, NULL, NULL);
    cl_kernel k_nv = clCreateKernel(prog_nv, "sgemm_uncoalesced", &err);

    const char *src_ig[] = { kernel_tiled };
    cl_program prog_ig = clCreateProgramWithSource(ctx_ig, 1, src_ig, NULL, &err);
    clBuildProgram(prog_ig, 1, &dev_ig, "-cl-fast-relaxed-math", NULL, NULL);
    cl_kernel k_ig = clCreateKernel(prog_ig, "sgemm_tiled", &err);

    // 5. Set Arguments
    // NVIDIA Args
    clSetKernelArg(k_nv, 0, sizeof(int), &M_NV);
    clSetKernelArg(k_nv, 1, sizeof(int), &N);
    clSetKernelArg(k_nv, 2, sizeof(int), &K);
    clSetKernelArg(k_nv, 3, sizeof(cl_mem), &d_A_nv);
    clSetKernelArg(k_nv, 4, sizeof(cl_mem), &d_B_nv);
    clSetKernelArg(k_nv, 5, sizeof(cl_mem), &d_C_nv);

    // Intel Args
    clSetKernelArg(k_ig, 0, sizeof(int), &M_IG);
    clSetKernelArg(k_ig, 1, sizeof(int), &N);
    clSetKernelArg(k_ig, 2, sizeof(int), &K);
    clSetKernelArg(k_ig, 3, sizeof(cl_mem), &d_A_ig);
    clSetKernelArg(k_ig, 4, sizeof(cl_mem), &d_B_ig);
    clSetKernelArg(k_ig, 5, sizeof(cl_mem), &d_C_ig);
    size_t localA = (size_t)TILE_M * TILE_K * sizeof(float);
    size_t localB = (size_t)TILE_K * TILE_N * sizeof(float);
    clSetKernelArg(k_ig, 6, localA, NULL);
    clSetKernelArg(k_ig, 7, localB, NULL);

    // Setup ND-Range limits
    size_t global_nv[2] = { N, M_NV }; // Uncoalesced
    size_t local_nv[2]  = { 16, 16 };

    size_t groupsX = (size_t)(N + TILE_N - 1) / TILE_N;
    size_t groupsY = (size_t)(M_IG + TILE_M - 1) / TILE_M;
    size_t global_ig[2] = { groupsX * WG_X, groupsY * WG_Y }; // Tiled
    size_t local_ig[2]  = { WG_X, WG_Y };

    printf("Launching kernels concurrently...\n");

    // Start timer
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // 6. Enqueue Kernels
    clEnqueueNDRangeKernel(q_nv, k_nv, 2, NULL, global_nv, local_nv, 0, NULL, NULL);
    clEnqueueNDRangeKernel(q_ig, k_ig, 2, NULL, global_ig, local_ig, 0, NULL, NULL);

    // 7. Read Results back to respective host memory blocks
    clEnqueueReadBuffer(q_nv, d_C_nv, CL_FALSE, 0, (size_t)M_NV * N * sizeof(float), h_C, 0, NULL, NULL);
    float *h_C_offset = h_C + (M_NV * N); // Shift pointer to where Intel writes
    clEnqueueReadBuffer(q_ig, d_C_ig, CL_FALSE, 0, (size_t)M_IG * N * sizeof(float), h_C_offset, 0, NULL, NULL);

    // 8. Wait for EVERYTHING to finish
    clFinish(q_nv);
    clFinish(q_ig);

    // Stop timer
    clock_gettime(CLOCK_MONOTONIC, &end);
    double exec_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    double ops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = (ops / exec_s) / 1e9;

    printf("========================================\n");
    printf("Dual-GPU Execution Time: %.6f seconds\n", exec_s);
    printf("Combined Performance: %.3f GFLOPS/sec\n", gflops);
    printf("========================================\n");

    // Cleanup
    clReleaseKernel(k_nv); clReleaseKernel(k_ig);
    clReleaseProgram(prog_nv); clReleaseProgram(prog_ig);
    clReleaseMemObject(d_A_nv); clReleaseMemObject(d_B_nv); clReleaseMemObject(d_C_nv);
    clReleaseMemObject(d_A_ig); clReleaseMemObject(d_B_ig); clReleaseMemObject(d_C_ig);
    clReleaseCommandQueue(q_nv); clReleaseCommandQueue(q_ig);
    clReleaseContext(ctx_nv); clReleaseContext(ctx_ig);
    free(h_A); free(h_B); free(h_C);

    return 0;
}