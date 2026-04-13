#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void print_build_log(cl_program program, cl_device_id device) {
    size_t log_size = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    if (log_size > 1) {
        char *log = (char*)malloc(log_size + 1);
        if (!log) return;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = '\0';
        fprintf(stderr, "\n===== BUILD LOG =====\n%s\n=====================\n", log);
        free(log);
    }
}

static cl_device_id pick_device(cl_platform_id *out_platform) {
    cl_uint num_platforms = 0;
    CHECK_CL(clGetPlatformIDs(0, NULL, &num_platforms), "clGetPlatformIDs(count)");
    if (!num_platforms) {
        fprintf(stderr, "No OpenCL platforms found.\n");
        exit(EXIT_FAILURE);
    }

    cl_platform_id *platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    if (!platforms) exit(EXIT_FAILURE);

    CHECK_CL(clGetPlatformIDs(num_platforms, platforms, NULL), "clGetPlatformIDs(list)");

    cl_device_id device = NULL;
    cl_int err = CL_DEVICE_NOT_FOUND;

    // Prefer GPU, fallback to CPU
    for (cl_uint p = 0; p < num_platforms && !device; ++p) {
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        if (err == CL_SUCCESS) {
            *out_platform = platforms[p];
            break;
        }
    }
    for (cl_uint p = 0; p < num_platforms && !device; ++p) {
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        if (err == CL_SUCCESS) {
            *out_platform = platforms[p];
            break;
        }
    }

    free(platforms);

    if (!device) {
        fprintf(stderr, "No OpenCL GPU/CPU device found.\n");
        exit(EXIT_FAILURE);
    }

    return device;
}

const char *kernelSource =
"__attribute__((reqd_work_group_size(16,16,1)))\n"
"__kernel void sgemm_tiled(\n"
"    const int M, const int N, const int K,\n"
"    __global const float* restrict A,\n"
"    __global const float* restrict B,\n"
"    __global float* restrict C,\n"
"    __local float* As,\n"
"    __local float* Bs)\n"
"{\n"
"    const int lx = get_local_id(0);\n"
"    const int ly = get_local_id(1);\n"
"    const int gx = get_group_id(0);\n"
"    const int gy = get_group_id(1);\n"
"\n"
"    const int rowBase = gy * 64 + ly * 4;\n"
"    const int colBase = gx * 64 + lx * 4;\n"
"    const int tid = ly * 16 + lx;\n"
"\n"
"    float acc[4][4] = {{0.0f,0.0f,0.0f,0.0f},\n"
"                       {0.0f,0.0f,0.0f,0.0f},\n"
"                       {0.0f,0.0f,0.0f,0.0f},\n"
"                       {0.0f,0.0f,0.0f,0.0f}};\n"
"\n"
"    for (int k0 = 0; k0 < K; k0 += 16) {\n"
"        // Load A tile: 64 x 16\n"
"        // Load B tile: 16 x 64\n"
"        #pragma unroll\n"
"        for (int t = 0; t < 4; ++t) {\n"
"            int idx = tid + t * 256;\n"
"\n"
"            int aRow = idx / 16;\n"
"            int aCol = idx - aRow * 16;\n"
"            int gAR = gy * 64 + aRow;\n"
"            int gAC = k0 + aCol;\n"
"            As[aRow * 16 + aCol] = (gAR < M && gAC < K) ? A[gAR * K + gAC] : 0.0f;\n"
"\n"
"            int bRow = idx / 64;\n"
"            int bCol = idx - bRow * 64;\n"
"            int gBR = k0 + bRow;\n"
"            int gBC = gx * 64 + bCol;\n"
"            Bs[bRow * 64 + bCol] = (gBR < K && gBC < N) ? B[gBR * N + gBC] : 0.0f;\n"
"        }\n"
"\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"        #pragma unroll\n"
"        for (int kk = 0; kk < 16; ++kk) {\n"
"            float a[4];\n"
"            float b[4];\n"
"\n"
"            #pragma unroll\n"
"            for (int i = 0; i < 4; ++i)\n"
"                a[i] = As[(ly * 4 + i) * 16 + kk];\n"
"\n"
"            #pragma unroll\n"
"            for (int j = 0; j < 4; ++j)\n"
"                b[j] = Bs[kk * 64 + (lx * 4 + j)];\n"
"\n"
"            #pragma unroll\n"
"            for (int i = 0; i < 4; ++i)\n"
"                #pragma unroll\n"
"                for (int j = 0; j < 4; ++j)\n"
"                    acc[i][j] += a[i] * b[j];\n"
"        }\n"
"\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"\n"
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

int main(void) {
    const int SIZE = 8192;
    const int M = SIZE, N = SIZE, K = SIZE;

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float *h_A = (float*)malloc(bytesA);
    float *h_B = (float*)malloc(bytesB);
    float *h_C = (float*)malloc(bytesC);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host allocation failed.\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 2.0f;
    memset(h_C, 0, bytesC);

    cl_int err;
    cl_platform_id platform = NULL;
    cl_device_id device = pick_device(&platform);

    char device_name[256] = {0};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("Using device: %s\n", device_name);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_CL(err, "clCreateCommandQueue");

    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytesA, h_A, &err);
    CHECK_CL(err, "clCreateBuffer A");
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytesB, h_B, &err);
    CHECK_CL(err, "clCreateBuffer B");
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytesC, NULL, &err);
    CHECK_CL(err, "clCreateBuffer C");

    const char *src[] = { kernelSource };
    cl_program program = clCreateProgramWithSource(context, 1, src, NULL, &err);
    CHECK_CL(err, "clCreateProgramWithSource");

    const char *build_opts =
        "-cl-fast-relaxed-math -cl-mad-enable -cl-denorms-are-zero";

    err = clBuildProgram(program, 1, &device, build_opts, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Program build failed.\n");
        print_build_log(program, device);
        CHECK_CL(err, "clBuildProgram");
    }

    cl_kernel kernel = clCreateKernel(program, "sgemm_tiled", &err);
    CHECK_CL(err, "clCreateKernel");

    err = clSetKernelArg(kernel, 0, sizeof(int), &M);
    err |= clSetKernelArg(kernel, 1, sizeof(int), &N);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &K);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_A);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_B);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_C);

    size_t localA = (size_t)TILE_M * TILE_K * sizeof(float); // 64*16
    size_t localB = (size_t)TILE_K * TILE_N * sizeof(float); // 16*64
    err |= clSetKernelArg(kernel, 6, localA, NULL);
    err |= clSetKernelArg(kernel, 7, localB, NULL);
    CHECK_CL(err, "clSetKernelArg");

    size_t groupsX = (size_t)(N + TILE_N - 1) / TILE_N;
    size_t groupsY = (size_t)(M + TILE_M - 1) / TILE_M;

    size_t localSize[2] = { WG_X, WG_Y };
    size_t globalSize[2] = { groupsX * WG_X, groupsY * WG_Y };

    // Warm-up run
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
    CHECK_CL(err, "warmup kernel");
    clFinish(queue);

    cl_event evt;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, &evt);
    CHECK_CL(err, "timed kernel");
    clWaitForEvents(1, &evt);

    cl_ulong t0 = 0, t1 = 0;
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, NULL);
    clReleaseEvent(evt);

    double exec_s = (double)(t1 - t0) * 1e-9;
    double ops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = (ops / exec_s) / 1e9;

    printf("========================================\n");
    printf("Matrix Size: %d x %d\n", SIZE, SIZE);
    printf("Execution Time: %.6f seconds\n", exec_s);
    printf("Performance: %.3f GFLOPS/sec\n", gflops);
    printf("========================================\n");

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}