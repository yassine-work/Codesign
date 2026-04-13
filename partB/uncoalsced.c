#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int main(void) {
    const int SIZE = 8192;
    const int M = SIZE, N = SIZE, K = SIZE;

    float *h_A = (float*)malloc((size_t)M * K * sizeof(float));
    float *h_B = (float*)malloc((size_t)K * N * sizeof(float));
    float *h_C = (float*)malloc((size_t)M * N * sizeof(float));
    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 2.0f;
    memset(h_C, 0, (size_t)M * N * sizeof(float));

    cl_int err;
    
    // Get NVIDIA Platform (Platform 0 based on your clinfo)
    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, NULL, &num_platforms);
    cl_platform_id *platforms = malloc(sizeof(cl_platform_id) * num_platforms);
    clGetPlatformIDs(num_platforms, platforms, NULL);
    cl_device_id dev_nv;
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 1, &dev_nv, NULL);
    free(platforms);

    cl_context ctx = clCreateContext(NULL, 1, &dev_nv, NULL, NULL, &err);
    cl_command_queue q = clCreateCommandQueue(ctx, dev_nv, CL_QUEUE_PROFILING_ENABLE, &err);

    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)M * K * sizeof(float), h_A, &err);
    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)K * N * sizeof(float), h_B, &err);
    cl_mem d_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (size_t)M * N * sizeof(float), NULL, &err);

    cl_program prog = clCreateProgramWithSource(ctx, 1, &kernel_uncoalesced, NULL, &err);
    clBuildProgram(prog, 1, &dev_nv, NULL, NULL, NULL);
    cl_kernel k = clCreateKernel(prog, "sgemm_uncoalesced", &err);

    clSetKernelArg(k, 0, sizeof(int), &M);
    clSetKernelArg(k, 1, sizeof(int), &N);
    clSetKernelArg(k, 2, sizeof(int), &K);
    clSetKernelArg(k, 3, sizeof(cl_mem), &d_A);
    clSetKernelArg(k, 4, sizeof(cl_mem), &d_B);
    clSetKernelArg(k, 5, sizeof(cl_mem), &d_C);

    size_t global[2] = { N, M };
    size_t local[2]  = { 16, 16 };

    cl_event evt;
    clEnqueueNDRangeKernel(q, k, 2, NULL, global, local, 0, NULL, &evt);
    clWaitForEvents(1, &evt);

    cl_ulong t0 = 0, t1 = 0;
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, NULL);

    double exec_s = (double)(t1 - t0) * 1e-9;
    double ops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = (ops / exec_s) / 1e9;

    printf("========================================\n");
    printf("NVIDIA UNCOALESCED Performance: %.3f GFLOPS/sec\n", gflops);
    printf("========================================\n");

    return 0;
}