#define CL_TARGET_OPENCL_VERSION 120
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

// The Coalesced Kernel written as a string
const char *kernelSource = 
"__kernel void myGEMM_Coalesced(const int M, const int N, const int K,\n"
"                               const __global float* A,\n"
"                               const __global float* B,\n"
"                               __global float* C) {\n"
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

int main() {
    // 1. Set Matrix Dimensions (Start with 1024 for testing)
    const int SIZE = 8192; 
    const int M = SIZE, N = SIZE, K = SIZE;
    size_t bytes = M * N * sizeof(float);

    // 2. Allocate Host Memory (CORRECTED: Added the '*' to make them pointers)
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    // Initialize matrices with some dummy data
    for(int i = 0; i < SIZE * SIZE; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 3. OpenCL Setup Boilerplate
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    clGetPlatformIDs(1, &platform_id, NULL);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
    
    // CRITICAL: Enable profiling to measure time
    cl_command_queue queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, NULL);

    // 4. Allocate Device Memory (Buffers)
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_A, NULL);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_B, NULL);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

    // 5. Compile the Kernel Program
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "myGEMM_Coalesced", NULL);

    // 6. Set Kernel Arguments
    clSetKernelArg(kernel, 0, sizeof(int), &M);
    clSetKernelArg(kernel, 1, sizeof(int), &N);
    clSetKernelArg(kernel, 2, sizeof(int), &K);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_C);

    // 7. Execute the Kernel
    size_t globalSize[2] = { (size_t)N, (size_t)M }; // Work items
    size_t localSize[2] = { 16, 16 };                // Work group size
    cl_event prof_event;

    printf("Executing Kernel...\n");
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, &prof_event);
    clWaitForEvents(1, &prof_event);

    // 8. Calculate Performance
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    double execution_time_s = (double)(time_end - time_start) / 1.0e9;
    double total_ops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = (total_ops / execution_time_s) / 1.0e9;

    printf("========================================\n");
    printf("Matrix Size: %d x %d\n", SIZE, SIZE);
    printf("Execution Time: %f seconds\n", execution_time_s);
    printf("Performance: %f GFLOPS/sec\n", gflops);
    printf("========================================\n");

    // 9. Cleanup
    clReleaseMemObject(d_A); clReleaseMemObject(d_B); clReleaseMemObject(d_C);
    clReleaseProgram(program); clReleaseKernel(kernel); clReleaseCommandQueue(queue); clReleaseContext(context);
    free(h_A); free(h_B); free(h_C);

    return 0;
}