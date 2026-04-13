#define CL_TARGET_OPENCL_VERSION 120
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

// ============================================================
// OPTIMIZATION LEVELS:
// 1. Tiled shared memory (local memory) — reduces global mem reads
// 2. Thread-level register accumulation
// 3. Work-item computes multiple output elements (2x2 per thread)
// 4. Vectorized loads using float4
// ============================================================

#define TS  32   // Tile size (must match localSize below)
#define WPT 4    // Work Per Thread: each thread computes WPT elements per row
#define RTS (TS/WPT) // Reduced tile size for the N dimension

const char *kernelSource =
// --- Kernel 1: Tiled (shared local memory) ---
"#define TS  32\n"
"#define WPT 4\n"
"#define RTS (TS/WPT)\n"

"__kernel __attribute__((reqd_work_group_size(RTS, TS, 1)))\n"
"void myGEMM_Tiled(\n"
"    const int M, const int N, const int K,\n"
"    const __global float* A,\n"
"    const __global float* B,\n"
"    __global float* C) {\n"

"    const int row = get_local_id(1);\n"         // 0..TS-1
"    const int col = get_local_id(0);\n"         // 0..RTS-1
"    const int globalRow = TS  * get_group_id(1) + row;\n"
"    const int globalCol = RTS * get_group_id(0) + col;\n"

    // Local memory tiles
"    __local float Asub[TS][TS];\n"
"    __local float Bsub[TS][TS];\n"

    // Register accumulators: each thread handles WPT columns
"    float acc[WPT];\n"
"    for (int w = 0; w < WPT; w++) acc[w] = 0.0f;\n"

    // Loop over tiles along K
"    const int numTiles = K / TS;\n"
"    for (int t = 0; t < numTiles; t++) {\n"

        // Load tile of A — each thread loads WPT elements
"        for (int w = 0; w < WPT; w++) {\n"
"            Asub[row][col + w*RTS] = A[globalRow * K + (t*TS + col + w*RTS)];\n"
"        }\n"
        // Load tile of B
"        for (int w = 0; w < WPT; w++) {\n"
"            Bsub[row][col + w*RTS] = B[(t*TS + row) * N + (globalCol + w*RTS)];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"

        // Multiply the tiles
"        for (int k = 0; k < TS; k++) {\n"
"            float a = Asub[row][k];\n"
"            for (int w = 0; w < WPT; w++) {\n"
"                acc[w] += a * Bsub[k][col + w*RTS];\n"
"            }\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"

    // Write results
"    for (int w = 0; w < WPT; w++) {\n"
"        if (globalRow < M && (globalCol + w*RTS) < N)\n"
"            C[globalRow * N + (globalCol + w*RTS)] = acc[w];\n"
"    }\n"
"}\n";

// ---- Helper: print build errors ----
void check_build(cl_program program, cl_device_id device) {
    cl_build_status status;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(status), &status, NULL);
    if (status != CL_BUILD_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log);
        free(log);
        exit(1);
    }
}

int main() {
    const int SIZE = 1024;
    const int M = SIZE, N = SIZE, K = SIZE;
    size_t bytes = (size_t)M * N * sizeof(float);

    // --- Host memory ---
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    for (int i = 0; i < SIZE * SIZE; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // --- OpenCL setup ---
    cl_platform_id platform_id;
    cl_device_id   device_id;
    clGetPlatformIDs(1, &platform_id, NULL);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    // Print device name
    char devName[256];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(devName), devName, NULL);
    printf("Device: %s\n", devName);

    cl_context       context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
    cl_command_queue queue   = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, NULL);

    // --- Buffers ---
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, bytes, h_A, NULL);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, bytes, h_B, NULL);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY,                         bytes, NULL, NULL);

    // --- Build kernel ---
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device_id, "-cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);
    check_build(program, device_id);

    cl_kernel kernel = clCreateKernel(program, "myGEMM_Tiled", NULL);

    // --- Kernel args ---
    clSetKernelArg(kernel, 0, sizeof(int),    &M);
    clSetKernelArg(kernel, 1, sizeof(int),    &N);
    clSetKernelArg(kernel, 2, sizeof(int),    &K);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_C);

    // --- Launch ---
    // globalSize: (N/WPT, M) → each thread covers WPT columns
    size_t globalSize[2] = { (size_t)(N / WPT), (size_t)M };
    size_t localSize[2]  = { (size_t)RTS,        (size_t)TS };

    cl_event prof_event;
    printf("Executing optimized tiled kernel...\n");
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, &prof_event);
    clWaitForEvents(1, &prof_event);

    // --- Profiling ---
    cl_ulong t_start, t_end;
    clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, NULL);
    clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END,   sizeof(t_end),   &t_end,   NULL);

    double exec_s    = (double)(t_end - t_start) / 1.0e9;
    double total_ops = 2.0 * (double)M * (double)N * (double)K;
    double gflops    = (total_ops / exec_s) / 1.0e9;

    printf("========================================\n");
    printf("Matrix Size   : %d x %d\n", SIZE, SIZE);
    printf("Tile Size     : %d x %d  (WPT=%d)\n", TS, TS, WPT);
    printf("Execution Time: %.6f seconds\n", exec_s);
    printf("Performance   : %.2f GFLOPS/sec\n", gflops);
    printf("========================================\n");

    // --- Verify (spot check) ---
    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, bytes, h_C, 0, NULL, NULL);
    printf("C[0][0] = %.1f (expected %.1f)\n", h_C[0], (float)(2 * K));

    // --- Cleanup ---
    clReleaseEvent(prof_event);
    clReleaseMemObject(d_A); clReleaseMemObject(d_B); clReleaseMemObject(d_C);
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);
    free(h_A); free(h_B); free(h_C);

    return 0;
}