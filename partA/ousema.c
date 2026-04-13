#define CL_TARGET_OPENCL_VERSION 120
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

// Improvement: Vectorized 2D Register Blocking (The ChatGPT Killer)
const char *kernelSource = 
"#define TS 64\n"
"#define WPT 4\n"
"#define RTS 16\n"
"__kernel void myGEMM_Vectorized(const int M, const int N, const int K,\n"
"                                const __global float4* A,\n"
"                                const __global float4* B,\n"
"                                __global float* C) {\n"
"    \n"
"    const int tidm = get_local_id(1);\n"
"    const int tidn = get_local_id(0);\n"
"    const int offsetM = TS * get_group_id(1);\n"
"    const int offsetN = TS * get_group_id(0);\n"
"    \n"
"    __local float Asub[TS][TS];\n"
"    __local float Bsub[TS][TS];\n"
"    \n"
"    float acc[WPT][WPT];\n"
"    for (int wm = 0; wm < WPT; wm++) {\n"
"        for (int wn = 0; wn < WPT; wn++) {\n"
"            acc[wm][wn] = 0.0f;\n"
"        }\n"
"    }\n"
"    \n"
"    // Loop over tiles - K is divided by 4 because we use float4\n"
"    const int numTiles = K / TS;\n"
"    for (int t = 0; t < numTiles; t++) {\n"
"        \n"
"        // Load tiles using float4 for maximum bandwidth\n"
"        for (int w = 0; w < WPT; w++) {\n"
"            int row = tidm + w * RTS;\n"
"            int col4 = tidn; // tidn goes 0..15, so 16 * 4 = 64\n"
"            \n"
"            // Load 4 floats at once from A and B\n"
"            float4 vecA = A[((offsetM + row) * (K/4)) + (t * (TS/4) + col4)];\n"
"            float4 vecB = B[((t * TS + row) * (N/4)) + (get_group_id(0) * (TS/4) + col4)];\n"
"            \n"
"            // Store into local memory as scalars for easy indexing in math\n"
"            Asub[row][col4*4 + 0] = vecA.x; Asub[row][col4*4 + 1] = vecA.y;\n"
"            Asub[row][col4*4 + 2] = vecA.z; Asub[row][col4*4 + 3] = vecA.w;\n"
"            \n"
"            Bsub[row][col4*4 + 0] = vecB.x; Bsub[row][col4*4 + 1] = vecB.y;\n"
"            Bsub[row][col4*4 + 2] = vecB.z; Bsub[row][col4*4 + 3] = vecB.w;\n"
"        }\n"
"        \n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        \n"
"        // Compute math with high register pressure to hide latency\n"
"        for (int k = 0; k < TS; k++) {\n"
"            float regB[WPT];\n"
"            for (int wn = 0; wn < WPT; wn++) {\n"
"                regB[wn] = Bsub[k][tidn + wn * RTS];\n"
"            }\n"
"            for (int wm = 0; wm < WPT; wm++) {\n"
"                float regA = Asub[tidm + wm * RTS][k];\n"
"                for (int wn = 0; wn < WPT; wn++) {\n"
"                    acc[wm][wn] += regA * regB[wn];\n"
"                }\n"
"            }\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    \n"
"    // Final write out\n"
"    for (int wm = 0; wm < WPT; wm++) {\n"
"        for (int wn = 0; wn < WPT; wn++) {\n"
"            int gR = offsetM + tidm + wm * RTS;\n"
"            int gC = offsetN + tidn + wn * RTS;\n"
"            C[gR * N + gC] = acc[wm][wn];\n"
"        }\n"
"    }\n"
"}\n";

int main() {
    const int SIZE = 8192; 
    const int M = SIZE, N = SIZE, K = SIZE;
    size_t bytes = (size_t)M * N * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    for(int i = 0; i < SIZE * SIZE; i++) { h_A[i] = 0.1f; h_B[i] = 0.2f; }

    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_A, NULL);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_B, NULL);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "myGEMM_Vectorized", NULL);

    clSetKernelArg(kernel, 0, sizeof(int), &M);
    clSetKernelArg(kernel, 1, sizeof(int), &N);
    clSetKernelArg(kernel, 2, sizeof(int), &K);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_C);

    // Global size is N/4, M/4 because of WPT=4
    size_t globalSize[2] = { (size_t)(N / 4), (size_t)(M / 4) }; 
    size_t localSize[2] = { 16, 16 };                
    cl_event prof_event;

    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, &prof_event);
    clWaitForEvents(1, &prof_event);

    cl_ulong start, end;
    clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);

    double time = (double)(end - start) / 1.0e9;
    double gflops = (2.0 * M * N * K) / (time * 1e9);

    printf("Vectorized GFLOPS: %f\n", gflops);

    clReleaseMemObject(d_A); clReleaseMemObject(d_B); clReleaseMemObject(d_C);
    clReleaseProgram(program); clReleaseKernel(kernel); clReleaseCommandQueue(queue); clReleaseContext(context);
    free(h_A); free(h_B); free(h_C);
    return 0;
}