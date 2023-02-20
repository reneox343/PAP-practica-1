#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <time.h>

using namespace std;

__global__ void idx_Calc_tid(int* input) {
    int tid = threadIdx.x;
    printf("[DEVICE] threadIdx.x: %d, data: %d\n\r", tid, input[tid]);
}

__global__ void idx_Calc_gid(int* input) {
    int tid = threadIdx.x;
    int offset = blockIdx.x * blockDim.x;
    int gid = tid + offset;
    printf("[DEVICE] blockIdx.x: %d, threadIdx.x: %d, gid: %d, data: %d\n\r", blockIdx.x, tid, gid, input[gid]);
}

__global__ void idx_Calc_gid2d(int* input) {
    int tid = threadIdx.x;
    int blockOffset = blockIdx.x * blockDim.x;
    int rowOffset = blockIdx.y * gridDim.x * blockDim.x;
    int gid = tid + blockOffset + rowOffset;
    printf("[DEVICE] gridDim.x: %d, blockIdx.x: %d, blockIdx.y: %d, threadIdx.x: %d, gid: %d, data: %d\n\r", gridDim.x, blockIdx.x, blockIdx.y, threadIdx.x, gid, input[gid]);
}

__global__ void idx_Calc_gid2dMax(int* input) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int blockOffset = blockIdx.x * blockDim.x * blockDim.y;
    int rowOffset = blockIdx.y * gridDim.x * blockDim.x * blockDim.y;
    int gid = tid + blockOffset + rowOffset;
    printf("[DEVICE] gridDim.x: %d, blockIdx.x: %d, blockIdx.y: %d, threadIdx.x: %d, gid: %d, data: %d\n\r", gridDim.x, blockIdx.x, blockIdx.y, threadIdx.x, gid, input[gid]);
}

int main(){
    const int n = 32;
    int size = n * sizeof(n);

    int vector[n] = { 2, 7, 14, 4, 13, 18, 99, 5, 1, 3, 88, 77, 33, 55, 11, 44, 10, 20, 30, 40, 50, 60, 70, 80, 90, 0, 95, 85, 75, 65, 45, 35};
 
    int* devA = 0;
    int* devB = 0;
    int* devC = 0;
    int* devD = 0;

    cudaMalloc((void**)&devA, size);
    cudaMalloc((void**)&devB, size);
    cudaMalloc((void**)&devC, size);
    cudaMalloc((void**)&devD, size);

    cudaMemcpy(devA, vector, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, vector, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devC, vector, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devD, vector, size, cudaMemcpyHostToDevice);

    dim3 grid(2,2);
    dim3 block(4,2);

    //idx_Calc_tid << <grid, block >> > (devA);
    //idx_Calc_gid << <grid, block >> > (devB);
    //idx_Calc_gid2d << <grid, block >> > (devC);
    idx_Calc_gid2dMax << <grid, block >> > (devD);

    cudaDeviceSynchronize();

    cudaDeviceReset();
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFree(devD);
    return 0;
}