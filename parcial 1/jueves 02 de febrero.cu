Practica 1 - 26/01/23

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void printKernel()
{
    printf("threads: %d  %d  %d \n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf(" blockId %d  %d %d \n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("gridDim %d %d %d \n", gridDim.x, gridDim.y, gridDim.z);
}

int main()
{
    int nx = 4;
    int ny = 4;
    int nz = 4;

    dim3 block(2, 2, 2);
    dim3 grid(nx / block.x, ny / block.y, nz / block.z);

    printKernel << <grid, block >> > ();

    return 0;
}

//warp - 32 threads 

Practica 2 - 26/01/23

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void multiplication(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int N = 3;
    const int a[N] = { 1, 0, 1 };
    const int b[N] = { 2,4,3 };
    int c[N] = { 0,0,0 };
    int size = N * sizeof(int);
    int* d_a = 0;
    int* d_b = 0;
    int* d_c = 0;

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    //copy from device to host
    cudaMemcpy(d_a,a,size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, a, size, cudaMemcpyHostToDevice);

    multiplication << <1, N >> > (d_a, d_b, d_c);

    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    printf("{1,0,1} * {2,4,3} = {%d,%d,%d}\n", c[0], c[1], c[2]);

    cudaDeviceReset();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}