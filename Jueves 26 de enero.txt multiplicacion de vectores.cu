
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <random>
using namespace std;
__global__ void matrixMult(int *a,int *b,int *c,int size)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id > size)return;
    c[id] = a[id] * b[id];
}
__global__ void printKernel() {
    printf("threadIdx %d %d %d \n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("blockIdx %d %d %d \n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("gridDim %d %d %d \n", gridDim.x, gridDim.y, gridDim.z);
}


int main()
{
    const int arraySize = 32;
    
    int *a_host = (int*)malloc(sizeof(int) * arraySize);
    int *b_host = (int*)malloc(sizeof(int) * arraySize);
    int *c_host = (int*)malloc(sizeof(int) * arraySize);
    int *a_device;
    int *b_device;
    int *c_device;
    cudaMalloc(&a_device,sizeof(int)*arraySize);
    cudaMalloc(&b_device,sizeof(int)*arraySize);
    cudaMalloc(&c_device,sizeof(int)*arraySize);
    for (int i = 0; i < arraySize; i++)
    {
        a_host[i] = rand() % 10 + 1;
        b_host[i] = rand() % 10 + 1;
    }
    //host to device
    cudaMemcpy(a_device, a_host, sizeof(int) * arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_host, sizeof(int) * arraySize, cudaMemcpyHostToDevice);
    
    matrixMult <<< 1, arraySize >>> (a_device, b_device, c_device, arraySize);

    cudaMemcpy(a_host, a_device, sizeof(int) * arraySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_host, b_device, sizeof(int) * arraySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(c_host, c_device, sizeof(int) * arraySize, cudaMemcpyDeviceToHost);

    cout << "a: ";
    for (int i = 0; i < arraySize; i++)
    {
        cout << a_host[i] << " ";
    }
    cout << endl;

    cout << "b: ";
    for (int i = 0; i < arraySize; i++)
    {
        cout << b_host[i] << " ";
    }
    cout << endl;

    cout << "c: ";
    for (int i = 0; i < arraySize; i++)
    {
        cout << c_host[i] << " ";
    }
    cout << endl;

    //matrixMult<< 1, arraySize >>(a, b, c);
    free(a_host);
    free(b_host);
    free(c_host);
    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(c_device);

    dim3 block1(2, 2, 2);
    dim3 grid1(4 / block1.x, 4 / block1.y, 4 / block1.z);
    printKernel << <grid1, block1 >> > ();
    return 0;
}

