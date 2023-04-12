#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <random>
using namespace std;

void bubbleSort(int* array, int size);
void printArray(int* array, int size, string name);

__global__ void bubbleSortGPU(int* array,int size)
{
    __shared__ int sharedmem[20];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    sharedmem[idx] = array[idx];
    __syncthreads();
    if (idx == 0) {
        return;
    }
    for (int i = 0; i < size; i++)
    {
        if (sharedmem[idx - 1] > sharedmem[idx]) {
            int aux = sharedmem[idx];
            sharedmem[idx] = sharedmem[idx - 1];
            sharedmem[idx - 1] = aux;
        }
    }
    array[idx] = sharedmem[idx];

}

int main()
{
    int size = 10;
    int memSize = size * sizeof(int);
    int *array_host = (int*)malloc(memSize);
    int *arraySortedCPU_host = (int*)malloc(memSize);
    int *arraySortedGPU_host = (int*)malloc(memSize);
    int* array_device;
    cudaMalloc(&array_device, memSize);

    for (int i = 0; i < size; i++)
    {
        array_host[i] = rand() % 50+1;
        arraySortedCPU_host[i] = array_host[i];
    }

    //gpu sorting
    cudaMemcpy(array_device, array_host,memSize, cudaMemcpyHostToDevice);
    bubbleSortGPU <<< 1, size >>> (array_device, size);
    cudaMemcpy(arraySortedGPU_host, array_device,memSize, cudaMemcpyDeviceToHost);

    //cpu sorting
    bubbleSort(arraySortedCPU_host, size);
    printArray(array_host, size, "original");
    printArray(arraySortedCPU_host, size, "sorted cpu");
    printArray(arraySortedGPU_host, size, "sorted gpu");
    free(array_host);
    free(arraySortedCPU_host);
    free(arraySortedGPU_host);
    cudaFree(array_device);
    return 0;
}

void printArray(int*array,int size,string name){
    cout << name << ": ";
    for (int i = 1; i < size; i++)
    {
        cout << array[i] << " ";
    }
    cout << endl;
}

void bubbleSort(int *array,int size){
   
    for (int i = 0; i < size; i++)
    {
        for (int j = 1; j < size; j++)
        {
            if (array[j - 1] > array[j]) {
                int aux = array[j];
                array[j] = array[j - 1];
                array[j - 1] = aux;

            }
        }
    }
}