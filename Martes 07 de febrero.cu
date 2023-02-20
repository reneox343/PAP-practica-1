#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <time.h>

using namespace std;

#define GPUErrorAssertion(ans) {gpuAssert((ans), __FILE__, __LINE__);}

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n\r", cudaGetErrorString(code), file, line);
        if (abort)exit(code);
    }
}

__global__ void matrix_mult(int* a, int* b, int* c, int size) {
    int row = threadIdx.x / size;
    int col = threadIdx.x - row * size;

    int suma = 0;
    if (row < size && col < size) {
        for (int i = 0; i < size; i++) {
            suma += a[row * size + i] * b[i * size + col];
        }
    }
    c[threadIdx.x] = suma;
}

int main() {

    const int width = 2;
    int* host_a, * host_b, * host_c;
    int* dev_a, * dev_b, * dev_c;
    host_a = (int*)malloc(width * width * sizeof(int));
    host_b = (int*)malloc(width * width * sizeof(int));
    host_c = (int*)malloc(width * width * sizeof(int));
    cudaMalloc(&dev_a, width * width * sizeof(int));
    cudaMalloc(&dev_b, width * width * sizeof(int));
    cudaMalloc(&dev_c, width * width * sizeof(int));
    for (int i = 0; i < width * width; i++) {
        int r1 = (rand() % (256));
        int r2 = (rand() % (256));
        host_a[i] = r1;
        host_b[i] = r2;
        host_c[i] = 0;
    }
    cudaMemcpy(dev_a, host_a, width * width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, width * width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, host_c, width * width * sizeof(int), cudaMemcpyHostToDevice);

    matrix_mult << <1, 32 >> > (dev_a, dev_b, dev_c, width);
    cudaMemcpy(host_c, dev_c, width * width * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaDeviceReset();

    cout << "A:\n";
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            cout << host_a[i * width + j] << " ";
        }
        cout << "\n";
    }
    cout << "B:\n";
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            cout << host_b[i * width + j] << " ";
        }
        cout << "\n";
    }

    cout << "C: \n";
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            cout << host_c[i * width + j] << " ";
        }
        cout << "\n";
    }
    free(host_a);
    free(host_b);
    free(host_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);


    return 0;
}