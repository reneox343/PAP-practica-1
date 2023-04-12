#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "random"
#include <stdio.h>
#include <iostream>
using namespace std;




__global__ void convolucion(int *image,int *a,int height,int width)
{
    int threadId = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int sum = 0;
    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++) {

            i = i * width;
            if (threadId + i + j > 0 && threadId + i + j < height * width) {
                sum += image[threadId + i + j];
            }
        }
    }
    image[threadId] = sum;

}
void printMatrix(int* image, int height, int width);
void printMatrix(int* image, int height, int width) {
    bool flag = 0;
    for (int i = 0; i < height*width; i++)
    {
        cout << image[i]<<" ";
        if (i % width == 0&&flag ==1)cout << endl;
        flag = 1;
    }
}
int main()

{
    int height = 10;
    int width = 10;
    int image_size = sizeof(int) * height * width;
    int conv_size = sizeof(int) *9;
    int *image_host = (int*)(malloc(image_size));
    int *conv_host = (int*)(malloc(conv_size));
    int *image_device, *conv_device;
    cudaMalloc((void**)&image_device, image_size);
    cudaMalloc((void**)&conv_device, conv_size);
    //init image
    for (int i = 0; i < height * width; i++)
    {
        image_host[i] = rand() % 100 + 1;
    }
    //init conv
    for (int i = 0; i < 9; i++)
    {
        conv_host[i] = 0;
    }
    conv_host[1] = 1;
    printMatrix(image_host, height, width);

    cudaMemcpy(image_device, image_host, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(conv_device, conv_host, conv_size, cudaMemcpyHostToDevice);
    dim3 grid(10,1,1);
    dim3 blocks(10,1,1);
    convolucion <<<grid, blocks >>> (image_device, conv_device, height, width);
    cudaMemcpy(image_host, image_device, image_size, cudaMemcpyDeviceToHost);
    cout << endl<<endl;
    printMatrix(image_host, height, width);


    return 0;
}

