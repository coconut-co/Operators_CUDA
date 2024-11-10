# include <stdio.h>
# include <cuda.h>
# include <string.h>

__global__ void sum(float *x)
{
    int block_id = blockIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_id = threadIdx.x;
    printf("block_id= %d,thread_id in block = %d, global_id= %d\n",block_id, thread_id, global_id);
    x[global_id] += 1;      // 并行处理，每个数都加1
}

int main(){
    int N = 12;
    int bytes = N * sizeof(float);
    float *device = nullptr;
    float *host = nullptr;

    // 在GPU上分配显存, 输入：分配的设备内存的地址
    cudaMalloc((void **)&device, bytes);    //  为啥是二级指针

    // 在CPU上分配内存
    host = (float *)malloc(bytes);

    // 在CPU上初始数据
    printf("host original\n");
    for (int i = 0; i < N; ++i){
        host[i] = 0;
        printf("%f ", host[i]);
    }
    printf("\n");

    // host ---> device
    cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice);

    // 处理数据，启动GPU kernel
    sum<<<1, N>>>(device);          // 一个block，每个block有n个thread

    // device ---> host，dst，src
    cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost);

    printf("host current\n");
    for (int i = 0; i < N; ++i){
        printf("%f", host[i]);
    }
    printf("\n");

    cudaFree(device);
    free(host);
}