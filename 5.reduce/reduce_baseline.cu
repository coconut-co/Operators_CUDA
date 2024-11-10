// 累加 
// reduce_baseline latency = 503.954346 ms
# include <cuda.h>
# include "cuda_runtime.h"
# include <stdio.h>

__global__ void reduce_baseline(const int* input, int* output, size_t n){
    // 由于只分配了1个block和thread,此时cuda程序相当于串行程序
    int sum = 0;
    for (int i = 0; i < n; ++i){
        sum += input[i];
    }
    *output = sum;
}

bool checkResult(int* out, int groundtruth){
    if (*out != groundtruth){
        return false;
    }
    return true;

}

int main(){
    // const int N = 32 * 1024 * 1024;
    const int N = 25600000;     

    //获取第0张卡的属性
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const int blockSize = 1;    // 一个thread
    const int gridSize = 1;     // 一个block
    // 定义分配的block数量和threads数量
    dim3 Grid(gridSize);
    dim3 Block(blockSize);

    // 分配内存和显存并初始化数据
    int* host_in = nullptr;
    int* host_out = nullptr;
    int* device_in = nullptr;
    int* device_out = nullptr;

    host_in = (int* )malloc(N * sizeof(int));
    host_out = (int* )malloc(1 * sizeof(int));
    cudaMalloc((void** )&device_in, N * sizeof(int));
    cudaMalloc((void** )&device_out, 1 * sizeof(int));

    // 初始化cpu数据
    for(int i = 0; i < N; i++){
        host_in[i] = 1;
    }   
    // 初始化gpu数据 
    cudaMemcpy(device_in, host_in, N * sizeof(int), cudaMemcpyHostToDevice);

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // 分配1个block和1个thread
    reduce_baseline<<<Grid, Block>>>(device_in, device_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop); 

    // 将结果拷回cpu
    cudaMemcpy(host_out, device_out, 1 * sizeof(int), cudaMemcpyDeviceToHost);

    // check准确性
    int groundtruth = N * 1;
    if (checkResult(host_out, groundtruth)){
        printf("the ans is right\n");
    }else{
        printf("the ans is flase\n");
        printf("groundtruth is %d\n", groundtruth);
        for (int i = 0; i < 1; i++){
            printf("device_out: %d\n", device_out[i]);
        }
    }
    printf("reduce_baseline latency = %f ms\n", milliseconds);

    cudaFree(device_in);
    cudaFree(device_out);
    free(host_in);
    free(host_out);

    return 0;
}