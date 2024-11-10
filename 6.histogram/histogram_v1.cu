// 原子操作（串行）, histogram latency = 1.247872 ms
# include<cuda.h>
# include<cuda_runtime.h>
# include<iostream>

__global__ void histgram(int* hist_device, int* bin_device){
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    // 资源竞争：如果线程0和线程1处理的数一样，导致bin_device[1]不知道如何++
    // bin_data[hist_data[gtid]]++;
    // 原子加法，将所有并行线程强制转化为串行
    atomicAdd(&bin_device[hist_device[gtid]], 1);
}
void checkResult(int* device_out, int* groundtruth, int n){
    for (int i = 0; i < n; i++){
        if (device_out[i] != groundtruth[i]){
            printf("the ans is flase\n");
        }
    }
    printf("the ans is right\n");
}

int main(){
    cudaSetDevice(0);
    cudaDeviceProp deviceprop;
    cudaGetDeviceProperties(&deviceprop, 0);
 
    const int N = 25600000;
    const int blockSize = 256;
    int gridSize = std::min((N + 256 - 1) / 256, deviceprop.maxGridSize[0]);    // gridSize=10000
    dim3 block(blockSize);
    dim3 grid(gridSize);

    int* hist = (int* )malloc(N * sizeof(int));
    int* bin = (int* )malloc(256 * sizeof(int));
    int* hist_device;
    int* bin_device;
    cudaMalloc((void** )&hist_device, N * sizeof(int));
    cudaMalloc((void** )&bin_device, 256 * sizeof(int));

    // 初始化数据
    for (int i = 0; i < N; i++){
        hist[i] = i % 256;  // %让数在0-255之间
    }
    cudaMemcpy(hist_device, hist, N * sizeof(int), cudaMemcpyHostToDevice);

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    histgram<<<grid, block>>>(hist_device, bin_device);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(bin, bin_device, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    int *groundtruth = (int* )malloc(256 * sizeof(int));
    for (int i = 0; i < 256; i++){
        groundtruth[i] = 100000;
    }
    checkResult(bin, groundtruth, 256);
    printf("histogram latency = %f ms\n", milliseconds);  

    cudaFree(bin_device);
    cudaFree(hist_device);
    free(bin);
    free(hist);

}