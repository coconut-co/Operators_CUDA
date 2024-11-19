# include <iostream>
# include <cuda.h>
# include <cuda_runtime.h>
# include <cuda_fp16.h>

# define LOOP_TIMES 1000

// 一共启动1024个thread
// 每个warp_schedule发射32个thread，一个sm有4个warp_schedule，一共128个thread同时工作
// 一个SM有64个FP32 CUDA CORE, 128个thread肯定能吃满64个FP32 CUDA CORE

__global__ void FP32FLOPS(int* start, int* stop, float* x, float* y, float* result){
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    float a = x[gtid];
    float b = y[gtid];
    float res = 0;
    int start_time = 0;

    // 只测量计算所需时间，排除访存所需时间
    // asm 内联汇编 volatile 不允许编译器优化
    // 获取将当前 GPU 的时钟周期数（%%clock 寄存器）存储到 start_time 
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start_time) :: "memory");

    // 问题：为什么使用 4 条 FMA 指令获得GPU的峰值性能
    // 回答：隐藏 for 循环的比较和i++时间上的开销
    for(int i = 0; i < LOOP_TIMES; i++){
        res = a * b + res;
        res = a * b + res;
        res = a * b + res;
        res = a * b + res;
    }

    // sync all threads而不是仅仅同步该block内的thread
    // 只有 bar.stnc 能做到
    asm volatile("bar.sync 0;");
    int stop_time = 0;
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop_time) :: "memory");
    start[gtid] = start_time;
    stop[gtid] = stop_time;
    result[gtid] = res;
}

int main(){
    int N = 1024;

    float* x_host = (float* )malloc(N * sizeof(float));
    float* y_host = (float* )malloc(N * sizeof(float));
    float* x_device;
    float* y_device;
    cudaMalloc((void**)&x_device, N * sizeof(float));
    cudaMalloc((void**)&y_device, N * sizeof(float));

    for (int i = 0; i < N; i++){
        x_host[i] = static_cast<float>(i);
        y_host[i] = static_cast<float>(i);
    }
    cudaMemcpy(x_device, x_host, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y_host, N * sizeof(float), cudaMemcpyHostToDevice);

    int* startClock_host = (int* )malloc(N * sizeof(int));
    int* stopClock_host = (int* )malloc(N * sizeof(int));
    float* result_device;
    int* startClock_device;
    int* stopClock_device;
    cudaMalloc((void** )&result_device, N * sizeof(float));
    cudaMalloc((void** )&startClock_device, N * sizeof(float));
    cudaMalloc((void** )&stopClock_device, N * sizeof(float));

    // 启动1024个线程
    // FLOPS 浮点运算次数（Floating Point Operations Per Second）
    FP32FLOPS<<<1, 1024>>>(startClock_device, stopClock_device, x_device, y_device, result_device);
    cudaMemcpy(startClock_host, startClock_device, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(stopClock_host, stopClock_device, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);  

    // 计算量：LOOP_TIMES * n次res * FMA(1次MUL 1次ADD) * 1024个thread / 时间 ->FP32/s
    float FLOPS = (LOOP_TIMES * 4 * 2 * 1024) / (stopClock_host[0] - startClock_host[0]);
    // GPU最大时钟频率, props.clockRate(kHz)
    printf("GPU Max Clock rate: %0.2f GHz\n" , props.clockRate * 1e-6f);
    // GPU 的 SM 数量
    printf("SM counts is %d\n", props.multiProcessorCount);
    // GPU 的理论峰值算力
    printf("actual GPU peak FLOPS is %f (TFLOPS) \n", FLOPS * props.clockRate * 1e-9 * props.multiProcessorCount);

    free(startClock_host);
    free(stopClock_host);
    free(x_host);
    free(y_host);
    cudaFree(startClock_device);
    cudaFree(stopClock_device);
    cudaFree(result_device);
    cudaFree(x_device);
    cudaFree(y_device);
}