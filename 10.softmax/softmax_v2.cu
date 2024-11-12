// v0：使用共享内存 smemSoftmax latency =  0.301440 ms
// v1: 向量化的读写 smemSoftmax latency =  0.126240 ms
// v2: 在warp内规约 warpSoftmax latency =  0.071232 ms
# include <cuda.h>
# include <cuda_runtime.h>
# include <iostream>

__device__ float warpReduceMax(float val){
    for (int offset = 16; offset > 0; offset /= 2){
        // 依次规约
        // offset = 16
        // 前16个thread和后16个thread两两比较，比较的两个线程会获得相同的值——即这两个线程中较大的那个值
        // offset = 8
        // 前16个thread和后16个thread两两比较
        //              .
        //              .
        // 直到offset = 1, 获得当前warp的最大值（传播到整个 warp 中的所有线程）
        val = max(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    // 在 warp 内进行求和归约
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// 并行性：每个 block 处理 1024 个数据，每个 block 独享一个shared memory
__global__ void softmax_v2(float* x, float* y, int n){
    int tid = threadIdx.x;      // 256
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;

    // 加载数据到 shared memory 空间
    __shared__ float smem[1024];
    smem[tid * 4 + 0] = x[gtid * 4 + 0];
    smem[tid * 4 + 1] = x[gtid * 4 + 1];
    smem[tid * 4 + 2] = x[gtid * 4 + 2];
    smem[tid * 4 + 3] = x[gtid * 4 + 3];
    __syncthreads();

    // 计算每个线程四个元素局部最大值
    float local_max =fmaxf(fmaxf(smem[tid * 4 + 0], smem[tid * 4 + 1]),
                            fmaxf(smem[tid * 4 + 2], smem[tid * 4 + 3]));

    // 计算 warp 内的最大值
    float warp_max_val = warpReduceMax(local_max);

    // 线程 0 获得每个 warp 的最大值，并归约到 block 内最大值
    __shared__ float max_val;
    if (tid % 32 == 0){
        atomicMax(reinterpret_cast<int*>(&max_val), warp_max_val);
    }
    __syncthreads();

    // 计算 softmax 的分子部分并归约求和
    float local_exp_sum = exp(smem[tid * 4 + 0] - max_val) + exp(smem[tid * 4 + 1] - max_val) +
                          exp(smem[tid * 4 + 2] - max_val) + exp(smem[tid * 4 + 3] - max_val);

    float warp_sum = warpReduceSum(local_exp_sum);

    // 线程 0 获取每个 warp 的和，并归约得到 block 内的和
    __shared__ float sum;
    if (tid % 32 == 0) {
        atomicAdd(&sum, warp_sum);
    }
    __syncthreads();

    // 计算每个元素的 softmax 输出
    y[gtid * 4 + 0] = exp(smem[tid * 4 + 0] - max_val) / sum;
    y[gtid * 4 + 1] = exp(smem[tid * 4 + 1] - max_val) / sum;
    y[gtid * 4 + 2] = exp(smem[tid * 4 + 2] - max_val) / sum;
    y[gtid * 4 + 3] = exp(smem[tid * 4 + 3] - max_val) / sum;  
}

// int N = 1000 * 1024
// softmax公式
// e^(xi - max(xi)) / sigma(e^(xi - max(xi)))
void softmaxCPU(float* input, float* groundtruth, int rows, int cols){
    for (int j = 0; j < rows; ++j){
        float total = 0;
        float MAX = 0;
        for (int i = 0; i < cols; ++i){
            MAX = max(input[j * cols + i], MAX);          // 找最大值 max(x1)
        }
        for (int i = 0; i < cols; ++i){
            total += exp(input[j * cols + i] - MAX);      // 防止softmax溢出
        }
        for (int i = 0; i < cols; ++i){
            groundtruth[j * cols + i] = exp(input[j * cols + i] - MAX) / total;
        }
    }
}
void checkResult(float* out, float* groundtruth, int n){
    for (int i = 0; i < n; i += 1024){
        if (out[i] - groundtruth[i] > 1e-5){
            printf("the ans is false");
            printf(" the out is: %f", out[i]);
            printf(" the groundtruth is: %f\n", groundtruth[i]);
        }else{
            printf("the ans is true!");
            printf(" the out is: %f", out[i]);
            printf(" the groundtruth is: %f\n", groundtruth[i]);
        }
    }
}

int main(){
    const int N = 1000 * 1024;

    float* x_host = (float* )malloc(N * sizeof(float));
    float* y_host = (float* )malloc(N * sizeof(float));
    float* groundtruth = (float* )malloc(N * sizeof(float));
    float* x_device;
    float* y_device;
    cudaMalloc((void** )&x_device, N * sizeof(float));
    cudaMalloc((void** )&y_device, N * sizeof(float));

    // 初始化
    for (int i = 0; i < N; ++i){
        x_host[i] = i % 10;
    }
    cudaMemcpy(x_device, x_host, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;    // 1024 / 4 = 256
    int gridSize = (N + 1024 - 1) / 1024;
    dim3 block(blockSize);
    dim3 grid(gridSize);

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    softmax_v2<<<grid, block>>>(x_device, y_device, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(y_host, y_device, N * sizeof(float), cudaMemcpyDeviceToHost);

    softmaxCPU(x_host, groundtruth, 1000, 1024);
    checkResult(y_host, groundtruth, N);

    printf("warpSoftmax latency = %f ms\n", milliseconds);

    cudaFree(x_device);
    cudaFree(y_device);
    free(x_host);
    free(y_host);
    free(groundtruth);
}