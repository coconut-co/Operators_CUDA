// v0：使用共享内存 smemSoftmax latency =  0.301440 ms

# include <cuda.h>
# include <cuda_runtime.h>
# include <iostream>

// 并行性：每个 block 处理 1024 个数据，每个 block 独享一个shared memory
// 选择 block 中 tid=0 的线程来求 max_val 和 sum
__global__ void softmax_v0(float* x, float* y, int n){
    int tid = threadIdx.x;
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;

    // 加载数据到 shared memory 空间
    __shared__ float smem[1024];
    smem[tid] = x[gtid];
    __syncthreads();

    // 每个 block 内求最大值
    __shared__ float max_val;
    max_val = 0;
    if (tid == 0){
        for (int i = 0; i < 1024; ++i){
            max_val = max(max_val, smem[i]);
        }
    }
    __syncthreads();

    // 每个 block 内求 sum
    __shared__ float sum;
    sum = 0;
    if (tid == 0){
        for (int i = 0; i < 1024; ++i){
            sum += exp(smem[i] - max_val);
        }
    }
    __syncthreads();

    // 计算softmax
    for (int i = tid; i < 1024; i += blockDim.x){
        y[gtid] = exp(smem[i] - max_val) / sum;
    }
    __syncthreads();
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
            printf("the ans is right\n");
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

    int blockSize = 1024;
    int gridSize = (N + blockSize - 1) / blockSize;
    dim3 block(blockSize);
    dim3 grid(gridSize);

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    softmax_v0<<<grid, block>>>(x_device, y_device, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(y_host, y_device, N * sizeof(float), cudaMemcpyDeviceToHost);

    softmaxCPU(x_host, groundtruth, 1000, 1024);
    checkResult(y_host, groundtruth, N);

    printf("smemSoftmax latency = %f ms\n", milliseconds);

    cudaFree(x_device);
    cudaFree(y_device);
    free(x_host);
    free(y_host);
    free(groundtruth);
}