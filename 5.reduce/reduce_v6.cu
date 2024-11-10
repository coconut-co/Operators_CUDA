// v0-v5cuda kernel中得到的是各个block负责范围内的总和
// 最终结果需要把各个block求得的和 再做 reduce sum

# include<cuda.h>
# include<iostream>
# include"cuda_runtime.h"

// 展开for循环
template<int blockSize>
__device__ void BlockSharedMemReduce(float* smem){
    // 一个block最多有1024个thread
    if (blockSize >= 1024) {
        if (threadIdx.x < 512){
            smem[threadIdx.x] += smem[threadIdx.x + 512];
        }
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (threadIdx.x < 256){
            smem[threadIdx.x] += smem[threadIdx.x + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (threadIdx.x < 128){
            smem[threadIdx.x] += smem[threadIdx.x + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (threadIdx.x < 64){
            smem[threadIdx.x] += smem[threadIdx.x + 64];
        }
        __syncthreads();
    }
    // 最后一个warp
    if (threadIdx.x < 32){
        volatile float* vshm = smem;
        if (blockDim.x >= 64){
            vshm[threadIdx.x] += vshm[threadIdx.x + 32];
        }
        vshm[threadIdx.x] += vshm[threadIdx.x + 16];
        vshm[threadIdx.x] += vshm[threadIdx.x + 8];
        vshm[threadIdx.x] += vshm[threadIdx.x + 4];
        vshm[threadIdx.x] += vshm[threadIdx.x + 2]; 
        vshm[threadIdx.x] += vshm[threadIdx.x + 1];
    }
}

template<int blockSize>
__global__ void reduce_v6(float* device_in, float* device_out, int nums){
    __shared__ float smem[blockSize];  
    
    unsigned int tid = threadIdx.x;
    unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total_thread_num = blockDim.x * gridDim.x;

    // 不用指定一个线程处理两个元素，而是通过循环来自动确定每个线程处理的元素个数
    float sum = 0.0f;
    for (int i = gtid; i < nums; i += total_thread_num){
        sum += device_in[i];
    }
    smem[tid] = sum;
    __syncthreads();

    // 在shared memory中累加
    BlockSharedMemReduce<blockSize>(smem);

    if (tid == 0){
        device_out[blockIdx.x] = smem[0];
    }
}

bool checkResult(float* device_out, float groundtruth){
    if (*device_out != groundtruth){
        printf("device_out is %f\n", *device_out);
        return false;
    }
    return true;
}

int main(){
    const int N = 25600000;

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const int blockSize = 256;
    // int gridSize = 100000;
    int gridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    dim3 grid(gridSize);
    dim3 block(blockSize);

    float* device_in;
    float* host_in = (float*)malloc(N * sizeof(float));
    cudaMalloc((void**)&device_in, N * sizeof(float));
    float* device_out;
    float* part_out;    // 存储每个block reduce的结果
    float* host_out = (float*)malloc(gridSize * sizeof(float));
    cudaMalloc((void**)&device_out, 1 * sizeof(float));
    cudaMalloc((void**)&part_out, gridSize * sizeof(float));

    for (int i = 0; i < N; ++i){
        host_in[i] = 1;
    }
    cudaMemcpy(device_in, host_in, N * sizeof(float), cudaMemcpyHostToDevice);

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v6<blockSize><<<grid,block>>>(device_in, part_out, N);
    reduce_v6<blockSize><<<1,block>>>(part_out, device_out, gridSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(host_out, device_out, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    float groundtruth = N * 1;
    bool is_right = checkResult(host_out, groundtruth);
    if (is_right){
        printf("the ans is right!\n");
    }else{
        printf("groundtruth is %f\n", groundtruth);
        printf("the ans is false\n");
    }
    printf("reduce_v6 latency = %f ms\n", milliseconds);

    cudaFree(device_in);
    cudaFree(device_out);
    free(host_in);
    free(host_out);

    
    return 0;
}