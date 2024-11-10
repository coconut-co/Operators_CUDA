// 累加
// baseline                              reduce_baseline latency = 503.954346 ms
// 引入shared memory且并行化处理           reduce_v0 latency = 0.719744 ms    
// 手动选择线程                            reduce_v1 latency = 0.503008 ms
// 用位运算来代替除余操作和除法操作         reduce_v1 latency = 0.509184 ms
// 消除shared memory 的 bank conflict     reduce_v2 latency = 0.478688 ms
// 减少空闲线程                            reduce_v3 latency = 0.269568 ms
// 展开for循环最后一个warp，只有warp0工作   reduce_v4 latency = 0.220992 ms
// 完全展开for循环                         reduce_v5 latency = 0.217792 ms

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
__global__ void reduce_v5(float* device_in, float* device_out){
    __shared__ float smem[blockSize];  
    
    unsigned int tid = threadIdx.x;
    unsigned int gtid = blockIdx.x * (blockSize * 2) + threadIdx.x;
    smem[tid] = device_in[gtid] + device_in[gtid + blockSize];
    __syncthreads();

    // 在shared memory中累加
    BlockSharedMemReduce<blockSize>(smem);

    if (tid == 0){
        device_out[blockIdx.x] = smem[0];
    }
}

bool checkResult(float* device_out, float groundtruth, int n){
    float sum = 0;
    for (int i = 0; i < n; i++){
        sum += device_out[i];
    }
    if (sum != groundtruth){
        printf("device_out is %f\n", sum);
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
    dim3 block(blockSize / 2);

    float* device_in;
    float* host_in = (float*)malloc(N * sizeof(float));
    cudaMalloc((void**)&device_in, N * sizeof(float));
    float* device_out;
    float* host_out = (float*)malloc(gridSize * sizeof(float));
    cudaMalloc((void**)&device_out, gridSize * sizeof(float));

    for (int i = 0; i < N; ++i){
        host_in[i] = 1;
    }
    cudaMemcpy(device_in, host_in, N * sizeof(float), cudaMemcpyHostToDevice);

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v5<blockSize / 2><<<grid,block>>>(device_in, device_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(host_out, device_out, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

        float groundtruth = N * 1;
    bool is_right = checkResult(host_out, groundtruth, gridSize);
    if (is_right){
        printf("the ans is right!\n");
    }else{
        printf("groundtruth is %f\n", groundtruth);
        printf("the ans is false\n");
    }
    printf("reduce_v5 latency = %f ms\n", milliseconds);

    cudaFree(device_in);
    cudaFree(device_out);
    free(host_in);
    free(host_out);

    
    return 0;
}