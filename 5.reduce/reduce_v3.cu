// 累加
// baseline                              reduce_baseline latency = 503.954346 ms
// 引入shared memory且并行化处理           reduce_v0 latency = 0.719744 ms    
// 手动选择线程                            reduce_v1 latency = 0.503008 ms
// 用位运算来代替除余操作和除法操作         reduce_v1 latency = 0.509184 ms
// 消除shared memory 的 bank conflict     reduce_v2 latency = 0.478688 ms
// 减少空闲线程                            reduce_v3 latency = 0.269568 ms
 
# include<iostream>
# include<cuda.h>
# include"cuda_runtime.h"

template<int blockSize>
__global__ void reduce_v3(float* device_in, float* device_out){
    unsigned int tid = threadIdx.x;
    unsigned int gtid = (blockSize * 2) * blockIdx.x + threadIdx.x;

    __shared__ float smem[blockSize];
    smem[tid] = device_in[gtid] + device_in[gtid + blockSize];
    __syncthreads();

    for (unsigned int index = blockDim.x / 2; index > 0; index >>= 1){
        if (tid < index){
            smem[tid] += smem[tid + index];
        }
        __syncthreads();
    } 

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

    const int blockSize = 256;      // thread数量
    int gridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    dim3 grid(gridSize);
    dim3 block(blockSize / 2);
    
    float* device_in;
    float* host_in = (float* )malloc(N * sizeof(float));
    cudaMalloc((void** )&device_in, N * sizeof(float));
    float* device_out;
    float* host_out = (float* )malloc(gridSize * sizeof(float));
    cudaMalloc((void** )&device_out, gridSize * sizeof(float));

    for (int i = 0; i < N; ++i){
        host_in[i] = 1;
    }
    
    cudaMemcpy(device_in, host_in, N * sizeof(float), cudaMemcpyHostToDevice);
    
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v3<blockSize / 2><<<grid, block>>>(device_in, device_out);
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
    printf("reduce_v3 latency = %f ms\n", milliseconds);

    cudaFree(device_in);
    cudaFree(device_out);
    free(host_in);
    free(host_out);

    
    return 0;
}
