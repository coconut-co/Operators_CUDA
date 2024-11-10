// 累加
// baseline                             reduce_baseline latency = 503.954346 ms
// 引入shared memory且并行化处理          reduce_v0 latency = 0.719744 ms    
// 手动选择线程                          reduce_v1 latency = 0.503008 ms
// 用位运算来代替除余操作和除法操作        reduce_v1 latency = 0.509896 ms()

# include<iostream>
# include<cuda.h>
# include"cuda_runtime.h"

// blockSize作为模板参数的主要效果在于静态shared memory的申请需要传入编译期间常量指定大小
template<int blockSize>
__global__ void reduce_v1(float* device_in, float* device_out){
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    // 把blocksize个显存数据加载到每个block所独自对应的shared memory的空间上
    __shared__ float smem[blockSize];
    // 每个线程都加载一个元素到shared memory对应的位置
    smem[tid] = device_in[gtid];
    __syncthreads();    // 保证一个block内的所有thread都加载完

    // // 手动选择
    // for (int s = 1; s < blockDim.x; s *= 2){
    //     int index = 2 * s * tid;
    //     if(index < blockDim.x){
    //         smem[index] += smem[index + s];
    //     }
    //     __syncthreads();
    // }

    // 用位运算
    for (int s = 1; s < blockDim.x; s *= 2){
        if ((tid & (2 * s - 1)) == 0){
            smem[tid] += smem[tid + s];
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
    dim3 block(blockSize);
    
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
    reduce_v1<blockSize><<<grid, block>>>(device_in, device_out);
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
    printf("reduce_v1 latency = %f ms\n", milliseconds);

    cudaFree(device_in);
    cudaFree(device_out);
    free(host_in);
    free(host_out);

    
    return 0;
}