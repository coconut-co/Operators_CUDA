# include <cuda.h>
# include <iostream>
# include "cuda_runtime.h"

// cpu
int fliter_cpu(int* src, int* dst, int n){
    int nres = 0;
    for (int i = 0; i < n; i++){
        if (src[i] > 0){
            dst[nres++] = src[i];
        }
    }
    return nres;
}

// gpu:naive
__global__ void filter_naive(int* src, int* dst, int* nres, int n)
{
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gtid < n && src[gtid] > 0){
        dst[atomicAdd(nres, 1)] = src[gtid];
    }
}

// gpu:block  filter_clock latency = 0.075168 ms,数据量2560000
// 先在shared_memory中统计一个block中大于0的数量，再把这个数累加到global_memory
__global__ void filter_block(int* src, int* dst, int* nres, int n){
    // shared_momory一个block内线程共享， 不同block不共享
    __shared__ int l_n;     // l_n每个block中大于0的个数
    int tid = threadIdx.x;
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    int total_thread_num = blockDim.x * gridDim.x;

    for (int i = gtid; i < n; i += total_thread_num){
        // 初始化每个block的l_n,只用一个线程
        if (tid == 0){
            l_n = 0;        
        }
        __syncthreads();

        // pos是每个线程私有的寄存器，且作为atomicAdd的返回值，表示当前线程对l_n原子加1之前的l_n
        int pos;
        if (i < n && src[i] > 0){
            // block0: 大于0的线程1，2，5，l_n=3
            // pos = 0, 1, 2
            // block1:大于0的线程1，2， l_n = 2
            // pos = 0, 1
            // block2:大于0的线程1，2，3，4， l_n = 4
            // pos = 0, 1, 2, 3
            pos = atomicAdd(&l_n, 1); // src[thread]>0的thread在当前block的index
        }
        __syncthreads();

        // 每个block tid0 作为leader，leader把每个block的l_n累加到全局计数器(nres)
        // 即所有block做一个reduce
        // l_n：每个block中大于0的数量
        // 原子加返回的l_n为 nres原子加l_n之前的值
        if (tid == 0){
            // block0: l_n = 3, nres = 3, 原子操作返回的是更新前nres的值l_n = 0
            // block1: l_n = 2, nres = 5，原子操作返回的是更新前nres的值l_n = 3
            // block2: l_n = 4, nres = 9，原子操作返回的是更新前nres的值l_n = 5
            l_n = atomicAdd(nres, l_n);
        }
        __syncthreads();

        if (i < n && src[i] > 0){
            // block 0:
            // l_n = 0; pos = 0, 1, 2
            // block 1;
            // l_n = 3; pos = 3, 4
            // block2; 
            // l_n = 5; pos = 5, 6, 7, 8
            pos += l_n;
            dst[pos] = src[i];
        }
        __syncthreads();
    }
} 

__device__ int atomicAggInc(int* ctr){
  unsigned int active = __activemask();   // 获得当前warp活跃的线程
  int leader = __ffs(active) - 1; // warp里面第一个src[threadIdx.x]>0的threadIdx.x
  int change = __popc(active);//warp mask中为1的数量
  int lane_mask_lt;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));   
  unsigned int rank = __popc(active & lane_mask_lt); 
  int warp_res;
  if(rank == 0)
    warp_res = atomicAdd(ctr, change);
  warp_res = __shfl_sync(active, warp_res, leader);
  return warp_res + rank; 
}


// warp
__global__ void filter_warp(int* src, int* dst, int* nres, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && src[i] > 0){
        dst[atomicAggInc(nres)] = src[i];
    }
}


void checkResult(int* device_out, int groundtruth, int n){
    if (*device_out != groundtruth){
        printf("the ans is false\n");
        printf("the groundtruth is: %d\n", groundtruth);
        printf("the device_out is: %d\n", *device_out);
    }
    printf("the ans is truth!\n");
    printf("the groundtruth is: %d\n", groundtruth);
    printf("the device_out is: %d\n", *device_out);
}

int main(){
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);    

    const int N = 2560000;
    const int blockSize = 256;
    int gridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    dim3 grid(gridSize);
    dim3 block(blockSize);

    int* src_host = (int* )malloc(N * sizeof(int));
    int* dst_host = (int* )malloc(N * sizeof(int));
    int* nres_host = (int* )malloc(1 * sizeof(int));    // 统计个数
    int* src_device;
    int* dst_device;
    int* nres_device;      // 统计个数
    cudaMalloc((void** )&src_device, N * sizeof(int));
    cudaMalloc((void** )&dst_device, N * sizeof(int));
    cudaMalloc((void** )&nres_device, 1 * sizeof(int));

    for (int i = 0; i < N; i++){
        src_host[i] = 1;
    }
    
    cudaMemcpy(src_device, src_host, N * sizeof(int), cudaMemcpyHostToDevice);

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    filter_warp<<<grid, block>>>(src_device, dst_device, nres_device, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(nres_host, nres_device, 1 * sizeof(int), cudaMemcpyDeviceToHost);

    int groundtruth = 0;
    for (int j = 0; j < N; j++){
        if (src_host[j] > 0){
            groundtruth++;
        }
    }
    checkResult(nres_host, groundtruth, N);
    printf("filter_clock latency = %f ms\n", milliseconds);   

    cudaFree(nres_device);
    cudaFree(dst_device);
    cudaFree(src_device);
    cudaFree(nres_host);
    cudaFree(dst_host);
    cudaFree(src_host);

}