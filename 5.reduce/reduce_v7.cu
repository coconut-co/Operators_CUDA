// 在warp层面做rudece，可以不考虑warp_divergence(shared memory)层面
# include<cuda.h>
# include<cuda_runtime.h>
# include<iostream>

#define warpSize 32

template <int blockSize>
__global__ void reduce_warp_level(float* device_in, float* device_out, int n){
    // 当前线程的私有寄存器，即每个线程都会拥有一个sum寄存器
    float sum = 0;

    unsigned int tid = threadIdx.x;
    unsigned int gtid = blockDim.x * blockIdx.x + threadIdx.x; // blockdim.x每个block中线程数量，blockidx.x当前执行的block在grid中的索引
    // 分配的线程总数， blockdim：block中thread数，grid：grid中block数
    unsigned int total_thread_num = blockDim.x * gridDim.x;
    // 不指定一个线程处理2个元素，而是for循环自动确定每个线程处理的元素个数
    for (int i = gtid; i < n; i += total_thread_num){
        sum += device_in[i];
    }

    // 存储每个block中 每个warp的和
    // warpsums位于shared memory上
    // 一个block中有8个warp 256/32=8
    __shared__ float warpSums[blockSize / warpSize];

    // 当前线程在其所有warp范围内的ID
    const int laneID = tid % 32;
    // 当前线程所在warp在所有warp范围内的ID
    const int warpID = tid / warpSize;

    // 对当前线程所在的warp做warpshuffle操作，直接交换warp内线程间的寄存器数据
    sum = warpShuffle<blockSize>(sum);

    // 将每个warp中的第 0 个线程中的 sum 取出来放入 warpsums，准备求和（block尺度）
    if (laneID == 0){
        warpSums[warpID] = sum;
    }
    __syncthreads();

    // 至此, 求得了每个block中 每个warp层次的求和
    // 接下来，再使用第一个warp（laneid=0-31）对每个warp的reduce sum结果求和

    // 首先，把warpsums存入前blocksize / warpsize个线程的sum寄存器中
    sum = (tid < blockSize / warpSize) ? warpSums[laneID] : 0;

    // 然后使用第一个warp（laneid=0-31）对每个warp的reduce sum结果求和
    if (warpID == 0){
        sum = warpShuffle<blockSize/warpSize>(sum);
    }
    if (tid == 0){
        device_out[blockIdx.x] = sum;
    }

    
}