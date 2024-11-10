# include<stdio.h>
# include<cuda.h>


__global__ void hello_cuda(){
    // 线程的ID, 当前线程在所有block范围内的全局id
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("block_id[ %d ], thread_id[ %d ], idx[ %d ] hello cuda\n", blockIdx.x, threadIdx.x, idx);
}

int main(){
    // 输入：第一个1表示cuda分配了几个block，第二个1表示cuda分配的每个block中的thread
    hello_cuda<<<2, 2>>>();     // 两个block， 每个block有2个thread   ，4个thread

    // 同步，在该函数处 强制CPU 等待GPU上的CUDA kernel执行
    cudaDeviceSynchronize();
    
    return 0;
}