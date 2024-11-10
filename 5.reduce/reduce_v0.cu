// 累加
// baseline                              reduce_baseline latency = 503.954346 ms
// v0引入shared memory且并行化处理         reduce_v0 latency = 0.719744 ms     


# include<cuda.h>
# include<cuda_runtime.h>
# include<iostream>

// blockSize是模板参数，在核函数被调用时就已经知道他的值<const int>
// blockSize作为模板参数的主要效果用于静态shared memory的申请需要传入编译期间常量指定大小
template<int blockSize>
__global__ void reduce_v0(int* device_in, int* device_out){ 

    int tid = threadIdx.x;      // 当前线程在其block内的id, tid最大为255
    // int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int gtid = blockIdx.x * blockSize + threadIdx.x;


    // 申请blocksize * float 个共享内存空间, 把blocksize个显存数据加载到每个block所独自对应的shared memory的空间上去
    __shared__ float smem[blockSize];
    // 每个线程加载一个元素到shared memory对应的位置
    smem[tid] = device_in[gtid];
    __syncthreads();    // 等待所有thread 都完成写入 shared memory操作

    // 在一个block内操作，小于线程数量
    // 最后每个block的累加结果都保存在tid为0的线程上面
    for (int idx = 1; idx < blockDim.x; idx *= 2){  
        // stage1: input: 0 1 2 3 4 5 6 
        //                |/  |/  |/ 
        // stage1: output:0   2   4
        // stage2: input: 0 2 4 6 8 10
        //                |/  |/  |/
        // stage2: output:0   4   8
        if (tid % (2 * idx) == 0){
            smem[tid] += smem[tid + idx];   // 干活的线程id
        }
        // v0 warp divergence?，没发生，因为没有else分支
        // 现在的v0和v1性能大体相似
        // v0慢的原因在于下一行使用了除余%，除余%是个非常耗时的指令，我会在下个版本对这里进一步修正
        // 可尝试把下一行替换为`if ((tid & (2 * index - 1)) == 0) {`, 性能大概可以提升30%～50%
            __syncthreads();
    }

    // reduce结果写回显存，10w个block内部的reduce和
    if (tid == 0){
        device_out[blockIdx.x] = smem[0];
    }
}

void checkResult(int* device_out, int groundtruth, int n){
    int result = 0;
    for (int i = 0; i < n; ++i){
        result += device_out[i];
    }
    if (result != groundtruth){
        printf("the ans is false!\n");
        printf("device_out = %d\n", result);
        printf("groundtruth = %d\n", groundtruth);
    }else{
        printf("the ans is truth!\n");
    }
}

int main(){

    const int N = 25600000;

    // 获取第0张卡的属性
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);   

    // 定义分配的block数量和threads数量
    const int blockSize = 256;      // threads
    int gridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);   // 10w个数据
    dim3 Grid(gridSize);
    dim3 Block(blockSize);

    // 分配内存和显存并初始化数据
    int* device_in;
    int* device_out;
    int* host_in = (int* )malloc(N * sizeof(int));
    int* host_out = (int* )malloc(gridSize * sizeof(int));
    cudaMalloc((void** )&device_in, N * sizeof(int));
    cudaMalloc((void** )&device_out, gridSize * sizeof(int));

    // 初始化cpu数据
    for(int i = 0; i < N; i++){
        host_in[i] = 1;
    }   

    // 初始化gpu数据 
    cudaMemcpy(device_in, host_in, N * sizeof(int), cudaMemcpyHostToDevice);

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // 分配1个block和1个thread
    reduce_v0<blockSize><<<Grid, Block>>>(device_in, device_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop); 

    // 将结果拷回cpu
    cudaMemcpy(host_out, device_out, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    // 验证
    int groundtruth = N * 1;
    checkResult(host_out, groundtruth, gridSize);
    printf("reduce_v0 latency = %f ms\n", milliseconds);

    cudaFree(device_in);
    cudaFree(device_out);
    free(host_in);
    free(host_out);

    return 0;
}