# include <cuda_runtime.h>
# include <cuda.h>
# include <iostream>

int main()
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cout << "there are no available device(s) that support cuda\n" << std::endl;
    } else {
        std::cout << "Detected " << deviceCount << " cuda capable device(s)\n" << std::endl;
    }

    for(int dev = 0; dev < deviceCount; dev++){
        cudaSetDevice(dev);                         // 选择要使用的设备
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);  // 获取设备的属性

        std::cout << "\n" << dev << " Device: " << deviceProp.name << std::endl;

    // 显存容量，单位为字节，1024*1024 = 1048576 转化为兆字节
    printf("  Total amount of global memory:                 %.0f MBytes ""(%llu bytes)\n",
             static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
             (unsigned long long)deviceProp.totalGlobalMem);

    // 时钟频率
    printf( "  GPU Max Clock rate:                            %.0f MHz (%0.2f ""GHz)\n",
        deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

    // L2 cache大小
    printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);

    // 每个线程块的共享内存总量
    printf("  Total amount of shared memory per block:       %zu bytes\n", deviceProp.sharedMemPerBlock);

    // 每个流式多处理器（sm）的共享内存总量
    printf("  Total shared memory per multiprocessor:        %zu bytes\n",deviceProp.sharedMemPerMultiprocessor);
    
    // 每个线程块可用的寄存数量
    printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
    
    // 线程束大小
    printf("  Warp size:                                     %d\n", deviceProp.warpSize);

    // 每个sm的最大线程数
    printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);

    // 每个线程块的最大线程数
    printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);

    // 线程块的维度
    printf("  Max dimension size of a block size (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    
    // grid的维度
    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
  }
  return 0;
}