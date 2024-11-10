# include <stdio.h>
# include <cuda.h>
# include <cuda_runtime.h>
# include <vector>
# include <iostream>

__global__ void vec_add(float *x, float *y, float *z, int N)
{
    // 2D grid
    int idx = (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x);
    // 1D grid
    // int idx = blockDim.x * blockIdx.x + threarIdx.x;
    if (idx < N) z[idx] = x[idx] + y[idx];
}

void vec_add_cpu(float *x, float *y, float *z, int N)
{
    for (int i = 0; i < N; ++i)
    {
        z[i] = x[i] + y[i];
    }
}

int main()
{
    int N = 10000;      // 10000个数据
    int nbytes = N * sizeof(float);

    int block_size = 256;        // 每个block有256个线程

    // 1D grid 
    // int block = ceil((N + block_size - 1) / block_size);         // block =（10000 + 256 - 1）/ 256 = 40 thread = 40 * 256 = 10240

    // dim3 grid(block);

    // 2D grid
    int block = ceil(sqrt((N + block_size - 1 / block_size)));      // ceil向上取整 7， thread = 7 * 7 * 256 = 12544
    dim3 grid(block, block);

    float *dx, *hx;
    float *dy, *hy;
    float *dz, *hz;
    
    // 申请显存
    cudaMalloc((void **)&dx, nbytes);
    cudaMalloc((void **)&dy, nbytes);
    cudaMalloc((void **)&dz, nbytes);

    // 申请内存
    hx = (float *)malloc(nbytes);
    hy = (float *)malloc(nbytes);
    hz = (float *)malloc(nbytes);

    for(int i = 0; i < N; ++i)
    {
        hx[i] = 1;
        hy[i] = 1;
    }

    // 数据 host --> device
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, nbytes, cudaMemcpyHostToDevice);   

    // 计时
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // GPU计算
    vec_add<<<grid, block_size>>>(dx, dy, dz, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop); 

    // 结果 device --> host
    cudaMemcpy(hz, dz, nbytes, cudaMemcpyDeviceToHost);

    std::vector<float> hz_cpu(N);
    std::vector<float> hz_gpu(N);

    // 存储gpu计算数据
    for(int i = 0; i < N; ++i)
    {
        hz_gpu.push_back(hz[i]);
    }
    // CPU计算
    vec_add_cpu(hx, hy, hz, N);
    // 存储cpu计算数据
    for(int i = 0; i < N; ++i)
    {
        hz_cpu.push_back(hz[i]);
    }

    // 比较计算结果
    for(int i = 0; i < N; ++i)
    {
        if(abs(hz_cpu[i] - hz_gpu[i]) > 1e-6){
            std::cout << "error" << i << hz_cpu[i] << hz_gpu[i] << std::endl;
        }
    }
    printf("Result right\n");

    //free
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    free(hx);
    free(hy);
    free(hz);
}