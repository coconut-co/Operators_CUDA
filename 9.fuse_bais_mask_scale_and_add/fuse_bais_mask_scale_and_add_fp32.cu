# include <cuda.h>
# include <cuda_runtime.h>
# include <iostream>

template<typename T>
struct MaskScaleAndElemwiseAddFunctor{
    // 有参构造函数
    MaskScaleAndElemwiseAddFunctor(const uint8_t* mask, const T* add_val, float scale)
    :_mask(mask), _add_val(add_val), _scale(scale)
    {}

    // 重载运算符（）
    __device__ T operator()(T x, int i) const{
        return x * static_cast<T>(static_cast<bool>(_mask[i]) * _scale) + _add_val[i];
    }

    const uint8_t* _mask;
    const T* _add_val;
    float _scale;
};

// 并行性：每个x可以独立计算
template<int biasSize, typename T, typename FUNCTION>
__global__ void FusedBaisAdd(FUNCTION functor, T* x, T* y, T* bias,
                                    const int n, const int bias_size){
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    // 最多分配gridDim * blockDim 个线程，
    // eg:n = 1050，gridDim = 2，blockDim = 512
    // tid = 1, -> 1 + 1024 = 1025, 线程1处理 1号和1025号数据
    for(int i = gtid; i < n; i += gridDim.x * blockDim.x){
        // 先加上偏置
        T tmp = x[i] + bias[i % bias_size];
        // 在做mask * scale + add
        y[i] = functor(tmp, i);
    }
};

// 向量化读写
template<int biasSize, typename T, typename FUNCTION>
__global__ void FusedBaisAddVecSmem(FUNCTION functor, T* x, T* y, T* bias,
                                    const int n, const int bias_size){
    __shared__ T smem[biasSize];
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    // 将bias放在shared_memory上，因为他被多次复用
    if (tid < bias_size){
        smem[tid] = bias[tid];
    }
    __syncthreads();
    // float4 向量化读取，一个线程读取四个数
    for(int i = gtid; i < n / 4; i += gridDim.x * blockDim.x){
        float4 a = reinterpret_cast<float4* >(x)[i];
        float4 b;

        b.x = functor(a.x + smem[(i * 4) % bias_size], i * 4);
        b.y = functor(a.y + smem[(i * 4 + 1) % bias_size], i * 4 + 1);
        b.z = functor(a.z + smem[(i * 4 + 2) % bias_size], i * 4 + 2);
        b.w = functor(a.w + smem[(i * 4 + 3) % bias_size], i * 4 + 3);

        reinterpret_cast<float4*>(y)[i] = b;
    }
}

void CheckRight(float* y_host ,float* groundtruth, int n){
    for (int i = 0; i < n; i++){
        if (y_host[i] == groundtruth[i]){
            //printf("the ans is right!\n");
        }else{
            printf("the ans is false\n");
        }
    }
}

// 实现fp32的算子fused biasadd mask scale add
// (x + bias) * mask * scale + add
int main(){
    constexpr int n = 100000;
    constexpr int bias_size = 10;
    float scale = 0.5;
    uint8_t* mask_tensor = new uint8_t[n];
    float* add_val = new float[n];

    // 初始化
    for (int i = 0; i < n; ++i)
    {
        mask_tensor[i] = (uint8_t)(i);
        add_val[i] = (float)(i);
    }
    
    float* x_host = (float* )malloc(sizeof(float) * n);
    float* y_host = (float* )malloc(sizeof(float) * n);
    float* y_host1 = (float* )malloc(sizeof(float) * n);
    float* bias_host = (float* )malloc(sizeof(float) * bias_size);
    for (int i = 0; i < n; ++i){
        x_host[i] = (float)(i);
        y_host[i] = 0.0f;
    }
    for (int i = 0; i < bias_size; ++i){
        bias_host[i] = (float)(i);
    }

    float* groundtruth = (float* )malloc(sizeof(float) * n);
    for (int i = 0; i < n; ++i){
        groundtruth[i] = (x_host[i] + bias_host[i % bias_size]) * static_cast<float>(static_cast<bool>(mask_tensor[i]) * scale) + add_val[i];
    }

    uint8_t* mask_device;
    float* add_device;
    cudaMalloc((void** )&mask_device, n * sizeof(uint8_t));
    cudaMalloc((void** )&add_device, n * sizeof(float));
    cudaMemcpy(mask_device, mask_tensor, n * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(add_device, add_val, n * sizeof(float), cudaMemcpyHostToDevice);
    float* x_device;
    float* y_device;
    float* bias_device;
    cudaMalloc((void** )&x_device, n * sizeof(float));
    cudaMalloc((void** )&y_device, n * sizeof(float));
    cudaMalloc((void** )&bias_device, bias_size * sizeof(float));
    cudaMemcpy(x_device, x_host, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y_host, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_device, bias_host, bias_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int blockSize = 512;
    int gridSize = std::min((n + blockSize - 1) / blockSize, deviceProp.maxGridSize[0]);
    dim3 block(blockSize);
    dim3 grid(gridSize);

    MaskScaleAndElemwiseAddFunctor<float> functor(mask_device, add_device, scale);   

    float milliseconds = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 朴素版
    cudaEventRecord(start);
    for (int i = 0; i < n; ++i){
        FusedBaisAdd<bias_size><<<grid, block>>>(functor, x_device, y_device, bias_device, n, bias_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(y_host, y_device, n * sizeof(float), cudaMemcpyDeviceToHost);

    float milliseconds1 = 0.0f;
    cudaEventRecord(start);
    for (int i = 0; i < n; ++i){
        FusedBaisAddVecSmem<bias_size><<<grid, block>>>(functor, x_device, y_device, bias_device, n, bias_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds1, start, stop);
    cudaMemcpy(y_host1, y_device, n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("朴素版\n");
    CheckRight(y_host, groundtruth, n);
    printf("it costs %f s \n", milliseconds/1000);   // 0.235473s 
    printf("向量版\n");
    CheckRight(y_host1, groundtruth, n);
    printf("it costs %f s \n", milliseconds1/1000);  // 0.265915s 

    cudaFree(x_device);
    cudaFree(y_device);
    cudaFree(bias_device);
    cudaFree(add_device);
    cudaFree(mask_device);
    free(x_host);
    free(y_host);
    free(y_host1);
    free(bias_host);
    free(groundtruth);
    delete mask_tensor;
    mask_tensor = nullptr;
    delete add_val;
    add_val = nullptr;
}