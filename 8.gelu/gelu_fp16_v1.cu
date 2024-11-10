# include<cuda.h>
# include<cuda_runtime.h>
# include<iostream>
#include <cuda_fp16.h>

// 让数据按内存对齐的方式存储
template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector{
    // 向量由size个类型为T的元素组成
    T value[Size];

    // 向量支持[]访问, 运算符重载：[重载对象]（传入的参数）{运算符重载的实现}
    __host__ __device__ inline const T& operator[](int i) const{
        return value[i];
    }

    __host__ __device__ inline T& operator[](int i){
        return value[i];
    }
};

__device__ float TanhApprox(float x){
    // ptx指令，是CUDA的更底层的语言，类似于汇编对于c/c++
    float r;
    asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
    return r;
}

// gelu公式：x / 2 * (1 + tan(0.7978845608028654 * (x + 0.044714998453855515 * x^3)))
template<typename T>
struct GeluFunctor{
    // static静态成员变量，多个对象可以共享，避免为每个对象重复存储
    static constexpr T alpha = static_cast<T>(0.7978845608028654);
    // constexpr 值在编译时就被确定，运行时bu'yong
    static constexpr T beta = static_cast<T>(0.044714998453855515);

    // 构造函数
    __device__ GeluFunctor() {};

    // 运算符重载，重载（）运算符
    __device__ T operator()(T x) const{
        const T half = static_cast<T>(0.5);
        const T one = static_cast<T>(1);
        const T tanh_in = alpha * (x + beta * x * x * x);
        return half * x * (one + tanh(tanh_in));  
    }
};

// 模板特化
// 专门处理half类型的GeluFunction
template<>
struct GeluFunctor<half>{
    static constexpr float alpha = GeluFunctor<float>::alpha;
    static constexpr float beta = GeluFunctor<float>::beta;

    // 处理float类型的GeluFunction
    GeluFunctor<float> float_functor;

    // 构造函数
    __device__ GeluFunctor() {};

    __device__ half operator()(const half x) const {
        // float_functor是一个 GeluFunctor<float>类型的对象，
        // GeluFunctor中重载了（）符号
        // 所以(x) 会直接对 x 进行 gelu 操作
        return static_cast<half>(float_functor(static_cast<float>(x)));
    }
};

// vecSize向量化读写的长度
template <int VecSize>
__global__ void FP16GeluCUDAKernel(const __half* x, __half* y, int n){
    // 向量化的load & store
    // 读取向量的offset--->每个线程处理元素的地址
    int offset = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) * VecSize;

    // 循环读取向量的stride
    int stride = static_cast<int>(blockDim.x * gridDim.x) * VecSize;

    GeluFunctor<half> gelu_fwd;
    __half y_reg[VecSize];

    // using 类型别名
    // ArrT: AlignedVector<__half, VecSize>类型的别名
    using ArrT = AlignedVector<__half, VecSize>;  // 声明向量类型
    for (; offset < n; offset += stride){
        // 每个线程所读向量的起始offset
        const __half* in = x + offset;

        if (VecSize == 1){
            y_reg[0] = gelu_fwd(in[0]);
        } else{
            // 标量计算
            for (int i = 0; i < VecSize; i++){
                y_reg[i] = gelu_fwd(in[i]);
            }
        }
        *reinterpret_cast<ArrT*>(y + offset) = *reinterpret_cast<ArrT*>(y_reg);
    }
}

int main(){
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);    

    int n = 1000;
    __half* device_x;
    __half* device_y;
    __half* host_x = new __half[n];
    __half* host_y = new __half[n];
    cudaMalloc((void** )&device_x, n * sizeof(__half));
    cudaMalloc((void** )&device_y, n * sizeof(__half));

    for (int i = 0; i < n; i++){
        host_x[i] = (__half)(i);
    }
    cudaMemcpy(device_x, host_x, n * sizeof(__half) , cudaMemcpyHostToDevice);
    // lambda表达式
    // 检查是否内存对齐
    // 返回一个bool值
    auto is_aligned = [](const void* p, int alignment){
        // p 转化为 uintprt_t类型
        // 取余，计算是否对齐
        return reinterpret_cast<uintptr_t>(p) % alignment == 0;
    };

    // alignof:返回内存对齐要求（对齐的字节数）
    constexpr auto kAlignment = alignof(AlignedVector<__half, 8>);

    // 每个线程一次处理8个元素
    if (n % 8 == 0 && is_aligned(host_x, kAlignment) && is_aligned(host_y, kAlignment)){
        int thread = std::min<int>(512, deviceProp.maxThreadsPerBlock); 
        int block = (n + thread - 1) / thread;
        block = std::min<int>(block, deviceProp.maxGridSize[0]);                                  
        FP16GeluCUDAKernel<4><<<block, thread>>>(device_x, device_y, n);  
        cudaMemcpy(host_y, device_y, sizeof(__half) * n, cudaMemcpyDeviceToHost); 
    }
    printf("pass\n");

    for (int i = 0; i < n; i++){
        std::cout << static_cast<float>(host_y[i]) << " ";
    }
    delete[] host_x;
    host_x = nullptr;
    delete[] host_y;
    host_y = nullptr;
    cudaFree(device_x);
    cudaFree(device_y);
}