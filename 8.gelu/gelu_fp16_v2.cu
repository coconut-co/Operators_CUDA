# include <cuda.h>
# include <iostream>
# include <cuda_runtime.h>
# include <cuda_fp16.h>

// alignas 指定对齐要求 （sizeof(T) * size）
// 使得 AlignedVector 对象可以像标准数组一样，通过下标访问其元素
template<typename T, int size>
struct alignas(sizeof(T) * size) AlignedVector{
    // 向量由size个T类型的元素组成
    T val[size];

    // 运算符重载
    // T&: 返回值类型（引用）opeartor：重载[]运算符， int i （参数列表：运算符需要的输入参数类型）
    __host__ __device__ inline const T& operator[](int i) const {
        return val[i];
    }
    __host__ __device__ inline T& operator[](int i){
        return val[i];
    }
};

// cuda内置math API： 计算tan(x)
__device__ float TanhApprox(float x) {
  // ptx指令，是CUDA的更底层的语言，类似于汇编对于C/C++
  float r;
  asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
  return r;
}

// gelu公式：x / 2 * (1 + tan(0.7978845608028654 * (x + 0.044714998453855515 * x^3))), 可上网自查
template<typename T>
struct GeluFunctor {
  static constexpr T alpha = static_cast<T>(0.7978845608028654);
  static constexpr T beta = static_cast<T>(0.044714998453855515);

  __device__ GeluFunctor() {};

  __device__ T operator()(T x) const {
    const T half = static_cast<T>(0.5);
    const T one = static_cast<T>(1);
    const T tanh_in = alpha * (x + beta * x * x * x);
    return half * x * (one + tanh(tanh_in));
  }
};

// 模板的特化
template<>
struct GeluFunctor<half>{
    static constexpr float alpha = GeluFunctor<float>::alpha;
    static constexpr float beta = GeluFunctor<float>::beta;
    GeluFunctor<float> float_functor;

    __device__ GeluFunctor() {};

    // half 返回值类型，重载（）运算符，参数列表：要传入的值
    __device__ half operator()(const half x) const {
        const float tan_in = 
            __half2float(__float2half_rn(alpha) * (x + __float2half_rn(beta) * x * x * x));

        // 计算tan值
        const float tanh_out = TanhApprox(tan_in);
        return __float2half_rn(0.5) * x * (__float2half_rn(1.0f) + __float2half_rn(tanh_out));
    }
// gelu公式：x / 2 * (1 + tan(0.7978845608028654 * (x + 0.044714998453855515 * x^3))),
    __device__ void apply2(half* y, const half* x) const {
        const half2 x2 = *(reinterpret_cast<const half2*>(x));   // 传入过来已经求出了offset，这里直接把指针转化为向量类型并解引用即可得到向量数据
        const float2 tanh_in = __half22float2(
            __hmul2(__float2half2_rn(alpha),
                __hadd2(x2, __hmul2(__hmul2(__hmul2(__float2half2_rn(beta), x2), x2), x2))));
        
        float2 tanh_out;
        // tanh_in是float 2类型变量； float2：两个float类型的元素组成的结构体
        tanh_out.x = TanhApprox(tanh_in.x);
        tanh_out.y = TanhApprox(tanh_in.y);
        const half2 y2 = __hmul2(__hmul2(__float2half2_rn(0.5F), x2), 
                                        __hadd2(__float2half2_rn(1.0F), __float22half2_rn(tanh_out)));

        // 向量化写回结果到缓存
        *reinterpret_cast<half2*>(y) = y2;
    }
};

template <int VecSize>
__global__ void FP16GeluCUDAKernel(const __half* x, __half* y, int n){
    // 向量化load & store
    // 读取向量的offset
    int offset = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) * VecSize;
    int stride = static_cast<int>(blockDim.x * gridDim.x) * VecSize;

    GeluFunctor<half> gelu_fwd;
    __half y_reg[VecSize];

    for (; offset < n; offset += stride){
        // 每个线程所读向量的起始offset
        const __half* in = x + offset;

        // 每个线程处理VecSize个元素
        if (VecSize == 1){
            y_reg[0] = gelu_fwd(in[0]);
        }else{
            for (int i = 0; i < VecSize; i += 2){
                gelu_fwd.apply2(y + offset + i, in + i);
            }
        }
    }
}

int main(){
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int n = 1000;
    __half* x_host = new __half[n];
    __half* y_host = new __half[n];
    __half* x_device;
    __half* y_device;
    cudaMalloc((void** )&x_device, n * sizeof(__half));
    cudaMalloc((void** )&y_device, n * sizeof(__half));

    for (int i = 0; i < n; i++){
        x_host[i] = (__half)(i);
    }
    cudaMemcpy(x_device, x_host, n * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y_host, n * sizeof(__half), cudaMemcpyHostToDevice);
    auto is_aligend = [](const void* p, int alignment){
        return reinterpret_cast<uintptr_t>(p) % alignment == 0;
    };

    // constexpr:常量表达式在编译时就能计算出结果
    // alignof:关键字，返回类型对齐要求（对齐的字节数）
    // 在使用alignof运算符时，传入的参数是一个类型，alignof的主要作用是返回这个类型的对齐要求
    // eg:size_t intAligment = alignof(int);   // 获取int类型的对齐要求
    // eg: struct Mystruct{
    //          char a;
    //          int b;
    // } 
    // size_t mystructAligment = alignof(Mystruct) // 获取Mystruct类型的对齐要求
    // 此时传入AlignedVector<__half, 8>，我们想知道8个__half类型的元素，对齐要求是什么，这些元素也是内存对其存储的
    constexpr auto kAlignment = alignof(AlignedVector<__half, 8>); 

    if (n % 8 == 0 && is_aligend(x_host, kAlignment) && is_aligend(y_host, kAlignment));
    {
        int thread = std::min<int>(512, deviceProp.maxThreadsPerBlock);
        int block = (n / 8 + thread - 1) / thread;
        block = std::min<int>(block, deviceProp.maxGridSize[0]);
        FP16GeluCUDAKernel<8><<<block, thread>>>(x_device, y_device, n);
        cudaMemcpy(y_host, y_device, n * sizeof(__half), cudaMemcpyDeviceToHost);        
    }
    printf("pass\n");

    for (int i = 0; i < n; i++){
        std::cout << static_cast<float>(y_host[i]) << " ";
    }
    delete[] x_host;
    x_host = nullptr;
    delete[] y_host;
    y_host = nullptr;
    cudaFree(x_device);
    cudaFree(y_device);

}