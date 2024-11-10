# include<cuda.h>
# include<cuda_runtime.h>
# include<iostream>

template<typename T>
__device__ T geluFunctor(T x){
    static constexpr T alpha = static_cast<T>(0.7978845608028654);
    static constexpr T beta = static_cast<T>(0.044714998453855515);
    const T half = static_cast<T>(0.5);
    const T one = static_cast<T>(1);
    const T tanh_in = alpha * (x + beta * x * x * x);
    return half * x * (one + tanh(tanh_in));
}

__global__ void geluCUDAKernel(float* x, float* y, int n){
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid < n){
        y[gtid] = geluFunctor<float>(x[gtid]);
    }
}

int main(){
    int N = 1000;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    float* x_host = (float* )malloc(N * sizeof(float));
    float* y_host = (float* )malloc(N * sizeof(float));
    float* x_device;
    float* y_device;
    cudaMalloc((void** )&x_device, N * sizeof(float));
    cudaMalloc((void** )&y_device, N * sizeof(float));

    for (int i = 0; i < N; i++){
        x_host[i] = static_cast<float>(i) ;
    }

    cudaMemcpy(x_device, x_host, N *sizeof(float), cudaMemcpyHostToDevice);
    geluCUDAKernel<<<1, 1024>>>(x_device, y_device, N);
    cudaMemcpy(y_host, y_device, N *sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++){
        printf("x: %f ", x_host[i]);
        printf("gelu(x): %f\n", y_host[i]);
    }

    cudaFree(y_device);
    cudaFree(x_device);
    free(y_host);
    free(x_host);
}