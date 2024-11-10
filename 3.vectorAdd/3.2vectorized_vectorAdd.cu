# include <stdio.h>
# include <cuda.h>
# include <cuda_runtime.h>

#define ARRAY_SIZE	  100000000 // 400M   Array size has to exceed L2 size to avoid L2 cache residence
#define MEMORY_OFFSET 10000000	// “关闭”缓存
#define BENCH_ITER    10
#define THREADS_NUM   256

// 专业操作
__device__ __forceinline__
float4 LoadFromGlobalPTX(float4 *ptr) {
    float4 ret;
    // ptx指令，是CUDA的更底层的语言，类似于汇编对于C/C++
    asm volatile (
        "ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w)
        : "l"(ptr)
    );

    return ret;
}

__global__ void mem_bw (float* A, float* B, float* C){
    
    // 当前线程在所有block内的全局id
    int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    for (int i = idx; i < MEMORY_OFFSET / 4; i += blockDim.x * gridDim.x){

        // 向量化的读
        // float4：4个float类型的数据
        // reinterpret_cast: 强制类型转换  flota* -> float4* 
        float4 a1 = reinterpret_cast<float4*>(A)[i];
        float4 b1 = reinterpret_cast<float4*>(B)[i];
        float4 c1;

        // 标量化计算
        c1.x = a1.x + b1.x;
        c1.y = a1.y + b1.y;
		c1.z = a1.z + b1.z;
		c1.w = a1.w + b1.w;       

		// 测量显存带宽方法2:copy操作,242.3g/s	
		// c1.x = a1.x;
		// c1.y = a1.y;
		// c1.z = a1.z;
		// c1.w = a1.w;

		// 向量化的写入
		reinterpret_cast<float4*>(C)[i] = c1;
    }
}
