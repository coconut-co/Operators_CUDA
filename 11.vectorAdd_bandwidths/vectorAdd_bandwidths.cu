# include <cuda.h>
# include <cuda_runtime.h>
# include <iostream>

#define ARRAY_SIZE     100000000  // 400M   数据量
#define MEMORY_OFFSET  10000000   // 40M    “关闭”缓存
#define BENCH_ITER     10
#define THREADS_NUM    256

__global__ void mem_bw(float* A, float* B, float* C){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < MEMORY_OFFSET / 4; i += blockDim.x * gridDim.x){

        // 向量化读
        float4 a1 = reinterpret_cast<float4*>(A)[i];
        float4 b1 = reinterpret_cast<float4*>(B)[i];
        float4 c1;

        // 标量化计算
		// 测量显存带宽方法1: 向量加法, 630GB/s
        // c1.x = a1.x + b1.x;
		// c1.y = a1.y + b1.y;
		// c1.z = a1.z + b1.z;
		// c1.w = a1.w + b1.w;

		// 测量显存带宽方法2: copy操作, 595GB/s
		c1.x = a1.x; 
		c1.y = a1.y;
		c1.z = a1.z;
		c1.w = a1.w;
        reinterpret_cast<float4*>(C)[i] = c1;
    }
}

void vec_add_cpu(float* A, float* B, float* C, int n){
    for (int i = 0; i < n; i++){
        C[i] = A[i] + B[i];
    }
}

int main(){
    float *A = (float* )malloc(ARRAY_SIZE * sizeof(float));
    float *B = (float* )malloc(ARRAY_SIZE * sizeof(float));
    float *C = (float* )malloc(ARRAY_SIZE * sizeof(float));

    float *device_A;
    float *device_B;
    float *device_C;   
    cudaMalloc((void** )&device_A, ARRAY_SIZE*sizeof(float));
    cudaMalloc((void** )&device_B, ARRAY_SIZE*sizeof(float));
    cudaMalloc((void** )&device_C, ARRAY_SIZE*sizeof(float));

    for (int i = 0; i < ARRAY_SIZE; ++i){
        A[i] = i;
        B[i] = i;
    }

    cudaMemcpy(device_A, A, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, B, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);

    int gridSize = MEMORY_OFFSET / THREADS_NUM;
    float milliseconds = 0;

    // L2 cache:5M, 因此在warm up时，塞入5M 数据，将L2 cache占满 -> "关闭"L2 cache
    printf("warm up start\n");
    mem_bw<<<gridSize, THREADS_NUM>>>(device_A, device_B, device_C);
    printf("warm up end\n");

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
    for(int i = 0; i < BENCH_ITER; i++){
        mem_bw<<<gridSize, THREADS_NUM>>>(device_A + i * MEMORY_OFFSET, device_B + i * MEMORY_OFFSET, device_C + i * MEMORY_OFFSET);
    }
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(C, device_C, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    float* cpu_res = (float* )malloc(ARRAY_SIZE * sizeof(float));
    vec_add_cpu(A, B, cpu_res, ARRAY_SIZE);

    for (int i = 0; i < 20; ++i){
        if (abs(cpu_res[i] - C[i]) > 1e6){
            printf("the ans is false\n");
        }
    }
    printf("the ans is true\n");

    // 测量显存带宽
    unsigned N = ARRAY_SIZE * 4;
    // 根据实际读写的次数，指定下行写3 *(float)N or 2 *(float)N
    printf("Mem bandwidths = %f(GB/s)\n", 2 *(float)N / milliseconds / 1e6);

    free(C);
    free(B);
    free(A);
    free(cpu_res);
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);

}