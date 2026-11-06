#include <cuda_runtime.h>

__constant__ float inv_sqrt_2 = 0.70710678118f;
__constant__ float half = 0.5f;
__constant__ float one = 1.0f;

__device__ __forceinline__ float geglu(float x1, float x2)
{
    return x1 * half * x2 * (one + erff(x2 * inv_sqrt_2));
}

__global__ void geglu_kernel(const float* __restrict__ input, float* __restrict__ output, int halfN) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= halfN)
        return;

    int x1_idx = idx;
    int x2_idx = halfN + idx;

    float x1 = input[x1_idx];
    float x2 = input[x2_idx];
    output[x1_idx] = geglu(x1, x2);
}

__global__ void geglu_kernel_op(const float* __restrict__ input,
                                 float* __restrict__ output, int halfN)
{
    int offset = blockDim.x * blockIdx.x + threadIdx.x;
    int NBy4   = halfN / 4;
    if (offset >= NBy4)
        return;

    const float4* input_v4      = reinterpret_cast<const float4*>(input);
    const float4* input_v4_post = reinterpret_cast<const float4*>(input + halfN);
    float4*       output_v4     = reinterpret_cast<float4*>(output);

    float4 temp      = input_v4[offset];
    float4 temp_post = input_v4_post[offset];
    float4 out;
    out.x = geglu(temp.x, temp_post.x);
    out.y = geglu(temp.y, temp_post.y);
    out.z = geglu(temp.z, temp_post.z);
    out.w = geglu(temp.w, temp_post.w);
    output_v4[offset] = out;
}

extern "C" void solve(const float* input, float* output, int N) 
{
    int halfN = N / 2;
    if (halfN == 0)
        return;

    int threadsPerBlock = 256;
    if (halfN % 4 == 0) {
        int NBy4 = halfN / 4;
        int blocksPerGrid = (NBy4 + threadsPerBlock - 1) / threadsPerBlock;
        geglu_kernel_op<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    } else {
        int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;
        geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    }

    cudaDeviceSynchronize();
}