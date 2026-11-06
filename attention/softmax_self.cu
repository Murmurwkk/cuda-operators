#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

__device__ __forceinline__ void get_max_sum(float& sum_self, float& max_self) 
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) 
    {
        float sum_other = __shfl_down_sync(0xffffffff, sum_self, offset);
        float max_other = __shfl_down_sync(0xffffffff, max_self, offset);
        float max_temp  = fmaxf(max_other, max_self);
        sum_self = sum_self * __expf(max_self - max_temp)
                 + sum_other * __expf(max_other - max_temp);
        max_self = max_temp;
    }
}

template<int threadsPerBlock>
__global__ void softmax_pass1_kernel(const float* input, int N, float* block_max, float* block_sum)
{
    __shared__ float smem_data[(threadsPerBlock / 32) * 2];
    int tid    = threadIdx.x;
    int id     = blockDim.x * blockIdx.x + tid;
    int stride = gridDim.x  * blockDim.x;
    float max_self = -FLT_MAX;
    float sum_self = 0.0f;
    const float4* in4 = reinterpret_cast<const float4*>(input);
    int N4 = (reinterpret_cast<uintptr_t>(input) % 16 == 0) ? N / 4 : 0;

    for (int i = id; i < N4; i += stride) 
    {
        float4 v = in4[i];
        float max_prev;

        max_prev = max_self; max_self = fmaxf(max_prev, v.x);
        sum_self = sum_self * __expf(max_prev - max_self) + __expf(v.x - max_self);

        max_prev = max_self; max_self = fmaxf(max_prev, v.y);
        sum_self = sum_self * __expf(max_prev - max_self) + __expf(v.y - max_self);

        max_prev = max_self; max_self = fmaxf(max_prev, v.z);
        sum_self = sum_self * __expf(max_prev - max_self) + __expf(v.z - max_self);

        max_prev = max_self; max_self = fmaxf(max_prev, v.w);
        sum_self = sum_self * __expf(max_prev - max_self) + __expf(v.w - max_self);
    }
    for (int i = N4 * 4 + id; i < N; i += stride) 
    {
        float val    = input[i];
        float max_prev = max_self;
        max_self = fmaxf(max_prev, val);
        sum_self = sum_self * __expf(max_prev - max_self) + __expf(val - max_self);
    }
    get_max_sum(sum_self, max_self);
    int warp_id  = tid / 32;
    int lane_id  = tid % 32;
    int num_warps = threadsPerBlock / 32;
    if (lane_id == 0) 
    {
        smem_data[warp_id]= sum_self;
        smem_data[warp_id + num_warps] = max_self;
    }
    __syncthreads();
    if (warp_id == 0) 
    {
        float b_sum = (tid < num_warps) ? smem_data[tid]: 0.0f;
        float b_max = (tid < num_warps) ? smem_data[tid + num_warps]: -FLT_MAX;
        get_max_sum(b_sum, b_max);
        if (tid == 0) 
        {
            block_max[blockIdx.x] = b_max;
            block_sum[blockIdx.x] = b_sum;
        }
    }
}

template<int threadsPerBlock>
__global__ void softmax_kernel_shfl(const float* input, float* output, int N, int num_blocks,
                                  const float* block_max, const float* block_sum)
{
    __shared__ float final_G_max, final_G_sum;
    if (threadIdx.x < 32) 
    {
        float b_max = -FLT_MAX, b_sum = 0.0f;
        for (int i = threadIdx.x; i < num_blocks; i += 32) 
        {
            float m = block_max[i], s = block_sum[i];
            float m_prev = b_max;
            b_max = fmaxf(m_prev, m);
            b_sum = b_sum * __expf(m_prev - b_max) + s * __expf(m - b_max);
        }
        get_max_sum(b_sum, b_max);
        if (threadIdx.x == 0) 
        {
            final_G_max = b_max;
            final_G_sum = b_sum;
        }
    }
    __syncthreads();
    float G_max = final_G_max;
    float inv_sum = __frcp_rn(final_G_sum);
    int id= blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x  * blockDim.x;
    const float4* in4  = reinterpret_cast<const float4*>(input);
    float4* out4 = reinterpret_cast<float4*>(output);
    int N4 = ((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(output)) % 16 == 0) ? N / 4 : 0;
    for (int i = id; i < N4; i += stride) 
    {
        float4 v = in4[i];
        v.x = __expf(v.x - G_max) * inv_sum;
        v.y = __expf(v.y - G_max) * inv_sum;
        v.z = __expf(v.z - G_max) * inv_sum;
        v.w = __expf(v.w - G_max) * inv_sum;
        out4[i] = v;
    }
    for (int i = N4 * 4 + id; i < N; i += stride) 
    {
        output[i] = __expf(input[i] - G_max) * inv_sum;
    }
}

// workspace contains at least 2*num_blocks floats and must remain alive until
// both launches complete. Concurrent requests require separate workspaces.
// Input/output may alias exactly; partially overlapping buffers are unsupported.
extern "C" cudaError_t softmax_vector_launch(const float* input, float* output, int N,
                                             float* workspace, int num_blocks, cudaStream_t stream)
{
    if (N < 0 || num_blocks < 1 || num_blocks > 1024) return cudaErrorInvalidValue;
    if (N == 0) return cudaSuccess;
    if (!input || !output || !workspace) return cudaErrorInvalidValue;
    softmax_pass1_kernel<512><<<num_blocks, 512, 0, stream>>>(input, N, workspace, workspace+num_blocks);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    softmax_kernel_shfl<512><<<num_blocks, 512, 0, stream>>>(input, output, N, num_blocks, workspace, workspace+num_blocks);
    return cudaGetLastError();
}

extern "C" void solve(const float* input, float* output, int N)
{
    if (N <= 0) return;
    const int threadsPerBlock = 512;
    int sm_count, device;
    if (cudaGetDevice(&device) != cudaSuccess) return;
    if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) return;
    int blocksPerGrid = min(1 + (N-1)/threadsPerBlock, min(sm_count * 4, 1024));
    float* workspace = nullptr;
    if (cudaMalloc(&workspace, 2*blocksPerGrid*sizeof(float)) != cudaSuccess) return;
    if (softmax_vector_launch(input, output, N, workspace, blocksPerGrid, nullptr) == cudaSuccess)
        cudaStreamSynchronize(nullptr);
    cudaFree(workspace);
}
