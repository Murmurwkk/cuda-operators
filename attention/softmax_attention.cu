#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>
#include <math.h>

#define TILE_WIDTH 16
#define TPB 256
#define COARSE_FACTOR 1

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K, const float scale) 
{
    __shared__ float sa[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sb[TILE_WIDTH][TILE_WIDTH];
    int row = TILE_WIDTH * blockIdx.y + threadIdx.y;
    int col = COARSE_FACTOR * TILE_WIDTH * blockIdx.x + threadIdx.x;
    float sum[COARSE_FACTOR];
    #pragma unroll
    for (int i = 0; i < COARSE_FACTOR; i++)
    {
        sum[i] = 0.0f;
    }
    for (int phase = 0; phase < (N + TILE_WIDTH - 1) / TILE_WIDTH; phase++)
    {
        sa[threadIdx.y][threadIdx.x] =
            (row < M && (phase * TILE_WIDTH + threadIdx.x) < N)
                ? A[row * N + phase * TILE_WIDTH + threadIdx.x]
                : 0.0f;
        #pragma unroll
        for (int c = 0; c < COARSE_FACTOR; c++)
        {
            int col_new = col + c * TILE_WIDTH;
            sb[threadIdx.y][threadIdx.x] =
                ((phase * TILE_WIDTH + threadIdx.y) < N && col_new < K)
                    ? B[(phase * TILE_WIDTH + threadIdx.y) * K + col_new]
                    : 0.0f;
            __syncthreads();
            for (int j = 0; j < TILE_WIDTH; j++)
            {
                sum[c] += sa[threadIdx.y][j] * sb[j][threadIdx.x];
            }
            __syncthreads();
        }
    }
    #pragma unroll
    for (int c = 0; c < COARSE_FACTOR; c++)
    {
        int col_new = col + c * TILE_WIDTH;
        if (row < M && col_new < K)
        {
            C[row * K + col_new] = sum[c] / scale;
        }
    }
}

__global__ void row_softmax_kernel(float* input, int M, int N) 
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row < M)
    {
        __shared__ float s_max[TPB];
        __shared__ float s_sum[TPB];
        float max_val = -FLT_MAX;
        float sum_val = 0.0f;
        #pragma unroll
        for(int j=tid;j<N;j+=TPB)
        {
            float val=input[row*N+j];
            if(val>max_val)
            {
                sum_val=sum_val*expf(max_val-val)+1.0f;
                max_val=val;
            }
            else
            {
                sum_val+=expf(val-max_val);
            }
        }
        s_max[tid]=max_val;
        s_sum[tid]=sum_val;
        __syncthreads();
        #pragma unroll
        for(int offset=TPB/2;offset>0;offset>>=1)
        {
            if(tid<offset)
            {
                float max_temp=s_max[tid];
                float max_next=s_max[tid+offset];
                float sum_temp=s_sum[tid];
                float sum_next=s_sum[tid+offset];
                // A finite valid logit can equal -FLT_MAX. Nonempty states have
                // sum >= 1; the numerical maximum is not an emptiness flag.
                if(sum_next>0.0f)
                {
                    if(max_next>max_temp)
                    {
                        s_sum[tid]=sum_temp*expf(max_temp-max_next)+sum_next;
                        s_max[tid]=max_next;
                    }
                    else
                    {
                        s_sum[tid]=sum_next*expf(max_next-max_temp)+sum_temp;
                    }
                }
            }
            __syncthreads();
        }
        float row_max=s_max[0];
        float row_sum=s_sum[0];
        #pragma unroll
        for(int j=tid;j<N;j+=TPB)
        {
            input[row*N+j]=expf(input[row*N+j]-row_max)/row_sum;
        }   
    }
}

__global__ void transpose_kernel(const float* input, float* output, int rows, int cols) 
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c < cols && r < rows) 
    {
        output[c  * rows + r] = input[r * cols + c];
    }
}

// Preallocated, asynchronous single-head dense attention. Finite FP32 logits only;
// no causal/padding mask, batch/head stride or backward pass. Workspaces may not alias.
extern "C" cudaError_t attention_materialized_launch(
    const float* Q, const float* K, const float* V, float* output,
    int M, int N, int d, float* d_score, float* d_KT, cudaStream_t stream)
{
    if (!Q || !K || !V || !output || !d_score || !d_KT || M <= 0 || N <= 0 || d <= 0)
        return cudaErrorInvalidValue;
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_trans((d + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);
    transpose_kernel<<<grid_trans, block, 0, stream>>>(K, d_KT, N, d);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    dim3 grid_score((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    gemm_kernel<<<grid_score, block, 0, stream>>>(Q, d_KT, d_score, M, d, N, sqrtf(static_cast<float>(d)));
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    row_softmax_kernel<<<M, TPB, 0, stream>>>(d_score, M, N);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    dim3 grid_out((d + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    gemm_kernel<<<grid_out, block, 0, stream>>>(d_score, V, output, M, N, d, 1.0f);
    return cudaGetLastError();
}

extern "C" void solve(  const float* Q, 
                        const float* K, 
                        const float* V, 
                        float* output, 
                        int M, int N, int d )
{
    if (!Q || !K || !V || !output || M <= 0 || N <= 0 || d <= 0) return;
    float* d_score = nullptr;
    float* d_KT = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_score), static_cast<size_t>(M) * N * sizeof(float)) != cudaSuccess) return;
    if (cudaMalloc(reinterpret_cast<void**>(&d_KT), static_cast<size_t>(d) * N * sizeof(float)) != cudaSuccess) {
        cudaFree(d_score);
        return;
    }
    // The legacy void ABI cannot report failures. New callers should use the checked entry.
    (void)attention_materialized_launch(Q, K, V, output, M, N, d, d_score, d_KT, nullptr);
    cudaFree(d_score);
    cudaFree(d_KT);
}
