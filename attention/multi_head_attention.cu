#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#define BLOCK 32
#define WARPS_PER_BLOCK 4   // number of Q-rows processed per thread block
#define MAX_DK_TILES 4      // supports d_k up to 128

// Warp-level row max reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

// Warp-level row sum reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// FlashAttention fused kernel
// Grid:  (ceil(N / WARPS_PER_BLOCK), h)
// Block: (BLOCK, WARPS_PER_BLOCK)  = (32, 4) = 128 threads
//   threadIdx.x  = lane within warp  (KV position or d_k column)
//   threadIdx.y  = warp index        (Q row within block)
//   Each warp (same ty, all tx) handles one Q row.
__global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ output,
    int N, int d_k, int d_model, float scale)
{
    const int tx = threadIdx.x;                              // lane [0,31]
    const int ty = threadIdx.y;                              // warp [0, WARPS_PER_BLOCK)
    const int q_row    = blockIdx.x * WARPS_PER_BLOCK + ty;
    const int head_off = blockIdx.y * d_k;
    const int d_k_tiles    = (d_k + BLOCK - 1) / BLOCK;
    const int num_kv_blocks = (N  + BLOCK - 1) / BLOCK;

    // Shared memory: each warp gets its own row in sQ/sS;
    // sK is shared across warps (all warps load the same K/V block).
    __shared__ float sQ[WARPS_PER_BLOCK][BLOCK + 1];
    __shared__ float sK[BLOCK][BLOCK + 1];   // K tile (reused for V)
    __shared__ float sS[WARPS_PER_BLOCK][BLOCK + 1];  // P row per warp

    float O_local[MAX_DK_TILES];
    #pragma unroll
    for (int t = 0; t < MAX_DK_TILES; t++) O_local[t] = 0.0f;
    float m_local = -INFINITY;
    float l_local = 0.0f;

    for (int j = 0; j < num_kv_blocks; j++) {
        int kv_base = j * BLOCK;

        // --- STEP 1: score = Q_row · K_col (tiled over d_k) ---
        float score = 0.0f;
        for (int t = 0; t < d_k_tiles; t++) {
            int dk_col = t * BLOCK + tx;

            // Each warp loads its own Q row
            sQ[ty][tx] = (q_row < N && dk_col < d_k)
                ? Q[q_row * d_model + head_off + dk_col] : 0.0f;

            // All warps cooperate to load the K block (ty used as row index)
            // Only the first BLOCK rows of the block participate; we use
            // ty * (BLOCK/WARPS_PER_BLOCK) striding so all BLOCK K-rows are covered.
            #pragma unroll
            for (int r = ty; r < BLOCK; r += WARPS_PER_BLOCK) {
                int kv_row = kv_base + r;
                sK[r][tx] = (kv_row < N && dk_col < d_k)
                    ? K[kv_row * d_model + head_off + dk_col] : 0.0f;
            }

            __syncthreads();

            // score += Q[q_row][dk_col_range] · K[kv_base+tx][dk_col_range]
            #pragma unroll
            for (int k = 0; k < BLOCK; k++)
                score += sQ[ty][k] * sK[tx][k];

            __syncthreads();
        }

        // Mask out-of-bounds KV positions
        int kv_col = kv_base + tx;
        score = (kv_col < N) ? score * scale : -INFINITY;

        // --- STEP 2: Online softmax ---
        float m_ij  = warp_reduce_max(score);
        float m_new = fmaxf(m_local, m_ij);
        float alpha = expf(m_local - m_new);
        float p     = expf(score - m_new);
        float l_ij  = warp_reduce_sum(p);

        #pragma unroll
        for (int t = 0; t < MAX_DK_TILES; t++) O_local[t] *= alpha;
        l_local = alpha * l_local + l_ij;
        m_local = m_new;

        sS[ty][tx] = p;
        __syncthreads();

        // --- STEP 3: O += P @ V_block (tiled over d_k) ---
        for (int t = 0; t < d_k_tiles; t++) {
            int dk_col = t * BLOCK + tx;

            // Load V tile cooperatively (same strided pattern as K)
            #pragma unroll
            for (int r = ty; r < BLOCK; r += WARPS_PER_BLOCK) {
                int kv_row = kv_base + r;
                sK[r][tx] = (kv_row < N && dk_col < d_k)
                    ? V[kv_row * d_model + head_off + dk_col] : 0.0f;
            }

            __syncthreads();

            float acc = 0.0f;
            #pragma unroll
            for (int jj = 0; jj < BLOCK; jj++)
                acc += sS[ty][jj] * sK[jj][tx];
            O_local[t] += acc;

            __syncthreads();
        }
    }

    // --- FINAL: normalize and write ---
    if (q_row < N) {
        float inv_l = 1.0f / l_local;
        for (int t = 0; t < d_k_tiles; t++) {
            int dk_col = t * BLOCK + tx;
            if (dk_col < d_k)
                output[q_row * d_model + head_off + dk_col] = O_local[t] * inv_l;
        }
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int N, int d_model, int h) {
    int d_k = d_model / h;
    float scale = 1.0f / sqrtf((float)d_k);

    dim3 block(BLOCK, WARPS_PER_BLOCK);                        // 128 threads/block
    dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, h);

    flash_attention_kernel<<<grid, block>>>(Q, K, V, output, N, d_k, d_model, scale);

    cudaDeviceSynchronize();
}