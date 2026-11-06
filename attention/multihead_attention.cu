#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

#define TILE_WIDTH 16
#define TPB 256
#define COARSE_FACTOR 1

__global__ void gemm_kernel_strided(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N,
    int lda,
    int ldb,
    int ldc,
    float scale)
{
    __shared__ float sa[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sb[TILE_WIDTH][TILE_WIDTH];

    int row = TILE_WIDTH * blockIdx.y + threadIdx.y;
    int col = COARSE_FACTOR * TILE_WIDTH * blockIdx.x + threadIdx.x;
    float sum[COARSE_FACTOR];

    #pragma unroll
    for (int i = 0; i < COARSE_FACTOR; ++i) {
        sum[i] = 0.0f;
    }

    for (int phase = 0; phase < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++phase) {
        int a_col = phase * TILE_WIDTH + threadIdx.x;
        sa[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * lda + a_col] : 0.0f;

        #pragma unroll
        for (int c = 0; c < COARSE_FACTOR; ++c) {
            int col_new = col + c * TILE_WIDTH;
            int b_row = phase * TILE_WIDTH + threadIdx.y;

            sb[threadIdx.y][threadIdx.x] =
                (b_row < K && col_new < N) ? B[b_row * ldb + col_new] : 0.0f;

            __syncthreads();

            #pragma unroll
            for (int j = 0; j < TILE_WIDTH; ++j) {
                sum[c] += sa[threadIdx.y][j] * sb[j][threadIdx.x];
            }

            __syncthreads();
        }
    }

    #pragma unroll
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int col_new = col + c * TILE_WIDTH;
        if (row < M && col_new < N) {
            C[row * ldc + col_new] = sum[c] / scale;
        }
    }
}

__global__ void row_softmax_kernel(float* input, int M, int N, int ld)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= M) {
        return;
    }

    __shared__ float s_max[TPB];
    __shared__ float s_sum[TPB];

    float max_val = -FLT_MAX;
    float sum_val = 0.0f;

    for (int j = tid; j < N; j += TPB) {
        float val = input[row * ld + j];
        if (val > max_val) {
            sum_val = sum_val * expf(max_val - val) + 1.0f;
            max_val = val;
        } else {
            sum_val += expf(val - max_val);
        }
    }

    s_max[tid] = max_val;
    s_sum[tid] = sum_val;
    __syncthreads();

    for (int offset = TPB / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float max_temp = s_max[tid];
            float max_next = s_max[tid + offset];
            float sum_temp = s_sum[tid];
            float sum_next = s_sum[tid + offset];

            if (max_next > -FLT_MAX) {
                if (max_next > max_temp) {
                    s_sum[tid] = sum_temp * expf(max_temp - max_next) + sum_next;
                    s_max[tid] = max_next;
                } else {
                    s_sum[tid] = sum_next * expf(max_next - max_temp) + sum_temp;
                }
            }
        }
        __syncthreads();
    }

    float row_max = s_max[0];
    float row_sum = s_sum[0];

    for (int j = tid; j < N; j += TPB) {
        input[row * ld + j] = expf(input[row * ld + j] - row_max) / row_sum;
    }
}

__global__ void transpose_kernel_strided(
    const float* input,
    float* output,
    int rows,
    int cols,
    int input_ld,
    int output_ld)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (c < cols && r < rows) {
        output[c * output_ld + r] = input[r * input_ld + c];
    }
}

// Q/K/V/output: [N, d_model].
// The last dimension is split into h contiguous head slices.
extern "C" void solve(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int N,
    int d_model,
    int h)
{
    if (Q == nullptr || K == nullptr || V == nullptr || output == nullptr) {
        return;
    }

    if (N <= 0 || d_model <= 0 || h <= 0 || (d_model % h) != 0) {
        return;
    }

    int seq_len = N;
    int head_dim = d_model / h;
    float scale = sqrtf(static_cast<float>(head_dim));

    float* d_score = nullptr;
    float* d_KT = nullptr;

    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&d_score),
        static_cast<size_t>(seq_len) * seq_len * sizeof(float));
    if (status != cudaSuccess) {
        return;
    }

    status = cudaMalloc(reinterpret_cast<void**>(&d_KT),
        static_cast<size_t>(head_dim) * N * sizeof(float));
    if (status != cudaSuccess) {
        cudaFree(d_score);
        return;
    }

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_trans((head_dim + TILE_WIDTH - 1) / TILE_WIDTH,
                    (seq_len + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 grid_score((seq_len + TILE_WIDTH * COARSE_FACTOR - 1) / (TILE_WIDTH * COARSE_FACTOR),
                    (seq_len + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 grid_out((head_dim + TILE_WIDTH * COARSE_FACTOR - 1) / (TILE_WIDTH * COARSE_FACTOR),
                  (seq_len + TILE_WIDTH - 1) / TILE_WIDTH);

    for (int head = 0; head < h; ++head) {
        const float* Q_head = Q + head * head_dim;
        const float* K_head = K + head * head_dim;
        const float* V_head = V + head * head_dim;
        float* output_head = output + head * head_dim;

        transpose_kernel_strided<<<grid_trans, block>>>(
            K_head, d_KT, seq_len, head_dim, d_model, seq_len);

        gemm_kernel_strided<<<grid_score, block>>>(
            Q_head, d_KT, d_score, seq_len, head_dim, seq_len, d_model, seq_len, seq_len, scale);

        row_softmax_kernel<<<seq_len, TPB>>>(d_score, seq_len, seq_len, seq_len);

        gemm_kernel_strided<<<grid_out, block>>>(
            d_score, V_head, output_head, seq_len, seq_len, head_dim, seq_len, d_model, d_model, 1.0f);
    }

    cudaDeviceSynchronize();
    cudaFree(d_score);
    cudaFree(d_KT);
}
