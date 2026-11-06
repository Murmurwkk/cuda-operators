#include <cuda_runtime.h>
#include <mma.h>

#include "ptx_helpers.cuh"

namespace {

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

constexpr int BLOCK_TILE_M = 64;
constexpr int BLOCK_TILE_N = 64;
constexpr int BLOCK_TILE_K = 32;

constexpr int WARP_TILE_M = 32;
constexpr int WARP_TILE_N = 32;

constexpr int WARPS_M = BLOCK_TILE_M / WARP_TILE_M;
constexpr int WARPS_N = BLOCK_TILE_N / WARP_TILE_N;
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
constexpr int PIPE_STAGES = 2;

constexpr int WARP_TILES_M = WARP_TILE_M / WMMA_M;
constexpr int WARP_TILES_N = WARP_TILE_N / WMMA_N;

constexpr int A_STAGE_FLOATS = BLOCK_TILE_M * BLOCK_TILE_K;
constexpr int B_STAGE_FLOATS = BLOCK_TILE_K * BLOCK_TILE_N;
constexpr int C_STAGE_FLOATS = BLOCK_TILE_M * BLOCK_TILE_N;

constexpr int A_STAGE_CHUNKS = A_STAGE_FLOATS / 4;
constexpr int B_STAGE_CHUNKS = B_STAGE_FLOATS / 4;
constexpr int C_STAGE_CHUNKS = C_STAGE_FLOATS / 4;

constexpr std::size_t OPT_SHARED_BYTES =
    static_cast<std::size_t>(PIPE_STAGES * A_STAGE_FLOATS +
                             PIPE_STAGES * B_STAGE_FLOATS + C_STAGE_FLOATS) *
    sizeof(float);

static_assert(BLOCK_TILE_K % WMMA_K == 0, "BLOCK_TILE_K must be a multiple of WMMA_K.");
static_assert(BLOCK_TILE_M % WARP_TILE_M == 0, "BLOCK_TILE_M must be divisible by WARP_TILE_M.");
static_assert(BLOCK_TILE_N % WARP_TILE_N == 0, "BLOCK_TILE_N must be divisible by WARP_TILE_N.");
static_assert(BLOCK_TILE_K % 4 == 0, "BLOCK_TILE_K must keep 16-byte vector lanes aligned.");
static_assert(BLOCK_TILE_N % 4 == 0, "BLOCK_TILE_N must keep 16-byte vector lanes aligned.");

__device__ __forceinline__ void issue_stage_copy(float *shared_a_stage,
                                                 float *shared_b_stage,
                                                 const float *A,
                                                 const float *B,
                                                 int block_row,
                                                 int block_col,
                                                 int k_base,
                                                 int M,
                                                 int K,
                                                 int N) {
  const int tid = threadIdx.x;
  constexpr int a_chunks_per_row = BLOCK_TILE_K / 4;
  constexpr int b_chunks_per_row = BLOCK_TILE_N / 4;

  for (int chunk = tid; chunk < A_STAGE_CHUNKS; chunk += THREADS_PER_BLOCK) {
    const int row = chunk / a_chunks_per_row;
    const int col = (chunk % a_chunks_per_row) * 4;
    const int global_row = block_row + row;
    const int global_col = k_base + col;
    float *dst = shared_a_stage + row * BLOCK_TILE_K + col;

    if (global_row < M) {
      const int remaining = K - global_col;
      const int valid_floats = remaining > 4 ? 4 : (remaining > 0 ? remaining : 0);
      if (valid_floats > 0) {
        ptx_helpers::cp_async_or_scalar(
            dst, A + global_row * K + global_col, valid_floats * 4);
      } else {
        ptx_helpers::zero4(dst);
      }
    } else {
      ptx_helpers::zero4(dst);
    }
  }

  for (int chunk = tid; chunk < B_STAGE_CHUNKS; chunk += THREADS_PER_BLOCK) {
    const int row = chunk / b_chunks_per_row;
    const int col = (chunk % b_chunks_per_row) * 4;
    const int global_row = k_base + row;
    const int global_col = block_col + col;
    float *dst = shared_b_stage + row * BLOCK_TILE_N + col;

    if (global_row < K) {
      const int remaining = N - global_col;
      const int valid_floats = remaining > 4 ? 4 : (remaining > 0 ? remaining : 0);
      if (valid_floats > 0) {
        ptx_helpers::cp_async_or_scalar(
            dst, B + global_row * N + global_col, valid_floats * 4);
      } else {
        ptx_helpers::zero4(dst);
      }
    } else {
      ptx_helpers::zero4(dst);
    }
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK) void tensor_gemm_tf32_optimized_kernel(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    int M,
    int K,
    int N) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const int block_row = blockIdx.y * BLOCK_TILE_M;
  const int block_col = blockIdx.x * BLOCK_TILE_N;
  if (block_row >= M || block_col >= N) {
    return;
  }

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;

  const int warp_tile_m = (warp_id / WARPS_N) * WARP_TILE_M;
  const int warp_tile_n = (warp_id % WARPS_N) * WARP_TILE_N;

  extern __shared__ __align__(32) float shared_storage[];
  float *shared_a = shared_storage;
  float *shared_b = shared_a + PIPE_STAGES * A_STAGE_FLOATS;
  float *shared_c = shared_b + PIPE_STAGES * B_STAGE_FLOATS;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      c_frag[WARP_TILES_M][WARP_TILES_N];

  #pragma unroll
  for (int i = 0; i < WARP_TILES_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILES_N; ++j) {
      wmma::fill_fragment(c_frag[i][j], 0.0f);
    }
  }

  const int k_tiles = (K + BLOCK_TILE_K - 1) / BLOCK_TILE_K;
  issue_stage_copy(shared_a, shared_b, A, B, block_row, block_col, 0, M, K, N);
  ptx_helpers::cp_async_commit();

  for (int k_tile_idx = 0; k_tile_idx < k_tiles; ++k_tile_idx) {
    const int read_stage = k_tile_idx & 1;
    const int write_stage = read_stage ^ 1;

    if (k_tile_idx + 1 < k_tiles) {
      const int next_k_base = (k_tile_idx + 1) * BLOCK_TILE_K;
      issue_stage_copy(shared_a + write_stage * A_STAGE_FLOATS,
                       shared_b + write_stage * B_STAGE_FLOATS,
                       A,
                       B,
                       block_row,
                       block_col,
                       next_k_base,
                       M,
                       K,
                       N);
      ptx_helpers::cp_async_commit();
      ptx_helpers::cp_async_wait<1>();
    } else {
      ptx_helpers::cp_async_wait<0>();
    }

    __syncthreads();

    float *read_a = shared_a + read_stage * A_STAGE_FLOATS;
    float *read_b = shared_b + read_stage * B_STAGE_FLOATS;

    for (int k_step = 0; k_step < BLOCK_TILE_K; k_step += WMMA_K) {
      wmma::fragment<wmma::matrix_a,
                     WMMA_M,
                     WMMA_N,
                     WMMA_K,
                     wmma::precision::tf32,
                     wmma::row_major>
          a_frag[WARP_TILES_M];
      wmma::fragment<wmma::matrix_b,
                     WMMA_M,
                     WMMA_N,
                     WMMA_K,
                     wmma::precision::tf32,
                     wmma::row_major>
          b_frag[WARP_TILES_N];

      #pragma unroll
      for (int i = 0; i < WARP_TILES_M; ++i) {
        float *a_ptr =
            read_a + (warp_tile_m + i * WMMA_M) * BLOCK_TILE_K + k_step;
        wmma::load_matrix_sync(a_frag[i], a_ptr, BLOCK_TILE_K);
        ptx_helpers::convert_fragment_to_tf32(a_frag[i]);
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILES_N; ++j) {
        float *b_ptr =
            read_b + k_step * BLOCK_TILE_N + warp_tile_n + j * WMMA_N;
        wmma::load_matrix_sync(b_frag[j], b_ptr, BLOCK_TILE_N);
        ptx_helpers::convert_fragment_to_tf32(b_frag[j]);
      }

      #pragma unroll
      for (int i = 0; i < WARP_TILES_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; ++j) {
          wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        }
      }
    }

    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < WARP_TILES_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILES_N; ++j) {
      float *dst =
          shared_c + (warp_tile_m + i * WMMA_M) * BLOCK_TILE_N + warp_tile_n +
          j * WMMA_N;
      wmma::store_matrix_sync(dst, c_frag[i][j], BLOCK_TILE_N, wmma::mem_row_major);
    }
  }

  __syncthreads();

  constexpr int c_chunks_per_row = BLOCK_TILE_N / 4;
  for (int chunk = tid; chunk < C_STAGE_CHUNKS; chunk += THREADS_PER_BLOCK) {
    const int row = chunk / c_chunks_per_row;
    const int col = (chunk % c_chunks_per_row) * 4;
    const int global_row = block_row + row;
    const int global_col = block_col + col;

    if (global_row >= M) {
      continue;
    }

    const float *src = shared_c + row * BLOCK_TILE_N + col;
    const int remaining = N - global_col;
    const int valid_floats = remaining > 4 ? 4 : (remaining > 0 ? remaining : 0);
    if (valid_floats == 0) {
      continue;
    }
    float4 value = make_float4(src[0], src[1], src[2], src[3]);

    if (valid_floats == 4 &&
        ptx_helpers::is_aligned_16(C + global_row * N + global_col)) {
      reinterpret_cast<float4 *>(C + global_row * N + global_col)[0] = value;
    } else {
      float *dst = C + global_row * N + global_col;
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (i < valid_floats) {
          dst[i] = src[i];
        }
      }
    }
  }
#else
  (void)A;
  (void)B;
  (void)C;
  (void)M;
  (void)K;
  (void)N;
#endif
}

__global__ void fallback_gemm_kernel(const float *__restrict__ A,
                                     const float *__restrict__ B,
                                     float *__restrict__ C,
                                     int M,
                                     int K,
                                     int N) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) {
    return;
  }

  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    sum += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = sum;
}

}  // namespace

extern "C" void tensor_gemm_launch(const float *A, const float *B, float *C, int M,
                                   int K, int N, cudaStream_t stream) {
  if (A == nullptr || B == nullptr || C == nullptr || M <= 0 || K < 0 ||
      N <= 0) {
    return;
  }

  // No stage may remain in flight when K=0 skips the consumer loop.
  if (K == 0) {
    cudaMemsetAsync(C, 0, static_cast<std::size_t>(M) * N * sizeof(float), stream);
    return;
  }

  int device = 0;
  cudaGetDevice(&device);

  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device);

  if (prop.major >= 8) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N,
              (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
    tensor_gemm_tf32_optimized_kernel<<<grid, block, OPT_SHARED_BYTES, stream>>>(
        A, B, C, M, K, N);
  } else {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    fallback_gemm_kernel<<<grid, block, 0, stream>>>(A, B, C, M, K, N);
  }
}

extern "C" void tensor_gemm(const float *A, const float *B, float *C, int M,
                            int K, int N) {
  tensor_gemm_launch(A, B, C, M, K, N, nullptr);
  cudaDeviceSynchronize();
}

extern "C" void solve(const float *A, const float *B, float *C, int M, int K,
                      int N) {
  tensor_gemm(A, B, C, M, K, N);
}
