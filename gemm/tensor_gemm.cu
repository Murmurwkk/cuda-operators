#include <cuda_runtime.h>
#include <mma.h>

namespace {

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

constexpr int A_TILE_ELEMS = WMMA_M * WMMA_K;
constexpr int B_TILE_ELEMS = WMMA_K * WMMA_N;
constexpr int C_TILE_ELEMS = WMMA_M * WMMA_N;
constexpr int SMEM_ELEMS_PER_WARP = A_TILE_ELEMS + B_TILE_ELEMS + C_TILE_ELEMS;

template <int WarpsPerBlock>
__global__ __launch_bounds__(WarpsPerBlock * 32) void tensor_gemm_tf32_kernel(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    int M,
    int K,
    int N) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane_id = tid & 31;

  const int tiles_n = (N + WMMA_N - 1) / WMMA_N;
  const int tile_id = blockIdx.x * WarpsPerBlock + warp_id;
  const int total_tiles = ((M + WMMA_M - 1) / WMMA_M) * tiles_n;
  if (tile_id >= total_tiles) {
    return;
  }

  const int tile_m = tile_id / tiles_n;
  const int tile_n = tile_id - tile_m * tiles_n;
  const int row_base = tile_m * WMMA_M;
  const int col_base = tile_n * WMMA_N;

  // WMMA load/store requires a 256-bit aligned base; every warp slice and
  // A/B/C offset below is a multiple of 32 bytes.
  __shared__ __align__(32) float shared[WarpsPerBlock * SMEM_ELEMS_PER_WARP];
  float *warp_shared = shared + warp_id * SMEM_ELEMS_PER_WARP;
  float *shared_a = warp_shared;
  float *shared_b = shared_a + A_TILE_ELEMS;
  float *shared_c = shared_b + B_TILE_ELEMS;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32,
                 wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32,
                 wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  wmma::fill_fragment(c_frag, 0.0f);

  for (int k_base = 0; k_base < K; k_base += WMMA_K) {
    for (int idx = lane_id; idx < A_TILE_ELEMS; idx += 32) {
      const int local_m = idx / WMMA_K;
      const int local_k = idx - local_m * WMMA_K;
      const int global_m = row_base + local_m;
      const int global_k = k_base + local_k;

      float value = 0.0f;
      if (global_m < M && global_k < K) {
        value = wmma::__float_to_tf32(A[global_m * K + global_k]);
      }
      shared_a[idx] = value;
    }

    for (int idx = lane_id; idx < B_TILE_ELEMS; idx += 32) {
      const int local_k = idx / WMMA_N;
      const int local_n = idx - local_k * WMMA_N;
      const int global_k = k_base + local_k;
      const int global_n = col_base + local_n;

      float value = 0.0f;
      if (global_k < K && global_n < N) {
        value = wmma::__float_to_tf32(B[global_k * N + global_n]);
      }
      shared_b[idx] = value;
    }

    __syncwarp();

    wmma::load_matrix_sync(a_frag, shared_a, WMMA_K);
    wmma::load_matrix_sync(b_frag, shared_b, WMMA_N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    __syncwarp();
  }

  wmma::store_matrix_sync(shared_c, c_frag, WMMA_N, wmma::mem_row_major);
  __syncwarp();

  for (int idx = lane_id; idx < C_TILE_ELEMS; idx += 32) {
    const int local_m = idx / WMMA_N;
    const int local_n = idx - local_m * WMMA_N;
    const int global_m = row_base + local_m;
    const int global_n = col_base + local_n;

    if (global_m < M && global_n < N) {
      C[global_m * N + global_n] = shared_c[idx];
    }
  }
}

}  // namespace

extern "C" void tensor_gemm_launch(const float *A, const float *B, float *C, int M,
                                   int K, int N, cudaStream_t stream) {
  if (A == nullptr || B == nullptr || C == nullptr || M <= 0 || K < 0 ||
      N <= 0) {
    return;
  }

  constexpr int warps_per_block = 8;
  const int tiles_m = (M + WMMA_M - 1) / WMMA_M;
  const int tiles_n = (N + WMMA_N - 1) / WMMA_N;
  const int total_tiles = tiles_m * tiles_n;

  dim3 block(warps_per_block * 32);
  dim3 grid((total_tiles + warps_per_block - 1) / warps_per_block);
  tensor_gemm_tf32_kernel<warps_per_block><<<grid, block, 0, stream>>>(A, B, C, M, K, N);
}

extern "C" void tensor_gemm(const float *A, const float *B, float *C, int M,
                            int K, int N) {
  tensor_gemm_launch(A, B, C, M, K, N, nullptr);
  cudaDeviceSynchronize();
}

// Keep the same entry name used by the existing standalone GEMM submissions.
extern "C" void solve(const float *A, const float *B, float *C, int M, int K,
                      int N) {
  tensor_gemm(A, B, C, M, K, N);
}
