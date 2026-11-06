#pragma once

#include <cuda_runtime.h>
#include <cub/block/block_load.cuh>
#include <cstdint>

namespace {

constexpr int kBlockThreads = 256;
constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 32;
constexpr int kThreadTileM = 4;
constexpr int kThreadTileN = 4;

static_assert(kTileM * kTileN / (kThreadTileM * kThreadTileN) ==
                  kBlockThreads,
              "tile shape must map exactly to one block");
static_assert((kTileM * kTileK) % kBlockThreads == 0,
              "A tile must divide evenly across block threads");
static_assert((kTileK * kTileN) % kBlockThreads == 0,
              "B tile must divide evenly across block threads");

__device__ __forceinline__ int valid_count(int total, int offset, int tile) {
  const int left = total - offset;
  if (left <= 0) {
    return 0;
  }
  return left < tile ? left : tile;
}

template <int TileCols>
struct PitchedTileLoadIterator {
  const float *__restrict__ base;
  int stride;
  int valid_rows;
  int valid_cols;

  __device__ __forceinline__ float operator[](int index) const {
    const int row = index / TileCols;
    const int col = index - row * TileCols;
    if (row < valid_rows && col < valid_cols) {
      return base[row * stride + col];
    }
    return 0.0f;
  }
};

// Loader 0: manual blocked scalar; 1: manual blocked float4; 2: CUB DIRECT.
// All three produce the same blocked item ownership and shared layout.
template <int Items, int TileCols, int Loader>
__device__ __forceinline__ void load_items(PitchedTileLoadIterator<TileCols> it,
                                          float (&items)[Items]) {
  const int first = threadIdx.x * Items;
#pragma unroll
  for (int i = 0; i < Items; i += 4) {
    const int index = first + i;
    const int row = index / TileCols;
    const int col = index % TileCols;
    if constexpr (Loader == 1) {
      if (row < it.valid_rows && col + 3 < it.valid_cols &&
          ((reinterpret_cast<uintptr_t>(it.base + row * it.stride + col) & 15) == 0)) {
        const float4 v = *reinterpret_cast<const float4*>(it.base + row * it.stride + col);
        items[i] = v.x; items[i+1] = v.y; items[i+2] = v.z; items[i+3] = v.w;
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) items[i+j] = it[index+j];
      }
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) items[i+j] = it[index+j];
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN, int Loader = 2>
__global__ __launch_bounds__(kBlockThreads) void gemm_cub_kernel(
    const float *__restrict__ A, const float *__restrict__ B,
    float *__restrict__ C, int M, int K, int N) {
  using BlockLoadA =
      cub::BlockLoad<float, kBlockThreads, (BM * BK) / kBlockThreads,
                     cub::BLOCK_LOAD_DIRECT>;
  using BlockLoadB =
      cub::BlockLoad<float, kBlockThreads, (BK * BN) / kBlockThreads,
                     cub::BLOCK_LOAD_DIRECT>;

  constexpr int kItemsA = (BM * BK) / kBlockThreads;
  constexpr int kItemsB = (BK * BN) / kBlockThreads;
  constexpr int kRowThreads = BN / TN;

  __shared__ float shared_a[BM * BK];
  __shared__ float shared_b[BK * BN];

  const int tid = threadIdx.x;
  const int tile_row = blockIdx.y * BM;
  const int tile_col = blockIdx.x * BN;
  const int local_m = (tid / kRowThreads) * TM;
  const int local_n = (tid - (tid / kRowThreads) * kRowThreads) * TN;

  float accum[TM][TN] = {};

  for (int k_base = 0; k_base < K; k_base += BK) {
    float a_items[kItemsA];
    float b_items[kItemsB];

    const PitchedTileLoadIterator<BK> a_it{
        A + tile_row * K + k_base,
        K,
        valid_count(M, tile_row, BM),
        valid_count(K, k_base, BK)};
    const PitchedTileLoadIterator<BN> b_it{
        B + k_base * N + tile_col,
        N,
        valid_count(K, k_base, BK),
        valid_count(N, tile_col, BN)};

    if constexpr (Loader == 2) {
      BlockLoadA().Load(a_it, a_items);
      BlockLoadB().Load(b_it, b_items);
    } else {
      load_items<kItemsA, BK, Loader>(a_it, a_items);
      load_items<kItemsB, BN, Loader>(b_it, b_items);
    }

#pragma unroll
    for (int i = 0; i < kItemsA; ++i) {
      const int index = tid * kItemsA + i;
      shared_a[index] = a_items[i];
    }
#pragma unroll
    for (int i = 0; i < kItemsB; ++i) {
      const int index = tid * kItemsB + i;
      shared_b[index] = b_items[i];
    }

    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float a_frag[TM];
      float b_frag[TN];

#pragma unroll
      for (int i = 0; i < TM; ++i) {
        a_frag[i] = shared_a[(local_m + i) * BK + kk];
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        b_frag[j] = shared_b[kk * BN + local_n + j];
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const float a_val = a_frag[i];
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          accum[i][j] = fmaf(a_val, b_frag[j], accum[i][j]);
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = tile_row + local_m + i;
    if (row < M) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int col = tile_col + local_n + j;
        if (col < N) {
          C[row * N + col] = accum[i][j];
        }
      }
    }
  }
}

} // namespace

// stream-aware entry point; loader is an experiment control, not a runtime
// autotuner. Existing gemm_cub/solve retain their synchronous (M,K,N) ABI.
extern "C" void gemm_cub_launch(const float *A, const float *B, float *C, int M,
                                int K, int N, int loader, cudaStream_t stream) {
  if (A == nullptr || B == nullptr || C == nullptr || M <= 0 || K < 0 ||
      N <= 0) {
    return;
  }

  dim3 block(kBlockThreads);
  dim3 grid((N + kTileN - 1) / kTileN, (M + kTileM - 1) / kTileM);
  if (loader == 0) {
    gemm_cub_kernel<kTileM, kTileN, kTileK, kThreadTileM, kThreadTileN, 0>
        <<<grid, block, 0, stream>>>(A, B, C, M, K, N);
  } else if (loader == 1) {
    gemm_cub_kernel<kTileM, kTileN, kTileK, kThreadTileM, kThreadTileN, 1>
        <<<grid, block, 0, stream>>>(A, B, C, M, K, N);
  } else {
    gemm_cub_kernel<kTileM, kTileN, kTileK, kThreadTileM, kThreadTileN, 2>
        <<<grid, block, 0, stream>>>(A, B, C, M, K, N);
  }
}

extern "C" void gemm_cub(const float *A, const float *B, float *C, int M,
                         int K, int N) {
  gemm_cub_launch(A, B, C, M, K, N, 2, nullptr);
  cudaDeviceSynchronize();
}

extern "C" void solve(const float *A, const float *B, float *C, int M, int K,
                      int N) {
  gemm_cub(A, B, C, M, K, N);
}
