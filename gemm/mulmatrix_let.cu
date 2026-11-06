#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda/pipeline>  

// -------------------------
// Vector helpers (safe path)
// -------------------------
__device__ __forceinline__ bool is_aligned_16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

__device__ __forceinline__ float4 FLOAT4(const float* p) {
  if (is_aligned_16(p)) {
    return *reinterpret_cast<const float4*>(p);
  } else {
    return make_float4(p[0], p[1], p[2], p[3]);
  }
}

__device__ __forceinline__ void STORE_FLOAT4(float* p, const float4& v) {
  if (is_aligned_16(p)) {
    *reinterpret_cast<float4*>(p) = v;
  } else {
    p[0] = v.x;
    p[1] = v.y;
    p[2] = v.z;
    p[3] = v.w;
  }
}

static constexpr int N_Thread = 256;

// ======================================================
// 1) INTERIOR kernel: full tiles only, NO bounds checks
//    Assumes:
//      - launched only on full tiles (row_a+BM<=M and col_b+BN<=N)
//      - K % BK == 0
//      - K % 4 == 0 and N % 4 == 0 (vector safety)
// ======================================================
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(N_Thread)
matrix_multiplication_kernel_interior(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int M, int K, int N) {
  (void)M; // M/N bounds are guaranteed by launch grid (full tiles)

  const int tid = threadIdx.x;

  const int row_a = blockIdx.y * BM;
  const int col_b = blockIdx.x * BN;

  const int tx = (tid / (BN / TN)) * TM;
  const int ty = (tid % (BN / TN)) * TN;

  constexpr int exa = 2;
  constexpr int ext = 0;
  __shared__ float sa[2][BK * (BM + exa)];
  __shared__ float sb[2][BK * (BN + ext)];

  float ra[TM] = {0.0f};
  float rb[TN] = {0.0f};
  float rc[TM * TN] = {0.0f};

  // vectorized load mapping
  const uint32_t ar = tid / (BK / 4);
  const uint32_t ac = tid % (BK / 4);
  const uint32_t br = tid / (BN / 4);
  const uint32_t bc = tid % (BN / 4);
  constexpr int step_a = N_Thread / (BK / 4);
  constexpr int step_b = N_Thread / (BN / 4);

  // base pointers for this C tile
  A += row_a * K;
  B += col_b;
  C += row_a * N + col_b;
  
  int f = 0;
  // K must be multiple of BK for interior fast path
  for (int ptr = 0; ptr < K; ptr += BK) {
    f = !f;
    // Load A tile into shared (no checks)
    for (int offset = 0; offset < BM; offset += step_a) {
      const float4 va =
          *reinterpret_cast<const float4*>(&A[(offset + ar) * K + ac * 4]);
      sa[f][(ac * 4 + 0) * (BM + exa) + (ar + offset)] = va.x;
      sa[f][(ac * 4 + 1) * (BM + exa) + (ar + offset)] = va.y;
      sa[f][(ac * 4 + 2) * (BM + exa) + (ar + offset)] = va.z;
      sa[f][(ac * 4 + 3) * (BM + exa) + (ar + offset)] = va.w;
    }

    // Load B tile into shared (no checks)
    for (int offset = 0; offset < BK; offset += step_b) {
      *reinterpret_cast<float4*>(&sb[f][(br + offset) * (BN + ext) + bc * 4]) =
          *reinterpret_cast<const float4*>(
              &B[(offset + br) * N + bc * 4]);
    }

    __syncthreads();

    // advance to next K-slice
    A += BK;
    B += BK * N;

#pragma unroll
    for (uint32_t i = 0; i < (uint32_t)BK; i++) {
#pragma unroll
      for (uint32_t j = 0; j < (uint32_t)TM; j++) {
        ra[j] = sa[f][i * (BM + exa) + (j + tx)];
      }
#pragma unroll
      for (uint32_t j = 0; j < (uint32_t)TN; j++) {
        rb[j] = sb[f][i * (BN + ext) + (j + ty)];
      }

#pragma unroll
      for (uint32_t x = 0; x < (uint32_t)TM; ++x) {
        const float a = ra[x];
#pragma unroll
        for (uint32_t y = 0; y < (uint32_t)TN; ++y) {
          rc[x * TN + y] = fmaf(a, rb[y], rc[x * TN + y]);
        }
      }
    }
  }

  // Store C tile (no checks), TN=8 => 2 float4 stores per row
#pragma unroll
  for (int x = 0; x < TM; x++) {
    float4 v0, v1;
    v0.x = rc[x * TN + 0];
    v0.y = rc[x * TN + 1];
    v0.z = rc[x * TN + 2];
    v0.w = rc[x * TN + 3];

    v1.x = rc[x * TN + 4];
    v1.y = rc[x * TN + 5];
    v1.z = rc[x * TN + 6];
    v1.w = rc[x * TN + 7];

    *reinterpret_cast<float4*>(&C[(tx + x) * N + (ty + 0)]) = v0;
    *reinterpret_cast<float4*>(&C[(tx + x) * N + (ty + 4)]) = v1;
  }
}

// ======================================================
// 2) BOUNDARY kernel: only edge tiles
//    - Handles partial tiles in M/N and K tail via checks
//    - Early-return skips interior tiles if interior pass ran
// ======================================================
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(N_Thread)
matrix_multiplication_kernel_boundary(const float* A,
                                      const float* B,
                                      float* C,
                                      int M, int K, int N,
                                      int grid_x_full, int grid_y_full) {
  // Skip interior blocks (computed by the fast kernel already)
  if (blockIdx.x < (uint32_t)grid_x_full && blockIdx.y < (uint32_t)grid_y_full)
    return;

  const int tid = threadIdx.x;
  const int row_a = blockIdx.y * BM;
  const int col_b = blockIdx.x * BN;
  const int tx = (tid / (BN / TN)) * TM;
  const int ty = (tid % (BN / TN)) * TN;

  constexpr int exa = 2;
  constexpr int ext = 0;
  __shared__ float sa[BK * (BM + exa)];
  __shared__ float sb[BK * (BN + ext)];

  float ra[TM] = {0.0f};
  float rb[TN] = {0.0f};
  float rc[TM * TN] = {0.0f};

  // vectorize load mapping
  const uint32_t ar = tid / (BK / 4);
  const uint32_t ac = tid % (BK / 4);
  const uint32_t br = tid / (BN / 4);
  const uint32_t bc = tid % (BN / 4);
  constexpr int step_a = N_Thread / (BK / 4);
  constexpr int step_b = N_Thread / (BN / 4);

  A += row_a * K;
  B += col_b;
  C += row_a * N + col_b;

  for (int ptr = 0; ptr < K; ptr += BK) {
    // Load A -> sa (with bounds)
    for (int offset = 0; offset < BM; offset += step_a) {
      if (row_a + offset + (int)ar < M && ptr + (int)ac * 4 + 3 < K) {
        float4 va = FLOAT4(&A[(offset + (int)ar) * K + (int)ac * 4]);
        sa[(ac * 4 + 0) * (BM + exa) + (ar + offset)] = va.x;
        sa[(ac * 4 + 1) * (BM + exa) + (ar + offset)] = va.y;
        sa[(ac * 4 + 2) * (BM + exa) + (ar + offset)] = va.z;
        sa[(ac * 4 + 3) * (BM + exa) + (ar + offset)] = va.w;
      } else {
        // scalar tail / out-of-bounds
#pragma unroll
        for (int i = 0; i < 4; i++) {
          if (row_a + offset + (int)ar < M && ptr + (int)ac * 4 + i < K) {
            sa[(ac * 4 + i) * (BM + exa) + (ar + offset)] =
                A[(offset + (int)ar) * K + (int)ac * 4 + i];
          } else {
            sa[(ac * 4 + i) * (BM + exa) + (ar + offset)] = 0.0f;
          }
        }
      }
    }

    // Load B -> sb (with bounds)
    for (int offset = 0; offset < BK; offset += step_b) {
      if (offset + (int)br + ptr < K && bc * 4 + col_b + 3 < N) {
        float4 vb = FLOAT4(&B[(offset + (int)br) * N + (int)bc * 4]);
        STORE_FLOAT4(&sb[(br + offset) * (BN + ext) + bc * 4], vb);
      } else {
#pragma unroll
        for (int i = 0; i < 4; i++) {
          if (offset + (int)br + ptr < K && (int)bc * 4 + col_b + i < N) {
            sb[(br + offset) * (BN + ext) + bc * 4 + i] =
                B[(offset + (int)br) * N + (int)bc * 4 + i];
          } else {
            sb[(br + offset) * (BN + ext) + bc * 4 + i] = 0.0f;
          }
        }
      }
    }

    __syncthreads();

    // advance to next K-slice
    A += BK;
    B += BK * N;

#pragma unroll
    for (uint32_t i = 0; i < (uint32_t)BK; i++) {
#pragma unroll
      for (uint32_t j = 0; j < (uint32_t)TM; j++) {
        ra[j] = sa[i * (BM + exa) + (j + tx)];
      }
#pragma unroll
      for (uint32_t j = 0; j < (uint32_t)TN; j++) {
        rb[j] = sb[i * (BN + ext) + (j + ty)];
      }

#pragma unroll
      for (uint32_t x = 0; x < (uint32_t)TM; ++x) {
        const float a = ra[x];
#pragma unroll
        for (uint32_t y = 0; y < (uint32_t)TN; ++y) {
          rc[x * TN + y] = fmaf(a, rb[y], rc[x * TN + y]);
        }
      }
    }

    __syncthreads();
  }

  // Store C (with bounds)
  for (int x = 0; x < TM; x++) {
    if (row_a + tx + x < M) {
      for (int y = 0; y < TN; y += 4) {
        if (col_b + ty + y + 3 < N) {
          float4 tmp;
          tmp.x = rc[x * TN + y + 0];
          tmp.y = rc[x * TN + y + 1];
          tmp.z = rc[x * TN + y + 2];
          tmp.w = rc[x * TN + y + 3];
          STORE_FLOAT4(&C[(tx + x) * N + (ty + y)], tmp);
        } else {
#pragma unroll
          for (int i = 0; i < 4; i++) {
            if (col_b + ty + y + i < N) {
              C[(tx + x) * N + (ty + y + i)] = rc[x * TN + y + i];
            }
          }
        }
      }
    }
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
// Asynchronous entry point for composition and measurement. The legacy solve
// below retains its original (M, K, N) ABI and synchronous completion.
extern "C" void mulmatrix_let_launch(const float* A, const float* B, float* C,
                                     int M, int K, int N, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K < 0 || C == nullptr ||
      (K > 0 && (A == nullptr || B == nullptr))) return;
      // Your tuned params
  constexpr int BM = 128;
  constexpr int BK = 16;
  constexpr int BN = 128;
  constexpr int TM = 8;
  constexpr int TN = 8;

  // 256 threads
  dim3 threadsPerBlock((BM * BN) / (TM * TN)); // 16384/64 = 256

  const int grid_x = (N + BN - 1) / BN;
  const int grid_y = (M + BM - 1) / BM;
  const int grid_x_full = N / BN;
  const int grid_y_full = M / BM;

  // Use interior fast path only when:
  // - there exists at least one full tile in both dims
  // - vector conditions are clean
  // - K is divisible by BK (no K-tail for the interior kernel)
  const bool can_use_interior_fast =
      (grid_x_full > 0) && (grid_y_full > 0) &&
      (N % 4 == 0) && (K % 4 == 0) &&
      (K % BK == 0) &&
      ((reinterpret_cast<uintptr_t>(A) & 15) == 0) &&
      ((reinterpret_cast<uintptr_t>(B) & 15) == 0) &&
      ((reinterpret_cast<uintptr_t>(C) & 15) == 0);

  if (can_use_interior_fast) {
    dim3 gridInterior(grid_x_full, grid_y_full);
    matrix_multiplication_kernel_interior<BM, BN, BK, TM, TN>
        <<<gridInterior, threadsPerBlock, 0, stream>>>(A, B, C, M, K, N);
  }

  // Boundary pass:
  // - If we didn't run interior, boundary covers everything (pass full=0 so no early return).
  // - If we did run interior, boundary skips interior blocks and only computes edge tiles.
  const bool need_boundary =
      (!can_use_interior_fast) || (grid_x_full != grid_x) || (grid_y_full != grid_y);

  if (need_boundary) {
    dim3 gridAll(grid_x, grid_y);
    const int fx = can_use_interior_fast ? grid_x_full : 0;
    const int fy = can_use_interior_fast ? grid_y_full : 0;
    matrix_multiplication_kernel_boundary<BM, BN, BK, TM, TN>
        <<<gridAll, threadsPerBlock, 0, stream>>>(A, B, C, M, K, N, fx, fy);
  }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int K, int N) {
  mulmatrix_let_launch(A, B, C, M, K, N, nullptr);
  cudaDeviceSynchronize();
}
