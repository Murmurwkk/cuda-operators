#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// Q: [M, d]
// K: [N, d]
// V: [N, d]
// output: [M, d]
// Computes: output = softmax(Q * K^T / sqrt(d)) * V
extern "C" void solve(const float* Q,
                      const float* K,
                      const float* V,
                      float* output,
                      int M,
                      int N,
                      int d);

namespace {

constexpr int kThreads = 128;
constexpr int kMaxAccumPerThread = 8;  // Supports d <= kThreads * kMaxAccumPerThread.

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

template <int Threads>
__device__ __forceinline__ float block_reduce_sum(float value, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  value = warp_reduce_sum(value);
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();

  if (warp == 0) {
    value = (lane < Threads / 32) ? scratch[lane] : 0.0f;
    value = warp_reduce_sum(value);
    if (lane == 0) {
      scratch[0] = value;
    }
  }
  __syncthreads();
  const float result = scratch[0];
  // Every warp must finish reading before a following reduction reuses scratch.
  __syncthreads();
  return result;
}

template <int Threads>
__device__ __forceinline__ float block_reduce_max(float value, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  value = warp_reduce_max(value);
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();

  if (warp == 0) {
    value = (lane < Threads / 32) ? scratch[lane] : -FLT_MAX;
    value = warp_reduce_max(value);
    if (lane == 0) {
      scratch[0] = value;
    }
  }
  __syncthreads();
  const float result = scratch[0];
  // The previous barrier orders writes, not these reads against the next call.
  __syncthreads();
  return result;
}

template <int TileN, int Threads>
__global__ void fused_attention_kernel(const float* __restrict__ Q,
                                       const float* __restrict__ K,
                                       const float* __restrict__ V,
                                       float* __restrict__ output,
                                       int M,
                                       int N,
                                       int d,
                                       float scale) {
  const int row = blockIdx.x;
  if (row >= M) {
    return;
  }

  extern __shared__ float smem[];
  float* s_q = smem;
  float* s_k = s_q + d;
  float* s_v = s_k + TileN * d;
  float* s_scores = s_v + TileN * d;
  float* s_reduce = s_scores + TileN;

  __shared__ float s_m;
  __shared__ float s_l;

  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 31;
  const int num_warps = Threads / 32;

  float acc[kMaxAccumPerThread];
#pragma unroll
  for (int i = 0; i < kMaxAccumPerThread; i++) {
    acc[i] = 0.0f;
  }

  for (int dim = threadIdx.x; dim < d; dim += Threads) {
    s_q[dim] = Q[static_cast<size_t>(row) * d + dim];
  }

  if (threadIdx.x == 0) {
    s_m = -FLT_MAX;
    s_l = 0.0f;
  }
  __syncthreads();

  for (int tile_start = 0; tile_start < N; tile_start += TileN) {
    const int tile_cols = min(TileN, N - tile_start);

    for (int idx = threadIdx.x; idx < tile_cols * d; idx += Threads) {
      const int local_col = idx / d;
      const int dim = idx - local_col * d;
      const size_t g_index = (static_cast<size_t>(tile_start) + local_col) * d + dim;
      s_k[static_cast<size_t>(local_col) * d + dim] = K[g_index];
      s_v[static_cast<size_t>(local_col) * d + dim] = V[g_index];
    }
    __syncthreads();

    for (int local_col = warp_id; local_col < tile_cols; local_col += num_warps) {
      float dot = 0.0f;
      const size_t base = static_cast<size_t>(local_col) * d;
      for (int dim = lane_id; dim < d; dim += 32) {
        dot = fmaf(s_q[dim], s_k[base + dim], dot);
      }
      dot = warp_reduce_sum(dot);
      if (lane_id == 0) {
        s_scores[local_col] = dot * scale;
      }
    }
    __syncthreads();

    float local_tile_max = -FLT_MAX;
    for (int local_col = threadIdx.x; local_col < tile_cols; local_col += Threads) {
      local_tile_max = fmaxf(local_tile_max, s_scores[local_col]);
    }
    const float tile_max = block_reduce_max<Threads>(local_tile_max, s_reduce);

    const float old_m = s_m;
    const float new_m = fmaxf(old_m, tile_max);
    const float prev_scale = __expf(old_m - new_m);

    float local_tile_sum = 0.0f;
    for (int local_col = threadIdx.x; local_col < tile_cols; local_col += Threads) {
      const float prob = __expf(s_scores[local_col] - new_m);
      s_scores[local_col] = prob;
      local_tile_sum += prob;
    }
    const float tile_sum = block_reduce_sum<Threads>(local_tile_sum, s_reduce);

#pragma unroll
    for (int i = 0; i < kMaxAccumPerThread; i++) {
      const int dim = threadIdx.x + i * Threads;
      if (dim < d) {
        float value = acc[i] * prev_scale;
        for (int local_col = 0; local_col < tile_cols; local_col++) {
          value = fmaf(s_scores[local_col], s_v[static_cast<size_t>(local_col) * d + dim], value);
        }
        acc[i] = value;
      }
    }

    if (threadIdx.x == 0) {
      s_l = s_l * prev_scale + tile_sum;
      s_m = new_m;
    }
    __syncthreads();
  }

  const float inv_l = __frcp_rn(s_l);
#pragma unroll
  for (int i = 0; i < kMaxAccumPerThread; i++) {
    const int dim = threadIdx.x + i * Threads;
    if (dim < d) {
      output[static_cast<size_t>(row) * d + dim] = acc[i] * inv_l;
    }
  }
}

template <int TileN>
cudaError_t launch_fused_attention(const float* Q,
                                   const float* K,
                                   const float* V,
                                   float* output,
                                   int M,
                                   int N,
                                   int d,
                                   float scale,
                                   int max_dynamic_smem,
                                   cudaStream_t stream) {
  const size_t shared_bytes =
      static_cast<size_t>(d + 2 * TileN * d + TileN + kThreads / 32) * sizeof(float);
  if (shared_bytes > static_cast<size_t>(max_dynamic_smem)) {
    return cudaErrorInvalidValue;
  }

  auto kernel = fused_attention_kernel<TileN, kThreads>;
  if (shared_bytes > 48 * 1024) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes));
    if (attr_status != cudaSuccess) {
      return attr_status;
    }
    attr_status = cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    if (attr_status != cudaSuccess) return attr_status;
  }

  kernel<<<M, kThreads, shared_bytes, stream>>>(Q, K, V, output, M, N, d, scale);
  return cudaGetLastError();
}

}  // namespace

// Dense single-head forward only. No mask; QK accumulation/logits must remain finite.
// Outputs are asynchronous on stream; callers must check both launch and completion.
extern "C" cudaError_t attention_online_launch(const float* Q,
                      const float* K,
                      const float* V,
                      float* output,
                      int M,
                      int N,
                      int d, cudaStream_t stream) {
  if (Q == nullptr || K == nullptr || V == nullptr || output == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (M <= 0 || N <= 0 || d <= 0) {
    return cudaErrorInvalidValue;
  }
  if (d > kThreads * kMaxAccumPerThread) {
    return cudaErrorInvalidValue;
  }

  int max_dynamic_smem = 0;
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(&max_dynamic_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (status != cudaSuccess) return status;
  if (max_dynamic_smem <= 0) {
    status = cudaDeviceGetAttribute(&max_dynamic_smem, cudaDevAttrMaxSharedMemoryPerBlock, device);
    if (status != cudaSuccess) return status;
  }

  const float scale = 1.0f / sqrtf(static_cast<float>(d));

  // Select by capacity before launch. A runtime failure is never hidden by retrying a tile.
#define TRY_TILE(T) \
  if ((N >= T || T == 8) && \
      static_cast<size_t>(d + 2 * T * d + T + kThreads / 32) * sizeof(float) <= static_cast<size_t>(max_dynamic_smem)) \
    return launch_fused_attention<T>(Q, K, V, output, M, N, d, scale, max_dynamic_smem, stream)
  TRY_TILE(128);
  TRY_TILE(64);
  TRY_TILE(32);
  TRY_TILE(16);
  TRY_TILE(8);
#undef TRY_TILE
  return cudaErrorInvalidValue;
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int M, int N, int d) {
  // Compatibility only: use attention_online_launch when errors must be observable.
  (void)attention_online_launch(Q, K, V, output, M, N, d, nullptr);
}
