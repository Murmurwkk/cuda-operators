#include <cuda_runtime.h>

namespace {

constexpr int WARP_THREADS = 32;
constexpr int SMALL_STATE_THRESHOLD = 64;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

static int g_shared_mem_per_block = -1;
static int g_shared_mem_limit = -1;

__device__ __forceinline__ int udy_index(int b, int t, int d, int seq_len,
                                         int d_model) {
  return (b * seq_len + t) * d_model + d;
}

__device__ __forceinline__ int bc_index(int b, int t, int n, int seq_len,
                                        int d_state) {
  return (b * seq_len + t) * d_state + n;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
  #pragma unroll
  for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_sum_masked(float value,
                                                        unsigned mask) {
  #pragma unroll
  for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_reduce_sum(float value,
                                                  float *warp_sums) {
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x >> 5;

  value = warp_reduce_sum(value);
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  value = (threadIdx.x < WARPS_PER_BLOCK) ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
    value = warp_reduce_sum(value);
  }
  return value;
}

__device__ __forceinline__ float update_state4(float4 &h, const float4 a,
                                               const float4 b, const float4 c,
                                               float delta, float delta_u) {
  h.x = fmaf(delta_u, b.x, expf(delta * a.x) * h.x);
  h.y = fmaf(delta_u, b.y, expf(delta * a.y) * h.y);
  h.z = fmaf(delta_u, b.z, expf(delta * a.z) * h.z);
  h.w = fmaf(delta_u, b.w, expf(delta * a.w) * h.w);
  return c.x * h.x + c.y * h.y + c.z * h.z + c.w * h.w;
}

__global__ __launch_bounds__(WARP_THREADS)
void selective_scan_warp_shared_kernel(const float *__restrict__ u,
                                       const float *__restrict__ delta,
                                       const float *__restrict__ A,
                                       const float *__restrict__ B,
                                       const float *__restrict__ C,
                                       const float *__restrict__ skip,
                                       float *__restrict__ y,
                                       int batch,
                                       int seq_len,
                                       int d_model,
                                       int d_state) {
  const int bd = blockIdx.x;
  const int b = bd / d_model;
  const int d = bd - b * d_model;
  if (b >= batch) {
    return;   
  }

  extern __shared__ float h[];
  const int lane = threadIdx.x;

  for (int n = lane; n < d_state; n += WARP_THREADS) {
    h[n] = 0.0f;
  }
  // Initialization and float4 ownership differ across lanes.
  __syncwarp();

  const float *a_row = A + d * d_state;
  const float skip_d = skip[d];
  const bool use_vec4 = ((d_state & 3) == 0);

  for (int t = 0; t < seq_len; ++t) {
    const int udy = udy_index(b, t, d, seq_len, d_model);
    const float u_val = u[udy];
    const float delta_val = delta[udy];
    const float delta_u = delta_val * u_val;
    float local_sum = 0.0f;

    const float *b_row = B + bc_index(b, t, 0, seq_len, d_state);
    const float *c_row = C + bc_index(b, t, 0, seq_len, d_state);

    if (use_vec4) {
      float4 *h4 = reinterpret_cast<float4 *>(h);
      const float4 *a4 = reinterpret_cast<const float4 *>(a_row);
      const float4 *b4 = reinterpret_cast<const float4 *>(b_row);
      const float4 *c4 = reinterpret_cast<const float4 *>(c_row);
      const int vec_count = d_state >> 2;

      for (int i = lane; i < vec_count; i += WARP_THREADS) {
        float4 hv = h4[i];
        local_sum += update_state4(hv, a4[i], b4[i], c4[i], delta_val, delta_u);
        h4[i] = hv;
      }
    } else {
      for (int n = lane; n < d_state; n += WARP_THREADS) {
        const float prev_h = h[n];
        const float a_bar = expf(delta_val * a_row[n]);
        const float new_h = fmaf(delta_u, b_row[n], a_bar * prev_h);
        h[n] = new_h;
        local_sum += c_row[n] * new_h;
      }
    }

    const float sum = warp_reduce_sum(local_sum);
    if (lane == 0) {
      y[udy] = sum + skip_d * u_val;
    }
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK)
void selective_scan_shared_kernel(const float *__restrict__ u,
                                  const float *__restrict__ delta,
                                  const float *__restrict__ A,
                                  const float *__restrict__ B,
                                  const float *__restrict__ C,
                                  const float *__restrict__ skip,
                                  float *__restrict__ y,
                                  int batch,
                                  int seq_len,
                                  int d_model,
                                  int d_state) {
  const int bd = blockIdx.x;
  const int b = bd / d_model;
  const int d = bd - b * d_model;
  if (b >= batch) {
    return;
  }

  extern __shared__ float shared[];
  float *h = shared;
  float *warp_sums = h + d_state;

  for (int n = threadIdx.x; n < d_state; n += blockDim.x) {
    h[n] = 0.0f;
  }
  __syncthreads();

  const float *a_row = A + d * d_state;
  const float skip_d = skip[d];
  const bool use_vec4 = ((d_state & 3) == 0);

  for (int t = 0; t < seq_len; ++t) {
    const int udy = udy_index(b, t, d, seq_len, d_model);
    const float u_val = u[udy];
    const float delta_val = delta[udy];
    const float delta_u = delta_val * u_val;
    float local_sum = 0.0f;

    const float *b_row = B + bc_index(b, t, 0, seq_len, d_state);
    const float *c_row = C + bc_index(b, t, 0, seq_len, d_state);

    if (use_vec4) {
      float4 *h4 = reinterpret_cast<float4 *>(h);
      const float4 *a4 = reinterpret_cast<const float4 *>(a_row);
      const float4 *b4 = reinterpret_cast<const float4 *>(b_row);
      const float4 *c4 = reinterpret_cast<const float4 *>(c_row);
      const int vec_count = d_state >> 2;

      for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
        float4 hv = h4[i];
        local_sum += update_state4(hv, a4[i], b4[i], c4[i], delta_val, delta_u);
        h4[i] = hv;
      }
    } else {
      for (int n = threadIdx.x; n < d_state; n += blockDim.x) {
        const float prev_h = h[n];
        const float a_bar = expf(delta_val * a_row[n]);
        const float new_h = fmaf(delta_u, b_row[n], a_bar * prev_h);
        h[n] = new_h;
        local_sum += c_row[n] * new_h;
      }
    }

    const float sum = block_reduce_sum(local_sum, warp_sums);
    if (threadIdx.x == 0) {
      y[udy] = sum + skip_d * u_val;
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(WARP_THREADS)
void selective_scan_warp_global_kernel(const float *__restrict__ u,
                                       const float *__restrict__ delta,
                                       const float *__restrict__ A,
                                       const float *__restrict__ B,
                                       const float *__restrict__ C,
                                       const float *__restrict__ skip,
                                       float *__restrict__ y,
                                       float *__restrict__ state,
                                       int batch,
                                       int seq_len,
                                       int d_model,
                                       int d_state) {
  const int bd = blockIdx.x;
  const int b = bd / d_model;
  const int d = bd - b * d_model;
  if (b >= batch) {
    return;
  }

  const int lane = threadIdx.x;
  float *h = state + static_cast<size_t>(bd) * d_state;

  for (int n = lane; n < d_state; n += WARP_THREADS) {
    h[n] = 0.0f;
  }
  // Initialization and float4 ownership differ across lanes.
  __syncwarp();

  const float *a_row = A + d * d_state;
  const float skip_d = skip[d];
  const bool use_vec4 = ((d_state & 3) == 0);

  for (int t = 0; t < seq_len; ++t) {
    const int udy = udy_index(b, t, d, seq_len, d_model);
    const float u_val = u[udy];
    const float delta_val = delta[udy];
    const float delta_u = delta_val * u_val;
    float local_sum = 0.0f;

    const float *b_row = B + bc_index(b, t, 0, seq_len, d_state);
    const float *c_row = C + bc_index(b, t, 0, seq_len, d_state);

    if (use_vec4) {
      float4 *h4 = reinterpret_cast<float4 *>(h);
      const float4 *a4 = reinterpret_cast<const float4 *>(a_row);
      const float4 *b4 = reinterpret_cast<const float4 *>(b_row);
      const float4 *c4 = reinterpret_cast<const float4 *>(c_row);
      const int vec_count = d_state >> 2;

      for (int i = lane; i < vec_count; i += WARP_THREADS) {
        float4 hv = h4[i];
        local_sum += update_state4(hv, a4[i], b4[i], c4[i], delta_val, delta_u);
        h4[i] = hv;
      }
    } else {
      for (int n = lane; n < d_state; n += WARP_THREADS) {
        const float prev_h = h[n];
        const float a_bar = expf(delta_val * a_row[n]);
        const float new_h = fmaf(delta_u, b_row[n], a_bar * prev_h);
        h[n] = new_h;
        local_sum += c_row[n] * new_h;
      }
    }

    const float sum = warp_reduce_sum(local_sum);
    if (lane == 0) {
      y[udy] = sum + skip_d * u_val;
    }
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK)
void selective_scan_global_kernel(const float *__restrict__ u,
                                  const float *__restrict__ delta,
                                  const float *__restrict__ A,
                                  const float *__restrict__ B,
                                  const float *__restrict__ C,
                                  const float *__restrict__ skip,
                                  float *__restrict__ y,
                                  float *__restrict__ state,
                                  int batch,
                                  int seq_len,
                                  int d_model,
                                  int d_state) {
  const int bd = blockIdx.x;
  const int b = bd / d_model;
  const int d = bd - b * d_model;
  if (b >= batch) {
    return;
  }

  __shared__ float warp_sums[WARPS_PER_BLOCK];
  float *h = state + static_cast<size_t>(bd) * d_state;

  for (int n = threadIdx.x; n < d_state; n += blockDim.x) {
    h[n] = 0.0f;
  }
  __syncthreads();

  const float *a_row = A + d * d_state;
  const float skip_d = skip[d];
  const bool use_vec4 = ((d_state & 3) == 0);

  for (int t = 0; t < seq_len; ++t) {
    const int udy = udy_index(b, t, d, seq_len, d_model);
    const float u_val = u[udy];
    const float delta_val = delta[udy];
    const float delta_u = delta_val * u_val;
    float local_sum = 0.0f;

    const float *b_row = B + bc_index(b, t, 0, seq_len, d_state);
    const float *c_row = C + bc_index(b, t, 0, seq_len, d_state);

    if (use_vec4) {
      float4 *h4 = reinterpret_cast<float4 *>(h);
      const float4 *a4 = reinterpret_cast<const float4 *>(a_row);
      const float4 *b4 = reinterpret_cast<const float4 *>(b_row);
      const float4 *c4 = reinterpret_cast<const float4 *>(c_row);
      const int vec_count = d_state >> 2;

      for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
        float4 hv = h4[i];
        local_sum += update_state4(hv, a4[i], b4[i], c4[i], delta_val, delta_u);
        h4[i] = hv;
      }
    } else {
      for (int n = threadIdx.x; n < d_state; n += blockDim.x) {
        const float prev_h = h[n];
        const float a_bar = expf(delta_val * a_row[n]);
        const float new_h = fmaf(delta_u, b_row[n], a_bar * prev_h);
        h[n] = new_h;
        local_sum += c_row[n] * new_h;
      }
    }

    const float sum = block_reduce_sum(local_sum, warp_sums);
    if (threadIdx.x == 0) {
      y[udy] = sum + skip_d * u_val;
    }
    __syncthreads();
  }
}

}  // namespace

extern "C" void solve(const float *u,
                      const float *delta,
                      const float *A,
                      const float *B,
                      const float *C,
                      const float *skip,
                      float *y,
                      int batch,
                      int seq_len,
                      int d_model,
                      int d_state) {
  if (u == nullptr || delta == nullptr || A == nullptr || B == nullptr ||
      C == nullptr || skip == nullptr || y == nullptr) {
    return;
  }
  if (batch <= 0 || seq_len <= 0 || d_model <= 0 || d_state <= 0) {
    return;
  }

  const int blocks = batch * d_model;
  dim3 grid(blocks);
  const bool use_warp_kernel = (d_state <= SMALL_STATE_THRESHOLD);
  dim3 block(use_warp_kernel ? WARP_THREADS : THREADS_PER_BLOCK);

  if (g_shared_mem_per_block < 0) {
    int device = 0;
    cudaGetDevice(&device);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    g_shared_mem_per_block = prop.sharedMemPerBlock;
    g_shared_mem_limit = prop.sharedMemPerBlock;

    if (prop.sharedMemPerBlockOptin > prop.sharedMemPerBlock) {
      const cudaError_t attr_status = cudaFuncSetAttribute(
          selective_scan_shared_kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          prop.sharedMemPerBlockOptin);
      if (attr_status == cudaSuccess) {
        g_shared_mem_limit = prop.sharedMemPerBlockOptin;
      }
    }
  }

  const size_t shared_bytes =
      use_warp_kernel ? static_cast<size_t>(d_state) * sizeof(float)
                      : static_cast<size_t>(d_state + WARPS_PER_BLOCK) *
                            sizeof(float);
  const bool use_shared_kernel =
      shared_bytes <= static_cast<size_t>(g_shared_mem_limit);

  if (use_shared_kernel) {
    if (use_warp_kernel) {
      selective_scan_warp_shared_kernel<<<grid, block, shared_bytes>>>(
          u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state);
    } else {
      selective_scan_shared_kernel<<<grid, block, shared_bytes>>>(
          u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state);
    }
  } else {
    float *state = nullptr;
    const size_t state_bytes =
        static_cast<size_t>(batch) * d_model * d_state * sizeof(float);
    cudaError_t alloc_status =
        cudaMallocAsync(reinterpret_cast<void **>(&state), state_bytes, 0);
    const bool used_async_alloc = (alloc_status == cudaSuccess);
    if (!used_async_alloc) {
      alloc_status = cudaMalloc(reinterpret_cast<void **>(&state), state_bytes);
    }
    if (alloc_status != cudaSuccess) {
      return;
    }
    if (use_warp_kernel) {
      selective_scan_warp_global_kernel<<<grid, block>>>(
          u, delta, A, B, C, skip, y, state, batch, seq_len, d_model, d_state);
    } else {
      selective_scan_global_kernel<<<grid, block>>>(
          u, delta, A, B, C, skip, y, state, batch, seq_len, d_model, d_state);
    }
    if (used_async_alloc) {
      cudaFreeAsync(state, 0);
    } else {
      cudaFree(state);
    }
  }

  //cudaDeviceSynchronize();
}
