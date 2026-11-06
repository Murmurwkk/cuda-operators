#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

namespace {

constexpr int WARP_SIZE = 32;
constexpr int CHUNK_THREADS = 256;
constexpr int MAX_SCAN_THREADS = 1024;
constexpr int MAX_CHUNKS = 4096;
constexpr int CHUNK_PREFIX_GROUP_SIZE = MAX_SCAN_THREADS;
constexpr int MAX_CHUNK_PREFIX_GROUPS = (MAX_CHUNKS + CHUNK_PREFIX_GROUP_SIZE - 1) / CHUNK_PREFIX_GROUP_SIZE;
constexpr int STATE_TILE = 32;
constexpr int REDUCE_THREADS = 256;
constexpr int REDUCE_WARPS = REDUCE_THREADS / WARP_SIZE;

__device__ __forceinline__ int udy_index(int b, int t, int d, int seq_len,
                                         int d_model) {
  return (b * seq_len + t) * d_model + d;
}

__device__ __forceinline__ int bc_index(int b, int t, int n, int seq_len,
                                        int d_state) {
  return (b * seq_len + t) * d_state + n;
}

__device__ __forceinline__ void compose_affine(float right_a, float right_b,
                                               float left_a, float left_b,
                                               float &out_a, float &out_b) {
  // right(left(h)) = right_a * (left_a * h + left_b) + right_b
  out_a = right_a * left_a;
  out_b = fmaf(right_a, left_b, right_b);
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_reduce_sum(float value,
                                                  float *warp_sums) {
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x >> 5;
  value = warp_reduce_sum(value);
  if (lane == 0) warp_sums[warp] = value;
  __syncthreads();
  value = (threadIdx.x < REDUCE_WARPS) ? warp_sums[lane] : 0.0f;
  if (warp == 0) value = warp_reduce_sum(value);
  return value;
}

int next_power_of_two(int value) {
  int power = 1;
  while (power < value) power <<= 1;
  return power;
}

int scan_threads_for(int count) {
  int threads = next_power_of_two(count);
  if (threads < WARP_SIZE) threads = WARP_SIZE;
  if (threads > MAX_SCAN_THREADS) threads = MAX_SCAN_THREADS;
  return threads;
}

size_t align_up(size_t x, size_t alignment) {
  return (x + alignment - 1) & ~(alignment - 1);
}

struct WorkspaceLayout {
  float *chunk_a;
  float *chunk_b;
  float *group_a;
  float *group_b;
  float *partial_y;
  size_t bytes;
};

WorkspaceLayout make_layout(void *workspace,
                            size_t state_count,
                            int num_chunks,
                            size_t output_count,
                            int num_state_tiles,
                            int num_chunk_prefix_groups) {
  uintptr_t base = reinterpret_cast<uintptr_t>(workspace);
  uintptr_t p = base;
  const size_t alignment = 256;
  const size_t chunk_count = state_count * static_cast<size_t>(num_chunks);
  const size_t group_count =
      state_count * static_cast<size_t>(num_chunk_prefix_groups);

  p = align_up(p, alignment);
  float *chunk_a = reinterpret_cast<float *>(p);
  p += chunk_count * sizeof(float);

  p = align_up(p, alignment);
  float *chunk_b = reinterpret_cast<float *>(p);
  p += chunk_count * sizeof(float);

  p = align_up(p, alignment);
  float *group_a = reinterpret_cast<float *>(p);
  p += group_count * sizeof(float);

  p = align_up(p, alignment);
  float *group_b = reinterpret_cast<float *>(p);
  p += group_count * sizeof(float);

  float *partial_y = nullptr;
  if (num_state_tiles > 1) {
    p = align_up(p, alignment);
    partial_y = reinterpret_cast<float *>(p);
    p += output_count * static_cast<size_t>(num_state_tiles) * sizeof(float);
  }

  p = align_up(p, alignment);
  WorkspaceLayout layout;
  layout.chunk_a = chunk_a;
  layout.chunk_b = chunk_b;
  layout.group_a = group_a;
  layout.group_b = group_b;
  layout.partial_y = partial_y;
  layout.bytes = p - base;
  return layout;
}

__global__ void build_chunk_summary_kernel(const float *__restrict__ u,
                                           const float *__restrict__ delta,
                                           const float *__restrict__ A,
                                           const float *__restrict__ B,
                                           float *__restrict__ chunk_a,
                                           float *__restrict__ chunk_b,
                                           int batch,
                                           int seq_len,
                                           int d_model,
                                           int d_state,
                                           int num_chunks) {
  const int state_idx = blockIdx.x;
  const int chunk = blockIdx.y;
  const int n = state_idx % d_state;
  const int bd = state_idx / d_state;
  const int b = bd / d_model;
  const int d = bd - b * d_model;
  const int local_t = threadIdx.x;
  const int t = chunk * CHUNK_THREADS + local_t;

  extern __shared__ float shared[];
  float *scan_a = shared;
  float *scan_b = scan_a + blockDim.x;

  float cur_a = 1.0f;
  float cur_b = 0.0f;
  if (b < batch && t < seq_len) {
    const int udy = udy_index(b, t, d, seq_len, d_model);
    const float delta_val = delta[udy];
    cur_a = expf(delta_val * A[d * d_state + n]);
    cur_b = delta_val * u[udy] * B[bc_index(b, t, n, seq_len, d_state)];
  }

  scan_a[local_t] = cur_a;
  scan_b[local_t] = cur_b;
  __syncthreads();

  for (int offset = 1; offset < CHUNK_THREADS; offset <<= 1) {
    float left_a = 1.0f;
    float left_b = 0.0f;
    if (local_t >= offset) {
      left_a = scan_a[local_t - offset];
      left_b = scan_b[local_t - offset];
    }
    __syncthreads();
    if (local_t >= offset) {
      float next_a, next_b;
      compose_affine(cur_a, cur_b, left_a, left_b, next_a, next_b);
      cur_a = next_a;
      cur_b = next_b;
    }
    __syncthreads();
    scan_a[local_t] = cur_a;
    scan_b[local_t] = cur_b;
    __syncthreads();
  }

  if (local_t == CHUNK_THREADS - 1) {
    const size_t idx = static_cast<size_t>(state_idx) * num_chunks + chunk;
    chunk_a[idx] = cur_a;
    chunk_b[idx] = cur_b;
  }
}

__global__ void scan_chunk_prefix_groups_kernel(float *__restrict__ chunk_a,
                                                float *__restrict__ chunk_b,
                                                float *__restrict__ group_a,
                                                float *__restrict__ group_b,
                                                int num_chunks,
                                                int num_groups) {
  const int state_idx = blockIdx.x;
  const int group = blockIdx.y;
  const int lane = threadIdx.x;
  const int chunk = group * CHUNK_PREFIX_GROUP_SIZE + lane;

  extern __shared__ float shared[];
  float *scan_a = shared;
  float *scan_b = scan_a + blockDim.x;

  float cur_a = 1.0f;
  float cur_b = 0.0f;
  if (chunk < num_chunks) {
    const size_t idx = static_cast<size_t>(state_idx) * num_chunks + chunk;
    cur_a = chunk_a[idx];
    cur_b = chunk_b[idx];
  }

  scan_a[lane] = cur_a;
  scan_b[lane] = cur_b;
  __syncthreads();

  for (int offset = 1; offset < blockDim.x; offset <<= 1) {
    float left_a = 1.0f;
    float left_b = 0.0f;
    if (lane >= offset) {
      left_a = scan_a[lane - offset];
      left_b = scan_b[lane - offset];
    }
    __syncthreads();
    if (lane >= offset) {
      float next_a, next_b;
      compose_affine(cur_a, cur_b, left_a, left_b, next_a, next_b);
      cur_a = next_a;
      cur_b = next_b;
    }
    __syncthreads();
    scan_a[lane] = cur_a;
    scan_b[lane] = cur_b;
    __syncthreads();
  }

  if (chunk < num_chunks) {
    float prefix_a = 1.0f;
    float prefix_b = 0.0f;
    if (lane > 0) {
      prefix_a = scan_a[lane - 1];
      prefix_b = scan_b[lane - 1];
    }
    const size_t idx = static_cast<size_t>(state_idx) * num_chunks + chunk;
    chunk_a[idx] = prefix_a;
    chunk_b[idx] = prefix_b;
  }

  if (lane == blockDim.x - 1) {
    const size_t group_idx = static_cast<size_t>(state_idx) * num_groups + group;
    group_a[group_idx] = scan_a[lane];
    group_b[group_idx] = scan_b[lane];
  }
}

__global__ void scan_chunk_prefix_group_prefix_kernel(float *__restrict__ group_a,
                                                      float *__restrict__ group_b,
                                                      int num_groups) {
  const int state_idx = blockIdx.x;
  const int group = threadIdx.x;

  extern __shared__ float shared[];
  float *scan_a = shared;
  float *scan_b = scan_a + blockDim.x;

  float cur_a = 1.0f;
  float cur_b = 0.0f;
  if (group < num_groups) {
    const size_t idx = static_cast<size_t>(state_idx) * num_groups + group;
    cur_a = group_a[idx];
    cur_b = group_b[idx];
  }

  scan_a[group] = cur_a;
  scan_b[group] = cur_b;
  __syncthreads();

  for (int offset = 1; offset < blockDim.x; offset <<= 1) {
    float left_a = 1.0f;
    float left_b = 0.0f;
    if (group >= offset) {
      left_a = scan_a[group - offset];
      left_b = scan_b[group - offset];
    }
    __syncthreads();
    if (group >= offset) {
      float next_a, next_b;
      compose_affine(cur_a, cur_b, left_a, left_b, next_a, next_b);
      cur_a = next_a;
      cur_b = next_b;
    }
    __syncthreads();
    scan_a[group] = cur_a;
    scan_b[group] = cur_b;
    __syncthreads();
  }

  if (group < num_groups) {
    float prefix_a = 1.0f;
    float prefix_b = 0.0f;
    if (group > 0) {
      prefix_a = scan_a[group - 1];
      prefix_b = scan_b[group - 1];
    }
    const size_t idx = static_cast<size_t>(state_idx) * num_groups + group;
    group_a[idx] = prefix_a;
    group_b[idx] = prefix_b;
  }
}

__global__ void apply_chunk_prefix_group_prefix_kernel(float *__restrict__ chunk_a,
                                                       float *__restrict__ chunk_b,
                                                       const float *__restrict__ group_a,
                                                       const float *__restrict__ group_b,
                                                       int num_chunks,
                                                       int num_groups) {
  const int state_idx = blockIdx.x;
  const int chunk = blockIdx.y * blockDim.x + threadIdx.x;
  if (chunk >= num_chunks) return;

  const int group = chunk / CHUNK_PREFIX_GROUP_SIZE;
  const size_t chunk_idx = static_cast<size_t>(state_idx) * num_chunks + chunk;
  const size_t group_idx = static_cast<size_t>(state_idx) * num_groups + group;

  const float local_a = chunk_a[chunk_idx];
  const float local_b = chunk_b[chunk_idx];
  const float base_a = group_a[group_idx];
  const float base_b = group_b[group_idx];

  float out_a, out_b;
  compose_affine(local_a, local_b, base_a, base_b, out_a, out_b);
  chunk_a[chunk_idx] = out_a;
  chunk_b[chunk_idx] = out_b;
}

__global__ void recompute_project_tile_kernel(const float *__restrict__ u,
                                              const float *__restrict__ delta,
                                              const float *__restrict__ A,
                                              const float *__restrict__ B,
                                              const float *__restrict__ C,
                                              const float *__restrict__ skip,
                                              const float *__restrict__ prefix_b,
                                              float *__restrict__ partial_y,
                                              float *__restrict__ y,
                                              int batch,
                                              int seq_len,
                                              int d_model,
                                              int d_state,
                                              int num_chunks,
                                              int num_state_tiles) {
  const int bd = blockIdx.x;
  const int chunk = blockIdx.y;
  const int tile = blockIdx.z;
  const int b = bd / d_model;
  const int d = bd - b * d_model;
  const int local_t = threadIdx.x;
  const int t = chunk * CHUNK_THREADS + local_t;

  extern __shared__ float shared[];
  float *scan_a = shared;
  float *scan_b = scan_a + blockDim.x;

  float acc = 0.0f;
  const int n_begin = tile * STATE_TILE;
  const int n_end = min(n_begin + STATE_TILE, d_state);

  for (int n = n_begin; n < n_end; ++n) {
    float cur_a = 1.0f;
    float cur_b = 0.0f;
    if (b < batch && t < seq_len) {
      const int udy = udy_index(b, t, d, seq_len, d_model);
      const float delta_val = delta[udy];
      cur_a = expf(delta_val * A[d * d_state + n]);
      cur_b = delta_val * u[udy] * B[bc_index(b, t, n, seq_len, d_state)];
    }

    scan_a[local_t] = cur_a;
    scan_b[local_t] = cur_b;
    __syncthreads();

    for (int offset = 1; offset < CHUNK_THREADS; offset <<= 1) {
      float left_a = 1.0f;
      float left_b = 0.0f;
      if (local_t >= offset) {
        left_a = scan_a[local_t - offset];
        left_b = scan_b[local_t - offset];
      }
      __syncthreads();
      if (local_t >= offset) {
        float next_a, next_b;
        compose_affine(cur_a, cur_b, left_a, left_b, next_a, next_b);
        cur_a = next_a;
        cur_b = next_b;
      }
      __syncthreads();
      scan_a[local_t] = cur_a;
      scan_b[local_t] = cur_b;
      __syncthreads();
    }

    if (b < batch && t < seq_len) {
      const size_t state_idx =
          (static_cast<size_t>(bd) * d_state) + static_cast<size_t>(n);
      const float h0 = prefix_b[state_idx * num_chunks + chunk];
      const float h = fmaf(cur_a, h0, cur_b);
      acc += C[bc_index(b, t, n, seq_len, d_state)] * h;
    }
    __syncthreads();
  }

  if (b < batch && t < seq_len) {
    const int out = udy_index(b, t, d, seq_len, d_model);
    if (num_state_tiles == 1) {
      y[out] = acc + skip[d] * u[out];
    } else {
      const size_t partial_idx =
          (static_cast<size_t>(tile) * batch * seq_len * d_model) + out;
      partial_y[partial_idx] = acc;
    }
  }
}

__global__ __launch_bounds__(REDUCE_THREADS)
void reduce_tiles_to_output_kernel(const float *__restrict__ partial_y,
                                   const float *__restrict__ u,
                                   const float *__restrict__ skip,
                                   float *__restrict__ y,
                                   int output_count,
                                   int seq_len,
                                   int d_model,
                                   int num_state_tiles) {
  const int out = blockIdx.x;
  if (out >= output_count) return;
  const int d = out % d_model;

  __shared__ float warp_sums[REDUCE_WARPS];
  float local = 0.0f;
  for (int tile = threadIdx.x; tile < num_state_tiles; tile += blockDim.x) {
    const size_t idx = static_cast<size_t>(tile) * output_count + out;
    local += partial_y[idx];
  }
  const float sum = block_reduce_sum(local, warp_sums);
  if (threadIdx.x == 0) {
    y[out] = sum + skip[d] * u[out];
  }
}

__global__ __launch_bounds__(REDUCE_THREADS)
void selective_scan_output_parallel_fallback_kernel(const float *__restrict__ u,
                                                    const float *__restrict__ delta,
                                                    const float *__restrict__ A,
                                                    const float *__restrict__ B,
                                                    const float *__restrict__ C,
                                                    const float *__restrict__ skip,
                                                    float *__restrict__ y,
                                                    int total,
                                                    int seq_len,
                                                    int d_model,
                                                    int d_state) {
  const int out_idx = blockIdx.x;
  if (out_idx >= total) return;

  const int d = out_idx % d_model;
  const int t = (out_idx / d_model) % seq_len;
  const int b = out_idx / (seq_len * d_model);

  __shared__ float warp_sums[REDUCE_WARPS];
  float local_sum = 0.0f;
  for (int n = threadIdx.x; n < d_state; n += blockDim.x) {
    float h = 0.0f;
    const float a_val = A[d * d_state + n];
    for (int s = 0; s <= t; ++s) {
      const int udy = udy_index(b, s, d, seq_len, d_model);
      const float delta_val = delta[udy];
      const float a_bar = expf(delta_val * a_val);
      const float b_val = B[bc_index(b, s, n, seq_len, d_state)];
      h = fmaf(delta_val * u[udy], b_val, a_bar * h);
    }
    local_sum += C[bc_index(b, t, n, seq_len, d_state)] * h;
  }

  const float sum = block_reduce_sum(local_sum, warp_sums);
  if (threadIdx.x == 0) y[out_idx] = sum + skip[d] * u[out_idx];
}

void launch_fallback(const float *u,
                     const float *delta,
                     const float *A,
                     const float *B,
                     const float *C,
                     const float *skip,
                     float *y,
                     int batch,
                     int seq_len,
                     int d_model,
                     int d_state,
                     cudaStream_t stream) {
  const int total = batch * seq_len * d_model;
  selective_scan_output_parallel_fallback_kernel<<<total, REDUCE_THREADS, 0, stream>>>(
      u, delta, A, B, C, skip, y, total, seq_len, d_model, d_state);
}

}  // namespace

extern "C" size_t selective_scan_streaming_workspace_size(int batch,
                                                           int seq_len,
                                                           int d_model,
                                                           int d_state) {
  if (batch <= 0 || seq_len <= 0 || d_model <= 0 || d_state <= 0) return 0;
  const int num_chunks = (seq_len + CHUNK_THREADS - 1) / CHUNK_THREADS;
  if (num_chunks > MAX_CHUNKS) return 0;
  const size_t state_count = static_cast<size_t>(batch) * d_model * d_state;
  const size_t output_count = static_cast<size_t>(batch) * seq_len * d_model;
  const int num_state_tiles = (d_state + STATE_TILE - 1) / STATE_TILE;
  const int num_chunk_prefix_groups =
      (num_chunks + CHUNK_PREFIX_GROUP_SIZE - 1) / CHUNK_PREFIX_GROUP_SIZE;
  return make_layout(nullptr, state_count, num_chunks, output_count,
                     num_state_tiles, num_chunk_prefix_groups)
      .bytes;
}

extern "C" void solve_with_workspace_stream(const float *u,
                                             const float *delta,
                                             const float *A,
                                             const float *B,
                                             const float *C,
                                             const float *skip,
                                             float *y,
                                             int batch,
                                             int seq_len,
                                             int d_model,
                                             int d_state,
                                             void *workspace,
                                             size_t workspace_bytes,
                                             cudaStream_t stream) {
  if (u == nullptr || delta == nullptr || A == nullptr || B == nullptr ||
      C == nullptr || skip == nullptr || y == nullptr) {
    return;
  }
  if (batch <= 0 || seq_len <= 0 || d_model <= 0 || d_state <= 0) return;

  const int output_count = batch * seq_len * d_model;
  const int num_chunks = (seq_len + CHUNK_THREADS - 1) / CHUNK_THREADS;
  if (num_chunks > MAX_CHUNKS) {
    launch_fallback(u, delta, A, B, C, skip, y, batch, seq_len, d_model,
                    d_state, stream);
    return;
  }

  const size_t state_count = static_cast<size_t>(batch) * d_model * d_state;
  const int num_state_tiles = (d_state + STATE_TILE - 1) / STATE_TILE;
  const int num_chunk_prefix_groups =
      (num_chunks + CHUNK_PREFIX_GROUP_SIZE - 1) / CHUNK_PREFIX_GROUP_SIZE;
  const size_t required = selective_scan_streaming_workspace_size(
      batch, seq_len, d_model, d_state);
  if (workspace == nullptr || workspace_bytes < required) {
    launch_fallback(u, delta, A, B, C, skip, y, batch, seq_len, d_model,
                    d_state, stream);
    return;
  }

  WorkspaceLayout layout = make_layout(workspace, state_count, num_chunks,
                                       output_count, num_state_tiles,
                                       num_chunk_prefix_groups);

  const size_t chunk_shared = 2 * CHUNK_THREADS * sizeof(float);
  dim3 summary_grid(static_cast<unsigned int>(state_count), num_chunks);
  build_chunk_summary_kernel<<<summary_grid, CHUNK_THREADS, chunk_shared, stream>>>(
      u, delta, A, B, layout.chunk_a, layout.chunk_b, batch, seq_len, d_model,
      d_state, num_chunks);

  const int local_prefix_threads = CHUNK_PREFIX_GROUP_SIZE;
  const size_t local_prefix_shared = 2 * local_prefix_threads * sizeof(float);
  dim3 local_prefix_grid(static_cast<unsigned int>(state_count),
                         num_chunk_prefix_groups);
  scan_chunk_prefix_groups_kernel<<<local_prefix_grid,
                                    local_prefix_threads,
                                    local_prefix_shared,
                                    stream>>>(layout.chunk_a,
                                              layout.chunk_b,
                                              layout.group_a,
                                              layout.group_b,
                                              num_chunks,
                                              num_chunk_prefix_groups);

  const int group_prefix_threads = scan_threads_for(num_chunk_prefix_groups);
  const size_t group_prefix_shared = 2 * group_prefix_threads * sizeof(float);
  scan_chunk_prefix_group_prefix_kernel<<<static_cast<unsigned int>(state_count),
                                          group_prefix_threads,
                                          group_prefix_shared,
                                          stream>>>(layout.group_a,
                                                    layout.group_b,
                                                    num_chunk_prefix_groups);

  const int apply_prefix_threads = 256;
  const int apply_prefix_blocks =
      (num_chunks + apply_prefix_threads - 1) / apply_prefix_threads;
  dim3 apply_prefix_grid(static_cast<unsigned int>(state_count),
                         apply_prefix_blocks);
  apply_chunk_prefix_group_prefix_kernel<<<apply_prefix_grid,
                                           apply_prefix_threads,
                                           0,
                                           stream>>>(layout.chunk_a,
                                                     layout.chunk_b,
                                                     layout.group_a,
                                                     layout.group_b,
                                                     num_chunks,
                                                     num_chunk_prefix_groups);

  dim3 project_grid(static_cast<unsigned int>(batch * d_model), num_chunks,
                    num_state_tiles);
  recompute_project_tile_kernel<<<project_grid, CHUNK_THREADS, chunk_shared, stream>>>(
      u, delta, A, B, C, skip, layout.chunk_b, layout.partial_y, y, batch,
      seq_len, d_model, d_state, num_chunks, num_state_tiles);

  if (num_state_tiles > 1) {
    reduce_tiles_to_output_kernel<<<output_count, REDUCE_THREADS, 0, stream>>>(
        layout.partial_y, u, skip, y, output_count, seq_len, d_model,
        num_state_tiles);
  }
}

extern "C" void solve_stream(const float *u,
                              const float *delta,
                              const float *A,
                              const float *B,
                              const float *C,
                              const float *skip,
                              float *y,
                              int batch,
                              int seq_len,
                              int d_model,
                              int d_state,
                              cudaStream_t stream) {
  const size_t workspace_bytes = selective_scan_streaming_workspace_size(
      batch, seq_len, d_model, d_state);
  if (workspace_bytes == 0) {
    launch_fallback(u, delta, A, B, C, skip, y, batch, seq_len, d_model,
                    d_state, stream);
    return;
  }

  void *workspace = nullptr;
#if CUDART_VERSION >= 11020
  cudaError_t err = cudaMallocAsync(&workspace, workspace_bytes, stream);
  if (err == cudaSuccess) {
    solve_with_workspace_stream(u, delta, A, B, C, skip, y, batch, seq_len,
                                d_model, d_state, workspace, workspace_bytes,
                                stream);
    cudaFreeAsync(workspace, stream);
    return;
  }
#endif
  if (cudaMalloc(&workspace, workspace_bytes) == cudaSuccess) {
    solve_with_workspace_stream(u, delta, A, B, C, skip, y, batch, seq_len,
                                d_model, d_state, workspace, workspace_bytes,
                                stream);
    cudaFree(workspace);
  } else {
    launch_fallback(u, delta, A, B, C, skip, y, batch, seq_len, d_model,
                    d_state, stream);
  }
}

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
  solve_stream(u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state,
               0);
}
