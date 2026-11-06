#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

namespace {

constexpr int WARP_SIZE = 32;
constexpr int REDUCE_THREADS = 256;
constexpr int REDUCE_WARPS = REDUCE_THREADS / WARP_SIZE;
constexpr int CHUNK_THREADS = 256;
constexpr int MAX_SCAN_THREADS = 1024;
constexpr int MAX_CHUNKS = 4096;

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
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  value = (threadIdx.x < REDUCE_WARPS) ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
    value = warp_reduce_sum(value);
  }
  return value;
}

int next_power_of_two(int value) {
  int power = 1;
  while (power < value) {
    power <<= 1;
  }
  return power;
}

int scan_threads_for(int count) {
  int threads = next_power_of_two(count);
  if (threads < WARP_SIZE) {
    threads = WARP_SIZE;
  }
  if (threads > MAX_SCAN_THREADS) {
    threads = MAX_SCAN_THREADS;
  }
  return threads;
}

size_t align_up(size_t x, size_t alignment) {
  return (x + alignment - 1) & ~(alignment - 1);
}

struct Workspace {
  float *local_a;
  float *local_b;
  float *chunk_a;
  float *chunk_b;
  float *contrib;
};

bool carve_workspace(void *workspace,
                     size_t workspace_bytes,
                     size_t elem_count,
                     size_t chunk_count,
                     Workspace &ws) {
  uintptr_t base = reinterpret_cast<uintptr_t>(workspace);
  uintptr_t p = (base + 255u) & ~uintptr_t(255u);
  const uintptr_t end = base + workspace_bytes;

  auto take = [&](size_t count) -> float * {
    p = (p + 255u) & ~uintptr_t(255u);
    const size_t bytes = count * sizeof(float);
    if (p + bytes > end) {
      return nullptr;
    }
    float *out = reinterpret_cast<float *>(p);
    p += bytes;
    return out;
  };

  ws.local_a = take(elem_count);
  ws.local_b = take(elem_count);
  ws.chunk_a = take(chunk_count);
  ws.chunk_b = take(chunk_count);
  ws.contrib = take(elem_count);

  return ws.local_a && ws.local_b && ws.chunk_a && ws.chunk_b && ws.contrib;
}

__global__ void scan_state_chunks_kernel(const float *__restrict__ u,
                                         const float *__restrict__ delta,
                                         const float *__restrict__ A,
                                         const float *__restrict__ B,
                                         float *__restrict__ local_a,
                                         float *__restrict__ local_b,
                                         float *__restrict__ chunk_a,
                                         float *__restrict__ chunk_b,
                                         int batch,
                                         int seq_len,
                                         int d_model,
                                         int d_state,
                                         int num_chunks) {
  const int state_idx = blockIdx.x;    // ((b * d_model + d) * d_state + n)
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

  if (b < batch && chunk < num_chunks && t < seq_len) {
    const int udy = udy_index(b, t, d, seq_len, d_model);
    const float delta_val = delta[udy];
    cur_a = expf(delta_val * A[d * d_state + n]);
    cur_b = delta_val * u[udy] * B[bc_index(b, t, n, seq_len, d_state)];
  }

  scan_a[local_t] = cur_a;
  scan_b[local_t] = cur_b;
  __syncthreads();

#pragma unroll
  for (int offset = 1; offset < CHUNK_THREADS; offset <<= 1) {
    float left_a = 1.0f;
    float left_b = 0.0f;
    if (local_t >= offset) {
      left_a = scan_a[local_t - offset];
      left_b = scan_b[local_t - offset];
    }
    __syncthreads();

    if (local_t >= offset) {
      float next_a;
      float next_b;
      compose_affine(cur_a, cur_b, left_a, left_b, next_a, next_b);
      cur_a = next_a;
      cur_b = next_b;
    }
    __syncthreads();

    scan_a[local_t] = cur_a;
    scan_b[local_t] = cur_b;
    __syncthreads();
  }

  if (t < seq_len) {
    const size_t elem_idx = static_cast<size_t>(state_idx) * seq_len + t;
    local_a[elem_idx] = cur_a;
    local_b[elem_idx] = cur_b;
  }

  if (local_t == CHUNK_THREADS - 1) {
    const size_t chunk_idx =
        static_cast<size_t>(state_idx) * num_chunks + chunk;
    chunk_a[chunk_idx] = cur_a;
    chunk_b[chunk_idx] = cur_b;
  }
}

__global__ void scan_chunk_prefix_kernel(float *__restrict__ chunk_a,
                                         float *__restrict__ chunk_b,
                                         int num_chunks) {
  const int state_idx = blockIdx.x;
  const int chunk = threadIdx.x;

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

  scan_a[chunk] = cur_a;
  scan_b[chunk] = cur_b;
  __syncthreads();

  for (int offset = 1; offset < blockDim.x; offset <<= 1) {
    float left_a = 1.0f;
    float left_b = 0.0f;

    if (chunk >= offset) {
      left_a = scan_a[chunk - offset];
      left_b = scan_b[chunk - offset];
    }
    __syncthreads();

    if (chunk >= offset) {
      float next_a;
      float next_b;
      compose_affine(cur_a, cur_b, left_a, left_b, next_a, next_b);
      cur_a = next_a;
      cur_b = next_b;
    }
    __syncthreads();

    scan_a[chunk] = cur_a;
    scan_b[chunk] = cur_b;
    __syncthreads();
  }

  if (chunk < num_chunks) {
    float prefix_a = 1.0f;
    float prefix_b = 0.0f;
    if (chunk > 0) {
      prefix_a = scan_a[chunk - 1];
      prefix_b = scan_b[chunk - 1];
    }

    const size_t idx = static_cast<size_t>(state_idx) * num_chunks + chunk;
    chunk_a[idx] = prefix_a;
    chunk_b[idx] = prefix_b;
  }
}

__global__ void apply_prefix_and_project_kernel(const float *__restrict__ local_a,
                                                const float *__restrict__ local_b,
                                                const float *__restrict__ chunk_a,
                                                const float *__restrict__ chunk_b,
                                                const float *__restrict__ C,
                                                float *__restrict__ contrib,
                                                size_t elem_count,
                                                int seq_len,
                                                int d_model,
                                                int d_state,
                                                int num_chunks) {
  const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (idx >= elem_count) {
    return;
  }

  const int t = static_cast<int>(idx % seq_len);
  const size_t state_idx = idx / seq_len;
  const int n = static_cast<int>(state_idx % d_state);
  const int bd = static_cast<int>(state_idx / d_state);
  const int b = bd / d_model;
  const int d = bd - b * d_model;
  const int chunk = t / CHUNK_THREADS;

  const size_t chunk_idx = state_idx * num_chunks + chunk;
  const float prefix_h0 = chunk_b[chunk_idx];  // h before this chunk, assuming global h_0 = 0.
  const float h = fmaf(local_a[idx], prefix_h0, local_b[idx]);

  contrib[idx] = C[bc_index(b, t, n, seq_len, d_state)] * h;
}

__global__ __launch_bounds__(REDUCE_THREADS)
void reduce_state_to_output_kernel(const float *__restrict__ contrib,
                                   const float *__restrict__ u,
                                   const float *__restrict__ skip,
                                   float *__restrict__ y,
                                   int output_count,
                                   int seq_len,
                                   int d_model,
                                   int d_state) {
  const int out_idx = blockIdx.x;
  if (out_idx >= output_count) {
    return;
  }

  const int d = out_idx % d_model;
  const int t = (out_idx / d_model) % seq_len;
  const int b = out_idx / (seq_len * d_model);
  const int bd = b * d_model + d;

  __shared__ float warp_sums[REDUCE_WARPS];

  float sum = 0.0f;
  for (int n = threadIdx.x; n < d_state; n += blockDim.x) {
    const size_t state_idx = static_cast<size_t>(bd) * d_state + n;
    sum += contrib[state_idx * seq_len + t];
  }

  sum = block_reduce_sum(sum, warp_sums);

  if (threadIdx.x == 0) {
    y[out_idx] = sum + skip[d] * u[out_idx];
  }
}

__global__ __launch_bounds__(REDUCE_THREADS)
void output_parallel_reference_kernel(const float *__restrict__ u,
                                      const float *__restrict__ delta,
                                      const float *__restrict__ A,
                                      const float *__restrict__ B,
                                      const float *__restrict__ C,
                                      const float *__restrict__ skip,
                                      float *__restrict__ y,
                                      int output_count,
                                      int seq_len,
                                      int d_model,
                                      int d_state) {
  const int out_idx = blockIdx.x;
  if (out_idx >= output_count) {
    return;
  }

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
      const float b_bar =
          delta_val * u[udy] * B[bc_index(b, s, n, seq_len, d_state)];
      h = fmaf(a_bar, h, b_bar);
    }

    local_sum += C[bc_index(b, t, n, seq_len, d_state)] * h;
  }

  const float sum = block_reduce_sum(local_sum, warp_sums);
  if (threadIdx.x == 0) {
    y[out_idx] = sum + skip[d] * u[out_idx];
  }
}

void launch_reference_fallback(const float *u,
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
  const int output_count = batch * seq_len * d_model;
  output_parallel_reference_kernel<<<output_count, REDUCE_THREADS, 0, stream>>>(
      u, delta, A, B, C, skip, y, output_count, seq_len, d_model, d_state);
}

}  // namespace

extern "C" size_t selective_scan_workspace_size(int batch,
                                                int seq_len,
                                                int d_model,
                                                int d_state) {
  if (batch <= 0 || seq_len <= 0 || d_model <= 0 || d_state <= 0) {
    return 0;
  }

  const int num_chunks = (seq_len + CHUNK_THREADS - 1) / CHUNK_THREADS;
  if (num_chunks > MAX_CHUNKS) {
    return 0;
  }

  const size_t state_count =
      static_cast<size_t>(batch) * d_model * d_state;
  const size_t elem_count = state_count * seq_len;
  const size_t chunk_count = state_count * num_chunks;

  size_t bytes = 256;
  bytes = align_up(bytes, 256) + elem_count * sizeof(float);   // local_a
  bytes = align_up(bytes, 256) + elem_count * sizeof(float);   // local_b
  bytes = align_up(bytes, 256) + chunk_count * sizeof(float);  // chunk_a
  bytes = align_up(bytes, 256) + chunk_count * sizeof(float);  // chunk_b
  bytes = align_up(bytes, 256) + elem_count * sizeof(float);   // contrib
  bytes = align_up(bytes, 256);
  return bytes;
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
  if (batch <= 0 || seq_len <= 0 || d_model <= 0 || d_state <= 0) {
    return;
  }

  const int output_count = batch * seq_len * d_model;
  const int num_chunks = (seq_len + CHUNK_THREADS - 1) / CHUNK_THREADS;
  if (num_chunks > MAX_CHUNKS || workspace == nullptr) {
    launch_reference_fallback(
        u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state, stream);
    return;
  }

  const size_t state_count =
      static_cast<size_t>(batch) * d_model * d_state;
  const size_t elem_count = state_count * seq_len;
  const size_t chunk_count = state_count * num_chunks;

  Workspace ws{};
  if (!carve_workspace(workspace, workspace_bytes, elem_count, chunk_count, ws)) {
    launch_reference_fallback(
        u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state, stream);
    return;
  }

  const size_t chunk_scan_shared = 2 * CHUNK_THREADS * sizeof(float);
  dim3 chunk_grid(static_cast<unsigned int>(state_count),
                  static_cast<unsigned int>(num_chunks));
  scan_state_chunks_kernel<<<chunk_grid,
                             CHUNK_THREADS,
                             chunk_scan_shared,
                             stream>>>(u,
                                       delta,
                                       A,
                                       B,
                                       ws.local_a,
                                       ws.local_b,
                                       ws.chunk_a,
                                       ws.chunk_b,
                                       batch,
                                       seq_len,
                                       d_model,
                                       d_state,
                                       num_chunks);

  const int prefix_threads = scan_threads_for(num_chunks);
  const size_t prefix_shared = 2 * prefix_threads * sizeof(float);
  scan_chunk_prefix_kernel<<<static_cast<unsigned int>(state_count),
                             prefix_threads,
                             prefix_shared,
                             stream>>>(ws.chunk_a, ws.chunk_b, num_chunks);

  const int apply_threads = 256;
  const int apply_blocks =
      static_cast<int>((elem_count + apply_threads - 1) / apply_threads);
  apply_prefix_and_project_kernel<<<apply_blocks, apply_threads, 0, stream>>>(
      ws.local_a,
      ws.local_b,
      ws.chunk_a,
      ws.chunk_b,
      C,
      ws.contrib,
      elem_count,
      seq_len,
      d_model,
      d_state,
      num_chunks);

  reduce_state_to_output_kernel<<<output_count, REDUCE_THREADS, 0, stream>>>(
      ws.contrib, u, skip, y, output_count, seq_len, d_model, d_state);
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
  const size_t workspace_bytes =
      selective_scan_workspace_size(batch, seq_len, d_model, d_state);

  if (workspace_bytes == 0) {
    launch_reference_fallback(
        u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state, stream);
    return;
  }

  void *workspace = nullptr;
#if CUDART_VERSION >= 11020
  if (cudaMallocAsync(&workspace, workspace_bytes, stream) != cudaSuccess) {
    launch_reference_fallback(
        u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state, stream);
    return;
  }

  solve_with_workspace_stream(u,
                              delta,
                              A,
                              B,
                              C,
                              skip,
                              y,
                              batch,
                              seq_len,
                              d_model,
                              d_state,
                              workspace,
                              workspace_bytes,
                              stream);

  cudaFreeAsync(workspace, stream);
#else
  if (cudaMalloc(&workspace, workspace_bytes) != cudaSuccess) {
    launch_reference_fallback(
        u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state, stream);
    return;
  }

  solve_with_workspace_stream(u,
                              delta,
                              A,
                              B,
                              C,
                              skip,
                              y,
                              batch,
                              seq_len,
                              d_model,
                              d_state,
                              workspace,
                              workspace_bytes,
                              stream);

  cudaFree(workspace);
#endif
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
  solve_stream(u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state, 0);
}
