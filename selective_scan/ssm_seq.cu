#include <cuda_runtime.h>

namespace {

constexpr int WARP_SIZE = 32;
constexpr int REDUCE_THREADS = 256;
constexpr int REDUCE_WARPS = REDUCE_THREADS / WARP_SIZE;
constexpr int MAX_SCAN_THREADS = 1024;
constexpr int CHUNK_THREADS = 256;
constexpr int MAX_CHUNKS = 1024;

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

__global__ void init_skip_kernel(const float *__restrict__ u,
                                 const float *__restrict__ skip,
                                 float *__restrict__ y,
                                 int total,
                                 int seq_len,
                                 int d_model) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int d = idx % d_model;
  y[idx] = skip[d] * u[idx];
}

__global__ void selective_scan_seq_kernel(const float *__restrict__ u,
                                          const float *__restrict__ delta,
                                          const float *__restrict__ A,
                                          const float *__restrict__ B,
                                          const float *__restrict__ C,
                                          float *__restrict__ y,
                                          int batch,
                                          int seq_len,
                                          int d_model,
                                          int d_state) {
  const int bd = blockIdx.x;
  const int n = blockIdx.y;
  const int b = bd / d_model;
  const int d = bd - b * d_model;
  if (b >= batch || n >= d_state) {
    return;
  }

  extern __shared__ float shared[];
  float *scan_a = shared;
  float *scan_b = scan_a + blockDim.x;

  const int t = threadIdx.x;
  float cur_a = 1.0f;
  float cur_b = 0.0f;

  if (t < seq_len) {
    const int udy = udy_index(b, t, d, seq_len, d_model);
    const float delta_val = delta[udy];
    cur_a = expf(delta_val * A[d * d_state + n]);
    cur_b = delta_val * u[udy] * B[bc_index(b, t, n, seq_len, d_state)];
  }

  scan_a[t] = cur_a;
  scan_b[t] = cur_b;
  __syncthreads();

  for (int offset = 1; offset < blockDim.x; offset <<= 1) {
    float left_a = 1.0f;
    float left_b = 0.0f;
    if (t >= offset) {
      left_a = scan_a[t - offset];
      left_b = scan_b[t - offset];
    }
    __syncthreads();

    if (t >= offset) {
      float next_a;
      float next_b;
      compose_affine(cur_a, cur_b, left_a, left_b, next_a, next_b);
      cur_a = next_a;
      cur_b = next_b;
    }
    __syncthreads();

    scan_a[t] = cur_a;
    scan_b[t] = cur_b;
    __syncthreads();
  }

  if (t < seq_len) {
    const int udy = udy_index(b, t, d, seq_len, d_model);
    const float c_val = C[bc_index(b, t, n, seq_len, d_state)];
    atomicAdd(y + udy, c_val * cur_b);
  }
}

__global__ void scan_chunks_kernel(const float *__restrict__ u,
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
  const int state_idx = blockIdx.x;
  const int chunk = blockIdx.y;
  const int n = state_idx % d_state;
  const int bd = state_idx / d_state;
  const int b = bd / d_model;
  const int d = bd - b * d_model;
  if (b >= batch || chunk >= num_chunks) {
    return;
  }

  extern __shared__ float shared[];
  float *scan_a = shared;
  float *scan_b = scan_a + blockDim.x;

  const int local_t = threadIdx.x;
  const int t = chunk * CHUNK_THREADS + local_t;
  float cur_a = 1.0f;
  float cur_b = 0.0f;

  if (t < seq_len) {
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
    const size_t elem_idx =
        (static_cast<size_t>(state_idx) * seq_len) + t;
    local_a[elem_idx] = cur_a;
    local_b[elem_idx] = cur_b;
  }

  if (local_t == CHUNK_THREADS - 1) {
    const size_t chunk_idx =
        (static_cast<size_t>(state_idx) * num_chunks) + chunk;
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
    const size_t idx =
        (static_cast<size_t>(state_idx) * num_chunks) + chunk;
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
    float prefix_b = 0.0f;
    if (chunk > 0) {
      prefix_b = scan_b[chunk - 1];
    }
    const size_t idx =
        (static_cast<size_t>(state_idx) * num_chunks) + chunk;
    chunk_b[idx] = prefix_b;
  }
}

__global__ void apply_chunk_prefix_kernel(const float *__restrict__ local_a,
                                          const float *__restrict__ local_b,
                                          const float *__restrict__ chunk_b,
                                          const float *__restrict__ C,
                                          float *__restrict__ y,
                                          size_t total_elems,
                                          int seq_len,
                                          int d_model,
                                          int d_state,
                                          int num_chunks) {
  const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (idx >= total_elems) {
    return;
  }

  const int t = static_cast<int>(idx % seq_len);
  const size_t state_idx = idx / seq_len;
  const int n = static_cast<int>(state_idx % d_state);
  const int bd = static_cast<int>(state_idx / d_state);
  const int b = bd / d_model;
  const int d = bd - b * d_model;
  const int chunk = t / CHUNK_THREADS;

  const float h = fmaf(local_a[idx],
                       chunk_b[state_idx * num_chunks + chunk],
                       local_b[idx]);
  const float c_val = C[bc_index(b, t, n, seq_len, d_state)];
  atomicAdd(y + udy_index(b, t, d, seq_len, d_model), c_val * h);
}

__global__ __launch_bounds__(REDUCE_THREADS)
void selective_scan_output_parallel_kernel(const float *__restrict__ u,
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
  if (out_idx >= total) {
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
      const float b_val = B[bc_index(b, s, n, seq_len, d_state)];
      h = fmaf(delta_val * u[udy], b_val, a_bar * h);
    }
    local_sum += C[bc_index(b, t, n, seq_len, d_state)] * h;
  }

  const float sum = block_reduce_sum(local_sum, warp_sums);
  if (threadIdx.x == 0) {
    y[out_idx] = sum + skip[d] * u[out_idx];
  }
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

bool allocate_float_buffer(float **ptr, size_t count) {
  *ptr = nullptr;
  if (count == 0) {
    return true;
  }
  return cudaMalloc(reinterpret_cast<void **>(ptr), count * sizeof(float)) ==
         cudaSuccess;
}

void free_float_buffer(float *ptr) {
  if (ptr != nullptr) {
    cudaFree(ptr);
  }
}

void launch_init_skip(const float *u,
                      const float *skip,
                      float *y,
                      int total,
                      int seq_len,
                      int d_model) {
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  init_skip_kernel<<<blocks, threads>>>(u, skip, y, total, seq_len, d_model);
}

void launch_output_parallel_fallback(const float *u,
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
  const int total = batch * seq_len * d_model;
  selective_scan_output_parallel_kernel<<<total, REDUCE_THREADS>>>(
      u, delta, A, B, C, skip, y, total, seq_len, d_model, d_state);
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

  const int output_count = batch * seq_len * d_model;
  const int bd_count = batch * d_model;

  if (seq_len <= MAX_SCAN_THREADS) {
    launch_init_skip(u, skip, y, output_count, seq_len, d_model);

    const int scan_threads = scan_threads_for(seq_len);
    const size_t shared_bytes = 2 * scan_threads * sizeof(float);
    dim3 grid(bd_count, d_state);
    selective_scan_seq_kernel<<<grid, scan_threads, shared_bytes>>>(
        u, delta, A, B, C, y, batch, seq_len, d_model, d_state);
    return;
  }

  const int num_chunks = (seq_len + CHUNK_THREADS - 1) / CHUNK_THREADS;
  if (num_chunks > MAX_CHUNKS) {
    launch_output_parallel_fallback(
        u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state);
    return;
  }

  const size_t state_count =
      static_cast<size_t>(batch) * d_model * d_state;
  const size_t elem_count = state_count * seq_len;
  const size_t chunk_count = state_count * num_chunks;

  float *local_a = nullptr;
  float *local_b = nullptr;
  float *chunk_a = nullptr;
  float *chunk_b = nullptr;

  const bool allocated =
      allocate_float_buffer(&local_a, elem_count) &&
      allocate_float_buffer(&local_b, elem_count) &&
      allocate_float_buffer(&chunk_a, chunk_count) &&
      allocate_float_buffer(&chunk_b, chunk_count);

  if (!allocated) {
    free_float_buffer(local_a);
    free_float_buffer(local_b);
    free_float_buffer(chunk_a);
    free_float_buffer(chunk_b);
    launch_output_parallel_fallback(
        u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state);
    return;
  }

  launch_init_skip(u, skip, y, output_count, seq_len, d_model);

  const size_t chunk_scan_shared = 2 * CHUNK_THREADS * sizeof(float);
  dim3 chunk_grid(static_cast<unsigned int>(state_count), num_chunks);
  scan_chunks_kernel<<<chunk_grid, CHUNK_THREADS, chunk_scan_shared>>>(
      u,
      delta,
      A,
      B,
      local_a,
      local_b,
      chunk_a,
      chunk_b,
      batch,
      seq_len,
      d_model,
      d_state,
      num_chunks);

  const int chunk_prefix_threads = scan_threads_for(num_chunks);
  const size_t chunk_prefix_shared =
      2 * chunk_prefix_threads * sizeof(float);
  scan_chunk_prefix_kernel<<<static_cast<unsigned int>(state_count),
                             chunk_prefix_threads,
                             chunk_prefix_shared>>>(chunk_a,
                                                    chunk_b,
                                                    num_chunks);

  const int apply_threads = 256;
  const int apply_blocks =
      static_cast<int>((elem_count + apply_threads - 1) / apply_threads);
  apply_chunk_prefix_kernel<<<apply_blocks, apply_threads>>>(
      local_a,
      local_b,
      chunk_b,
      C,
      y,
      elem_count,
      seq_len,
      d_model,
      d_state,
      num_chunks);

  free_float_buffer(local_a);
  free_float_buffer(local_b);
  free_float_buffer(chunk_a);
  free_float_buffer(chunk_b);
}
