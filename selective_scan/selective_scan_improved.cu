#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <limits.h>

namespace {

constexpr int WARP_SIZE = 32;
constexpr int REDUCE_THREADS = 256;
constexpr int REDUCE_WARPS = REDUCE_THREADS / WARP_SIZE;
constexpr int MAX_SCAN_THREADS = 1024;
constexpr int CHUNK_THREADS = 256;
constexpr int MAX_CHUNKS = 1024;
constexpr int SMALL_SEQ_OUTPUT_PARALLEL_LIMIT = 64;

__device__ __forceinline__ int udy_index(int b, int t, int d, int seq_len,
                                         int d_model) {
  return (b * seq_len + t) * d_model + d;
}

__device__ __forceinline__ int bc_index(int b, int t, int n, int seq_len,
                                        int d_state) {
  return (b * seq_len + t) * d_state + n;
}

// right(left(x)) = right_a * (left_a * x + left_b) + right_b.
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
    const size_t elem_idx = static_cast<size_t>(state_idx) * seq_len + t;
    local_a[elem_idx] = cur_a;
    local_b[elem_idx] = cur_b;
  }

  // The final lane contains the complete chunk transform because invalid tail
  // elements are identity transforms and do not change the composed pair.
  if (local_t == CHUNK_THREADS - 1) {
    const size_t chunk_idx =
        static_cast<size_t>(state_idx) * num_chunks + chunk;
    chunk_a[chunk_idx] = cur_a;
    chunk_b[chunk_idx] = cur_b;
  }
}

// Converts per-chunk inclusive transforms into exclusive prefix transforms.
// After this kernel, chunk_a/chunk_b at chunk k store the transform of all
// chunks before k: F_{k-1}(...F_0(x)). Keeping prefix_a fixes the old code's
// loss of affine scale information and keeps the representation correct for
// non-zero initial states or any future multi-level prefix pass.
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

__global__ void apply_chunk_prefix_kernel(const float *__restrict__ local_a,
                                          const float *__restrict__ local_b,
                                          const float *__restrict__ prefix_a,
                                          const float *__restrict__ prefix_b,
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
  const size_t chunk_idx = state_idx * num_chunks + chunk;

  // Current recurrence assumes h_{-1}=0, so the exclusive prefix contributes
  // prefix_b. prefix_a is intentionally kept in memory/interface so the prefix
  // representation remains complete and the kernel can be extended to non-zero
  // initial state as: h0 = prefix_a[chunk_idx] * initial + prefix_b[chunk_idx].
  (void)prefix_a;
  const float h0 = prefix_b[chunk_idx];
  const float h = fmaf(local_a[idx], h0, local_b[idx]);
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

size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

float *take_float_buffer(char *base, size_t &offset, size_t count) {
  offset = align_up(offset, 256);
  float *ptr = reinterpret_cast<float *>(base + offset);
  offset += count * sizeof(float);
  return ptr;
}

bool valid_shape(int batch, int seq_len, int d_model, int d_state) {
  return batch > 0 && seq_len > 0 && d_model > 0 && d_state > 0;
}

bool checked_mul_size(size_t a, size_t b, size_t *out) {
  if (a != 0 && b > SIZE_MAX / a) {
    return false;
  }
  *out = a * b;
  return true;
}

void launch_init_skip(const float *u,
                      const float *skip,
                      float *y,
                      int total,
                      int d_model,
                      cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  init_skip_kernel<<<blocks, threads, 0, stream>>>(u, skip, y, total, d_model);
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
                                     int d_state,
                                     cudaStream_t stream) {
  const int total = batch * seq_len * d_model;
  selective_scan_output_parallel_kernel<<<total, REDUCE_THREADS, 0, stream>>>(
      u, delta, A, B, C, skip, y, total, seq_len, d_model, d_state);
}

cudaError_t allocate_async(void **ptr, size_t bytes, cudaStream_t stream) {
  *ptr = nullptr;
  if (bytes == 0) {
    return cudaSuccess;
  }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
  return cudaMallocAsync(ptr, bytes, stream);
#else
  (void)stream;
  return cudaMalloc(ptr, bytes);
#endif
}

cudaError_t free_async(void *ptr, cudaStream_t stream) {
  if (ptr == nullptr) {
    return cudaSuccess;
  }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
  return cudaFreeAsync(ptr, stream);
#else
  (void)stream;
  return cudaFree(ptr);
#endif
}

}  // namespace

extern "C" size_t selective_scan_workspace_size(int batch,
                                                int seq_len,
                                                int d_model,
                                                int d_state) {
  if (!valid_shape(batch, seq_len, d_model, d_state)) {
    return 0;
  }
  if (seq_len <= MAX_SCAN_THREADS) {
    return 0;
  }
  const int num_chunks = (seq_len + CHUNK_THREADS - 1) / CHUNK_THREADS;
  if (num_chunks > MAX_CHUNKS) {
    return 0;
  }

  size_t state_count = 0;
  size_t elem_count = 0;
  size_t chunk_count = 0;
  if (!checked_mul_size(static_cast<size_t>(batch), static_cast<size_t>(d_model), &state_count) ||
      !checked_mul_size(state_count, static_cast<size_t>(d_state), &state_count) ||
      !checked_mul_size(state_count, static_cast<size_t>(seq_len), &elem_count) ||
      !checked_mul_size(state_count, static_cast<size_t>(num_chunks), &chunk_count)) {
    return 0;
  }

  size_t bytes = 0;
  bytes = align_up(bytes, 256) + elem_count * sizeof(float);   // local_a
  bytes = align_up(bytes, 256) + elem_count * sizeof(float);   // local_b
  bytes = align_up(bytes, 256) + chunk_count * sizeof(float);  // chunk/prefix_a
  bytes = align_up(bytes, 256) + chunk_count * sizeof(float);  // chunk/prefix_b
  return bytes;
}

extern "C" cudaError_t solve_with_workspace_stream(const float *u,
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
    return cudaErrorInvalidValue;
  }
  if (!valid_shape(batch, seq_len, d_model, d_state)) {
    return cudaErrorInvalidValue;
  }

  size_t output_count_size = 0;
  if (!checked_mul_size(static_cast<size_t>(batch), static_cast<size_t>(seq_len), &output_count_size) ||
      !checked_mul_size(output_count_size, static_cast<size_t>(d_model), &output_count_size) ||
      output_count_size > static_cast<size_t>(INT32_MAX)) {
    return cudaErrorInvalidValue;
  }
  const int output_count = static_cast<int>(output_count_size);
  const int bd_count = batch * d_model;

  // Very short sequences avoid d_state-way atomic contention and excessive
  // scan block count by using one CTA per output. This is O(L^2) in sequence
  // length, so it is deliberately capped at a small threshold.
  if (seq_len <= SMALL_SEQ_OUTPUT_PARALLEL_LIMIT) {
    launch_output_parallel_fallback(
        u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state, stream);
    return cudaGetLastError();
  }

  if (seq_len <= MAX_SCAN_THREADS) {
    launch_init_skip(u, skip, y, output_count, d_model, stream);

    const int scan_threads = scan_threads_for(seq_len);
    const size_t shared_bytes = 2 * scan_threads * sizeof(float);
    dim3 grid(bd_count, d_state);
    selective_scan_seq_kernel<<<grid, scan_threads, shared_bytes, stream>>>(
        u, delta, A, B, C, y, batch, seq_len, d_model, d_state);
    return cudaGetLastError();
  }

  const int num_chunks = (seq_len + CHUNK_THREADS - 1) / CHUNK_THREADS;
  if (num_chunks > MAX_CHUNKS) {
    launch_output_parallel_fallback(
        u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state, stream);
    return cudaGetLastError();
  }

  size_t state_count = 0;
  size_t elem_count = 0;
  size_t chunk_count = 0;
  if (!checked_mul_size(static_cast<size_t>(batch), static_cast<size_t>(d_model), &state_count) ||
      !checked_mul_size(state_count, static_cast<size_t>(d_state), &state_count) ||
      !checked_mul_size(state_count, static_cast<size_t>(seq_len), &elem_count) ||
      !checked_mul_size(state_count, static_cast<size_t>(num_chunks), &chunk_count) ||
      state_count > static_cast<size_t>(UINT32_MAX)) {
    return cudaErrorInvalidValue;
  }

  const size_t required_workspace =
      selective_scan_workspace_size(batch, seq_len, d_model, d_state);
  if (workspace == nullptr || workspace_bytes < required_workspace) {
    launch_output_parallel_fallback(
        u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state, stream);
    return cudaGetLastError();
  }

  char *base = reinterpret_cast<char *>(workspace);
  size_t offset = 0;
  float *local_a = take_float_buffer(base, offset, elem_count);
  float *local_b = take_float_buffer(base, offset, elem_count);
  float *chunk_a = take_float_buffer(base, offset, chunk_count);
  float *chunk_b = take_float_buffer(base, offset, chunk_count);

  launch_init_skip(u, skip, y, output_count, d_model, stream);

  const size_t chunk_scan_shared = 2 * CHUNK_THREADS * sizeof(float);
  dim3 chunk_grid(static_cast<unsigned int>(state_count), num_chunks);
  scan_chunks_kernel<<<chunk_grid, CHUNK_THREADS, chunk_scan_shared, stream>>>(
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
                             chunk_prefix_shared,
                             stream>>>(chunk_a, chunk_b, num_chunks);

  const int apply_threads = 256;
  const int apply_blocks =
      static_cast<int>((elem_count + apply_threads - 1) / apply_threads);
  apply_chunk_prefix_kernel<<<apply_blocks, apply_threads, 0, stream>>>(
      local_a,
      local_b,
      chunk_a,
      chunk_b,
      C,
      y,
      elem_count,
      seq_len,
      d_model,
      d_state,
      num_chunks);

  return cudaGetLastError();
}

extern "C" cudaError_t solve_stream(const float *u,
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
    return solve_with_workspace_stream(u,
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
                                       nullptr,
                                       0,
                                       stream);
  }

  void *workspace = nullptr;
  cudaError_t status = allocate_async(&workspace, workspace_bytes, stream);
  if (status != cudaSuccess) {
    return solve_with_workspace_stream(u,
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
                                       nullptr,
                                       0,
                                       stream);
  }

  status = solve_with_workspace_stream(u,
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
  cudaError_t free_status = free_async(workspace, stream);
  return status != cudaSuccess ? status : free_status;
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
  (void)solve_stream(u,
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
                     0);
}
