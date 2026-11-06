#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int WARP_SIZE = 32;
constexpr int MAX_STATE_THREADS = 1024;
constexpr int FALLBACK_THREADS = 256;

__device__ __forceinline__ int udy_index(int b, int t, int d, int seq_len,
                                         int d_model) {
  return (b * seq_len + t) * d_model + d;
}

__device__ __forceinline__ int bc_index(int b, int t, int n, int seq_len,
                                        int d_state) {
  return (b * seq_len + t) * d_state + n;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
  __shared__ float warp_sums[32];
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x >> 5;
  const int warp_count = (blockDim.x + WARP_SIZE - 1) >> 5;

  v = warp_reduce_sum(v);
  if (lane == 0) {
    warp_sums[warp] = v;
  }
  __syncthreads();

  v = (threadIdx.x < warp_count) ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
    v = warp_reduce_sum(v);
  }
  return v;
}

// Main path. One CTA owns one (batch, d_model) stream. Time is advanced in
// recurrence order; d_state is mapped to threads; the C*h reduction is done
// inside the CTA, so y is written exactly once and no atomicAdd is needed.
__global__ void selective_scan_stream_kernel(const float *__restrict__ u,
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
  const int n = threadIdx.x;

  if (b >= batch || n >= d_state) {
    // Threads with n >= d_state still participate in reductions with zero.
  }

  float h = 0.0f;
  const float a_base = (n < d_state) ? A[d * d_state + n] : 0.0f;

  for (int t = 0; t < seq_len; ++t) {
    const int udy = udy_index(b, t, d, seq_len, d_model);
    const float delta_val = delta[udy];
    const float u_val = u[udy];

    float contrib = 0.0f;
    if (n < d_state) {
      const int btn = bc_index(b, t, n, seq_len, d_state);
      const float a_bar = expf(delta_val * a_base);
      const float b_bar = delta_val * u_val * B[btn];
      h = fmaf(a_bar, h, b_bar);
      contrib = C[btn] * h;
    }

    const float sum = block_reduce_sum(contrib);
    if (threadIdx.x == 0) {
      y[udy] = sum + skip[d] * u_val;
    }
    __syncthreads();
  }
}

// Correctness fallback for very large d_state. It computes one output element
// per CTA and recomputes the recurrence from the beginning of the sequence.
// This is intentionally not the fast path; it avoids atomics and large scratch
// buffers for shapes outside the one-state-per-thread regime.
__global__ void selective_scan_output_fallback_kernel(
    const float *__restrict__ u,
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

  float local_sum = 0.0f;
  for (int n = threadIdx.x; n < d_state; n += blockDim.x) {
    float h = 0.0f;
    const float a_base = A[d * d_state + n];
    for (int s = 0; s <= t; ++s) {
      const int udy = udy_index(b, s, d, seq_len, d_model);
      const float delta_val = delta[udy];
      const float a_bar = expf(delta_val * a_base);
      const float b_bar = delta_val * u[udy] * B[bc_index(b, s, n, seq_len, d_state)];
      h = fmaf(a_bar, h, b_bar);
    }
    local_sum += C[bc_index(b, t, n, seq_len, d_state)] * h;
  }

  const float sum = block_reduce_sum(local_sum);
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

int state_threads_for(int d_state) {
  int threads = next_power_of_two(d_state);
  if (threads < WARP_SIZE) {
    threads = WARP_SIZE;
  }
  if (threads > MAX_STATE_THREADS) {
    threads = MAX_STATE_THREADS;
  }
  return threads;
}

bool valid_args(const float *u,
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
  return u != nullptr && delta != nullptr && A != nullptr && B != nullptr &&
         C != nullptr && skip != nullptr && y != nullptr && batch > 0 &&
         seq_len > 0 && d_model > 0 && d_state > 0;
}

}  // namespace

extern "C" size_t selective_scan_workspace_size(int batch,
                                                 int seq_len,
                                                 int d_model,
                                                 int d_state) {
  (void)batch;
  (void)seq_len;
  (void)d_model;
  (void)d_state;
  return 0;
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
  if (!valid_args(u, delta, A, B, C, skip, y, batch, seq_len, d_model,
                  d_state)) {
    return cudaSuccess;
  }

  const int64_t total64 = static_cast<int64_t>(batch) * seq_len * d_model;
  const int64_t bd64 = static_cast<int64_t>(batch) * d_model;
  if (total64 > INT32_MAX || bd64 > INT32_MAX) {
    return cudaErrorInvalidValue;
  }

  if (d_state <= MAX_STATE_THREADS) {
    const int threads = state_threads_for(d_state);
    selective_scan_stream_kernel<<<static_cast<int>(bd64), threads, 0, stream>>>(
        u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state);
  } else {
    selective_scan_output_fallback_kernel<<<static_cast<int>(total64),
                                            FALLBACK_THREADS, 0, stream>>>(
        u, delta, A, B, C, skip, y, static_cast<int>(total64), seq_len,
        d_model, d_state);
  }

  return cudaGetLastError();
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
  (void)workspace;
  (void)workspace_bytes;
  return solve_stream(u, delta, A, B, C, skip, y, batch, seq_len, d_model,
                      d_state, stream);
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
  (void)solve_stream(u, delta, A, B, C, skip, y, batch, seq_len, d_model,
                     d_state, 0);
}

