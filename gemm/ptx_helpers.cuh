#pragma once

#include <cstdint>

#include <cuda_runtime.h>
#include <mma.h>

namespace ptx_helpers {

__device__ __forceinline__ bool is_aligned_16(const void *ptr) {
  return (reinterpret_cast<std::uintptr_t>(ptr) & 0xF) == 0;
}

__device__ __forceinline__ void zero4(float *dst) {
  dst[0] = 0.0f;
  dst[1] = 0.0f;
  dst[2] = 0.0f;
  dst[3] = 0.0f;
}

__device__ __forceinline__ void copy4_or_zero(float *dst,
                                              const float *src,
                                              int valid_bytes) {
  const int valid_floats = valid_bytes >> 2;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    dst[i] = i < valid_floats ? src[i] : 0.0f;
  }
}

__device__ __forceinline__ void cp_async_or_scalar(float *dst,
                                                   const float *src,
                                                   int valid_bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  if (valid_bytes == 16 && is_aligned_16(dst) && is_aligned_16(src)) {
    const unsigned int shared_addr =
        static_cast<unsigned int>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(shared_addr), "l"(src) : "memory");
    return;
  }
#endif

  if (valid_bytes > 0) {
    copy4_or_zero(dst, src, valid_bytes);
  } else {
    zero4(dst);
  }
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" : : : "memory");
#endif
}

template <int PendingGroups>
__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" : : "n"(PendingGroups) : "memory");
#endif
}

template <typename Fragment>
__device__ __forceinline__ void convert_fragment_to_tf32(Fragment &frag) {
  #pragma unroll
  for (int i = 0; i < Fragment::num_elements; ++i) {
    frag.x[i] = nvcuda::wmma::__float_to_tf32(frag.x[i]);
  }
}

}  // namespace ptx_helpers
