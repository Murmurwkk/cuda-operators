// Compatibility entry point. The historical signature named its arguments
// (M,N,K) but immediately swapped the latter two, so its effective ABI was
// (M,K,N). Preserve that behavior with explicit parameter names.
#include "gemm_pipeline.cuh"

extern "C" void mulmatrix_let2_launch(const float* A,const float* B,float* C,
                                      int M,int K,int N,cudaStream_t stream) {
  mur_gemm_pipeline::launch(A,B,C,M,N,K,3,stream);
}
extern "C" void solve(const float* A,const float* B,float* C,int M,int K,int N) {
  mulmatrix_let2_launch(A,B,C,M,K,N,nullptr);
  cudaDeviceSynchronize();
}
