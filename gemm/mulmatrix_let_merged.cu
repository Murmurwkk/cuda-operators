// Corrected fixed-tile SIMT pipeline. The original source is retained in the
// experiment archive, including its missing first register fragment and padded
// grid without output bounds. This interface retains the original (M,N,K) ABI.
#include "gemm_pipeline.cuh"

extern "C" void mulmatrix_pipeline_launch(const float* A,const float* B,float* C,
                                          int M,int N,int K,cudaStream_t stream) {
  mur_gemm_pipeline::launch(A,B,C,M,N,K,3,stream);
}
extern "C" void solve(const float* A,const float* B,float* C,int M,int N,int K) {
  mulmatrix_pipeline_launch(A,B,C,M,N,K,nullptr);
  cudaDeviceSynchronize();
}
