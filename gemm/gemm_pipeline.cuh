#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace mur_gemm_pipeline {
constexpr int BM=128, BN=64, BK=32, THREADS=128, TM=8, TN=8;

template<bool Swizzle>
__device__ __forceinline__ int a_index(int row,int col) {
  if constexpr(Swizzle) return row*BK + ((col/4)^((row/TM)%8))*4 + col%4;
  return row*BK+col;
}
template<bool Async>
__device__ __forceinline__ void copy4(float* dst,const float* src) {
#if __CUDA_ARCH__ >= 800
  if constexpr(Async) {
    const uint32_t smem=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem),"l"(src) : "memory");
    return;
  }
#endif
  *reinterpret_cast<float4*>(dst)=*reinterpret_cast<const float4*>(src);
}
template<bool Async> __device__ __forceinline__ void commit() {
#if __CUDA_ARCH__ >= 800
  if constexpr(Async) asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}
template<bool Async> __device__ __forceinline__ void wait() {
#if __CUDA_ARCH__ >= 800
  if constexpr(Async) asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}
template<bool Swizzle,bool Async>
__device__ __forceinline__ void load_tile(const float* A,const float* B,float* sa,float* sb,
                                        int M,int N,int K,int row0,int col0,int k0) {
  for(int v=threadIdx.x;v<BM*BK/4;v+=THREADS) {
    const int r=v/(BK/4),c=(v%(BK/4))*4;
    const bool full=row0+r<M && k0+c+3<K;
    const float* src=full?A+size_t(row0+r)*K+k0+c:A;
    float* dst=sa+a_index<Swizzle>(r,c);
    if(full && (reinterpret_cast<uintptr_t>(src)&15)==0) copy4<Async>(dst,src);
    else {
#pragma unroll
      for(int j=0;j<4;++j) dst[j]=(row0+r<M && k0+c+j<K)?A[size_t(row0+r)*K+k0+c+j]:0.0f;
    }
  }
  for(int v=threadIdx.x;v<BK*BN/4;v+=THREADS) {
    const int r=v/(BN/4),c=(v%(BN/4))*4;
    const bool full=k0+r<K && col0+c+3<N;
    const float* src=full?B+size_t(k0+r)*N+col0+c:B;
    float* dst=sb+r*BN+c;
    if(full && (reinterpret_cast<uintptr_t>(src)&15)==0) copy4<Async>(dst,src);
    else {
#pragma unroll
      for(int j=0;j<4;++j) dst[j]=(k0+r<K && col0+c+j<N)?B[size_t(k0+r)*N+col0+c+j]:0.0f;
    }
  }
  commit<Async>();
}

// Same compute mapping for all variants. Stages=2 alone does not imply overlap;
// Async=false remains an ordinary global->register->shared load path.
template<bool Swizzle,bool Async,int Stages>
__global__ __launch_bounds__(THREADS) void kernel(const float* A,const float* B,float* C,int M,int N,int K) {
  static_assert(Stages==1 || Stages==2,"single or double buffering only");
  __shared__ __align__(16) float storage[Stages*(BM*BK+BK*BN)];
  float* sa=storage;float* sb=storage+Stages*BM*BK;
  const int warp=threadIdx.x/32,lane=threadIdx.x%32;
  const int mr=(warp/2)*64+(lane/4)*TM,nc=(warp%2)*32+(lane%4)*TN;
  const int row0=blockIdx.y*BM,col0=blockIdx.x*BN;
  const int tiles=(K+BK-1)/BK;
  float accum[TM][TN]={};
  float ra[2][TM],rb[2][TN];
  if(tiles>0) {
    load_tile<Swizzle,Async>(A,B,sa,sb,M,N,K,row0,col0,0);
    wait<Async>();__syncthreads();
  }
  int slot=0;
  for(int tile=0;tile<tiles;++tile) {
    const int next_slot=Stages==2?slot^1:0;
    if constexpr(Stages==2) {
      if(tile+1<tiles) load_tile<Swizzle,Async>(A,B,sa+next_slot*BM*BK,sb+next_slot*BK*BN,M,N,K,row0,col0,(tile+1)*BK);
    }
    const float* ca=sa+slot*BM*BK;const float* cb=sb+slot*BK*BN;
    // Prime fragment 0 before the first FMA, on EVERY tile. A zero initialized
    // register would hide the defect but still omit k=0's contribution.
#pragma unroll
    for(int i=0;i<TM;++i)ra[0][i]=ca[a_index<Swizzle>(mr+i,0)];
#pragma unroll
    for(int j=0;j<TN;++j)rb[0][j]=cb[nc+j];
#pragma unroll
    for(int kk=0;kk<BK;++kk) {
      const int cur=kk&1,nxt=cur^1;
      if(kk+1<BK) {
#pragma unroll
        for(int i=0;i<TM;++i)ra[nxt][i]=ca[a_index<Swizzle>(mr+i,kk+1)];
#pragma unroll
        for(int j=0;j<TN;++j)rb[nxt][j]=cb[(kk+1)*BN+nc+j];
      }
#pragma unroll
      for(int i=0;i<TM;++i) {
#pragma unroll
        for(int j=0;j<TN;++j)accum[i][j]=fmaf(ra[cur][i],rb[cur][j],accum[i][j]);
      }
    }
    if constexpr(Stages==1) {
      // All consumers finish BEFORE the only slot is overwritten.
      __syncthreads();
      if(tile+1<tiles) load_tile<Swizzle,Async>(A,B,sa,sb,M,N,K,row0,col0,(tile+1)*BK);
    }
    wait<Async>();
    // Wait handles this thread's copies; the barrier publishes all producers
    // and also ensures the old slot has no remaining readers before reuse.
    __syncthreads();slot=next_slot;
  }
#pragma unroll
  for(int i=0;i<TM;++i) {
#pragma unroll
    for(int j=0;j<TN;++j)if(row0+mr+i<M && col0+nc+j<N)C[size_t(row0+mr+i)*N+col0+nc+j]=accum[i][j];
  }
}

inline void launch(const float* A,const float* B,float* C,int M,int N,int K,int mode,cudaStream_t stream) {
  if(M<=0||N<=0||K<0||!A||!B||!C)return;
  const dim3 grid((N+BN-1)/BN,(M+BM-1)/BM);
  if(mode==0)kernel<false,false,1><<<grid,THREADS,0,stream>>>(A,B,C,M,N,K);
  else if(mode==1)kernel<true,false,1><<<grid,THREADS,0,stream>>>(A,B,C,M,N,K);
  else if(mode==2)kernel<true,false,2><<<grid,THREADS,0,stream>>>(A,B,C,M,N,K);
  else kernel<true,true,2><<<grid,THREADS,0,stream>>>(A,B,C,M,N,K);
}
} // namespace mur_gemm_pipeline
