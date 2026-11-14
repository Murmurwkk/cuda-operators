// Allocation-only ablation: include the measured kernels without changing them.
#define solve allocating_solve
#include "selective_scan_improved.cu"
#undef solve
#include <cstdio>
#include <cstdlib>
static void* workspace = nullptr;
static size_t workspace_bytes = 0;
extern "C" void bench_prepare(int b,int l,int d,int n) {
  workspace_bytes = selective_scan_workspace_size(b,l,d,n);
  if (workspace_bytes && cudaMalloc(&workspace,workspace_bytes) != cudaSuccess) std::abort();
}
extern "C" void bench_release() { if(workspace) cudaFree(workspace); }
extern "C" void solve(const float* u,const float* dt,const float* a,const float* b,const float* c,const float* s,float* y,int B,int L,int D,int N) {
  const auto e=solve_with_workspace_stream(u,dt,a,b,c,s,y,B,L,D,N,workspace,workspace_bytes,0);
  if(e!=cudaSuccess) { std::fprintf(stderr,"workspace solve: %s\n",cudaGetErrorString(e)); std::abort(); }
}
