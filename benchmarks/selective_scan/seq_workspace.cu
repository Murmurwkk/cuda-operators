// Allocation-only ablation. The four kernels below are the original source's kernels.
#define solve allocating_solve
#include "ssm_seq.cu"
#undef solve
#include <cstdlib>
static float *la=nullptr,*lb=nullptr,*ca=nullptr,*cb=nullptr;
extern "C" void bench_prepare(int b,int l,int d,int n) {
  if(l<=MAX_SCAN_THREADS) return;
  const size_t states=size_t(b)*d*n, chunks=(l+CHUNK_THREADS-1)/CHUNK_THREADS;
  if(chunks>MAX_CHUNKS) std::abort();
  if(!allocate_float_buffer(&la,states*l)||!allocate_float_buffer(&lb,states*l)||
     !allocate_float_buffer(&ca,states*chunks)||!allocate_float_buffer(&cb,states*chunks)) std::abort();
}
extern "C" void bench_release() { free_float_buffer(la);free_float_buffer(lb);free_float_buffer(ca);free_float_buffer(cb); }
extern "C" void solve(const float* u,const float* dt,const float* a,const float* b,const float* c,const float* s,float* y,int B,int L,int D,int N) {
  if(L<=MAX_SCAN_THREADS) { allocating_solve(u,dt,a,b,c,s,y,B,L,D,N);return; }
  const int chunks=(L+CHUNK_THREADS-1)/CHUNK_THREADS;
  const size_t states=size_t(B)*D*N, elems=states*L;
  launch_init_skip(u,s,y,B*L*D,L,D);
  dim3 grid(static_cast<unsigned>(states),chunks);
  scan_chunks_kernel<<<grid,CHUNK_THREADS,2*CHUNK_THREADS*sizeof(float)>>>(u,dt,a,b,la,lb,ca,cb,B,L,D,N,chunks);
  const int threads=scan_threads_for(chunks);
  scan_chunk_prefix_kernel<<<static_cast<unsigned>(states),threads,2*threads*sizeof(float)>>>(ca,cb,chunks);
  const int blocks=static_cast<int>((elems+255)/256);
  apply_chunk_prefix_kernel<<<blocks,256>>>(la,lb,cb,c,y,elems,L,D,N,chunks);
}
