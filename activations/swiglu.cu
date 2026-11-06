#include <cuda_runtime.h>
#include <cstdint>

// x is contiguous [2, n]: SiLU(x[0]) * x[1]. Float32, out-of-place.
namespace {
__device__ __forceinline__ float silu_times(float a, float b) {
    // Avoid exp(-a) overflow for large negative finite a.
    float e=__expf(-fabsf(a));
    float sigmoid=a>=0.f ? 1.f/(1.f+e) : e/(1.f+e);
    return (a*sigmoid)*b;
}
__global__ void swiglu_scalar(const float* x,float* out,int64_t n) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
        i<n;i+=int64_t(blockDim.x)*gridDim.x)
        out[i]=silu_times(x[i],x[n+i]);
}
__global__ void swiglu_vector(const float* x,float* out,int64_t n) {
    const float4* a=reinterpret_cast<const float4*>(x);
    const float4* b=reinterpret_cast<const float4*>(x+n);
    float4* y=reinterpret_cast<float4*>(out);
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
        i<n/4;i+=int64_t(blockDim.x)*gridDim.x) {
        float4 u=a[i],v=b[i];
        y[i]=make_float4(silu_times(u.x,v.x),silu_times(u.y,v.y),
                        silu_times(u.z,v.z),silu_times(u.w,v.w));
    }
}
}
// Asynchronous checked entry; total_elements counts BOTH halves.
// No CUDA work for zero; odd/negative counts are invalid. Input and output
// must not overlap. Pointers need float alignment; float4 is conditional.
extern "C" cudaError_t swiglu_launch(
    const float* input,float* output,int64_t total_elements,cudaStream_t stream) {
    if(total_elements<0 || total_elements%2) return cudaErrorInvalidValue;
    if(total_elements==0) return cudaSuccess;
    if(!input || !output) return cudaErrorInvalidValue;
    const int64_t n=total_elements/2;
    const bool vector_ok=n%4==0 &&
        ((reinterpret_cast<uintptr_t>(input) |
          reinterpret_cast<uintptr_t>(input+n) |
          reinterpret_cast<uintptr_t>(output)) & 15u)==0;
    const int64_t work=vector_ok?n/4:n;
    const int threads=256;
    const int blocks=int((work+threads-1)/threads>65535?65535:(work+threads-1)/threads);
    if(vector_ok) swiglu_vector<<<blocks,threads,0,stream>>>(input,output,n);
    else swiglu_scalar<<<blocks,threads,0,stream>>>(input,output,n);
    return cudaGetLastError();
}
// Compatibility wrapper for the original standalone call convention.
// Framework integration uses swiglu_launch to preserve status and stream.
extern "C" void solve(const float* input,float* output,int total_elements) {
    (void)swiglu_launch(input,output,total_elements,nullptr);
}
