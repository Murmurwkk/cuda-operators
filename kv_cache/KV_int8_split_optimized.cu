#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <math.h>
#include <vector>

#define WARP_SIZE 32
#define QK_WARPS_PER_BLOCK 8
#define QK_THREADS (QK_WARPS_PER_BLOCK * WARP_SIZE)
#define SOFTMAX_THREADS 256
#define PV_THREADS 256
#define PV_SEQ_TILE 256

#ifndef KV_CUDA_DEBUG_SYNC
#define KV_CUDA_DEBUG_SYNC 0
#endif

namespace {
    __global__ void qk_warp_kernel(
        const float* __restrict__ Q,
        const int8_t* __restrict__ K_int8,
        const float* __restrict__ k_scale,
        float* __restrict__ S,
        int num_heads,
        int seq_len,
        int head_dim,
        float inv_sqrt_dim
    );

    inline bool cuda_check(cudaError_t err,const char* expr,const char* file,int line)
    {
        if(err==cudaSuccess)
        {
            return true;
        }
        fprintf(stderr,"[KV_int8_split_optimized] CUDA error at %s:%d: %s failed: %s (%s)\n",
                file,line,expr,cudaGetErrorName(err),cudaGetErrorString(err));
        return false;
    }

    bool prepare_rtx5070_launch(int head_dim)
    {
        int device=0;
        if(!cuda_check(cudaGetDevice(&device),"cudaGetDevice(&device)",__FILE__,__LINE__))
        {
            return false;
        }

        cudaDeviceProp prop{};
        if(!cuda_check(cudaGetDeviceProperties(&prop,device),"cudaGetDeviceProperties(&prop,device)",__FILE__,__LINE__))
        {
            return false;
        }

        static bool warned_arch=false;
        if((prop.major!=12 || prop.minor!=0) && !warned_arch)
        {
            fprintf(stderr,
                    "[KV_int8_split_optimized] warning: current GPU is %s with compute capability %d.%d; "
                    "this file is tuned for RTX 5070 / sm_120.\n",
                    prop.name,prop.major,prop.minor);
            warned_arch=true;
        }

        size_t q_smem_bytes=sizeof(float)*static_cast<size_t>(head_dim);
        size_t smem_limit=prop.sharedMemPerBlockOptin>0
            ? static_cast<size_t>(prop.sharedMemPerBlockOptin)
            : static_cast<size_t>(prop.sharedMemPerBlock);

        if(q_smem_bytes>smem_limit)
        {
            fprintf(stderr,
                    "[KV_int8_split_optimized] CUDA launch error: q_smem requires %zu bytes, "
                    "but device shared memory limit per block is %zu bytes.\n",
                    q_smem_bytes,smem_limit);
            return false;
        }

        if(q_smem_bytes>static_cast<size_t>(prop.sharedMemPerBlock))
        {
            if(!cuda_check(
                    cudaFuncSetAttribute(
                        qk_warp_kernel,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(q_smem_bytes)),
                    "cudaFuncSetAttribute(qk_warp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, q_smem_bytes)",
                    __FILE__,
                    __LINE__))
            {
                return false;
            }
        }

        return true;
    }

    __device__ __forceinline__ float warp_sum(float v)
    {
        #pragma unroll
        for(int offset=WARP_SIZE>>1;offset>0;offset>>=1)
        {
            v+=__shfl_down_sync(0xffffffff,v,offset);
        }
        return v;
    }

    __global__ void qk_warp_kernel(
        const float* __restrict__ Q,
        const int8_t* __restrict__ K_int8,
        const float* __restrict__ k_scale,
        float* __restrict__ S,
        int num_heads,
        int seq_len,
        int head_dim,
        float inv_sqrt_dim
    )
    {
        int head=blockIdx.y;
        int warp=threadIdx.x>>5;
        int lane=threadIdx.x&(WARP_SIZE-1);
        int token=blockIdx.x*QK_WARPS_PER_BLOCK+warp;

        extern __shared__ float q_smem[];
        for(int d=threadIdx.x;d<head_dim;d+=blockDim.x)
        {
            q_smem[d]=Q[head*head_dim+d];
        }
        __syncthreads();

        if(head<num_heads && token<seq_len)
        {
            const int8_t* k_row=K_int8+(head*seq_len+token)*head_dim;
            float sum=0.0f;
            for(int d=lane;d<head_dim;d+=WARP_SIZE)
            {
                sum=fmaf(q_smem[d],static_cast<float>(k_row[d]),sum);
            }
            sum=warp_sum(sum);
            if(lane==0)
            {
                S[head*seq_len+token]=sum*k_scale[head*seq_len+token]*inv_sqrt_dim;
            }
        }
    }

    __global__ void softmax_kernel(float* __restrict__ S, int seq_len)
    {
        __shared__ float s_max[SOFTMAX_THREADS];
        __shared__ float s_sum[SOFTMAX_THREADS];

        int head=blockIdx.x;
        int tid=threadIdx.x;
        float local_max=-FLT_MAX;

        for(int i=tid;i<seq_len;i+=blockDim.x)
        {
            local_max=fmaxf(local_max,S[head*seq_len+i]);
        }
        s_max[tid]=local_max;
        __syncthreads();

        for(int offset=blockDim.x>>1;offset>0;offset>>=1)
        {
            if(tid<offset)
            {
                s_max[tid]=fmaxf(s_max[tid],s_max[tid+offset]);
            }
            __syncthreads();
        }

        float row_max=s_max[0];
        float local_sum=0.0f;
        for(int i=tid;i<seq_len;i+=blockDim.x)
        {
            float v=expf(S[head*seq_len+i]-row_max);
            S[head*seq_len+i]=v;
            local_sum+=v;
        }
        s_sum[tid]=local_sum;
        __syncthreads();

        for(int offset=blockDim.x>>1;offset>0;offset>>=1)
        {
            if(tid<offset)
            {
                s_sum[tid]+=s_sum[tid+offset];
            }
            __syncthreads();
        }

        float row_sum=s_sum[0];
        for(int i=tid;i<seq_len;i+=blockDim.x)
        {
            S[head*seq_len+i]/=row_sum;
        }
    }

    __global__ void pv_tiled_kernel(
        const float* __restrict__ S,
        const int8_t* __restrict__ V_int8,
        const float* __restrict__ v_scale,
        float* __restrict__ output,
        int num_heads,
        int seq_len,
        int head_dim
    )
    {
        __shared__ float weight[PV_SEQ_TILE];

        int head=blockIdx.y;
        int dim=blockIdx.x*blockDim.x+threadIdx.x;
        int tid=threadIdx.x;
        float sum=0.0f;

        for(int base=0;base<seq_len;base+=PV_SEQ_TILE)
        {
            int token=base+tid;
            if(tid<PV_SEQ_TILE && token<seq_len)
            {
                int idx=head*seq_len+token;
                weight[tid]=S[idx]*v_scale[idx];
            }
            __syncthreads();

            int tile_count=min(PV_SEQ_TILE,seq_len-base);
            if(head<num_heads && dim<head_dim)
            {
                for(int i=0;i<tile_count;i++)
                {
                    const int8_t* v_row=V_int8+(head*seq_len+base+i)*head_dim;
                    sum=fmaf(weight[i],static_cast<float>(v_row[dim]),sum);
                }
            }
            __syncthreads();
        }

        if(head<num_heads && dim<head_dim)
        {
            output[head*head_dim+dim]=sum;
        }
    }
}

#define CUDA_CHECK_RETURN(expr) \
    do { \
        if(!cuda_check((expr),#expr,__FILE__,__LINE__)) { \
            return; \
        } \
    } while(0)

#if KV_CUDA_DEBUG_SYNC
#define CUDA_KERNEL_CHECK_RETURN() \
    do { \
        CUDA_CHECK_RETURN(cudaGetLastError()); \
        CUDA_CHECK_RETURN(cudaDeviceSynchronize()); \
    } while(0)
#else
#define CUDA_KERNEL_CHECK_RETURN() \
    do { \
        CUDA_CHECK_RETURN(cudaGetLastError()); \
    } while(0)
#endif

// Checked/preallocated asynchronous entry. Same contiguous single-query/head
// layout as solve; the caller owns H*N float scratch and checks completion.
extern "C" cudaError_t kv_int8_split_launch(
    const float* Q, const int8_t* K_int8, const int8_t* V_int8,
    const float* k_scale, const float* v_scale, float* output,
    int num_heads, int seq_len, int head_dim, float* S, cudaStream_t stream) {
    if(!Q||!K_int8||!V_int8||!k_scale||!v_scale||!output||!S||num_heads<=0||seq_len<=0||head_dim<=0)
        return cudaErrorInvalidValue;
    int device=0,limit=0;cudaError_t status=cudaGetDevice(&device);
    if(status!=cudaSuccess)return status;
    status=cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,device);
    if(status!=cudaSuccess)return status;
    const size_t bytes=sizeof(float)*static_cast<size_t>(head_dim);
    if(bytes>static_cast<size_t>(limit))return cudaErrorInvalidValue;
    if(bytes>48*1024){
        status=cudaFuncSetAttribute(qk_warp_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,static_cast<int>(bytes));
        if(status!=cudaSuccess)return status;
    }
    dim3 qk_grid((seq_len+QK_WARPS_PER_BLOCK-1)/QK_WARPS_PER_BLOCK,num_heads);
    qk_warp_kernel<<<qk_grid,QK_THREADS,bytes,stream>>>(Q,K_int8,k_scale,S,num_heads,seq_len,head_dim,1.f/sqrtf(float(head_dim)));
    status=cudaGetLastError();if(status!=cudaSuccess)return status;
    softmax_kernel<<<num_heads,SOFTMAX_THREADS,0,stream>>>(S,seq_len);
    status=cudaGetLastError();if(status!=cudaSuccess)return status;
    dim3 pv_grid((head_dim+PV_THREADS-1)/PV_THREADS,num_heads);
    pv_tiled_kernel<<<pv_grid,PV_THREADS,0,stream>>>(S,V_int8,v_scale,output,num_heads,seq_len,head_dim);
    return cudaGetLastError();
}

extern "C" void solve(
    const float* Q,
    const int8_t* K_int8,
    const int8_t* V_int8,
    const float* k_scale,
    const float* v_scale,
    float* output,
    int num_heads,
    int seq_len,
    int head_dim
)
{
    if(Q==nullptr || K_int8==nullptr || V_int8==nullptr || k_scale==nullptr || v_scale==nullptr || output==nullptr)
    {
        fprintf(stderr,"[KV_int8_split_optimized] null pointer input: Q=%p K_int8=%p V_int8=%p k_scale=%p v_scale=%p output=%p\n",
                static_cast<const void*>(Q),
                static_cast<const void*>(K_int8),
                static_cast<const void*>(V_int8),
                static_cast<const void*>(k_scale),
                static_cast<const void*>(v_scale),
                static_cast<void*>(output));
        return;
    }

    if(num_heads<=0 || seq_len<=0 || head_dim<=0)
    {
        fprintf(stderr,
                "[KV_int8_split_optimized] invalid shape: num_heads=%d, seq_len=%d, head_dim=%d\n",
                num_heads,seq_len,head_dim);
        return;
    }

    if(!prepare_rtx5070_launch(head_dim))
    {
        return;
    }

    float* S=nullptr;
    size_t score_bytes=sizeof(float)*static_cast<size_t>(num_heads)*static_cast<size_t>(seq_len);
    CUDA_CHECK_RETURN(cudaMalloc(reinterpret_cast<void**>(&S),score_bytes));

    float inv_sqrt_dim=1.0f/sqrtf(static_cast<float>(head_dim));
    dim3 qk_grid((seq_len+QK_WARPS_PER_BLOCK-1)/QK_WARPS_PER_BLOCK,num_heads);
    qk_warp_kernel<<<qk_grid,QK_THREADS,sizeof(float)*head_dim>>>(
        Q,K_int8,k_scale,S,num_heads,seq_len,head_dim,inv_sqrt_dim);
    if(!cuda_check(cudaGetLastError(),"qk_warp_kernel launch",__FILE__,__LINE__))
    {
        cuda_check(cudaFree(S),"cudaFree(S)",__FILE__,__LINE__);
        return;
    }
#if KV_CUDA_DEBUG_SYNC
    if(!cuda_check(cudaDeviceSynchronize(),"cudaDeviceSynchronize() after qk_warp_kernel",__FILE__,__LINE__))
    {
        cuda_check(cudaFree(S),"cudaFree(S)",__FILE__,__LINE__);
        return;
    }
#endif

    softmax_kernel<<<num_heads,SOFTMAX_THREADS>>>(S,seq_len);
    if(!cuda_check(cudaGetLastError(),"softmax_kernel launch",__FILE__,__LINE__))
    {
        cuda_check(cudaFree(S),"cudaFree(S)",__FILE__,__LINE__);
        return;
    }
#if KV_CUDA_DEBUG_SYNC
    if(!cuda_check(cudaDeviceSynchronize(),"cudaDeviceSynchronize() after softmax_kernel",__FILE__,__LINE__))
    {
        cuda_check(cudaFree(S),"cudaFree(S)",__FILE__,__LINE__);
        return;
    }
#endif

    dim3 pv_grid((head_dim+PV_THREADS-1)/PV_THREADS,num_heads);
    pv_tiled_kernel<<<pv_grid,PV_THREADS>>>(S,V_int8,v_scale,output,num_heads,seq_len,head_dim);
    if(!cuda_check(cudaGetLastError(),"pv_tiled_kernel launch",__FILE__,__LINE__))
    {
        cuda_check(cudaFree(S),"cudaFree(S)",__FILE__,__LINE__);
        return;
    }
#if KV_CUDA_DEBUG_SYNC
    if(!cuda_check(cudaDeviceSynchronize(),"cudaDeviceSynchronize() after pv_tiled_kernel",__FILE__,__LINE__))
    {
        cuda_check(cudaFree(S),"cudaFree(S)",__FILE__,__LINE__);
        return;
    }
#endif

    CUDA_CHECK_RETURN(cudaFree(S));
}

#ifndef KV_INT8_NO_MAIN
static void cpu_reference(
    const std::vector<float>& Q,
    const std::vector<int8_t>& K_int8,
    const std::vector<int8_t>& V_int8,
    const std::vector<float>& k_scale,
    const std::vector<float>& v_scale,
    std::vector<float>& output,
    int num_heads,
    int seq_len,
    int head_dim
)
{
    std::vector<float> scores(seq_len);
    const float inv_sqrt_dim=1.0f/sqrtf(static_cast<float>(head_dim));

    for(int h=0;h<num_heads;h++)
    {
        float max_score=-FLT_MAX;
        for(int t=0;t<seq_len;t++)
        {
            float dot=0.0f;
            for(int d=0;d<head_dim;d++)
            {
                int q_idx=h*head_dim+d;
                int kv_idx=(h*seq_len+t)*head_dim+d;
                dot=fmaf(Q[q_idx],static_cast<float>(K_int8[kv_idx]),dot);
            }
            float score=dot*k_scale[h*seq_len+t]*inv_sqrt_dim;
            scores[t]=score;
            max_score=fmaxf(max_score,score);
        }

        float denom=0.0f;
        for(int t=0;t<seq_len;t++)
        {
            scores[t]=expf(scores[t]-max_score);
            denom+=scores[t];
        }

        for(int d=0;d<head_dim;d++)
        {
            float sum=0.0f;
            for(int t=0;t<seq_len;t++)
            {
                int kv_idx=(h*seq_len+t)*head_dim+d;
                float prob=scores[t]/denom;
                float v=static_cast<float>(V_int8[kv_idx])*v_scale[h*seq_len+t];
                sum=fmaf(prob,v,sum);
            }
            output[h*head_dim+d]=sum;
        }
    }
}

int main(int argc,char** argv)
{
    int num_heads=4;
    int seq_len=128;
    int head_dim=64;

    if(argc>=4)
    {
        num_heads=std::atoi(argv[1]);
        seq_len=std::atoi(argv[2]);
        head_dim=std::atoi(argv[3]);
    }

    if(num_heads<=0 || seq_len<=0 || head_dim<=0)
    {
        fprintf(stderr,"usage: %s [num_heads seq_len head_dim]\n",argv[0]);
        return 1;
    }

    printf("[KV_int8_split_optimized] test shape: heads=%d seq_len=%d head_dim=%d\n",
           num_heads,seq_len,head_dim);

    size_t q_count=static_cast<size_t>(num_heads)*static_cast<size_t>(head_dim);
    size_t kv_count=static_cast<size_t>(num_heads)*static_cast<size_t>(seq_len)*static_cast<size_t>(head_dim);
    size_t scale_count=static_cast<size_t>(num_heads)*static_cast<size_t>(seq_len);

    std::vector<float> h_Q(q_count);
    std::vector<int8_t> h_K(kv_count);
    std::vector<int8_t> h_V(kv_count);
    std::vector<float> h_ks(scale_count);
    std::vector<float> h_vs(scale_count);
    std::vector<float> h_out(q_count,0.0f);
    std::vector<float> h_ref(q_count,0.0f);

    for(size_t i=0;i<q_count;i++)
    {
        h_Q[i]=static_cast<float>(static_cast<int>(i%17)-8)*0.125f;
    }
    for(size_t i=0;i<kv_count;i++)
    {
        h_K[i]=static_cast<int8_t>(static_cast<int>(i%23)-11);
        h_V[i]=static_cast<int8_t>(static_cast<int>((i*3)%29)-14);
    }
    for(size_t i=0;i<scale_count;i++)
    {
        h_ks[i]=0.015625f+static_cast<float>(i%5)*0.001f;
        h_vs[i]=0.03125f+static_cast<float>(i%7)*0.001f;
    }

    float* d_Q=nullptr;
    int8_t* d_K=nullptr;
    int8_t* d_V=nullptr;
    float* d_ks=nullptr;
    float* d_vs=nullptr;
    float* d_out=nullptr;

    if(!cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Q),q_count*sizeof(float)),"cudaMalloc(d_Q)",__FILE__,__LINE__)) return 1;
    if(!cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_K),kv_count*sizeof(int8_t)),"cudaMalloc(d_K)",__FILE__,__LINE__)) return 1;
    if(!cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_V),kv_count*sizeof(int8_t)),"cudaMalloc(d_V)",__FILE__,__LINE__)) return 1;
    if(!cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ks),scale_count*sizeof(float)),"cudaMalloc(d_ks)",__FILE__,__LINE__)) return 1;
    if(!cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_vs),scale_count*sizeof(float)),"cudaMalloc(d_vs)",__FILE__,__LINE__)) return 1;
    if(!cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_out),q_count*sizeof(float)),"cudaMalloc(d_out)",__FILE__,__LINE__)) return 1;

    if(!cuda_check(cudaMemcpy(d_Q,h_Q.data(),q_count*sizeof(float),cudaMemcpyHostToDevice),"cudaMemcpy(d_Q)",__FILE__,__LINE__)) return 1;
    if(!cuda_check(cudaMemcpy(d_K,h_K.data(),kv_count*sizeof(int8_t),cudaMemcpyHostToDevice),"cudaMemcpy(d_K)",__FILE__,__LINE__)) return 1;
    if(!cuda_check(cudaMemcpy(d_V,h_V.data(),kv_count*sizeof(int8_t),cudaMemcpyHostToDevice),"cudaMemcpy(d_V)",__FILE__,__LINE__)) return 1;
    if(!cuda_check(cudaMemcpy(d_ks,h_ks.data(),scale_count*sizeof(float),cudaMemcpyHostToDevice),"cudaMemcpy(d_ks)",__FILE__,__LINE__)) return 1;
    if(!cuda_check(cudaMemcpy(d_vs,h_vs.data(),scale_count*sizeof(float),cudaMemcpyHostToDevice),"cudaMemcpy(d_vs)",__FILE__,__LINE__)) return 1;

    solve(d_Q,d_K,d_V,d_ks,d_vs,d_out,num_heads,seq_len,head_dim);
    if(!cuda_check(cudaDeviceSynchronize(),"cudaDeviceSynchronize() after solve",__FILE__,__LINE__)) return 1;
    if(!cuda_check(cudaMemcpy(h_out.data(),d_out,q_count*sizeof(float),cudaMemcpyDeviceToHost),"cudaMemcpy(h_out)",__FILE__,__LINE__)) return 1;

    cpu_reference(h_Q,h_K,h_V,h_ks,h_vs,h_ref,num_heads,seq_len,head_dim);

    float max_abs_err=0.0f;
    for(size_t i=0;i<q_count;i++)
    {
        max_abs_err=fmaxf(max_abs_err,fabsf(h_out[i]-h_ref[i]));
    }

    printf("[KV_int8_split_optimized] max_abs_err = %.8g\n",max_abs_err);
    printf("[KV_int8_split_optimized] first outputs:");
    for(size_t i=0;i<q_count && i<8;i++)
    {
        printf(" %.6f",h_out[i]);
    }
    printf("\n");

    cuda_check(cudaFree(d_Q),"cudaFree(d_Q)",__FILE__,__LINE__);
    cuda_check(cudaFree(d_K),"cudaFree(d_K)",__FILE__,__LINE__);
    cuda_check(cudaFree(d_V),"cudaFree(d_V)",__FILE__,__LINE__);
    cuda_check(cudaFree(d_ks),"cudaFree(d_ks)",__FILE__,__LINE__);
    cuda_check(cudaFree(d_vs),"cudaFree(d_vs)",__FILE__,__LINE__);
    cuda_check(cudaFree(d_out),"cudaFree(d_out)",__FILE__,__LINE__);

    if(max_abs_err>1e-4f)
    {
        fprintf(stderr,"[KV_int8_split_optimized] validation failed\n");
        return 1;
    }

    printf("[KV_int8_split_optimized] validation passed\n");
    return 0;
}
#endif

#undef CUDA_KERNEL_CHECK_RETURN
#undef CUDA_CHECK_RETURN
