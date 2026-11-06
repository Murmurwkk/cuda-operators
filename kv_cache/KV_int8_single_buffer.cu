#include<cuda_runtime.h>
#include<algorithm>
#include<cfloat>
#include<stdint.h>

#define TILE_SIZE 128
#define WARP_SIZE 32
#define WARP_POW_BI 5
#define PAD 4

namespace{
    __device__ __forceinline__ void copy_int8_rows_to_padded(
        const int8_t* __restrict__ src_base,
        int8_t* __restrict__ dst_base,
        int rows,
        int head_dim,
        int head_dim_pad,
        int tid,
        int block_threads
    )
    {
        for(int t=tid;t<rows;t+=block_threads)
        {
            const int8_t* src=src_base+t*head_dim;
            int8_t* dst=dst_base+t*head_dim_pad;
            int copied=0;

            uintptr_t src_addr=reinterpret_cast<uintptr_t>(src);
            uintptr_t dst_addr=reinterpret_cast<uintptr_t>(dst);
            if(((src_addr|dst_addr)&3u)==0)
            {
                const int32_t* src4=reinterpret_cast<const int32_t*>(src);
                int32_t* dst4=reinterpret_cast<int32_t*>(dst);
                int words=head_dim>>2;
                for(int j=0;j<words;j++)
                {
                    dst4[j]=src4[j];
                }
                copied=words<<2;
            }

            for(int j=copied;j<head_dim;j++)
            {
                dst[j]=src[j];
            }
        }
    }

    template<int block_dim>
    __global__ void kv(
        const float* __restrict__ Q,
        const int8_t* __restrict__ K_int8,
        const int8_t* __restrict__ V_int8,
        const float* __restrict__ k_scale,
        const float* __restrict__ v_scale,
        float* __restrict__ output,
        int num_heads,
        int seq_len,
        int head_dim
    )
    {
        //init
        const int hid=blockIdx.x;
        const int tid=threadIdx.x;
        const int dim=blockDim.x;
        int lane=tid%WARP_SIZE;
        int warp=tid>>WARP_POW_BI;
        int warp_dim=(head_dim+WARP_SIZE-1)>>WARP_POW_BI;
        int warp_num=block_dim>>WARP_POW_BI;
        int head_dim_pad=head_dim+PAD;

        //single shared memory tile buffer
        extern __shared__ char smem[];
        int tile_int8=TILE_SIZE*head_dim_pad;
        int tile_float=TILE_SIZE;

        int8_t* buffer_k=(int8_t*)smem;
        float* buffer_ks=(float*)(buffer_k+tile_int8);
        int8_t* buffer_v=(int8_t*)(buffer_ks+tile_float);
        float* buffer_vs=(float*)(buffer_v+tile_int8);
        float* buffer_scores=(float*)(buffer_vs+tile_float);
        float* buffer_reduce=(float*)(buffer_scores+tile_float);

        //register
        float q[8];
        float max=-FLT_MAX;
        float sum=0.0f;
        float acc[8];
        int tile_num=(seq_len+TILE_SIZE-1)/TILE_SIZE;
        float d_inv_sqrt=rsqrtf((float)head_dim);

        #pragma unroll
        for(int i=0;i<8;i++)
        {
            acc[i]=0.0f;
        }

        #pragma unroll
        for(int i=0;i<warp_dim;i++)
        {
            int idx=lane+i*WARP_SIZE;
            if(idx<head_dim)
                q[i]=Q[hid*head_dim+idx];
            else
                q[i]=0.0f;
        }

        //load and compute one tile at a time, without prefetching the next tile
        for(int tile=0;tile<tile_num;tile++)
        {
            int tile_start=tile*TILE_SIZE;
            int tile_load=min(TILE_SIZE,seq_len-tile_start);

            const int8_t* k_start=K_int8+hid*seq_len*head_dim+tile_start*head_dim;
            copy_int8_rows_to_padded(k_start,buffer_k,tile_load,head_dim,head_dim_pad,tid,dim);

            const float* ks_start=k_scale+hid*seq_len+tile_start;
            for(int i=tid;i<tile_load;i+=dim)
            {
                buffer_ks[i]=ks_start[i];
            }

            const int8_t* v_start=V_int8+hid*seq_len*head_dim+tile_start*head_dim;
            copy_int8_rows_to_padded(v_start,buffer_v,tile_load,head_dim,head_dim_pad,tid,dim);

            const float* vs_start=v_scale+hid*seq_len+tile_start;
            for(int i=tid;i<tile_load;i+=dim)
            {
                buffer_vs[i]=vs_start[i];
            }
            __syncthreads();

            //mha
            //QK^T
            for(int i=warp;i<tile_load;i+=warp_num)
            {
                float tmp=0.0f;
                #pragma unroll
                for(int j=0;j<warp_dim;j++)
                {
                    int idx=lane+j*WARP_SIZE;
                    if(idx<head_dim)
                    {
                        float k=buffer_k[i*head_dim_pad+idx]*buffer_ks[i];
                        tmp=fmaf(q[j],k,tmp);
                    }
                }
                #pragma unroll
                for(int offset=WARP_SIZE>>1;offset>0;offset>>=1)
                {
                    tmp+=__shfl_xor_sync(0xffffffff,tmp,offset);
                }
                if(lane==0) buffer_scores[i]=tmp*d_inv_sqrt;
            }
            __syncthreads();

            //online softmax
            float tmp_max=-FLT_MAX;
            for(int i=tid;i<tile_load;i+=dim)
            {
                tmp_max=fmaxf(tmp_max,buffer_scores[i]);
            }
            #pragma unroll
            for(int offset=WARP_SIZE>>1;offset>0;offset>>=1)
            {
                tmp_max=fmaxf(tmp_max,__shfl_xor_sync(0xffffffff,tmp_max,offset));
            }
            if(lane==0) buffer_reduce[warp]=tmp_max;
            __syncthreads();

            float tile_max=-FLT_MAX;
            if(warp==0)
            {
                tile_max=(lane<warp_num)?buffer_reduce[lane]:-FLT_MAX;
                #pragma unroll
                for(int offset=WARP_SIZE>>1;offset>0;offset>>=1)
                {
                    tile_max=fmaxf(tile_max,__shfl_xor_sync(0xffffffff,tile_max,offset));
                }
                if(lane==0) buffer_reduce[0]=tile_max;
            }
            __syncthreads();
            tile_max=buffer_reduce[0];

            //V add
            if(warp==0)
            {
                float new_max=fmaxf(max,tile_max);
                float alpha=expf(max-new_max);

                #pragma unroll
                for(int j=0;j<warp_dim;j++)
                {
                    acc[j]*=alpha;
                }
                sum*=alpha;

                for(int i=0;i<tile_load;i++)
                {
                    float w=expf(buffer_scores[i]-new_max);
                    sum+=w;
                    #pragma unroll
                    for(int j=0;j<warp_dim;j++)
                    {
                        int idx=lane+j*WARP_SIZE;
                        if(idx<head_dim)
                        {
                            float v=buffer_v[i*head_dim_pad+idx]*buffer_vs[i];
                            acc[j]=fmaf(w,v,acc[j]);
                        }
                    }
                }
                max=new_max;
            }
            __syncthreads();
        }

        //epilogue
        if(warp==0)
        {
            #pragma unroll
            for(int j=0;j<warp_dim;j++)
            {
                int idx=lane+j*WARP_SIZE;
                if(idx<head_dim) output[hid*head_dim+idx]=acc[j]/sum;
            }
        }
    }

    template<int block_dim>
    void launch_kv(
        dim3 grid,
        dim3 block,
        int smem_size,
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
        cudaFuncSetAttribute(
            kv<block_dim>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size);
        cudaFuncSetAttribute(
            kv<block_dim>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        kv<block_dim><<<grid, block, smem_size>>>(
            Q, K_int8, V_int8, k_scale, v_scale, output,
            num_heads, seq_len, head_dim);
    }
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
    int block_dim=std::min(head_dim,256);
    block_dim=(block_dim+WARP_SIZE-1)/WARP_SIZE*WARP_SIZE;
    if(block_dim<WARP_SIZE) block_dim=WARP_SIZE;
    const int grid_dim=num_heads;
    const int warp_num=block_dim>>5;
    dim3 grid(grid_dim);
    dim3 block(block_dim);

    const int head_dim_pad=head_dim+PAD;
    const int tile_buffer_size=head_dim_pad*TILE_SIZE;
    const int scale_buffer_size=sizeof(float)*TILE_SIZE;
    const int float_size=sizeof(float)*TILE_SIZE;
    const int k_tile=tile_buffer_size;
    const int v_tile=tile_buffer_size;
    const int k_sc=scale_buffer_size;
    const int v_sc=scale_buffer_size;
    const int scores=float_size;
    const int reduce=sizeof(float)*warp_num;
    const int smem_size=k_tile+v_tile+k_sc+v_sc+scores+reduce;

    switch (block_dim) {
    case 32:
        launch_kv<32>(grid, block, smem_size,
            Q, K_int8, V_int8, k_scale, v_scale, output,
            num_heads, seq_len, head_dim);
        break;
    case 64:
        launch_kv<64>(grid, block, smem_size,
            Q, K_int8, V_int8, k_scale, v_scale, output,
            num_heads, seq_len, head_dim);
        break;
    case 96:
        launch_kv<96>(grid, block, smem_size,
            Q, K_int8, V_int8, k_scale, v_scale, output,
            num_heads, seq_len, head_dim);
        break;
    case 128:
        launch_kv<128>(grid, block, smem_size,
            Q, K_int8, V_int8, k_scale, v_scale, output,
            num_heads, seq_len, head_dim);
        break;
    case 160:
        launch_kv<160>(grid, block, smem_size,
            Q, K_int8, V_int8, k_scale, v_scale, output,
            num_heads, seq_len, head_dim);
        break;
    case 192:
        launch_kv<192>(grid, block, smem_size,
            Q, K_int8, V_int8, k_scale, v_scale, output,
            num_heads, seq_len, head_dim);
        break;
    case 224:
        launch_kv<224>(grid, block, smem_size,
            Q, K_int8, V_int8, k_scale, v_scale, output,
            num_heads, seq_len, head_dim);
        break;
    case 256:
        launch_kv<256>(grid, block, smem_size,
            Q, K_int8, V_int8, k_scale, v_scale, output,
            num_heads, seq_len, head_dim);
        break;
    }

    //cudaDeviceSynchronize();
}
