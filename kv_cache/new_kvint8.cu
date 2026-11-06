#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

__global__ void qk_kernel(const float* Q, float* S, const int8_t* K_int8, const float* k_scale, 
                          int num_heads, int seq_len, int head_dim){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(row<num_heads && col<seq_len){
        float sum = 0.0f;
        for(int i=0;i<head_dim;i++){
            float K = k_scale[row * seq_len + col] * static_cast<float>(K_int8[(row * seq_len + col) * head_dim + i]);
            sum += Q[row * head_dim + i] * K;
        }
        S[row * seq_len + col] = sum / sqrtf(static_cast<float>(head_dim));
    }
}

__global__ void softmax_kernel(float* S, int seq_len){
    __shared__ float sdata_max[256];
    __shared__ float sdata_sum[256];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    float temp_max = -FLT_MAX;
    float temp_sum = 0.0f;
    for(int i = tid; i < seq_len; i += stride){
        if(i < seq_len){
            temp_max = fmaxf(temp_max, S[bid * seq_len + i]);       
        }
    }
    sdata_max[tid] = temp_max;
    __syncthreads();

    for(int offset = blockDim.x /2; offset >0; offset >>= 1){
        if(tid < offset){
            sdata_max[tid] = fmaxf(sdata_max[tid], sdata_max[tid + offset]);
        }
        __syncthreads();
    }

    for(int i = tid; i < seq_len; i += stride){
        if(i < seq_len){
            S[bid * seq_len + i] = expf(S[bid * seq_len + i] - sdata_max[0]);
            temp_sum += S[bid * seq_len + i];
        }
    }
    sdata_sum[tid] = temp_sum;
    __syncthreads();

    for(int offset = blockDim.x /2; offset > 0; offset >>= 1){
        if(tid < offset){
            sdata_sum[tid] += sdata_sum[tid + offset];
        }
        __syncthreads();
    }

    for(int i = tid; i < seq_len; i += stride){
        if(i < seq_len){
            S[bid * seq_len + i] = S[bid * seq_len + i] / sdata_sum[0];
        }
    }

}

__global__ void pv_kernel(const float* S, float* output, const int8_t* V_int8, const float* v_scale, 
                          int num_heads, int seq_len, int head_dim){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(row<num_heads && col<head_dim){
        float sum = 0.0f;
        for(int i=0;i<seq_len;i++){
            float V = v_scale[row * seq_len + i] * static_cast<float>(V_int8[(row * seq_len + i)* head_dim + col]);
            sum += S[row * seq_len + i] * V;
        }
        output[row * head_dim + col] = sum;
    }
}


// One query per head; contiguous [heads, sequence, dimension] int8 cache with
// one FP32 scale per token/head. No mask, paging, GQA mapping or quantization.
extern "C" cudaError_t kv_int8_simple_launch(
    const float* Q, const int8_t* K_int8, const int8_t* V_int8,
    const float* k_scale, const float* v_scale, float* output,
    int num_heads, int seq_len, int head_dim, float* S, cudaStream_t stream) {
    if(!Q||!K_int8||!V_int8||!k_scale||!v_scale||!output||!S||num_heads<=0||seq_len<=0||head_dim<=0)
        return cudaErrorInvalidValue;
    dim3 block(16,16);
    dim3 qk_grid((seq_len+15)/16,(num_heads+15)/16);
    dim3 pv_grid((head_dim+15)/16,(num_heads+15)/16);
    qk_kernel<<<qk_grid,block,0,stream>>>(Q,S,K_int8,k_scale,num_heads,seq_len,head_dim);
    cudaError_t status=cudaGetLastError();if(status!=cudaSuccess)return status;
    softmax_kernel<<<num_heads,256,0,stream>>>(S,seq_len);
    status=cudaGetLastError();if(status!=cudaSuccess)return status;
    pv_kernel<<<pv_grid,block,0,stream>>>(S,output,V_int8,v_scale,num_heads,seq_len,head_dim);
    return cudaGetLastError();
}

extern "C" void solve(const float* Q, const int8_t* K_int8, const int8_t* V_int8,
                      const float* k_scale, const float* v_scale, float* output, int num_heads,
                      int seq_len, int head_dim) {
    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid_qk((seq_len + 15) / 16, (num_heads + 15) / 16);
    dim3 blocksPerGrid_pv((head_dim + 15) / 16, (num_heads + 15) / 16);

    float* S;
    cudaMalloc(&S, sizeof(float) * num_heads * seq_len);

    qk_kernel<<<blocksPerGrid_qk,threadsPerBlock>>>(Q, S, K_int8, k_scale, num_heads, seq_len, head_dim);

    softmax_kernel<<<num_heads,256>>>(S, seq_len);

    pv_kernel<<<blocksPerGrid_pv, threadsPerBlock>>>(S, output, V_int8, v_scale, num_heads, seq_len, head_dim);

    cudaFree(S);
}
