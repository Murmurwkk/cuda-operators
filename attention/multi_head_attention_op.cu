#include <cuda_runtime.h>
#include <iostream>

static __device__ __forceinline__ float warp_reduce_max(float x){
#pragma unroll
    for(int i = 16; i >= 1; i /= 2)
        x = max(x, __shfl_xor_sync(0xffffffff, x, i));
    return x;
}

static __device__ __forceinline__ float warp_reduce_sum(float x){
#pragma unroll
    for(int i = 16; i >= 1; i /= 2)
        x += __shfl_xor_sync(0xffffffff, x, i);
    return x;
}

__global__ void mha_simple(const float* Q, const float* K, const float* V, float* output, int M, int N, int d, int dk){
    int head = blockIdx.x;
    int mi = blockIdx.y * blockDim.x + threadIdx.x;
    if(mi >= M)
        return;

    float maxi = -1e18, sumi = 0;
    float scale = sqrt((float)dk);
    const float* q_row = Q + mi * d + head * dk;
    float* o_row = output + mi * d + head * dk;
    for(int i = 0; i < dk; i++)
        o_row[i] = 0;
    for(int ni = 0; ni < N; ni++){
        const float* k_row = K + ni * d + head * dk;
        float s = 0;
        for(int i = 0; i < dk; i++)
            s += q_row[i] * k_row[i];
        s /= scale;

        float m = max(maxi, s);
        sumi = sumi * exp(maxi - m) + exp(s - m);
        maxi = m;
    }
    for(int ni = 0; ni < N; ni++){
        const float* k_row = K + ni * d + head * dk;
        const float* v_row = V + ni * d + head * dk;
        float s = 0;
        for(int i = 0; i < dk; i++)
            s += q_row[i] * k_row[i];
        s /= scale;

        float x = exp(s - maxi) / sumi;
        for(int i = 0; i < dk; i++)
            o_row[i] += x * v_row[i];
    }
}

template<int M, int N, int BSIZE>
static __device__ __forceinline__ void clearmat(float* dst){
#pragma unroll
    for(int i = 0; i < M * N; i += BSIZE * 4){
        const int index = i + threadIdx.x * 4;
        float4* d = (float4*)(dst + index);
        *d = make_float4(0, 0, 0, 0);
    }
}

template<int TILE_M, int TILE_N, int BSIZE>
static __device__ __forceinline__ void ldmat(float* dst, const float* src, int stride, int size_m, int size_n){
#pragma unroll
    for(int i = 0; i < TILE_M * TILE_N; i += BSIZE * 4){
        const int index = i + threadIdx.x * 4;
        const int x = index % TILE_N;
        const int y = index / TILE_N;
        float4* d = (float4*)(dst + index);
        
        if(x < size_n && y < size_m){
            *d = *(const float4*)(src + y * stride + x);
        }else 
            *d = make_float4(0.0, 0.0, 0.0, 0.0);
    }
}

template<int TILE_M, int TILE_N, int BSIZE>
static __device__ __forceinline__ void stmat(float* dst, const float* src, int stride, int size_m, int size_n){
#pragma unroll
    for(int i = 0; i < TILE_M * TILE_N; i += BSIZE * 4){
        const int index = i + threadIdx.x * 4;
        const int x = index % TILE_N;
        const int y = index / TILE_N;
        const float4* s = (const float4*)(src + index);
        float4* d = (float4*)(dst + y * stride + x);
        
        if(x < size_n && y < size_m)
            *d = *s;
    }
}

template<int M, int N, int K>
static __device__ __forceinline__ void mma_trans(const float* A, const float* B, float* C){
    // A: M*N, B: K*N, C: M*K
#pragma unroll
    for(int mi = 0; mi < M; mi++){
#pragma unroll
        for(int ki = 0; ki < K; ki++){
#pragma unroll
            for(int ni = 0; ni < N; ni++){
                C[mi * K + ki] += A[mi * N + ni] * B[ki * N + ni];
            }
        }
    }
}

template<int M, int N, int K>
static __device__ __forceinline__ void mma(const float* A, const float* B, float* C){
    // A: M*N, B:N*K, C:M*K
#pragma unroll
    for(int mi = 0; mi < M; mi++){
#pragma unroll
        for(int ni = 0; ni < N; ni++){
#pragma unroll
            for(int ki = 0; ki < K; ki++)    
                C[mi * K + ki] += A[mi * N + ni] * B[ni * K + ki];
        }
    }
}

template<int m, int n, int M, int N>
static __device__ __forceinline__ void ldmat_reg(float* dst, const float* src){
#pragma unroll
    for(int i = 0; i < m; i++){
#pragma unroll
        for(int j = 0; j < n; j++)
            dst[i * n + j] = src[i * N + j];
    }
}

template<int TILE_M, int TILE_N, int TILE_D, int SMALL_TILE_M, int SMALL_TILE_N, int SMALL_TILE_D, int BSIZE>
__global__ void mha(const float* Q, const float* K, const float* V, float* output, int M, int N, int d, int dk){
    __shared__ float cached_k[TILE_N][TILE_D];  // also cached v
    __shared__ float cached_s[TILE_M][TILE_N];
    __shared__ float cached_o[TILE_M][TILE_D];

    extern __shared__ float exts[];
    float* cached_q = exts;
    float* gmax = cached_q + TILE_M * TILE_D;
    float* gsum = gmax + TILE_M;
    float* delta = gsum + TILE_M;
    float* delta_b = delta + TILE_M;
#define cached_v cached_k

    const int head = blockIdx.x;
    const int start_m = blockIdx.y * TILE_M;
    const int size_m = min(TILE_M, M - start_m);
    const float scale = sqrtf((float)dk);
    const int warp_id = threadIdx.x / 32;
    const int id_in_warp = threadIdx.x % 32;
    const int warps = blockDim.x / 32;
    const int gsize_d = min(TILE_D, dk);

// init
    for(int i = threadIdx.x; i < TILE_M; i += blockDim.x){
        gmax[i] = -1e18;
        gsum[i] = 0;
    }
    clearmat<TILE_M, TILE_D, BSIZE>(&cached_o[0][0]);
    ldmat<TILE_M, TILE_D, BSIZE>(&cached_q[0], Q + start_m * d + head * dk, d, size_m, gsize_d);

    for(int start_n = 0; start_n < N; start_n += TILE_N){
        const int size_n = min(TILE_N, N - start_n);
        const int ssn = (threadIdx.x % (TILE_N / SMALL_TILE_N)) * SMALL_TILE_N;
        const int ssm = (threadIdx.x / (TILE_N / SMALL_TILE_N)) * SMALL_TILE_M;
        const int ssd_sv = (threadIdx.x % (TILE_D / SMALL_TILE_D)) * SMALL_TILE_D;
        const int ssm_sv = (threadIdx.x / (TILE_D / SMALL_TILE_D)) * SMALL_TILE_M;

        float reg_s[SMALL_TILE_M * SMALL_TILE_N] = {0.0};
        float reg_q[SMALL_TILE_M * SMALL_TILE_D];
        float reg_k[SMALL_TILE_N * SMALL_TILE_D];

        // 1. calc QK^T
        for(int start_d = 0; start_d < dk; start_d += TILE_D){
            const int size_d = min(TILE_D, dk - start_d);
            ldmat<TILE_N, TILE_D, BSIZE>(&cached_k[0][0], K + start_n * d + head * dk + start_d, d, size_n, size_d);
            __syncthreads();

#pragma unroll
            for(int di = 0; di < TILE_D; di += SMALL_TILE_D){
                ldmat_reg<SMALL_TILE_M, SMALL_TILE_D, TILE_M, TILE_D>(reg_q, &cached_q[ssm * TILE_D + di]);
                ldmat_reg<SMALL_TILE_N, SMALL_TILE_D, TILE_N, TILE_D>(reg_k, &cached_k[ssn][di]);
                mma_trans<SMALL_TILE_M, SMALL_TILE_D, SMALL_TILE_N>(reg_q, reg_k, reg_s);
            }
            __syncthreads();
        }

        // 2. calc softmax
#pragma unroll
        for(int mi = 0; mi < SMALL_TILE_M; mi++){
#pragma unroll
            for(int ni = 0; ni < SMALL_TILE_N; ni++){
                float s = reg_s[mi * SMALL_TILE_N + ni] / scale;
                cached_s[ssm + mi][ssn + ni] = s;
            }
        }
        __syncthreads();

        for(int mi = warp_id; mi < TILE_M; mi += warps){
            float mx = -1e18;
            float s = 0;
            for(int ni = id_in_warp; ni < size_n; ni += 32)
                mx = max(mx, cached_s[mi][ni]);
            mx = warp_reduce_max(mx);
            for(int ni = id_in_warp; ni < size_n; ni += 32){
                float x = exp(cached_s[mi][ni] - mx);
                s += x;
                cached_s[mi][ni] = x;
            }
            s = warp_reduce_sum(s);
            for(int ni = id_in_warp; ni < size_n; ni += 32){
                cached_s[mi][ni] /= s;
            }
            if(id_in_warp == 0){
                float os = gsum[mi];
                float rm = max(mx, gmax[mi]);
                float d = exp(gmax[mi] - rm);
                float db = exp(mx - rm);
                float rs = os * d + s * db;
                delta[mi] = (start_n == 0) ? 0 : (os / rs) * d;
                delta_b[mi] = (start_n == 0) ? 1 : (s / rs) * db;
                gsum[mi] = rs;
                gmax[mi] = rm; 
            }
        }
        __syncthreads();

        // 3. calc SV
        for(int start_d = 0; start_d < dk; start_d += TILE_D){
            const int size_d = min(TILE_D, dk - start_d);
            ldmat<TILE_N, TILE_D, BSIZE>(&cached_v[0][0], V + start_n * d + head * dk + start_d, d, size_n, size_d);
            __syncthreads();

            float reg_o[SMALL_TILE_M * SMALL_TILE_D] = {0.0};
            float reg_v[SMALL_TILE_N * SMALL_TILE_D];

#pragma unroll
            for(int ni = 0; ni < TILE_N; ni += SMALL_TILE_N){
                ldmat_reg<SMALL_TILE_N, SMALL_TILE_D, TILE_N, TILE_D>(reg_v, &cached_v[ni][ssd_sv]);
                ldmat_reg<SMALL_TILE_M, SMALL_TILE_N, TILE_M, TILE_N>(reg_s, &cached_s[ssm_sv][ni]);
                mma<SMALL_TILE_M, SMALL_TILE_N, SMALL_TILE_D>(reg_s, reg_v, reg_o);
            }

#pragma unroll
            for(int mi = 0; mi < SMALL_TILE_M; mi++){
#pragma unroll
                for(int di = 0; di < SMALL_TILE_D; di++){
                    cached_o[ssm_sv + mi][ssd_sv + di] = 
                        cached_o[ssm_sv + mi][ssd_sv + di] * delta[ssm_sv + mi] + reg_o[mi * SMALL_TILE_D + di] * delta_b[ssm_sv + mi];
                }
            }
            __syncthreads();
        }
    }

    stmat<TILE_M, TILE_D, BSIZE>(output + start_m * d + head * dk, &cached_o[0][0], d, size_m, gsize_d);
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N, int d_model, int h) {
    int max_shared = 70 * 1024;
    constexpr int TILE_N = 64;
    constexpr int TILE_M = 64;
    constexpr int TILE_D = TILE_N;
    constexpr int SMALL_TILE_N = 4;
    constexpr int SMALL_TILE_M = 4;
    constexpr int SMALL_TILE_D = SMALL_TILE_N;
    constexpr int THREADS = (TILE_M / SMALL_TILE_M) * (TILE_N / SMALL_TILE_N);
    CUDA_CHECK(cudaFuncSetAttribute((void*)mha<TILE_M, TILE_N, TILE_D, SMALL_TILE_M, SMALL_TILE_N, SMALL_TILE_D, THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared)); 
    
    const int dk = d_model / h;

    if(dk > 64)
        return;

    if(dk % 4 == 0){
        dim3 blocks(h, (N + TILE_M - 1) / TILE_M);
        mha<TILE_M, TILE_N, TILE_D, SMALL_TILE_M, SMALL_TILE_N, SMALL_TILE_D, THREADS>
            <<<blocks, THREADS, sizeof(float) * (4 * TILE_M + TILE_M * TILE_D + TILE_N * TILE_D + TILE_M * TILE_N)>>>(Q, K, V, output, N, N, d_model, dk);
    }else{
        int threads = 256;
        dim3 blocks(h, (N + threads - 1) / threads);
        mha_simple<<<blocks, threads>>>(Q, K, V, output, N, N, d_model, dk);
    }
    cudaDeviceSynchronize();
}