#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#define ll long long
#define tile_size 32

__global__ void matrixmul(float *a, float *b, float *c, int m, int k, int n)
{
    __shared__ float sa[tile_size][tile_size];
    __shared__ float sb_trans[tile_size][tile_size];
    int row = blockIdx.y * tile_size + threadIdx.y;
    int col = blockIdx.x * tile_size + threadIdx.x;
    float sum = 0.0f;
    for(int idx=0;idx<(k+tile_size-1)/tile_size;idx++)
    {
        if(row < m && idx * tile_size + threadIdx.x < k)
            sa[threadIdx.y][threadIdx.x] = a[row * k + idx * tile_size + threadIdx.x];
        else
            sa[threadIdx.y][threadIdx.x] = 0.0f;
        if(col < n && idx * tile_size + threadIdx.y < k)
            sb_trans[threadIdx.x][threadIdx.y] = b[(idx * tile_size + threadIdx.y) * n + col];
        else
            sb_trans[threadIdx.x][threadIdx.y] = 0.0f;
        __syncthreads();
        for(int j=0;j<tile_size;j++)
            sum += sa[threadIdx.y][j] * sb_trans[threadIdx.x][j];
        __syncthreads();
    }
    if(row < m &&  col < n)
        c[row * n + col] = sum;
}

int main()
{
    int m ,n ,k;
    std::cin >> m >> k >> n;
    dim3 threadsPerBlock(tile_size, tile_size);
    dim3 numBlocks((n + tile_size - 1) / tile_size, (m + tile_size - 1) / tile_size);

    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    a = (float *)malloc(m * k * sizeof(float));
    b = (float *)malloc(k * n * sizeof(float));
    c = (float *)malloc(m * n * sizeof(float));
    cudaMalloc(&d_a, m * k * sizeof(float));
    cudaMalloc(&d_b, k * n * sizeof(float));
    cudaMalloc(&d_c, m * n * sizeof(float));
    for(int i = 0; i < m * k; i++)
        a[i]=(i%m)+(i/m)+2;
    for(int i = 0; i < k * n; i++)
        b[i]=(i%n)+(i/n)+2;
    cudaMemcpy(d_a,a,sizeof(float)*m*k,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,sizeof(float)*k*n,cudaMemcpyHostToDevice);
    matrixmul<<<numBlocks, threadsPerBlock>>>(d_a,d_b,d_c,m,k,n);
    cudaMemcpy(c,d_c,sizeof(float)*m*n,cudaMemcpyDeviceToHost);
    for(int i = 0; i < m * n; i++)
        std::cout << c[i] << " ";   
    free(a);
    free(b);    
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}