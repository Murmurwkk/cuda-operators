# CUDA Operators

Hand-written CUDA kernels for the building blocks of LLM training and inference: GEMM, attention, INT8 KV-cache attention, selective scan and gated activations. Most of them are the kernels behind a write-up on [murmurwkk.github.io](https://murmurwkk.github.io/blog/), where the measurements, profiler evidence and correctness checks are explained.

Kernels are FP32 unless noted (the tensor-core GEMMs use TF32), and were developed and measured on an RTX 5070 Laptop GPU (SM 120) with CUDA 13.1.

## What's here

| Folder | Kernels | Write-up |
|---|---|---|
| [`gemm/`](gemm/) | Shared-memory tiled GEMM; a 128×64-tile SIMT GEMM with an XOR-swizzled layout, double buffering and `cp.async`; scalar, `float4` and CUB loaders; WMMA tensor-core GEMMs | [Where a Naive CUDA GEMM Loses 5× to cuBLAS](https://murmurwkk.github.io/blog/gemm-reuse-ldst/) · [When Removing Bank Conflicts Made My GEMM Slower](https://murmurwkk.github.io/blog/cuda-double-buffering/) · [Where TF32 Loses Precision, and What It Buys Back](https://murmurwkk.github.io/blog/tf32-precision-tensor-cores/) |
| [`attention/`](attention/) | Row softmax, online-softmax single-head attention without a score matrix, fused multi-head attention | [Online Softmax in CUDA: No Score Matrix, More Time](https://murmurwkk.github.io/blog/online-softmax/) · [CUDA Graphs Save Launches, Not Work](https://murmurwkk.github.io/blog/cuda-graph-runtime/) |
| [`kv_cache/`](kv_cache/) | Decode attention over an INT8 K/V cache with per-token scales: tiled, single-buffer and split QK/PV variants | [What Quantizing the KV Cache Costs in Speed and Accuracy](https://murmurwkk.github.io/blog/kv-cache-scheduling/) |
| [`selective_scan/`](selective_scan/) | Mamba-style selective scan: a sequential baseline, a CTA-per-stream rewrite, and sequence-parallel versions built on composing affine transforms | [Serial and Parallel Selective-Scan Kernels Compared on a Laptop GPU](https://murmurwkk.github.io/blog/selective-scan/) |
| [`activations/`](activations/) | Elementwise SwiGLU (overflow-safe sigmoid) and GEGLU (erf-based GELU) | [A Custom SwiGLU Op in PyTorch: Everything but the Kernel](https://murmurwkk.github.io/blog/pytorch-custom-operator-contracts/) |
| [`common/`](common/) | `ordered_workspace.cuh`: a scratch-buffer owner that keeps asynchronous uses on one stream | [CUDA Graphs Save Launches, Not Work](https://murmurwkk.github.io/blog/cuda-graph-runtime/) |

Some folders keep several versions of the same operator side by side, because the write-ups compare them. Each file's header comment says what it assumes about layout, alignment and lifetimes.

## Building

Each `.cu` file is its own translation unit. Most expose an `extern "C"` entry point (`solve(...)`, and in the newer kernels a stream-aware launch function); `gemm/matrixmul_tile.cu` and `kv_cache/KV_int8_split_optimized.cu` also contain a `main`. Headers sit next to the files that include them. For example:

```sh
nvcc -O3 -arch=sm_120 -c gemm/gemm_cub.cu
```

Change `-arch` to match your GPU.

## Reproducing the measurements

- [`benchmarks/selective_scan/`](benchmarks/selective_scan/): an independent FP64
  reference, correctness checks, build and run commands, per-run timing records,
  initialization-synchronization regression, workspace ablation and chain-count
  sweep. The runner works from this public checkout.
- [`benchmarks/parallel_scaling/`](benchmarks/parallel_scaling/): ParScale subset
  scoring, explicit cached-decode timing, numerical results and plotting scripts.
  Download the named public checkpoints separately; model weights are not included.

Each package records the conditions and source hashes used for its results.
Timings depend on the GPU, software and workload; read the package's timing
definition before comparing numbers. Historical unsafe code, where included for
regression comparison, is labelled and is not the default implementation.

## License

No license is granted: the code is published to be read, and all rights are reserved by Murmurwkk. Please ask before reusing it.
