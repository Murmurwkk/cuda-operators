// Independent FP64 oracle, checked before timing. No external data is required.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void solve(const float*, const float*, const float*, const float*,
                      const float*, const float*, float*, int, int, int, int);
#ifdef USE_WORKSPACE_ADAPTER
extern "C" void bench_prepare(int, int, int, int);
extern "C" void bench_release();
#endif

static void check(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(status));
        std::exit(2);
    }
}

struct Errors { double absolute = 0, normalized = 0; size_t bad = 0; };
static Errors compare(const std::vector<float>& actual, const std::vector<double>& ref,
                      double atol, double rtol) {
    Errors e;
    if (actual.size() != ref.size()) throw std::runtime_error("output shape mismatch");
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(ref[i])) { ++e.bad; continue; }
        const double delta = std::fabs(double(actual[i]) - ref[i]);
        e.absolute = std::max(e.absolute, delta);
        e.normalized = std::max(e.normalized, delta / (atol + rtol * std::fabs(ref[i])));
        if (delta > atol + rtol * std::fabs(ref[i])) ++e.bad;
    }
    return e;
}

static void self_test() {
    const std::vector<double> ref{0.0, 1.0};
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    if (compare({0.0f, 1.0f}, ref, 2e-5, 2e-4).bad ||
        !compare({nan, 1.0f}, ref, 2e-5, 2e-4).bad ||
        !compare({inf, 1.0f}, ref, 2e-5, 2e-4).bad ||
        !compare({0.0f, 1.1f}, ref, 2e-5, 2e-4).bad)
        throw std::runtime_error("checker self-test failed");
    std::puts("{\"checker_self_test\":true}");
}

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--self-test") { self_test(); return 0; }
    if (argc < 5) throw std::runtime_error("usage: bench B L D N [--seed S] [--iters I] [--warmup W] [--check-only]");
    const int batch = std::stoi(argv[1]), len = std::stoi(argv[2]);
    const int dim = std::stoi(argv[3]), state = std::stoi(argv[4]);
    int seed = 17, iters = 30, warmup = 6;
    bool check_only = false;
    for (int i = 5; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--check-only") check_only = true;
      else if (i + 1 < argc && arg == "--seed") seed = std::stoi(argv[++i]);
      else if (i + 1 < argc && arg == "--iters") iters = std::stoi(argv[++i]);
      else if (i + 1 < argc && arg == "--warmup") warmup = std::stoi(argv[++i]);
      else throw std::runtime_error("unknown or incomplete option");
    }
    if (batch <= 0 || len <= 0 || dim <= 0 || state <= 0 || iters <= 0 || warmup < 0)
      throw std::runtime_error("all dimensions and iters must be positive");
    const size_t nu = size_t(batch) * len * dim, nbc = size_t(batch) * len * state;
    const size_t na = size_t(dim) * state;
    if (nu * size_t(state) > 2000000000ULL) throw std::runtime_error("oracle work limit exceeded; no unchecked result is emitted");
    std::vector<float> hu(nu), hd(nu), hA(na), hB(nbc), hC(nbc), hs(dim);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& x : hu) x = dist(rng);
    for (auto& x : hd) x = std::fabs(dist(rng)) * 0.1f + 0.01f;
    for (auto& x : hA) x = -std::fabs(dist(rng)) - 0.1f;
    for (auto& x : hB) x = dist(rng);
    for (auto& x : hC) x = dist(rng);
    for (auto& x : hs) x = dist(rng);
    std::vector<double> ref(nu);
    for (int b = 0; b < batch; ++b) for (int d = 0; d < dim; ++d) {
      std::vector<double> h(state, 0.0);
      for (int t = 0; t < len; ++t) {
        const size_t i = (size_t(b) * len + t) * dim + d;
        double y = double(hs[d]) * hu[i];
        for (int n = 0; n < state; ++n) {
          const size_t j = (size_t(b) * len + t) * state + n;
          h[n] = std::exp(double(hd[i]) * hA[size_t(d) * state + n]) * h[n]
               + double(hd[i]) * hB[j] * hu[i];
          y += h[n] * hC[j];
        }
        ref[i] = y;
      }
    }
    float *du, *dd, *dA, *dB, *dC, *ds, *dy;
    check(cudaMalloc(&du, nu * sizeof(float)), "malloc u");
    check(cudaMalloc(&dd, nu * sizeof(float)), "malloc delta");
    check(cudaMalloc(&dA, na * sizeof(float)), "malloc A");
    check(cudaMalloc(&dB, nbc * sizeof(float)), "malloc B");
    check(cudaMalloc(&dC, nbc * sizeof(float)), "malloc C");
    check(cudaMalloc(&ds, size_t(dim) * sizeof(float)), "malloc skip");
    check(cudaMalloc(&dy, nu * sizeof(float)), "malloc output");
    check(cudaMemcpy(du, hu.data(), nu * 4, cudaMemcpyHostToDevice), "copy u");
    check(cudaMemcpy(dd, hd.data(), nu * 4, cudaMemcpyHostToDevice), "copy delta");
    check(cudaMemcpy(dA, hA.data(), na * 4, cudaMemcpyHostToDevice), "copy A");
    check(cudaMemcpy(dB, hB.data(), nbc * 4, cudaMemcpyHostToDevice), "copy B");
    check(cudaMemcpy(dC, hC.data(), nbc * 4, cudaMemcpyHostToDevice), "copy C");
    check(cudaMemcpy(ds, hs.data(), size_t(dim) * 4, cudaMemcpyHostToDevice), "copy skip");
#ifdef USE_WORKSPACE_ADAPTER
    bench_prepare(batch, len, dim, state);
#endif
    auto launch = [&]() { solve(du, dd, dA, dB, dC, ds, dy, batch, len, dim, state); check(cudaGetLastError(), "launch"); };
    Errors errors;
    // Poison output before both checks, so a missed write cannot reuse a previous answer.
    for (int repeat = 0; repeat < 2; ++repeat) {
      check(cudaMemset(dy, 0xff, nu * 4), "poison y");
      launch(); check(cudaDeviceSynchronize(), "correctness solve");
      std::vector<float> actual(nu);
      check(cudaMemcpy(actual.data(), dy, nu * 4, cudaMemcpyDeviceToHost), "copy result");
      const Errors e = compare(actual, ref, 2e-5, 2e-4);
      errors.absolute = std::max(errors.absolute, e.absolute);
      errors.normalized = std::max(errors.normalized, e.normalized);
      errors.bad += e.bad;
    }
    if (errors.bad) {
      std::fprintf(stderr, "correctness failed: %zu entries; abs=%g normalized=%g\n", errors.bad, errors.absolute, errors.normalized);
      return 3;
    }
    double ms = 0.0, wall_ms = 0.0;
    if (!check_only) {
      for (int i = 0; i < warmup; ++i) launch();
      check(cudaDeviceSynchronize(), "warmup");
      cudaEvent_t start, end;
      check(cudaEventCreate(&start), "create event"); check(cudaEventCreate(&end), "create event");
      const auto wall_start = std::chrono::steady_clock::now();
      check(cudaEventRecord(start), "start event");
      for (int i = 0; i < iters; ++i) launch();
      check(cudaEventRecord(end), "end event"); check(cudaEventSynchronize(end), "wait event");
      const auto wall_end = std::chrono::steady_clock::now();
      float total = 0;
      check(cudaEventElapsedTime(&total, start, end), "elapsed");
      ms = total / iters;
      wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count() / iters;
      check(cudaEventDestroy(start), "destroy event"); check(cudaEventDestroy(end), "destroy event");
    }
    std::printf("{\"B\":%d,\"L\":%d,\"D\":%d,\"N\":%d,\"seed\":%d,\"iters\":%d,\"warmup\":%d,\"ms\":%.9f,\"wall_ms\":%.9f,\"max_abs_error\":%.12g,\"max_normalized_error\":%.12g,\"atol\":0.00002,\"rtol\":0.0002,\"correct\":true,\"check_only\":%s}\n",
                batch,len,dim,state,seed,iters,warmup,ms,wall_ms,errors.absolute,errors.normalized,check_only?"true":"false");
#ifdef USE_WORKSPACE_ADAPTER
    bench_release();
#endif
    cudaFree(du); cudaFree(dd); cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(ds); cudaFree(dy);
    return 0;
  } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
