// expf + SwiGLU parity: does CUDA reproduce the CPU's bits?
// The pool evaluates SwiGLU as (g / (1.f + std::exp(-g))) * u between the two expert products. If CUDA's expf
// disagrees by one ulp anywhere, a "bit-exact CUDA port" of the expert dot is not bit-exact end to end.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

__global__ void k_exp(const float* x, float* y, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = expf(x[i]);
}
__global__ void k_swiglu(const float* g, const float* u, float* y, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { const float gv = g[i]; y[i] = (gv / (1.f + expf(-gv))) * u[i]; }
}
static bool same_bits(float a, float b) { uint32_t x, y; std::memcpy(&x,&a,4); std::memcpy(&y,&b,4); return x==y; }
static int ulp_diff(float a, float b) {
    int32_t x, y; std::memcpy(&x,&a,4); std::memcpy(&y,&b,4);
    if (x < 0) x = (int32_t)0x80000000u - x; if (y < 0) y = (int32_t)0x80000000u - y;
    long long d = (long long)x - (long long)y; return (int)(d < 0 ? -d : d);
}
int main() {
    const long long n = 40000000;
    std::vector<float> g(n), u(n), hg(n), hgs(n), hsw(n), d_exp(n), d_sw(n);
    uint64_t s = 88172645463325252ull;
    auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (float)((s >> 11) * (1.0/9007199254740992.0)); };
    for (long long i = 0; i < n; ++i) {
        // half uniform in [-40, 5] (the range a gate value can occupy), half dense linspace
        g[i] = (i & 1) ? (-40.f + 45.f * rnd()) : (-40.f + 45.f * (float)i / (float)n);
        u[i] = -2.f + 4.f * rnd();
    }
    for (long long i = 0; i < n; ++i) { hg[i] = std::exp(g[i]); hgs[i] = std::exp(-g[i]);
        hsw[i] = (g[i] / (1.f + hgs[i])) * u[i]; }
    float *dg, *du, *dy;
    cudaMalloc(&dg, n*4); cudaMalloc(&du, n*4); cudaMalloc(&dy, n*4);
    cudaMemcpy(dg, g.data(), n*4, cudaMemcpyHostToDevice);
    cudaMemcpy(du, u.data(), n*4, cudaMemcpyHostToDevice);
    k_exp<<<(n+255)/256, 256>>>(dg, dy, n);
    cudaDeviceSynchronize();
    cudaMemcpy(d_exp.data(), dy, n*4, cudaMemcpyDeviceToHost);
    // swiglu on device, using the same inputs (host used exp(-g))
    std::vector<float> ng(n); for (long long i=0;i<n;++i) ng[i] = -g[i];
    float* dng; cudaMalloc(&dng, n*4); cudaMemcpy(dng, ng.data(), n*4, cudaMemcpyHostToDevice);
    k_swiglu<<<(n+255)/256, 256>>>(dg, du, dy, n);
    cudaDeviceSynchronize();
    cudaMemcpy(d_sw.data(), dy, n*4, cudaMemcpyDeviceToHost);

    long long bad_exp_pos = 0, bad_sw = 0; int worst_ulp = 0; float worst_x = 0;
    for (long long i = 0; i < n; ++i) {
        if (!same_bits(hg[i], d_exp[i])) { ++bad_exp_pos; int u = ulp_diff(hg[i], d_exp[i]); if (u > worst_ulp) { worst_ulp = u; worst_x = g[i]; } }
        if (!same_bits(hsw[i], d_sw[i])) ++bad_sw;
    }
    std::printf("exp(+x): %lld of %lld differ bitwise (%.4f%%), worst %d ulp at x=%.9g\n",
                bad_exp_pos, n, 100.0*bad_exp_pos/n, worst_ulp, worst_x);
    std::printf("swiglu (g/(1+exp(-g)))*u: %lld of %lld differ bitwise (%.4f%%)\n", bad_sw, n, 100.0*bad_sw/n);
    return 0;
}
