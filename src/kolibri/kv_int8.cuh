#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
namespace strata::kolibri::experimental {
// 64 signed codes and a rounded FP16 scale, deliberately not padded to 68B.
struct Kv8Block { int8_t q[64]; __half scale; };
static_assert(sizeof(Kv8Block)==66,"KV int8 wire size");
__global__ void kv8_pack(const __half* src, Kv8Block* dst, int heads, int dim, int rows,
                         int64_t pos0, int count, bool ring, __half* roundtrip=nullptr) {
    const int chunks=dim/64;
    const int chunk=blockIdx.x%chunks, t=(blockIdx.x/chunks)%count;
    const int head=blockIdx.x/(chunks*count), lane=threadIdx.x;
    if(head>=heads || lane>=64) return;
    const int64_t slot=ring?(pos0+t)%rows:pos0+t;
    const size_t index=((size_t)head*rows+slot)*chunks+chunk;
    const float v=__half2float(src[index*64+lane]);
    __shared__ float maxima[64];
    maxima[lane]=fabsf(v); __syncthreads();
    for(int stride=32;stride;stride/=2) {
        if(lane<stride) maxima[lane]=fmaxf(maxima[lane],maxima[lane+stride]);
        __syncthreads();
    }
    __half hs=__float2half_rn(maxima[0]/127.f);
    if(maxima[0]>0.f && __half2float(hs)==0.f) hs=__ushort_as_half(1);
    const float scale=__half2float(hs);
    const int q=scale==0.f?0:__float2int_rn(__fdiv_rn(v,scale));
    const int clipped=max(-127,min(127,q));
    dst[index].q[lane]=(int8_t)clipped;
    if(lane==0) dst[index].scale=hs;
    if(roundtrip) roundtrip[index*64+lane]=__float2half_rn(scale*float(clipped));
}
// Restore only reachable rows, respecting the per-head physical stride.
__global__ void kv8_restore(const Kv8Block* src, __half* dst, int dim, int rows, int visible) {
    const int chunks=dim/64, b=blockIdx.x%chunks;
    const int token=(blockIdx.x/chunks)%visible, head=blockIdx.x/(chunks*visible);
    const size_t index=((size_t)head*rows+token)*chunks+b;
    dst[index*64+threadIdx.x]=__float2half_rn(__half2float(src[index].scale)*float(src[index].q[threadIdx.x]));
}
__global__ void kv8_unpack(const Kv8Block* src, __half* dst, size_t blocks) {
    const size_t index=blockIdx.x;
    if(index>=blocks || threadIdx.x>=64) return;
    dst[index*64+threadIdx.x]=__float2half_rn(__half2float(src[index].scale)*float(src[index].q[threadIdx.x]));
}
}
