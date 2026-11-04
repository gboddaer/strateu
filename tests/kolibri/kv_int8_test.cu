// Experimental KV format unit test, independent of model and engine.
#include "../../src/kolibri/kv_int8.cuh"
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstring>
#define CHECK(x) do { cudaError_t ec=(x); if(ec!=cudaSuccess) { std::fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(ec)); return 2; } } while(0)
int main() {
    using namespace strata::kolibri::experimental;
    constexpr int H=2,D=128,R=7;
    const int count=H*D*R, blocks=count/64;
    std::vector<__half> src(count),got(count);
    for(int i=0;i<count;++i) src[i]=__float2half(float((i*17)%503-251)*.013f);
    // zero block and smallest half: exercise scale underflow and zero branches.
    for(int i=0;i<64;++i) src[i]=__float2half(0);
    for(int i=64;i<128;++i) src[i]=__ushort_as_half(1);
    __half *d,*out; Kv8Block* packed;
    CHECK(cudaMalloc(&d,count*2)); CHECK(cudaMalloc(&out,count*2)); CHECK(cudaMalloc(&packed,blocks*sizeof(Kv8Block)));
    CHECK(cudaMemcpy(d,src.data(),count*2,cudaMemcpyHostToDevice));
    CHECK(cudaMemset(packed,0,blocks*sizeof(Kv8Block)));
    // p0=5, n=4 crosses the 7-slot ring; only slots 5,6,0,1 must change.
    kv8_pack<<<H*4*(D/64),64>>>(d,packed,H,D,R,5,4,true);
    kv8_unpack<<<blocks,64>>>(packed,out,blocks);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(got.data(),out,count*2,cudaMemcpyDeviceToHost));
    int errors=0;
    for(int h=0;h<H;++h) for(int r=0;r<R;++r) for(int b=0;b<D/64;++b) {
        const int start=(h*R+r)*D+b*64;
        const bool touched=r==5||r==6||r==0||r==1;
        float m=0; for(int i=0;i<64;++i) m=fmaxf(m,fabsf(__half2float(src[start+i])));
        float scale=__half2float(__float2half(m/127.f));
        if(m>0 && scale==0) scale=__half2float(__ushort_as_half(1));
        for(int i=0;i<64;++i) {
            const float f=__half2float(src[start+i]);
            int q=scale==0?0:int(nearbyintf(f/scale)); q=q < -127 ? -127 : (q>127?127:q);
            const __half want=__float2half(touched?scale*q:0.f);
            if(std::memcmp(&want,&got[start+i],2)) ++errors;
        }
    }
    // Repeated packing and restoration must not re-quantize old restored rows.
    std::vector<__half> again(count);
    kv8_unpack<<<blocks,64>>>(packed,out,blocks);
    CHECK(cudaMemcpy(again.data(),out,count*2,cudaMemcpyDeviceToHost));
    if(std::memcmp(again.data(),got.data(),count*2)) ++errors;
    // Partial restore must use the full per-head stride, not a dense visible-row stride.
    CHECK(cudaMemset(out,0x3c,count*2));
    kv8_restore<<<H*3*(D/64),64>>>(packed,out,D,R,3);
    CHECK(cudaMemcpy(again.data(),out,count*2,cudaMemcpyDeviceToHost));
    for(int h=0;h<H;++h) for(int r=0;r<R;++r) for(int k=0;k<D;++k) {
        const int i=(h*R+r)*D+k;
        const __half want=r<3?got[i]:__ushort_as_half(0x3c3c);
        if(std::memcmp(&want,&again[i],2)) ++errors;
    }
    kv8_pack<<<H*R*(D/64),64>>>(d,packed,H,D,R,0,R,false,d);
    kv8_unpack<<<blocks,64>>>(packed,out,blocks);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(got.data(),out,count*2,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(again.data(),d,count*2,cudaMemcpyDeviceToHost));
    if(std::memcmp(again.data(),got.data(),count*2)) ++errors;
    CHECK(cudaFree(d)); CHECK(cudaFree(out)); CHECK(cudaFree(packed));
    std::printf("kv_int8: %d errors; block=%zu bytes / 64 values; ring, zeros, tiny scale, repeat checked\n",errors,sizeof(Kv8Block));
    return errors?1:0;
}
