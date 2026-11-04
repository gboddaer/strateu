// One-block oracle: instrument the REAL AVX2 source at its FMA boundaries.
// No copied/rederived AVX2 body. Compare its integer lanes with a scalar
// interpretation and isolate the sign of the minimum correction.
#include <immintrin.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
static float avx_sumi[8], avx_prod[4], avx_d, avx_dmin;
static __m256 capture256(__m256 d, __m256 v, __m256 a) {
    _mm256_storeu_ps(avx_sumi, v); avx_d = _mm_cvtss_f32(_mm256_castps256_ps128(d));
    return _mm256_fmadd_ps(d, v, a);
}
static __m128 capture128(__m128 d, __m128 v, __m128 a) {
    _mm_storeu_ps(avx_prod, v); avx_dmin = _mm_cvtss_f32(d);
    return _mm_fmadd_ps(d, v, a);
}
#define _mm256_fmadd_ps capture256
#define _mm_fmadd_ps capture128
#include "../src/kernels/cpu/kq_avx2.cpp"
#undef _mm256_fmadd_ps
#undef _mm_fmadd_ps

static uint64_t rng = 0x243f6a8885a308d3ULL;
static uint32_t random32() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng >> 32; }
static float tree8(const float* a) { return ((a[0]+a[4])+(a[2]+a[6]))+((a[1]+a[5])+(a[3]+a[7])); }
static float tree4(const float* a) { return (a[0]+a[2])+(a[1]+a[3]); }
static bool same(float a, float b) { return std::memcmp(&a, &b, 4) == 0; }
int main() {
    int errors=0, old_sign_matches=0;
    constexpr int cases=4096;
    for (int trial=0; trial<cases; ++trial) {
        block_q4_K x{}; block_q8_K y{};
        // All 6-bit scale/min encodings, signed/zero scales and both signs
        // of Q8 (including -128) are covered, with finite normal FP16 scales.
        const uint16_t hd = uint16_t((random32()%2 ? 0x8000 : 0) | (0x2000 + random32()%0x1800));
        const uint16_t hm = uint16_t((random32()%2 ? 0x8000 : 0) | (0x2000 + random32()%0x1800));
        std::memcpy(&x, &hd, 2); std::memcpy((uint8_t*)&x+2, &hm, 2);
        for (auto& v:x.scales) v=random32();
        for (auto& v:x.qs) v=random32();
        y.d = (int(random32()%20001)-10000)*1e-5f;
        for (auto& v:y.qs) v=int8_t(random32());
        for (int j=0;j<16;++j) { int sum=0; for(int k=0;k<16;++k) sum+=y.qs[16*j+k]; y.bsums[j]=sum; }
        const void* act=&y; float actual;
        strata::kernels::cpu::q4k_dot(&x,1,&act,1,&actual);
        int s[8],m[8];
        for(int j=0;j<4;++j) { s[j]=x.scales[j]&63; m[j]=x.scales[j+4]&63; }
        for(int j=4;j<8;++j) {
            s[j]=(x.scales[j+4]&15)|((x.scales[j-4]>>6)<<4);
            m[j]=(x.scales[j+4]>>4)|((x.scales[j]>>6)<<4);
        }
        int sumi[8]={},prod[4]={}; double exact=0;
        const float xd=_cvtsh_ss(hd), xm=_cvtsh_ss(hm);
        for(int j=0;j<8;++j) for(int k=0;k<32;++k) {
            const int q=(x.qs[32*(j/2)+k]>>(4*(j%2)))&15;
            sumi[k/4]+=s[j]*q*int(y.qs[32*j+k]);
            exact+=(double(xd)*s[j]*q-double(xm)*m[j])*double(y.d)*y.qs[32*j+k];
        }
        float acc[8],mins[4],oldmins[4];
        for(int k=0;k<8;++k) acc[k]=std::fma(y.d*xd,float(sumi[k]),0.0f);
        for(int k=0;k<4;++k) {
            prod[k]=m[2*k]*(int(y.bsums[4*k])+y.bsums[4*k+1])+m[2*k+1]*(int(y.bsums[4*k+2])+y.bsums[4*k+3]);
            mins[k]=std::fma(-y.d*xm,float(prod[k]),0.0f);
            oldmins[k]=std::fma(y.d*xm,float(prod[k]),0.0f);
        }
        bool lanes=true;
        for(int k=0;k<8;++k) lanes &= avx_sumi[k]==sumi[k];
        for(int k=0;k<4;++k) lanes &= avx_prod[k]==prod[k];
        const float scalar=tree8(acc)+tree4(mins), old=tree8(acc)+tree4(oldmins);
        old_sign_matches+=same(old,actual);
        if(!lanes || !same(scalar,actual)) ++errors;
        if(trial==0) {
            std::printf("sumi AVX/scalar:"); for(int k=0;k<8;++k) std::printf(" %.0f/%d",avx_sumi[k],sumi[k]);
            std::printf("\nprod AVX/scalar:"); for(int k=0;k<4;++k) std::printf(" %.0f/%d",avx_prod[k],prod[k]);
            std::printf("\nd=%a dmin AVX=%a old-port=%a\nactual=%a scalar=%a old-sign=%a float64=%.17g\n",avx_d,avx_dmin,y.d*xm,actual,scalar,old,exact);
        }
    }
    std::printf("one-block: %d/%d exact lane + result matches; old-sign matches %d/%d\n",cases-errors,cases,old_sign_matches,cases);
    return errors?1:0;
}
