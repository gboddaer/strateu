// tools/kolibri/expert_stream_capture.hpp - Task 1 of plan 2026-10-31-kolibri-gpu-streaming.
//
// The streamed expert prefill's workload capture, schema v1: one record per (chunk, layer) expert call, holding
// everything the CPU expert leg consumed and produced, so the same work can be replayed later without the engine,
// the model or the router.
//
//   <dir>/capture.bin        header, manifest, length-delimited records, trailer
//   <dir>/blobs/<sha256>.bin one file per unique touched expert blob (deduplicated by content)
//
// A record is [u32 payload_len][payload][32-byte SHA-256 of the payload].  The payload's every size is derivable
// from its own geometry, and the reader rejects the record unless the length, the derived sizes and the digest all
// agree - so truncation, a stale version and a mis-framed record cannot be mistaken for data.  Nothing here touches
// CUDA, so the engine (a .cu, C++17) and the host replay tool both include it.
#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "strata/kernels/cpu/native_expert.hpp"

namespace strata::kolibri::stream::capture {

inline constexpr char kMagic[8] = {'K', 'S', 'T', 'R', 'C', 'A', 'P', '1'};
inline constexpr char kTrailerMagic[8] = {'K', 'S', 'T', 'R', 'E', 'N', 'D', '1'};
inline constexpr uint32_t kVersion = 1;
inline constexpr uint32_t kEndian = 0x01020304u;
/// Default cap on the blob store: past it the writer refuses, rather than filling the disk mid-campaign.
inline constexpr uint64_t kDefaultStoreCapBytes = uint64_t(200) << 30;

using Digest = std::array<uint8_t, 32>;

// ----------------------------------------------------------------------------------------------------------------
// SHA-256 (FIPS 180-4), the one hash both halves of the capture use.
// ----------------------------------------------------------------------------------------------------------------
namespace sha256_detail {

inline constexpr uint32_t kK[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

inline uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

inline void compress(uint32_t h[8], const uint8_t block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
        w[i] = ((uint32_t) block[i * 4] << 24) | ((uint32_t) block[i * 4 + 1] << 16) |
               ((uint32_t) block[i * 4 + 2] << 8) | (uint32_t) block[i * 4 + 3];
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t s1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t t1 = k + s1 + ch + kK[i] + w[i];
        const uint32_t s0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = s0 + maj;
        k = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += k;
}

}  // namespace sha256_detail

/// SHA-256 of `size` bytes.  Streamed over the caller's buffer once; no allocation.
inline Digest sha256(const void* data, size_t size) {
    using namespace sha256_detail;
    uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                     0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    const uint8_t* p = (const uint8_t*) data;
    size_t left = size;
    while (left >= 64) {
        compress(h, p);
        p += 64;
        left -= 64;
    }
    uint8_t tail[128];
    std::memset(tail, 0, sizeof tail);
    if (left != 0) std::memcpy(tail, p, left);
    tail[left] = 0x80;
    const size_t tail_len = left + 1 + 8 <= 64 ? 64 : 128;
    const uint64_t bits = (uint64_t) size * 8u;
    for (int i = 0; i < 8; ++i) tail[tail_len - 1 - (size_t) i] = (uint8_t) (bits >> (8 * i));
    compress(h, tail);
    if (tail_len == 128) compress(h, tail + 64);
    Digest d{};
    for (int i = 0; i < 8; ++i) {
        d[(size_t) i * 4] = (uint8_t) (h[i] >> 24);
        d[(size_t) i * 4 + 1] = (uint8_t) (h[i] >> 16);
        d[(size_t) i * 4 + 2] = (uint8_t) (h[i] >> 8);
        d[(size_t) i * 4 + 3] = (uint8_t) h[i];
    }
    return d;
}

inline std::string hex(const Digest& d) {
    static const char* digits = "0123456789abcdef";
    std::string s(64, '0');
    for (size_t i = 0; i < 32; ++i) {
        s[i * 2] = digits[d[i] >> 4];
        s[i * 2 + 1] = digits[d[i] & 0xf];
    }
    return s;
}

// ----------------------------------------------------------------------------------------------------------------
// The record and its serialized form.
// ----------------------------------------------------------------------------------------------------------------

/// One layer's format, field by field.  NativeFmt itself is never memcpy'd: its padding and its borrowed sign
/// pointers are not part of the capture.
struct Fmt {
    int32_t gu_type = -1, d_type = -1, gu_act = -1, d_act = -1;
    int64_t n_embd = 0, n_ff = 0;
    uint64_t gu_row = 0, d_row = 0, up_off = 0, down_off = 0, bytes = 0, act_bytes = 0, h_bytes = 0;

    static Fmt from(const strata::kernels::cpu::NativeFmt& f) {
        Fmt o;
        o.gu_type = f.gu_type;
        o.d_type = f.d_type;
        o.gu_act = f.gu_act;
        o.d_act = f.d_act;
        o.n_embd = f.n_embd;
        o.n_ff = f.n_ff;
        o.gu_row = f.gu_row;
        o.d_row = f.d_row;
        o.up_off = f.up_off;
        o.down_off = f.down_off;
        o.bytes = f.bytes;
        o.act_bytes = f.act_bytes;
        o.h_bytes = f.h_bytes;
        return o;
    }
    /// Rebuilds the layer's format.  The fields this path does not stream (the NVFP4 tail, a rotated basis, the
    /// per-projection rotation flags) keep their defaults, which is what the capture's validation requires.
    strata::kernels::cpu::NativeFmt to_native() const {
        strata::kernels::cpu::NativeFmt f;
        f.gu_type = gu_type;
        f.d_type = d_type;
        f.gu_act = gu_act;
        f.d_act = d_act;
        f.n_embd = n_embd;
        f.n_ff = n_ff;
        f.gu_row = (size_t) gu_row;
        f.d_row = (size_t) d_row;
        f.up_off = (size_t) up_off;
        f.down_off = (size_t) down_off;
        f.bytes = (size_t) bytes;
        f.act_bytes = (size_t) act_bytes;
        f.h_bytes = (size_t) h_bytes;
        return f;
    }
};

/// A touched expert blob, referenced by content.  The writer stores the bytes once; the reader loads them from
/// `<dir>/blobs/<sha256>.bin`.
struct BlobRef {
    int32_t expert = -1;
    uint64_t size = 0;
    Digest sha{};
};

struct Record {
    int32_t layer = -1;
    int64_t chunk = -1;
    int32_t top_k = 0, selections = 0, tokens = 0, experts = 0;
    uint64_t weight_stride = 0;
    Fmt fmt;
    std::vector<int32_t> ids;     ///< one expert id per selection, original selection order
    std::vector<uint8_t> acts;    ///< tokens x fmt.act_bytes of already-quantized activations
    std::vector<BlobRef> blobs;   ///< the unique touched experts this record read
    std::vector<float> out;       ///< selections x fmt.n_embd, the CPU expert result in selection order
    Digest digest{};              ///< of the serialized payload (set by the writer, re-checked by the reader)
};

/// Every refusal in this header goes through one function, so a caller can catch one type and the message always
/// names the capture.
[[noreturn]] inline void refuse(const std::string& why) { throw std::runtime_error("capture: " + why); }

namespace detail {

inline constexpr uint64_t kPayloadFixed = 4 + 8 + 4 * 4 + 8 + (4 * 4 + 2 * 8 + 7 * 8);
inline constexpr uint64_t kBlobRefBytes = 4 + 8 + 32;

inline void put_u32(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back((uint8_t) (v >> (8 * i)));
}
inline void put_u64(std::vector<uint8_t>& b, uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back((uint8_t) (v >> (8 * i)));
}
inline void put_i32(std::vector<uint8_t>& b, int32_t v) { put_u32(b, (uint32_t) v); }
inline void put_i64(std::vector<uint8_t>& b, int64_t v) { put_u64(b, (uint64_t) v); }
inline void put_bytes(std::vector<uint8_t>& b, const void* p, size_t n) {
    const uint8_t* s = (const uint8_t*) p;
    b.insert(b.end(), s, s + n);
}

inline uint64_t mul_or_refuse(uint64_t a, uint64_t b) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) refuse("a size overflows");
    return a * b;
}
inline uint64_t add_or_refuse(uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a) refuse("a size overflows");
    return a + b;
}

/// The scalar validation the reader applies before it trusts any size: an impossible geometry or an unsupported
/// format cannot be turned into a buffer length.
inline void validate_geometry(const Record& r) {
    if (r.selections < 0 || r.tokens < 0 || r.top_k <= 0 || r.experts <= 0) refuse("an impossible record geometry");
    if ((uint64_t) r.selections % (uint64_t) r.top_k != 0) refuse("selections are not a multiple of top_k");
    if ((uint64_t) r.selections / (uint64_t) r.top_k != (uint64_t) r.tokens)
        refuse("the token count is not selections/top_k");
    if (r.fmt.n_embd <= 0 || r.fmt.n_ff <= 0 || r.fmt.n_embd % 256 != 0 || r.fmt.n_ff % 256 != 0)
        refuse("a geometry that is not whole 256-value blocks");
    if (r.fmt.act_bytes != (uint64_t) (r.fmt.n_embd / 256) * 292u ||
        r.fmt.h_bytes != (uint64_t) (r.fmt.n_ff / 256) * 292u)
        refuse("activation sizes that are not Q8_K of the geometry");
    if (r.fmt.gu_type != 12 || (r.fmt.d_type != 12 && r.fmt.d_type != 14) || r.fmt.gu_act != 15 ||
        r.fmt.d_act != 15)
        refuse("a weight or activation type outside the supported path");
    if (r.weight_stride < r.fmt.bytes) refuse("a weight stride shorter than the blob");
}

/// The payload length every field of `r` implies (the blob count must already be known).
inline uint64_t expected_payload(const Record& r) {
    validate_geometry(r);
    if (r.blobs.size() > std::numeric_limits<uint32_t>::max()) refuse("too many blobs in one record");
    uint64_t n = kPayloadFixed;
    n = add_or_refuse(n, mul_or_refuse((uint64_t) r.selections, 4));
    n = add_or_refuse(n, mul_or_refuse((uint64_t) r.tokens, r.fmt.act_bytes));
    n = add_or_refuse(n, 4);
    n = add_or_refuse(n, mul_or_refuse((uint64_t) r.blobs.size(), kBlobRefBytes));
    n = add_or_refuse(n, mul_or_refuse(mul_or_refuse((uint64_t) r.selections, (uint64_t) r.fmt.n_embd), 4));
    return n;
}

/// Bounds-checked reader over one payload: a bad length cannot walk off the buffer.
struct Cursor {
    const uint8_t* p;
    size_t left;
    uint32_t u32() {
        if (left < 4) refuse("a truncated payload");
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) v |= (uint32_t) p[i] << (8 * i);
        p += 4;
        left -= 4;
        return v;
    }
    uint64_t u64() {
        if (left < 8) refuse("a truncated payload");
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= (uint64_t) p[i] << (8 * i);
        p += 8;
        left -= 8;
        return v;
    }
    int32_t i32() { return (int32_t) u32(); }
    int64_t i64() { return (int64_t) u64(); }
    const uint8_t* take(size_t n) {
        if (left < n) refuse("a truncated payload");
        const uint8_t* q = p;
        p += n;
        left -= n;
        return q;
    }
};

}  // namespace detail

/// Serializes one record's payload (without the length prefix and the digest).
inline std::vector<uint8_t> encode(const Record& r) {
    std::vector<uint8_t> b;
    b.reserve((size_t) detail::expected_payload(r));
    detail::put_i32(b, r.layer);
    detail::put_i64(b, r.chunk);
    detail::put_i32(b, r.top_k);
    detail::put_i32(b, r.selections);
    detail::put_i32(b, r.tokens);
    detail::put_i32(b, r.experts);
    detail::put_u64(b, r.weight_stride);
    detail::put_i32(b, r.fmt.gu_type);
    detail::put_i32(b, r.fmt.d_type);
    detail::put_i32(b, r.fmt.gu_act);
    detail::put_i32(b, r.fmt.d_act);
    detail::put_i64(b, r.fmt.n_embd);
    detail::put_i64(b, r.fmt.n_ff);
    detail::put_u64(b, r.fmt.gu_row);
    detail::put_u64(b, r.fmt.d_row);
    detail::put_u64(b, r.fmt.up_off);
    detail::put_u64(b, r.fmt.down_off);
    detail::put_u64(b, r.fmt.bytes);
    detail::put_u64(b, r.fmt.act_bytes);
    detail::put_u64(b, r.fmt.h_bytes);
    for (int32_t id : r.ids) detail::put_i32(b, id);
    if (!r.acts.empty()) detail::put_bytes(b, r.acts.data(), r.acts.size());
    detail::put_u32(b, (uint32_t) r.blobs.size());
    for (const BlobRef& blob : r.blobs) {
        detail::put_i32(b, blob.expert);
        detail::put_u64(b, blob.size);
        detail::put_bytes(b, blob.sha.data(), blob.sha.size());
    }
    if (!r.out.empty()) detail::put_bytes(b, r.out.data(), r.out.size() * sizeof(float));
    if (b.size() != detail::expected_payload(r)) refuse("the encoded payload is not the size its geometry implies");
    return b;
}

/// Parses one payload.  Rejects a length that disagrees with the geometry, a bad id, an impossible size, and any
/// geometry outside this initial path's supported formats.
inline Record decode(const uint8_t* payload, size_t size) {
    Record r;
    detail::Cursor c{payload, size};
    r.layer = c.i32();
    r.chunk = c.i64();
    r.top_k = c.i32();
    r.selections = c.i32();
    r.tokens = c.i32();
    r.experts = c.i32();
    r.weight_stride = c.u64();
    r.fmt.gu_type = c.i32();
    r.fmt.d_type = c.i32();
    r.fmt.gu_act = c.i32();
    r.fmt.d_act = c.i32();
    r.fmt.n_embd = c.i64();
    r.fmt.n_ff = c.i64();
    r.fmt.gu_row = c.u64();
    r.fmt.d_row = c.u64();
    r.fmt.up_off = c.u64();
    r.fmt.down_off = c.u64();
    r.fmt.bytes = c.u64();
    r.fmt.act_bytes = c.u64();
    r.fmt.h_bytes = c.u64();
    detail::validate_geometry(r);
    r.ids.resize((size_t) r.selections);
    for (int32_t& id : r.ids) {
        id = c.i32();
        if (id < 0 || id >= r.experts) refuse("an expert id out of range");
    }
    {
        const size_t n = (size_t) r.tokens * (size_t) r.fmt.act_bytes;
        const uint8_t* q = c.take(n);
        r.acts.assign(q, q + n);
    }
    const uint32_t blob_count = c.u32();
    r.blobs.resize(blob_count);
    for (BlobRef& blob : r.blobs) {
        blob.expert = c.i32();
        blob.size = c.u64();
        const uint8_t* sha = c.take(32);
        std::memcpy(blob.sha.data(), sha, 32);
        if (blob.expert < 0 || blob.expert >= r.experts) refuse("a blob's expert id is out of range");
    }
    {
        const size_t n = (size_t) r.selections * (size_t) r.fmt.n_embd;
        const uint8_t* q = c.take(n * sizeof(float));
        r.out.resize(n);
        std::memcpy(r.out.data(), q, n * sizeof(float));
    }
    if (c.left != 0) refuse("trailing bytes in the payload");
    if (detail::expected_payload(r) != size) refuse("the payload length is not the length its geometry implies");
    return r;
}

// ----------------------------------------------------------------------------------------------------------------
// Writer / reader.
// ----------------------------------------------------------------------------------------------------------------

/// One touched expert's bytes as the caller holds them (the engine's pinned table, a replay's mapped blob).
struct BlobInput {
    int32_t expert = -1;
    const uint8_t* bytes = nullptr;
    size_t size = 0;
};

/// Writes a capture into a NEW directory.  Refuses an existing non-empty directory, and refuses to grow the blob
/// store past its cap.  Records are appended and flushed as they are written, so a killed run leaves a truncated
/// capture the reader rejects rather than a silently short one.
///
/// **PRECONDITION: bytes handed to add() are immutable for the writer's lifetime.**  The blob store is
/// content-addressed, and a digest is memoised per source pointer: a long capture passes the same expert blob to
/// add() once per chunk (16 times for an 8191-id prompt), and re-hashing 737 GB of unchanged weights was measured
/// to cost an hour on this box while the record itself is unchanged.  The engine's pinned expert table is
/// allocated at load and never written, which is exactly this precondition; a caller that rewrites a buffer it
/// already captured must use a fresh Writer.
class Writer {
public:
    Writer(std::string dir, std::string manifest_json, uint64_t cap_bytes = kDefaultStoreCapBytes)
        : dir_(std::move(dir)), cap_(cap_bytes) {
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::path d(dir_);
        if (fs::exists(d, ec) && !fs::is_empty(d, ec))
            throw std::runtime_error("capture: refusing to write into a non-empty directory: " + dir_);
        fs::create_directories(d / "blobs", ec);
        if (ec) throw std::runtime_error("capture: cannot create " + dir_ + ": " + ec.message());
        out_.open((d / "capture.bin").string(), std::ios::binary | std::ios::trunc);
        if (!out_) throw std::runtime_error("capture: cannot open capture.bin for writing in " + dir_);
        out_.write(kMagic, sizeof kMagic);
        detail::put_u32(header_, kVersion);
        detail::put_u32(header_, kEndian);
        detail::put_u64(header_, manifest_json.size());
        header_.insert(header_.end(), manifest_json.begin(), manifest_json.end());
        const Digest md = sha256(manifest_json.data(), manifest_json.size());
        header_.insert(header_.end(), md.begin(), md.end());
        out_.write((const char*) header_.data(), (std::streamsize) header_.size());
        if (!out_) throw std::runtime_error("capture: cannot write the header in " + dir_);
    }
    ~Writer() {
        if (!closed_) {   // never throw out of a destructor; a truncated file is rejected by the reader
            try {
                close();
            } catch (...) {
            }
        }
    }
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;

    /// Stores `blobs` (deduplicated by content), then frames and writes `r`.  Fills r.blobs and r.digest.
    void add(Record& r, const std::vector<BlobInput>& blobs) {
        r.blobs.clear();
        r.blobs.reserve(blobs.size());
        for (const BlobInput& in : blobs) {
            if (in.bytes == nullptr && in.size != 0) throw std::runtime_error("capture: a blob has no bytes");
            BlobRef ref;
            ref.expert = in.expert;
            ref.size = in.size;
            ref.sha = digest_of(in.bytes, in.size);
            const std::filesystem::path path = std::filesystem::path(dir_) / "blobs" / (hex(ref.sha) + ".bin");
            std::error_code ec;
            if (!std::filesystem::exists(path, ec)) {
                if (stored_ + in.size > cap_) {
                    char msg[256];
                    std::snprintf(msg, sizeof msg,
                                  "capture: the blob store would pass its cap (%llu + %llu > %llu bytes); "
                                  "delete this capture or raise the cap",
                                  (unsigned long long) stored_, (unsigned long long) in.size,
                                  (unsigned long long) cap_);
                    throw std::runtime_error(msg);
                }
                const std::filesystem::path tmp = path.string() + ".tmp";
                {
                    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
                    if (!f) throw std::runtime_error("capture: cannot create " + tmp.string());
                    f.write((const char*) in.bytes, (std::streamsize) in.size);
                    if (!f) throw std::runtime_error("capture: cannot write " + tmp.string());
                }
                std::filesystem::rename(tmp, path, ec);
                if (ec) throw std::runtime_error("capture: cannot publish " + path.string() + ": " + ec.message());
                stored_ += in.size;
            }
            r.blobs.push_back(ref);
        }
        const std::vector<uint8_t> payload = encode(r);
        r.digest = sha256(payload.data(), payload.size());
        std::vector<uint8_t> len;
        detail::put_u32(len, (uint32_t) payload.size());
        out_.write((const char*) len.data(), 4);
        out_.write((const char*) payload.data(), (std::streamsize) payload.size());
        out_.write((const char*) r.digest.data(), 32);
        if (!out_) throw std::runtime_error("capture: cannot write a record in " + dir_);
        ++records_;
    }

    void close() {
        if (closed_) return;
        out_.write(kTrailerMagic, sizeof kTrailerMagic);
        out_.flush();
        out_.close();
        if (!out_) throw std::runtime_error("capture: cannot write the trailer in " + dir_);
        closed_ = true;
    }

    const std::string& dir() const { return dir_; }
    uint64_t records() const { return records_; }
    uint64_t blob_bytes() const { return stored_; }

private:
    Digest digest_of(const uint8_t* bytes, size_t size) {
        if (bytes != nullptr) {
            const auto it = memo_.find(bytes);
            if (it != memo_.end() && it->second.size == size) return it->second.sha;
        }
        const Digest d = sha256(bytes, size);
        if (bytes != nullptr) memo_[bytes] = Memo{size, d};
        return d;
    }

    struct Memo {
        size_t size;
        Digest sha;
    };
    std::unordered_map<const void*, Memo> memo_;   // source pointer -> its digest (see the class precondition)
    std::string dir_;
    uint64_t cap_ = 0;
    uint64_t stored_ = 0;
    uint64_t records_ = 0;
    bool closed_ = false;
    std::vector<uint8_t> header_;
    std::ofstream out_;
};

/// Reads a capture back, record by record.  The header, the manifest and every record are checksum-verified; a
/// truncated, reordered or foreign file is refused rather than partially replayed.
class Reader {
public:
    explicit Reader(std::string dir) : dir_(std::move(dir)) {
        in_.open((std::filesystem::path(dir_) / "capture.bin").string(), std::ios::binary);
        if (!in_) throw std::runtime_error("capture: cannot open " + dir_ + "/capture.bin");
        char magic[8];
        in_.read(magic, 8);
        if (!in_ || std::memcmp(magic, kMagic, sizeof kMagic) != 0)
            throw std::runtime_error("capture: bad magic (not a v1 capture) in " + dir_);
        const uint32_t version = read_u32();
        const uint32_t endian = read_u32();
        if (version != kVersion) throw std::runtime_error("capture: unsupported version in " + dir_);
        if (endian != kEndian) throw std::runtime_error("capture: unsupported byte order in " + dir_);
        const uint64_t mlen = read_u64();
        if (mlen > (uint64_t) 1 << 20) throw std::runtime_error("capture: an impossible manifest length in " + dir_);
        manifest_.resize((size_t) mlen);
        if (mlen != 0) in_.read(&manifest_[0], (std::streamsize) mlen);
        Digest expect{};
        in_.read((char*) expect.data(), 32);
        if (!in_) throw std::runtime_error("capture: truncated header in " + dir_);
        const Digest got = sha256(manifest_.data(), manifest_.size());
        if (got != expect) throw std::runtime_error("capture: the manifest checksum does not match in " + dir_);
    }

    const std::string& manifest() const { return manifest_; }

    /// Next record, or false at the verified trailer.  Throws on a short/framed/digest/geometry failure.
    bool next(Record& out) {
        if (done_) return false;
        char head[8];
        in_.read(head, 8);
        if (!in_) throw std::runtime_error("capture: missing trailer (truncated capture) in " + dir_);
        if (std::memcmp(head, kTrailerMagic, sizeof kTrailerMagic) == 0) {
            if (in_.peek() != std::char_traits<char>::eof())
                throw std::runtime_error("capture: bytes after the trailer in " + dir_);
            done_ = true;
            return false;
        }
        in_.seekg(-8, std::ios::cur);
        const uint32_t len = read_u32();
        std::vector<uint8_t> payload(len);
        if (len != 0) in_.read((char*) payload.data(), (std::streamsize) len);
        Digest expect{};
        in_.read((char*) expect.data(), 32);
        if (!in_) throw std::runtime_error("capture: a truncated record in " + dir_);
        const Digest got = sha256(payload.data(), payload.size());
        if (got != expect) throw std::runtime_error("capture: a record checksum does not match in " + dir_);
        out = decode(payload.data(), payload.size());
        out.digest = expect;
        return true;
    }

    /// Loads one referenced blob into `dst` (capacity >= ref.size), verifying size and checksum.  The destination is
    /// the caller's, so a replay can keep the blob on the same alignment the engine's pinned table has.
    void load_into(const BlobRef& ref, uint8_t* dst, size_t capacity) const {
        if (capacity < ref.size) throw std::runtime_error("capture: the destination is smaller than the blob");
        const std::filesystem::path path = std::filesystem::path(dir_) / "blobs" / (hex(ref.sha) + ".bin");
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("capture: missing blob " + path.string());
        const std::streamoff size = f.tellg();
        if (size < 0 || (uint64_t) size != ref.size)
            throw std::runtime_error("capture: blob size does not match the record: " + path.string());
        f.seekg(0);
        if (ref.size != 0) f.read((char*) dst, (std::streamsize) ref.size);
        if (!f) throw std::runtime_error("capture: cannot read blob " + path.string());
        if (sha256(dst, (size_t) ref.size) != ref.sha)
            throw std::runtime_error("capture: blob checksum does not match the record: " + path.string());
    }

    /// Loads one referenced blob and verifies its size and checksum: a store that lost or changed a file is refused.
    void load(const BlobRef& ref, std::vector<uint8_t>& bytes) const {
        const std::filesystem::path path =
            std::filesystem::path(dir_) / "blobs" / (hex(ref.sha) + ".bin");
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("capture: missing blob " + path.string());
        const std::streamoff size = f.tellg();
        if (size < 0 || (uint64_t) size != ref.size)
            throw std::runtime_error("capture: blob size does not match the record: " + path.string());
        f.seekg(0);
        bytes.resize((size_t) ref.size);
        if (ref.size != 0) f.read((char*) bytes.data(), (std::streamsize) ref.size);
        if (!f) throw std::runtime_error("capture: cannot read blob " + path.string());
        if (sha256(bytes.data(), bytes.size()) != ref.sha)
            throw std::runtime_error("capture: blob checksum does not match the record: " + path.string());
    }

private:
    uint32_t read_u32() {
        unsigned char b[4];
        in_.read((char*) b, 4);
        if (!in_) throw std::runtime_error("capture: truncated file in " + dir_);
        return (uint32_t) b[0] | ((uint32_t) b[1] << 8) | ((uint32_t) b[2] << 16) | ((uint32_t) b[3] << 24);
    }
    uint64_t read_u64() {
        unsigned char b[8];
        in_.read((char*) b, 8);
        if (!in_) throw std::runtime_error("capture: truncated file in " + dir_);
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= (uint64_t) b[i] << (8 * i);
        return v;
    }

    std::string dir_;
    std::string manifest_;
    std::ifstream in_;
    bool done_ = false;
};

}  // namespace strata::kolibri::stream::capture
