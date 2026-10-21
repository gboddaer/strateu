#!/bin/bash
# tools/kolibri/speed_config.sh - THE passing speed configuration for kolibri-1 on this box, in one place.
#
# Measured (the ledger's rows): pp512 4,744 ms = 107.9 tok/s, tg128 6,004 ms = 21.3 tok/s (node-1).
# KL 0.00634 PASS, the plain-path oracle BITWISE.
#
# GOTCHA (cost a full diagnostic round on 2026-10-21): the CPU pool is OPT-IN via --cpu-threads.  Without it
# the engine has no pool, every routed expert takes the per-expert GPU mmvq fallback (measured: 364,350 mmvq
# launches = 26.8 s of 35 s GPU busy) and the prefill is 4.6x SLOWER (22,126 ms vs 4,744 ms; 23.1 vs 107.9
# tok/s) - with NO warning in the log.  The counters show it: "CNT branches per tok: pool 0.0 gpu-expert 50.0".
#
# Usage:  source tools/kolibri/speed_config.sh   (exports the vars; BENCH_EXTRA carries the engine flags)
export STRATA_KOLIBRI_PF_STAGE=1
export STRATA_KOLIBRI_EXPERT_F16=24
# The fp16 promotion budget per chunk-layer (default 48).  Measured curve, 4K prefill, same binary and fixture:
# 12 -> 90.7 tok/s, 24 -> 119.2, 48 -> 134.2, 56 -> 139.0, 64 -> 138.7, 72 -> 133.0, 96 -> 112.4, 192 -> 84.6.
# Promotions buy pool rows (one promotion ~21 rows ~0.85 ms of CPU for ~0.16 ms of PCIe); past ~64 the GPU side
# grows faster than the pool shrinks.  56 is worth +3.6% at BOTH 4K and 32K (32K 119.5 -> 123.8, two runs).
export STRATA_KOLIBRI_RES_TOP=56
export STRATA_KOLIBRI_ROUTER_TILE=1
export STRATA_KOLIBRI_ATTN_MB=256
# K27: the transposed KV mirror for the sliding layers. ON by default (it is NOT a numerics change - same values,
# same accumulation order, contiguous addresses - and the oracle prints BITWISE PASS with it). Measured: tg128
# 22.8 -> 25.7 tok/s (+12.7%), tg after 32K 14.5 -> 15.5 (+7%), prefill untouched. STRATA_KOLIBRI_ATTN_KT=0 is the
# A/B arm. Mirrors are sliding-layer only BY MEASUREMENT: mirroring the full layers too is slower (25.3 vs 25.7)
# because their 11000-half row stride puts each d-step in a new line.
export GGML_CUDA_DISABLE_GRAPHS=1
export KOLIBRI_MODEL=/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf
export BENCH_EXTRA="--prefill --counters --cpu-threads 24"
export KOLIBRI_BENCH=/tmp/kolibri_bench_node1.sh   # numactl --cpunodebind=1 --membind=1
# PREFILL ONLY: both sockets' CPUs for the pool (+7.4% at 32K, +6.1% at 4K; the pool leg goes 114 s -> 89 s at
# 231-236 GFLOP/s against the 226 GFLOP/s node-1 roof, i.e. the roof was a one-socket number).  Decode is SLOWER
# this way (32K-after: tg 13.3-13.5 vs 14.6) because every per-token pool call then crosses the socket link, so
# keep decode on node 1.  Memory stays bound to node 1 (-membind=1): the 42.9 GB pinned expert table lives there.
# NOTE: node 0 also hosts the concurrent work on this box, so this trades some of its bandwidth for ours.
export KOLIBRI_BENCH_BOTH=/tmp/kolibri_bench_both.sh   # numactl --cpunodebind=0,1 --membind=1
