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
export GGML_CUDA_DISABLE_GRAPHS=1
export KOLIBRI_MODEL=/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf
export BENCH_EXTRA="--prefill --counters --cpu-threads 24"
export KOLIBRI_BENCH=/tmp/kolibri_bench_node1.sh   # numactl --cpunodebind=1 --membind=1
