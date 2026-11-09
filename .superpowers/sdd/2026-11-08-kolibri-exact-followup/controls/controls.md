# Task 0 flag-off controls (sealed engine)

Engine SHA-256: `8a4c1fc1f4eece04fee923f763887ec12643ebfdaaae6198f9250b86dd0c63c2` (sealed binary)  
Model: `/home/gbo/models/Kolibri-1-Q4_K_M.gguf` SHA-256 `c2ac1301424441ef210b6de50ce25e8ccf69f86494df53d6ba52ed558456062e`  
Tool: `tools/kolibri/bench_unpinned.py`, no new engine switches, stats off.  
Runtime: ROCm wheels in `LD_LIBRARY_PATH`, `HIP_VISIBLE_DEVICES=1`, Python 3.13.5.  
The first runs of each case used `--timeout 7200`, below the plan's 10800 s minimum. Their reruns use
`--timeout 10800` and reproduce the exact same complete-logit hashes, so the deviation did not mask a timeout.

## control-512-auto

- input 512 tokens; decode 128 steps; wall 287.8 s; watchdog 7200 s
- startup 251.2 s; prefill 21.37 tok/s; decode 11.16 tok/s
- logits 65536000 bytes / 16384000 floats (finite check passed); SHA-256 `6ffefa8f65b3cb8ca451ae77f23ee05ba197bd23e8e25febbf8121d806fa8721`

## control-512-auto-rerun

- input 512 tokens; decode 128 steps; wall 186.5 s; watchdog 10800 s
- startup 143.9 s; prefill 17.30 tok/s; decode 10.70 tok/s
- logits 65536000 bytes / 16384000 floats (finite check passed); SHA-256 `6ffefa8f65b3cb8ca451ae77f23ee05ba197bd23e8e25febbf8121d806fa8721`

## control-8-cache8

- input 8 tokens; decode 32 steps; wall 36.4 s; watchdog 7200 s
- startup 3.4 s; prefill 1.05 tok/s; decode 1.24 tok/s
- logits 16384000 bytes / 4096000 floats (finite check passed); SHA-256 `466f48f21dacab36cd4f7bcabb42d90c5bb32ceaba4aa784cfe9b6a8d027c3d6`

## control-8-cache8-rerun

- input 8 tokens; decode 32 steps; wall 30.2 s; watchdog 10800 s
- startup 3.4 s; prefill 1.20 tok/s; decode 1.57 tok/s
- logits 16384000 bytes / 4096000 floats (finite check passed); SHA-256 `466f48f21dacab36cd4f7bcabb42d90c5bb32ceaba4aa784cfe9b6a8d027c3d6`

Raw JSON and logs are beside this file; `.bin` dumps are not committed.
