# Task 9 - host-copy ceiling diagnosis

## Probe

New standalone `tools/kolibri/transfer_probe.cu` (HIP test-tree target `kolibri_transfer_probe`, not part of the
engine). It allocates one 64 MiB pageable and one 64 MiB pinned source (the plan's cap), copies 1/4/16/64 MiB H2D,
takes the best of five event-timed copies, verifies every byte, and times verification and a CPU memcpy separately.

## Results (identical bytes verified in every row)

| source | 1 MiB | 4 MiB | 16 MiB | 64 MiB |
|---|---:|---:|---:|---:|
| pageable GB/s | 1.86 | 1.96 | 1.99 | 2.00 |
| pinned GB/s | 1.91 | 1.98 | 1.99 | 2.00 |

`HSA_ENABLE_SDMA=1` and `=0` show no improvement over the default: pinned 64 MiB is 33.618/33.618/33.614 ms
(default/SDMA=1/SDMA=0). Pinning does not lift the observed rate; the copy is about 2.0 GB/s in all three processes.
CPU memcpy rates were single-shot and state-dependent (4.5-63 GB/s), so no CPU-rate conclusion is drawn.

The engine's measured expert PCIe traffic is 5.4 MB/token at ~13 tok/s, about 70 MB/s. That rules against sustained
bulk-bandwidth saturation in that sample, but not against burst, small-copy, staging or synchronization costs: at
2.0 GB/s the 5.4 MB token alone is about 2.7 ms of bulk transfer time. This probe does not bound total copy time.

## Decision: REJECT (no switch)

Neither pageable/pinned nor SDMA setting gives a repeatable gain, and bytes are correct. No engine change and no
system/root setting was attempted. The probe is kept as a standalone diagnostic because it records the observed rate and
verifies every byte, and it now exits nonzero on any mismatch; it is not wired into the engine. `task9/topology.txt`
records the read-only root complex, PCIe 16 GT/s x16 link, `rocminfo` and environment. This says nothing about a
zero-transfer workload.

CUDA/SYCL NOT RUN. No server.
