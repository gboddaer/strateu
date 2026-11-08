# Checked overnight results — 2026-11-08

Run source:6c241aa4, amdw7800probuild, sealed HIP W7800 engine. No server.
Ten-hour deadline reached2026-11-08T02:41:47Z. Controller ended; no sealed-engine processes remain.

54 attempts:49 successful finite complete dumps, including27 bitwise comparisons;4 CLI context-budget
rejections;1 case stopped at the global deadline. This is NOT completion of the entire scheduled matrix.

Fresh audit reopened all49 successful binary dumps:114,688,000 finite floats, expected sizes, nonconstant
rows, hashes identical to their recorded results, and no conflicting forced-history hashes per fixture.
Completed paired lengths:1025,4095,4096,4097,8191,8192,8193,16383,16384,16385,32767,65537.
The32768 demand reference completed, but its balanced comparison hit the deadline.
Small-cache8 vsauto comparison reached16385tokens. No bitwise mismatch or nonfinite completed dump.

Retrieval:9/9 exact first-answer matches CEDAR-7419, using real-tokenizer synthetic English/German/code
records with the needle at2%,50%,95%, at8192,65536,131000 input tokens. These are successful actual
131000-token inputs, not just context allocation. They are one GPU strategy, not independent full-model
fidelity or a general task-quality qualification. No paired full-logit cache-placement check at131000 yet.

Four failed attempts were a CONTROLLER budget error, not an inference crash: it validated
prompt+steps-1<=131072, but engineCLI requires prompt+steps<=ctx (engine7364–7369).
131065+8 and131072+1 were each rejected in auto and cache8 modes before prefill, with exit2 and the
explicit exceeds-ctx message. Preserve original controller as evidence; future runs must follow the CLI
contract. Valid near-edge replacements include131064+8 and131071+1. Do not relax engine guards to pass tests.

Diagnostic-counter prefill rates (not unprofiled production throughput):4097tokens38.90tok/s;
16385tokens77.65;32768tokens76.34;65537tokens51.29;131000-token middle-needle run20.62tok/s.
131000 probes each took roughly106minutes. Long-context attention tiling is a profiling priority,
but these timings alone do not establish the bottleneck. OS cache was not purged.

Not reached:32769/65535/65536 forced comparisons, remaining32768 arms, serial4097 pair and64K soak.
Exact131072-token input was not exercised because of the harness/CLI boundary mismatch.
No engine or optimization code was modified, and no tolerance relaxed.
