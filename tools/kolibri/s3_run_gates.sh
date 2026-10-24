#!/usr/bin/env bash
# S3 gate driver: the fp16-KV build gets the full gate (linearity + after-32K); the pre-S3 fp32-KV binary
# gets the after-32K point only, which is the number the before/after ratio needs.
# Kept as a file: the nested quoting of a `nohup bash -c '...'` one-liner is a trap (it silently killed a
# launch on 2026-10-18 by breaking on `","` inside a nested python -c).
set -uo pipefail
cd $(dirname "$0")/../..
M=${M:-/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf}
mkdir -p /tmp/kolibri_s3 /tmp/kolibri_s3_fp32

bash tools/kolibri/s3_kv_gate.sh /tmp/kolibri_s3 build/strata-kolibri > /tmp/kolibri_s3/fp16.txt 2>&1
echo "FP16_EXIT=$?" >> /tmp/kolibri_s3/fp16.txt

# before/after pair: the same 512-short + 32768-after-32K pass on the pre-S3 (fp32 KV) binary
bash tools/kolibri/s3_kv_gate.sh /tmp/kolibri_s3_fp32 /tmp/kolibri-f32kv > /tmp/kolibri_s3/fp32.txt 2>&1
echo "FP32_EXIT=$?" >> /tmp/kolibri_s3/fp32.txt
echo ALLDONE >> /tmp/kolibri_s3/fp16.txt
