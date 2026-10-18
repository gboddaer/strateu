#!/usr/bin/env bash
# S3 gate driver: fp16-KV build first, then the fp32-KV binary for the before/after pair at 32K.
# Kept as a file because the nested quoting in a `nohup bash -c '...'` one-liner is a trap.
set -uo pipefail
cd /data2/develop/strata-speed-wt
M=${M:-/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf}
export STRATA_DUMP_CTX=8192 GGML_CUDA_DISABLE_GRAPHS=1
mkdir -p /tmp/kolibri_s3

bash tools/kolibri/s3_kv_gate.sh /tmp/kolibri_s3 build/strata-kolibri > /tmp/kolibri_s3/fp16.txt 2>&1
echo "FP16_EXIT=$?" >> /tmp/kolibri_s3/fp16.txt

# before/after pair: the same 32K prompt on the pre-S3 (fp32 KV) binary, same flags
python3 -c 'open("/tmp/kolibri_s3/ids_32768.txt","w").write(",".join(str(1000+(i*7919)%100000) for i in range(32768)))'
/tmp/kolibri-f32kv --model "$M" --tokens-file /tmp/kolibri_s3/ids_32768.txt --n 32 --ctx 33024 --prefill \
  > /tmp/kolibri_s3/fp32_tg32k.log 2>&1
echo "FP32_EXIT=$?" >> /tmp/kolibri_s3/fp32_tg32k.log
echo ALLDONE >> /tmp/kolibri_s3/fp16.txt
