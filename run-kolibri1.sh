#!/usr/bin/env bash
# run-kolibri1.sh - launch the Kolibri-1 service on this box.
#
# Kolibri-1 is a 47 GB Q4_K_M Mixture-of-Experts model: the expert table lives pinned in host RAM
# (42.90 GiB) and only a bounded slice is mirrored in VRAM, so it runs on a card far smaller than
# the model.  The quant is not a knob worth turning here: Q8_0 doubles the pinned table and the
# per-token weight traffic (roughly half the decode speed) for accuracy this box cannot use.
#
# Usage:
#   ./run-kolibri1.sh                 # foreground, Ctrl-C to stop
#   ./run-kolibri1.sh --detach        # background, logs in $RUNDIR
#   ./run-kolibri1.sh --stop          # stop a detached instance
#   ./run-kolibri1.sh --print-config  # show what would run, launch nothing
#
# Everything is an environment override; the defaults are the measured configuration.
#   MODEL      path to the GGUF            (default /fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf)
#   CTX        context tokens              (default 131072; 32768 leaves more VRAM headroom)
#   PORT/HOST  service address             (default 8080 / 127.0.0.1 - see the binding note below)
#   THREADS    prefill pool lanes          (default: this box's physical cores, one lane each)
#   EXPERT_F16 fp16 expert slots per layer (default 0 = the engine's own path, measured fastest here;
#                                     24 costs about a third of the prompt speed on an R9700)
#   ATTN_MB    attention tile size         (default 512; 1024 may be faster at 32K - re-measure)
#   RES_TOP    resident top-k per layer    (default 56, clamped to the slot count)
#   VRAM_GUARD_PCT  warn when the card already has this share of its VRAM in use (default 25)
#   GPU_NAME   the card this launcher is allowed to use (default "AMD Radeon AI PRO R9700").
#              The card is chosen by name and HIP_VISIBLE_DEVICES is pinned to it: this box has a
#              second card (a Radeon PRO W7800 48GB) and an iGPU, and Kolibri-1 must never land on
#              them.  If the named card is not there, the launcher stops instead of guessing.
#   HIP_VISIBLE_DEVICES  set it to override the choice; it is checked against GPU_NAME and the run
#                  is refused when it points at a different card
#   API_MONITOR=1                          expose /api/requests history
#   PYTHON     interpreter                 (auto-detected: must have the 'regex' package,
#                                          which the server's tokenizer imports)
#   EXE        engine binary               (default ./build/strata-kolibri)
#   TOKENIZER  tokenizer pack dir          (default ./.tok/tokenizer)
#
# WHAT THE DEFAULTS ARE, AND WHERE THEY COME FROM.  ATTN_MB / RES_TOP / the router tiles are the
# configuration measured on the CUDA campaign box (16 GB Tesla T4, 27 pinnable cores) -
# docs/strateu-campaign-report.md.  On the AMD Radeon AI PRO R9700 (gfx1201, 32 GB) they are
# starting points, not verdicts: that report's §9.3 names them as the verdicts to re-derive on a
# 32 GB card rather than inherit.  EXPERT_F16 has been re-derived here and is 0: see the sweep below.
#
# EXPERT_F16 (measured 2026-10-28 on this box: Radeon AI PRO R9700, gfx1201, Ryzen 7 5700G, 93 GiB,
# Q4_K_M, ctx 131072, 8 lanes, a 512-token prompt and 128 greedy tokens per request, two requests
# per start, one arm per load): with the fp16 expert cache at 24 slots/layer the prompt read
# 57.6/70.1 and 60.1/71.8 tok/s; at 0 it read 108.3/116.3, 102.0/122.3 and 74.7/83.4 tok/s - every
# run faster than every run with the cache, decode the same either way (15.8-16.2 tok/s both).  The
# cache is the T4's prefill helper (it pays there with PF_PARTIAL); here it only costs the VRAM and
# the time.  0 is also the engine's own default, i.e. the path the bitwise oracle measures.
#
# BINDING.  HOST defaults to 127.0.0.1.  Do not widen it to 0.0.0.0 without --api-key: an open
# port on a machine that is not yours is an open inference endpoint (AGENTS.md).
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

MODEL=${MODEL:-/fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf}
PORT=${PORT:-8080}
HOST=${HOST:-127.0.0.1}
CTX=${CTX:-131072}
EXPERT_F16=${EXPERT_F16:-0}
ATTN_MB=${ATTN_MB:-512}
RES_TOP=${RES_TOP:-56}
VRAM_GUARD_PCT=${VRAM_GUARD_PCT:-25}
GPU_NAME=${GPU_NAME:-AMD Radeon AI PRO R9700}
PYTHON=${PYTHON:-python3}
EXE=${EXE:-$HERE/build/strata-kolibri}
TOKENIZER=${TOKENIZER:-$HERE/.tok/tokenizer}
RUNDIR=${RUNDIR:-${XDG_CACHE_HOME:-$HOME/.cache}/strateu}
API_MONITOR=${API_MONITOR:-0}

# One prefill lane per pinnable physical core.  The pool wants a physical core to itself; a lane
# that shares a core with another lane makes the per-chunk barrier wait for whichever lane got
# descheduled, which is a placement problem and looks like a slow kernel.  SMT siblings do not
# count as separate lanes.  The CUDA box reported 27 pinnable cores and used 24; this box is
# derived, not copied.
physical_cores() {
  local n=""
  if command -v lscpu >/dev/null; then
    n=$(lscpu -p=CPU,CORE,SOCKET 2>/dev/null | grep -v '^#' | grep -v '^$' | sort -u -t, -k2,3 | wc -l)
  fi
  if [ -z "$n" ] || [ "$n" -lt 1 ] 2>/dev/null; then
    n=$(nproc)
  fi
  echo "$n"
}
THREADS=${THREADS:-$(physical_cores)}

MODE=run
for a in "$@"; do
  case "$a" in
    --detach)       MODE=detach ;;
    --stop)         MODE=stop ;;
    --print-config) MODE=print ;;
    -h|--help)      awk 'NR>1{ if (/^#/) { sub(/^# ?/,""); print; next } exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "run-kolibri1.sh: unknown option '$a' (try --help)" >&2; exit 2 ;;
  esac
done

PATTERN='serve[.]server --engine strata'

if [ "$MODE" = stop ]; then
  pids=$(pgrep -f "$PATTERN" || true)
  if [ -z "$pids" ]; then echo "run-kolibri1.sh: nothing to stop"; exit 0; fi
  echo "stopping: $pids"
  kill $pids 2>/dev/null || true
  for _ in $(seq 1 30); do pgrep -f "$PATTERN" >/dev/null || { echo "stopped"; exit 0; }; sleep 1; done
  echo "run-kolibri1.sh: still running, sending SIGKILL" >&2; pkill -9 -f "$PATTERN" || true; exit 0
fi

# ---- preflight: fail loudly here rather than halfway through a 47 GB load -------------
# The server imports `regex` for the tokenizer pack; a bare system python3 often lacks it, so pick an
# interpreter that has it (an explicit PYTHON wins, then the repo's own venvs, then a sibling checkout's).
has_regex() { [ -n "${1:-}" ] && [ -x "$1" ] && "$1" -c 'import regex' >/dev/null 2>&1; }
PY=""
for c in "${PYTHON:-}" "$HERE/.venv/bin/python" "$HERE/venv/bin/python" \
         "$HERE/../.venv/bin/python" "$HERE/../Strata/.venv/bin/python" \
         "$(command -v python3 2>/dev/null || true)" "$(command -v python 2>/dev/null || true)"; do
  if has_regex "$c"; then PY=$c; break; fi
done
if [ "$MODE" = print ]; then
  # --print-config reports what would run; a missing build or interpreter is a line of output here,
  # not an error, so the config can be inspected before the engine has been compiled.
  [ -n "$PY" ] || echo "WARNING: no python with 'regex' - run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
else
  if [ -z "$PY" ]; then
    echo "run-kolibri1.sh: no Python interpreter with the 'regex' package was found." >&2
    echo "  Fix: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    echo "  or point at one: PYTHON=/path/to/venv/bin/python $0" >&2
    exit 1
  fi
fi
[ -n "$PY" ] && PYTHON=$PY

# A missing model or tokenizer is always an error; a missing engine is only a print-mode warning,
# because --print-config is also the "what would run" check you run before building.
[ -f "$MODEL" ] || { echo "run-kolibri1.sh: model not found: $MODEL" >&2; exit 1; }
if [ ! -x "$EXE" ]; then
  if [ "$MODE" = print ]; then
    echo "WARNING: engine not built: $EXE" >&2
  else
    echo "run-kolibri1.sh: engine not built: $EXE" >&2
    echo "  build it:  cmake --build build -j $THREADS --target strata-kolibri" >&2
    exit 1
  fi
fi
[ -f "$TOKENIZER/vocab.json" ] || { echo "run-kolibri1.sh: tokenizer pack missing: $TOKENIZER/vocab.json" >&2; exit 1; }
if [ "$MODE" != print ] && command -v ss >/dev/null && ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "run-kolibri1.sh: port $PORT is already in use (another instance? ./run-kolibri1.sh --stop)" >&2; exit 1
fi

# ---- which card, and only this card -------------------------------------------------
# This box has three GPUs the runtime can see: the Radeon AI PRO R9700 (gfx1201, 32 GB) this
# launcher is written for, a Radeon PRO W7800 48GB (gfx1100) that belongs to something else, and a
# Cezanne iGPU.  Left unset, HIP_VISIBLE_DEVICES exposes all three and the engine takes device 0 -
# which is the right card today only because rocm-smi and HIP happen to enumerate in the same order
# here.  So the card is resolved by name, pinned for the engine, and re-checked: a wrong card is a
# wrong measurement, and a 48 GB card would make every number here look better than the R9700 is.
# rocm-smi and HIP agree on the index on this box (checked with hipGetDeviceProperties against
# rocm-smi --showproductname: hip[0]=R9700/gfx1201, hip[1]=W7800/gfx1100, hip[2]=iGPU/gfx90c).
gpu_index_by_name() {
  command -v rocm-smi >/dev/null || return 1
  rocm-smi --showproductname --csv 2>/dev/null | awk -F, -v want="$GPU_NAME" '
    tolower($2) ~ tolower(want) { sub(/^card/, "", $1); print $1; exit }'
}
WANT_IDX=$(gpu_index_by_name || true)
if [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
  if [ -z "$WANT_IDX" ]; then
    echo "run-kolibri1.sh: no card matching '$GPU_NAME' was found - refusing to guess a GPU."
    echo "  Cards present: $(rocm-smi --showproductname --csv 2>/dev/null | awk -F, 'NR>1{printf "%s=%s ", $1, $2}')" >&2
    echo "  Override with GPU_NAME=<series> or HIP_VISIBLE_DEVICES=<index>." >&2
    exit 1
  fi
  export HIP_VISIBLE_DEVICES=$WANT_IDX
elif [ -n "$WANT_IDX" ] && [ "${HIP_VISIBLE_DEVICES%%,*}" != "$WANT_IDX" ]; then
  echo "run-kolibri1.sh: HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES is not the '$GPU_NAME' card (that is card $WANT_IDX) - refusing to run on another GPU." >&2
  exit 1
fi

# One instance per card: the resident footprint is a large slice of the card, and a second instance
# on top of it OOMs.  The CUDA box read this from nvidia-smi; on AMD it comes from rocm-smi, as
# bytes, for the card the engine will actually get (HIP_VISIBLE_DEVICES, else the first card).
# The threshold is a share of the card rather than a fixed MiB, so the same guard means the same
# thing on a 16 GB, a 32 GB and a 48 GB card.
vram_used_pct() {
  local idx=${HIP_VISIBLE_DEVICES%%,*}
  idx=${idx:-0}
  if command -v rocm-smi >/dev/null; then
    rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, -v i="$idx" '
      $1=="card"i || $1=="card"i"" { if ($2+0 > 0) printf "%d", ($3*100)/$2 }'
  elif command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null |
      awk -F', *' -v i="$idx" 'NR==i+1 && $2+0>0 { printf "%d", ($1*100)/$2 }'
  fi
}
USED_PCT=$(vram_used_pct || true)
if [ -n "${USED_PCT:-}" ] && [ "$USED_PCT" -gt "$VRAM_GUARD_PCT" ] 2>/dev/null; then
  echo "run-kolibri1.sh: WARNING: ${USED_PCT}% of this card's VRAM is already in use (guard: ${VRAM_GUARD_PCT}%) - a second Kolibri-1 instance will OOM" >&2
fi

mkdir -p "$RUNDIR"
CFG="$RUNDIR/strata.json"
"$PYTHON" - "$CFG" "$EXE" "$MODEL" "$HERE" "$CTX" "$THREADS" \
          "$EXPERT_F16" "$ATTN_MB" "$RES_TOP" "$HIP_VISIBLE_DEVICES" "$TOKENIZER" <<'PY'
import json, sys
cfg, exe, model, cwd, ctx, threads, slots, attn_mb, res_top, gpu, tokenizer = sys.argv[1:12]
json.dump({
    # the engine's own flags live in "args": the serve front-end passes them verbatim
    "exe": exe,
    "model_name": "kolibri-1",
    # the family: the server takes the model card's sampling and its own turn/stop handling from it, and the auto-tuner
    # (strata tune) reads it to pick this family's levers - its arguments alone look like a single-GGUF flash-next run
    "arch": "kolibri1",
    # the card, by the index resolved above: a config that names none leaves the tuner reading the smallest card in
    # the machine, which on a box with an iGPU is not the card the model runs on
    "gpu": int(gpu),
    # the pack the tokenizer and the chat template are read from: the app and `strata tune` read it here, the
    # server also accepts --tokenizer on the command line
    "tokenizer": tokenizer,
    "cwd": cwd,
    "args": ["--model", model, "--prefill", "--cpu-threads", str(threads), "--ctx", str(ctx)],
    # the measured configuration (see STRATEU.md): staging on, bounded fp16 residency,
    # tiled router, larger attention tile.  These are the CUDA box's measured values; on AMD
    # they are the A/B baseline the §9.3 re-derivations start from.
    "env": {
        # the card resolved by name above, kept in the config so a start that does not go through this script (the
        # tuner, the app) gets the same one: without it the runtime exposes every card and the engine takes device 0
        "HIP_VISIBLE_DEVICES": gpu,
        "STRATA_KOLIBRI_PF_STAGE": "1",
        "STRATA_KOLIBRI_EXPERT_F16": str(slots),
        "STRATA_KOLIBRI_ROUTER_TILE": "1",
        "STRATA_KOLIBRI_ROUTER_TK": "3",
        "STRATA_KOLIBRI_ROUTER_TJ": "16",
        "STRATA_KOLIBRI_ATTN_MB": str(attn_mb),
        "STRATA_KOLIBRI_RES_TOP": str(res_top),
        # ggml's CUDA backend is what a HIP build compiles too, so this graph kill-switch
        # applies on AMD as well; the campaign found plain paths safer with graphs off.
        "GGML_CUDA_DISABLE_GRAPHS": "1",
    },
    # derive from the CONFIG path, not sys.argv[0]: inside a heredoc argv[0] is "-" (the script is stdin)
    "log": cfg.rsplit("/", 1)[0] + "/engine.log",
}, open(cfg, "w"), indent=2)
PY

MON=""
[ "$API_MONITOR" = "1" ] && MON="--api-monitor"

if [ "$MODE" = print ]; then
  echo "engine : $EXE"
  echo "model  : $MODEL  ($(du -sh "$MODEL" 2>/dev/null | cut -f1))"
  echo "python : ${PYTHON:-<none with regex>}"
  echo "ctx    : $CTX tokens   threads: $THREADS   fp16 slots/layer: $EXPERT_F16"
  echo "serve  : http://$HOST:$PORT   (api-monitor: $API_MONITOR)"
  echo "gpu    : '$GPU_NAME' -> HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
  echo "vram   : card $HIP_VISIBLE_DEVICES at ${USED_PCT:-?}% used (guard ${VRAM_GUARD_PCT}%)"
  echo "config : $CFG"; sed 's/^/           /' "$CFG"; exit 0
fi

ARGS=(--engine strata --config "$CFG" --tokenizer "$TOKENIZER" --host "$HOST" --port "$PORT")
[ -n "$MON" ] && ARGS+=("$MON")

echo "starting Kolibri-1: ctx $CTX, $THREADS lanes, $EXPERT_F16 fp16 slots/layer on '$GPU_NAME' (device $HIP_VISIBLE_DEVICES)"
echo "  UI   : http://${HOST}:$PORT"
echo "  API  : http://${HOST}:$PORT/v1/chat/completions"
echo "  logs : $RUNDIR/{server,engine}.log"

if [ "$MODE" = detach ]; then
  setsid nohup "$PYTHON" -m serve.server "${ARGS[@]}" \
      > "$RUNDIR/server.log" 2>&1 < /dev/null &
  echo "  detached; follow with: tail -f $RUNDIR/server.log"
else
  exec "$PYTHON" -m serve.server "${ARGS[@]}"
fi
