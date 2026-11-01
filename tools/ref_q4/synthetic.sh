#!/bin/bash
# The synthetic Qwen4 validation end to end: per preset a random checkpoint of the preset's geometry
# (tools/capy_flash/gen_synthetic.py), its artifact (tools.convert), the FP64 oracle, the HF-BF16
# reference and their floor (ref_dump.py, tf_compare.py), then the engine against the oracle
# (engine_check.sh), on every arm of the preset. A step whose output exists is skipped: after a
# rebuild only the engine runs again.
#
#   tools/ref_q4/synthetic.sh <build-dir> <flash-next-checkpoint> <tokens.ids.txt> <work-dir> [preset...]
#
# <flash-next-checkpoint>: a Qwen3.8-Flash-Next checkpoint, source of the config and tokenizer files.
# <tokens.ids.txt>: at least 160 prompt ids (the oracle scores them; the decode runs force 32 after 128).
# Presets (default: all), 4 layers + MTP, seed 1234:
#   q4d-2560         hidden 2560, dense FFN 6912, PLE; qwen4_bf16
#   q4d-2560-nople   the same without PLE
#   q4d-2560-silu    q4d-2560 with output_gate_type silu (the GDN output gate of Qwen3.5)
#   q4d-5120         hidden 5120, 4 KV heads, FFN 14336, PLE; qwen4_bf16
#   q4d-5120-nople   the same without PLE
#   q4d-5120-hcq8    the q4d-5120 checkpoint with Q8 hyper-connection mixers (qwen4_bf16_hcq8);
#                    arms: small-T route, general route (NINFER_QWEN4_HC_GENERAL=1)
#   q4d-5120-nvfp4   q4d-5120 snapped onto the NVFP4 grid (snap_nvfp4.py), qwen4_nvfp4;
#                    arms: fused routes, generic chain (NINFER_QWEN4_FUSED=0)
# PYTHON: an interpreter with numpy, safetensors, torch and transformers with qwen4_exp (versions in
# tools/README.md; default python3). TPS and RUNS reach engine_check.sh.
# PREPARE_ONLY=1 stops after the CPU steps (checkpoints, artifacts, references), before the GPUs.
# Results: <work-dir>/check-<preset>[-<arm>]/summary.txt and one line per arm in <work-dir>/summary.txt.
# Exit 0 when every arm of every preset passes.
set -u
[ $# -ge 4 ] || { sed -n 2,25p "$0"; exit 2; }
build=$1 source=$2 tokens=$3 work=$4; shift 4
presets=("${@:-q4d-2560 q4d-2560-nople q4d-2560-silu q4d-5120 q4d-5120-nople q4d-5120-hcq8 q4d-5120-nvfp4}")
read -r -a presets <<< "${presets[*]}"
py=${PYTHON:-python3}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
mkdir -p "$work"
log() { echo "$(date +%T) $*" | tee -a "$work/summary.txt"; }

# step <output> <log> <command...>: run unless the output exists; a failure stops the script.
step() {
  local output=$1 logfile=$2; shift 2
  [ -e "$output" ] && return 0
  "$@" > "$logfile" 2>&1 || { log "FAIL $(basename "$logfile" .log): see $logfile"; exit 1; }
}
checkpoint() { # name generator-options...
  local name=$1; shift
  step "$work/$name/model.safetensors.index.json" "$work/$name.gen.log" \
    "$py" "$root/tools/capy_flash/gen_synthetic.py" --source "$source" --out "$work/$name" \
    --layers 4 --mtp --seed 1234 --ffn dense "$@"
}
in_root() { (cd "$root" && "$@"); }
artifact() { # checkpoint recipe artifact-name
  step "$work/$3.ninfer" "$work/$3.convert.log" \
    in_root "$py" -m tools.convert --model "$work/$1" --recipe "$2" --components text,mtp \
    --device cpu --name "capy-flash-$3" --out "$work/$3.ninfer"
}
oracle() { # checkpoint: FP64 oracle, HF-BF16 reference, floor
  local c=$1
  step "$work/$c.fp64.tfdump" "$work/$c.fp64.log" \
    "$py" "$here/ref_dump.py" --checkpoint "$work/$c" --impl fp64 --tokens "$tokens" --mtp --out "$work/$c.fp64.tfdump"
  step "$work/$c.hf.tfdump" "$work/$c.hf.log" \
    "$py" "$here/ref_dump.py" --checkpoint "$work/$c" --impl hf --threads 8 --tokens "$tokens" --mtp --out "$work/$c.hf.tfdump"
  step "$work/$c.floor.json" "$work/$c.floor.log" \
    "$py" "$here/tf_compare.py" "$work/$c.fp64.tfdump" "$work/$c.hf.tfdump" --json "$work/$c.floor.json"
}
failed=0
check() { # checkpoint artifact-name arm [env...]
  local c=$1 a=$2 arm=$3; shift 3
  [ "${PREPARE_ONLY:-0}" = 1 ] && return 0
  local out=$work/check-$a${arm:+-$arm}
  rm -rf "$out"
  env "$@" "$here/engine_check.sh" "$build" "$work/$a.ninfer" "$work/$c" "$work/$c.fp64.tfdump" \
    "$work/$c.floor.json" "$tokens" "$out" > "$out.log" 2>&1
  local rc=$?
  log "$a${arm:+ $arm}: $([ $rc = 0 ] && echo PASS || echo "FAIL rc=$rc") ($(grep -c PASS "$out/summary.txt" 2>/dev/null) PASS, $(grep -c FAIL "$out/summary.txt" 2>/dev/null) FAIL)"
  [ $rc = 0 ] || failed=1
}
for preset in "${presets[@]}"; do
  log "preset $preset"
  case $preset in
    q4d-2560)       checkpoint q4d-2560 --intermediate 6912 ;;
    q4d-2560-nople) checkpoint q4d-2560-nople --intermediate 6912 --no-ple ;;
    q4d-2560-silu)  checkpoint q4d-2560-silu --intermediate 6912 --output-gate silu ;;
    q4d-5120-nople) checkpoint q4d-5120-nople --intermediate 14336 --hidden 5120 --kv-heads 4 --no-ple ;;
    q4d-5120 | q4d-5120-hcq8 | q4d-5120-nvfp4)
      checkpoint q4d-5120 --intermediate 14336 --hidden 5120 --kv-heads 4 ;;
    *) log "unknown preset $preset"; exit 2 ;;
  esac
  if [ "$preset" = q4d-5120-nvfp4 ]; then
    step "$work/q4d-5120-snap/model.safetensors.index.json" "$work/q4d-5120-snap.log" \
      "$py" "$here/snap_nvfp4.py" --checkpoint "$work/q4d-5120" --out "$work/q4d-5120-snap"
  fi
  case $preset in
    q4d-5120-hcq8)
      artifact q4d-5120 qwen4_bf16_hcq8 q4d-5120-hcq8; oracle q4d-5120
      check q4d-5120 q4d-5120-hcq8 small-t
      check q4d-5120 q4d-5120-hcq8 general NINFER_QWEN4_HC_GENERAL=1 ;;
    q4d-5120-nvfp4)
      artifact q4d-5120-snap qwen4_nvfp4 q4d-5120-nvfp4; oracle q4d-5120-snap
      check q4d-5120-snap q4d-5120-nvfp4 fused
      check q4d-5120-snap q4d-5120-nvfp4 generic NINFER_QWEN4_FUSED=0 ;;
    *)
      artifact "$preset" qwen4_bf16 "$preset"; oracle "$preset"
      check "$preset" "$preset" "" ;;
  esac
done
exit $failed
