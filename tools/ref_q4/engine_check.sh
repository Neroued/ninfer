#!/bin/bash
# A synthetic Qwen4 artifact's engine forward against the FP64 oracle (synthetic.sh prepares the
# artifact and the references and runs this per preset). Per tp: the validation driver
# (<build-dir>/tests/ninfer_qwen4_tf_driver) scores the prompt in one chunk (`score`) and in chunks
# of 64 (`chunk64`), runs every prompt column through the MTP layer (`tfmtp`, one device only), and
# decodes 32 forced greedy tokens after the first 128 (`dec`, ordinary rounds with CUDA Graph;
# `decmtp`, MTP K=3 verify rounds compared on the engine's own sequence). Each dump is compared with
# tf_compare.py against the oracle under the bf16 profile (2 x floor), which decides, and the w4a4
# profile (10 x floor), reported for information only: it is strictly less severe, so it cannot fail
# where bf16 passes.
#
#   tools/ref_q4/engine_check.sh <build-dir> <artifact.ninfer> <checkpoint-dir> \
#       <oracle.fp64.tfdump> <floor.json> <prompt.ids.txt> <out-dir>
#
# PYTHON: an interpreter with numpy (tf_compare.py), default python3. TPS: tp widths
# (default "1 2"). RUNS: subset of "score chunk64 tfmtp dec decmtp". The caller's environment
# reaches every run (e.g. NINFER_QWEN4_FUSED=0). Exit 0 when every comparison passes bf16.
set -u
[ $# -eq 7 ] || { sed -n 2,17p "$0"; exit 2; }
build=$1 artifact=$2 checkpoint=$3 oracle=$4 floor=$5 tokens=$6 out=$7
py=${PYTHON:-python3}
here=$(cd "$(dirname "$0")" && pwd)
driver=$build/tests/ninfer_qwen4_tf_driver
mkdir -p "$out"
tr -s " \n" "\n" < "$tokens" | grep . | head -128 | tr "\n" " " > "$out/prompt128.ids.txt"
# tokens.npy of a generate-mode dump: the run's prompt + generated ids.
add_tokens() {
  "$py" - "$1" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
dump = Path(sys.argv[1])
if not (dump / "tokens.npy").exists():
    ids = json.loads(Path(str(dump) + ".json").read_text().splitlines()[0])["ids"]
    np.save(dump / "tokens.npy", np.array(ids, np.int64))
PY
}
compare() { # dump reference
  local dump=$1 ref=$2 prof
  for prof in w4a4 bf16; do
    "$py" "$here/tf_compare.py" "$ref" "$dump" --profile $prof --floor "$floor" --json "$dump.$prof.json" \
      > "$dump.$prof.txt" 2>&1
    local rc=$?
    local label=$prof
    [ $prof = w4a4 ] && label="w4a4 (information)"
    echo "$(basename "$dump") $label $([ $rc = 0 ] && echo PASS || echo "FAIL rc=$rc") :: $(grep -E '^(logits|mtp_logits)' "$dump.$prof.txt" | head -1 | cut -c1-120)" \
      | tee -a "$out/summary.txt"
    [ $rc = 0 ] || [ $prof = w4a4 ] || failed=1
  done
}
for run in ${RUNS:-}; do
  case $run in score | chunk64 | tfmtp | dec | decmtp) ;; *) echo "unknown run $run" >&2; exit 2 ;; esac
done
failed=0
for tp in ${TPS:-1 2}; do
  base=$out/tp$tp
  common=(--kv bf16 --tp "$tp" --context 4096)
  for run in ${RUNS:-score chunk64 tfmtp dec decmtp}; do
    dump=$base.$run
    rm -rf "$dump"
    case $run in
      score)   NINFER_QWEN4_TFDUMP=$dump "$driver" score "$artifact" "$tokens" "${common[@]}" --out "$dump.json" ;;
      chunk64) NINFER_QWEN4_TFDUMP=$dump "$driver" score "$artifact" "$tokens" "${common[@]}" --chunk 64 --out "$dump.json" ;;
      tfmtp)   [ "$tp" = 1 ] || continue
               NINFER_QWEN4_TFDUMP=$dump NINFER_QWEN4_TFDUMP_MTP=1 NINFER_QWEN4_TFDUMP_LAYERS=none \
                 "$driver" generate "$artifact" "$tokens" "${common[@]}" --max-new 1 --mtp 3 --no-cuda-graph --out "$dump.json" ;;
      dec)     NINFER_QWEN4_VALIDATION=1 NINFER_QWEN4_TFDUMP=$dump NINFER_QWEN4_TFDUMP_LAYERS=none NINFER_QWEN4_TFDUMP_FORCE=$tokens \
                 "$driver" generate "$artifact" "$out/prompt128.ids.txt" "${common[@]}" --max-new 32 --out "$dump.json" ;;
      decmtp)  NINFER_QWEN4_VALIDATION=1 NINFER_QWEN4_TFDUMP=$dump NINFER_QWEN4_TFDUMP_LAYERS=none NINFER_QWEN4_TFDUMP_FORCE=$tokens \
                 "$driver" generate "$artifact" "$out/prompt128.ids.txt" "${common[@]}" --max-new 32 --mtp 3 --out "$dump.json" ;;
    esac > "$dump.log" 2>&1
    rc=$?
    [ $rc = 0 ] || { echo "$(basename "$dump") run FAIL rc=$rc" | tee -a "$out/summary.txt"; failed=1; continue; }
    add_tokens "$dump"
    if [ $run = decmtp ]; then
      # MTP rounds are not forced: the reference follows the engine's own sequence.
      "$py" -c 'import sys, numpy as np; print(" ".join(map(str, np.load(sys.argv[1]))))' "$dump/tokens.npy" \
        > "$dump.ids.txt"
      "$py" "$here/ref_dump.py" --checkpoint "$checkpoint" --impl fp64 --tokens "$dump.ids.txt" --layers none \
        --out "$dump.fp64.tfdump" > "$dump.fp64.log" 2>&1 || { echo "$(basename "$dump") reference FAIL" | tee -a "$out/summary.txt"; failed=1; continue; }
      compare "$dump" "$dump.fp64.tfdump"
    else
      compare "$dump" "$oracle"
    fi
  done
done
exit $failed
