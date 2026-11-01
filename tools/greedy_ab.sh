#!/bin/bash
# Greedy A/B of two `ninfer` runs: two binaries (--bin-b; e.g. a build against a release) or one
# binary under two environments (--env-a / --env-b): the generated token ids of every prompt,
# compared exactly.
#
#   tools/greedy_ab.sh [options] <ninfer> <artifact> [ninfer flags...]
#     --env-a 'K=V ...'   environment of arm A (default: none)
#     --env-b 'K=V ...'   environment of arm B (default: none)
#     --bin-b PATH        arm B's binary (default: <ninfer>)
#     --flags-b 'FLAGS'   ninfer flags of arm B only (a flag the binary of arm A does not have)
#     --prompts FILE      one prompt per line (default: three built-in prompts)
#     --out DIR           logs and summary (default: ./greedy_ab-<time>)
#     --max-new N         generated tokens per prompt (default 64)
# The flags after the artifact are passed to every run (e.g. --tp 2 --devices 0,1 --kv-dtype int8
# --spec mtp --draft-tokens 3 --embedding-host). Exit 0 when every prompt matches, 1 otherwise.
set -u
env_a=""
env_b=""
prompts=""
bin_b=""
flags_b=""
out="greedy_ab-$(date +%Y%m%d-%H%M%S)"
max_new=64
while [ $# -gt 0 ]; do
  case "$1" in
    --env-a) env_a=$2; shift 2 ;;
    --env-b) env_b=$2; shift 2 ;;
    --bin-b) bin_b=$2; shift 2 ;;
    --flags-b) flags_b=$2; shift 2 ;;
    --prompts) prompts=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --max-new) max_new=$2; shift 2 ;;
    *) break ;;
  esac
done
[ $# -ge 2 ] || { sed -n 2,15p "$0"; exit 2; }
bin=$1 artifact=$2; shift 2
bin_b=${bin_b:-$bin}
mkdir -p "$out"
if [ -n "$prompts" ]; then
  mapfile -t list < "$prompts"
else
  list=("Explain in three sentences why the sky is blue."
        "Write a Python function that returns the n-th Fibonacci number iteratively."
        "Qual e la capitale del Portogallo e quanti abitanti ha?")
fi
ids() { sed -n 's/.*generated ids[[:space:]]*//p' "$1" | head -1; }
fail=0
for i in "${!list[@]}"; do
  for arm in a b; do
    e=$env_a run_bin=$bin extra=()
    [ $arm = b ] && { e=$env_b run_bin=$bin_b; read -r -a extra <<< "$flags_b"; }
    env $e "$run_bin" "$artifact" --prompt "${list[$i]}" --greedy --max-new "$max_new" \
      --print-token-ids --no-thinking "$@" "${extra[@]}" > "$out/p$i-$arm.out" 2> "$out/p$i-$arm.err"
    echo "p$i $arm rc=$?" >> "$out/runs.txt"
  done
  a=$(ids "$out/p$i-a.err")
  b=$(ids "$out/p$i-b.err")
  if [ -n "$a" ] && [ "$a" = "$b" ]; then
    verdict=identical
  else
    verdict=DIFFERENT; fail=1
    verdict="$verdict $(awk -v a="$a" -v b="$b" 'BEGIN {
      n = split(a, x, " "); m = split(b, y, " ")
      for (i = 1; i <= (n < m ? n : m); i++) if (x[i] != y[i]) { print "at token " i - 1; exit }
      print "lengths " n "/" m }')"
  fi
  acc_a=$(grep -oE "[a-z0-9]+ acceptance rate[[:space:]]+[0-9.]+" "$out/p$i-a.err" | awk '{print $4}')
  acc_b=$(grep -oE "[a-z0-9]+ acceptance rate[[:space:]]+[0-9.]+" "$out/p$i-b.err" | awk '{print $4}')
  echo "p$i: $verdict (acceptance A ${acc_a:--} B ${acc_b:--})" | tee -a "$out/summary.txt"
done
exit $fail
