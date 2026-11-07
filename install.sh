#!/usr/bin/env bash
# capy-flash installer: checks this machine, downloads a release and a model, verifies both and prints
# the command that serves it. Nothing is installed system-wide; everything goes under --dir.
#
#   curl -fsSL https://raw.githubusercontent.com/ValerioDolci/capy-flash/main/install.sh | bash -s -- [options]
#
#   --dir DIR          where the release and the model go (default ~/capy-flash)
#   --model NAME       flash-next (default, Qwen3.8-Flash-Next, 126.6 GB), flash-next-64gb (64 GB of RAM,
#                      102 GB), 27b (Qwen3.8-27B, two GPUs, 17.4 GB) or 27b-16gb (one GPU, 3-bit, 12.5 GB)
#   --version vX.Y.Z   release tag (default: the latest release)
#   --check            only check the machine, download nothing
#   --no-model         download the release only
#   --verify-model     also check the model's SHA-256 digests (reads every byte: minutes for Flash-Next)
set -euo pipefail

usage() {
  cat <<'EOF'
usage: install.sh [--dir DIR] [--model flash-next|flash-next-64gb|27b|27b-16gb] [--version vX.Y.Z] [--check] [--no-model] [--verify-model]
  --dir DIR          where the release and the model go (default ~/capy-flash)
  --model NAME       flash-next (default, Qwen3.8-Flash-Next, 126.6 GB), flash-next-64gb (64 GB of RAM,
                     102 GB), 27b (Qwen3.8-27B, two GPUs, 17.4 GB) or 27b-16gb (one GPU, 3-bit, 12.5 GB)
  --version vX.Y.Z   release tag (default: the latest release)
  --check            only check the machine, download nothing
  --no-model         download the release only
  --verify-model     also check the model's SHA-256 digests (reads every byte: minutes for Flash-Next)
EOF
}

repo=ValerioDolci/capy-flash
dir=$HOME/capy-flash
model=flash-next
launcher_flags=""
version=
check_only=0
fetch_model=1
verify_model=0
while [ $# -gt 0 ]; do
  case $1 in
    --dir) dir=$2; shift 2 ;;
    --model) model=$2; shift 2 ;;
    --version) version=$2; shift 2 ;;
    --check) check_only=1; shift ;;
    --no-model) fetch_model=0; shift ;;
    --verify-model) verify_model=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "install.sh: unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
done

case $model in
  flash-next)
    hf_repo=Feyd89/Qwen3.8-Flash-Next-NVFP4-capy-flash
    artifact=qwen3_8_flash_next_nvfp4_idxgdnq8.ninfer
    launcher=serve-flash-next.sh
    model_id=capy-flash
    need_ram_gib=70 need_disk_gb=135 need_simd=1 need_gpus=2 ;;
  flash-next-64gb)
    # v0.7: GSQ-RCO i-quant experts (48 GiB page-locked instead of 65); small RAM caches for 64 GB.
    hf_repo=Feyd89/Qwen3.8-Flash-Next-GSQ-RCO-capy-flash
    artifact=qwen3_8_flash_next_gsqrco_mtpB.ninfer
    launcher=serve-flash-next.sh
    launcher_flags=" --host-kv-mib 1024 --host-state-slots 2"
    model_id=capy-flash
    need_ram_gib=56 need_disk_gb=110 need_simd=1 need_gpus=1 ;;
  27b-16gb)
    # v0.7: ISTA-DASLab's 3-bit GSQ weights on ONE 16 GB GPU (MMLU-Pro below the NVFP4 27B: see the README).
    hf_repo=Feyd89/Qwen3.8-27B-GSQ-3bit-capy-flash
    artifact=qwen3_8_27b_gsq3.ninfer
    launcher=serve-27b-16gb.sh
    model_id=qwen3.8-27b
    need_ram_gib=8 need_disk_gb=15 need_simd=0 need_gpus=1 ;;
  27b)
    hf_repo=Feyd89/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer
    artifact=qwen3_8_27b_quasar_nvfp4.ninfer
    launcher=serve-tp2.sh
    model_id=qwen3.8-27b
    need_ram_gib=8 need_disk_gb=20 need_simd=0 need_gpus=2 ;;
  *) echo "install.sh: --model must be flash-next, flash-next-64gb, 27b or 27b-16gb" >&2; exit 2 ;;
esac

failed=0
ok()   { printf '  ok    %s\n' "$*"; }
warn() { printf '  WARN  %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; failed=1; }

echo "capy-flash: checking this machine for $model"
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] && ok "Linux x86_64" \
  || fail "Linux x86_64 is required (this is $(uname -s) $(uname -m))"

if command -v nvidia-smi > /dev/null 2>&1; then
  gpus=$(nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader,nounits 2>/dev/null || true)
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  usable=$(printf '%s\n' "$gpus" | awk -F', ' '$3 >= 15000 && $4 == "12.0"' | wc -l)
  if [ "$usable" -ge "$need_gpus" ]; then
    ok "$usable sm_120 GPUs with 16 GB ($(printf '%s\n' "$gpus" | head -1 | cut -d, -f2 | sed 's/^ //'))"
  else
    fail "$need_gpus sm_120 GPU(s) (RTX 50 series) with 16 GB are needed; found:"
    printf '%s\n' "$gpus" | sed 's/^/          /'
  fi
  if [ "${driver%%.*}" -ge 580 ] 2>/dev/null; then ok "NVIDIA driver $driver"
  else fail "NVIDIA driver $driver is too old: CUDA 13.1 needs R580 or newer"; fi
else
  fail "nvidia-smi not found: install the NVIDIA driver (R580 or newer)"
fi

if [ "$need_simd" = 1 ]; then
  # The experts outside VRAM run on the CPU: AVX-512 VNNI kernels, or AVX2 + FMA + F16C ones (same
  # results bit for bit), picked at startup.
  flags=$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null || true)
  lacks() { for f in "$@"; do case " $flags " in *" $f "*) ;; *) printf ' %s' "$f" ;; esac; done; }
  missing512=$(lacks avx512f avx512bw avx512vl avx512_vnni avx2 fma f16c)
  missing2=$(lacks avx2 fma f16c)
  if [ -z "$missing512" ]; then ok "CPU has AVX-512 F/BW/VL/VNNI"
  elif [ -z "$missing2" ]; then ok "CPU has AVX2/FMA/F16C (no AVX-512 VNNI: the AVX2 expert kernels run)"
  else fail "the CPU lacks$missing2: Flash-Next computes the experts outside VRAM with AVX2 or AVX-512"; fi
fi

total_gib=$(awk '/^MemTotal:/ {printf "%d", $2 / 1048576}' /proc/meminfo)
avail_gib=$(awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo)
if [ "$total_gib" -lt "$need_ram_gib" ]; then
  fail "about $need_ram_gib GiB of free RAM are needed, the machine has $total_gib GiB in total"
elif [ "$avail_gib" -lt "$need_ram_gib" ]; then
  warn "$avail_gib GiB of RAM free now, $need_ram_gib GiB needed at startup: stop other models before serving"
else
  ok "RAM: $avail_gib GiB free of $total_gib GiB"
fi

probe=$dir
while [ ! -d "$probe" ]; do probe=$(dirname "$probe"); done # --check creates nothing
free_gb=$(df -P -B1000000000 "$probe" | awk 'NR == 2 {print $4}')
if [ "$fetch_model" = 1 ] && [ ! -e "$dir/models/$model/$artifact" ] && [ "$free_gb" -lt "$need_disk_gb" ]; then
  fail "$need_disk_gb GB of disk are needed in $dir, $free_gb GB free (choose another --dir)"
else
  ok "disk: $free_gb GB free in $dir"
fi

if [ "$fetch_model" = 1 ]; then
  if command -v hf > /dev/null 2>&1; then ok "Hugging Face CLI (hf)"
  else fail "the Hugging Face CLI is missing: pip install -U 'huggingface_hub[cli]' (or pipx install huggingface_hub)"; fi
fi
command -v curl > /dev/null 2>&1 || fail "curl is missing"

if [ "$failed" = 1 ]; then echo "capy-flash: fix the FAIL lines above and run again."; exit 1; fi
[ "$check_only" = 1 ] && { echo "capy-flash: this machine can serve $model."; exit 0; }
mkdir -p "$dir"

if [ -z "$version" ]; then
  version=$(curl -fsSL "https://api.github.com/repos/$repo/releases/latest" | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)
  [ -n "$version" ] || { echo "capy-flash: cannot find the latest release; pass --version" >&2; exit 1; }
fi
tarball=capy-flash-$version-linux-x86_64-sm120a-ubuntu26.04.tar.gz
release=$dir/capy-flash-$version
if [ ! -x "$release/bin/ninfer-serve" ]; then
  echo "capy-flash: downloading release $version"
  url=https://github.com/$repo/releases/download/$version
  curl -fL --retry 3 -C - --progress-bar -o "$dir/$tarball" "$url/$tarball"
  curl -fsSL -o "$dir/$tarball.sha256" "$url/$tarball.sha256"
  (cd "$dir" && sha256sum -c "$tarball.sha256") || { echo "capy-flash: the release download is corrupt; delete $dir/$tarball and retry" >&2; exit 1; }
  unpack=$(mktemp -d "$dir/.unpack.XXXXXX")
  tar -xzf "$dir/$tarball" -C "$unpack"
  rm -rf "$release"
  mv "$unpack"/*/ "$release" # the tarball holds one top directory
  rmdir "$unpack"
  rm -f "$dir/$tarball" "$dir/$tarball.sha256"
fi
(cd "$release" && sha256sum --quiet -c SHA256SUMS) || {
  echo "capy-flash: release verification failed in $release; installation stopped" >&2
  exit 1
}
ok "release $version verified ($release)"
ln -sfn "$release" "$dir/current" # stable path for units and llama-swap (examples/serving)
missing_libs=$(LD_LIBRARY_PATH="$release/lib" ldd "$release/bin/ninfer-serve" | awk '/not found/ {print $1}' | tr '\n' ' ')
if [ -n "$missing_libs" ]; then
  warn "missing libraries: $missing_libs"
  echo "        on Ubuntu/Debian: sudo apt install libavcodec-dev libavformat-dev libswscale-dev libcurl4"
  echo "        (the tarball is built on Ubuntu 26.04; on another distribution use the Dockerfile)"
fi

if [ "$fetch_model" = 1 ]; then
  echo "capy-flash: downloading $hf_repo (resumable: run again if it stops)"
  hf download "$hf_repo" --local-dir "$dir/models/$model"
  if [ "$verify_model" = 1 ]; then
    (cd "$dir/models/$model" && sha256sum -c SHA256SUMS) || {
      echo "capy-flash: model verification failed in $dir/models/$model; installation stopped" >&2
      exit 1
    }
    ok "model verified"
  fi
fi

port=8080
cat <<EOF

capy-flash $version is ready. Serve $model with:

  $dir/current/$launcher $dir/models/$model/$artifact$launcher_flags
$( [ "$model" = flash-next-64gb ] && echo "
(one GPU: NINFER_DEVICES=0 $dir/current/$launcher $dir/models/$model/$artifact)" )
$( case "$model" in 27b|27b-16gb) echo "
(text only, more room for the KV: NINFER_VISION=0 $dir/current/$launcher $dir/models/$model/$artifact)" ;; esac )

then, from another shell (the first start takes a minute or two):

  curl -s http://127.0.0.1:$port/v1/chat/completions -H 'Content-Type: application/json' \\
    -d '{"model":"$model_id","messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":512}'

NINFER_PORT=9000 changes the port; extra ninfer-serve flags go after the model path.
To run it as a service: examples/serving in https://github.com/$repo (systemd unit, llama-swap entry).
EOF
