#!/bin/bash
set -e

export TMPDIR=/hy-tmp/tmp
export TEMP=/hy-tmp/tmp
export TMP=/hy-tmp/tmp
export RAY_TMPDIR=/hy-tmp/ray_tmp
export RAY_TEMP_DIR=/hy-tmp/ray_tmp   # 注意：有些版本用 RAY_TEMP_DIR 而非 --temp-dir
export TRITON_CACHE_DIR=/hy-tmp/triton_cache
export TORCH_EXTENSIONS_DIR=/hy-tmp/torch_extensions
export XDG_CACHE_HOME=/hy-tmp/.cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p \
  /hy-tmp/tmp \
  /hy-tmp/ray_tmp \
  /hy-tmp/triton_cache \
  /hy-tmp/torch_extensions \
  /hy-tmp/.cache

ray stop --force 2>/dev/null || true
ray start --head --temp-dir=/hy-tmp/ray_tmp
