#!/usr/bin/env bash

export CUDA_HOME=/usr/local/cuda-12.2
export PATH="${CUDA_HOME}/bin:${PATH}"

# Triton compiles a small Python/CUDA driver helper on first use. This server
# does not have python3.10-dev installed system-wide, so headers are extracted
# into this user-local directory.
export CPATH="/home/chendepeng/.local/python3.10-dev-root/usr/include/python3.10:/home/chendepeng/.local/python3.10-dev-root/usr/include:${CPATH:-}"
