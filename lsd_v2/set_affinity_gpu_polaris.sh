#!/bin/bash -l
num_gpus=4
# Assign GPUs in reverse order due to Polaris node topology
gpu=$((${num_gpus} - 1 - ${PMI_LOCAL_RANK} % ${num_gpus}))
export CUDA_VISIBLE_DEVICES=$gpu
exec "$@"