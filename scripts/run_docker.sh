#!/usr/bin/env bash
set -euo pipefail

action="${1:-shell}"
if [[ "$#" -gt 0 ]]; then shift; fi
case "$action" in
  test) command_args=(bash scripts/test.sh "$@") ;;
  train) command_args=(bash scripts/retrain_and_evaluate.sh "$@") ;;
  evaluate) command_args=(bash scripts/evaluate.sh "$@") ;;
  shell) command_args=(bash "$@") ;;
  *) echo 'Usage: run_docker.sh {test|train|evaluate|shell} [arguments]' >&2; exit 2 ;;
esac

data_root="${EO_DATA_ROOT:-/data}"
docker_args=(run --rm
  --name "${EO_CONTAINER_NAME:-eo-dev}"
  --gpus all
  --shm-size=8g
  -p 127.0.0.1:6006:6006
  -v "$data_root/projects/eo-ml:/workspace"
  -v "$data_root/fireData:/datasets/fireData:ro"
  -v "$data_root/biomassData:/datasets/biomassData:ro"
  -v "$data_root/processed:/processed"
  -v "$data_root/model_cache:/cache"
  -v "$data_root/checkpoints:/checkpoints"
  -v "$data_root/outputs:/outputs"
  -e HF_HOME=/cache/huggingface
  -e TORCH_HOME=/cache/torch
  -e XDG_CACHE_HOME=/cache
  -e PYTHONUNBUFFERED=1
  -e "OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}"
  -e "MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}"
  -w /workspace)
if [[ "$action" == shell ]]; then
  docker_args+=(-i)
  if [[ -t 0 && -t 1 ]]; then docker_args+=(-t); fi
fi
exec docker "${docker_args[@]}" "${EO_ML_IMAGE:-eo-ml:latest}" "${command_args[@]}"
