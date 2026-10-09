#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ "$#" -lt 1 ]]; then
  echo 'Usage: bash scripts/evaluate.sh /checkpoints/fire/<run>/best.pt [--config PATH]' >&2
  exit 2
fi
checkpoint_path="$1"
shift
exec python -u -m src.evaluate_fire --checkpoint "$checkpoint_path" "$@"
