#!/usr/bin/env bash

# Usage:
#   ./prem_dec.sh <dataset> <start_step> <end_step> [device]
#
# Examples:
#   ./prem_dec.sh /path/to/dataset 0 64 cuda
#   ./prem_dec.sh false_injection/interrupt_dataset 0 64 cuda
#
# Expects:
#   <dataset>/*/output_hidden.pt
#
# Writes:
#   <dataset>/<sample>_output_hidden_final_token_logprob_<start>_<end>.png

set -euo pipefail

die() {
  echo "[prem_dec] ERROR: $*" >&2
  return 1 2>/dev/null || exit 1
}

DATASET_ARG="${1:-}"
START_STEP="${2:-}"
END_STEP="${3:-}"
DEVICE="${4:-cuda}"
PYTHON_BIN="${PYTHON:-python3}"

[[ -n "$DATASET_ARG" ]] || die "dataset required"
[[ -n "$START_STEP" ]] || die "start_step required"
[[ -n "$END_STEP" ]] || die "end_step required"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python executable not found: $PYTHON_BIN"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$HOME/personaplex}"

if [[ -d "$DATASET_ARG" ]]; then
  DATASET="$DATASET_ARG"
elif [[ -d "$ROOT/Full-Duplex-Bench/data/$DATASET_ARG" ]]; then
  DATASET="$ROOT/Full-Duplex-Bench/data/$DATASET_ARG"
elif [[ -d "$ROOT/$DATASET_ARG" ]]; then
  DATASET="$ROOT/$DATASET_ARG"
else
  die "dataset directory not found: $DATASET_ARG"
fi

(
  cd "$SCRIPT_DIR/moshi"
  "$PYTHON_BIN" -m moshi.persona_vector.premature_decode \
    --root-dir "$DATASET" \
    --start "$START_STEP" \
    --end "$END_STEP" \
    --device "$DEVICE" \
    --output-dir "$DATASET" \
    --only-final-token-logprob
)

echo "[prem_dec] Done. Wrote final-token log-probability heatmaps under $DATASET"
