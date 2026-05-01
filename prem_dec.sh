#!/usr/bin/env bash

# Usage:
#   source prem_dec.sh <dataset_rel> <start_step> <end_step>
#
# Examples:
#   source prem_dec.sh false_injection/interrupt_dataset 0 64
#   source prem_dec.sh false_injection2 0 64
#
# Expects:
#   <dataset>/*/output_hidden.pt
#
# Writes:
#   <dataset>/<sample>_output_hidden_final_token_logprob_<start>_<end>.png

die() {
  echo "[prem_dec] ERROR: $*" >&2
  return 1 2>/dev/null || exit 1
}

DATASET_REL="${1:?dataset_rel required (e.g. false_injection/interrupt_dataset)}"
START_STEP="${2:?start_step required}"
END_STEP="${3:?end_step required}"
DEVICE="cuda"
PYTHON_BIN="${PYTHON:-python3}"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python executable not found: $PYTHON_BIN"

ROOT="${ROOT:-$HOME/personaplex}"
DATA_ROOT="${DATA_ROOT:-/home/chang168/orcd/pool/personaplex/data}"

if [[ "$DATASET_REL" = /* ]]; then
  DATASET="$DATASET_REL"
else
  DATASET="$DATA_ROOT/$DATASET_REL"
fi

if [[ -d "$DATASET/interrupt_dataset" ]]; then
  DATASET="$DATASET/interrupt_dataset"
fi

if [[ ! -d "$DATASET" ]]; then
  die "dataset directory not found: $DATASET"
fi

hidden_count=$(find "$DATASET" -mindepth 2 -maxdepth 2 -type f -name "output_hidden.pt" | wc -l | tr -d ' ')
input_transcript_count=$(find "$DATASET" -mindepth 2 -maxdepth 2 -type f -name "input_transcript.json" | wc -l | tr -d ' ')
output_transcript_count=$(find "$DATASET" -mindepth 2 -maxdepth 2 -type f -name "output_transcript.json" | wc -l | tr -d ' ')

if [[ "$hidden_count" -gt 0 && ( "$input_transcript_count" -lt "$hidden_count" || "$output_transcript_count" -lt "$hidden_count" ) ]]; then
  if [[ "$DATASET" == "$DATA_ROOT/"* ]]; then
    TRANSCRIPT_DATASET_REL="${DATASET#"$DATA_ROOT"/}"
    cd "$ROOT" || die "failed to cd to $ROOT"
    source venv_fdb.sh || die "failed to activate venv_fdb.sh"
    if [[ "$input_transcript_count" -lt "$hidden_count" ]]; then
      echo "[prem_dec] Missing input_transcript.json files; generating input transcripts for $TRANSCRIPT_DATASET_REL"
      source fdb_transcript.sh "$TRANSCRIPT_DATASET_REL" input transcript || die "failed to generate input transcripts"
    fi
    if [[ "$output_transcript_count" -lt "$hidden_count" ]]; then
      echo "[prem_dec] Missing output_transcript.json files; generating output transcripts for $TRANSCRIPT_DATASET_REL"
      source fdb_transcript.sh "$TRANSCRIPT_DATASET_REL" output transcript || die "failed to generate output transcripts"
    fi
    cd "$ROOT" || die "failed to cd to $ROOT"
    source venv_moshi.sh || die "failed to activate venv_moshi.sh"
  else
    echo "[prem_dec] WARNING: transcript files are missing, but $DATASET is outside DATA_ROOT=$DATA_ROOT; skipping auto transcript generation." >&2
  fi
fi

if ! (
  cd "$ROOT/personaplex/moshi"
  "$PYTHON_BIN" -m moshi.persona_vector.premature_decode \
    --root-dir "$DATASET" \
    --start "$START_STEP" \
    --end "$END_STEP" \
    --device "$DEVICE" \
    --output-dir "$DATASET" \
    --only-final-token-logprob
); then
  cd ~/personaplex
  return 1 2>/dev/null || exit 1
fi

echo "[prem_dec] Done. Wrote final-token log-probability heatmaps under $DATASET"
cd ~/personaplex
