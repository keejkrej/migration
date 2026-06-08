#!/usr/bin/env bash
# Run migration segment + track for every ND2 position with timers.
# Skips positions whose trajectory CSV and overlay PNG already exist.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ND2=""
OUTPUT=""
CHANNEL="all"
Z=0
FROM=0
TO=""
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: run_all_positions.sh --nd2 PATH --output DIR [options] [-- extra migration track args]

Required:
  --nd2 PATH       Input ND2 file
  --output DIR     Output directory

Options:
  --channel SPEC   Channel selection (default: all)
  --z INDEX        Z-slice index (default: 0)
  --from INDEX     First position (default: 0)
  --to INDEX       Last position inclusive (default: last in file)
  -h, --help       Show this help

Examples:
  scripts/run_all_positions.sh \
    --nd2 /data/experiment.nd2 \
    --output ~/data/experiment \
    --channel all --z 0

  scripts/run_all_positions.sh \
    --nd2 /data/experiment.nd2 \
    --output ~/data/experiment \
    -- --delta-t 2 --min-track-length 50
EOF
}

format_duration() {
  local total="$1"
  local hours=$((total / 3600))
  local minutes=$(((total % 3600) / 60))
  local seconds=$((total % 60))
  if ((hours > 0)); then
    printf "%dh %02dm %02ds" "$hours" "$minutes" "$seconds"
  else
    printf "%dm %02ds" "$minutes" "$seconds"
  fi
}

position_is_complete() {
  local position="$1"
  (
    cd "$REPO_ROOT"
    uv run python - "$ND2" "$OUTPUT" "$CHANNEL" "$Z" "$position" <<'PY'
import sys
from pathlib import Path

from migration.core.nd2 import parse_channel_option
from migration.core.outputs import build_output_stem
from migration.core.types import Nd2Selection

nd2_path, output_dir, channel_arg, z_arg, position_arg = sys.argv[1:6]
selection = Nd2Selection(
    position=int(position_arg),
    channel=parse_channel_option(channel_arg),
    z=int(z_arg),
)
stem = build_output_stem(nd2_path, selection)
output = Path(output_dir).expanduser()
trajectories = output / f"{stem}_trajectories.csv"
overlay = output / f"{stem}_overlay.png"
sys.exit(0 if trajectories.is_file() and overlay.is_file() else 1)
PY
  )
}

run_timed() {
  local label="$1"
  shift
  local start end elapsed
  start=$(date +%s)
  echo "==> $label"
  if "$@"; then
    end=$(date +%s)
    elapsed=$((end - start))
    echo "==> $label finished in $(format_duration "$elapsed")"
    return 0
  fi
  end=$(date +%s)
  elapsed=$((end - start))
  echo "==> $label FAILED after $(format_duration "$elapsed")" >&2
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nd2)
      ND2="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --channel)
      CHANNEL="$2"
      shift 2
      ;;
    --z)
      Z="$2"
      shift 2
      ;;
    --from)
      FROM="$2"
      shift 2
      ;;
    --to)
      TO="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$ND2" || -z "$OUTPUT" ]]; then
  echo "Error: --nd2 and --output are required." >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$ND2" ]]; then
  echo "Error: ND2 file not found: $ND2" >&2
  exit 1
fi

mkdir -p "$OUTPUT"

N_POS="$(
  cd "$REPO_ROOT"
  uv run python - "$ND2" <<'PY'
import sys
from mdat.core.formats.input.session import inspect_input

print(inspect_input(sys.argv[1]).n_pos)
PY
)"

if [[ -z "$TO" ]]; then
  TO=$((N_POS - 1))
fi

if ((FROM < 0 || TO >= N_POS || FROM > TO)); then
  echo "Error: invalid position range ${FROM}..${TO} (file has ${N_POS} positions)." >&2
  exit 1
fi

PIPELINE_START=$(date +%s)
skipped=0
processed=0
failed=0

echo "ND2:       $ND2"
echo "Output:    $OUTPUT"
echo "Channel:   $CHANNEL"
echo "Z:         $Z"
echo "Positions: ${FROM}..${TO} (${N_POS} in file)"
echo

for ((position = FROM; position <= TO; position++)); do
  echo "----------------------------------------"
  echo "Position ${position}/${TO}"

  if position_is_complete "$position"; then
    echo "Skip position ${position}: trajectories and overlay already exist"
    ((skipped += 1)) || true
    continue
  fi

  position_start=$(date +%s)
  position_ok=true

  if ! run_timed "segment position ${position}" \
    bash -c "cd \"$REPO_ROOT\" && uv run migration segment \"$ND2\" --position \"$position\" --channel \"$CHANNEL\" --z \"$Z\" --output \"$OUTPUT\""; then
    position_ok=false
  elif ! run_timed "track position ${position}" \
    bash -c "cd \"$REPO_ROOT\" && uv run migration track \"$ND2\" --position \"$position\" --channel \"$CHANNEL\" --z \"$Z\" --output \"$OUTPUT\" $(printf '%q ' "${EXTRA_ARGS[@]}")"; then
    position_ok=false
  fi

  position_end=$(date +%s)
  position_elapsed=$((position_end - position_start))

  if [[ "$position_ok" == true ]]; then
    ((processed += 1)) || true
    echo "Position ${position} total: $(format_duration "$position_elapsed")"
  else
    ((failed += 1)) || true
    echo "Position ${position} failed (elapsed $(format_duration "$position_elapsed"))" >&2
  fi
  echo
done

PIPELINE_END=$(date +%s)
PIPELINE_ELAPSED=$((PIPELINE_END - PIPELINE_START))

echo "========================================"
echo "Done in $(format_duration "$PIPELINE_ELAPSED")"
echo "Processed: ${processed}"
echo "Skipped:   ${skipped}"
echo "Failed:    ${failed}"

if ((failed > 0)); then
  exit 1
fi
