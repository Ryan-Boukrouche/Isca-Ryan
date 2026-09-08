#!/usr/bin/env bash
# Use bash for portability and modern shell features.
set -euo pipefail  # Exit on error, undefined variable, and fail on pipe errors.
IFS=$'\n\t'  # Set a safe internal field separator for word splitting.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # Resolve this script's directory.
LOOPER="${SCRIPT_DIR}/looper.sh"  # Locate the looper driver in the same directory.

if [[ ! -f "$LOOPER" ]]; then
  echo "Cannot find looper script at $LOOPER" >&2
  exit 1
fi

# This driver runs the north-east dayside gridpoint region in chunks of 32 points.
# Each chunk is invoked using looper.sh with TASK_OFFSET/TASK_COUNT set.
# The north-east dayside region is:
#   J = 32..63, I = 0..31 
# and we sweep through K ranges in this order:
#   36..47, 24..35, 12..23, 0..11.
# Alternatively, set J_INDICES and I_INDICES to select arbitrary comma-separated indices.

N_PARALLEL=${N_PARALLEL:-32}  # Default number of parallel Isca tasks per batch.
TASK_COUNT=${TASK_COUNT:-32}  # Default number of gridpoints per chunk.
VERBOSE=${VERBOSE:-no}  # Default verbosity setting for downstream scripts.
CLEANUP_TASK_DIRS=${CLEANUP_TASK_DIRS:-yes}  # Default cleanup behavior for task directories.

J_INDICES=${J_INDICES:-$(seq -s, 32 63)}  # J indices to process, by default 32..63
I_INDICES=${I_INDICES:-$(seq -s, 0 31)}  # I indices to process, by default 0..31

K_RANGES=("36-47" "24-35" "12-23" "0-11")  # K bands to execute in priority order.

for krange in "${K_RANGES[@]}"; do
  K_MIN_RANGE=${krange%-*}  # Extract the lower bound of the current K range.
  K_MAX_RANGE=${krange#*-}  # Extract the upper bound of the current K range.
  j_count=$(awk -F, '{print NF}' <<< "$J_INDICES")  # Count selected J indices separated by commas.
  i_count=$(awk -F, '{print NF}' <<< "$I_INDICES")  # Count selected I indices separated by commas.
  total_points=$(( (K_MAX_RANGE - K_MIN_RANGE + 1) * j_count * i_count ))  # Compute total gridpoints in the current K slab.
  num_chunks=$(( (total_points + TASK_COUNT - 1) / TASK_COUNT ))  # Divide into full 32-point chunks, rounding up.

  echo "=== K=${K_MIN_RANGE}-${K_MAX_RANGE} : total points=${total_points}, chunks=${num_chunks} ==="  # Report batch size.

  for ((chunk=0; chunk<num_chunks; chunk++)); do
    offset=$((chunk * TASK_COUNT))  # Calculate task offset for this chunk.
    chunk_id=$(printf '%03d' "$((chunk + 1))")  # Format the chunk index with leading zeros.
    run_id="ne_j032-063_i000-031_k${K_MIN_RANGE}-${K_MAX_RANGE}_chunk${chunk_id}"  # Build a descriptive run ID.

    echo "Running chunk ${chunk_id}/${num_chunks} for K=${K_MIN_RANGE}-${K_MAX_RANGE}, TASK_OFFSET=${offset}, TASK_COUNT=${TASK_COUNT}, RUN_ID=${run_id}"  # Log the exact invocation.

    K_MIN=${K_MIN_RANGE} K_MAX=${K_MAX_RANGE} \
    J_INDICES=${J_INDICES} I_INDICES=${I_INDICES} \
    TASK_OFFSET=${offset} TASK_COUNT=${TASK_COUNT} \
    N_PARALLEL=${N_PARALLEL} CLEANUP_TASK_DIRS=${CLEANUP_TASK_DIRS} VERBOSE=${VERBOSE} \
    RUN_ID="${run_id}" \
    bash "${LOOPER}"  # Call looper.sh with the chunk-specific environment.
  done

done
