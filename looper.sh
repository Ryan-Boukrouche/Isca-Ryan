#!/usr/bin/env bash
#
# Radiative-timescale loop for Isca
#
# What this script does:
#   1. Take one "reference" restart archive from a finished run.
#   2. Copy it into a temporary working directory.
#   3. Add +1 K to one temperature gridpoint in atmosphere.res.nc.
#   4. Run Isca for one timestep from that temporary restart.
#   5. Postprocess the output with your existing pressure-level interpolation script.
#   6. Read soc_tdt_rad at the chosen gridpoint.
#   7. Store the result.
#   8. Repeat for the next gridpoint.
#
# This assumes:
#   - your experiment script already knows how to run one timestep,
#   - your interpolation script already produces atmos_monthly_interp_full.nc,
#   - you want tau = -1 / soc_tdt_rad when the perturbation is +1 K.
#
# Because the model run is the expensive part, the script only uses Python
# for tiny file-editing tasks.

set -euo pipefail # stops the script if: commands fail (-e), if variables are unset (-u), and if any part of a pipe fails (pipefail)

# Temporarily relax unset-variable strictness
set +u
# Initialize micromamba in this shell
eval "$(micromamba shell hook --shell bash)"
micromamba activate isca_env
# Reenable unset-variable strictness
set -u

# ----------------------------
# User-editable settings
# ----------------------------

# Experiment name, used in your Isca directory tree.
EXP="2_1320_as007"

# The completed run that provides the baseline restart.
BASE_RUN="0273"

# The next run number, used as the first perturbed run index.
NEXT_RUN="0274"

# Root folder containing run0273/, run0274/, etc.
ROOT="/proj/bolinc/users/x_ryabo/Isca-Ryan_outputs/${EXP}"

# Where the final table and NetCDF output will be written.
OUTROOT="${ROOT}/radiative_timescale_output"

# This is the archive that never changes.
ORIGINAL_RESTART_ARCHIVE="${ROOT}/restarts/res${BASE_RUN}_original.tar.gz"

# This is the archive that Isca should read for the current iteration.
WORKING_RESTART_ARCHIVE="${ROOT}/restarts/res${BASE_RUN}.tar.gz"

# ----------------------------

# Isca experiment script.
# Replace this with the actual script you normally use to launch the model.
ISCA_EXPERIMENT_SCRIPT="/home/x_ryabo/Isca-Ryan/exp/${EXP}/socrates_aquaplanet_nodyn.py"

# Your pressure-level interpolation script.
PLEVEL_SCRIPT="/home/x_ryabo/Isca-Ryan/postprocessing/plevel_interpolation/scripts/run_plevel.py"

# Helper wrappers for parallel execution.
# The wrapper scripts are separate Python files that run one task at a time.
# This avoids race conditions inside the original experiment and interpolation scripts.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISCA_RUN_WRAPPER="${SCRIPT_DIR}/run_isca_wrapper.py"
PLEVEL_INTERP_WRAPPER="${SCRIPT_DIR}/run_plevel_wrapper.py"

# Parallel execution settings.
# Number of independent gridpoint tasks to run simultaneously.
# This is the maximum number of background jobs that will be active.
N_PARALLEL=${N_PARALLEL:-16}
# Number of CPU cores assigned to each Isca task.
# For example, with 16 total cores, use 1 core per task for 16 parallel tasks.
NCORES_PER_TASK=${NCORES_PER_TASK:-1}
# The first run number assigned to the generated parallel task outputs.
# Each task will get a unique run number starting from this base.
BASE_TASK_RUN_NUM=$((10#$NEXT_RUN))
# Whether to remove task-specific temporary directories after each job finishes.
CLEANUP_TASK_DIRS=${CLEANUP_TASK_DIRS:-yes}

# Number of model levels, latitudes, longitudes in the restart/grid.
# These should match your run.
NZ=48
NY=64
NX=128

# Temperature perturbation size in Kelvin.
DT=1.0

# ----------------------------
# Sanity checks and folders
# ----------------------------

# Stop immediately if the baseline restart directory is missing.
if [[ ! -d "$BASE_RESTART_DIR" ]]; then
  echo "Missing baseline restart directory: $BASE_RESTART_DIR" >&2
  exit 1
fi
rm -rf "$ORIGINAL_RESTART_DIR"
cp -a "$BASE_RESTART_DIR" "$ORIGINAL_RESTART_DIR"

# Stop immediately if the experiment script is missing.
if [[ ! -f "$ISCA_EXPERIMENT_SCRIPT" ]]; then
  echo "Missing experiment script: $ISCA_EXPERIMENT_SCRIPT" >&2
  exit 1
fi

# Stop immediately if the interpolation script is missing.
if [[ ! -f "$PLEVEL_SCRIPT" ]]; then
  echo "Missing interpolation script: $PLEVEL_SCRIPT" >&2
  exit 1
fi

# Create output folders if they do not already exist.
mkdir -p "$OUTROOT"

# Working directories for parallel tasks and results.
WORKDIR="${OUTROOT}/parallel_work"
JOB_RESULTS_DIR="${OUTROOT}/job_results"
mkdir -p "$WORKDIR" "$JOB_RESULTS_DIR"

# A simple text file to store one result per line:
#   k j i tau
# This master TSV is overwritten at start and rebuilt from task outputs.
RESULTS_TSV="${OUTROOT}/tau_rad_results.tsv"
: > "$RESULTS_TSV"

# ----------------------------
# Helper 1:
# restore the working restart archive from the pristine original archive
# ----------------------------
refresh_working_restart_archive() {
  # Copy the pristine archive over the working archive.
  # This gives us a clean baseline for the next gridpoint.
  cp -f "$ORIGINAL_RESTART_ARCHIVE" "$WORKING_RESTART_ARCHIVE"
}

# ----------------------------
# Helper 2:
# copy baseline restart and perturb one temperature gridpoint
# ----------------------------
perturb_restart_temperature() {
  # Arguments:
  #   $1 = vertical index k
  #   $2 = latitude index j
  #   $3 = longitude index i
  local k="$1"
  local j="$2"
  local i="$3"

  # Run Python because the restart archive helpers live in Python.
  python - \
    "$WORKING_RESTART_ARCHIVE" \
    "$RESTART_EDIT_TMPDIR" \
    "$k" \
    "$j" \
    "$i" \
    "$DT" <<'PY'
import os
import sys
from isca.util import edit_restart_archive
from isca.util import edit_restart_file

# The restart archive that Isca will read for this gridpoint.
archive = sys.argv[1]

# The temporary directory used while the archive is unpacked and edited.
tmp_dir = sys.argv[2]

# Gridpoint indices to perturb.
k = int(sys.argv[3])
j = int(sys.argv[4])
i = int(sys.argv[5])

# The temperature increment in Kelvin.
dt = float(sys.argv[6])

# Build a temporary output archive name first.
# We write to a separate file, then rename it over the working archive.
out_archive = archive + ".new"

# -------------------------
# 1. Perturb atmosphere.res.nc (gridpoint temperature)
# -------------------------

# Open the archive, unpack it into tmp_dir, edit the extracted files,
# and repack the result into out_archive when the context closes.
with edit_restart_archive(archive, outfile=out_archive, tmp_dir=tmp_dir) as files:

    # Open atmosphere.res.nc and perturb the temperature field.
    with edit_restart_file(files["atmosphere.res.nc"]) as ds:

        # Read the old value for a sanity check.
        before = ds["tg"][0, k, j, i].item()

        # Add the perturbation.
        ds["tg"][0, k, j, i] += dt

        # Read the new value for a sanity check.
        after = ds["tg"][0, k, j, i].item()

        # Print the change so you can verify that the edit really happened.
        print(f"[ATMOS] tg before = {before}")
        print(f"[ATMOS] tg after  = {after}")
        print(f"[ATMOS] delta     = {after - before}")

# -------------------------
# 2. Perturb spectral_dynamics.res.nc (grid temperature field)
# -------------------------
# Open spectral_dynamics.res.nc and perturb the matching grid-space field.
    with edit_restart_file(files["spectral_dynamics.res.nc"]) as ds:

        # Read the old value for a sanity check.
        before = ds["tg"][0, k, j, i].item()

        # Add the same perturbation here too.
        ds["tg"][0, k, j, i] += dt

        # Read the new value for a sanity check.
        after = ds["tg"][0, k, j, i].item()

        # Print the change so you can verify that the edit really happened.
        print(f"[SPECTRAL] tg before = {before}")
        print(f"[SPECTRAL] tg after  = {after}")
        print(f"[SPECTRAL] delta     = {after - before}")

# Replace the old working archive with the newly repacked one.
# This keeps only one active restart archive at a time.
mv -f "$out_archive" "$archive"
PY
}

# ----------------------------
# Helper 3:
# run one Isca timestep from the temporary restart
# ----------------------------
run_isca_one_step() {
  python "$ISCA_EXPERIMENT_SCRIPT"
}

# ----------------------------
# Helper 4:
# run the pressure-level interpolation step
# ----------------------------
run_interpolation() {
  # Arguments:
  #   $1 = run number for this task
  local run_num="$1"

  # Use the wrapper so the interpolation step only processes this task's output.
  python "$PLEVEL_INTERP_WRAPPER" "$(dirname "$PLEVEL_SCRIPT")" "$EXP" "$run_num" "$ROOT" "$OUTROOT"
}

# ----------------------------
# Helper 5:
# read soc_tdt_rad at one point and convert to a timescale
# ----------------------------
compute_tau_from_output() {
  # Arguments:
  #   $1 = interpolated NetCDF file
  #   $2 = vertical index k
  #   $3 = latitude index j
  #   $4 = longitude index i
  local interp_file="$1"
  local k="$2"
  local j="$3"
  local i="$4"

  python - "$interp_file" "$k" "$j" "$i" <<'PY'
import sys
import numpy as np
import xarray as xr

interp_file = sys.argv[1]
k = int(sys.argv[2])
j = int(sys.argv[3])
i = int(sys.argv[4])

# Open the interpolated output file.
with xr.open_dataset(interp_file, decode_times=False) as ds:
    # Socrates radiative temperature tendency, in K/s.
    tdot = ds["soc_tdt_rad"].isel(time=0, pfull=k, lat=j, lon=i).item()

# For a +1 K perturbation:
# tau = -ΔT / Δ(dT/dt) = -1 / tdot
# If tdot is zero or missing, return NaN.
if np.isfinite(tdot) and abs(tdot) > 1e-30:
  tau = 1.0 / np.abs(tdot)
else:
  tau = np.nan

print(tau)
PY
}

wait_for_slot() {
  # Wait until the number of running background jobs is below the configured limit.
  while true; do
    local running
    running=$(jobs -pr | wc -l)
    if (( running < N_PARALLEL )); then
      break
    fi
    # Wait for the next background job to finish before checking again.
    if ! wait -n; then
      exit_code=1
    fi
  done
}

process_gridpoint() {
  # Arguments:
  #   $1 = k index
  #   $2 = j index
  #   $3 = i index
  #   $4 = string used to name this task uniquely
  #   $5 = numeric run number for this task
  #   $6 = zero-padded run number string for folder names
  local k="$1"
  local j="$2"
  local i="$3"
  local task_id="$4"
  local run_num="$5"
  local run_num_str="$6"

  # Each task gets its own temporary directory to avoid file conflicts.
  local task_dir="${WORKDIR}/task_${task_id}"
  local tmp_restart_dir="${task_dir}/restart"
  # This is one reusable temporary directory used while editing the archive.
  # It is not kept after the helper finishes if the helper succeeds.
  RESTART_EDIT_TMPDIR="${OUTROOT}/restart_edit_tmp"
  local tmp_dir="${task_dir}/restart_edit_tmp"

  # This task writes one TSV line to a unique result file.
  local result_file="${JOB_RESULTS_DIR}/tau_${run_num_str}_${task_id}.tsv"
  local interp_file="${OUTROOT}/${EXP}/run${run_num_str}/atmos_monthly_interp_full.nc"

  mkdir -p "$task_dir"
  # Copy the baseline restart into the task directory.
  cp -a "$ORIGINAL_RESTART_DIR" "$tmp_restart_dir"

  perturb_restart_temperature "$tmp_restart_dir" "$k" "$j" "$i"
  run_isca_one_step "$run_num" "$tmp_restart_dir"
  run_interpolation "$run_num"

  local tau
  # Read the interpolated output and compute the timescale for this gridpoint.
  tau="$(compute_tau_from_output "$interp_file" "$k" "$j" "$i")"
  echo "$k $j $i $tau" > "$result_file"

  # Remove temporary directories for this task if cleanup is enabled.
  if [[ "$CLEANUP_TASK_DIRS" == "yes" ]]; then
    rm -rf "$task_dir"
    rm -rf "${OUTROOT}/${EXP}/run${run_num_str}"
  fi
}

# ----------------------------
# Main loop
# ----------------------------
exit_code=0
for k in $(seq 0 $((NZ - 1))); do
  for j in $(seq 0 $((NY - 1))); do
    for i in $(seq 0 $((NX - 1))); do
      # Create a unique integer index for this gridpoint task.
      task_index=$((k * NY * NX + j * NX + i))
      # Turn that index into a unique run number for this task.
      run_num=$((BASE_TASK_RUN_NUM + task_index))
      # Format the run number as a four-digit string for folder names.
      run_num_str=$(printf '%04d' "$run_num")
      # Build a human-readable task ID from k,j,i.
      task_id=$(printf '%02d_%03d_%03d' "$k" "$j" "$i")

      # Wait until we can start a new parallel task.
      wait_for_slot
      # Start the gridpoint task in the background.
      process_gridpoint "$k" "$j" "$i" "$task_id" "$run_num" "$run_num_str" &
    done
  done
done

# Wait for all remaining tasks to complete.
while (( $(jobs -pr | wc -l) > 0 )); do
  if ! wait -n; then
    exit_code=1
  fi
done

if (( exit_code != 0 )); then
  echo "One or more parallel tasks failed. Check logs and temporary directories." >&2
  exit 1
fi

# Combine all task results into a single TSV file in sorted order.
sort -n -k1,1 -k2,2 -k3,3 "${JOB_RESULTS_DIR}"/tau_*.tsv > "$RESULTS_TSV"

# ----------------------------
# Build the final 3D NetCDF file
# ----------------------------
python - "$RESULTS_TSV" "$OUTROOT/tau_rad.nc" "$NZ" "$NY" "$NX" <<'PY'
import sys
import numpy as np
import xarray as xr

results_tsv = sys.argv[1]
out_nc = sys.argv[2]
nz = int(sys.argv[3])
ny = int(sys.argv[4])
nx = int(sys.argv[5])

# Start with an array full of NaNs.
tau = np.full((nz, ny, nx), np.nan, dtype=np.float64)

# Read each line of the TSV file and fill the corresponding point.
with open(results_tsv) as f:
    for line in f:
        if not line.strip():
            continue
        k, j, i, val = line.split()
        tau[int(k), int(j), int(i)] = float(val)

# Make a simple NetCDF file using index coordinates.
ds = xr.Dataset(
    data_vars={
        "tau_rad": (("pfull_index", "lat_index", "lon_index"), tau)
    },
    coords={
        "pfull_index": np.arange(nz),
        "lat_index": np.arange(ny),
        "lon_index": np.arange(nx),
    },
    attrs={
        "description": "Radiative timescale",
        "definition": "tau = 1 / |soc_tdt_rad|",
        "units": "s",
    },
)
ds.to_netcdf(out_nc)
PY

echo "Finished."
echo "Text results: ${RESULTS_TSV}"
echo "NetCDF output: ${OUTROOT}/tau_rad.nc"