#!/usr/bin/env bash
#
# Radiative-timescale computer for Isca
#
# What this script does:
#   1. Take the latest restart archive from a finished monthly averaged run.
#   2. Copy it into Isca_outputs/experiment_name/restarts/resXXXX_k[kkk]_j[jjj]_i[iii].tar.gz.
#   3. Use the Isca utilities edit_restart_archive and edit_restart_file to add +1 K at gridpoint (k,j,i) in atmosphere.res.nc and spectral_dynamics.res.nc
#   4. Run Isca for one timestep from the temporary restart file resXXXX_k[kkk]_j[jjj]_i[iii].tar.gz using a Python wrapper
#   5. Postprocess the output with the interpolation script run_level.py using a Python wrapper.
#   6. Read soc_tdt_rad at gridpoint (k,j,i).
#   7. Store the result in a .tsv file.
#   8. Run 16, 32, or 64 gridpoints in parallel with 1 CPU per gridpoint.
#   9. Assemble the 3D output file tau_rad.tsv and build the final netCDF file tau_rad.nc with variables pfull_index, lat_index, lon_index, tau_rad

set -euo pipefail # stops the script if: commands fail (-e), if variables are unset (-u), and if any part of a pipe fails (pipefail)
IFS=$'\n\t'

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

# Experiment name.
EXP="2_1320_as007"

# The completed run that provides the baseline restart.
BASE_RUN="0273"

# Root folder containing run0273/, run0274/, etc.
ROOT="/proj/bolinc/users/x_ryabo/Isca-Ryan_outputs/${EXP}"

# Where the final table and NetCDF output will be written.
OUTROOT="${ROOT}/radiative_timescale_output"

# This is the archive that never changes.
ORIGINAL_RESTART_ARCHIVE="${ROOT}/restarts/res${BASE_RUN}_original.tar.gz"

# ----------------------------

# Isca experiment script.
ISCA_EXPERIMENT_SCRIPT="/home/x_ryabo/Isca-Ryan/exp/${EXP}/socrates_aquaplanet_nodyn.py"

# Sigma pressure to real pressure interpolation script.
PLEVEL_SCRIPT="/home/x_ryabo/Isca-Ryan/postprocessing/plevel_interpolation/scripts/run_plevel.py"

# Helper wrappers for parallel execution.
# The wrapper scripts are separate Python files that run one task at a time.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISCA_RUN_WRAPPER="${SCRIPT_DIR}/run_isca_wrapper.py"
PLEVEL_INTERP_WRAPPER="${SCRIPT_DIR}/run_plevel_wrapper.py"

# Parallel execution settings.
# Number of independent gridpoint tasks to run simultaneously.
# This is the maximum number of background jobs that will be active.
N_PARALLEL=${N_PARALLEL:-1}
# Number of CPU cores assigned to each Isca task.
# For example, with 16 total cores, use 1 core per task for 16 parallel tasks.
NCORES_PER_TASK=${NCORES_PER_TASK:-1}

# A run identifier to isolate temporary outputs across separate looper invocations.
# If not explicitly set, this will be derived from the selected gridpoint range.
RUN_ID=${RUN_ID:-}

# Whether to remove task-specific temporary directories after each job finishes.
CLEANUP_TASK_DIRS=${CLEANUP_TASK_DIRS:-yes}

# Whether to print verbose task progress. Set VERBOSE=no to silence non-error output.
VERBOSE=${VERBOSE:-yes}

log() {
  if [[ "$VERBOSE" == "yes" || "$VERBOSE" == "true" || "$VERBOSE" == "1" ]]; then
    echo "$@"
  fi
}

# Number of model levels, latitudes, longitudes in the restart/grid.
NZ=48
NY=64
NX=128

# Temperature perturbation size in Kelvin.
DT=1.0

# ----------------------------
# Sanity checks and folders
# ----------------------------

# Stop immediately if the original restart archive is missing.
if [[ ! -f "$ORIGINAL_RESTART_ARCHIVE" ]]; then
  echo "Missing original restart archive: $ORIGINAL_RESTART_ARCHIVE" >&2
  exit 1
fi

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

# ----------------------------
# Helper 1:
# copy baseline restart and perturb one temperature gridpoint
# ----------------------------
perturb_restart_temperature() {
  # Arguments:
  #   $1 = restart archive path
  #   $2 = temporary edit directory
  #   $3 = vertical index k
  #   $4 = latitude index j
  #   $5 = longitude index i
  local archive="$1"
  local tmp_dir="$2"
  local k="$3"
  local j="$4"
  local i="$5"

  # Run Python because the restart archive helpers live in Python.
  python - \
    "$archive" \
    "$tmp_dir" \
    "$k" \
    "$j" \
    "$i" \
    "$DT" <<'PY'
import os
import sys
from isca.util import edit_restart_archive
from isca.util import edit_restart_file

verbose = os.environ.get('VERBOSE', 'yes').lower() in ('yes', 'true', '1')

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

        # Print the change to verify that the edit really happened.
        if verbose:
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

        # Print the change to verify that the edit really happened.
        if verbose:
            print(f"[SPECTRAL] tg before = {before}")
            print(f"[SPECTRAL] tg after  = {after}")
            print(f"[SPECTRAL] delta     = {after - before}")

# Replace the old working archive with the newly repacked one.
# This keeps only one active restart archive at a time.
os.replace(out_archive, archive)
PY
}

# ----------------------------
# Helper 2:
# run one Isca timestep from the temporary restart
# ----------------------------
run_isca_one_step() {
  local run_num="$1"
  local restart_archive="$2"
  local isca_output_root="$3"
  local task_id="$4"

  python "$ISCA_RUN_WRAPPER" \
      "$ISCA_EXPERIMENT_SCRIPT" \
      "$run_num" \
      "$restart_archive" \
      "$NCORES_PER_TASK" \
      "$isca_output_root" \
      "$task_id"
}

# ----------------------------
# Helper 3:
# run the pressure-level interpolation step
# ----------------------------
run_interpolation() {
  # Arguments:
  #   $1 = run number for this task
  #   $2 = Isca output root for this task
  local run_num="$1"
  local isca_output_root="$2"

  # Use the wrapper so the interpolation step only processes this task's output.
  python "$PLEVEL_INTERP_WRAPPER" "$(dirname "$PLEVEL_SCRIPT")" "$EXP" "$run_num" "$isca_output_root" "$isca_output_root"
}

# ----------------------------
# Helper 4:
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

  log "START $task_id run=$run_num"

  # Each task gets its own temporary directory to avoid file conflicts.
  local task_dir="${WORKDIR}/task_${task_id}"
  local tmp_restart_dir="${task_dir}/restart"
  local k_fmt
  local j_fmt
  local i_fmt
  k_fmt=$(printf '%03d' "$k")
  j_fmt=$(printf '%03d' "$j")
  i_fmt=$(printf '%03d' "$i")
  local task_restart_archive="${task_dir}/res${BASE_RUN}_k${k_fmt}_j${j_fmt}_i${i_fmt}.tar.gz"

  mkdir -p "$tmp_restart_dir"

  # Copy the pristine archive into the per-task workspace.
  cp -f "$ORIGINAL_RESTART_ARCHIVE" "$task_restart_archive"

  # Each gridpoint gets its own output folder under the run root.
  local point_root="${OUTROOT_RUN}/${task_id}"
  local result_file="${JOB_RESULTS_DIR}/tau_${run_num_str}_${task_id}.tsv"
  local interp_file="${point_root}/atmos_monthly_interp_full.nc"

  perturb_restart_temperature "$task_restart_archive" "$tmp_restart_dir" "$k" "$j" "$i"
  # Tell Isca to write its output under this gridpoint's folder so
  # the final structure becomes: OUTROOT/kXXX_jYYY_iZZZ/runXXXX/...
  run_isca_one_step "$run_num" "$task_restart_archive" "$point_root" "$task_id"
  run_interpolation "$run_num" "$point_root"

  local tau
  # Read the interpolated output and compute the timescale for this gridpoint.
  tau="$(compute_tau_from_output "$interp_file" "$k" "$j" "$i")"
  echo "$k $j $i $tau" > "$result_file"
  log "DONE  $task_id run=$run_num"

  # Remove temporary directories for this task if cleanup is enabled.
  if [[ "$CLEANUP_TASK_DIRS" == "yes" ]]; then
    rm -rf "$task_dir"
    rm -rf "$point_root"
  fi
}

# ----------------------------
# Main loop
# ----------------------------
exit_code=0

# To run one gridpoint: 
# K_MIN=10 K_MAX=10 J_MIN=20 J_MAX=20 I_MIN=30 I_MAX=30 N_PARALLEL=1 CLEANUP_TASK_DIRS=no VERBOSE=yes bash looper.sh
# 
# To run two gridpoints in parallel:
# K_MIN=10 K_MAX=10 J_MIN=20 J_MAX=20 I_MIN=30 I_MAX=31 N_PARALLEL=2 CLEANUP_TASK_DIRS=yes VERBOSE=no bash looper.sh

# To run 4 gridpoints in parallel, surface level (47), substellar longitude (0), and a span of latitudes around the substellar point (30-34):
# K_MIN=47 K_MAX=47 J_MIN=30 J_MAX=34 I_MIN=0 I_MAX=0 N_PARALLEL=4 CLEANUP_TASK_DIRS=yes VERBOSE=no bash looper.sh

# One may want to divide up the grid into managable sizes:
# North-east dayside: J=32–63, I=0–31.
# North-west dayside: J=32–63, I=96–127
# South-east dayside: J=0–31, I=0–31
# South-west dayside: J=0–31, I=96–127
# North-east nightside: J=32–63, I=32–63
# North-west nightside: J=32–63, I=64–95
# South-east nightside: J=0–31, I=32–63
# South-west nightside: J=0–31, I=64–95
# Vertical levels can also be divided into chunks, e.g. K=0–11, 12–23, 24–35, 36–47.


K_MIN=${K_MIN:-0}
K_MAX=${K_MAX:-$((NZ - 1))}
J_MIN=${J_MIN:-0}
J_MAX=${J_MAX:-$((NY - 1))}
I_MIN=${I_MIN:-0}
I_MAX=${I_MAX:-$((NX - 1))}

# Allow chunked execution within the selected range.
# TASK_OFFSET selects the first point from the range to process.
# TASK_COUNT selects how many gridpoints to run in this invocation.
TASK_OFFSET=${TASK_OFFSET:-0}
TASK_COUNT=${TASK_COUNT:-0}

if (( K_MIN < 0 || K_MAX >= NZ || J_MIN < 0 || J_MAX >= NY || I_MIN < 0 || I_MAX >= NX || K_MIN > K_MAX || J_MIN > J_MAX || I_MIN > I_MAX )); then
  echo "Invalid gridpoint range: k=${K_MIN}:${K_MAX}, j=${J_MIN}:${J_MAX}, i=${I_MIN}:${I_MAX}" >&2
  exit 1
fi

TOTAL_POINTS=$(( (K_MAX - K_MIN + 1) * (J_MAX - J_MIN + 1) * (I_MAX - I_MIN + 1) ))
if (( TASK_OFFSET < 0 || TASK_OFFSET >= TOTAL_POINTS )); then
  echo "Invalid TASK_OFFSET=${TASK_OFFSET}; valid 0..$((TOTAL_POINTS - 1))" >&2
  exit 1
fi
if (( TASK_COUNT <= 0 || TASK_COUNT > TOTAL_POINTS - TASK_OFFSET )); then
  TASK_COUNT=$((TOTAL_POINTS - TASK_OFFSET))
fi

# If RUN_ID is explicitly provided, isolate this invocation under a dedicated
# output folder so multiple looper runs can coexist in the same OUTROOT.
# Otherwise, use OUTROOT directly so each gridpoint creates its own folder
# directly under OUTROOT.
if [[ -n "${RUN_ID:-}" ]]; then
  OUTROOT_RUN="${OUTROOT}/${RUN_ID}"
else
  OUTROOT_RUN="${OUTROOT}"
fi

# Working directories for parallel tasks and results.
WORKDIR="${OUTROOT_RUN}/parallel_work"
JOB_RESULTS_DIR="${OUTROOT_RUN}/job_results"
mkdir -p "$OUTROOT_RUN" "$WORKDIR" "$JOB_RESULTS_DIR"

# A simple text file to store one result per line:
#   k j i tau
# This master TSV is overwritten at start and rebuilt from task outputs.
RESULTS_TSV="${OUTROOT_RUN}/tau_rad_results${RUN_ID:+_${RUN_ID}}.tsv"
OUT_NC="${OUTROOT_RUN}/tau_rad${RUN_ID:+_${RUN_ID}}.nc"
: > "$RESULTS_TSV"

# Report the active run settings.
log "looper.sh starting run ${RUN_ID}"
log "  selected gridpoint range: k=${K_MIN:-0}:${K_MAX:-$((NZ - 1))}, j=${J_MIN:-0}:${J_MAX:-$((NY - 1))}, i=${I_MIN:-0}:${I_MAX:-$((NX - 1))}"
log "  parallel tasks: ${N_PARALLEL}, cores per task: ${NCORES_PER_TASK}, cleanup: ${CLEANUP_TASK_DIRS}"

num_tasks=$TASK_COUNT
if (( TASK_COUNT < TOTAL_POINTS )); then
  echo "  candidate gridpoints in range: ${TOTAL_POINTS}"
  echo "  running chunk: TASK_OFFSET=${TASK_OFFSET}, TASK_COUNT=${TASK_COUNT}"
fi
log "  total tasks to run: ${num_tasks}"

task_counter=0
tasks_run=0
done_selection=0
for k in $(seq "$K_MIN" "$K_MAX"); do
  for j in $(seq "$J_MIN" "$J_MAX"); do
    for i in $(seq "$I_MIN" "$I_MAX"); do
      if (( task_counter < TASK_OFFSET )); then
        task_counter=$((task_counter + 1))
        continue
      fi
      if (( tasks_run >= TASK_COUNT )); then
        done_selection=1
        break
      fi

      # Turn that index into a unique run number for this selected task, starting at 1.
      run_num=$((tasks_run + 1))
      # Format the run number as a four-digit string for folder names.
      run_num_str=$(printf '%04d' "$run_num")
      # Build a human-readable task ID from k,j,i.
      task_id=$(printf 'k%03d_j%03d_i%03d' "$k" "$j" "$i")

      # Wait until we can start a new parallel task.
      wait_for_slot
      # Start the gridpoint task in the background.
      process_gridpoint "$k" "$j" "$i" "$task_id" "$run_num" "$run_num_str" &

      tasks_run=$((tasks_run + 1))
      task_counter=$((task_counter + 1))
    done
    if (( done_selection )); then
      break
    fi
  done
  if (( done_selection )); then
    break
  fi
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
shopt -s nullglob
result_files=("${JOB_RESULTS_DIR}"/tau_*.tsv)
if (( ${#result_files[@]} == 0 )); then
  echo "No task result files found in ${JOB_RESULTS_DIR}" >&2
  exit 1
fi
sort -n -k1,1 -k2,2 -k3,3 "${result_files[@]}" > "$RESULTS_TSV"
shopt -u nullglob

# ----------------------------
# Build the final 3D NetCDF file
# ----------------------------
python - "$RESULTS_TSV" "$OUT_NC" "$NZ" "$NY" "$NX" <<'PY'
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

log "Finished."
log "Text results: ${RESULTS_TSV}"
log "NetCDF output: ${OUT_NC}"