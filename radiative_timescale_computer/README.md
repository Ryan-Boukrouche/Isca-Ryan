# Radiative Timescale Computer

This repository contains scripts for computing radiative timescales from Isca restart fields by perturbing individual gridpoints and running Isca for one timestep.

## Files

- `looper.sh`
  - Main driver that selects gridpoints, perturbs one temperature point per task, runs Isca, interpolates output, extracts `soc_tdt_rad`, and writes results.
  - Supports parallel execution with `N_PARALLEL`, chunked range selection with `TASK_OFFSET`/`TASK_COUNT`, and isolated output per `RUN_ID`.

- `run_ne_dayside_chunks.sh`
  - Driver for north-east dayside execution in 32-point chunks.
  - Uses `looper.sh` with repeating `RUN_ID` values and `TASK_OFFSET`/`TASK_COUNT` to process range subsets.

## Dependencies

- `bash` with `set -euo pipefail`
- `micromamba` for activating `isca_env`
- Python packages required by Isca and the interpolation scripts
- `xarray`, `numpy`
- `run_isca_wrapper.py` and `run_plevel_wrapper.py` must be present in the same directory as `looper.sh`

## Usage

### Run a chunked north-east dayside sequence

```bash
cd /.../radiative_timescale_computer
./run_ne_dayside_chunks.sh
```

Optional environment overrides:

```bash
VERBOSE=yes N_PARALLEL=32 TASK_COUNT=32 ./run_ne_dayside_chunks.sh
```

### Run `looper.sh` directly

```bash
K_MIN=36 K_MAX=47 J_MIN=32 J_MAX=63 I_MIN=0 I_MAX=31 \
TASK_OFFSET=0 TASK_COUNT=32 N_PARALLEL=32 CLEANUP_TASK_DIRS=yes VERBOSE=yes \
RUN_ID="ne_j032-063_i000-031_k36-47_chunk001" \
bash looper.sh
```

### Example single-point invocation

```bash
K_MIN=10 K_MAX=10 J_MIN=20 J_MAX=20 I_MIN=30 I_MAX=30 \
N_PARALLEL=1 CLEANUP_TASK_DIRS=no VERBOSE=yes bash looper.sh
```

## Output structure

`looper.sh` writes outputs under `OUTROOT`, which defaults to the experiment `radiative_timescale_output` folder.

When `RUN_ID` is set, outputs are isolated under:

```
${OUTROOT}/${RUN_ID}/
```

Inside that folder:

- `parallel_work/` — temporary task directories
- `job_results/` — one TSV per task: `tau_####_kXXX_jYYY_iZZZ.tsv`
- `tau_rad_results_<RUN_ID>.tsv` — merged, sorted result table
- `tau_rad_<RUN_ID>.nc` — 3D NetCDF with `tau_rad`
- `kXXX_jYYY_iZZZ/` — per-gridpoint Isca outputs

## Notes

- `looper.sh` expects the baseline restart archive at `ORIGINAL_RESTART_ARCHIVE`.
- `run_ne_dayside_chunks.sh` is a convenience wrapper for the north-east dayside region and does not modify Isca internals.

## License

TBC
