#!/usr/bin/env python3
# Wrapper script to run the pressure-level interpolation for a single Isca task.

import os
import sys

# Expect exactly five command-line arguments.
if len(sys.argv) != 6:
    print("Usage: run_plevel_wrapper.py <plevel_script_dir> <exp_name> <run_num> <base_dir> <out_dir>", file=sys.stderr)
    sys.exit(1)

# Directory containing the original plevel script and plevel_fn module.
plevel_script_dir = sys.argv[1]
# The experiment name from the Isca output tree.
exp_name = sys.argv[2]
# The run number for this task.
run_num = int(sys.argv[3])
# Root directory containing the Isca output folders.
base_dir = sys.argv[4]
# Output directory where interpolated files will be written.
output_dir = sys.argv[5]

# Add the plevel script directory to Python search path.
sys.path.insert(0, plevel_script_dir)
# Is verbose output enabled for this wrapper? Otherwise silence informational prints.
verbose = os.environ.get('VERBOSE', 'yes').lower() in ('yes', 'true', '1')
try:
    from plevel_fn import plevel_call
except Exception as exc:
    raise RuntimeError(f"Unable to import plevel_fn from {plevel_script_dir}: {exc}")

# Format the run number as a zero-padded four-digit string.
run_num_str = f"{run_num:04d}"

# Build candidate input paths for the Isca task output.
direct_input = os.path.join(base_dir, "atmos_monthly.nc")
input_dir = os.path.join(base_dir, f"run{run_num_str}")
alt_input_dir = os.path.join(base_dir, exp_name, f"run{run_num_str}")
if os.path.isfile(direct_input):
    input_file = direct_input
elif os.path.isfile(os.path.join(input_dir, "atmos_monthly.nc")):
    input_file = os.path.join(input_dir, "atmos_monthly.nc")
elif os.path.isfile(os.path.join(alt_input_dir, "atmos_monthly.nc")):
    input_file = os.path.join(alt_input_dir, "atmos_monthly.nc")
else:
    raise FileNotFoundError(
        f"Input file not found in {direct_input}, {input_dir}, or {alt_input_dir}"
    )

# Use the output directory exactly as provided.
os.makedirs(output_dir, exist_ok=True)
# Interpolated file path.
out_file = os.path.join(output_dir, "atmos_monthly_interp_full.nc")

# Ensure the input file exists before calling plevel.
if not os.path.isfile(input_file):
    raise FileNotFoundError(f"Input file not found: {input_file}")

# Use the same full-level pressure-list and variable list from the original script.
plevs = {
    'full': ' -p "1 3 6 11 18 27 41 62 91 132 190 270 378 524 719 976 1310 1743 2296 2998 3880 4979 6335 7998 10020 12459 15381 18857 22962 27777 33360 39683 46538 53543 60291 66509 72085 76998 81270 84945 88077 90728 92957 94823 96379 97671 98739 99611"'
}
var_names = {
    'full': ' rh sphum ucomp vcomp omega height temp soc_tdt_lw soc_tdt_sw soc_tdt_rad dt_tg_convection dt_tg_condensation cf reff_rad frac_liq qcl_rad rh_in_cf'
}

# Mask values below the surface pressure during interpolation.
mask_below_surface_option = '-x'

# Remove an existing output file before writing a fresh one.
if os.path.isfile(out_file):
    os.remove(out_file)

# Call the pressure-level interpolation function.
plevel_call(
    input_file,
    out_file,
    var_names=var_names['full'],
    p_levels=plevs['full'],
    mask_below_surface_option=mask_below_surface_option,
)

# Print the output path so the caller can see it.
if verbose:
    print(out_file)
