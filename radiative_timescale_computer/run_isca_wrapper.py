#!/usr/bin/env python3
# Wrapper script to launch the Isca experiment for a single task.

import importlib.util
import os
import sys

# Expect four or five arguments from the bash driver.
if len(sys.argv) not in (5, 6):
    print("Usage: run_isca_wrapper.py <experiment_script> <run_num> <restart_archive> <num_cores> [<output_root>]", file=sys.stderr)
    sys.exit(1)

# Read the path to the Isca experiment script.
script_path = sys.argv[1]
# The integer run number assigned to this parallel task.
run_num = int(sys.argv[2])
# The restart archive created for this task.
restart_archive = sys.argv[3]
# Number of CPU cores to use for this Isca execution.
num_cores = int(sys.argv[4])
# Optional root directory where Isca output should be written.
output_root = sys.argv[5] if len(sys.argv) == 6 else None

# Validate that the experiment script actually exists.
if not os.path.isfile(script_path):
    raise FileNotFoundError(f"Experiment script not found: {script_path}")

# Dynamically load the experiment script as a Python module.
spec = importlib.util.spec_from_file_location('isca_experiment', script_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# The experiment script must define an `exp` object for Isca execution.
if not hasattr(module, 'exp'):
    raise AttributeError(f"Module {script_path} does not define exp")

# Set a custom output root for the experiment, if requested.
if output_root:
    os.makedirs(output_root, exist_ok=True)
    module.exp.datadir = output_root
    print(f"Setting exp.datadir = {output_root}")

# Build the restart archive path that Isca expects.
print(f"Running Isca exp.run({run_num}) with restart {restart_archive} and {num_cores} cores")

# Launch the model for exactly one timestep using the given restart.
module.exp.run(run_num, restart_file=restart_archive, use_restart=True, num_cores=num_cores, overwrite_data=True)
