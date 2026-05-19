#!/usr/bin/env python3  # Use the system Python interpreter to run this wrapper.
# Wrapper script to launch the Isca experiment for a single task.

import importlib.util  # Dynamically load the Isca experiment script by filename.
import os             # Filesystem and environment helpers.
import sys            # Access command-line arguments.

# Expect five, six, or seven arguments from the bash driver.
# 1: experiment script path
# 2: run number
# 3: restart archive path
# 4: number of cores
# 5: optional output root
# 6: optional task id for isolating temporary workdirs
if len(sys.argv) not in (5, 6, 7):
    print(
        "Usage: run_isca_wrapper.py <experiment_script> <run_num> <restart_archive> <num_cores> [<output_root> [<task_id>]]",
        file=sys.stderr,
    )
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
output_root = sys.argv[5] if len(sys.argv) >= 6 else None
# Optional per-task ID used to isolate temporary Isca workdirs.
task_id = sys.argv[6] if len(sys.argv) == 7 else None

# Is verbose output enabled for this wrapper? Otherwise silence informational prints.
verbose = os.environ.get('VERBOSE', 'yes').lower() in ('yes', 'true', '1')

# Validate that the experiment script actually exists.
if not os.path.isfile(script_path):
    raise FileNotFoundError(f"Experiment script not found: {script_path}")

# Dynamically load the experiment script as a Python module so we can access its exp object.
spec = importlib.util.spec_from_file_location('isca_experiment', script_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# The experiment script must define an `exp` object for Isca execution.
if not hasattr(module, 'exp'):
    raise AttributeError(f"Module {script_path} does not define exp")

# Set a custom output root for the experiment, if requested.
# This controls where Isca writes the final run output folder, e.g. run0001.
if output_root:
    os.makedirs(output_root, exist_ok=True)
    module.exp.datadir = output_root
    if verbose:
        print(f"Setting exp.datadir = {output_root}")

# If a task-specific ID is provided, isolate the temporary Isca workdir for this
# gridpoint so multiple tasks do not race on the shared GFDL_WORK/experiment name.
if task_id:
    workbase = os.environ.get('GFDL_WORK')
    if not workbase:
        raise RuntimeError(
            'GFDL_WORK is not set in the environment, cannot isolate per-task workdir'
        )

    # Use the existing experiment work directory base under GFDL_WORK.
    module.exp.workdir = os.path.join(workbase, 'experiment', module.exp.name)
    # Replace the standard single run folder with a unique run_<task_id> directory.
    module.exp.rundir = os.path.join(module.exp.workdir, f'run_{task_id}')
    os.makedirs(module.exp.rundir, exist_ok=True)
    # Avoid the default output folder nesting of run#### inside the point_root.
    module.exp.runfmt = ''
    if verbose:
        print(f"Setting exp.workdir = {module.exp.workdir}")
        print(f"Setting exp.rundir = {module.exp.rundir}")
        print(f"Setting exp.runfmt = {module.exp.runfmt!r}")

# Log the command before launching Isca.
if verbose:
    print(f"Running Isca exp.run({run_num}) with restart {restart_archive} and {num_cores} cores")

# Launch the model for exactly one timestep using the given restart archive.
module.exp.run(
    run_num,
    restart_file=restart_archive,
    use_restart=True,
    num_cores=num_cores,
    overwrite_data=True,
)
