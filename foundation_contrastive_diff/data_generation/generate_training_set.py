"""Generate the synthetic training set for the foundation_contrastive_diff study.

Policy (see ../RQ1_PLAN.md sec. 3a): each (prior, current) pair has
    * AT MOST ONE pathology  (consolidation | pleural_effusion | pneumothorax | fluid_overload)
    * a projection-angle difference between prior and current
    * devices (nuisance) rendered in most pairs but IGNORED in the GT diff map

This is a thin driver that reuses the existing generator
`python_files/CT_entities/DRR_generator.py` via its new `--single_pathology` mode.
It does not re-implement any DRR logic; it only fixes the RQ1 entity policy and
forwards CT paths / output / CT-list slice to the underlying generator.

Meant to run on the university Linux PCs (HUJI josko cluster) inside the same
virtualenv used to generate the original DRRs. See `run_generate_pairs.slurm`.

Example (single machine):
    python generate_training_set.py -n 12000 -o /cs/labs/josko/.../final/fcd_train

Parallel split across N array tasks (task i of N):
    python generate_training_set.py -n 12000 -o <out> --slice_index i --num_slices N
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# Underlying generator (reused as-is).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
_DRR_GENERATOR = os.path.join(_REPO_ROOT, "python_files", "CT_entities", "DRR_generator.py")

# RQ1 taxonomy: 4 pathologies, balanced, with the remaining mass = clean ("none") pairs.
# Cardiomegaly is intentionally 0 (excluded from this study).
DEFAULT_PATHOLOGY_PROBS = {
    "Consolidation": 0.20,
    "PleuralEffusion": 0.20,
    "Pneumothorax": 0.20,
    "FluidOverload": 0.20,
    # -> 0.20 remaining probability mass = "none" (clean) pairs
    "Cardiomegaly": 0.0,
}
# Devices are nuisance: present in most pairs so the head learns to ignore them.
DEFAULT_DEVICE_PROB = 0.8
# Angle distribution: [max_abs_per_axis_deg, max_sum_deg, min_sum_deg, exponent].
DEFAULT_ROTATION_PARAMS = [17.5, 37.5, 0.0, 1.75]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("-n", "--number_pairs", type=int, required=True,
                   help="Total number of synthetic pairs to generate (across this process).")
    p.add_argument("-o", "--output", type=str, required=True,
                   help="Output directory for the generated pairs.")
    p.add_argument("-i", "--input", nargs="+", default=None,
                   help="CT directories. Defaults to the generator's built-in cluster paths.")

    # Pathology / device policy (override only if you need to).
    p.add_argument("--consolidation", type=float, default=DEFAULT_PATHOLOGY_PROBS["Consolidation"])
    p.add_argument("--pleural_effusion", type=float, default=DEFAULT_PATHOLOGY_PROBS["PleuralEffusion"])
    p.add_argument("--pneumothorax", type=float, default=DEFAULT_PATHOLOGY_PROBS["Pneumothorax"])
    p.add_argument("--fluid_overload", type=float, default=DEFAULT_PATHOLOGY_PROBS["FluidOverload"])
    p.add_argument("--devices_prob", type=float, default=DEFAULT_DEVICE_PROB)
    p.add_argument("--rotation_params", nargs=4, type=float, default=DEFAULT_ROTATION_PARAMS)
    p.add_argument("--memory_threshold", type=float, default=25.0)

    # Parallelization: split the CT list into num_slices contiguous chunks and run chunk slice_index.
    p.add_argument("--slice_index", type=int, default=0, help="Which CT-list chunk to process (0-based).")
    p.add_argument("--num_slices", type=int, default=1, help="Total number of CT-list chunks.")

    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter to run the generator with (defaults to the current venv).")
    p.add_argument("--dry_run", action="store_true", help="Print the generator command and exit.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.num_slices < 1 or not (0 <= args.slice_index < args.num_slices):
        raise SystemExit(f"Invalid slice: slice_index={args.slice_index}, num_slices={args.num_slices}")

    if not os.path.isfile(_DRR_GENERATOR):
        raise SystemExit(f"Could not find DRR_generator.py at: {_DRR_GENERATOR}")

    a = args.slice_index / args.num_slices
    b = (args.slice_index + 1) / args.num_slices

    cmd = [
        args.python, _DRR_GENERATOR,
        "-n", str(args.number_pairs),
        "-o", args.output,
        "--single_pathology",
        "-CO", str(args.consolidation),
        "-PL", str(args.pleural_effusion),
        "-PN", str(args.pneumothorax),
        "-FL", str(args.fluid_overload),
        "-CA", "0.0",                       # cardiomegaly excluded from this study
        "-EX", str(args.devices_prob),
        "-r", *[str(x) for x in args.rotation_params],
        "-s", f"{a:.6f}", f"{b:.6f}",
        "-m", str(args.memory_threshold),
    ]
    if args.input:
        cmd += ["-i", *args.input]

    print("[generate_training_set] Running:\n  " + " ".join(cmd), flush=True)
    if args.dry_run:
        return 0

    os.makedirs(args.output, exist_ok=True)
    # Run from the generator's own directory so its top-level imports resolve.
    return subprocess.run(cmd, cwd=os.path.dirname(_DRR_GENERATOR)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
