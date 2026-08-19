"""Convert PNIMIT batch 2 JPGs to NIfTI and build sequential pair folders.

Input layout:
    python_files/Sahar_work/Pnimit_batch_2/A 21/SynapseExport (57).jpg
                                          /A 21/SynapseExport (58).jpg
                                          ...

Output:
    1) A *.nii.gz file is written next to every *.jpg in the source tree.
       Conversion is byte-identical to python_files/jpg_to_nii (grayscale
       array, identity affine, no rotate / no mirror / no transpose).
    2) python_files/Sahar_work/Pnimit_batch_2_pairs/pair_A{N}_{i}_{i+1}/
       directories are created, each containing the two consecutive
       SynapseExport (k).nii.gz files for that patient (copied, not moved).

This mirrors the batch-1 layout under
python_files/annotation tool/Pairs_PNIMIT_1_pairs/.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR / "Pnimit_batch_2"
PAIRS_ROOT = SCRIPT_DIR / "Pnimit_batch_2_pairs"

# "A 21" -> patient id "A21"
PATIENT_DIR_RE = re.compile(r"^A\s*(\d+)$")
# "SynapseExport (57).jpg" -> 57
SYNAPSE_RE = re.compile(r"SynapseExport \((\d+)\)\.jpg$", re.IGNORECASE)


def jpg_to_nii(jpg_path: Path, nii_path: Path) -> None:
    """Convert a single JPG to NIfTI matching batch-1 orientation.

    The annotation tool (python_files/annotation tool/main.py) loads NIfTI
    via ``nii.get_fdata().T``. Batch 1 files store the array transposed
    on disk so that this load-time ``.T`` recovers the original (H, W)
    image orientation. We replicate that here -- save ``arr.T`` -- so the
    images render upright and un-mirrored in the annotation tool.
    """
    img = Image.open(jpg_path).convert("L")  # grayscale
    data = np.array(img).T  # store (W, H) so tool's .T -> (H, W)
    nii = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(nii, str(nii_path))


def convert_patient_dir(patient_dir: Path) -> list[Path]:
    """Convert every *.jpg in patient_dir to *.nii.gz next to it.

    Returns the produced .nii.gz paths sorted by SynapseExport number.
    """
    jpgs = sorted(
        (p for p in patient_dir.iterdir() if SYNAPSE_RE.search(p.name)),
        key=lambda p: int(SYNAPSE_RE.search(p.name).group(1)),
    )
    nii_paths: list[Path] = []
    for jpg in jpgs:
        nii = jpg.with_suffix("")  # strip .jpg
        nii = nii.with_name(nii.name + ".nii.gz")
        jpg_to_nii(jpg, nii)
        print(f"  wrote {nii.name}")
        nii_paths.append(nii)
    return nii_paths


def build_pairs(patient_id: str, nii_paths: list[Path]) -> None:
    """Create pair_A{patient_id}_{i}_{i+1}/ folders for consecutive timepoints."""
    if len(nii_paths) < 2:
        print(f"  [{patient_id}] only {len(nii_paths)} timepoint(s); no pairs")
        return
    for idx in range(len(nii_paths) - 1):
        a, b = nii_paths[idx], nii_paths[idx + 1]
        pair_name = f"pair_{patient_id}_{idx + 1}_{idx + 2}"
        pair_dir = PAIRS_ROOT / pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)
        for src in (a, b):
            dst = pair_dir / src.name
            shutil.copy2(src, dst)
        print(f"  built {pair_name} ({a.name} + {b.name})")


def main() -> None:
    if not SRC_ROOT.is_dir():
        raise SystemExit(f"missing source dir: {SRC_ROOT}")
    PAIRS_ROOT.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted(
        (p for p in SRC_ROOT.iterdir() if p.is_dir() and PATIENT_DIR_RE.match(p.name)),
        key=lambda p: int(PATIENT_DIR_RE.match(p.name).group(1)),
    )
    if not patient_dirs:
        raise SystemExit(f"no 'A NN' folders found under {SRC_ROOT}")

    for pdir in patient_dirs:
        patient_id = "A" + PATIENT_DIR_RE.match(pdir.name).group(1)
        print(f"[{patient_id}] {pdir}")
        nii_paths = convert_patient_dir(pdir)
        build_pairs(patient_id, nii_paths)

    print(f"\nDone. Pairs written to: {PAIRS_ROOT}")


if __name__ == "__main__":
    main()
