"""Run Observer Variability for all 8 ROI types sequentially."""
import subprocess
import sys
import os

ROI_TYPES = [
    "full_image",
    "lungs",
    "lungs_heart",
    "lungs_mediastinum",
    "full_thorax",
    "lungs_margin5",
    "lungs_med_margin5",
    "lungs_convex_hull",
]

base = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(base, "archive", "evaluation", "Evaluation", "Observer_Variability_original_39cb6f8.py")
python = sys.executable

for roi in ROI_TYPES:
    print(f"\n{'='*60}")
    print(f"  Running OV for: {roi}")
    print(f"{'='*60}\n")
    cmd = [
        python, script,
        "--model-preds-dir", os.path.join(base, "Sahar_work", "files", "predictions", roi),
        "--annotations-dir", os.path.join(base, "annotation tool", "Annotations"),
        "--pairs-roots",
            os.path.join(base, "annotation tool", "Pairs1"),
            os.path.join(base, "annotation tool", "Pairs2"),
            os.path.join(base, "annotation tool", "Pairs3"),
            os.path.join(base, "annotation tool", "Pairs4"),
        "--out-dir", os.path.join(base, "Sahar_work", "files", "ov_results", roi),
        "--num-pairs", "60",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ERROR: {roi} failed with return code {result.returncode}")
    else:
        print(f"  DONE: {roi}")

print("\n\nAll 8 ROI types completed.")
