School PC Prediction Setup

This setup keeps Prediction.py unchanged and uses its default model path.

Prepared datasets inside this folder:
1) Pairs 1-100
- Input pairs root: school_pc_prediction\input_pairs_1_100
- Segs dir: school_pc_prediction\segs_1_100
- Output dir used by script: school_pc_prediction\predictions_1_100

2) PNIMIT pairs
- Input pairs root: school_pc_prediction\input_pairs_pnimit
- Segs dir: school_pc_prediction\segs_pnimit
- Output dir used by script: school_pc_prediction\predictions_pnimit

Run options:
- Double-click run_prediction_pairs_1_100.bat
- Double-click run_prediction_pnimit.bat

Equivalent commands:
python Prediction.py --use-segs --pairs-roots "school_pc_prediction\input_pairs_1_100" --segs-dir "school_pc_prediction\segs_1_100" --preds-dir "school_pc_prediction\predictions_1_100"
python Prediction.py --use-segs --pairs-roots "school_pc_prediction\input_pairs_pnimit" --segs-dir "school_pc_prediction\segs_pnimit" --preds-dir "school_pc_prediction\predictions_pnimit"
python Prediction.py --no-segs --pairs-roots "school_pc_prediction\input_pairs_1_100" --preds-dir "school_pc_prediction\predictions_1_100"

Seg naming expected by Prediction.py:
- <image_name>_seg.nii.gz

Notes:
- This package was populated from annotation tool/Pairs1..Pairs8 and annotation tool/Pairs_PNIMIT_1_pairs.
- If Python is not in PATH on school PCs, run with the full Python executable path.
- In the current source data, Pair99 and Pair100 do not have matching seg files, so running 1-100 with --use-segs may fail on those two pairs.
