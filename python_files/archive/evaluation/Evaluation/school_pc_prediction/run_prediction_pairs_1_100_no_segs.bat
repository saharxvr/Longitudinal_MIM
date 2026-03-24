@echo off
setlocal

REM Run Prediction.py for pairs 1-100 without segmentation dependency.
REM Uses default model path from Prediction.py.

cd /d "%~dp0.."

python Prediction.py --no-segs ^
  --pairs-roots "school_pc_prediction\input_pairs_1_100" ^
  --preds-dir "school_pc_prediction\predictions_1_100"

echo.
echo Done. Outputs are in school_pc_prediction\predictions_1_100
pause
