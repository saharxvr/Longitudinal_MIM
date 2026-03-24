@echo off
setlocal

REM Run Prediction.py for pairs 1-100 with local segs.
REM Uses default model path from Prediction.py.

cd /d "%~dp0.."

python Prediction.py --use-segs ^
  --pairs-roots "school_pc_prediction\input_pairs_1_100" ^
  --segs-dir "school_pc_prediction\segs_1_100" ^
  --preds-dir "school_pc_prediction\predictions_1_100"

echo.
echo Done. Outputs are in school_pc_prediction\predictions_1_100
pause
