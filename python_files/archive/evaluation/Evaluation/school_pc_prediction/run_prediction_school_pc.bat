@echo off
setlocal

REM Run Prediction.py with local school-PC paths.
REM Uses default model path from Prediction.py (no --model-path flag).

cd /d "%~dp0.."

python Prediction.py --use-segs ^
  --pairs-roots "school_pc_prediction\input_pairs" ^
  --segs-dir "school_pc_prediction\segs" ^
  --preds-dir "school_pc_prediction\predictions"

echo.
echo Done. Outputs are in school_pc_prediction\predictions
pause
