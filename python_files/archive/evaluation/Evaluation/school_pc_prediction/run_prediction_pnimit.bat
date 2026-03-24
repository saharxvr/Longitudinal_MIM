@echo off
setlocal

REM Run Prediction.py for PNIMIT pairs with local segs.
REM Uses default model path from Prediction.py.

cd /d "%~dp0.."

python Prediction.py --use-segs ^
  --pairs-roots "school_pc_prediction\input_pairs_pnimit" ^
  --segs-dir "school_pc_prediction\segs_pnimit" ^
  --preds-dir "school_pc_prediction\predictions_pnimit"

echo.
echo Done. Outputs are in school_pc_prediction\predictions_pnimit
pause
