import json
import os

base = "Sahar_work/files/ov_results"
dirs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))

rows = []
for d in dirs:
    fp = os.path.join(base, d, "precision_measures.json")
    if not os.path.exists(fp):
        continue
    with open(fp) as f:
        data = json.load(f)
    rows.append({
        "ROI": d,
        "HMDR_P": data.get("Model HMDR (Positive)", None),
        "HMDR_N": data.get("Model HMDR (Negative)", None),
        "UDPP_P": data.get("UDPP Model (Positive)", None),
        "UDPP_N": data.get("UDPP Model (Negative)", None),
    })

header = f"{'ROI':<22} {'HMDR Pos':>10} {'HMDR Neg':>10} {'UDPP Pos':>10} {'UDPP Neg':>10}"
print(header)
print("-" * len(header))
for r in rows:
    print(f"{r['ROI']:<22} {r['HMDR_P']:>10.4f} {r['HMDR_N']:>10.4f} {r['UDPP_P']:>10.4f} {r['UDPP_N']:>10.4f}")
