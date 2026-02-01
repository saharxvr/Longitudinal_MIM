
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ResolvedColumns:
	subject_id: str
	study_date: str
	study_time: str | None
	view_position: str | None
	study_id: str | None
	dicom_id: str | None
	performed_procedure_step_description: str | None
	procedure_code_meaning: str | None
	view_code_meaning: str | None


def _normalize_col_key(name: str) -> str:
	return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _resolve_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
	normalized_map: dict[str, str] = {_normalize_col_key(c): c for c in columns}
	for candidate in candidates:
		key = _normalize_col_key(candidate)
		if key in normalized_map:
			return normalized_map[key]
	return None


def _resolve_columns(df: pd.DataFrame) -> ResolvedColumns:
	subject_id = _resolve_column(df.columns, ["subject_id", "subjectid", "patient_id", "patientid"])
	if subject_id is None:
		raise KeyError(
			"Could not find subject_id column. Available columns: " + ", ".join(map(str, df.columns))
		)

	study_date = _resolve_column(df.columns, ["StudyDate", "study_date", "date"])
	if study_date is None:
		raise KeyError(
			"Could not find StudyDate column. Available columns: " + ", ".join(map(str, df.columns))
		)

	study_time = _resolve_column(df.columns, ["StudyTime", "study_time", "time"])

	return ResolvedColumns(
		subject_id=subject_id,
		study_date=study_date,
		study_time=study_time,
		view_position=_resolve_column(df.columns, ["ViewPosition", "view_position", "view"]),
		study_id=_resolve_column(df.columns, ["study_id", "studyid"]),
		dicom_id=_resolve_column(df.columns, ["dicom_id", "dicomid", "image_id", "imageid"]),
		performed_procedure_step_description=_resolve_column(
			df.columns, ["PerformedProcedureStepDescription", "performed_procedure_step_description"]
		),
		procedure_code_meaning=_resolve_column(
			df.columns, ["ProcedureCodeSequence_CodeMeaning", "procedure_code_sequence_codemeaning"]
		),
		view_code_meaning=_resolve_column(
			df.columns, ["ViewCodeSequence_CodeMeaning", "view_code_sequence_codemeaning"]
		),
	)


def _to_yyyymmdd(series: pd.Series) -> pd.Series:
	s = series.astype("string")
	s = s.str.replace("\\.0$", "", regex=True)
	s = s.str.replace(r"\D+", "", regex=True)
	s = s.str.zfill(8)
	s = s.where(s.str.len() == 8)
	return s


def _to_hhmmss(series: pd.Series | None, *, default: str = "000000") -> pd.Series:
	if series is None:
		return pd.Series([default] * 0, dtype="string")

	s = series.astype("string")
	s = s.fillna(default)
	s = s.str.replace("\\.0$", "", regex=True)
	s = s.str.replace(r"\D+", "", regex=True)
	s = s.str.slice(0, 6)
	s = s.str.zfill(6)
	s = s.where(s.str.len() == 6, default)
	return s


def _compute_study_datetime(df: pd.DataFrame, cols: ResolvedColumns) -> pd.Series:
	date_str = _to_yyyymmdd(df[cols.study_date])
	if cols.study_time is None:
		time_str = pd.Series(["000000"] * len(df), index=df.index, dtype="string")
	else:
		time_str = _to_hhmmss(df[cols.study_time])
		time_str = time_str.reindex(df.index, fill_value="000000")

	dt_str = date_str.fillna("") + time_str.fillna("000000")
	study_dt = pd.to_datetime(dt_str, format="%Y%m%d%H%M%S", errors="coerce")
	return study_dt


def _compute_study_date_only(df: pd.DataFrame, cols: ResolvedColumns) -> pd.Series:
	date_str = _to_yyyymmdd(df[cols.study_date])
	return pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")


def _ap_mask(df: pd.DataFrame, cols: ResolvedColumns, *, ap_mode: str) -> pd.Series:
	if cols.view_position is None:
		if ap_mode == "strict":
			raise KeyError(
				"AP filtering requires a ViewPosition column in strict mode. "
				"Use --ap-mode broad or ensure ViewPosition exists."
			)
		base = pd.Series([False] * len(df), index=df.index)
	else:
		base = df[cols.view_position].astype("string").str.upper().eq("AP")

	if ap_mode == "strict":
		return base.fillna(False)

	# Broad: include portable AP and coded antero-posterior markers.
	mask = base.fillna(False)
	portable_text = "CHEST (PORTABLE AP)"
	ap_code_text = "antero-posterior"

	if cols.performed_procedure_step_description is not None:
		mask |= df[cols.performed_procedure_step_description].astype("string").str.upper().eq(
			portable_text
		)

	if cols.procedure_code_meaning is not None:
		mask |= df[cols.procedure_code_meaning].astype("string").str.upper().eq(portable_text)

	if cols.view_code_meaning is not None:
		mask |= df[cols.view_code_meaning].astype("string").str.lower().eq(ap_code_text)

	return mask.fillna(False)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Filter MIMIC-CXR metadata to AP scans, remove patients with <2 scans, "
			"and sort by patient then scan date."
		)
	)

	parser.add_argument(
		"--input",
		type=Path,
		default=Path("mimic-cxr-2.0.0-metadata.xlsx"),
		help="Path to mimic-cxr-2.0.0-metadata.xlsx",
	)
	parser.add_argument(
		"--sheet",
		type=str,
		default=None,
		help="Excel sheet name (optional). If omitted, uses the first sheet.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("ap_sorted_patient_date.csv"),
		help="Output CSV path.",
	)
	parser.add_argument(
		"--ap-mode",
		choices=["strict", "broad"],
		default="broad",
		help=(
			"AP filter mode. strict = ViewPosition == AP only. "
			"broad = also includes portable AP / coded antero-posterior fields when present."
		),
	)
	parser.add_argument(
		"--min-scans",
		type=int,
		default=2,
		help="Keep only patients with at least this many AP scans.",
	)
	parser.add_argument(
		"--dedupe-study",
		action="store_true",
		help="If study_id exists, keep only the earliest image per (subject_id, study_id).",
	)

	parser.add_argument(
		"--pairs-only",
		action="store_true",
		help=(
			"If set, keep only rows that have a previous scan for the same patient "
			"(i.e., consecutive scan pairs)."
		),
	)

	return parser


def main() -> int:
	args = build_parser().parse_args()

	if not args.input.exists():
		raise FileNotFoundError(f"Input not found: {args.input}")

	sheet_name: str | int
	sheet_name = args.sheet if args.sheet is not None else 0
	df = pd.read_excel(args.input, sheet_name=sheet_name, engine="openpyxl")
	cols = _resolve_columns(df)

	# Filter to AP scans.
	ap_mask = _ap_mask(df, cols, ap_mode=args.ap_mode)
	df = df.loc[ap_mask].copy()

	# Build date and sort by patient then date (time only breaks ties).
	df["study_date_only"] = _compute_study_date_only(df, cols)
	sort_cols = [cols.subject_id, "study_date_only"]

	# Optional tie-breaker when StudyTime exists: keep a stable within-day ordering.
	if cols.study_time is not None:
		df["study_datetime"] = _compute_study_datetime(df, cols)
		sort_cols.append("study_datetime")

	df = df.sort_values(by=sort_cols, na_position="last")

	# Optional: dedupe multiple images per study (keep earliest by time).
	if args.dedupe_study and cols.study_id is not None:
		df = df.drop_duplicates(subset=[cols.subject_id, cols.study_id], keep="first")

	# Drop patients with too few scans.
	df["n_scans_patient"] = df.groupby(cols.subject_id)[cols.subject_id].transform("size")
	df = df.loc[df["n_scans_patient"] >= args.min_scans].copy()

	# Assign per-patient scan order after all filtering.
	df = df.sort_values(by=sort_cols, na_position="last")
	df["scan_order"] = df.groupby(cols.subject_id).cumcount() + 1

	# Compute delta time to previous scan using StudyDate only (ignores time-of-day).
	df["prev_study_date_only"] = df.groupby(cols.subject_id)["study_date_only"].shift(1)
	delta_days = (df["study_date_only"] - df["prev_study_date_only"]).dt.days
	df["delta_days"] = delta_days

	if args.pairs_only:
		df = df.loc[df["prev_study_date_only"].notna()].copy()

	df.to_csv(args.output, index=False)

	n_patients = df[cols.subject_id].nunique(dropna=True)
	print(
		"Wrote",
		args.output,
		f"({len(df)} rows, {n_patients} patients; ap_mode={args.ap_mode}, min_scans={args.min_scans})",
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
