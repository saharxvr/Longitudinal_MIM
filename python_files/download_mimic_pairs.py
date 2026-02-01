from __future__ import annotations

import argparse
import csv
import os
import shutil
import urllib.request
from base64 import b64encode
import urllib.error
from dataclasses import dataclass
import getpass
from pathlib import Path
from typing import Iterable
import re

import requests

import pandas as pd


@dataclass(frozen=True)
class PairRow:
    subject_id: str
    baseline_study_id: str
    baseline_dicom_id: str
    followup_study_id: str
    followup_dicom_id: str
    pair_index: int
    delta_days: float | int | None


@dataclass(frozen=True)
class ScanRow:
    subject_id: str
    study_id: str
    dicom_id: str
    study_date_yyyymmdd: str
    scan_order: int | None


def _normalize_subject_id(subject_id: str | int) -> str:
    # MIMIC-CXR uses folders like p10000032
    sid = str(subject_id)
    sid = sid.strip()
    if sid.startswith("p"):
        sid = sid[1:]
    return f"p{sid}"


def _normalize_study_id(study_id: str | int) -> str:
    # MIMIC-CXR uses folders like s53911762
    st = str(study_id)
    st = st.strip()
    if st.startswith("s"):
        st = st[1:]
    return f"s{st}"


def _normalize_ext(ext: str) -> str:
    ext = ext.strip()
    if not ext:
        raise ValueError("Empty --ext")
    if not ext.startswith("."):
        ext = "." + ext
    return ext


def _pXX_folder(p_subject_id: str) -> str:
    # In the repo code: sub_folder = subject_id[:3] where subject_id is like 'p10000032' -> 'p10'
    # That matches PhysioNet structure (p10/p10000032/...)
    return p_subject_id[:3]


def _build_rel_path(subject_id: str, study_id: str, dicom_id: str, ext: str) -> Path:
    p_subject = _normalize_subject_id(subject_id)
    s_study = _normalize_study_id(study_id)
    pxx = _pXX_folder(p_subject)

    did = str(dicom_id).strip()
    if did.endswith(ext):
        filename = did
    else:
        filename = did + ext

    return Path(pxx) / p_subject / s_study / filename


def _find_local_file(mimic_files_root: Path, rel_path: Path) -> Path | None:
    # Accept either a direct ".../files" root or a higher directory containing "files".
    candidates = []
    if mimic_files_root.name.lower() == "files":
        candidates.append(mimic_files_root / rel_path)
    candidates.append(mimic_files_root / "files" / rel_path)

    for path in candidates:
        if path.exists():
            return path
    return None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _copy_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists():
        if overwrite:
            dst.unlink()
        else:
            return
    _ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def _pairs_from_sorted_scans(df: pd.DataFrame) -> list[PairRow]:
    required = {"subject_id", "study_id", "dicom_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Input CSV missing required columns: {sorted(missing)}")

    delta_col = "delta_days" if "delta_days" in df.columns else None

    # Ensure stable order: if the file is already sorted by patient/date, this preserves it.
    df = df.copy()
    df["_subject_id"] = df["subject_id"].astype("string")

    pairs: list[PairRow] = []
    for subject_id, group in df.groupby("_subject_id", sort=False):
        group = group.reset_index(drop=True)
        if len(group) < 2:
            continue

        for i in range(1, len(group)):
            baseline = group.iloc[i - 1]
            followup = group.iloc[i]
            pairs.append(
                PairRow(
                    subject_id=str(subject_id),
                    baseline_study_id=str(baseline["study_id"]),
                    baseline_dicom_id=str(baseline["dicom_id"]),
                    followup_study_id=str(followup["study_id"]),
                    followup_dicom_id=str(followup["dicom_id"]),
                    pair_index=i,
                    delta_days=(None if delta_col is None else followup[delta_col]),
                )
            )

    return pairs


def _parse_yyyymmdd(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Accept "YYYYMMDD" or "YYYY-MM-DD"
    s = s.replace("-", "")
    s = "".join(ch for ch in s if ch.isdigit())
    if len(s) != 8:
        return None
    return s


def _date_str_from_row(row: pd.Series) -> str:
    # Prefer StudyDate, fallback to study_date_only
    if "StudyDate" in row:
        d = _parse_yyyymmdd(row["StudyDate"])
        if d:
            return d
    if "study_date_only" in row:
        d = _parse_yyyymmdd(row["study_date_only"])
        if d:
            return d
    raise KeyError("Could not find a valid StudyDate/study_date_only in input CSV")


def _scan_row_from_series(row: pd.Series) -> ScanRow:
    scan_order = None
    if "scan_order" in row and pd.notna(row["scan_order"]):
        try:
            scan_order = int(row["scan_order"])
        except Exception:
            scan_order = None
    return ScanRow(
        subject_id=str(row["subject_id"]),
        study_id=str(row["study_id"]),
        dicom_id=str(row["dicom_id"]),
        study_date_yyyymmdd=_date_str_from_row(row),
        scan_order=scan_order,
    )


def _select_pairs_by_gap(
    df: pd.DataFrame,
    *,
    min_gap_days: int,
    max_gap_days: int,
    max_patients: int | None,
    max_pairs: int | None,
) -> tuple[list[tuple[ScanRow, ScanRow, int]], dict[tuple[str, str, str], ScanRow]]:
    """Select consecutive pairs with gap in [min_gap_days, max_gap_days].

    Returns:
      - list of (baseline_scan, followup_scan, gap_days)
      - dict of unique scans referenced by those pairs
    """
    required = {"subject_id", "study_id", "dicom_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Input CSV missing required columns: {sorted(missing)}")

    df = df.copy()
    df["_subject_id"] = df["subject_id"].astype("string")

    selected_pairs: list[tuple[ScanRow, ScanRow, int]] = []
    unique_scans: dict[tuple[str, str, str], ScanRow] = {}

    n_patients_done = 0
    for subject_id, group in df.groupby("_subject_id", sort=False):
        if max_patients is not None and n_patients_done >= max_patients:
            break

        group = group.reset_index(drop=True)
        if len(group) < 2:
            continue

        n_patients_done += 1

        for i in range(1, len(group)):
            bl = _scan_row_from_series(group.iloc[i - 1])
            fu = _scan_row_from_series(group.iloc[i])

            # Compute gap days from date strings.
            bl_dt = pd.to_datetime(bl.study_date_yyyymmdd, format="%Y%m%d", errors="coerce")
            fu_dt = pd.to_datetime(fu.study_date_yyyymmdd, format="%Y%m%d", errors="coerce")
            if pd.isna(bl_dt) or pd.isna(fu_dt):
                continue

            gap = int((fu_dt - bl_dt).days)
            if gap < min_gap_days or gap > max_gap_days:
                continue

            selected_pairs.append((bl, fu, gap))

            unique_scans[(bl.subject_id, bl.study_id, bl.dicom_id)] = bl
            unique_scans[(fu.subject_id, fu.study_id, fu.dicom_id)] = fu

            if max_pairs is not None and len(selected_pairs) >= max_pairs:
                break
        if max_pairs is not None and len(selected_pairs) >= max_pairs:
            break

    return selected_pairs, unique_scans


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Create/download consecutive AP scan pairs from MIMIC-CXR v2.0.0 metadata output. "
            "Default behavior copies files from an existing local MIMIC-CXR folder; it can also emit PhysioNet URLs."
        )
    )

    p.add_argument(
        "--pairs-csv",
        type=Path,
        default=Path("ap_sorted_patient_date_with_deltas.csv"),
        help="CSV with at least subject_id, study_id, dicom_id, sorted by patient then date.",
    )

    p.add_argument(
        "--layout",
        choices=["pairs", "patient"],
        default="pairs",
        help=(
            "Output layout. pairs = pair_#### folders with baseline/followup. "
            "patient = one folder per patient, download each scan once, and write a pairs manifest."
        ),
    )
    p.add_argument(
        "--mimic-files-root",
        type=Path,
        default=None,
        help=(
            "Path to your local PhysioNet MIMIC-CXR folder (either the 'files' directory itself, "
            "or a parent directory that contains 'files'). If omitted, no local copying is done."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("mimic_pairs_out"),
        help="Where to write pair folders.",
    )
    p.add_argument(
        "--ext",
        type=str,
        default="dcm",
        help="File extension to fetch/copy (e.g., dcm for DICOM, jpg for MIMIC-CXR-JPG).",
    )

    p.add_argument(
        "--dataset",
        type=str,
        default="mimic-cxr",
        help="PhysioNet dataset slug for URL generation (e.g., mimic-cxr or mimic-cxr-jpg).",
    )
    p.add_argument(
        "--version",
        type=str,
        default="2.0.0",
        help="Dataset version for URL generation.",
    )
    p.add_argument(
        "--write-urls",
        type=Path,
        default=None,
        help="If set, write a urls.txt file (one URL per image) for downloading via your preferred tool.",
    )

    p.add_argument(
        "--min-gap-days",
        type=int,
        default=1,
        help="(patient layout) Keep only consecutive pairs with gap >= this.",
    )
    p.add_argument(
        "--max-gap-days",
        type=int,
        default=8,
        help="(patient layout) Keep only consecutive pairs with gap <= this.",
    )

    p.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="(patient layout) Optional cap on number of patients to process.",
    )

    p.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="(patient layout) Optional cap on number of unique scans to download.",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not download/copy files; only write manifests/URL lists.",
    )

    p.add_argument(
        "--test-url",
        type=str,
        default=None,
        help=(
            "Download a single PhysioNet URL (useful for debugging access). "
            "Example: https://physionet.org/files/.../file.dcm?download"
        ),
    )

    p.add_argument(
        "--download",
        action="store_true",
        help=(
            "Attempt to download missing files from PhysioNet. Requires access approval for MIMIC-CXR and "
            "credentials via env vars PHYSIONET_USERNAME and PHYSIONET_PASSWORD."
        ),
    )

    p.add_argument(
        "--auth-mode",
        choices=["session", "basic"],
        default="session",
        help=(
            "How to authenticate to PhysioNet. session=login like a browser (recommended). "
            "basic=HTTP basic auth (may fail with 403 on some endpoints)."
        ),
    )

    p.add_argument(
        "--download-query",
        type=str,
        default="?download",
        help=(
            "Query string to append to file URLs when downloading (PhysioNet often uses '?download'). "
            "Use empty string to disable."
        ),
    )

    p.add_argument(
        "--credentials-file",
        type=Path,
        default=None,
        help=(
            "Path to a local credentials file (DO NOT COMMIT). Format: key=value per line. "
            "Accepts PHYSIONET_USERNAME / PHYSIONET_PASSWORD or username / password."
        ),
    )
    p.add_argument(
        "--download-cache",
        type=Path,
        default=None,
        help=(
            "Where to cache downloaded files (mirrors PhysioNet relpaths). "
            "Defaults to --mimic-files-root if provided, else <out-dir>/physionet_cache."
        ),
    )

    p.add_argument(
        "--prompt-credentials",
        action="store_true",
        help="Prompt for PhysioNet credentials even if env vars are already set.",
    )

    p.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on number of pairs (useful for quick tests).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing copied files in out-dir.",
    )

    return p


def _physionet_url(dataset: str, version: str, rel_path: Path) -> str:
    # Note: Access to MIMIC-CXR on PhysioNet requires credentials + data use agreement.
    # This function just builds the URL.
    rel = str(rel_path).replace(os.sep, "/")
    return f"https://physionet.org/files/{dataset}/{version}/files/{rel}"


def _add_query(url: str, query: str) -> str:
    q = (query or "").strip()
    if not q:
        return url
    if q.startswith("?"):
        if "?" in url:
            return url + "&" + q.lstrip("?")
        return url + q
    # If user passes e.g. "download" or "dl=1"
    if "?" in url:
        return url + "&" + q
    return url + "?" + q


def _download_file(url: str, dst: Path, *, overwrite: bool) -> bool:
    """Download a single file from PhysioNet.

    Uses HTTP basic auth if PHYSIONET_USERNAME / PHYSIONET_PASSWORD are set.
    Returns True if the file exists at dst after the call.
    """
    if dst.exists() and not overwrite:
        return True

    username = os.environ.get("PHYSIONET_USERNAME")
    password = os.environ.get("PHYSIONET_PASSWORD")

    request = urllib.request.Request(url)
    if username and password:
        token = b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        request.add_header("Authorization", f"Basic {token}")

    _ensure_dir(dst.parent)
    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    try:
        with urllib.request.urlopen(request) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.replace(dst)
        return True
    except urllib.error.HTTPError as e:
        # Common causes: 401/403 (credentials/access), 404 (wrong dataset/version/path)
        msg = ""
        try:
            body = e.read(512)
            msg = body.decode("utf-8", errors="ignore").strip().replace("\n", " ")
        except Exception:
            msg = ""

        if msg:
            print(f"Download failed: HTTP {e.code} for {url} | {msg[:200]}")
        else:
            print(f"Download failed: HTTP {e.code} for {url}")
        if tmp.exists():
            tmp.unlink()
        return False


def _login_physionet_session(username: str, password: str) -> requests.Session:
    """Login to PhysioNet using a browser-like session.

    PhysioNet uses CSRF + cookies; direct file URLs can return 403 without a logged-in session.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    login_url = "https://physionet.org/login/"
    r = session.get(login_url, timeout=60)
    r.raise_for_status()

    # CSRF token can be in cookie and/or HTML form.
    csrf_cookie = session.cookies.get("csrftoken")
    m = re.search(r"name=\"csrfmiddlewaretoken\" value=\"([^\"]+)\"", r.text)
    csrf_form = m.group(1) if m else None
    csrf = csrf_form or csrf_cookie
    if not csrf:
        raise RuntimeError("Could not find CSRF token on PhysioNet login page")

    payload = {
        "username": username,
        "password": password,
        "csrfmiddlewaretoken": csrf,
        "next": "/",
    }

    headers = {
        "Referer": login_url,
    }
    # Some deployments require csrftoken cookie + header.
    session.headers.update({"X-CSRFToken": csrf})

    r2 = session.post(login_url, data=payload, headers=headers, timeout=60, allow_redirects=True)
    r2.raise_for_status()

    # Heuristic: if we're still on login page, authentication likely failed.
    if "/login/" in r2.url:
        raise RuntimeError("PhysioNet login did not succeed (still on /login/)")

    return session


def _download_file_session(url: str, dst: Path, *, overwrite: bool, session: requests.Session) -> bool:
    if dst.exists() and not overwrite:
        return True

    _ensure_dir(dst.parent)
    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    try:
        with session.get(url, stream=True, timeout=120, allow_redirects=True) as r:
            # 403 here means not approved for dataset or login not valid for files endpoint.
            if r.status_code != 200:
                snippet = ""
                try:
                    snippet = r.text[:200].replace("\n", " ")
                except Exception:
                    snippet = ""
                if snippet:
                    print(f"Download failed: HTTP {r.status_code} for {url} | {snippet}")
                else:
                    print(f"Download failed: HTTP {r.status_code} for {url}")
                return False

            content_type = (r.headers.get("Content-Type") or "").lower()
            if "text/html" in content_type:
                # Often indicates we got redirected to an HTML page rather than the file.
                snippet = r.text[:200].replace("\n", " ")
                print(f"Download failed: got HTML instead of file for {url} | {snippet}")
                return False

            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        tmp.replace(dst)
        return True
    except requests.RequestException as e:
        print(f"Download failed: {type(e).__name__} for {url}")
        if tmp.exists():
            tmp.unlink()
        return False
    except Exception as e:
        print(f"Download failed: {type(e).__name__} for {url}")
        # Leave no partials behind
        if tmp.exists():
            tmp.unlink()
        return False


def _ensure_physionet_credentials() -> None:
    """Ensure PHYSIONET_USERNAME / PHYSIONET_PASSWORD are available.

    If not present, prompt interactively (password hidden).
    """
    # Interactive prompt until non-empty
    while not os.environ.get("PHYSIONET_USERNAME"):
        os.environ["PHYSIONET_USERNAME"] = input("PhysioNet username: ").strip()
    while not os.environ.get("PHYSIONET_PASSWORD"):
        os.environ["PHYSIONET_PASSWORD"] = getpass.getpass("PhysioNet password (hidden): ")


def _load_credentials_file(path: Path, *, override_env: bool) -> None:
    """Load PHYSIONET_USERNAME / PHYSIONET_PASSWORD from a simple key=value file."""
    if not path.exists():
        raise FileNotFoundError(f"credentials file not found: {path}")

    username: str | None = None
    password: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lower()
        value = value.strip().strip("\"'")

        if key in {"physionet_username", "username"}:
            username = value
        elif key in {"physionet_password", "password"}:
            password = value

    if username is None or password is None:
        raise ValueError(
            "Credentials file must contain both username and password. "
            "Expected keys: PHYSIONET_USERNAME/PHYSIONET_PASSWORD (or username/password)."
        )

    if override_env:
        os.environ["PHYSIONET_USERNAME"] = username
        os.environ["PHYSIONET_PASSWORD"] = password
    else:
        if not os.environ.get("PHYSIONET_USERNAME"):
            os.environ["PHYSIONET_USERNAME"] = username
        if not os.environ.get("PHYSIONET_PASSWORD"):
            os.environ["PHYSIONET_PASSWORD"] = password


def main() -> int:
    args = build_parser().parse_args()

    # Test mode: download a single URL and exit.
    if args.test_url is not None:
        if not args.download:
            raise ValueError("--test-url requires --download")
        if args.prompt_credentials:
            os.environ.pop("PHYSIONET_USERNAME", None)
            os.environ.pop("PHYSIONET_PASSWORD", None)
        elif args.credentials_file is not None:
            _load_credentials_file(args.credentials_file, override_env=True)
        _ensure_physionet_credentials()

        session: requests.Session | None = None
        if args.auth_mode == "session":
            session = _login_physionet_session(
                os.environ.get("PHYSIONET_USERNAME") or "",
                os.environ.get("PHYSIONET_PASSWORD") or "",
            )

        url = _add_query(args.test_url, args.download_query)
        _ensure_dir(args.out_dir)
        out_name = Path(url.split("?", 1)[0]).name or "downloaded_file"
        dst = args.out_dir / out_name

        ok = (
            _download_file_session(url, dst, overwrite=args.overwrite, session=session)
            if args.auth_mode == "session" and session is not None
            else _download_file(url, dst, overwrite=args.overwrite)
        )
        if ok:
            print("Downloaded:", dst)
            return 0
        raise SystemExit(2)

    if not args.pairs_csv.exists():
        raise FileNotFoundError(f"pairs csv not found: {args.pairs_csv}")

    ext = _normalize_ext(args.ext)
    df = pd.read_csv(args.pairs_csv)

    # For patient layout, we filter by gap and then download unique scans only once.
    patient_pairs: list[tuple[ScanRow, ScanRow, int]] | None = None
    unique_scans: dict[tuple[str, str, str], ScanRow] | None = None
    pairs: list[PairRow] | None = None

    if args.layout == "patient":
        patient_pairs, unique_scans = _select_pairs_by_gap(
            df,
            min_gap_days=args.min_gap_days,
            max_gap_days=args.max_gap_days,
            max_patients=args.max_patients,
            max_pairs=args.max_pairs,
        )
    else:
        pairs = _pairs_from_sorted_scans(df)
        if args.max_pairs is not None:
            pairs = pairs[: args.max_pairs]

    _ensure_dir(args.out_dir)

    download_cache = args.download_cache
    if download_cache is None:
        download_cache = args.mimic_files_root if args.mimic_files_root is not None else (args.out_dir / "physionet_cache")

    # If the user asked to download but didn't provide a local root, use the cache as the root.
    if args.download and args.mimic_files_root is None:
        args.mimic_files_root = download_cache

    if args.download:
        # Precedence:
        # 1) --prompt-credentials forces interactive prompt
        # 2) --credentials-file loads if env vars not already set
        # 3) fallback interactive prompt if still missing
        if args.prompt_credentials:
            os.environ.pop("PHYSIONET_USERNAME", None)
            os.environ.pop("PHYSIONET_PASSWORD", None)
        elif args.credentials_file is not None:
            # Override env vars to avoid stale values from a prior terminal session.
            _load_credentials_file(args.credentials_file, override_env=True)
        _ensure_physionet_credentials()

    session: requests.Session | None = None
    if args.download and args.auth_mode == "session":
        username = os.environ.get("PHYSIONET_USERNAME") or ""
        password = os.environ.get("PHYSIONET_PASSWORD") or ""
        try:
            session = _login_physionet_session(username, password)
        except Exception as e:
            raise RuntimeError(
                "Could not login to PhysioNet with session auth. "
                "If you can login in a browser but this fails, you likely need to accept dataset access/DUA, "
                "or PhysioNet may be blocking automated logins from your network."
            ) from e

    url_lines: list[str] = []

    copied = 0
    missing_local = 0
    downloaded = 0
    failed_downloads = 0

    def _download_or_find(rel_path: Path, *, url: str) -> Path | None:
        nonlocal downloaded, failed_downloads
        src = _find_local_file(args.mimic_files_root, rel_path) if args.mimic_files_root is not None else None
        if src is not None:
            return src

        if not args.download or args.dry_run:
            return None

        cache_path = download_cache / "files" / rel_path
        ok = (
            _download_file_session(url, cache_path, overwrite=args.overwrite, session=session)
            if args.auth_mode == "session" and session is not None
            else _download_file(url, cache_path, overwrite=args.overwrite)
        )
        if ok:
            downloaded += 1
            return cache_path
        failed_downloads += 1
        return None

    try:
        if args.layout == "patient":
            assert patient_pairs is not None and unique_scans is not None

            # Write per-patient pairs manifest and download unique scans.
            pairs_by_patient: dict[str, list[tuple[ScanRow, ScanRow, int]]] = {}
            for bl, fu, gap in patient_pairs:
                p_subject = _normalize_subject_id(bl.subject_id)
                pairs_by_patient.setdefault(p_subject, []).append((bl, fu, gap))

            # Download unique scans; optional cap.
            scan_items = list(unique_scans.values())
            if args.max_files is not None:
                scan_items = scan_items[: args.max_files]

            # Ensure deterministic ordering.
            scan_items.sort(key=lambda s: (_normalize_subject_id(s.subject_id), s.study_date_yyyymmdd, s.study_id, s.dicom_id))

            for scan in scan_items:
                p_subject = _normalize_subject_id(scan.subject_id)
                patient_dir = args.out_dir / p_subject
                _ensure_dir(patient_dir)

                study_id_norm = _normalize_study_id(scan.study_id)
                # Filename includes date so you can infer gaps later.
                order_part = f"_o{scan.scan_order:03d}" if scan.scan_order is not None else ""
                out_name = f"{scan.study_date_yyyymmdd}{order_part}_{study_id_norm}_{scan.dicom_id}{ext}"
                dst = patient_dir / out_name

                rel = _build_rel_path(scan.subject_id, scan.study_id, scan.dicom_id, ext)
                url = _add_query(_physionet_url(args.dataset, args.version, rel), args.download_query)
                url_lines.append(url)

                src = _download_or_find(rel, url=url)
                if src is None:
                    missing_local += 1
                    continue
                if not args.dry_run:
                    _copy_file(src, dst, overwrite=args.overwrite)
                    copied += 1

            # Write per-patient manifest describing selected pairs and gaps.
            for p_subject, p_pairs in pairs_by_patient.items():
                patient_dir = args.out_dir / p_subject
                _ensure_dir(patient_dir)
                manifest_path = patient_dir / "pairs_1_8_days.csv"
                with open(manifest_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "baseline_date",
                            "baseline_study_id",
                            "baseline_dicom_id",
                            "followup_date",
                            "followup_study_id",
                            "followup_dicom_id",
                            "gap_days",
                        ]
                    )
                    for bl, fu, gap in p_pairs:
                        writer.writerow(
                            [
                                bl.study_date_yyyymmdd,
                                _normalize_study_id(bl.study_id),
                                bl.dicom_id,
                                fu.study_date_yyyymmdd,
                                _normalize_study_id(fu.study_id),
                                fu.dicom_id,
                                gap,
                            ]
                        )
        else:
            assert pairs is not None
            for pair in pairs:
                p_subject = _normalize_subject_id(pair.subject_id)

                pair_dir = args.out_dir / p_subject / f"pair_{pair.pair_index:04d}_d{pair.delta_days}"
                _ensure_dir(pair_dir)

                bl_rel = _build_rel_path(pair.subject_id, pair.baseline_study_id, pair.baseline_dicom_id, ext)
                fu_rel = _build_rel_path(pair.subject_id, pair.followup_study_id, pair.followup_dicom_id, ext)

                url_lines.append(_physionet_url(args.dataset, args.version, bl_rel))
                url_lines.append(_physionet_url(args.dataset, args.version, fu_rel))

                if args.mimic_files_root is not None:
                    bl_url = _add_query(_physionet_url(args.dataset, args.version, bl_rel), args.download_query)
                    fu_url = _add_query(_physionet_url(args.dataset, args.version, fu_rel), args.download_query)

                    bl_src = _download_or_find(bl_rel, url=bl_url)
                    fu_src = _download_or_find(fu_rel, url=fu_url)

                    if bl_src is None or fu_src is None:
                        missing_local += 1
                        continue

                    if not args.dry_run:
                        bl_dst = pair_dir / ("baseline" + ext)
                        fu_dst = pair_dir / ("followup" + ext)
                        _copy_file(bl_src, bl_dst, overwrite=args.overwrite)
                        _copy_file(fu_src, fu_dst, overwrite=args.overwrite)
                        copied += 1
    except KeyboardInterrupt:
        print("Interrupted by user. Partial results were saved.")
        print(
            f"Copied: {copied}. Missing local: {missing_local}. Downloaded: {downloaded}. Failed downloads: {failed_downloads}."
        )
        return 130

    if args.write_urls is not None:
        # De-duplicate URL lines while preserving order
        seen: set[str] = set()
        uniq = []
        for u in url_lines:
            if u not in seen:
                seen.add(u)
                uniq.append(u)
        args.write_urls.write_text("\n".join(uniq) + "\n", encoding="utf-8")
        print("Wrote URLs:", args.write_urls, f"({len(uniq)} files)")

    if args.mimic_files_root is None:
        if args.layout == "patient":
            n_pairs = 0 if patient_pairs is None else len(patient_pairs)
            n_scans = 0 if unique_scans is None else len(unique_scans)
            print(
                f"Selected {n_pairs} pairs with gap {args.min_gap_days}-{args.max_gap_days} days. "
                f"Unique scans referenced: {n_scans}. "
                f"(No local copy; provide --download and/or --mimic-files-root)."
            )
        else:
            print(
                f"Built {len(pairs)} pairs. (No local copy; provide --mimic-files-root to copy files.) "
                f"URLs can be written via --write-urls."
            )
    else:
        if args.layout == "patient":
            n_pairs = 0 if patient_pairs is None else len(patient_pairs)
            n_scans = 0 if unique_scans is None else len(unique_scans)
            print(
                f"Selected {n_pairs} pairs with gap {args.min_gap_days}-{args.max_gap_days} days. "
                f"Unique scans referenced: {n_scans}. "
                f"Copied {copied} files to {args.out_dir}. "
                f"Missing local files: {missing_local}. Downloaded files: {downloaded}. Failed downloads: {failed_downloads}."
            )
            if n_pairs == 0:
                print(
                    "No qualifying pairs found with current limits. "
                    "Try increasing --max-patients or using --max-pairs without --max-patients."
                )
        else:
            print(
                f"Built {len(pairs)} pairs. Copied {copied} pairs to {args.out_dir}. "
                f"Pairs missing local files: {missing_local}. Downloaded files: {downloaded}. Failed downloads: {failed_downloads}."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
