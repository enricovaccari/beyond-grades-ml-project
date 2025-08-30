# -*- coding: utf-8 -*-
#**********************************************************************************
# content       = cleaning.py functions
#
# version       = 1.0.0
# date          = 26-08-2025
#
# how to        = gets imported automatically from notebooks
# dependencies  = Python 3, typing, pandas, numpy, scipy
# todos         = -
#
# license       = MIT
# author        = Enrico Vaccari <e.vaccari99@gmail.com>
#
# © ALL RIGHTS RESERVED
#**********************************************************************************

# -------------------------------------------------------------------
# LIBRARIES (for cleaning.py)
# -------------------------------------------------------------------

import re
import os
import sys
import json
import importlib
import subprocess
from typing import Optional, Tuple, Dict, Any, List

from pathlib import Path
from datetime import datetime

required_packages = ["pandas", "numpy", "scipy"]

_loaded = {}
for _pkg in required_packages:
    try:
        _mod = importlib.import_module(_pkg)
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", _pkg])
            _mod = importlib.import_module(_pkg)
        except subprocess.CalledProcessError:
            _mod = None
    if _mod:
        _loaded[_pkg] = _mod

pd = _loaded.get("pandas")
np = _loaded.get("numpy")
scipy = _loaded.get("scipy")
if scipy is not None:
    from scipy import stats

# -------------------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------------------

def data_quality_check(
    df: "pd.DataFrame",
    study_time_col: str = "StudyTimeWeekly",
    gpa_col: str = "GPA",
    absences_col: str = "Absences",
    date_like_cols: Optional[List[str]] = None,
    drop_student_id: bool = True,
    drop_grade_class: bool = True,
    study_time_range: Tuple[float, float] = (0.0, 100.0),
    remove_study_time_out_of_range: bool = False
) -> Tuple["pd.DataFrame", Dict[str, Any]]:
    """
    Run quick, opinionated sanity checks + lightweight fixes for key student columns.
    Always returns a full report dict (with 'ok' where no issues are found) and a
    *copy* of df with only minimal safe mutations (type coercions, optional drops).

    Parameters
    ----------
    df : pd.DataFrame
    study_time_col : str
    gpa_col : str
    absences_col : str
    date_like_cols : list[str] | None
        Extra columns to attempt parsing as datetimes (in addition to name heuristics).
    drop_student_id : bool
        If True, drop "StudentID" when present (non-predictive identifier).
    drop_grade_class : bool
        If True, drop "grade_class"/"GradeClass" if present (often leakage-ish or redundant).
    study_time_range : (float, float)
        Logical bounds in hours/week.
    remove_study_time_out_of_range : bool
        If True, drop rows where study time is outside range.

    Returns
    -------
    (clean_df, report)
    """
    report: Dict[str, Any] = {}
    out = df.copy()

    # ---- structural drops ----
    dropped = []
    if drop_student_id and "StudentID" in out.columns:
        out.drop(columns=["StudentID"], inplace=True, errors="ignore")
        dropped.append("StudentID")
    if drop_grade_class:
        for gc in ("grade_class", "GradeClass"):
            if gc in out.columns:
                out.drop(columns=[gc], inplace=True, errors="ignore")
                dropped.append(gc)
    report["dropped_columns"] = dropped if dropped else "ok"

    # ---- study time ----
    study_report: Dict[str, Any] = {}
    if study_time_col in out.columns:
        raw_na = out[study_time_col].isna().sum()
        st = pd.to_numeric(out[study_time_col], errors="coerce")
        coerced_to_nan = int(max(st.isna().sum() - raw_na, 0))
        study_report["non_numeric_coerced"] = coerced_to_nan

        if st.notna().any():
            study_report["min"] = float(st.min())
            study_report["max"] = float(st.max())
        else:
            study_report["min"], study_report["max"] = None, None

        lo, hi = study_time_range
        oor = (st < lo) | (st > hi)
        study_report["out_of_range"] = {"count": int(oor.sum()), "range_used": [lo, hi]}
        out[study_time_col] = st

        if remove_study_time_out_of_range and oor.any():
            out = out.loc[~oor].copy()
            study_report["rows_removed"] = int(oor.sum())
    else:
        study_report["status"] = "missing"
    report["study_time"] = study_report

    # ---- GPA ----
    gpa_report: Dict[str, Any] = {}
    if gpa_col in out.columns:
        raw_na = out[gpa_col].isna().sum()
        gpa = pd.to_numeric(out[gpa_col], errors="coerce")
        gpa_report["non_numeric_coerced"] = int(max(gpa.isna().sum() - raw_na, 0))
        if gpa.notna().any():
            gpa_report["min"] = float(gpa.min())
            gpa_report["max"] = float(gpa.max())
        else:
            gpa_report["min"], gpa_report["max"] = None, None
        out[gpa_col] = gpa
    else:
        gpa_report["status"] = "missing"
    report["gpa"] = gpa_report

    # ---- absences ----
    abs_report: Dict[str, Any] = {}
    if absences_col in out.columns:
        raw_na = out[absences_col].isna().sum()
        ab = pd.to_numeric(out[absences_col], errors="coerce")
        abs_report["non_numeric_coerced"] = int(max(ab.isna().sum() - raw_na, 0))
        if ab.notna().any():
            abs_report["min"] = float(ab.min())
            abs_report["max"] = float(ab.max())
            abs_report["negative_count"] = int((ab < 0).sum())

            v = ab.dropna()
            if len(v) >= 5:
                q1, q3 = np.percentile(v, [25, 75])
                iqr = q3 - q1
                upper_fence = q3 + 1.5 * iqr
                abs_report["upper_outliers"] = int((ab > upper_fence).sum())
                abs_report["upper_fence"] = float(upper_fence)
        else:
            abs_report["min"], abs_report["max"] = None, None
        out[absences_col] = ab
    else:
        abs_report["status"] = "missing"
    report["absences"] = abs_report

    # ---- date parsing (optional) ----
    candidates: List[str] = []
    if date_like_cols:
        candidates.extend([c for c in date_like_cols if c in out.columns])
    # heuristic add
    for c in out.columns:
        n = str(c).lower()
        if ("date" in n or "time" in n) and c not in candidates:
            candidates.append(c)

    date_report: Dict[str, Any] = {}
    if candidates:
        now_ts = pd.Timestamp(datetime.now())
        for c in candidates:
            parsed = pd.to_datetime(out[c], errors="coerce", utc=False, infer_datetime_format=True)
            parse_rate = float(parsed.notna().mean()) if len(out) else 0.0
            future_cnt = int((parsed > now_ts).sum())
            date_report[c] = {
                "parse_success_rate": round(parse_rate, 3),
                "future_dates": future_cnt
            }
    report["date_columns"] = date_report if date_report else "ok"

    # ---- categorical inconsistencies (case/whitespace) ----
    cat_issues: Dict[str, Any] = {}
    for c in out.select_dtypes(include=["object", "category"]).columns:
        s = out[c].dropna().astype(str)
        if s.empty:
            continue
        raw_u = int(s.nunique())
        norm_u = int(s.str.strip().str.lower().nunique())
        if norm_u < raw_u:
            cat_issues[c] = {
                "raw_unique": raw_u,
                "normalized_unique": norm_u,
                "note": "Likely case/whitespace inconsistencies."
            }
    report["categorical_format_inconsistencies"] = cat_issues if cat_issues else "ok"

    # ---- encoding issues (non-ascii / control chars) ----
    enc_issues: Dict[str, Any] = {}
    control_re = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
    for c in out.select_dtypes(include=["object", "category"]).columns:
        s = out[c].dropna().astype(str)
        if s.empty:
            continue
        non_ascii = int(s.map(lambda x: not all(ord(ch) < 128 for ch in x)).sum())
        control = int(s.map(lambda x: bool(control_re.search(x))).sum())
        if non_ascii or control:
            enc_issues[c] = {
                "non_ascii_count": non_ascii,
                "control_char_count": control
            }
    report["encoding_issues"] = enc_issues if enc_issues else "ok"

    return out, report


def print_quality_report(report: Dict[str, Any]) -> None:
    """Pretty-print the nested report produced by data_quality_check()."""
    def _rec(d, indent=0):
        pad = "  " * indent
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{pad}- {k}:")
                _rec(v, indent + 1)
            else:
                print(f"{pad}- {k}: {v}")
    print("=== Data Quality Report ===")
    _rec(report)

# -------------------------------------------------------------------

