"""Classification: bucket mapping, cost behavior, forecastability and atypical items."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

import sc_config as C
from sc_utils import clean_code, clean_text, is_blank, norm_key, strip_accents


# ---------------------------------------------------------------------------
# Bucket names and GL mapping
# ---------------------------------------------------------------------------
def normalize_bucket(name) -> str:
    """Return the canonical bucket name; unknown names are kept (title case)."""
    if is_blank(name):
        return ""
    key = norm_key(name)
    if key in C.BUCKET_SYNONYMS:
        return C.BUCKET_SYNONYMS[key]
    for bucket in C.STANDARD_BUCKETS:
        if key == norm_key(bucket):
            return bucket
    return str(name).strip()


def suggest_bucket(text: str) -> str | None:
    """Suggest a bucket from an account description using keyword rules (matched at word starts)."""
    probe = strip_accents(str(text)).lower()
    for bucket, keywords in C.KEYWORD_RULES:
        if any(re.search(r"(?<![a-z0-9])" + k, probe) for k in keywords):
            return bucket
    return None


def map_gl_rows(gl_rows: pd.DataFrame, mapping: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign a bucket to GL rows. Priority: GL + cost center, GL, cost center, keyword, Other."""
    gl_rows = gl_rows.copy()
    by_pair: dict[tuple[str, str], str] = {}
    by_gl: dict[str, str] = {}
    by_cc: dict[str, str] = {}
    if mapping is not None and not mapping.empty:
        for _, row in mapping.iterrows():
            gl = clean_code(row.get("GL_Account"))
            cc = clean_code(row.get("Cost_Center"))
            bucket = normalize_bucket(row.get("Bucket"))
            if not bucket:
                continue
            if gl and cc:
                by_pair[(gl, cc)] = bucket
            elif gl:
                by_gl[gl] = bucket
            elif cc:
                by_cc[cc] = bucket

    combos = (
        gl_rows.groupby(["GL_Account", "Cost_Center"], dropna=False)["Description"]
        .first()
        .reset_index()
    )
    records = []
    for _, row in combos.iterrows():
        gl, cc, desc = row["GL_Account"], row["Cost_Center"], clean_text(row["Description"])
        if (gl, cc) in by_pair:
            bucket, status = by_pair[(gl, cc)], "Mapping sheet (GL + cost center)"
        elif gl in by_gl:
            bucket, status = by_gl[gl], "Mapping sheet (GL)"
        elif cc and cc in by_cc:
            bucket, status = by_cc[cc], "Mapping sheet (cost center)"
        else:
            suggestion = suggest_bucket(f"{gl} {desc}")
            if suggestion:
                bucket, status = suggestion, "Suggested by keyword (review)"
            else:
                bucket, status = C.OTHER, "Unmapped (sent to Other)"
        records.append({"GL_Account": gl, "Cost_Center": cc, "Description": desc,
                        "Bucket": bucket, "Mapping_Status": status})
    review = pd.DataFrame(records, columns=["GL_Account", "Cost_Center", "Description", "Bucket", "Mapping_Status"])
    gl_rows = gl_rows.drop(columns=["Bucket"], errors="ignore").merge(
        review[["GL_Account", "Cost_Center", "Bucket", "Mapping_Status"]],
        on=["GL_Account", "Cost_Center"], how="left",
    )
    return gl_rows, review


# ---------------------------------------------------------------------------
# Atypical items (outliers)
# ---------------------------------------------------------------------------
def detect_outliers(y: pd.Series, z_threshold: float = C.OUTLIER_Z, min_dev: float = C.OUTLIER_MIN_DEV,
                    window: int = C.OUTLIER_WINDOW) -> pd.DataFrame:
    """Flag spikes, dips and reversals.

    A point is atypical when the modified z-score (Iglewicz and Hoaglin) of its residual against a
    centered rolling median is above `z_threshold` and the residual is at least `min_dev` of the
    baseline. A point is not flagged when the same calendar month of another year shows the same
    deviation (a recurring pattern such as a shutdown month or a year-end peak is seasonality).
    Negative values in a series whose median is positive are always flagged.
    """
    s = pd.Series(y, dtype=float)
    out = pd.DataFrame({"Value": s})
    if s.dropna().empty:
        out["Baseline"] = np.nan
        out["Z"] = np.nan
        out["Flag"] = False
        out["Reason"] = ""
        return out
    baseline = s.rolling(window, center=True, min_periods=3).median()
    baseline = baseline.fillna(s.median())
    resid = s - baseline
    med = resid.median()
    mad = (resid - med).abs().median()
    if mad > 0:
        z = 0.6745 * (resid - med) / mad
    else:
        mean_ad = (resid - resid.mean()).abs().mean()
        z = (resid - med) / (1.253314 * mean_ad) if mean_ad > 0 else resid * 0.0
    z = z.fillna(0.0)
    material = resid.abs() >= min_dev * baseline.abs()
    spike = np.array(((z.abs() > z_threshold) & material).to_numpy(), dtype=bool, copy=True)
    ratio = (s / baseline.replace(0, np.nan)).to_numpy(float)
    for i in np.flatnonzero(spike):
        for k in (-24, -12, 12, 24):
            j = i + k
            if 0 <= j < len(s) and np.isfinite(ratio[j]) and np.isfinite(ratio[i]):
                same_size = abs(ratio[j] - ratio[i]) <= min_dev
                same_side = np.sign(ratio[j] - 1) == np.sign(ratio[i] - 1)
                if same_size and same_side and abs(ratio[j] - 1) >= min_dev / 2:
                    spike[i] = False
                    break
    spike = pd.Series(spike, index=s.index)
    negative = (s < 0) & (s.median() > 0)
    reason = np.where(negative, "Negative value (possible reversal)",
                      np.where(spike & (z > 0), "Spike", np.where(spike & (z < 0), "Dip", "")))
    out["Baseline"] = baseline
    out["Z"] = z
    out["Flag"] = (spike | negative).astype(bool)
    out["Reason"] = reason
    return out


def apply_overrides(flags: pd.DataFrame, overrides: dict | None) -> pd.DataFrame:
    """Overrides map a Timestamp to True (exclude) or False (keep)."""
    if not overrides:
        flags["Excluded"] = flags["Flag"]
        return flags
    excluded = flags["Flag"].copy()
    for ts, value in overrides.items():
        if ts in excluded.index:
            excluded.loc[ts] = bool(value)
    flags["Excluded"] = excluded.astype(bool)
    return flags


def clean_series(y: pd.Series, z_threshold: float, overrides: dict | None = None,
                 min_dev: float = C.OUTLIER_MIN_DEV) -> tuple[pd.Series, pd.DataFrame]:
    """Replace excluded points by the rolling-median baseline."""
    flags = apply_overrides(detect_outliers(y, z_threshold, min_dev), overrides)
    cleaned = y.astype(float).where(~flags["Excluded"], flags["Baseline"])
    return cleaned, flags


# ---------------------------------------------------------------------------
# Cost behavior: fixed vs variable
# ---------------------------------------------------------------------------
def cost_behavior(y_clean: pd.Series, volume: pd.Series | None, months: int = C.CLASSIFY_MONTHS) -> dict:
    empty = {"Behavior": "No activity data", "Variable_Share": np.nan, "Fixed_per_Month": np.nan,
             "Rate_per_Activity": np.nan, "R2": np.nan, "P_Value": np.nan, "N": 0}
    if volume is None or volume.dropna().empty:
        return empty
    df = pd.concat({"y": y_clean, "v": volume}, axis=1).dropna()
    df = df.iloc[-months:]
    n = len(df)
    if n < 12 or df["v"].std() == 0:
        return {**empty, "Behavior": "Insufficient data", "N": n}
    x = df["v"].to_numpy(float)
    yy = df["y"].to_numpy(float)
    x_mean, y_mean = x.mean(), yy.mean()
    sxx = ((x - x_mean) ** 2).sum()
    slope = ((x - x_mean) * (yy - y_mean)).sum() / sxx
    intercept = y_mean - slope * x_mean
    fitted = intercept + slope * x
    resid = yy - fitted
    sse = (resid ** 2).sum()
    sst = ((yy - y_mean) ** 2).sum()
    r2 = 1 - sse / sst if sst > 0 else np.nan
    if n > 2 and sse > 0:
        se = np.sqrt(sse / (n - 2) / sxx)
        t_stat = slope / se
        from scipy import stats

        p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 2))
    else:
        p_value = 0.0 if sse == 0 and slope != 0 else 1.0
    var_share = slope * x_mean / y_mean if y_mean != 0 else np.nan
    if slope <= 0 or p_value >= C.P_VALUE_MAX or not np.isfinite(var_share) or var_share < C.VAR_SHARE_FIXED:
        behavior = "Fixed"
    elif var_share >= C.VAR_SHARE_VARIABLE:
        behavior = "Variable"
    else:
        behavior = "Semi-variable"
    return {"Behavior": behavior, "Variable_Share": float(var_share), "Fixed_per_Month": float(intercept),
            "Rate_per_Activity": float(slope), "R2": float(r2) if np.isfinite(r2) else np.nan,
            "P_Value": float(p_value), "N": n}


# ---------------------------------------------------------------------------
# Forecastability
# ---------------------------------------------------------------------------
def forecastability(y_clean: pd.Series, months: int = C.CLASSIFY_MONTHS) -> dict:
    s = y_clean.dropna().iloc[-months:]
    if len(s) < 3:
        return {"Segment": "Insufficient data", "CV": np.nan, "Naive_WAPE_1m": np.nan, "Zero_Share": np.nan}
    mean = s.mean()
    # Volatility around a linear trend, so steady growth (a product ramp) does not count as noise.
    x = np.arange(len(s), dtype=float)
    slope, intercept = np.polyfit(x, s.to_numpy(float), 1) if len(s) >= 3 else (0.0, float(mean))
    resid = s.to_numpy(float) - (intercept + slope * x)
    cv = float(np.std(resid, ddof=1) / abs(mean)) if mean != 0 else np.nan
    zero_share = float((s.abs() < 1e-9).mean())
    last = s.iloc[-13:]
    denom = last.iloc[1:].abs().sum()
    naive_wape = float(last.diff().abs().iloc[1:].sum() / denom) if denom > 0 else np.nan
    if zero_share > C.INTERMITTENT_ZERO_SHARE:
        segment = "Intermittent"
    elif not np.isfinite(cv):
        segment = "Erratic"
    elif cv < C.CV_STABLE:
        segment = "Stable"
    elif cv <= C.CV_VARIABLE:
        segment = "Variable"
    else:
        segment = "Erratic"
    return {"Segment": segment, "CV": float(cv) if np.isfinite(cv) else np.nan,
            "Naive_WAPE_1m": naive_wape, "Zero_Share": zero_share}
