"""Small helpers shared by the package."""

from __future__ import annotations

import math
import re
import unicodedata

import numpy as np
import pandas as pd


def strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def is_blank(value) -> bool:
    """True for None, NaN, NaT and empty strings."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if np.ndim(result) == 0 else False


def norm_key(value) -> str:
    """Lowercase, accent-free key with single spaces; used to match names and aliases."""
    if is_blank(value):
        return ""
    text = strip_accents(str(value)).lower().strip()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_code(value) -> str:
    """Account or cost center code as text (610100.0 becomes '610100')."""
    if is_blank(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return str(int(value)) if float(value).is_integer() else str(value)
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".")[0]
    return text


def clean_text(value) -> str:
    return "" if is_blank(value) else str(value).strip()


def month_start(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year=int(year), month=int(month), day=1)


def shift_months(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    return pd.Timestamp(ts) + pd.DateOffset(months=n)


def month_label(ts) -> str:
    return pd.Timestamp(ts).strftime("%b-%y")


def to_number(series: pd.Series) -> pd.Series:
    """Convert a column to float, accepting '1,234', '$1,234' and '(1,234)'."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)
    text = series.astype("object").map(lambda v: "" if is_blank(v) else str(v).strip())
    negative = text.str.match(r"^\(.*\)$")
    text = text.str.replace(r"[\$,\s\(\)]", "", regex=True)
    numbers = pd.to_numeric(text.replace("", np.nan), errors="coerce").astype(float)
    numbers[negative.fillna(False).astype(bool)] *= -1
    return numbers
