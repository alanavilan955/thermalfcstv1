"""Read and validate the input workbook (.xlsx or .xlsb, wide or long sheets)."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

import sc_config as C
from sc_classify import map_gl_rows, normalize_bucket
from sc_utils import clean_code, clean_text, is_blank, norm_key, to_number

# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------
SHEET_ALIASES = {
    "Spend": ["spend", "gasto", "gastos", "data", "datos", "costs", "costos"],
    "Drivers": ["drivers", "driver", "indicadores"],
    "Mapping": ["mapping", "mapeo", "gl mapping", "account mapping"],
    "Maintenance_Events": ["maintenance events", "events", "eventos", "eventos mantenimiento",
                           "maintenance plan", "plan de mantenimiento"],
    "FX": ["fx", "fx rates", "exchange rates", "tipo de cambio", "tasas"],
    "Adjustments": ["adjustments", "ajustes", "management adjustments", "ajustes de direccion"],
}

COLUMN_ALIASES = {
    "Plant": ["plant", "planta", "plant code", "plant name", "site", "sitio"],
    "Product_Line": ["product line", "linea de producto", "linea", "line", "product family", "familia",
                     "family", "value stream", "mini plant", "miniplant", "mini planta", "miniplanta"],
    "Currency": ["currency", "moneda", "curr", "ccy"],
    "Version": ["version", "scenario", "escenario"],
    "Year": ["year", "ano", "anio", "fiscal year", "ejercicio"],
    "Month": ["month", "mes", "period", "periodo", "fiscal period"],
    "Date": ["date", "fecha"],
    "Bucket": ["bucket", "rubro", "spend bucket", "cost bucket", "cost category"],
    "GL_Account": ["gl account", "gl", "account", "cuenta", "cuenta gl", "cost element",
                   "elemento de costo", "gl acct"],
    "Cost_Center": ["cost center", "cost centre", "ceco", "centro de costo", "centro de costos", "cc"],
    "Description": ["description", "descripcion", "gl description", "account description", "text", "texto"],
    "Driver": ["driver", "indicador", "kpi", "metric", "metrica"],
    "Asset": ["asset", "activo", "equipment", "equipo"],
    "Type": ["type", "tipo", "event type"],
    "Owner": ["owner", "responsable", "dueno"],
    "Reason": ["reason", "motivo", "comment", "comentario"],
    "Source": ["source", "fuente"],
}
VALUE_ALIASES = ["value", "valor", "amount", "monto", "importe"]
RATE_ALIASES = ["rate to usd", "rate", "fx rate", "tasa", "tipo de cambio", "usd rate", "rate usd"]

ALIAS_TO_CANONICAL = {alias: canon for canon, aliases in COLUMN_ALIASES.items() for alias in aliases}
ALL_ALIAS_KEYS = set(ALIAS_TO_CANONICAL) | set(VALUE_ALIASES) | set(RATE_ALIASES)
CANONICAL_NAMES = set(COLUMN_ALIASES) | {"Value", "Amount", "Rate_to_USD"}

MONTH_NAMES = {
    "jan": 1, "january": 1, "ene": 1, "enero": 1,
    "feb": 2, "february": 2, "febrero": 2,
    "mar": 3, "march": 3, "marzo": 3,
    "apr": 4, "april": 4, "abr": 4, "abril": 4,
    "may": 5, "mayo": 5,
    "jun": 6, "june": 6, "junio": 6,
    "jul": 7, "july": 7, "julio": 7,
    "aug": 8, "august": 8, "ago": 8, "agosto": 8,
    "sep": 9, "sept": 9, "september": 9, "septiembre": 9, "set": 9, "setiembre": 9,
    "oct": 10, "october": 10, "octubre": 10,
    "nov": 11, "november": 11, "noviembre": 11,
    "dec": 12, "december": 12, "dic": 12, "diciembre": 12,
}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class Inputs:
    spend: pd.DataFrame
    gl_detail: pd.DataFrame
    mapping_review: pd.DataFrame
    drivers: pd.DataFrame
    events: pd.DataFrame
    fx: pd.DataFrame
    adjustments: pd.DataFrame
    plant_currency: dict
    issues: pd.DataFrame
    formats: dict = field(default_factory=dict)
    fatal: bool = False


class IssueLog:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, severity: str, sheet: str, check: str, message: str) -> None:
        self.rows.append({"Severity": severity, "Sheet": sheet, "Check": check, "Message": message})

    def frame(self) -> pd.DataFrame:
        order = {"Error": 0, "Warning": 1, "Info": 2}
        df = pd.DataFrame(self.rows, columns=["Severity", "Sheet", "Check", "Message"])
        if df.empty:
            return df
        return df.sort_values("Severity", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Period parsing
# ---------------------------------------------------------------------------
def parse_period_label(label) -> tuple[int | None, int | None]:
    """Return (year, month) from a header or cell; year is None when not present."""
    if is_blank(label):
        return None, None
    if isinstance(label, (pd.Timestamp, datetime, date)):
        ts = pd.Timestamp(label)
        return int(ts.year), int(ts.month)
    if isinstance(label, (bool, np.bool_)):
        return None, None
    if isinstance(label, (int, float, np.integer, np.floating)):
        v = float(label)
        if v.is_integer() and 1 <= v <= 12:
            return None, int(v)
        if 20000 < v < 80000:  # Excel serial date (for example from .xlsb files)
            ts = pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(v))
            return int(ts.year), int(ts.month)
        if v.is_integer() and 190001 <= v <= 210012 and 1 <= int(v) % 100 <= 12:
            return int(v) // 100, int(v) % 100
        return None, None
    raw = str(label).strip()
    key = norm_key(raw).replace(".", " ").replace("'", " ")
    key = re.sub(r"\s+", " ", key).strip()
    m = re.fullmatch(r"(\d{4})[ /](\d{1,2})(?:[ /]\d{1,2})?(?: \d{2}:\d{2}(?::\d{2})?)?", key)
    if m and 1 <= int(m.group(2)) <= 12:
        return int(m.group(1)), int(m.group(2))
    m = re.fullmatch(r"(\d{1,2})[ /](\d{4})", key)
    if m and 1 <= int(m.group(1)) <= 12:
        return int(m.group(2)), int(m.group(1))
    m = re.fullmatch(r"(?:p|m|per|period|periodo|mes|month)\s?0?(\d{1,2})", key)
    if m and 1 <= int(m.group(1)) <= 12:
        return None, int(m.group(1))
    m = re.fullmatch(r"([a-z]+)(?: ?(\d{2}|\d{4}))?", key)
    if m and m.group(1) in MONTH_NAMES:
        year = m.group(2)
        if year is not None:
            year = int(year) + (2000 if len(year) == 2 else 0)
        return year, MONTH_NAMES[m.group(1)]
    m = re.fullmatch(r"(\d{4}) ?([a-z]+)", key)
    if m and m.group(2) in MONTH_NAMES:
        return int(m.group(1)), MONTH_NAMES[m.group(2)]
    m = re.fullmatch(r"\d{1,2}", key)
    if m and 1 <= int(key) <= 12:
        return None, int(key)
    return None, None


def _is_period_header(label, numeric_ok: bool) -> bool:
    if isinstance(label, str) and (label in CANONICAL_NAMES or norm_key(label) in ALL_ALIAS_KEYS):
        return False
    if isinstance(label, (int, float, np.integer, np.floating)) and not numeric_ok:
        v = float(label)
        if v.is_integer() and 1 <= v <= 12:
            return False
    return parse_period_label(label)[1] is not None


# ---------------------------------------------------------------------------
# Sheet helpers
# ---------------------------------------------------------------------------
def _find_header_row(raw: pd.DataFrame, max_rows: int = 20) -> int:
    for i in range(min(max_rows, len(raw))):
        values = raw.iloc[i].tolist()
        hits = sum(1 for v in values if isinstance(v, str) and norm_key(v) in ALL_ALIAS_KEYS)
        periods = sum(1 for v in values if isinstance(v, (str, datetime, date, pd.Timestamp))
                      and not (isinstance(v, str) and norm_key(v) in ALL_ALIAS_KEYS)
                      and parse_period_label(v)[1] is not None)
        if hits >= 2 or (hits >= 1 and periods >= 3):
            return i
    return 0


def _frame_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    if raw.empty:
        return pd.DataFrame()
    raw = raw.reset_index(drop=True)
    header_idx = _find_header_row(raw)
    header = raw.iloc[header_idx].tolist()
    columns, seen = [], set()
    for j, h in enumerate(header):
        label = f"_col{j}" if is_blank(h) else (h.strip() if isinstance(h, str) else h)
        if isinstance(label, (pd.Timestamp, datetime, date)):
            label = pd.Timestamp(label)
        if label in seen:
            label = f"{label}_{j}"
        seen.add(label)
        columns.append(label)
    df = raw.iloc[header_idx + 1:].copy()
    df.columns = columns
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def _canonicalize(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    value_aliases = RATE_ALIASES if value_name == "Rate_to_USD" else VALUE_ALIASES
    rename, used = {}, set()
    for col in df.columns:
        if not isinstance(col, str):
            continue
        key = norm_key(col)
        canon = None
        if key in ALIAS_TO_CANONICAL:
            canon = ALIAS_TO_CANONICAL[key]
        elif key in value_aliases or key == norm_key(value_name):
            canon = value_name
        if canon and canon not in used:
            rename[col] = canon
            used.add(canon)
    return df.rename(columns=rename)


def _to_long(df: pd.DataFrame, value_name: str, sheet: str, log: IssueLog,
             require_month: bool = True) -> tuple[pd.DataFrame, str]:
    numeric_headers = [c for c in df.columns if isinstance(c, (int, float, np.integer, np.floating))
                       and float(c).is_integer() and 1 <= float(c) <= 12]
    numeric_ok = len(numeric_headers) >= 6
    period_cols = [c for c in df.columns if _is_period_header(c, numeric_ok)]
    if len(period_cols) >= 3:
        id_cols = [c for c in df.columns if c not in period_cols]
        long = df.melt(id_vars=id_cols, value_vars=period_cols, var_name="_period", value_name=value_name)
        parsed = [parse_period_label(v) for v in long["_period"]]
        fmt = "wide (months in columns)"
    else:
        source = "Month" if "Month" in df.columns else ("Date" if "Date" in df.columns else None)
        if source is None:
            if require_month:
                log.add("Error", sheet, "Structure",
                        "No month columns (Jan..Dec) and no Month or Date column were found.")
                return pd.DataFrame(), "unreadable"
            long = df.copy()
            long["Year"] = pd.to_numeric(long.get("Year"), errors="coerce") if "Year" in long else np.nan
            long["Month"] = np.nan
            if value_name in long:
                long[value_name] = to_number(long[value_name])
            return long, "long (no month)"
        if value_name not in df.columns:
            log.add("Error", sheet, "Structure", f"Long format needs a {value_name} column.")
            return pd.DataFrame(), "unreadable"
        long = df.copy()
        parsed = [parse_period_label(v) for v in long[source]]
        fmt = "long (one row per month)"

    years = pd.Series([p[0] for p in parsed], index=long.index, dtype="float")
    months = pd.Series([p[1] for p in parsed], index=long.index, dtype="float")
    if "Year" in long.columns:
        year_col = to_number(long["Year"])
        years = years.where(years.notna(), year_col)
    long["Year"] = years
    long["Month"] = months
    long[value_name] = to_number(long[value_name])
    long = long.dropna(subset=[value_name])
    bad = long["Year"].isna() | long["Month"].isna()
    if bad.any():
        log.add("Warning", sheet, "Periods",
                f"{int(bad.sum())} rows without a readable year or month were ignored.")
        long = long[~bad]
    long = long.drop(columns=["_period"], errors="ignore").copy()
    long["Year"] = long["Year"].astype(int)
    long["Month"] = long["Month"].astype(int)
    long["Date"] = pd.to_datetime({"year": long["Year"], "month": long["Month"], "day": 1})
    return long, fmt


def _text_col(df: pd.DataFrame, name: str, upper: bool = False) -> pd.Series:
    if name not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    values = df[name].map(clean_text)
    return values.str.upper() if upper else values


def _code_col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return df[name].map(clean_code)


def _find_sheet(sheets: dict, canonical: str):
    aliases = {norm_key(a) for a in SHEET_ALIASES[canonical]} | {norm_key(canonical)}
    for name in sheets:
        if norm_key(name) in aliases:
            return name
    return None


# ---------------------------------------------------------------------------
# Spend
# ---------------------------------------------------------------------------
SPEND_KEYS = ["Plant", "Product_Line", "Bucket", "Version", "Date"]


def _process_spend(raw: pd.DataFrame, mapping: pd.DataFrame | None, log: IssueLog):
    df = _canonicalize(_frame_from_raw(raw), "Value")
    empty = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, "unreadable")
    if df.empty:
        log.add("Error", "Spend", "Structure", "The Spend sheet is empty.")
        return empty
    missing = [c for c in ("Plant", "Version") if c not in df.columns]
    if missing:
        log.add("Error", "Spend", "Structure", f"Missing required column(s): {', '.join(missing)}.")
        return empty
    if "Bucket" not in df.columns and "GL_Account" not in df.columns:
        log.add("Error", "Spend", "Structure", "Spend needs a Bucket column, a GL_Account column, or both.")
        return empty
    long, fmt = _to_long(df, "Value", "Spend", log)
    if long.empty:
        return empty

    long["Plant"] = _text_col(long, "Plant")
    long["Product_Line"] = _text_col(long, "Product_Line")
    long["Currency"] = _text_col(long, "Currency", upper=True)
    long = long[long["Plant"] != ""].copy()

    versions = long["Version"].map(lambda v: C.VERSION_ALIASES.get(norm_key(v)))
    unknown = sorted({clean_text(v) for v, k in zip(long["Version"], versions) if k is None})
    if unknown:
        log.add("Info", "Spend", "Versions",
                f"Versions not used by the model were ignored: {', '.join(unknown[:8])}.")
    long["Version"] = versions
    long = long[long["Version"].notna()].copy()

    bucket_raw = _text_col(long, "Bucket")
    long["GL_Account"] = _code_col(long, "GL_Account")
    long["Cost_Center"] = _code_col(long, "Cost_Center")
    long["Description"] = _text_col(long, "Description")
    is_bucket = bucket_raw != ""
    is_gl = (~is_bucket) & (long["GL_Account"] != "")
    dropped = int((~is_bucket & ~is_gl).sum())
    if dropped:
        log.add("Warning", "Spend", "Structure", f"{dropped} rows without bucket or GL account were ignored.")

    bucket_rows = long[is_bucket].copy()
    bucket_rows["Bucket"] = bucket_raw[is_bucket].map(normalize_bucket)
    bucket_rows["Level"] = "Bucket"

    gl_rows = long[is_gl].copy()
    review = pd.DataFrame(columns=["GL_Account", "Cost_Center", "Description", "Bucket", "Mapping_Status"])
    if not gl_rows.empty:
        gl_rows, review = map_gl_rows(gl_rows, mapping)
        gl_rows["Level"] = "GL"
        suggested = int((review["Mapping_Status"] == "Suggested by keyword (review)").sum())
        unmapped = int((review["Mapping_Status"] == "Unmapped (sent to Other)").sum())
        if suggested:
            log.add("Warning", "Mapping", "Unmapped accounts",
                    f"{suggested} GL account(s) were not in the Mapping sheet; a bucket was suggested by keyword. "
                    "Review them in the Classification tab.")
        if unmapped:
            log.add("Warning", "Mapping", "Unmapped accounts",
                    f"{unmapped} GL account(s) could not be mapped and were sent to Other.")

    # Plant currency
    plant_currency = {}
    for plant, grp in long.groupby("Plant"):
        cur = grp["Currency"][grp["Currency"] != ""]
        if cur.empty:
            plant_currency[plant] = "USD"
            log.add("Warning", "Spend", "Currency", f"No currency for plant {plant}; USD was assumed.")
        else:
            counts = cur.value_counts()
            plant_currency[plant] = counts.index[0]
            if len(counts) > 1:
                log.add("Warning", "Spend", "Currency",
                        f"Plant {plant} has more than one currency ({', '.join(counts.index)}); "
                        f"{counts.index[0]} was used.")

    # Double counting between GL rows and bucket rows
    if not gl_rows.empty and not bucket_rows.empty:
        gl_sum = gl_rows.groupby(SPEND_KEYS)["Value"].sum()
        b_sum = bucket_rows.groupby(SPEND_KEYS)["Value"].sum()
        both = gl_sum.index.intersection(b_sum.index)
        if len(both):
            diff = (gl_sum.loc[both] - b_sum.loc[both]).reset_index(name="Diff")
            for (plant, mini, bucket, version), grp in diff.groupby(["Plant", "Product_Line", "Bucket", "Version"]):
                label = f"{plant}{' / ' + mini if mini else ''} / {bucket} ({version})"
                log.add("Warning", "Spend", "Double counting",
                        f"{label}: {len(grp)} month(s) have both GL rows and bucket rows. GL rows were used and "
                        f"bucket rows ignored. GL minus bucket difference: {grp['Diff'].sum():,.0f}.")
            keep = ~bucket_rows.set_index(SPEND_KEYS).index.isin(both)
            bucket_rows = bucket_rows[keep]

    cols = ["Plant", "Product_Line", "Bucket", "Version", "Date", "Value", "Level",
            "GL_Account", "Cost_Center", "Description"]
    combined = pd.concat([bucket_rows.reindex(columns=cols), gl_rows.reindex(columns=cols)], ignore_index=True)
    combined["Currency"] = combined["Plant"].map(plant_currency)

    dup = combined.duplicated(subset=["Plant", "Product_Line", "Bucket", "Version", "Date", "Level",
                                      "GL_Account", "Cost_Center"], keep=False)
    if dup.any():
        log.add("Info", "Spend", "Duplicates", f"{int(dup.sum())} duplicated rows were summed.")

    spend = (combined.groupby(["Plant", "Product_Line", "Currency", "Bucket", "Version", "Date"], as_index=False)
             ["Value"].sum())
    gl_detail = combined[combined["Level"] == "GL"].groupby(
        ["Plant", "Product_Line", "Currency", "Bucket", "GL_Account", "Cost_Center", "Version", "Date"],
        as_index=False).agg(Value=("Value", "sum"), Description=("Description", "first"))

    _integrity_checks(spend, log)
    return spend, gl_detail, review, plant_currency, fmt


def _integrity_checks(spend: pd.DataFrame, log: IssueLog) -> None:
    act = spend[spend["Version"] == C.ACT]
    zero_msgs, gap_msgs, neg_count = [], [], 0
    for (plant, mini, bucket), grp in act.groupby(["Plant", "Product_Line", "Bucket"]):
        s = grp.set_index("Date")["Value"].sort_index()
        label = f"{plant}{' / ' + mini if mini else ''} / {bucket}"
        if s.abs().median() > 0:
            zeros = s[s == 0]
            if len(zeros):
                zero_msgs.append(f"{label} ({', '.join(d.strftime('%b-%y') for d in zeros.index[:4])})")
        full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
        missing = full.difference(s.index)
        if len(missing):
            gap_msgs.append(f"{label} ({len(missing)} month(s))")
        neg_count += int((s < 0).sum())
    if zero_msgs:
        log.add("Warning", "Spend", "Zero actual months",
                "Months with zero actuals in active series: " + "; ".join(zero_msgs[:10])
                + (" ..." if len(zero_msgs) > 10 else ""))
    if gap_msgs:
        log.add("Warning", "Spend", "Missing months",
                "Months missing inside the actual history (filled by interpolation): " + "; ".join(gap_msgs[:10]))
    if neg_count:
        log.add("Info", "Spend", "Negative values",
                f"{neg_count} negative actual values found; they are flagged as possible reversals.")


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------
def normalize_driver_name(raw) -> str:
    key = norm_key(raw)
    if key in C.DRIVER_ALIASES:
        return C.DRIVER_ALIASES[key]
    m = re.match(r"^(?:util(?:ity|ities)? units?|utility consumption|consumption|consumo)\s*(.*)$", key)
    if m:
        rest = m.group(1).strip()
        return C.UTIL_UNITS + ("_" + rest.title().replace(" ", "_") if rest else "")
    m = re.match(r"^(?:util(?:ity|ities)? tariffs?|utility rate|tariff|tarifa)\s*(.*)$", key)
    if m:
        rest = m.group(1).strip()
        return C.UTIL_TARIFF + ("_" + rest.title().replace(" ", "_") if rest else "")
    return clean_text(raw)


def is_rate_driver(name: str) -> bool:
    return name in C.RATE_DRIVERS or str(name).startswith(C.UTIL_TARIFF)


def _process_drivers(raw: pd.DataFrame, log: IssueLog) -> tuple[pd.DataFrame, str]:
    cols = ["Plant", "Product_Line", "Driver", "Version", "Date", "Value"]
    df = _canonicalize(_frame_from_raw(raw), "Value")
    if df.empty:
        return pd.DataFrame(columns=cols), "empty"
    missing = [c for c in ("Plant", "Driver", "Version") if c not in df.columns]
    if missing:
        log.add("Error", "Drivers", "Structure",
                f"Missing required column(s): {', '.join(missing)}. Drivers were ignored.")
        return pd.DataFrame(columns=cols), "unreadable"
    long, fmt = _to_long(df, "Value", "Drivers", log)
    if long.empty:
        return pd.DataFrame(columns=cols), fmt
    long["Plant"] = _text_col(long, "Plant")
    long["Product_Line"] = _text_col(long, "Product_Line")
    long["Driver"] = long["Driver"].map(normalize_driver_name)
    long["Version"] = long["Version"].map(lambda v: C.DRIVER_VERSION_ALIASES.get(norm_key(v)))
    bad_version = long["Version"].isna()
    if bad_version.any():
        log.add("Warning", "Drivers", "Versions",
                f"{int(bad_version.sum())} driver rows with a version other than Act, Plan or AOP were ignored.")
    long = long[~bad_version & (long["Plant"] != "")]
    known = {C.UNITS, C.STD_HOURS, C.EARNED_HOURS, C.LABOR_HOURS, C.EFFICIENCY, C.WAGE}
    unknown = sorted({d for d in long["Driver"].unique()
                      if d not in known and not d.startswith(C.UTIL_UNITS) and not d.startswith(C.UTIL_TARIFF)})
    if unknown:
        log.add("Info", "Drivers", "Driver names",
                f"Drivers not used by the model: {', '.join(unknown[:10])}.")
    rate_mask = long["Driver"].map(is_rate_driver)
    agg_sum = long[~rate_mask].groupby(cols[:-1], as_index=False)["Value"].sum()
    agg_mean = long[rate_mask].groupby(cols[:-1], as_index=False)["Value"].mean()
    drivers = pd.concat([agg_sum, agg_mean], ignore_index=True)[cols]

    # Copied build plans: 3 or more consecutive identical months
    plan_vol = drivers[drivers["Driver"].isin([C.UNITS, C.EARNED_HOURS]) & (drivers["Version"] == C.DRIVER_PLAN)]
    for (plant, mini, driver), grp in plan_vol.groupby(["Plant", "Product_Line", "Driver"]):
        s = grp.sort_values("Date")["Value"].round(6).to_numpy()
        run, best = 1, 1
        for a, b in zip(s[:-1], s[1:]):
            run = run + 1 if (a == b and a != 0) else 1
            best = max(best, run)
        if best >= 3:
            label = f"{plant}{' / ' + mini if mini else ''}"
            log.add("Info", "Drivers", "Repeated plan values",
                    f"{label}: the {driver} plan repeats the same value in {best} consecutive months. "
                    "Confirm with S&OP that it is a level-loaded plan and not a copied one.")
    return drivers, fmt


# ---------------------------------------------------------------------------
# Events, FX and adjustments
# ---------------------------------------------------------------------------
def _process_events(raw: pd.DataFrame, log: IssueLog) -> tuple[pd.DataFrame, str]:
    cols = ["Plant", "Product_Line", "Asset", "Type", "Date", "Amount", "Source"]
    df = _canonicalize(_frame_from_raw(raw), "Amount")
    if df.empty:
        return pd.DataFrame(columns=cols), "empty"
    if "Plant" not in df.columns:
        log.add("Error", "Maintenance_Events", "Structure", "Missing Plant column. Events were ignored.")
        return pd.DataFrame(columns=cols), "unreadable"
    long, fmt = _to_long(df, "Amount", "Maintenance_Events", log)
    if long.empty:
        return pd.DataFrame(columns=cols), fmt
    out = pd.DataFrame({
        "Plant": _text_col(long, "Plant"), "Product_Line": _text_col(long, "Product_Line"),
        "Asset": _text_col(long, "Asset"), "Type": _text_col(long, "Type"),
        "Date": long["Date"], "Amount": long["Amount"], "Source": _text_col(long, "Source"),
    })
    return out[out["Plant"] != ""].reset_index(drop=True), fmt


def _process_fx(raw: pd.DataFrame, log: IssueLog) -> tuple[pd.DataFrame, str]:
    cols = ["Currency", "Year", "Month", "Rate_to_USD"]
    df = _canonicalize(_frame_from_raw(raw), "Rate_to_USD")
    if df.empty:
        return pd.DataFrame(columns=cols), "empty"
    if "Currency" not in df.columns:
        log.add("Error", "FX", "Structure", "Missing Currency column. FX rates were ignored.")
        return pd.DataFrame(columns=cols), "unreadable"
    numeric_headers = [c for c in df.columns if isinstance(c, (int, float, np.integer, np.floating))
                       and float(c).is_integer() and 1 <= float(c) <= 12]
    period_cols = [c for c in df.columns if _is_period_header(c, len(numeric_headers) >= 6)]
    if len(period_cols) >= 3:
        long, fmt = _to_long(df, "Rate_to_USD", "FX", log)
    else:
        long, fmt = df.copy(), "table (one rate per row; Year and Month optional)"
        if "Rate_to_USD" not in long.columns:
            log.add("Error", "FX", "Structure", "No Rate_to_USD column found. FX rates were ignored.")
            return pd.DataFrame(columns=cols), "unreadable"
        long["Year"] = to_number(long["Year"]) if "Year" in long.columns else np.nan
        long["Month"] = ([parse_period_label(v)[1] for v in long["Month"]] if "Month" in long.columns
                         else np.nan)
        long["Rate_to_USD"] = to_number(long["Rate_to_USD"])
    if long.empty:
        log.add("Error", "FX", "Structure", "No Rate_to_USD values found. FX rates were ignored.")
        return pd.DataFrame(columns=cols), "unreadable"
    out = pd.DataFrame({
        "Currency": _text_col(long, "Currency", upper=True),
        "Year": pd.to_numeric(long["Year"], errors="coerce") if "Year" in long else np.nan,
        "Month": pd.to_numeric(long["Month"], errors="coerce") if "Month" in long else np.nan,
        "Rate_to_USD": pd.to_numeric(long["Rate_to_USD"], errors="coerce"),
    })
    out = out[(out["Currency"] != "") & out["Rate_to_USD"].notna()]
    return out.reset_index(drop=True), fmt


def _process_adjustments(raw: pd.DataFrame, log: IssueLog) -> tuple[pd.DataFrame, str]:
    cols = ["Plant", "Product_Line", "Bucket", "Year", "Month", "Amount", "Owner", "Reason"]
    df = _canonicalize(_frame_from_raw(raw), "Amount")
    if df.empty:
        return pd.DataFrame(columns=cols), "empty"
    missing = [c for c in ("Plant", "Bucket") if c not in df.columns]
    if missing:
        log.add("Error", "Adjustments", "Structure",
                f"Missing required column(s): {', '.join(missing)}. Adjustments were ignored.")
        return pd.DataFrame(columns=cols), "unreadable"
    long, fmt = _to_long(df, "Amount", "Adjustments", log)
    if long.empty:
        return pd.DataFrame(columns=cols), fmt
    out = pd.DataFrame({
        "Plant": _text_col(long, "Plant"), "Product_Line": _text_col(long, "Product_Line"),
        "Bucket": long["Bucket"].map(normalize_bucket), "Year": long["Year"], "Month": long["Month"],
        "Amount": long["Amount"], "Owner": _text_col(long, "Owner"), "Reason": _text_col(long, "Reason"),
    })
    return out[out["Plant"] != ""].reset_index(drop=True), fmt


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def load_inputs(file_bytes: bytes, filename: str) -> Inputs:
    log = IssueLog()
    engine = "pyxlsb" if filename.lower().endswith(".xlsb") else "openpyxl"
    try:
        sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, header=None, engine=engine)
    except Exception as exc:  # noqa: BLE001 - shown to the user
        log.add("Error", "File", "Read", f"The file could not be read ({type(exc).__name__}: {exc}).")
        return _empty_inputs(log)

    names = {canon: _find_sheet(sheets, canon) for canon in SHEET_ALIASES}
    if names["Spend"] is None:
        if len(sheets) == 1:
            names["Spend"] = next(iter(sheets))
            log.add("Info", "File", "Sheets", f"Sheet '{names['Spend']}' was read as Spend.")
        else:
            log.add("Error", "File", "Sheets", "No Spend sheet was found (accepted names: Spend, Gasto, Data).")
            return _empty_inputs(log)

    formats = {}
    mapping = None
    if names["Mapping"] is not None:
        mapping = _canonicalize(_frame_from_raw(sheets[names["Mapping"]]), "Value")
        formats["Mapping"] = "table"
        if "Bucket" not in mapping.columns or ("GL_Account" not in mapping.columns
                                               and "Cost_Center" not in mapping.columns):
            log.add("Error", "Mapping", "Structure",
                    "Mapping needs Bucket plus GL_Account and/or Cost_Center columns. It was ignored.")
            mapping = None

    spend, gl_detail, review, plant_currency, formats["Spend"] = _process_spend(sheets[names["Spend"]], mapping, log)
    fatal = spend.empty or (spend["Version"] == C.ACT).sum() == 0
    if not spend.empty and (spend["Version"] == C.ACT).sum() == 0:
        log.add("Error", "Spend", "Versions", "No actual (Act) rows were found.")

    drivers, events, fx, adjustments = (pd.DataFrame(columns=["Plant", "Product_Line", "Driver", "Version", "Date", "Value"]),
                                        pd.DataFrame(columns=["Plant", "Product_Line", "Asset", "Type", "Date", "Amount", "Source"]),
                                        pd.DataFrame(columns=["Currency", "Year", "Month", "Rate_to_USD"]),
                                        pd.DataFrame(columns=["Plant", "Product_Line", "Bucket", "Year", "Month", "Amount", "Owner", "Reason"]))
    if names["Drivers"] is not None:
        drivers, formats["Drivers"] = _process_drivers(sheets[names["Drivers"]], log)
    else:
        log.add("Info", "Drivers", "Sheets", "No Drivers sheet; driver buckets fall back to Auto.")
    if names["Maintenance_Events"] is not None:
        events, formats["Maintenance_Events"] = _process_events(sheets[names["Maintenance_Events"]], log)
    if names["FX"] is not None:
        fx, formats["FX"] = _process_fx(sheets[names["FX"]], log)
    if names["Adjustments"] is not None:
        adjustments, formats["Adjustments"] = _process_adjustments(sheets[names["Adjustments"]], log)

    non_usd = sorted({c for c in plant_currency.values() if c != "USD"})
    known_fx = set(fx["Currency"]) if not fx.empty else set()
    missing_fx = [c for c in non_usd if c not in known_fx]
    if missing_fx:
        log.add("Warning", "FX", "Missing rates",
                f"No AOP rate to USD for: {', '.join(missing_fx)}. Those plants are shown in local currency only.")

    return Inputs(spend=spend, gl_detail=gl_detail, mapping_review=review, drivers=drivers, events=events,
                  fx=fx, adjustments=adjustments, plant_currency=plant_currency, issues=log.frame(),
                  formats=formats, fatal=bool(fatal))


def _empty_inputs(log: IssueLog) -> Inputs:
    return Inputs(spend=pd.DataFrame(), gl_detail=pd.DataFrame(), mapping_review=pd.DataFrame(),
                  drivers=pd.DataFrame(), events=pd.DataFrame(), fx=pd.DataFrame(),
                  adjustments=pd.DataFrame(), plant_currency={}, issues=log.frame(), formats={}, fatal=True)


def detect_actuals_through(spend: pd.DataFrame, year: int) -> int:
    """Last month of `year` where at least half of the active series have actuals."""
    act = spend[(spend["Version"] == C.ACT) & (spend["Date"].dt.year == year)]
    if act.empty:
        return 0
    counts = act[act["Value"] != 0].groupby(act["Date"].dt.month).size()
    if counts.empty:
        return 0
    threshold = 0.5 * counts.max()
    good = counts[counts >= threshold]
    return int(good.index.max())
