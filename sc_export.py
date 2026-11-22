"""Excel outputs: forecast workbook and input template."""

from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd

import sc_config as C

PCT_KEYS = ("Accuracy", "WAPE", "Share", "Bias", "FVA", "_%", "CV")
DEC_KEYS = ("R2", "P_Value", "Rate_per_Activity", "Rate_to_USD", "Z", "Cost_per_Activity")
PLAIN_KEYS = ("Year", "Month", "Horizon", "Points", "Series", "N", "Atypical_Excluded")


def _col_format(col: str, series: pd.Series, fmts: dict):
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return None
    if any(k in col for k in PCT_KEYS):
        return fmts["pct"]
    if any(col == k or col.endswith("_" + k) for k in DEC_KEYS) or col in DEC_KEYS:
        return fmts["dec"]
    if col in PLAIN_KEYS or col.startswith("Points"):
        return fmts["int"]
    return fmts["money"]


def write_workbook(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="mmm-yy", date_format="mmm-yy") as writer:
        wb = writer.book
        fmts = {
            "header": wb.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": C.NAVY,
                                     "text_wrap": True, "valign": "vcenter"}),
            "money": wb.add_format({"num_format": "#,##0;[Red]-#,##0"}),
            "pct": wb.add_format({"num_format": "0.0%"}),
            "dec": wb.add_format({"num_format": "0.0000"}),
            "int": wb.add_format({"num_format": "0"}),
        }
        for name, df in sheets.items():
            sheet = name[:31]
            frame = df.copy() if df is not None else pd.DataFrame()
            if frame.empty:
                frame = pd.DataFrame({"Note": ["No data for this sheet."]})
            for col in frame.columns:
                if isinstance(frame[col].dtype, pd.DatetimeTZDtype):
                    frame[col] = frame[col].dt.tz_localize(None)
            frame.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            for j, col in enumerate(frame.columns):
                ws.write(0, j, str(col), fmts["header"])
                sample = frame[col].head(300).astype(str).str.len()
                width = int(max(len(str(col)), sample.max() if len(sample) else 8) * 1.1) + 2
                width = max(9, min(48, width))
                ws.set_column(j, j, width, _col_format(str(col), frame[col], fmts))
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, max(len(frame), 1), max(len(frame.columns) - 1, 0))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------
README_LINES = [
    "Thermal Plant Spend Forecast: input template (data center cooling equipment plants)",
    "",
    "Sheets: Spend (required); Drivers, Mapping, Maintenance_Events, FX and Adjustments (optional).",
    "Accepted files: .xlsx, .xlsm and .xlsb. Title rows above a table are skipped automatically.",
    "",
    "SPEND",
    "One row per plant, product line (optional), bucket (or GL account + cost center) and version.",
    "Months can be columns (Jan..Dec plus a Year column, or headers such as 2026-01) or rows (Year, Month, Value).",
    "Versions: Act (actuals), AOP and PPF (prior forecast). Other versions are ignored.",
    "Buckets: Direct Labor, Indirect Labor, Maintenance, Utilities, Consumables & Tooling, Scrap & Rework, Freight,",
    "Depreciation, Fees, Allocations, Other. Spanish names are accepted (Mano de obra directa, Retrabajo, Fletes, ...).",
    "GL rows: leave Bucket blank and fill GL_Account, Cost_Center and Description; the Mapping sheet assigns the bucket.",
    "If GL rows and bucket rows exist for the same plant, bucket, version and month, the GL rows are used.",
    "Product_Line is optional (for example Perimeter Cooling, Chillers, Liquid Cooling). Leave it blank for plant-level rows.",
    "History: 24 months or more of Act is recommended; seasonality is switched off with less than 24.",
    "",
    "DRIVERS",
    "Version Act for history and Plan for the remaining months (AOP drivers fill gaps in the plan).",
    "Build plan: Units and Std_Hours_per_Unit (from routings) by product line, or Earned_Hours directly.",
    "Earned hours = Units x Std_Hours_per_Unit. Without standards, units are used as the activity measure.",
    "Labor: Labor_Hours (actual direct labor hours), optional Labor_Efficiency plan (earned / actual hours), Wage_Rate.",
    "Utilities: Util_Units_<Type> and Util_Tariff_<Type>, for example Util_Units_Electricity (kWh) and",
    "Util_Tariff_Electricity ($/kWh). Plant-level meters are fine: product lines roll up to the plant.",
    "Plant-level spend uses plant-level drivers; when a driver exists only by product line, the lines are summed",
    "(rates are weighted by hours or consumption). Every active line needs a build plan for the plant total.",
    "",
    "MAPPING",
    "GL_Account and/or Cost_Center plus Bucket. Priority: GL + cost center, GL, cost center.",
    "Accounts not found are suggested by keyword and flagged for review.",
    "",
    "MAINTENANCE_EVENTS",
    "Planned PM, shutdowns and overhauls by asset and month (test stands, brazing furnaces, paint lines...).",
    "Include past events so they are removed from the unplanned base.",
    "",
    "FX",
    "Rate_to_USD = USD per 1 unit of local currency at the AOP rate. Year and Month are optional.",
    "",
    "ADJUSTMENTS",
    "Management adjustments in plant currency for open months, with owner and reason. They appear as a separate line.",
]


def template_frames() -> dict[str, pd.DataFrame]:
    months = C.MONTH_ABBR
    spend_rows = [
        {"Plant": "US01", "Product_Line": "", "Currency": "USD", "Version": "Act", "Year": 2025,
         "Bucket": "Direct Labor", "GL_Account": "", "Cost_Center": "", "Description": "",
         **{m: 520_000 + 4_000 * i for i, m in enumerate(months)}},
        {"Plant": "US01", "Product_Line": "", "Currency": "USD", "Version": "Act", "Year": 2026,
         "Bucket": "Direct Labor", "GL_Account": "", "Cost_Center": "", "Description": "",
         **{m: (545_000 + 3_000 * i if i < 6 else None) for i, m in enumerate(months)}},
        {"Plant": "US01", "Product_Line": "", "Currency": "USD", "Version": "AOP", "Year": 2026,
         "Bucket": "Direct Labor", "GL_Account": "", "Cost_Center": "", "Description": "",
         **{m: 540_000 for m in months}},
        {"Plant": "US02", "Product_Line": "", "Currency": "USD", "Version": "Act", "Year": 2026,
         "Bucket": "", "GL_Account": "510100", "Cost_Center": "US02-ASM", "Description": "Assembly Direct Labor",
         **{m: (330_000 if i < 6 else None) for i, m in enumerate(months)}},
        {"Plant": "MX01", "Product_Line": "Liquid Cooling", "Currency": "MXN", "Version": "PPF", "Year": 2026,
         "Bucket": "Scrap & Rework", "GL_Account": "", "Cost_Center": "", "Description": "",
         **{m: 650_000 for m in months}},
    ]
    drivers = pd.DataFrame([
        ["US01", "Chillers", "Units", "Act", 2026, 6, 26],
        ["US01", "Chillers", "Units", "Plan", 2026, 7, 27],
        ["US01", "Chillers", "Std_Hours_per_Unit", "Act", 2026, 6, 235],
        ["US01", "Chillers", "Labor_Hours", "Act", 2026, 6, 7_250],
        ["US01", "", "Wage_Rate", "Act", 2026, 6, 34.4],
        ["US01", "", "Wage_Rate", "Plan", 2026, 7, 34.4],
        ["US01", "", "Util_Units_Electricity", "Act", 2026, 6, 790_000],
        ["US01", "", "Util_Tariff_Electricity", "Act", 2026, 6, 0.098],
        ["US01", "", "Util_Tariff_Electricity", "Plan", 2026, 7, 0.0975],
        ["MX01", "Liquid Cooling", "Labor_Efficiency", "Plan", 2026, 7, 0.86],
    ], columns=["Plant", "Product_Line", "Driver", "Version", "Year", "Month", "Value"])
    mapping = pd.DataFrame([
        ["510100", "", "Direct Labor", "Assembly Direct Labor"],
        ["560100", "", "Scrap & Rework", "Scrap Material"],
        ["530100", "US02-MNT", "Maintenance", "Repair Parts (maintenance cost center)"],
    ], columns=["GL_Account", "Cost_Center", "Bucket", "Description"])
    events = pd.DataFrame([
        ["US01", "", "Test lab chiller loop", "Overhaul", 2026, 8, 220_000, "Maintenance plan 2026"],
        ["US01", "", "Fiber laser", "PM", 2026, 11, 90_000, "Work order 4500123"],
    ], columns=["Plant", "Product_Line", "Asset", "Type", "Year", "Month", "Amount", "Source"])
    fx = pd.DataFrame([["MXN", 2026, None, 0.055], ["EUR", 2026, None, 1.08]],
                      columns=["Currency", "Year", "Month", "Rate_to_USD"])
    adjustments = pd.DataFrame([
        ["US01", "", "Maintenance", 2026, 9, 60_000, "Plant Controller US01", "Test lab scope increase"],
    ], columns=["Plant", "Product_Line", "Bucket", "Year", "Month", "Amount", "Owner", "Reason"])
    return {
        "README": pd.DataFrame({"Instructions": README_LINES}),
        "Spend": pd.DataFrame(spend_rows),
        "Drivers": drivers,
        "Mapping": mapping,
        "Maintenance_Events": events,
        "FX": fx,
        "Adjustments": adjustments,
    }


def build_template() -> bytes:
    return write_workbook(template_frames())


def assumptions_frame(meta: dict) -> pd.DataFrame:
    rows = [{"Item": k, "Value": v} for k, v in meta.items()]
    rows.append({"Item": "Generated", "Value": datetime.now().strftime("%Y-%m-%d %H:%M")})
    return pd.DataFrame(rows)


def to_bool_text(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = np.where(out[col], "Yes", "No")
    return out
