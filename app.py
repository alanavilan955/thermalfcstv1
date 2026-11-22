"""Thermal Plant Spend Forecast: Streamlit app for data center cooling equipment plants.

Run locally with:  streamlit run app.py

Every module sits next to this file (flat layout), so the app runs when all files are uploaded to
the root of the GitHub repository, even with an upload method that does not keep folders.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Thermal Plant Spend Forecast", layout="wide")

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

MODULE_FILES = ["sc_charts.py", "sc_classify.py", "sc_config.py", "sc_context.py", "sc_driverprep.py",
                "sc_drivers.py", "sc_engine.py", "sc_export.py", "sc_fx.py", "sc_io_excel.py", "sc_metrics.py",
                "sc_models.py", "sc_sample.py", "sc_scenario.py", "sc_utils.py"]
PACKAGES = {"numpy": "numpy", "pandas": "pandas", "plotly": "plotly", "statsmodels": "statsmodels",
            "scipy": "scipy", "openpyxl": "openpyxl", "pyxlsb": "pyxlsb", "xlsxwriter": "XlsxWriter"}


def _setup_problems() -> list[str]:
    """Missing module files or packages, described in plain words for the deployment owner."""
    problems = []
    missing_files = [name for name in MODULE_FILES if not (APP_DIR / name).is_file()]
    if missing_files:
        problems.append("Files missing next to app.py (upload them to the same folder as app.py, the repository "
                        "root): " + ", ".join(missing_files))
    missing_pkgs = [pip_name for module, pip_name in PACKAGES.items() if importlib.util.find_spec(module) is None]
    if missing_pkgs:
        problems.append("Packages not installed: " + ", ".join(missing_pkgs) + ". Check that requirements.txt "
                        "is in the repository root next to app.py, then reboot the app.")
    return problems


_problems = _setup_problems()
if _problems:
    st.error("The app cannot start because the deployment is incomplete.")
    for _text in _problems:
        st.markdown(f"- {_text}")
    st.caption(f"App folder on the server: {APP_DIR}")
    st.stop()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sc_charts as charts  # noqa: E402
import sc_config as C  # noqa: E402
from sc_engine import Settings, run_models  # noqa: E402
from sc_export import assumptions_frame, build_template, to_bool_text, write_workbook  # noqa: E402
from sc_fx import FxTable  # noqa: E402
from sc_io_excel import detect_actuals_through, load_inputs  # noqa: E402
from sc_sample import make_sample_workbook  # noqa: E402
from sc_scenario import (Scenario, add_activity, allocate_gl, bridge, build_table, clean_adjustments,  # noqa: E402
                         monthly_cost_per_activity, summarize_fy)

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
ADJ_COLUMNS = ["Plant", "Product_Line", "Bucket", "Year", "Month", "Amount", "Owner", "Reason"]


def active_palette() -> dict:
    """Chart palette that follows the active Streamlit theme (dark by default).

    On the first run of a browser session, st.context.theme reports the browser's light or dark
    preference instead of the theme set in .streamlit/config.toml, so the configured base is used on
    that run. Later runs follow st.context.theme, which also reflects a theme chosen in Settings.
    """
    configured = st.get_option("theme.base")
    try:
        kind = st.context.theme.type
    except Exception:  # noqa: BLE001 - older Streamlit versions
        kind = None
    first_run = "_theme_checked" not in st.session_state
    st.session_state["_theme_checked"] = True
    if configured in ("dark", "light") and (first_run or kind not in ("dark", "light")):
        kind = configured
    if kind not in ("dark", "light"):
        kind = "dark"
    return C.PALETTES[kind]


PAL = active_palette()


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def sample_bytes() -> bytes:
    return make_sample_workbook()


@st.cache_data(show_spinner=False)
def template_bytes() -> bytes:
    return build_template()


@st.cache_data(show_spinner=False, max_entries=6)
def cached_inputs(file_bytes: bytes, filename: str):
    return load_inputs(file_bytes, filename)


@st.cache_data(show_spinner=False, max_entries=6)
def cached_results(file_bytes: bytes, filename: str, settings_key: tuple):
    year, at, points, z, min_dev, target, method_over, outlier_over = settings_key
    settings = Settings(year=year, actuals_through=at, backtest_points=points, outlier_z=z,
                        outlier_min_dev=min_dev, accuracy_target=target, method_overrides=method_over,
                        outlier_overrides=outlier_over)
    return run_models(cached_inputs(file_bytes, filename), settings)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def money(v) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:,.0f}"


def pct(v, signed: bool = False) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:+.1%}" if signed else f"{v:.1%}"


def money_config(cols) -> dict:
    return {c: st.column_config.NumberColumn(c.replace("_", " "), format="localized") for c in cols}


def pct_config(cols) -> dict:
    return {c: st.column_config.NumberColumn(c.replace("_", " "), format="percent") for c in cols}


def round_money(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(0)
    return out


def series_label(plant: str, mini: str) -> str:  # mini = product line
    return f"{plant} / {mini}" if mini else plant


def bucket_order(b: str) -> int:
    return C.STANDARD_BUCKETS.index(b) if b in C.STANDARD_BUCKETS else 99


def editor_frame(base: pd.DataFrame, key: str) -> pd.DataFrame:
    """Rebuild the edited table from the data_editor state, before the editor is drawn."""
    state = st.session_state.get(key)
    df = base.copy().reset_index(drop=True)
    if not state:
        return df
    for idx, changes in state.get("edited_rows", {}).items():
        for col, val in changes.items():
            if col in df.columns:
                df.loc[int(idx), col] = val
    deleted = [i for i in state.get("deleted_rows", []) if i in df.index]
    if deleted:
        df = df.drop(index=deleted)
    added = [r for r in state.get("added_rows", []) if r]
    if added:
        df = pd.concat([df, pd.DataFrame(added)], ignore_index=True)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sidebar: data
# ---------------------------------------------------------------------------
st.sidebar.title("Thermal Plant Spend Forecast")
st.sidebar.caption("Data center cooling equipment: driver-based and ETS/ARIMA year-end landing · "
                   f"version {C.APP_VERSION}")

st.sidebar.subheader("1. Data")
uploaded = st.sidebar.file_uploader("Upload workbook (.xlsx, .xlsm, .xlsb)", type=["xlsx", "xlsm", "xlsb"])
if uploaded is not None:
    file_bytes, filename = uploaded.getvalue(), uploaded.name
    st.sidebar.caption(f"Using: {uploaded.name}")
else:
    file_bytes, filename = sample_bytes(), "sample_plant_spend.xlsx"
    st.sidebar.info("Showing synthetic sample data. Upload your workbook to replace it.")
st.sidebar.download_button("Download input template", template_bytes(), file_name="plant_spend_template.xlsx",
                           mime=XLSX_MIME, width="stretch")
if uploaded is None:
    st.sidebar.download_button("Download sample data", file_bytes, file_name="sample_plant_spend.xlsx",
                               mime=XLSX_MIME, width="stretch")

signature = hashlib.md5(file_bytes).hexdigest()
if st.session_state.get("file_sig") != signature:
    st.session_state["file_sig"] = signature
    st.session_state["outlier_overrides"] = {}
    st.session_state.pop("adj_base", None)
    st.session_state.pop("adj_editor", None)
    st.session_state.pop("workbook", None)

inputs = cached_inputs(file_bytes, filename)
if inputs.fatal:
    st.title("Thermal Plant Spend Forecast")
    st.error("The workbook could not be used. Review the checks below, fix the file and upload it again.")
    st.dataframe(inputs.issues, width="stretch", hide_index=True)
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: period, model, scenario, view
# ---------------------------------------------------------------------------
st.sidebar.subheader("2. Period")
act_years = sorted(int(y) for y in inputs.spend.loc[inputs.spend["Version"] == C.ACT, "Date"].dt.year.unique())
year = st.sidebar.selectbox("Forecast year", act_years, index=len(act_years) - 1)
detected = detect_actuals_through(inputs.spend, year) or 1
at = st.sidebar.selectbox("Actuals through", list(range(1, 13)), index=min(max(detected, 1), 12) - 1,
                          format_func=lambda m: f"{C.MONTH_ABBR[m - 1]}-{str(year)[2:]}",
                          help=f"Detected from the file: {C.MONTH_ABBR[detected - 1]}-{str(year)[2:]}. "
                               "Choose an earlier month to test how the model would have landed.")
if at == 12:
    st.sidebar.warning("All 12 months are closed; there is nothing left to forecast.")

with st.sidebar.expander("3. Model settings"):
    bt_points = st.slider("Backtest points (closed months)", 3, 12, C.BACKTEST_POINTS)
    outlier_z = st.slider("Atypical item threshold (modified z)", 2.5, 6.0, C.OUTLIER_Z, 0.5)
    min_dev_pct = st.slider("Atypical item minimum deviation (%)", 5, 50, int(C.OUTLIER_MIN_DEV * 100), 5,
                            help="A point is atypical only if it also deviates this much from its rolling median.")
    target_pct = st.slider("Accuracy target (%)", 80, 99, int(C.ACCURACY_TARGET * 100))
    st.caption("Method by bucket")
    buckets_present = sorted(inputs.spend["Bucket"].unique(), key=bucket_order)
    method_overrides = []
    for b in buckets_present:
        default = C.DEFAULT_METHOD.get(b, C.M_AUTO)
        choice = st.selectbox(f"{b} (default: {default})", C.METHOD_CHOICES, index=0, key=f"method_{b}")
        if choice != C.M_DEFAULT:
            method_overrides.append((b, choice))

SCENARIO_KEYS = ("sc_volume", "sc_efficiency", "sc_wage", "sc_tariff")


def reset_scenario() -> None:
    for k in SCENARIO_KEYS:
        st.session_state[k] = 0


with st.sidebar.expander("4. Scenario"):
    sc_volume = st.slider("Build plan (units / earned hours) %", -20, 20, 0, key="sc_volume")
    sc_prod = st.slider("Labor efficiency (earned / actual hours) %", -15, 15, 0, key="sc_efficiency")
    sc_wage = st.slider("Wage rate %", -10, 10, 0, key="sc_wage")
    sc_tariff = st.slider("Energy tariff %", -30, 30, 0, key="sc_tariff")
    st.caption("Applies to Direct Labor, Utilities and activity-regression buckets (consumables, scrap, freight "
               "when they behave as variable).")
    st.button("Reset scenario", on_click=reset_scenario)
scenario = Scenario(sc_volume / 100, sc_prod / 100, sc_wage / 100, sc_tariff / 100)

st.sidebar.subheader("5. View")
view = st.sidebar.radio("Currency", ["USD (AOP rate)", "Local currency"], index=0)
usd = view.startswith("USD")
st.sidebar.caption("Files are processed in memory and are not stored by the app.")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
overrides_tuple = tuple(sorted((k[0], k[1], k[2], k[3], bool(v))
                               for k, v in st.session_state["outlier_overrides"].items()))
settings_key = (int(year), int(at), int(bt_points), float(outlier_z), min_dev_pct / 100, target_pct / 100,
                tuple(method_overrides), overrides_tuple)
with st.spinner("Classifying series, running backtests and forecasting..."):
    results = cached_results(file_bytes, filename, settings_key)
target = target_pct / 100
fx = FxTable(inputs.fx, year)

if "adj_base" not in st.session_state:
    base_adj = inputs.adjustments.copy() if not inputs.adjustments.empty else pd.DataFrame(columns=ADJ_COLUMNS)
    st.session_state["adj_base"] = base_adj.reindex(columns=ADJ_COLUMNS).reset_index(drop=True)
adjustments_now = editor_frame(st.session_state["adj_base"], "adj_editor")
table, adj_warnings = build_table(results, scenario, adjustments_now, fx)

plants = sorted({sr.plant for sr in results.series.values()})
missing_rate = sorted({c for c in results.plant_currency.values() if not fx.has(c)})
ccy_label = "USD" if usd else "local currency"
cutoff_label = results.cutoff.strftime("%b-%y")

st.title("Thermal Plant Spend Forecast")
st.caption(f"Forecast year {year} · actuals through {cutoff_label} · {len(results.forecast_months)} month(s) to "
           f"forecast · {len(results.series)} series · models ran in {results.elapsed_s:.1f} s")
if usd and missing_rate:
    st.warning(f"No AOP rate to USD for {', '.join(missing_rate)}. Those plants are blank in USD views and left "
               "out of USD totals; switch to local currency or add the rate in the FX sheet.")
if not scenario.is_base:
    st.info(f"Scenario active: build plan {sc_volume:+d}%, labor efficiency {sc_prod:+d}%, wage {sc_wage:+d}%, "
            f"energy tariff {sc_tariff:+d}%.")

tabs = st.tabs(["Overview", "Forecast", "Classification", "Accuracy", "Bridge", "Adjustments", "Data check",
                "Download"])

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    by = ["Plant"] if usd else ["Plant", "Currency"]
    fy = summarize_fy(table, by, usd=usd)
    acc_plant = results.accuracy_plant.rename(columns={"Scope": "Plant"})
    fy = fy.merge(acc_plant[["Plant", "Accuracy_1m", "Accuracy_3m"]], on="Plant", how="left")
    fy = add_activity(fy, results)
    if usd:
        tot = fy[["YTD_Actual", "YTG_Forecast", "Mgmt_Adjustments", "FY_Landing", "FY_AOP", "FY_PPF",
                  "Scenario_Impact"]].sum(min_count=1)
        net = results.accuracy_network
        k = st.columns(5)
        k[0].metric("FY landing (USD)", money(tot["FY_Landing"]),
                    help="YTD actual + remaining-month forecast + management adjustments.")
        for col_idx, ref in ((1, "FY_AOP"), (2, "FY_PPF")):
            ref_val = tot[ref]
            has_ref = bool(np.isfinite(ref_val) and ref_val != 0)
            delta_val = tot["FY_Landing"] - ref_val if has_ref else np.nan
            k[col_idx].metric("vs " + ref.split("_")[1], money(delta_val),
                              pct(delta_val / abs(ref_val), True) if has_ref else None, delta_color="inverse",
                              help=None if has_ref else f"No {ref.split('_')[1]} values in the file.")
        k[3].metric("Network accuracy (1m)", pct(net["Accuracy_1m"].iloc[0]) if len(net) else "n/a",
                    help="1 - WAPE of the backtest, all plants in USD at the AOP rate. 3 months ahead: "
                         + (pct(net["Accuracy_3m"].iloc[0]) if len(net) else "n/a") + ".")
        acc_s = results.accuracy_series["Accuracy_1m"].dropna()
        k[4].metric(f"Series at or above {target_pct}% (1m)", f"{int((acc_s >= target).sum())} of {len(acc_s)}")
        notes = []
        bases = set(fy["Activity_Basis"].dropna())
        if len(bases) == 1 and fy["FY_Activity"].notna().all() and fy["FY_Landing"].notna().all():
            unit_name = "earned hour" if bases == {"earned hours"} else "unit"
            cpa = fy["FY_Landing"].sum() / fy["FY_Activity"].sum()
            notes.append(f"Network conversion cost per {unit_name}: {cpa:,.2f} USD "
                         f"({fy['FY_Activity'].sum():,.0f} {bases.pop()} in the year, actual + build plan).")
        if not scenario.is_base:
            notes.append(f"Scenario impact on the remaining months: {money(tot['Scenario_Impact'])} USD.")
        for n in notes:
            st.caption(n)

    st.subheader("Landing by plant")
    show = fy.copy()
    show["Status_1m"] = np.where(show["Accuracy_1m"] >= target, "Meets target",
                                 np.where(show["Accuracy_1m"].notna(), "Below target", "n/a"))
    money_cols = ["YTD_Actual", "YTG_Forecast", "Mgmt_Adjustments", "FY_Landing", "FY_AOP", "Vs_AOP", "FY_PPF",
                  "Vs_PPF", "Scenario_Impact"]
    cols = by + money_cols[:6] + ["Vs_AOP_%", "FY_PPF", "Vs_PPF", "FY_Activity", "Activity_Basis", "Cost_per_Activity",
                                  "Accuracy_1m", "Accuracy_3m", "Status_1m"]
    st.dataframe(round_money(show[cols], money_cols + ["FY_Activity"]), hide_index=True, width="stretch",
                 column_config={**money_config(money_cols + ["FY_Activity"]),
                                **pct_config(["Vs_AOP_%", "Accuracy_1m", "Accuracy_3m"]),
                                "Cost_per_Activity": st.column_config.NumberColumn(
                                    "Cost per earned hour", format="%.2f",
                                    help="FY landing / FY activity (earned hours, or units when standards are not "
                                         "loaded). Activity = actual through the cutoff + build plan.")})

    st.subheader("Monthly profile")
    scope_opts = (["All plants"] if usd else []) + plants
    scope = st.selectbox("Scope", scope_opts, key="ov_scope")
    sub = table if scope == "All plants" else table[table["Plant"] == scope]
    sfx = "_USD" if usd else ""
    monthly = sub.groupby("Date", as_index=False).agg(
        Actual=("Actual" + sfx, lambda s: s.sum(min_count=1)),
        Forecast=("Forecast" + sfx, lambda s: s.sum(min_count=1)),
        Adj=("Adjustment" + sfx, "sum"),
        AOP=("AOP" + sfx, lambda s: s.sum(min_count=1)),
        PPF=("PPF" + sfx, lambda s: s.sum(min_count=1)),
    )
    is_fc = monthly["Date"] > results.cutoff
    monthly["Forecast"] = np.where(is_fc, monthly["Forecast"].fillna(0) + monthly["Adj"].fillna(0), np.nan)
    monthly["Actual"] = np.where(is_fc, np.nan, monthly["Actual"])
    unit = "USD" if usd else results.plant_currency.get(scope, "local")
    st.plotly_chart(charts.overview_chart(monthly, f"{scope}: actual, forecast and AOP ({year})", unit, pal=PAL),
                    width="stretch", theme=None)
    cpa_df = monthly_cost_per_activity(table, results, None if scope == "All plants" else scope, usd)
    if not cpa_df.empty and cpa_df["Cost_per_Activity"].notna().any():
        basis = cpa_df["Basis"].iloc[0] or "activity"
        per = "earned hour" if basis == "earned hours" else "unit"
        st.plotly_chart(charts.cost_per_activity_chart(cpa_df, f"{scope}: conversion cost per {per}",
                                                       f"{unit} per {per}", pal=PAL),
                        width="stretch", theme=None)
        st.caption("Spend (actual, then forecast plus adjustments) divided by activity (actual, then build plan). "
                   "A rising line with a flat build plan points to fixed-cost absorption or efficiency issues.")

# ---------------------------------------------------------------------------
# Forecast detail
# ---------------------------------------------------------------------------
with tabs[1]:
    c1, c2, c3 = st.columns(3)
    plant = c1.selectbox("Plant", plants, key="fc_plant")
    minis = sorted({sr.mini for sr in results.series.values() if sr.plant == plant})
    mini = c2.selectbox("Product line", minis, format_func=lambda m: m or "(plant level)", key="fc_mini") \
        if minis != [""] else ""
    buckets = sorted({sr.bucket for sr in results.series.values() if sr.plant == plant and sr.mini == mini},
                     key=bucket_order)
    bucket = c3.selectbox("Bucket", ["All buckets"] + buckets, key="fc_bucket")
    currency = results.plant_currency.get(plant, "USD")
    unit = "USD" if usd else currency

    def conv(s: pd.Series) -> pd.Series:
        if not usd or s is None or len(s) == 0:
            return s
        return s * fx.rates_for(currency, s.index)

    sel = table[(table["Plant"] == plant) & (table["Product_Line"] == mini)]
    if bucket == "All buckets":
        sfx = "_USD" if usd else ""
        g = sel.groupby("Date").agg(**{k: (k + sfx, lambda s: s.sum(min_count=1))
                                       for k in ("Actual", "Forecast", "Forecast_Base", "AOP", "PPF",
                                                 "Actual_After_Cutoff", "Adjustment")})
        hist_keys = [sr for sr in results.series.values() if sr.plant == plant and sr.mini == mini]
        hist = pd.concat([conv(sr.history) for sr in hist_keys], axis=1).sum(axis=1, min_count=1)
        hist = hist[hist.index >= pd.Timestamp(year - 1, 1, 1)]
        fc = g["Forecast"][g.index > results.cutoff] + g["Adjustment"][g.index > results.cutoff]
        nan = pd.Series(np.nan, index=fc.index)
        fig = charts.series_chart(hist, fc, nan, nan, g["AOP"], g["PPF"], g["Actual_After_Cutoff"],
                                  f"{series_label(plant, mini)}: all buckets", unit, pal=PAL)
        st.plotly_chart(fig, width="stretch", theme=None)
        st.caption("Plant total = sum of bucket forecasts plus management adjustments. Intervals are shown by bucket.")
        rows = []
        for sr in sorted(hist_keys, key=lambda s: bucket_order(s.bucket)):
            rows.append({"Bucket": sr.bucket, "Method": sr.method, "Behavior": sr.behavior.get("Behavior"),
                         "Forecastability": sr.forecastability.get("Segment"),
                         "Accuracy_1m": sr.metrics["Accuracy_1m"], "Accuracy_3m": sr.metrics["Accuracy_3m"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config=pct_config(["Accuracy_1m", "Accuracy_3m"]))
    else:
        sr = results.series[(plant, mini, bucket)]
        hist = conv(sr.history)
        hist = hist[hist.index >= pd.Timestamp(year - 1, 1, 1)]
        row = sel[sel["Bucket"] == bucket].set_index("Date")
        sfx = "_USD" if usd else ""
        fc_mask = row.index > results.cutoff
        fig = charts.series_chart(hist, row["Forecast" + sfx][fc_mask], row["Lower" + sfx][fc_mask],
                                  row["Upper" + sfx][fc_mask], row["AOP" + sfx], row["PPF" + sfx],
                                  row["Actual_After_Cutoff" + sfx], f"{sr.label}", unit,
                                  base=row["Forecast_Base" + sfx][fc_mask], pal=PAL)
        st.plotly_chart(fig, width="stretch", theme=None)
        m = sr.metrics
        k = st.columns(5)
        k[0].metric("Method", sr.method)
        k[1].metric("Accuracy 1m", pct(m["Accuracy_1m"]),
                    help="1 - WAPE of forecasts made 1 month before each closed month in the backtest.")
        k[2].metric("Accuracy 3m", pct(m["Accuracy_3m"]))
        k[3].metric("Bias", pct(m["Bias"], True), help="Positive = forecast above actual (over-forecast).")
        k[4].metric("FVA vs naive (1m)", pct(m["FVA_1m"], True),
                    help="Accuracy gain over repeating the last closed month. Negative means naive was better.")
        st.markdown(f"**How it was forecast:** {sr.note or C.METHOD_DESCRIPTIONS.get(sr.method, sr.method)}")
        relevant = {
            "labor": lambda d: d in (C.ACTIVITY, C.UNITS, C.EARNED_HOURS, C.LABOR_HOURS, C.EFFICIENCY, C.WAGE),
            "utilities": lambda d: d == C.ACTIVITY or d.startswith(C.UTIL_UNITS) or d.startswith(C.UTIL_TARIFF),
            "regression": lambda d: d == C.ACTIVITY,
        }.get(sr.kind)
        shown_drivers = [d for d in sr.planned_drivers if relevant and relevant(d) and d != C.ACTIVITY]
        st.markdown(f"**Classification:** {sr.behavior.get('Behavior', 'n/a')} cost · "
                    f"{sr.forecastability.get('Segment', 'n/a')} series"
                    + (f" · planned drivers used: {', '.join(shown_drivers)}" if shown_drivers else ""))
        if sr.metrics["Accuracy_1m"] < target:
            st.warning(f"Backtest accuracy at 1 month is below the {target_pct}% target. "
                       "Review atypical items, drivers or the method for this bucket.")
        detail = row[["Actual" + sfx, "Forecast_Base" + sfx, "Forecast" + sfx, "Lower" + sfx, "Upper" + sfx,
                      "Adjustment" + sfx, "Landing" + sfx, "AOP" + sfx, "PPF" + sfx]].copy()
        detail.columns = ["Actual", "Forecast_Base", "Forecast", "Lower_80", "Upper_80", "Adjustment", "Landing",
                          "AOP", "PPF"]
        detail.index = detail.index.strftime("%b-%y")
        detail = detail.reset_index().rename(columns={"Date": "Month", "index": "Month"})
        mcols = list(detail.columns[1:])
        st.dataframe(round_money(detail, mcols), hide_index=True, width="stretch", column_config=money_config(mcols))
        if not sr.candidates.empty and len(sr.candidates) > 1:
            with st.expander("Method comparison in the backtest"):
                st.dataframe(sr.candidates, hide_index=True, width="stretch",
                             column_config=pct_config(["WAPE_1m", "WAPE_3m", "WAPE_pooled"]))
        if sr.components:
            with st.expander("Driver components (base case, local currency)"):
                comp = pd.DataFrame(index=sr.forecast.index.strftime("%b-%y"))
                c = sr.components
                if sr.kind == "labor":
                    comp["Earned_hours"] = c["activity"]
                    comp["Labor_efficiency"] = c["eff"]
                    comp["Labor_hours"] = c["hours"]
                    comp["Wage_rate"] = c["wage"]
                    st.caption(f"Calibration factor (actual vs formula, last {C.CALIBRATION_MONTHS} months): "
                               f"{c['k']:.3f}" + ("" if c.get("uses_hours") else
                                                   ". No labor hours loaded: the factor is a $ per earned hour."))
                elif sr.kind == "utilities":
                    for name, u, t in zip(c["types"], c["units"], c["tariff"]):
                        comp[f"{name}_units"] = u
                        comp[f"{name}_tariff"] = t
                    st.caption(f"Fixed charges per month: {money(c['fixed'])}; calibration factor {c['k']:.3f}")
                elif sr.kind == "maintenance":
                    comp["Unplanned_base"] = c["base"]
                    comp["Planned_events"] = c["events"]
                elif sr.kind == "regression":
                    comp["Activity"] = c["activity"]
                    st.caption(f"Fixed per month {money(c['a'])}; rate per activity unit {c['b']:.2f}")
                st.dataframe(comp.reset_index().rename(columns={"index": "Month"}), hide_index=True,
                             width="stretch")

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Series classification")
    st.caption("Behavior: regression of spend on activity (earned standard hours, or units when standards are not "
               "loaded) over the last 24 months; variable share = rate x average activity / average spend. "
               "Forecastability: coefficient of variation around a linear trend of the cleaned series, so a "
               "product ramp does not count as noise.")
    cl = results.classification.copy()
    cl = cl.sort_values(["Plant", "Product_Line", "Bucket"], key=lambda s: s.map(bucket_order) if s.name == "Bucket" else s)
    st.dataframe(round_money(cl, ["Fixed_per_Month"]), hide_index=True, width="stretch",
                 column_config={**money_config(["Fixed_per_Month"]),
                                **pct_config(["Variable_Share", "CV", "Naive_WAPE_1m", "Zero_Share"]),
                                "R2": st.column_config.NumberColumn("R2", format="%.2f"),
                                "P_Value": st.column_config.NumberColumn("P value", format="%.3f"),
                                "Rate_per_Activity": st.column_config.NumberColumn("Rate per earned hour",
                                                                                   format="%.2f")})
    counts = cl["Behavior"].value_counts()
    st.caption("Behavior count: " + ", ".join(f"{k} {v}" for k, v in counts.items()))

    if not inputs.mapping_review.empty:
        st.subheader("GL account mapping")
        st.dataframe(inputs.mapping_review, hide_index=True, width="stretch")

    st.subheader("Atypical items")
    st.caption(f"Flagged when the modified z-score against a 5-month rolling median is above {outlier_z} and the "
               f"deviation is {min_dev_pct}% or more, plus negative values. Excluded "
               "points are replaced by the rolling median when models are fitted; actuals are not changed. "
               "Uncheck Exclude to keep a point, then apply.")
    out = results.outliers.copy()
    if out.empty:
        st.info("No atypical items were flagged.")
    else:
        with st.form("outlier_form"):
            edited = st.data_editor(
                round_money(out, ["Value", "Baseline"]), hide_index=True, width="stretch",
                disabled=[c for c in out.columns if c != "Exclude"],
                column_config={**money_config(["Value", "Baseline"]),
                               "Z": st.column_config.NumberColumn("Z", format="%.1f"),
                               "Exclude": st.column_config.CheckboxColumn("Exclude")},
                key="outlier_editor")
            if st.form_submit_button("Apply and re-run"):
                new = {}
                for r in edited.itertuples(index=False):
                    if bool(r.Exclude) != bool(r.Auto_Flag):
                        new[(r.Plant, r.Product_Line, r.Bucket, r.Month)] = bool(r.Exclude)
                st.session_state["outlier_overrides"] = new
                st.rerun()
    with st.expander("Exclude a month that was not flagged"):
        keys = sorted(results.series.keys(), key=lambda k: (k[0], k[1], bucket_order(k[2])))
        choice = st.selectbox("Series", keys, format_func=lambda k: f"{series_label(k[0], k[1])} / {k[2]}",
                              key="manual_series")
        months_avail = [d.strftime("%Y-%m") for d in results.series[choice].history.index]
        month_pick = st.selectbox("Month", months_avail[::-1], key="manual_month")
        if st.button("Exclude this month"):
            st.session_state["outlier_overrides"][(choice[0], choice[1], choice[2], month_pick)] = True
            st.rerun()

# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------
with tabs[3]:
    hz = st.radio("Horizon", ["1 month ahead", "3 months ahead"], horizontal=True, key="acc_hz")
    col = "Accuracy_1m" if hz.startswith("1") else "Accuracy_3m"
    acc = results.accuracy_series.copy()
    acc["Row"] = [series_label(p, m) for p, m in zip(acc["Plant"], acc["Product_Line"])]
    if acc[col].notna().any():
        st.plotly_chart(charts.accuracy_heatmap(acc, col, target, f"Backtest accuracy (1 - WAPE), {hz}", pal=PAL),
                        width="stretch", theme=None)
    plant_tbl = pd.concat([results.accuracy_plant, results.accuracy_network], ignore_index=True)
    plant_tbl["Status"] = np.where(plant_tbl[col] >= target, "Meets target",
                                   np.where(plant_tbl[col].notna(), "Below target", "n/a"))
    st.subheader("Plant and network totals")
    st.dataframe(plant_tbl, hide_index=True, width="stretch",
                 column_config=pct_config(["Accuracy_1m", "Accuracy_3m", "Bias", "FVA_1m", "FVA_3m"]))
    st.subheader("By series")
    acc["Status"] = np.where(acc[col] >= target, "Meets target", np.where(acc[col].notna(), "Below target", "n/a"))
    st.dataframe(acc.drop(columns=["Row"]), hide_index=True, width="stretch",
                 column_config=pct_config(["Accuracy_1m", "Accuracy_3m", "Bias", "FVA_1m", "FVA_3m"]))
    st.caption(f"Backtest: each of the last {bt_points} closed months is forecast 1 and 3 months before it, using "
               "only data available at that time. Driver buckets use actual volume, crew, wage and tariffs in the "
               "backtest because historical plans are not loaded; their score measures the driver formula, not the "
               "error of the volume plan. Plant and network totals use only months where every series has a point.")

# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------
with tabs[4]:
    b1, b2 = st.columns(2)
    scope_opts = (["All plants"] if usd else []) + plants
    scope = b1.selectbox("Scope", scope_opts, key="br_scope")
    reference = b2.radio("Reference", [C.AOP, C.PPF], horizontal=True, key="br_ref")
    br = bridge(table, reference, None if scope == "All plants" else scope, usd)
    unit = "USD" if usd else results.plant_currency.get(scope, "local")
    ref_total, model_total, adj_total = br["Ref"].sum(), br["Model"].sum(), br["Adj"].sum()
    if ref_total == 0:
        st.info(f"No {reference} values for this scope.")
    else:
        st.plotly_chart(charts.waterfall(br, f"FY {reference}", ref_total, model_total + adj_total, adj_total,
                                         f"{scope}: FY {reference} to FY landing ({unit})", unit, pal=PAL),
                        width="stretch", theme=None)
        shown = br.rename(columns={"Ref": f"FY_{reference}", "Model": "FY_Landing_before_adj", "Adj": "Mgmt_Adj"})
        shown["Delta_%"] = np.where(shown[f"FY_{reference}"] != 0, shown["Delta"] / shown[f"FY_{reference}"].abs(),
                                    np.nan)
        mcols = [f"FY_{reference}", "FY_Landing_before_adj", "Delta", "Mgmt_Adj"]
        st.dataframe(round_money(shown, mcols), hide_index=True, width="stretch",
                     column_config={**money_config(mcols), **pct_config(["Delta_%"])})
        st.caption("Red bars add spend versus the reference (unfavorable); navy bars reduce it. The axis starts "
                   "above zero so the steps are readable.")

# ---------------------------------------------------------------------------
# Adjustments
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Management adjustments")
    st.caption("Amounts in plant currency for open months. They are added on top of the model and shown as a "
               "separate line in the landing, the bridge and the download. Rows without an owner are marked "
               "TO BE CONFIRMED.")
    bucket_opts = sorted(set(C.STANDARD_BUCKETS) | set(inputs.spend["Bucket"].unique()), key=bucket_order)
    st.data_editor(
        st.session_state["adj_base"], key="adj_editor", num_rows="dynamic", hide_index=True, width="stretch",
        column_config={
            "Plant": st.column_config.SelectboxColumn("Plant", options=plants, required=True),
            "Product_Line": st.column_config.TextColumn("Product_Line"),
            "Bucket": st.column_config.SelectboxColumn("Bucket", options=bucket_opts, required=True),
            "Year": st.column_config.NumberColumn("Year", default=int(year), format="%d"),
            "Month": st.column_config.NumberColumn("Month", min_value=1, max_value=12, step=1, format="%d"),
            "Amount": st.column_config.NumberColumn("Amount", format="localized"),
            "Owner": st.column_config.TextColumn("Owner"),
            "Reason": st.column_config.TextColumn("Reason"),
        })
    for w in adj_warnings:
        st.warning(w)
    adj_clean, _ = clean_adjustments(adjustments_now, year)
    if not adj_clean.empty:
        st.caption("Total adjustments by plant (plant currency): " + ", ".join(
            f"{p} {money(v)}" for p, v in adj_clean.groupby("Plant")["Amount"].sum().items()))

# ---------------------------------------------------------------------------
# Data check
# ---------------------------------------------------------------------------
with tabs[6]:
    issues = inputs.issues.copy()
    if adj_warnings:
        issues = pd.concat([issues, pd.DataFrame([{"Severity": "Warning", "Sheet": "Adjustments",
                                                   "Check": "Adjustments", "Message": w} for w in adj_warnings])],
                           ignore_index=True)
    n_err = int((issues["Severity"] == "Error").sum()) if not issues.empty else 0
    n_warn = int((issues["Severity"] == "Warning").sum()) if not issues.empty else 0
    k = st.columns(4)
    k[0].metric("Errors", n_err)
    k[1].metric("Warnings", n_warn)
    k[2].metric("Series", len(results.series))
    act_dates = inputs.spend.loc[inputs.spend["Version"] == C.ACT, "Date"]
    k[3].metric("Months of actuals", act_dates.dt.to_period("M").nunique())
    st.caption(f"Actual history: {act_dates.min():%b-%y} to {act_dates.max():%b-%y}. Formats detected: "
               + "; ".join(f"{s}: {f}" for s, f in inputs.formats.items()))
    if issues.empty:
        st.success("No issues found.")
    else:
        st.dataframe(issues, hide_index=True, width="stretch",
                     column_config={"Message": st.column_config.TextColumn("Message", width="large")})
    with st.expander("Parsed spend data (first 500 rows)"):
        st.dataframe(inputs.spend.head(500), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def output_sheets() -> dict[str, pd.DataFrame]:
    sheets: dict[str, pd.DataFrame] = {}
    usd_ok = not missing_rate
    if usd_ok:
        s_usd = summarize_fy(table, ["Plant"], usd=True)
        total = s_usd.drop(columns=["Plant"]).sum(numeric_only=True)
        total["Vs_AOP_%"] = total["Vs_AOP"] / abs(total["FY_AOP"]) if total["FY_AOP"] else np.nan
        sheets["Summary_USD"] = pd.concat([s_usd, pd.DataFrame([{"Plant": "Total", **total.to_dict()}])],
                                          ignore_index=True)
    sheets["Summary_Local"] = summarize_fy(table, ["Plant", "Currency"], usd=False)
    sheets["By_Bucket_Local"] = summarize_fy(table, ["Plant", "Product_Line", "Currency", "Bucket"], usd=False)
    cols = ["Plant", "Product_Line", "Bucket", "Currency", "Date", "Period", "Method", "Actual", "Forecast_Base",
            "Forecast", "Lower", "Upper", "Adjustment", "Landing", "AOP", "PPF", "Rate_to_USD", "Landing_USD",
            "AOP_USD", "PPF_USD"]
    sheets["Forecast_Monthly"] = table[[c for c in cols if c in table.columns]]
    wide = table.assign(Month=table["Date"].dt.strftime("%b-%y")).pivot_table(
        index=["Plant", "Product_Line", "Bucket", "Currency"], columns="Month", values="Landing", aggfunc="sum",
        sort=False)
    sheets["Landing_Wide_Local"] = wide.reset_index()
    gl = allocate_gl(table, inputs.gl_detail, results)
    if not gl.empty:
        sheets["GL_Detail"] = gl
    sheets["Classification"] = results.classification
    sheets["Atypical_Items"] = to_bool_text(results.outliers)
    sheets["Accuracy_Series"] = results.accuracy_series
    sheets["Accuracy_Plant"] = pd.concat([results.accuracy_plant, results.accuracy_network], ignore_index=True)
    sel_rows, bt_rows = [], []
    for sr in results.series.values():
        if not sr.candidates.empty:
            sel_rows.append(sr.candidates.assign(Plant=sr.plant, Product_Line=sr.mini, Bucket=sr.bucket))
        if not sr.backtest.empty:
            bt_rows.append(sr.backtest.assign(Plant=sr.plant, Product_Line=sr.mini, Bucket=sr.bucket, Method=sr.method))
    if sel_rows:
        ms = pd.concat(sel_rows, ignore_index=True)
        sheets["Method_Selection"] = to_bool_text(ms[["Plant", "Product_Line", "Bucket"] +
                                                    [c for c in ms.columns if c not in ("Plant", "Product_Line", "Bucket")]])
    if bt_rows:
        bt = pd.concat(bt_rows, ignore_index=True)
        sheets["Backtest_Points"] = bt[["Plant", "Product_Line", "Bucket", "Method", "Eval_Date", "Horizon", "Actual",
                                        "Forecast", "Naive"]]
    if not inputs.mapping_review.empty:
        sheets["Mapping_Review"] = inputs.mapping_review
    sheets["Adjustments"] = clean_adjustments(adjustments_now, year)[0]
    fx_used = {c: fx.rate(c) for c in sorted(set(results.plant_currency.values()))}
    meta = {
        "Source file": filename,
        "Forecast year": year,
        "Actuals through": cutoff_label,
        "Months forecast": len(results.forecast_months),
        "Backtest points": bt_points,
        "Atypical item threshold (modified z)": outlier_z,
        "Atypical item minimum deviation": f"{min_dev_pct}%",
        "Accuracy target": f"{target_pct}%",
        "Method overrides": ", ".join(f"{b}: {m}" for b, m in method_overrides) or "None (defaults)",
        "User changes to atypical items": len(st.session_state["outlier_overrides"]),
        "Scenario": (f"build plan {sc_volume:+d}%, labor efficiency {sc_prod:+d}%, wage {sc_wage:+d}%, "
                     f"energy tariff {sc_tariff:+d}%") if not scenario.is_base else "Base (no changes)",
        "FX (USD per local unit, AOP)": ", ".join(f"{c} {r:.4f}" for c, r in fx_used.items()),
        "Accuracy definition": "1 - WAPE of a rolling-origin backtest at 1 and 3 months",
        "Driver backtest note": "Driver buckets are backtested with actual drivers (ex post).",
    }
    sheets["Assumptions"] = assumptions_frame(meta)
    sheets["Data_Issues"] = inputs.issues
    return sheets


with tabs[7]:
    st.subheader("Download")
    st.write("The workbook includes the landing by plant and bucket, the monthly forecast in local currency and USD, "
             "GL detail (when GL rows were loaded), classification, atypical items, backtest accuracy, method "
             "selection, adjustments, assumptions and data checks.")
    if st.button("Build forecast workbook", type="primary"):
        with st.spinner("Building workbook..."):
            st.session_state["workbook"] = write_workbook(output_sheets())
    if st.session_state.get("workbook"):
        st.download_button("Download forecast workbook (.xlsx)", st.session_state["workbook"],
                           file_name=f"plant_spend_forecast_{year}_{cutoff_label}.xlsx", mime=XLSX_MIME)
        st.caption("Rebuild the workbook after changing settings, scenarios or adjustments.")
    st.download_button("Download input template", template_bytes(), file_name="plant_spend_template.xlsx",
                       mime=XLSX_MIME, key="tpl2")
