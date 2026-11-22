"""Scenarios, management adjustments, landing tables, bridges and GL detail."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import sc_config as C
from sc_drivers import recompute
from sc_engine import Results
from sc_fx import FxTable
from sc_utils import month_start

MONEY_COLS = ["Actual", "Forecast_Base", "Forecast", "Lower", "Upper", "Adjustment", "Landing", "AOP", "PPF",
              "Actual_After_Cutoff"]


@dataclass(frozen=True)
class Scenario:
    volume: float = 0.0       # build plan (units and earned hours)
    efficiency: float = 0.0   # labor efficiency (earned hours / labor hours)
    wage: float = 0.0
    tariff: float = 0.0       # energy tariffs

    @property
    def is_base(self) -> bool:
        return not any((self.volume, self.efficiency, self.wage, self.tariff))


def clean_adjustments(adj: pd.DataFrame | None, year: int) -> tuple[pd.DataFrame, list[str]]:
    cols = ["Plant", "Product_Line", "Bucket", "Year", "Month", "Amount", "Owner", "Reason"]
    warnings: list[str] = []
    if adj is None or adj.empty:
        return pd.DataFrame(columns=cols), warnings
    df = adj.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df["Plant"] = df["Plant"].fillna("").astype(str).str.strip()
    df["Product_Line"] = df["Product_Line"].fillna("").astype(str).str.strip()
    df["Bucket"] = df["Bucket"].fillna("").astype(str).str.strip()
    df["Owner"] = df["Owner"].fillna("").astype(str).str.strip()
    df["Reason"] = df["Reason"].fillna("").astype(str).str.strip()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    bad = (df["Plant"] == "") | (df["Bucket"] == "") | df["Month"].isna() | df["Amount"].isna()
    if bad.any():
        warnings.append(f"{int(bad.sum())} adjustment row(s) without plant, bucket, month or amount were ignored.")
    df = df[~bad].copy()
    df["Year"] = df["Year"].fillna(year).astype(int)
    df["Month"] = df["Month"].astype(int)
    other_year = df["Year"] != year
    if other_year.any():
        warnings.append(f"{int(other_year.sum())} adjustment row(s) outside {year} were ignored.")
        df = df[~other_year]
    no_owner = df["Owner"] == ""
    if no_owner.any():
        warnings.append(f"{int(no_owner.sum())} adjustment row(s) have no owner (shown as TO BE CONFIRMED).")
        df.loc[no_owner, "Owner"] = "TO BE CONFIRMED"
    return df[cols].reset_index(drop=True), warnings


def build_table(results: Results, scenario: Scenario, adjustments: pd.DataFrame, fx: FxTable) -> tuple[pd.DataFrame, list[str]]:
    """Monthly table for the forecast year, one row per series and month, local and USD."""
    months = results.year_months
    cutoff = results.cutoff
    frames = []
    for sr in results.series.values():
        df = pd.DataFrame(index=months)
        df.index.name = "Date"
        df["Plant"], df["Product_Line"], df["Bucket"], df["Currency"] = sr.plant, sr.mini, sr.bucket, sr.currency
        df["Method"] = sr.method
        df["Actual"] = sr.history.reindex(months).where(months <= cutoff)
        base = sr.forecast.reindex(months)
        df["Forecast_Base"] = base
        scen = recompute(sr.kind, sr.components, scenario) if not scenario.is_base else None
        if scen is not None and len(sr.forecast):
            df["Forecast"] = pd.Series(np.asarray(scen, float), index=sr.forecast.index).reindex(months)
        else:
            df["Forecast"] = base
        shift = (df["Forecast"] - df["Forecast_Base"]).fillna(0.0)
        df["Lower"] = sr.lower.reindex(months) + shift
        df["Upper"] = sr.upper.reindex(months) + shift
        df["AOP"] = sr.aop.reindex(months)
        df["PPF"] = sr.ppf.reindex(months)
        df["Actual_After_Cutoff"] = sr.later.reindex(months)
        frames.append(df.reset_index())
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    adj, warnings = clean_adjustments(adjustments, results.year)
    if not adj.empty:
        adj = adj.copy()
        adj["Date"] = [month_start(y, m) for y, m in zip(adj["Year"], adj["Month"])]
        closed = adj["Date"] <= cutoff
        if closed.any():
            warnings.append(f"{int(closed.sum())} adjustment row(s) fall in closed months and were ignored.")
            adj = adj[~closed]
        keys = set(zip(table["Plant"], table["Product_Line"], table["Bucket"])) if not table.empty else set()
        mask = pd.Series([k not in keys for k in zip(adj["Plant"], adj["Product_Line"], adj["Bucket"])],
                         index=adj.index, dtype=bool)
        missing = adj.loc[mask]
        for (plant, mini, bucket), _grp in missing.groupby(["Plant", "Product_Line", "Bucket"]):
            extra = pd.DataFrame({"Date": months})
            extra["Plant"], extra["Product_Line"], extra["Bucket"] = plant, mini, bucket
            extra["Currency"] = results.plant_currency.get(plant, "USD")
            extra["Method"] = "Adjustment only"
            frames.append(extra)
        if len(missing):
            table = pd.concat(frames, ignore_index=True)
        adj_sum = adj.groupby(["Plant", "Product_Line", "Bucket", "Date"], as_index=False)["Amount"].sum()
        table = table.merge(adj_sum.rename(columns={"Amount": "Adjustment"}),
                            on=["Plant", "Product_Line", "Bucket", "Date"], how="left")
    if table.empty:
        return table, warnings
    if "Adjustment" not in table.columns:
        table["Adjustment"] = 0.0
    table["Adjustment"] = table["Adjustment"].fillna(0.0)
    closed = table["Date"] <= cutoff
    table["Landing"] = np.where(closed, table["Actual"].fillna(0.0), table["Forecast"].fillna(0.0)) + table["Adjustment"]
    table["Period"] = np.where(closed, "Actual", "Forecast")
    rates = np.array([fx.rate(c, d.month) for c, d in zip(table["Currency"], table["Date"])], dtype=float)
    table["Rate_to_USD"] = rates
    for col in MONEY_COLS:
        if col in table.columns:
            table[col + "_USD"] = table[col] * rates
    return table, warnings


def summarize_fy(table: pd.DataFrame, by: list[str], usd: bool = True) -> pd.DataFrame:
    """YTD actual, YTG forecast, adjustments, FY landing, AOP and PPF, with deltas."""
    sfx = "_USD" if usd else ""
    t = table.copy()
    t["YTD_Actual"] = np.where(t["Period"] == "Actual", t["Actual" + sfx].fillna(0.0), 0.0)
    t["YTG_Forecast"] = np.where(t["Period"] == "Forecast", t["Forecast" + sfx].fillna(0.0), 0.0)
    t["Mgmt_Adjustments"] = t["Adjustment" + sfx].fillna(0.0)
    t["FY_Landing"] = t["Landing" + sfx].fillna(0.0)
    t["FY_AOP"] = t["AOP" + sfx]
    t["FY_PPF"] = t["PPF" + sfx]
    if "Forecast_Base" + sfx in t:
        t["YTG_Base"] = np.where(t["Period"] == "Forecast", t["Forecast_Base" + sfx].fillna(0.0), 0.0)
    out = t.groupby(by, as_index=False)[["YTD_Actual", "YTG_Forecast", "YTG_Base", "Mgmt_Adjustments",
                                         "FY_Landing", "FY_AOP", "FY_PPF"]].sum(min_count=1)
    if usd:
        missing = t.assign(_no_rate=t["Rate_to_USD"].isna()).groupby(by)["_no_rate"].any().reset_index()
        out = out.merge(missing, on=by, how="left")
        money = ["YTD_Actual", "YTG_Forecast", "YTG_Base", "Mgmt_Adjustments", "FY_Landing", "FY_AOP", "FY_PPF"]
        out.loc[out["_no_rate"].fillna(False).astype(bool), money] = np.nan
        out = out.drop(columns="_no_rate")
    out["Vs_AOP"] = out["FY_Landing"] - out["FY_AOP"]
    out["Vs_AOP_%"] = np.where(out["FY_AOP"] != 0, out["Vs_AOP"] / out["FY_AOP"].abs(), np.nan)
    out["Vs_PPF"] = out["FY_Landing"] - out["FY_PPF"]
    out["Scenario_Impact"] = out["YTG_Forecast"] - out["YTG_Base"]
    return out


def bridge(table: pd.DataFrame, reference: str, plant: str | None, usd: bool) -> pd.DataFrame:
    """Landing minus reference by bucket, plus management adjustments as a separate step."""
    sfx = "_USD" if usd else ""
    t = table if plant is None else table[table["Plant"] == plant]
    ref_col = ("AOP" if reference == C.AOP else "PPF") + sfx
    t = t.assign(Model=np.where(t["Period"] == "Actual", t["Actual" + sfx].fillna(0.0), t["Forecast" + sfx].fillna(0.0)),
                 Ref=t[ref_col].fillna(0.0), Adj=t["Adjustment" + sfx].fillna(0.0))
    by_bucket = t.groupby("Bucket", as_index=False)[["Ref", "Model", "Adj"]].sum()
    by_bucket["Delta"] = by_bucket["Model"] - by_bucket["Ref"]
    order = {b: i for i, b in enumerate(C.STANDARD_BUCKETS)}
    by_bucket = by_bucket.sort_values("Bucket", key=lambda s: s.map(lambda b: order.get(b, 99)))
    return by_bucket.reset_index(drop=True)


def allocate_gl(table: pd.DataFrame, gl_detail: pd.DataFrame, results: Results) -> pd.DataFrame:
    """Spread bucket forecasts to GL accounts using the mix of the last 6 closed months."""
    if gl_detail is None or gl_detail.empty or table.empty:
        return pd.DataFrame()
    cutoff = results.cutoff
    act = gl_detail[(gl_detail["Version"] == C.ACT) & (gl_detail["Date"] <= cutoff)]
    if act.empty:
        return pd.DataFrame()
    start = cutoff - pd.DateOffset(months=C.GL_MIX_MONTHS - 1)
    keys = ["Plant", "Product_Line", "Bucket"]
    acct = ["GL_Account", "Cost_Center"]
    recent = act[act["Date"] >= start]
    mix = recent.groupby(keys + acct, as_index=False)["Value"].sum()
    tot = mix.groupby(keys)["Value"].transform("sum")
    mix["Share"] = np.where(tot != 0, mix["Value"] / tot, np.nan)
    counts = mix.groupby(keys)["Value"].transform("count")
    mix["Share"] = mix["Share"].fillna(1.0 / counts)
    desc = act.groupby(acct, as_index=False)["Description"].first()
    fc = table[table["Period"] == "Forecast"][keys + ["Date", "Forecast", "Adjustment"]]
    alloc = fc.merge(mix[keys + acct + ["Share"]], on=keys, how="inner")
    alloc["Value"] = (alloc["Forecast"].fillna(0.0) + alloc["Adjustment"].fillna(0.0)) * alloc["Share"]
    alloc["Period"] = "Forecast (allocated)"
    ytd = act[act["Date"].dt.year == results.year][keys + acct + ["Date", "Value"]].copy()
    ytd["Period"] = "Actual"
    out = pd.concat([ytd, alloc[keys + acct + ["Date", "Value", "Period"]]], ignore_index=True)
    out = out.merge(desc, on=acct, how="left")
    return out.sort_values(keys + acct + ["Date"]).reset_index(drop=True)


def add_activity(fy: pd.DataFrame, results: Results, landing_col: str = "FY_Landing") -> pd.DataFrame:
    """Add FY activity (earned hours or units) and conversion cost per activity unit by plant."""
    out = fy.copy()
    act = results.activity
    if act is None or act.empty or "Plant" not in out.columns:
        out["FY_Activity"], out["Activity_Basis"], out["Cost_per_Activity"] = np.nan, "", np.nan
        return out
    agg = act.groupby("Plant").agg(FY_Activity=("Activity", lambda s: s.sum(min_count=1)),
                                   Months=("Activity", "count"), Activity_Basis=("Basis", "first")).reset_index()
    agg.loc[agg["Months"] < 12, "FY_Activity"] = np.nan
    out = out.merge(agg.drop(columns="Months"), on="Plant", how="left")
    out["Cost_per_Activity"] = out[landing_col] / out["FY_Activity"].replace(0, np.nan)
    return out


def monthly_cost_per_activity(table: pd.DataFrame, results: Results, plant: str | None, usd: bool) -> pd.DataFrame:
    """Monthly conversion cost per activity unit for one plant or all plants (USD)."""
    act = results.activity
    if act is None or act.empty:
        return pd.DataFrame()
    sfx = "_USD" if usd else ""
    t = table if plant is None else table[table["Plant"] == plant]
    a = act if plant is None else act[act["Plant"] == plant]
    if a.empty or a["Activity"].isna().all():
        return pd.DataFrame()
    if plant is None and a["Basis"].nunique() > 1:
        return pd.DataFrame()
    spend = t.groupby("Date")["Landing" + sfx].sum(min_count=1)
    activity = a.groupby("Date")["Activity"].sum(min_count=1)
    df = pd.concat({"Spend": spend, "Activity": activity}, axis=1)
    df["Cost_per_Activity"] = df["Spend"] / df["Activity"].replace(0, np.nan)
    df["Period"] = np.where(df.index <= results.cutoff, "Actual", "Forecast")
    df["Basis"] = a["Basis"].iloc[0]
    return df.reset_index()
