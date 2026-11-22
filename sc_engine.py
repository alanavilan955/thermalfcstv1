"""Orchestration: classification, backtest, method selection and final forecast for each series."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import sc_config as C
from sc_classify import clean_series, cost_behavior, forecastability
from sc_context import MethodOutput, SeriesContext, Unavailable
from sc_driverprep import build_driver_frames
from sc_drivers import DRIVER_FUNCTIONS
from sc_fx import FxTable
from sc_io_excel import Inputs
from sc_metrics import accuracy, bias, rmse, wape
from sc_models import aop_adjusted, naive, runrate_mean, runrate_median, seasonal_naive, volume_regression
from sc_utils import month_start, shift_months

BT_COLUMNS = ["Eval_Date", "Horizon", "Actual", "Forecast"]


@dataclass(frozen=True)
class Settings:
    year: int
    actuals_through: int
    backtest_points: int = C.BACKTEST_POINTS
    outlier_z: float = C.OUTLIER_Z
    outlier_min_dev: float = C.OUTLIER_MIN_DEV
    accuracy_target: float = C.ACCURACY_TARGET
    method_overrides: tuple = ()      # ((bucket, method), ...)
    outlier_overrides: tuple = ()     # ((plant, mini, bucket, "YYYY-MM", exclude), ...)


@dataclass
class SeriesResult:
    plant: str
    mini: str
    bucket: str
    currency: str
    requested: str
    method: str
    note: str
    forecast: pd.Series
    lower: pd.Series
    upper: pd.Series
    kind: str | None
    components: dict | None
    history: pd.Series
    later: pd.Series
    aop: pd.Series
    ppf: pd.Series
    backtest: pd.DataFrame
    candidates: pd.DataFrame
    metrics: dict
    behavior: dict
    forecastability: dict
    planned_drivers: list = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str, str]:
        return self.plant, self.mini, self.bucket

    @property
    def label(self) -> str:
        return f"{self.plant}{' / ' + self.mini if self.mini else ''} / {self.bucket}"


@dataclass
class Results:
    year: int
    actuals_through: int
    cutoff: pd.Timestamp
    year_months: pd.DatetimeIndex
    forecast_months: pd.DatetimeIndex
    series: dict
    plant_currency: dict
    classification: pd.DataFrame
    outliers: pd.DataFrame
    accuracy_series: pd.DataFrame
    accuracy_plant: pd.DataFrame
    accuracy_network: pd.DataFrame
    settings: Settings
    elapsed_s: float
    activity: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------
def run_method(method: str, ctx: SeriesContext, train_end: pd.Timestamp, months: pd.DatetimeIndex,
               mode: str) -> MethodOutput:
    h = len(months)
    if method == C.M_NAIVE:
        return MethodOutput(naive(ctx.y_raw(train_end), h), note=C.METHOD_DESCRIPTIONS[C.M_NAIVE])
    if method == C.M_SNAIVE:
        return MethodOutput(seasonal_naive(ctx.y_raw(train_end), h), note=C.METHOD_DESCRIPTIONS[C.M_SNAIVE])
    if method == C.M_RR3:
        return MethodOutput(runrate_mean(ctx.y_clean(train_end), h, 3), note=C.METHOD_DESCRIPTIONS[C.M_RR3])
    if method == C.M_RR6:
        return MethodOutput(runrate_median(ctx.y_clean(train_end), h, 6), note=C.METHOD_DESCRIPTIONS[C.M_RR6])
    if method == C.M_ENSEMBLE:
        return MethodOutput(ctx.stat("y", ctx.y_clean(train_end), h, mode), note=C.METHOD_DESCRIPTIONS[C.M_ENSEMBLE])
    if method == C.M_REG:
        act_h = ctx.driver_hist(C.ACTIVITY, train_end)
        if act_h is None:
            raise Unavailable("no Units or Earned_Hours actuals")
        act_f = ctx.driver_future(C.ACTIVITY, months, mode, train_end)
        out = volume_regression(ctx.y_clean(train_end), act_h, act_f)
        if out is None:
            raise Unavailable("needs 12+ months of spend and activity plus a full build plan")
        fc, a, b = out
        return MethodOutput(fc, "regression", {"activity": np.asarray(act_f, float), "a": a, "b": b},
                            f"Spend = {a:,.0f} fixed per month + {b:,.2f} per {ctx.basis or 'activity unit'}"
                            " x build plan")
    if method == C.M_AOPADJ:
        aop_f = ctx.aop.reindex(months).to_numpy(float)
        if not np.all(np.isfinite(aop_f)):
            raise Unavailable("AOP missing for these months")
        start = month_start(ctx.year, 1)
        if train_end >= start:
            ytd_act = float(ctx.y_raw(train_end).loc[start:].sum())
            ytd_aop = float(ctx.aop.loc[start:train_end].sum())
        else:
            ytd_act = ytd_aop = 0.0
        fc, ratio = aop_adjusted(aop_f, ytd_act, ytd_aop)
        return MethodOutput(fc, None, None, f"AOP phasing x {ratio:.3f} (YTD actual / YTD AOP)")
    if method == C.M_DRIVER:
        fn = DRIVER_FUNCTIONS.get(ctx.bucket)
        if fn is None:
            raise Unavailable(f"no driver formula for {ctx.bucket}")
        return fn(ctx, train_end, months, mode)
    raise Unavailable(f"unknown method {method}")


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def backtest(ctx: SeriesContext, methods: list[str], n_points: int) -> dict[str, pd.DataFrame]:
    """Rolling-origin backtest. Each of the last n_points closed months is forecast 1 and 3 months ahead."""
    y = ctx.y
    eval_months = list(y.index[-n_points:])
    origins: dict[pd.Timestamp, list[tuple[pd.Timestamp, int]]] = {}
    for e in eval_months:
        for hz in C.HORIZONS:
            origin = shift_months(e, -(hz - 1))
            train_end = shift_months(origin, -1)
            if len(y.loc[:train_end]) < C.MIN_TRAIN_OBS:
                continue
            origins.setdefault(origin, []).append((e, hz))
    rows: dict[str, list[dict]] = {m: [] for m in methods}
    for origin in sorted(origins):
        entries = origins[origin]
        train_end = shift_months(origin, -1)
        last_eval = max(e for e, _ in entries)
        months = pd.date_range(origin, last_eval, freq="MS")
        for m in methods:
            try:
                out = run_method(m, ctx, train_end, months, "backtest")
            except Unavailable:
                continue
            for e, hz in entries:
                rows[m].append({"Eval_Date": e, "Horizon": hz, "Actual": float(y.loc[e]),
                                "Forecast": float(out.forecast[hz - 1])})
    return {m: pd.DataFrame(r, columns=BT_COLUMNS) for m, r in rows.items()}


def summarize(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"WAPE_1m": np.nan, "WAPE_3m": np.nan, "WAPE_pooled": np.nan, "Points": 0}
    h1, h3 = df[df["Horizon"] == 1], df[df["Horizon"] == 3]
    return {"WAPE_1m": wape(h1["Actual"], h1["Forecast"]), "WAPE_3m": wape(h3["Actual"], h3["Forecast"]),
            "WAPE_pooled": wape(df["Actual"], df["Forecast"]), "Points": len(df)}


def select_method(bt: dict[str, pd.DataFrame], candidates: list[str]) -> str | None:
    frames = {m: bt[m].set_index(["Eval_Date", "Horizon"]) for m in candidates if m in bt and not bt[m].empty}
    if not frames:
        return None
    common = None
    for f in frames.values():
        common = f.index if common is None else common.intersection(f.index)
    scores = {}
    for m, f in frames.items():
        sub = f.loc[common] if common is not None and len(common) >= 3 else f
        scores[m] = wape(sub["Actual"], sub["Forecast"])
    return min(scores, key=lambda m: scores[m] if np.isfinite(scores[m]) else np.inf)


def interval_widths(h: int, rmse1: float, rmse3: float) -> np.ndarray:
    if not np.isfinite(rmse1) and not np.isfinite(rmse3):
        return np.full(h, np.nan)
    r1 = rmse1 if np.isfinite(rmse1) else rmse3 / np.sqrt(3)
    r3 = rmse3 if np.isfinite(rmse3) else rmse1 * np.sqrt(3)
    widths = []
    for i in range(1, h + 1):
        if i == 1:
            r = r1
        elif i == 2:
            r = (r1 + r3) / 2
        else:
            r = r3 * np.sqrt(i / 3)
        widths.append(C.INTERVAL_Z * r)
    return np.asarray(widths, float)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def _prepare_history(y_hist: pd.Series, cutoff: pd.Timestamp) -> tuple[pd.Series, list[str]]:
    notes = []
    if y_hist.empty:
        return y_hist, notes
    idx = pd.date_range(y_hist.index.min(), cutoff, freq="MS")
    y = y_hist.reindex(idx)
    last_valid = y.last_valid_index()
    inner_missing = int(y.loc[:last_valid].isna().sum())
    if inner_missing:
        notes.append(f"{inner_missing} missing month(s) filled by interpolation")
    y = y.loc[:last_valid].interpolate(limit_area="inside").reindex(idx)
    if last_valid < cutoff:
        y.loc[y.index > last_valid] = 0.0
        notes.append(f"no actuals after {last_valid.strftime('%b-%y')}; later months treated as 0")
    return y.astype(float), notes


def _pivot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(index="Date", columns=["Plant", "Product_Line", "Bucket"], values="Value",
                          aggfunc="sum").sort_index()


def run_models(inputs: Inputs, settings: Settings) -> Results:
    t0 = time.perf_counter()
    year, at = int(settings.year), int(settings.actuals_through)
    cutoff = month_start(year, at)
    year_months = pd.date_range(month_start(year, 1), periods=12, freq="MS")
    forecast_months = year_months[year_months > cutoff]

    spend = inputs.spend
    act_p = _pivot(spend[spend["Version"] == C.ACT])
    in_year = spend["Date"].dt.year == year
    aop_p = _pivot(spend[(spend["Version"] == C.AOP) & in_year])
    ppf_p = _pivot(spend[(spend["Version"] == C.PPF) & in_year])
    keys = sorted(set(act_p.columns) | set(aop_p.columns))

    drv_act_map, drv_plan_map, basis_map = build_driver_frames(inputs.drivers)

    events_map = {}
    if inputs.events is not None and not inputs.events.empty:
        for (plant, mini), grp in inputs.events.groupby(["Plant", "Product_Line"]):
            events_map[(plant, mini)] = grp.groupby("Date")["Amount"].sum().sort_index()

    overrides: dict[tuple, dict] = {}
    for plant, mini, bucket, ym, flag in settings.outlier_overrides:
        overrides.setdefault((plant, mini, bucket), {})[pd.Timestamp(f"{ym}-01")] = bool(flag)
    method_over = dict(settings.method_overrides)
    fx = FxTable(inputs.fx, year)

    series: dict[tuple, SeriesResult] = {}
    class_rows, outlier_rows = [], []
    for key in keys:
        plant, mini, bucket = key
        currency = inputs.plant_currency.get(plant, "USD")
        y_full = act_p[key].dropna() if key in act_p.columns else pd.Series(dtype=float)
        aop_s = aop_p[key].reindex(year_months) if key in aop_p.columns else pd.Series(np.nan, index=year_months)
        ppf_s = ppf_p[key].reindex(year_months) if key in ppf_p.columns else pd.Series(np.nan, index=year_months)
        y_hist, notes = _prepare_history(y_full.loc[:cutoff], cutoff)
        later = y_full[y_full.index > cutoff]
        events = events_map.get((plant, mini), pd.Series(dtype=float)) if bucket == C.MAINTENANCE else pd.Series(dtype=float)
        ctx = SeriesContext(
            plant=plant, mini=mini, bucket=bucket, currency=currency, y=y_hist, later=later,
            drv_act=drv_act_map.get((plant, mini), pd.DataFrame()),
            drv_plan=drv_plan_map.get((plant, mini), pd.DataFrame()),
            events=events, aop=aop_s, ppf=ppf_s, year=year, cutoff=cutoff,
            forecast_months=forecast_months, outlier_z=settings.outlier_z,
            overrides=overrides.get(key, {}), outlier_min_dev=settings.outlier_min_dev,
            basis=basis_map.get((plant, mini)),
        )
        ctx.detect_planned()
        sr, flags = _run_series(ctx, settings, method_over, notes)
        series[key] = sr

        class_rows.append({
            "Plant": plant, "Product_Line": mini, "Bucket": bucket, "Currency": currency,
            "Behavior": sr.behavior.get("Behavior"), "Variable_Share": sr.behavior.get("Variable_Share"),
            "Fixed_per_Month": sr.behavior.get("Fixed_per_Month"),
            "Rate_per_Activity": sr.behavior.get("Rate_per_Activity"), "Activity_Basis": ctx.basis or "none",
            "R2": sr.behavior.get("R2"), "P_Value": sr.behavior.get("P_Value"),
            "Forecastability": sr.forecastability.get("Segment"), "CV": sr.forecastability.get("CV"),
            "Naive_WAPE_1m": sr.forecastability.get("Naive_WAPE_1m"),
            "Zero_Share": sr.forecastability.get("Zero_Share"),
            "Atypical_Excluded": int(flags["Excluded"].sum()) if flags is not None and "Excluded" in flags else 0,
            "Default_Method": C.DEFAULT_METHOD.get(bucket, C.M_AUTO), "Method_Used": sr.method,
        })
        if flags is not None and not flags.empty:
            shown = flags[flags["Flag"] | flags["Excluded"]]
            for ts, row in shown.iterrows():
                outlier_rows.append({
                    "Plant": plant, "Product_Line": mini, "Bucket": bucket, "Month": ts.strftime("%Y-%m"),
                    "Value": row["Value"], "Baseline": row["Baseline"], "Z": row["Z"],
                    "Reason": row["Reason"] or "Forced by user", "Auto_Flag": bool(row["Flag"]),
                    "Exclude": bool(row["Excluded"]),
                })

    classification = pd.DataFrame(class_rows)
    outliers = pd.DataFrame(outlier_rows, columns=["Plant", "Product_Line", "Bucket", "Month", "Value", "Baseline",
                                                   "Z", "Reason", "Auto_Flag", "Exclude"])
    activity = _activity_table(drv_act_map, drv_plan_map, basis_map, year_months, cutoff)
    acc_series = _series_accuracy_table(series)
    acc_plant = _aggregate_accuracy(series, fx, by_plant=True)
    acc_network = _aggregate_accuracy(series, fx, by_plant=False)
    return Results(year=year, actuals_through=at, cutoff=cutoff, year_months=year_months,
                   forecast_months=forecast_months, series=series, plant_currency=inputs.plant_currency,
                   classification=classification, outliers=outliers, accuracy_series=acc_series,
                   accuracy_plant=acc_plant, accuracy_network=acc_network, settings=settings,
                   elapsed_s=time.perf_counter() - t0, activity=activity)


def _empty_result(ctx: SeriesContext, method: str, note: str, forecast: np.ndarray) -> SeriesResult:
    fm = ctx.forecast_months
    nan = pd.Series(np.nan, index=fm, dtype=float)
    return SeriesResult(
        plant=ctx.plant, mini=ctx.mini, bucket=ctx.bucket, currency=ctx.currency, requested=method,
        method=method, note=note, forecast=pd.Series(forecast, index=fm, dtype=float), lower=nan, upper=nan,
        kind=None, components=None, history=ctx.y, later=ctx.later, aop=ctx.aop, ppf=ctx.ppf,
        backtest=pd.DataFrame(columns=BT_COLUMNS + ["Naive"]), candidates=pd.DataFrame(),
        metrics={"Accuracy_1m": np.nan, "Accuracy_3m": np.nan, "Bias": np.nan, "FVA_1m": np.nan,
                 "FVA_3m": np.nan, "RMSE_1m": np.nan, "RMSE_3m": np.nan, "Points": 0},
        behavior={"Behavior": "n/a"}, forecastability={"Segment": "n/a"},
    )


def _run_series(ctx: SeriesContext, settings: Settings, method_over: dict, notes: list[str]):
    fm = ctx.forecast_months
    h = len(fm)
    if ctx.y.empty:
        aop_f = ctx.aop.reindex(fm).fillna(0.0).to_numpy(float)
        return _empty_result(ctx, "AOP (no actuals)", "No actual history; AOP used for the remaining months.", aop_f), None
    if ctx.y.iloc[-12:].abs().sum() == 0:
        return _empty_result(ctx, "Inactive", "No spend in the last 12 closed months; forecast set to 0.",
                             np.zeros(h)), None
    if len(ctx.y) < C.MIN_HISTORY_FOR_MODELS:
        return _empty_result(ctx, C.M_RR3, f"Only {len(ctx.y)} months of history; 3-month average used.",
                             runrate_mean(ctx.y, h, 3)), None

    y_clean_full, flags = clean_series(ctx.y, ctx.outlier_z, ctx.overrides, ctx.outlier_min_dev)
    if ctx.bucket == C.MAINTENANCE and len(ctx.events):
        base = ctx.y - ctx.events.reindex(ctx.y.index).fillna(0.0)
        _, flags = clean_series(base, ctx.outlier_z, ctx.overrides, ctx.outlier_min_dev)
    fcast = forecastability(y_clean_full)
    activity_hist = ctx.driver_hist(C.ACTIVITY, ctx.cutoff)
    behavior = cost_behavior(y_clean_full, activity_hist)

    requested_raw = method_over.get(ctx.bucket, C.M_DEFAULT)
    requested = C.DEFAULT_METHOD.get(ctx.bucket, C.M_AUTO) if requested_raw == C.M_DEFAULT else requested_raw
    effective = requested
    final_out: MethodOutput | None = None
    if requested != C.M_AUTO and h > 0:
        try:
            final_out = run_method(requested, ctx, ctx.cutoff, fm, "final")
        except Unavailable as exc:
            notes.append(f"{requested} not available ({exc}); Auto used")
            effective = C.M_AUTO

    if effective == C.M_AUTO:
        candidates = [C.M_ENSEMBLE, C.M_NAIVE, C.M_SNAIVE]
        if behavior.get("Behavior") in ("Variable", "Semi-variable") and ctx.planned.get(C.ACTIVITY):
            candidates.append(C.M_REG)
    else:
        candidates = [effective]
    methods = list(dict.fromkeys(candidates + [C.M_NAIVE]))
    bt = backtest(ctx, methods, settings.backtest_points)

    if effective == C.M_AUTO:
        chosen = select_method(bt, candidates) or C.M_ENSEMBLE
    else:
        chosen = effective
        if h == 0 and bt.get(chosen, pd.DataFrame()).empty:
            notes.append(f"{chosen} could not be backtested")

    if h > 0 and (final_out is None or chosen != requested):
        for method in [chosen, C.M_ENSEMBLE, C.M_SNAIVE, C.M_NAIVE]:
            try:
                final_out = run_method(method, ctx, ctx.cutoff, fm, "final")
                if method != chosen:
                    notes.append(f"{chosen} failed in the final run; {method} used")
                    chosen = method
                break
            except Unavailable:
                continue
    if final_out is None:
        final_out = MethodOutput(np.zeros(h))

    bt_chosen = bt.get(chosen, pd.DataFrame(columns=BT_COLUMNS))
    bt_naive = bt.get(C.M_NAIVE, pd.DataFrame(columns=BT_COLUMNS))
    merged = bt_chosen.merge(bt_naive[["Eval_Date", "Horizon", "Forecast"]].rename(columns={"Forecast": "Naive"}),
                             on=["Eval_Date", "Horizon"], how="left")
    h1, h3 = merged[merged["Horizon"] == 1], merged[merged["Horizon"] == 3]
    acc1, acc3 = accuracy(h1["Actual"], h1["Forecast"]), accuracy(h3["Actual"], h3["Forecast"])
    metrics = {
        "Accuracy_1m": acc1, "Accuracy_3m": acc3,
        "Bias": bias(merged["Actual"], merged["Forecast"]),
        "FVA_1m": acc1 - accuracy(h1["Actual"], h1["Naive"]) if len(h1) else np.nan,
        "FVA_3m": acc3 - accuracy(h3["Actual"], h3["Naive"]) if len(h3) else np.nan,
        "RMSE_1m": rmse(h1["Actual"], h1["Forecast"]), "RMSE_3m": rmse(h3["Actual"], h3["Forecast"]),
        "Points": len(merged),
    }
    cand_rows = []
    for m in methods:
        s = summarize(bt.get(m))
        cand_rows.append({"Method": m, **s, "Candidate": m in candidates, "Selected": m == chosen})
    candidates_df = pd.DataFrame(cand_rows)

    fc = np.asarray(final_out.forecast, float)
    widths = interval_widths(h, metrics["RMSE_1m"], metrics["RMSE_3m"])
    note_parts = [final_out.note] if final_out.note else []
    if effective == C.M_AUTO:
        note_parts.insert(0, f"Auto selected {chosen} (lowest backtest WAPE)")
    note_parts += notes
    sr = SeriesResult(
        plant=ctx.plant, mini=ctx.mini, bucket=ctx.bucket, currency=ctx.currency, requested=requested,
        method=chosen, note="; ".join(p for p in note_parts if p),
        forecast=pd.Series(fc, index=fm, dtype=float),
        lower=pd.Series(fc - widths, index=fm, dtype=float), upper=pd.Series(fc + widths, index=fm, dtype=float),
        kind=final_out.kind, components=final_out.components, history=ctx.y, later=ctx.later,
        aop=ctx.aop, ppf=ctx.ppf, backtest=merged, candidates=candidates_df, metrics=metrics,
        behavior=behavior, forecastability=fcast,
        planned_drivers=sorted(k for k, v in ctx.planned.items() if v),
    )
    return sr, flags


# ---------------------------------------------------------------------------
# Accuracy tables
# ---------------------------------------------------------------------------
def _series_accuracy_table(series: dict) -> pd.DataFrame:
    rows = []
    for sr in series.values():
        m = sr.metrics
        rows.append({"Plant": sr.plant, "Product_Line": sr.mini, "Bucket": sr.bucket, "Method": sr.method,
                     "Accuracy_1m": m["Accuracy_1m"], "Accuracy_3m": m["Accuracy_3m"], "Bias": m["Bias"],
                     "FVA_1m": m["FVA_1m"], "FVA_3m": m["FVA_3m"], "Points": m["Points"]})
    return pd.DataFrame(rows)


def _aggregate_accuracy(series: dict, fx: FxTable, by_plant: bool) -> pd.DataFrame:
    groups: dict[str, list[SeriesResult]] = {}
    for sr in series.values():
        if sr.backtest is None or sr.backtest.empty:
            continue
        if not by_plant and not fx.has(sr.currency):
            continue
        groups.setdefault(sr.plant if by_plant else "Network (USD)", []).append(sr)
    rows = []
    for name, members in groups.items():
        frames = []
        for i, sr in enumerate(members):
            f = sr.backtest[["Eval_Date", "Horizon", "Actual", "Forecast", "Naive"]].copy()
            if not by_plant:
                rate = fx.rates_for(sr.currency, f["Eval_Date"])
                for col in ("Actual", "Forecast", "Naive"):
                    f[col] = f[col] * rate
            f["_sid"] = i
            frames.append(f)
        allf = pd.concat(frames, ignore_index=True)
        g = allf.groupby(["Eval_Date", "Horizon"]).agg(n=("_sid", "nunique"), Actual=("Actual", "sum"),
                                                        Forecast=("Forecast", "sum"), Naive=("Naive", "sum"))
        g = g[g["n"] == len(members)].reset_index()
        row = {"Scope": name, "Series": len(members)}
        for hz in C.HORIZONS:
            sub = g[g["Horizon"] == hz]
            acc = accuracy(sub["Actual"], sub["Forecast"])
            row[f"Accuracy_{hz}m"] = acc
            row[f"FVA_{hz}m"] = acc - accuracy(sub["Actual"], sub["Naive"]) if len(sub) else np.nan
            row[f"Points_{hz}m"] = len(sub)
        row["Bias"] = bias(g["Actual"], g["Forecast"])
        rows.append(row)
    cols = ["Scope", "Series", "Accuracy_1m", "Accuracy_3m", "Bias", "FVA_1m", "FVA_3m", "Points_1m", "Points_3m"]
    return pd.DataFrame(rows, columns=cols)


def _activity_table(act_map: dict, plan_map: dict, basis_map: dict, year_months: pd.DatetimeIndex,
                    cutoff: pd.Timestamp) -> pd.DataFrame:
    """Plant-level activity for the forecast year: actual through the cutoff, plan afterwards."""
    rows = []
    for (plant, line), act in act_map.items():
        if line != "" or C.ACTIVITY not in act.columns:
            continue
        plan = plan_map.get((plant, line), pd.DataFrame())
        plan_s = plan[C.ACTIVITY] if C.ACTIVITY in plan.columns else pd.Series(dtype=float)
        for d in year_months:
            if d <= cutoff:
                value, source = act[C.ACTIVITY].get(d, np.nan), "Act"
            else:
                value, source = plan_s.get(d, np.nan), "Plan"
            rows.append({"Plant": plant, "Date": d, "Activity": value, "Source": source,
                         "Basis": basis_map.get((plant, line))})
    return pd.DataFrame(rows, columns=["Plant", "Date", "Activity", "Source", "Basis"])
