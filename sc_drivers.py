"""Driver formulas for Direct Labor, Utilities and Maintenance, and scenario recalculation.

Activity is measured in earned standard hours (units x standard hours per unit). When standards are
not loaded, units are used instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import sc_config as C
from sc_classify import clean_series
from sc_context import MethodOutput, SeriesContext, Unavailable
from sc_models import runrate_median


def _finite(a) -> bool:
    return bool(np.all(np.isfinite(np.asarray(a, dtype=float))))


def _actual_series(ctx: SeriesContext, name: str) -> pd.Series:
    return ctx.drv_act[name] if name in ctx.drv_act.columns else pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Direct labor
# ---------------------------------------------------------------------------
def labor(ctx: SeriesContext, train_end: pd.Timestamp, months: pd.DatetimeIndex, mode: str) -> MethodOutput:
    """Labor $ = earned hours / labor efficiency x wage rate x calibration factor.

    Labor efficiency = earned hours / actual direct labor hours. When actual labor hours are not
    loaded, the calibration factor becomes a $ per earned hour rate.
    """
    h = len(months)
    activity_h = ctx.driver_hist(C.ACTIVITY, train_end)
    if activity_h is None:
        raise Unavailable("no Units or Earned_Hours actuals")
    activity_f = ctx.driver_future(C.ACTIVITY, months, mode, train_end)
    if not _finite(activity_f):
        raise Unavailable("build plan (Units or Earned_Hours) missing for some forecast months")

    labor_h = ctx.driver_hist(C.LABOR_HOURS, train_end)
    uses_hours = labor_h is not None and len(labor_h) >= 6
    if uses_hours:
        eff_h = (activity_h / labor_h).replace([np.inf, -np.inf], np.nan).dropna()
        if ctx.planned.get(C.LABOR_HOURS):
            hours_f = (_actual_series(ctx, C.LABOR_HOURS).reindex(months).to_numpy(float) if mode == "backtest"
                       else ctx.driver_future(C.LABOR_HOURS, months, mode, train_end))
            eff_f = activity_f / hours_f
        else:
            if ctx.planned.get(C.EFFICIENCY):
                if mode == "backtest":
                    eff_all = (_actual_series(ctx, C.ACTIVITY) / _actual_series(ctx, C.LABOR_HOURS))
                    eff_f = eff_all.reindex(months).to_numpy(float)
                else:
                    eff_f = ctx.driver_future(C.EFFICIENCY, months, mode, train_end)
            else:
                eff_f = ctx.stat("efficiency", eff_h, h, mode)
            if not _finite(eff_f) or np.any(np.asarray(eff_f) <= 0):
                raise Unavailable("labor efficiency forecast is not valid")
            hours_f = activity_f / eff_f
        base_h = labor_h
    else:
        eff_f = np.full(h, np.nan)
        hours_f = np.asarray(activity_f, float)
        base_h = activity_h
    if not _finite(hours_f):
        raise Unavailable("labor hours could not be derived for some forecast months")

    wage_h = ctx.driver_hist(C.WAGE, train_end)
    if wage_h is not None:
        wage_f = (ctx.driver_future(C.WAGE, months, mode, train_end) if ctx.planned.get(C.WAGE)
                  else np.repeat(wage_h.iloc[-1], h))
        rate_h = wage_h
    else:
        wage_f = np.ones(h)
        rate_h = pd.Series(1.0, index=base_h.index)
    if not _finite(wage_f):
        raise Unavailable("wage rate missing for some forecast months")

    formula_h = (base_h * rate_h).dropna()
    y_c = ctx.y_clean(train_end)
    df = pd.concat({"y": y_c, "f": formula_h}, axis=1).dropna().iloc[-C.CALIBRATION_MONTHS:]
    if df.empty or df["f"].sum() == 0:
        raise Unavailable("labor formula cannot be calibrated (no overlap between spend and drivers)")
    k = float(df["y"].sum() / df["f"].sum())
    fc = np.asarray(hours_f, float) * np.asarray(wage_f, float) * k
    if uses_hours and wage_h is not None:
        note = "Labor = earned hours / labor efficiency x wage rate x calibration factor"
    elif uses_hours:
        note = "Labor = earned hours / labor efficiency x calibrated $ per labor hour (no wage data)"
    else:
        note = "Labor = earned hours x calibrated $ per earned hour (no labor hours loaded)"
    comps = {"activity": np.asarray(activity_f, float), "hours": np.asarray(hours_f, float),
             "eff": np.asarray(eff_f, float), "wage": np.asarray(wage_f, float), "k": k,
             "uses_hours": bool(uses_hours), "has_wage": wage_h is not None}
    return MethodOutput(fc, "labor", comps, note)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def utilities(ctx: SeriesContext, train_end: pd.Timestamp, months: pd.DatetimeIndex, mode: str) -> MethodOutput:
    """Utilities $ = sum over types of consumption x tariff, plus fixed demand charges.

    Consumption = consumption per earned hour (forecast) x earned hours of the plan, unless the
    consumption itself is planned.
    """
    if not ctx.util_types:
        raise Unavailable("no Util_Units drivers")
    h = len(months)
    activity_h = ctx.driver_hist(C.ACTIVITY, train_end)
    activity_f = None
    units_all, tariff_all, labels = [], [], []
    var_h = None
    all_tariffs = True
    for suffix in ctx.util_types:
        units_name, tariff_name = C.UTIL_UNITS + suffix, C.UTIL_TARIFF + suffix
        units_h = ctx.driver_hist(units_name, train_end)
        if units_h is None or len(units_h) < 6:
            continue
        if ctx.planned.get(units_name):
            units_f = ctx.driver_future(units_name, months, mode, train_end)
        else:
            if activity_h is None:
                raise Unavailable("Units or Earned_Hours needed to forecast utility consumption")
            if activity_f is None:
                activity_f = ctx.driver_future(C.ACTIVITY, months, mode, train_end)
                if not _finite(activity_f):
                    raise Unavailable("build plan missing for some forecast months")
            intensity_h = (units_h / activity_h).replace([np.inf, -np.inf], np.nan).dropna()
            intensity_f = ctx.stat("intensity" + suffix, intensity_h, h, mode)
            units_f = intensity_f * activity_f
        tariff_h = ctx.driver_hist(tariff_name, train_end)
        if tariff_h is not None:
            tariff_f = (ctx.driver_future(tariff_name, months, mode, train_end) if ctx.planned.get(tariff_name)
                        else np.repeat(tariff_h.iloc[-1], h))
            contrib_h = units_h * tariff_h
        else:
            all_tariffs = False
            tariff_f = np.ones(h)
            contrib_h = units_h
        if not _finite(units_f) or not _finite(tariff_f):
            raise Unavailable(f"consumption or tariff values missing for {suffix.strip('_') or 'utilities'}")
        var_h = contrib_h if var_h is None else var_h + contrib_h
        units_all.append(np.asarray(units_f, float))
        tariff_all.append(np.asarray(tariff_f, float))
        labels.append(suffix.strip("_") or "Utilities")
    if not units_all:
        raise Unavailable("not enough utility consumption history")

    y_c = ctx.y_clean(train_end)
    var_f = sum(u * t for u, t in zip(units_all, tariff_all))
    if all_tariffs:
        resid = (y_c - var_h).dropna().iloc[-C.CALIBRATION_MONTHS:]
        if resid.empty:
            raise Unavailable("utility formula cannot be calibrated")
        fixed = float(resid.median())
        fc = var_f + fixed
        comps = {"mode": "additive", "units": units_all, "tariff": tariff_all, "fixed": fixed, "k": 1.0,
                 "types": labels}
        note = ("Utilities = consumption per earned hour x earned hours x tariff, plus fixed charges "
                "(median of last 6 months)")
    else:
        df = pd.concat({"y": y_c, "f": var_h}, axis=1).dropna().iloc[-C.CALIBRATION_MONTHS:]
        if df.empty or df["f"].sum() == 0:
            raise Unavailable("utility formula cannot be calibrated")
        k = float(df["y"].sum() / df["f"].sum())
        fc = var_f * k
        comps = {"mode": "multiplicative", "units": units_all, "tariff": tariff_all, "fixed": 0.0, "k": k,
                 "types": labels}
        note = "Utilities = consumption x calibrated $ per unit (tariffs not loaded)"
    return MethodOutput(np.asarray(fc, float), "utilities", comps, note)


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------
def maintenance(ctx: SeriesContext, train_end: pd.Timestamp, months: pd.DatetimeIndex, mode: str) -> MethodOutput:
    """Maintenance $ = unplanned base (actual minus events, atypical months removed) + planned events."""
    h = len(months)
    y_raw = ctx.y_raw(train_end)
    has_events = ctx.events is not None and len(ctx.events) > 0
    events_h = ctx.events.reindex(y_raw.index).fillna(0.0) if has_events else pd.Series(0.0, index=y_raw.index)
    base_raw = y_raw - events_h
    base_clean, _ = clean_series(base_raw, ctx.outlier_z, ctx.overrides, ctx.outlier_min_dev)
    if len(base_clean.dropna()) >= C.MIN_TRAIN_OBS:
        base_f = ctx.stat("maintenance_base", base_clean, h, mode)
    else:
        base_f = runrate_median(base_clean, h)
    events_f = ctx.events.reindex(months).fillna(0.0).to_numpy(float) if has_events else np.zeros(h)
    fc = np.asarray(base_f, float) + events_f
    note = ("Maintenance = unplanned base (ETS+ARIMA on actual minus events, atypical months removed) "
            "+ planned events")
    if not has_events:
        note += " (no events loaded for this plant or line)"
    return MethodOutput(fc, "maintenance", {"base": np.asarray(base_f, float), "events": events_f}, note)


DRIVER_FUNCTIONS = {
    C.DIRECT_LABOR: labor,
    C.UTILITIES: utilities,
    C.MAINTENANCE: maintenance,
}


# ---------------------------------------------------------------------------
# Scenario recalculation
# ---------------------------------------------------------------------------
def recompute(kind: str | None, comps: dict | None, scenario) -> np.ndarray | None:
    """Recalculate a driver or regression forecast under scenario multipliers.

    Volume scales earned hours (and consumption); efficiency divides labor hours; wage and tariff
    scale rates. Returns None when the method is not affected by scenarios.
    """
    if not kind or not comps:
        return None
    v = 1.0 + scenario.volume
    if kind == "labor":
        hours = comps["hours"] * v / (1.0 + scenario.efficiency)
        wage = comps["wage"] * (1.0 + scenario.wage)
        return hours * wage * comps["k"]
    if kind == "utilities":
        var = sum(u * v * t * (1.0 + scenario.tariff) for u, t in zip(comps["units"], comps["tariff"]))
        if comps["mode"] == "additive":
            return var + comps["fixed"]
        return var * comps["k"]
    if kind == "regression":
        return comps["a"] + comps["b"] * comps["activity"] * v
    return None
