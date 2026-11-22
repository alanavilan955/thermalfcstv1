"""Prepare driver frames: derive earned hours and roll product lines up to plant level.

Earned_Hours = Units x Std_Hours_per_Unit when earned hours are not loaded. Plant-level frames are
built from the product lines when the plant has no plant-level drivers for a given name; drivers
loaded at plant level (for example, utility meters) always take precedence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import sc_config as C
from sc_io_excel import is_rate_driver

BASIS_EARNED = "earned hours"
BASIS_UNITS = "units"


def _pivot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(index="Date", columns="Driver", values="Value", aggfunc="mean").sort_index()


def _add_earned(frame: pd.DataFrame, std_fallback: float | None) -> pd.DataFrame:
    """Earned_Hours = Units x Std_Hours_per_Unit wherever earned hours are missing."""
    if frame.empty or C.UNITS not in frame.columns:
        return frame
    f = frame.copy()
    std = f[C.STD_HOURS] if C.STD_HOURS in f.columns else pd.Series(np.nan, index=f.index)
    if std_fallback is not None and np.isfinite(std_fallback):
        std = std.ffill().fillna(std_fallback)
    derived = f[C.UNITS] * std
    if C.EARNED_HOURS in f.columns:
        f[C.EARNED_HOURS] = f[C.EARNED_HOURS].combine_first(derived)
    elif derived.notna().any():
        f[C.EARNED_HOURS] = derived
    return f


def _weighted(values: list[pd.Series], weights: list[pd.Series | None], idx: pd.Index) -> pd.Series:
    v = pd.concat([s.reindex(idx) for s in values], axis=1)
    if all(w is not None for w in weights):
        w = pd.concat([s.reindex(idx) for s in weights], axis=1).where(v.notna())
        w.columns = v.columns
        num = (v * w).sum(axis=1, min_count=1)
        den = w.sum(axis=1, min_count=1)
        out = num / den.replace(0, np.nan)
        return out.where(out.notna(), v.mean(axis=1))
    return v.mean(axis=1)


def _rollup(frames: list[pd.DataFrame], required: dict[str, int] | None = None) -> pd.DataFrame:
    """Sum quantities and weight-average rates across product-line frames of one version.

    `required` gives, per quantity driver, how many lines must report a value for a month to be kept
    (used for plans, so a missing line plan does not understate the plant).
    """
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    idx = frames[0].index
    for f in frames[1:]:
        idx = idx.union(f.index)
    cols = []
    for f in frames:
        cols += [c for c in f.columns if c not in cols]
    out = pd.DataFrame(index=idx)
    for col in cols:
        members = [f for f in frames if col in f.columns]
        series = [f[col] for f in members]
        if col == C.STD_HOURS:
            continue
        if col in C.QUANTITY_DRIVERS or col.startswith(C.UTIL_UNITS):
            need = (required or {}).get(col, 1)
            stacked = pd.concat([s.reindex(idx) for s in series], axis=1)
            out[col] = stacked.sum(axis=1, min_count=max(1, need)) if len(series) >= need else np.nan
        elif col == C.WAGE:
            weights = [f[C.LABOR_HOURS] if C.LABOR_HOURS in f.columns else
                       (f[C.EARNED_HOURS] if C.EARNED_HOURS in f.columns else None) for f in members]
            out[col] = _weighted(series, weights, idx)
        elif col.startswith(C.UTIL_TARIFF):
            units_col = C.UTIL_UNITS + col[len(C.UTIL_TARIFF):]
            weights = [f[units_col] if units_col in f.columns else None for f in members]
            out[col] = _weighted(series, weights, idx)
        elif col == C.EFFICIENCY:
            weights = [f[C.EARNED_HOURS] if C.EARNED_HOURS in f.columns else None for f in members]
            out[col] = _weighted(series, weights, idx)
        elif is_rate_driver(col):
            out[col] = _weighted(series, [None] * len(series), idx)
        else:
            out[col] = pd.concat([s.reindex(idx) for s in series], axis=1).sum(axis=1, min_count=1)
    return out


def _active_lines(act_frames: dict[str, pd.DataFrame], col: str) -> int:
    """Lines with actual values for `col` in their last 3 months of history."""
    count = 0
    for f in act_frames.values():
        if col in f.columns and f[col].dropna().size:
            last = f.index.max()
            recent = f.loc[f.index > last - pd.DateOffset(months=3), col]
            if recent.notna().any():
                count += 1
    return count


def build_driver_frames(drivers: pd.DataFrame | None) -> tuple[dict, dict, dict]:
    """Return (actual frames, plan frames, activity basis), keyed by (plant, product line)."""
    act_map: dict[tuple, pd.DataFrame] = {}
    plan_map: dict[tuple, pd.DataFrame] = {}
    basis_map: dict[tuple, str | None] = {}
    if drivers is None or drivers.empty:
        return act_map, plan_map, basis_map
    for (plant, line), grp in drivers.groupby(["Plant", "Product_Line"]):
        act = _pivot(grp[grp["Version"] == C.DRIVER_ACT])
        plan = _pivot(grp[grp["Version"] == C.DRIVER_PLAN])
        aop = _pivot(grp[grp["Version"] == C.DRIVER_AOP])
        if not aop.empty:
            plan = plan.combine_first(aop) if not plan.empty else aop
        std_last = None
        if not act.empty and C.STD_HOURS in act.columns and act[C.STD_HOURS].notna().any():
            std_last = float(act[C.STD_HOURS].dropna().iloc[-1])
        act_map[(plant, line)] = _add_earned(act, None)
        plan_map[(plant, line)] = _add_earned(plan, std_last)

    for plant in sorted({p for p, _ in act_map}):
        lines = sorted(line for p, line in act_map if p == plant and line != "")
        if not lines:
            continue
        line_act = {line: act_map[(plant, line)] for line in lines}
        rolled_act = _rollup(list(line_act.values()))
        all_earned = all(C.EARNED_HOURS in f.columns and f[C.EARNED_HOURS].notna().any() for f in line_act.values())
        plan_frames = [plan_map[(plant, line)] for line in lines]
        quantity_cols = {c for f in plan_frames for c in f.columns
                         if c in C.QUANTITY_DRIVERS or str(c).startswith(C.UTIL_UNITS)}
        required = {c: _active_lines(line_act, c) for c in quantity_cols}
        rolled_plan = _rollup(plan_frames, required)
        if not all_earned:
            rolled_act = rolled_act.drop(columns=[C.EARNED_HOURS], errors="ignore")
            rolled_plan = rolled_plan.drop(columns=[C.EARNED_HOURS], errors="ignore")
        given_act = act_map.get((plant, ""))
        given_plan = plan_map.get((plant, ""))
        act_map[(plant, "")] = (given_act.combine_first(rolled_act)
                                if given_act is not None and not given_act.empty else rolled_act)
        plan_map[(plant, "")] = (given_plan.combine_first(rolled_plan)
                                 if given_plan is not None and not given_plan.empty else rolled_plan)

    for key, act in act_map.items():
        plan = plan_map.get(key, pd.DataFrame())
        basis, col = None, None
        if C.EARNED_HOURS in act.columns and act[C.EARNED_HOURS].notna().any():
            basis, col = BASIS_EARNED, C.EARNED_HOURS
        elif C.UNITS in act.columns and act[C.UNITS].notna().any():
            basis, col = BASIS_UNITS, C.UNITS
        if basis:
            act = act.copy()
            act[C.ACTIVITY] = act[col]
            act_map[key] = act
            if not plan.empty and col in plan.columns:
                plan = plan.copy()
                plan[C.ACTIVITY] = plan[col]
                plan_map[key] = plan
        basis_map[key] = basis
    return act_map, plan_map, basis_map
