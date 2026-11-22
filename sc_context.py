"""Per-series context shared by the forecasting methods."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sc_classify import clean_series
from sc_io_excel import is_rate_driver
from sc_models import StatEngine


class Unavailable(Exception):
    """Raised when a method cannot run for a series (for example, missing drivers)."""


@dataclass
class MethodOutput:
    forecast: np.ndarray
    kind: str | None = None
    components: dict | None = None
    note: str = ""


@dataclass
class SeriesContext:
    plant: str
    mini: str
    bucket: str
    currency: str
    y: pd.Series                 # raw actuals through the cutoff (gaps filled)
    later: pd.Series             # actuals after the cutoff (display only)
    drv_act: pd.DataFrame        # driver actuals, index = month
    drv_plan: pd.DataFrame       # driver plan (AOP drivers fill gaps), index = month
    events: pd.Series            # planned maintenance events by month
    aop: pd.Series               # AOP for the forecast year
    ppf: pd.Series               # prior forecast for the forecast year
    year: int
    cutoff: pd.Timestamp
    forecast_months: pd.DatetimeIndex
    outlier_z: float
    overrides: dict
    outlier_min_dev: float = 0.15
    basis: str | None = None
    planned: dict = field(default_factory=dict)
    engines: dict = field(default_factory=dict)
    util_types: list = field(default_factory=list)
    clean_cache: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.plant}{' / ' + self.mini if self.mini else ''} / {self.bucket}"

    # -- actual series ---------------------------------------------------------
    def y_raw(self, train_end: pd.Timestamp) -> pd.Series:
        return self.y.loc[:train_end]

    def y_clean(self, train_end: pd.Timestamp) -> pd.Series:
        if train_end not in self.clean_cache:
            cleaned, _ = clean_series(self.y_raw(train_end), self.outlier_z, self.overrides, self.outlier_min_dev)
            self.clean_cache[train_end] = cleaned
        return self.clean_cache[train_end]

    # -- drivers ----------------------------------------------------------------
    def driver_hist(self, name: str, train_end: pd.Timestamp) -> pd.Series | None:
        if name not in self.drv_act.columns:
            return None
        s = self.drv_act[name].loc[:train_end].dropna()
        return s if len(s) else None

    def driver_future(self, name: str, months: pd.DatetimeIndex, mode: str,
                      train_end: pd.Timestamp) -> np.ndarray:
        """Backtest mode uses actual drivers; final mode uses the plan."""
        source = self.drv_act if mode == "backtest" else self.drv_plan
        if name in source.columns:
            s = source[name].reindex(months).astype(float)
        else:
            s = pd.Series(np.nan, index=months, dtype=float)
        if is_rate_driver(name) and s.isna().any():
            hist = self.driver_hist(name, train_end)
            last = hist.iloc[-1] if hist is not None else np.nan
            s = s.ffill().fillna(last)
        return s.to_numpy(float)

    def detect_planned(self) -> None:
        for name in self.drv_plan.columns:
            self.planned[name] = bool(self.drv_plan[name].reindex(self.forecast_months).notna().any())
        prefix = "Util_Units"
        types = sorted({c[len(prefix):] for c in self.drv_act.columns if c.startswith(prefix)})
        self.util_types = types

    # -- statistical sub-forecasts ----------------------------------------------
    def stat(self, name: str, series: pd.Series, h: int, mode: str) -> np.ndarray:
        if mode == "backtest":
            engine = self.engines.setdefault(name, StatEngine())
            return engine.forecast(series, h, reuse_arima=True)["ensemble"]
        return StatEngine().forecast(series, h)["ensemble"]
