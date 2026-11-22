"""Statistical forecasting methods for monthly series."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import sc_config as C

ARIMA_CANDIDATES = [((0, 1, 1), (0, 0, 0, 0)), ((1, 1, 0), (0, 0, 0, 0)), ((1, 1, 1), (0, 0, 0, 0))]
ARIMA_CANDIDATES_SEASONAL = [((0, 1, 1), (0, 0, 0, 0)), ((1, 1, 0), (0, 0, 0, 0)), ((0, 1, 1), (1, 0, 0, 12))]


def _values(y) -> np.ndarray:
    return np.asarray(pd.Series(y, dtype=float).dropna(), dtype=float)


def naive(y, h: int) -> np.ndarray:
    v = _values(y)
    return np.repeat(v[-1], h) if len(v) else np.full(h, np.nan)


def seasonal_naive(y, h: int) -> np.ndarray:
    v = _values(y)
    if len(v) < 12:
        return naive(v, h)
    return np.array([v[len(v) - 12 + (i % 12)] for i in range(h)])


def runrate_mean(y, h: int, n: int = 3) -> np.ndarray:
    v = _values(y)
    return np.repeat(v[-n:].mean(), h) if len(v) else np.full(h, np.nan)


def runrate_median(y, h: int, n: int = 6) -> np.ndarray:
    v = _values(y)
    return np.repeat(np.median(v[-n:]), h) if len(v) else np.full(h, np.nan)


def ets_forecast(y, h: int) -> np.ndarray:
    """Damped additive-trend exponential smoothing; additive seasonality with 24+ months."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    v = _values(y)
    if len(v) < 4:
        return naive(v, h)
    if np.std(v) == 0:
        return np.repeat(v[-1], h)
    scale = np.abs(v).mean() or 1.0
    z = v / scale
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(v) >= C.MIN_SEASONAL_OBS:
                model = ExponentialSmoothing(z, trend="add", damped_trend=True, seasonal="add",
                                             seasonal_periods=12, initialization_method="heuristic")
            else:
                model = ExponentialSmoothing(z, trend="add", damped_trend=True, initialization_method="estimated")
            fit = model.fit(use_brute=False)
            fc = np.asarray(fit.forecast(h), dtype=float) * scale
        if not np.all(np.isfinite(fc)):
            raise ValueError("non-finite forecast")
        return fc
    except Exception:  # noqa: BLE001 - fall back to a simple method
        return seasonal_naive(v, h) if len(v) >= 12 else naive(v, h)


class ArimaState:
    """ARIMA order chosen once by AIC, then reused on longer windows without re-estimation."""

    def __init__(self) -> None:
        self.result = None
        self.scale = 1.0
        self.spec = None

    def _fit(self, z: np.ndarray, order, sorder):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(z, order=order, seasonal_order=sorder, trend="n",
                            enforce_stationarity=False, enforce_invertibility=False)
            return model.fit(disp=False, maxiter=50)

    def select(self, v: np.ndarray) -> None:
        self.scale = np.abs(v).mean() or 1.0
        z = v / self.scale
        candidates = ARIMA_CANDIDATES_SEASONAL if len(v) >= C.MIN_SEASONAL_OBS else ARIMA_CANDIDATES
        best, best_aic = None, np.inf
        for order, sorder in candidates:
            try:
                res = self._fit(z, order, sorder)
                if np.isfinite(res.aic) and res.aic < best_aic:
                    best, best_aic = (res, (order, sorder)), res.aic
            except Exception:  # noqa: BLE001
                continue
        if best is None:
            self.result, self.spec = None, None
        else:
            self.result, self.spec = best

    def forecast(self, v: np.ndarray, h: int, reuse: bool) -> np.ndarray:
        if not reuse or self.result is None:
            self.select(v)
            if self.result is None:
                return np.full(h, np.nan)
            res = self.result
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = self.result.apply(v / self.scale, refit=False)
            except Exception:  # noqa: BLE001
                self.select(v)
                if self.result is None:
                    return np.full(h, np.nan)
                res = self.result
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = np.asarray(res.forecast(h), dtype=float) * self.scale
        return fc


class StatEngine:
    """ETS + ARIMA average. In backtests the ARIMA order is chosen at the first origin and reused."""

    def __init__(self) -> None:
        self.arima = ArimaState()
        self.calls = 0

    def forecast(self, y, h: int, reuse_arima: bool = False) -> dict:
        v = _values(y)
        if len(v) < C.MIN_HISTORY_FOR_MODELS:
            fc = runrate_mean(v, h, n=min(3, max(len(v), 1)))
            return {"ets": fc, "arima": fc, "ensemble": fc}
        if np.std(v) == 0:
            fc = np.repeat(v[-1], h)
            return {"ets": fc, "arima": fc, "ensemble": fc}
        ets = ets_forecast(v, h)
        try:
            arima = self.arima.forecast(v, h, reuse=reuse_arima and self.calls > 0)
        except Exception:  # noqa: BLE001
            arima = np.full(h, np.nan)
        self.calls += 1
        if np.all(np.isfinite(arima)) and np.all(np.isfinite(ets)):
            ens = (ets + arima) / 2.0
        elif np.all(np.isfinite(ets)):
            ens = ets
        else:
            ens = arima
        if not np.all(np.isfinite(ens)):
            ens = seasonal_naive(v, h) if len(v) >= 12 else naive(v, h)
        return {"ets": ets, "arima": arima, "ensemble": ens}


def volume_regression(y_hist: pd.Series, vol_hist: pd.Series, vol_future: np.ndarray,
                      months: int = C.CLASSIFY_MONTHS) -> tuple[np.ndarray, float, float] | None:
    df = pd.concat({"y": y_hist, "v": vol_hist}, axis=1).dropna().iloc[-months:]
    if len(df) < 12 or df["v"].std() == 0 or np.any(~np.isfinite(vol_future)):
        return None
    slope, intercept = np.polyfit(df["v"].to_numpy(float), df["y"].to_numpy(float), 1)
    return intercept + slope * np.asarray(vol_future, dtype=float), float(intercept), float(slope)


def aop_adjusted(aop_future: np.ndarray, ytd_act: float, ytd_aop: float) -> tuple[np.ndarray, float]:
    ratio = ytd_act / ytd_aop if ytd_aop and np.isfinite(ytd_aop) and ytd_aop != 0 else 1.0
    ratio = float(np.clip(ratio, 0.5, 2.0))
    return np.asarray(aop_future, dtype=float) * ratio, ratio
