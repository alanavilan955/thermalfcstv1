"""Constant-currency conversion to USD at the AOP rate of the forecast year."""

from __future__ import annotations

import numpy as np
import pandas as pd


class FxTable:
    """Rate_to_USD lookup: USD per 1 unit of local currency.

    Priority: currency + forecast year + month, currency + forecast year, currency only,
    then the average of the monthly rates of the forecast year.
    """

    def __init__(self, fx: pd.DataFrame | None, year: int) -> None:
        self.year = int(year)
        self.monthly: dict[tuple[str, int, int], float] = {}
        self.yearly: dict[tuple[str, int], float] = {}
        self.flat: dict[str, float] = {}
        if fx is None or fx.empty:
            return
        for row in fx.itertuples(index=False):
            cur = str(row.Currency).upper()
            rate = float(row.Rate_to_USD)
            year_ok = pd.notna(row.Year)
            month_ok = pd.notna(row.Month)
            if year_ok and month_ok:
                self.monthly[(cur, int(row.Year), int(row.Month))] = rate
            elif year_ok:
                self.yearly[(cur, int(row.Year))] = rate
            else:
                self.flat[cur] = rate

    def rate(self, currency: str, month: int | None = None) -> float:
        cur = str(currency).upper()
        if cur == "USD":
            return 1.0
        if month is not None and (cur, self.year, int(month)) in self.monthly:
            return self.monthly[(cur, self.year, int(month))]
        if (cur, self.year) in self.yearly:
            return self.yearly[(cur, self.year)]
        if cur in self.flat:
            return self.flat[cur]
        values = [v for (c, y, _m), v in self.monthly.items() if c == cur and y == self.year]
        return float(np.mean(values)) if values else np.nan

    def has(self, currency: str) -> bool:
        return bool(np.isfinite(self.rate(currency)))

    def rates_for(self, currency: str, dates) -> np.ndarray:
        return np.array([self.rate(currency, pd.Timestamp(d).month) for d in dates], dtype=float)
