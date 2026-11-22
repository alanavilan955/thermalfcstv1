"""Plotly figures. Every function takes a palette from config.PALETTES ("dark" or "light")."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import sc_config as C

DEFAULT = C.PALETTES["dark"]


def _layout(fig: go.Figure, title: str, y_title: str, pal: dict, height: int = 420) -> go.Figure:
    fig.update_layout(
        template=pal["template"],
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=15, color=pal["title"])),
        paper_bgcolor=pal["bg"], plot_bgcolor=pal["bg"],
        font=dict(family="Arial, Helvetica, sans-serif", size=12, color=pal["font"]),
        height=height, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=pal["font"])),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=pal["hover_bg"], font=dict(color=pal["font"])),
    )
    fig.update_yaxes(title=y_title, tickformat=",.0f", gridcolor=pal["grid"], zeroline=False,
                     linecolor=pal["axis"], color=pal["font"], automargin=True)
    fig.update_xaxes(showgrid=False, linecolor=pal["axis"], color=pal["font"], automargin=True)
    return fig


def series_chart(history: pd.Series, forecast: pd.Series, lower: pd.Series, upper: pd.Series,
                 aop: pd.Series, ppf: pd.Series, later: pd.Series, title: str, y_title: str,
                 base: pd.Series | None = None, pal: dict = DEFAULT) -> go.Figure:
    fig = go.Figure()
    fc = forecast.dropna()
    if len(fc) and lower.notna().any():
        fig.add_trace(go.Scatter(x=list(upper.index) + list(lower.index[::-1]),
                                 y=list(upper.values) + list(lower.values[::-1]),
                                 fill="toself", fillcolor=pal["band"], line=dict(width=0), mode="lines",
                                 hoverinfo="skip", name="Approx. 80% interval"))
    fig.add_trace(go.Scatter(x=history.index, y=history.values, name="Actual", mode="lines+markers",
                             line=dict(color=pal["actual"], width=2.4), marker=dict(size=5)))
    if len(fc):
        bridge_x = ([history.index[-1]] if len(history) else []) + list(fc.index)
        bridge_y = ([history.values[-1]] if len(history) else []) + list(fc.values)
        fig.add_trace(go.Scatter(x=bridge_x, y=bridge_y, name="Forecast", mode="lines+markers",
                                 line=dict(color=pal["forecast"], width=2.4, dash="dash"), marker=dict(size=6)))
    if base is not None and len(base.dropna()) and not np.allclose(base.fillna(0).values, forecast.fillna(0).values):
        fig.add_trace(go.Scatter(x=base.index, y=base.values, name="Base (no scenario)", mode="lines",
                                 line=dict(color=pal["base"], width=1.5, dash="dot")))
    if aop is not None and aop.notna().any():
        fig.add_trace(go.Scatter(x=aop.index, y=aop.values, name="AOP", mode="lines",
                                 line=dict(color=pal["aop"], width=1.8, dash="dot")))
    if ppf is not None and ppf.notna().any():
        fig.add_trace(go.Scatter(x=ppf.index, y=ppf.values, name="PPF", mode="lines",
                                 line=dict(color=pal["ppf"], width=1.8, dash="dashdot")))
    if later is not None and len(later.dropna()):
        fig.add_trace(go.Scatter(x=later.index, y=later.values, name="Actual after cutoff", mode="markers",
                                 marker=dict(color=pal["later"], symbol="circle-open", size=8)))
    return _layout(fig, title, y_title, pal)


def overview_chart(monthly: pd.DataFrame, title: str, y_title: str, pal: dict = DEFAULT) -> go.Figure:
    """monthly: Date, Actual, Forecast, AOP, PPF (already summed)."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly["Date"], y=monthly["Actual"], name="Actual", marker_color=pal["bar_actual"]))
    fig.add_trace(go.Bar(x=monthly["Date"], y=monthly["Forecast"], name="Forecast + adjustments",
                         marker_color=pal["bar_forecast"]))
    fig.add_trace(go.Scatter(x=monthly["Date"], y=monthly["AOP"], name="AOP", mode="lines+markers",
                             line=dict(color=pal["aop"], width=2, dash="dot")))
    if monthly["PPF"].notna().any() and monthly["PPF"].abs().sum() > 0:
        fig.add_trace(go.Scatter(x=monthly["Date"], y=monthly["PPF"], name="PPF", mode="lines",
                                 line=dict(color=pal["ppf"], width=2, dash="dashdot")))
    fig.update_layout(barmode="stack")
    return _layout(fig, title, y_title, pal, height=380)


def cost_per_activity_chart(df: pd.DataFrame, title: str, y_title: str, pal: dict = DEFAULT) -> go.Figure:
    """df: Date, Cost_per_Activity, Period."""
    fig = go.Figure()
    act = df[df["Period"] == "Actual"]
    fc = df[df["Period"] == "Forecast"]
    fig.add_trace(go.Scatter(x=act["Date"], y=act["Cost_per_Activity"], name="Actual", mode="lines+markers",
                             line=dict(color=pal["actual"], width=2.4)))
    if len(fc):
        bx = ([act["Date"].iloc[-1]] if len(act) else []) + list(fc["Date"])
        by = ([act["Cost_per_Activity"].iloc[-1]] if len(act) else []) + list(fc["Cost_per_Activity"])
        fig.add_trace(go.Scatter(x=bx, y=by, name="Forecast", mode="lines+markers",
                                 line=dict(color=pal["forecast"], width=2.4, dash="dash")))
    _layout(fig, title, y_title, pal, height=320)
    fig.update_yaxes(tickformat=",.2f")
    return fig


def accuracy_heatmap(table: pd.DataFrame, value_col: str, target: float, title: str,
                     pal: dict = DEFAULT) -> go.Figure:
    """table: Row, Bucket, value_col."""
    piv = table.pivot_table(index="Row", columns="Bucket", values=value_col, aggfunc="mean")
    order = [b for b in C.STANDARD_BUCKETS if b in piv.columns] + [b for b in piv.columns if b not in C.STANDARD_BUCKETS]
    piv = piv[order]
    z = piv.values.astype(float)
    text = np.where(np.isfinite(z), np.char.mod("%.1f%%", np.nan_to_num(z) * 100), "n/a")
    lo = max(0.0, min(np.nanmin(z) if np.isfinite(z).any() else target - 0.2, target - 0.15))
    c0, c1, c2, c3 = pal["heat"]
    colorscale = [[0.0, c0], [max(0.01, (target - 0.05 - lo) / (1 - lo)), c1],
                  [max(0.02, (target - lo) / (1 - lo)), c2], [1.0, c3]]
    fig = go.Figure(go.Heatmap(z=z, x=list(piv.columns), y=list(piv.index), text=text, texttemplate="%{text}",
                               textfont=dict(color=pal["heat_text"]), colorscale=colorscale, zmin=lo, zmax=1.0,
                               hoverongaps=False, xgap=2, ygap=2,
                               colorbar=dict(tickformat=".0%", thickness=14, tickfont=dict(color=pal["font"]))))
    height = max(260, 70 + 36 * len(piv.index))
    fig.update_layout(template=pal["template"], title=dict(text=title, x=0, font=dict(size=15, color=pal["title"])),
                      paper_bgcolor=pal["bg"], plot_bgcolor=pal["bg"],
                      font=dict(family="Arial, Helvetica, sans-serif", size=12, color=pal["font"]),
                      height=height, margin=dict(l=10, r=10, t=50, b=10))
    fig.update_yaxes(autorange="reversed", color=pal["font"], automargin=True, showgrid=False)
    fig.update_xaxes(color=pal["font"], automargin=True, showgrid=False)
    return fig


def waterfall(bridge_df: pd.DataFrame, ref_label: str, ref_total: float, landing_total: float,
              adj_total: float, title: str, y_title: str, pal: dict = DEFAULT) -> go.Figure:
    labels = [ref_label] + list(bridge_df["Bucket"])
    values = [ref_total] + list(bridge_df["Delta"])
    measures = ["absolute"] + ["relative"] * len(bridge_df)
    if abs(adj_total) > 0:
        labels.append("Mgmt adjustments")
        values.append(adj_total)
        measures.append("relative")
    labels.append("FY landing")
    values.append(landing_total)
    measures.append("total")
    texts = [f"{v:,.0f}" if m != "relative" else f"{v:+,.0f}" for v, m in zip(values, measures)]
    fig = go.Figure(go.Waterfall(
        x=labels, y=values, measure=measures, text=texts, textposition="outside",
        textfont=dict(color=pal["font"]), connector=dict(line=dict(color=pal["axis"])),
        increasing=dict(marker=dict(color=pal["up"])), decreasing=dict(marker=dict(color=pal["down"])),
        totals=dict(marker=dict(color=pal["total"])),
    ))
    _layout(fig, title, y_title, pal, height=440)
    levels = np.cumsum([ref_total] + list(bridge_df["Delta"]) + ([adj_total] if abs(adj_total) > 0 else []))
    low, high = float(np.min(levels)), float(np.max(levels))
    span = max(high - low, abs(high) * 0.01, 1.0)
    fig.update_yaxes(range=[low - 0.6 * span, high + 0.35 * span])
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig
