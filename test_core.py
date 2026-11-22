"""Core tests for the thermal equipment plant model: run with `pytest -q`."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

import sc_config as C
from sc_classify import cost_behavior, detect_outliers, forecastability, normalize_bucket, suggest_bucket
from sc_driverprep import build_driver_frames
from sc_drivers import recompute
from sc_engine import Settings, run_models
from sc_export import build_template, write_workbook
from sc_fx import FxTable
from sc_io_excel import detect_actuals_through, load_inputs, parse_period_label
from sc_metrics import accuracy, bias, wape
from sc_sample import make_sample_workbook
from sc_scenario import Scenario, add_activity, allocate_gl, build_table, summarize_fy


@pytest.fixture(scope="module")
def sample():
    inputs = load_inputs(make_sample_workbook(), "sample.xlsx")
    results = run_models(inputs, Settings(year=2026, actuals_through=6))
    return inputs, results


# -- metrics -------------------------------------------------------------------
def test_wape_accuracy_bias():
    a, f = [100, 200, 300], [110, 190, 300]
    assert wape(a, f) == pytest.approx(20 / 600)
    assert accuracy(a, f) == pytest.approx(1 - 20 / 600)
    assert bias(a, f) == pytest.approx(0.0)
    assert np.isnan(wape([0, 0], [1, 1]))


# -- parsing -------------------------------------------------------------------
@pytest.mark.parametrize("label,expected", [
    ("Jan", (None, 1)), ("Enero", (None, 1)), ("2026-03", (2026, 3)), ("03/2026", (2026, 3)),
    ("P07", (None, 7)), ("Sep-26", (2026, 9)), (pd.Timestamp("2026-11-01"), (2026, 11)),
    (46023.0, (2026, 1)), (202612, (2026, 12)), (5, (None, 5)), ("Plant", (None, None)),
])
def test_parse_period_label(label, expected):
    assert parse_period_label(label) == expected


def test_bucket_names_and_keywords():
    assert normalize_bucket("Mano de obra directa") == C.DIRECT_LABOR
    assert normalize_bucket("Retrabajo") == C.SCRAP
    assert normalize_bucket("Consumibles y herramentales") == C.CONSUMABLES
    assert suggest_bucket("Brazing Alloy and Welding Gas") == C.CONSUMABLES
    assert suggest_bucket("Rework Labor and Material") == C.SCRAP
    assert suggest_bucket("Certification and Testing Fees") == C.FEES
    assert suggest_bucket("540300 Water and Sewer") == C.UTILITIES
    assert suggest_bucket("Natural Gas") == C.UTILITIES
    assert suggest_bucket("Gastos varios") is None
    assert suggest_bucket("Misc Plant Expense") is None


def test_long_format_and_title_rows():
    rows = []
    for d in pd.date_range("2024-01-01", "2026-06-01", freq="MS"):
        rows.append(["P1", "USD", "Actual", "Mantenimiento", d.strftime("%Y-%m"), 100_000 + d.month * 1_000])
    df = pd.DataFrame(rows, columns=["Planta", "Moneda", "Version", "Rubro", "Mes", "Monto"])
    title = pd.DataFrame([["Spend export"] + [None] * 5], columns=df.columns)
    header = pd.DataFrame([list(df.columns)], columns=df.columns)
    raw = pd.concat([title, header, df], ignore_index=True)
    data = io.BytesIO()
    with pd.ExcelWriter(data, engine="xlsxwriter") as w:
        raw.to_excel(w, sheet_name="Gasto", index=False, header=False)
    inputs = load_inputs(data.getvalue(), "long.xlsx")
    assert not inputs.fatal
    assert inputs.formats["Spend"].startswith("long")
    assert set(inputs.spend["Bucket"]) == {C.MAINTENANCE}
    assert detect_actuals_through(inputs.spend, 2026) == 6


def test_double_counting_uses_gl_rows():
    months = {m: 100.0 for m in C.MONTH_ABBR}
    spend = pd.DataFrame([
        {"Plant": "P1", "Currency": "USD", "Version": "Act", "Year": 2025, "Bucket": "Fees", "GL_Account": "",
         **{m: 250.0 for m in C.MONTH_ABBR}},
        {"Plant": "P1", "Currency": "USD", "Version": "Act", "Year": 2025, "Bucket": "", "GL_Account": "590100",
         **months},
        {"Plant": "P1", "Currency": "USD", "Version": "Act", "Year": 2025, "Bucket": "", "GL_Account": "590200",
         **months},
    ])
    mapping = pd.DataFrame([["590100", "", "Fees"], ["590200", "", "Fees"]],
                           columns=["GL_Account", "Cost_Center", "Bucket"])
    inputs = load_inputs(write_workbook({"Spend": spend, "Mapping": mapping}), "dc.xlsx")
    act = inputs.spend[inputs.spend["Version"] == "Act"]
    assert act["Value"].sum() == pytest.approx(12 * 200.0)
    assert (inputs.issues["Check"] == "Double counting").any()


# -- classification ----------------------------------------------------------------
def test_outliers_flag_one_off_but_not_recurring_shutdowns():
    idx = pd.date_range("2024-01-01", periods=30, freq="MS")
    rng = np.random.default_rng(0)
    y = pd.Series(100_000 * (1 + rng.normal(0, 0.01, 30)), index=idx)
    y[idx.month == 8] *= 0.55          # August shutdown every year: seasonality, not an atypical item
    y.iloc[15] = 180_000               # one-off spike (Apr-25)
    flags = detect_outliers(y)
    assert bool(flags["Flag"].iloc[15])
    assert not flags.loc[idx.month == 8, "Flag"].any()
    assert int(flags["Flag"].sum()) == 1


def test_cost_behavior_and_forecastability():
    idx = pd.date_range("2024-01-01", periods=24, freq="MS")
    rng = np.random.default_rng(1)
    hours = pd.Series(12_000 * (1 + 0.15 * np.sin(np.arange(24))) * (1 + rng.normal(0, 0.01, 24)), index=idx)
    variable = 4.5 * hours * (1 + rng.normal(0, 0.01, 24))
    fixed = pd.Series(50_000 * (1 + rng.normal(0, 0.01, 24)), index=idx)
    assert cost_behavior(variable, hours)["Behavior"] == "Variable"
    assert cost_behavior(fixed, hours)["Behavior"] == "Fixed"
    assert forecastability(fixed)["Segment"] == "Stable"


# -- drivers ------------------------------------------------------------------------
def _drivers_frame(rows):
    df = pd.DataFrame(rows, columns=["Plant", "Product_Line", "Driver", "Version", "Date", "Value"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def test_earned_hours_roll_up_to_plant():
    rows = []
    for d in ("2026-05-01", "2026-06-01"):
        rows += [["P1", "A", "Units", "Act", d, 10], ["P1", "A", "Std_Hours_per_Unit", "Act", d, 40],
                 ["P1", "A", "Labor_Hours", "Act", d, 500], ["P1", "B", "Units", "Act", d, 2],
                 ["P1", "B", "Std_Hours_per_Unit", "Act", d, 200], ["P1", "B", "Labor_Hours", "Act", d, 450],
                 ["P1", "", "Wage_Rate", "Act", d, 30]]
    rows += [["P1", "A", "Units", "Plan", "2026-07-01", 11], ["P1", "B", "Units", "Plan", "2026-07-01", 3]]
    act, plan, basis = build_driver_frames(_drivers_frame(rows))
    plant = act[("P1", "")]
    assert basis[("P1", "")] == "earned hours"
    assert plant.loc["2026-06-01", "Earned_Hours"] == pytest.approx(10 * 40 + 2 * 200)
    assert plant.loc["2026-06-01", "Labor_Hours"] == pytest.approx(950)
    assert plant.loc["2026-06-01", "Wage_Rate"] == pytest.approx(30)
    # the plan uses the last actual standard of each line
    assert plan[("P1", "")].loc["2026-07-01", "Activity"] == pytest.approx(11 * 40 + 3 * 200)


def test_plant_plan_needs_every_active_line():
    rows = []
    for d in ("2026-05-01", "2026-06-01"):
        rows += [["P1", "A", "Units", "Act", d, 10], ["P1", "B", "Units", "Act", d, 5]]
    rows += [["P1", "A", "Units", "Plan", "2026-07-01", 11]]  # line B has no plan
    _act, plan, _basis = build_driver_frames(_drivers_frame(rows))
    assert np.isnan(plan[("P1", "")].loc["2026-07-01", "Activity"])


def test_labor_scenario_recompute():
    comps = {"hours": np.array([1_000.0]), "wage": np.array([30.0]), "k": 1.2}
    assert recompute("labor", comps, Scenario())[0] == pytest.approx(1_000 * 30 * 1.2)
    up = recompute("labor", comps, Scenario(volume=0.10, efficiency=0.10, wage=0.05))
    assert up[0] == pytest.approx(1_000 * 1.10 / 1.10 * 30 * 1.05 * 1.2)


# -- end to end ---------------------------------------------------------------------
def test_sample_end_to_end(sample):
    inputs, results = sample
    assert not inputs.fatal
    assert len(results.series) == 44
    for sr in results.series.values():
        assert len(sr.forecast) == 6
        assert np.isfinite(sr.forecast.to_numpy()).all()
    drivers = [sr for sr in results.series.values() if sr.bucket in C.DRIVER_BUCKETS]
    assert all(sr.method == C.M_DRIVER for sr in drivers)
    assert results.accuracy_network["Accuracy_1m"].iloc[0] > 0.9
    assert set(results.activity["Plant"]) == {"US01", "MX01", "IT01"}
    assert results.activity["Activity"].notna().all()


def test_landing_identity_activity_and_gl_allocation(sample):
    inputs, results = sample
    fx = FxTable(inputs.fx, 2026)
    table, warnings = build_table(results, Scenario(), inputs.adjustments, fx)
    assert warnings == []
    fy = add_activity(summarize_fy(table, ["Plant"], usd=True), results)
    assert np.allclose(fy["FY_Landing"], fy["YTD_Actual"] + fy["YTG_Forecast"] + fy["Mgmt_Adjustments"])
    assert (fy["Cost_per_Activity"] > 0).all()
    gl = allocate_gl(table, inputs.gl_detail, results)
    fc = gl[gl["Period"] != "Actual"].groupby(["Bucket", "Date"])["Value"].sum()
    us01 = table[(table["Plant"] == "US01") & (table["Period"] == "Forecast")].set_index(["Bucket", "Date"])
    assert np.allclose(fc.sort_index(), (us01["Forecast"] + us01["Adjustment"]).sort_index())


def test_template_loads():
    inputs = load_inputs(build_template(), "template.xlsx")
    assert not inputs.fatal
    assert not inputs.issues["Severity"].isin(["Error", "Warning"]).any()
