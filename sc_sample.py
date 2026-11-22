"""Synthetic sample workbook for data center cooling equipment plants.

All plants, product lines, assets, accounts and figures are invented for demonstration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import sc_config as C
from sc_export import write_workbook

MONTHS = pd.date_range("2024-01-01", "2026-12-01", freq="MS")
CUTOFF = pd.Timestamp("2026-06-01")
PLAN_START = pd.Timestamp("2026-07-01")
N = len(MONTHS)
T = np.arange(N)
MOY = MONTHS.month.to_numpy()

PERIMETER, CHILLERS, LIQUID = "Perimeter Cooling", "Chillers", "Liquid Cooling"

# plant, line, base units/month, monthly growth, std hours 2024-25, std hours 2026, efficiency start, end, noise
LINES = [
    ("US01", PERIMETER, 140, 0.0025, 42.0, 42.0, 0.86, 0.88, 0.07),
    ("US01", CHILLERS, 24, 0.0025, 240.0, 235.0, 0.82, 0.84, 0.12),
    ("MX01", LIQUID, 60, 0.040, 65.0, 60.0, 0.78, 0.86, 0.08),
    ("MX01", PERIMETER, 90, 0.0025, 42.0, 42.0, 0.84, 0.85, 0.07),
    ("IT01", CHILLERS, 30, 0.0020, 240.0, 235.0, 0.88, 0.89, 0.10),
]
SEASON = {"US01": {3: 1.04, 7: 0.92, 12: 1.10}, "MX01": {4: 0.95, 12: 0.85}, "IT01": {8: 0.55, 12: 0.90}}
CURRENCY = {"US01": "USD", "MX01": "MXN", "IT01": "EUR"}

# Spend entities: (plant, product line in Spend) -> lines that drive it
ENTITIES = {("US01", ""): [PERIMETER, CHILLERS], ("MX01", LIQUID): [LIQUID], ("MX01", PERIMETER): [PERIMETER],
            ("IT01", ""): [CHILLERS]}

PARAMS = {
    ("US01", ""): dict(wage=31.0, wage_step=("apr", 0.035), elec_int=60, elec_season=([6, 7, 8], 0.12),
                       elec_t=0.092, elec_growth=0.03, elec_noise=0.015, gas_int=1.2, gas_t=0.68, gas_noise=0.08,
                       fixed_u=38_000, maint=160_000, consum_rate=4.5, consum_fixed=12_000, scrap_rate=3.0,
                       freight={PERIMETER: 180, CHILLERS: 650}, dep=410_000, dep_step=("2025-01", 35_000),
                       fees=55_000, il=520_000, il_step=("apr", 0.03), alloc=210_000, other=45_000,
                       plan_tariffs={"Electricity": 0.092 * 1.06, "Gas": 0.70}),
    ("MX01", LIQUID): dict(wage=118.0, wage_step=("jan", 0.08), elec_int=55, elec_season=([4, 5, 6, 7, 8, 9], 0.10),
                           elec_t=2.35, elec_growth=0.05, elec_noise=0.03, gas_int=None, fixed_u=250_000,
                           maint=1_300_000, consum_rate=70.0, consum_fixed=150_000, scrap_rate=55.0,
                           freight={LIQUID: 9_000}, dep=3_200_000, dep_step=("2025-07", 1_800_000),
                           fees=450_000, il=4_200_000, il_step=("jan", 0.08), alloc=1_900_000, other=350_000,
                           plan_tariffs={"Electricity": 2.35 * 1.1025}),
    ("MX01", PERIMETER): dict(wage=118.0, wage_step=("jan", 0.08), elec_int=40,
                              elec_season=([4, 5, 6, 7, 8, 9], 0.10), elec_t=2.35, elec_growth=0.05,
                              elec_noise=0.03, gas_int=None, fixed_u=200_000, maint=1_000_000, consum_rate=60.0,
                              consum_fixed=120_000, scrap_rate=40.0, freight={PERIMETER: 6_500},
                              dep=2_100_000, dep_step=None, fees=350_000, il=3_300_000, il_step=("jan", 0.08),
                              alloc=1_600_000, other=250_000, plan_tariffs={"Electricity": 2.35 * 1.1025}),
    ("IT01", ""): dict(wage=29.0, wage_step=("jul", 0.025), elec_int=80, elec_season=([7], 0.10), elec_t=0.19,
                       elec_growth=0.0, elec_noise=0.10, gas_int=1.0, gas_t=0.55, gas_noise=0.12, fixed_u=26_000,
                       maint=110_000, consum_rate=3.8, consum_fixed=8_000, scrap_rate=2.6, freight={CHILLERS: 900},
                       dep=240_000, dep_step=None, fees=35_000, il=310_000, il_step=("jul", 0.025), alloc=150_000,
                       other=25_000, plan_tariffs={"Electricity": 0.20, "Gas": 0.56}),
}

EVENTS = {
    ("US01", ""): [("Brazing furnace", "Overhaul", "2024-09", 180_000), ("Powder coat line", "Shutdown", "2025-07", 140_000),
                   ("Test lab chiller loop", "Overhaul", "2026-08", 220_000), ("Fiber laser", "PM", "2026-11", 90_000)],
    ("MX01", LIQUID): [("CDU test stand", "Overhaul", "2025-10", 1_600_000), ("Brazing cell", "PM", "2026-09", 1_100_000)],
    ("MX01", PERIMETER): [("Coil line", "Overhaul", "2025-04", 1_300_000), ("Plant", "Shutdown", "2026-12", 900_000)],
    ("IT01", ""): [("Plant", "Summer shutdown maintenance", "2024-08", 160_000),
                   ("Plant", "Summer shutdown maintenance", "2025-08", 170_000),
                   ("Plant", "Summer shutdown maintenance", "2026-08", 180_000),
                   ("Run-test cell", "Overhaul", "2026-10", 120_000)],
}

# One-off items to show the atypical-item checks
SPIKES = {
    ("US01", ""): {C.MAINTENANCE: [("2025-03", 150_000)], C.SCRAP: [("2026-02", 80_000)],
                   C.FREIGHT: [("2025-09", 45_000)], C.FEES: [("2024-10", None)]},
    ("MX01", LIQUID): {C.SCRAP: [("2025-06", 1_200_000)]},
    ("IT01", ""): {C.FEES: [("2025-05", 90_000)], C.OTHER: [("2024-08", None)]},
}

US01_GL = {
    C.DIRECT_LABOR: [("510100", "US01-ASM", "Assembly Direct Labor", 0.62), ("510200", "US01-ASM", "Overtime Premium", 0.14),
                     ("510300", "US01-ASM", "Temp Agency Labor", 0.10), ("510400", "US01-TST", "Test Technicians Labor", 0.14)],
    C.INDIRECT_LABOR: [("520100", "US01-ADM", "Salaries Supervision", 0.55),
                       ("520200", "US01-QLT", "Quality and Engineering Salaries", 0.45)],
    C.MAINTENANCE: [("530100", "US01-MNT", "Repair Parts", 0.50), ("530200", "US01-MNT", "Contract Maintenance Services", 0.50)],
    C.UTILITIES: [("540100", "US01-FAC", "Electricity", 0.80), ("540200", "US01-FAC", "Natural Gas", 0.15),
                  ("540300", "US01-FAC", "Water and Sewer", 0.05)],
    C.CONSUMABLES: [("550100", "US01-ASM", "Shop Supplies", 0.50), ("550200", "US01-ASM", "Brazing Alloy and Welding Gas", 0.30),
                    ("550300", "US01-ASM", "Perishable Tooling", 0.20)],
    C.SCRAP: [("560100", "US01-QLT", "Scrap Material", 0.60), ("560200", "US01-QLT", "Rework Labor and Material", 0.40)],
    C.FREIGHT: [("570100", "US01-LOG", "Outbound Freight", 0.75), ("570200", "US01-LOG", "Expedite Freight", 0.25)],
    C.DEPRECIATION: [("580100", "US01-FIN", "Depreciation Machinery and Test Lab", 1.0)],
    C.FEES: [("590100", "US01-QLT", "Certification and Testing Fees", 0.60), ("590200", "US01-ADM", "Consulting Fees", 0.40)],
    C.ALLOCATIONS: [("595100", "US01-ADM", "Corporate Allocation In", 1.0)],
    C.OTHER: [("599100", "US01-ADM", "Misc Plant Expense", 1.0)],
}
NOT_IN_MAPPING = {"540300", "599100"}

AOP_FACTOR = {C.DIRECT_LABOR: 0.97, C.INDIRECT_LABOR: 0.99, C.MAINTENANCE: 0.95, C.UTILITIES: 0.94,
              C.CONSUMABLES: 0.96, C.SCRAP: 0.85, C.FREIGHT: 0.95, C.DEPRECIATION: 1.0, C.FEES: 0.92,
              C.ALLOCATIONS: 1.0, C.OTHER: 0.95}


def _steps(rule: tuple[str, float] | None) -> np.ndarray:
    """Cumulative step factor for an annual increase in a given month (jan, apr or jul)."""
    if rule is None:
        return np.ones(N)
    month = {"jan": 1, "apr": 4, "jul": 7}[rule[0]]
    count = np.array([sum(1 for y in (2024, 2025, 2026) if d >= pd.Timestamp(year=y, month=month, day=1)
                          and not (month == 1 and y == 2024)) for d in MONTHS])
    return (1 + rule[1]) ** count


def _season(plant: str) -> np.ndarray:
    return np.array([SEASON[plant].get(m, 1.0) for m in MOY])


def _line_drivers(plant, line, base, growth, std0, std1, eff0, eff1, noise, rng):
    trend = base * (1 + growth) ** T
    if line == LIQUID:  # the build ramp slows from 4% to 3% a month after the cutoff
        k = int((MONTHS <= CUTOFF).sum()) - 1
        trend = np.where(T <= k, trend, trend[k] * 1.03 ** (T - k))
    units_exp = trend * _season(plant)
    units = np.round(units_exp * (1 + rng.normal(0, noise, N)))
    std = np.where(MONTHS.year >= 2026, std1, std0)
    if line == LIQUID:
        eff_exp = eff0 + (eff1 - eff0) * (1 - np.exp(-T / 12))
    else:
        eff_exp = eff0 + (eff1 - eff0) * T / (N - 1)
    eff = eff_exp * (1 + rng.normal(0, 0.02, N))
    earned = units * std
    labor_hours = earned / eff
    return dict(units=units, units_exp=units_exp, std=std, eff=eff, eff_exp=eff_exp, earned=earned,
                earned_exp=units_exp * std, labor_hours=labor_hours)


def make_sample_frames(seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    act_mask, plan_mask = MONTHS <= CUTOFF, MONTHS >= PLAN_START
    noise = lambda sd: 1 + rng.normal(0, sd, N)  # noqa: E731

    lines = {(p, l): _line_drivers(p, l, *rest, rng) for p, l, *rest in LINES}
    driver_rows: list[list] = []

    def add(plant, line, name, values, mask, version):
        for d, v in zip(MONTHS[mask], np.asarray(values, float)[mask]):
            driver_rows.append([plant, line, name, version, d.year, d.month, round(float(v), 4)])

    spend_rows, event_rows = [], []
    for (plant, sline), driving in ENTITIES.items():
        p = PARAMS[(plant, sline)]
        cur = CURRENCY[plant]
        L = [lines[(plant, l)] for l in driving]
        earned = sum(x["earned"] for x in L)
        earned_exp = sum(x["earned_exp"] for x in L)
        labor_hours = sum(x["labor_hours"] for x in L)
        units = {l: lines[(plant, l)]["units"] for l in driving}
        units_exp = {l: lines[(plant, l)]["units_exp"] for l in driving}

        wage = p["wage"] * _steps(p["wage_step"])
        dl = labor_hours * wage * 1.22 * noise(0.015)
        dl_exp = sum(x["earned_exp"] / x["eff_exp"] for x in L) * wage * 1.22

        season_e = 1 + p["elec_season"][1] * np.isin(MOY, p["elec_season"][0])
        elec_int = p["elec_int"] * season_e * noise(0.03)
        elec_units = elec_int * earned
        elec_tariff = p["elec_t"] * (1 + p["elec_growth"] * (MONTHS.year.to_numpy() - 2024)) * noise(p["elec_noise"])
        util = elec_units * elec_tariff + p["fixed_u"] * noise(0.02)
        util_exp = p["elec_int"] * season_e * earned_exp * p["elec_t"] * (1 + p["elec_growth"] * (MONTHS.year.to_numpy() - 2024)) + p["fixed_u"]
        gas_units = gas_tariff = None
        if p["gas_int"]:
            season_g = 1 + 0.5 * np.isin(MOY, [12, 1, 2])
            gas_units = p["gas_int"] * season_g * earned * noise(0.04)
            gas_tariff = p["gas_t"] * noise(p["gas_noise"])
            util = util + gas_units * gas_tariff
            util_exp = util_exp + p["gas_int"] * season_g * earned_exp * p["gas_t"]

        events = pd.Series(0.0, index=MONTHS)
        for asset, etype, ym, amount in EVENTS[(plant, sline)]:
            events[pd.Timestamp(ym + "-01")] += amount
            event_rows.append([plant, sline, asset, etype, int(ym[:4]), int(ym[5:]), amount, "Maintenance plan"])
        maint = p["maint"] * noise(0.12) + events.to_numpy()
        maint_exp = p["maint"] + events.to_numpy()

        consum = (p["consum_rate"] * earned + p["consum_fixed"]) * noise(0.06)
        consum_exp = p["consum_rate"] * earned_exp + p["consum_fixed"]
        scrap = p["scrap_rate"] * earned * noise(0.15)
        scrap_exp = p["scrap_rate"] * earned_exp
        freight = sum(p["freight"][l] * units[l] for l in driving) * noise(0.08)
        freight_exp = sum(p["freight"][l] * units_exp[l] for l in driving)
        dep = np.full(N, float(p["dep"]))
        if p["dep_step"]:
            dep[MONTHS >= pd.Timestamp(p["dep_step"][0] + "-01")] += p["dep_step"][1]
        dep_exp = dep.copy()
        dep = dep * noise(0.0005)
        fees_exp = np.full(N, float(p["fees"]))
        if plant == "US01":
            fees_exp = fees_exp + 30_000 * np.isin(MOY, [3, 9])  # certification testing
        fees = fees_exp * noise(0.05)
        il_exp = p["il"] * _steps(p["il_step"]) * (1 + (0.015 * (T // 3) if sline == LIQUID else 0))
        il = il_exp * noise(0.012)
        alloc_exp = p["alloc"] * (1 + 0.12 * np.isin(MOY, [3, 6, 9, 12]))
        alloc = alloc_exp * noise(0.03)
        other = p["other"] * noise(0.12)

        actual = {C.DIRECT_LABOR: dl, C.INDIRECT_LABOR: il, C.MAINTENANCE: maint, C.UTILITIES: util,
                  C.CONSUMABLES: consum, C.SCRAP: scrap, C.FREIGHT: freight, C.DEPRECIATION: dep, C.FEES: fees,
                  C.ALLOCATIONS: alloc, C.OTHER: other}
        expected = {C.DIRECT_LABOR: dl_exp, C.INDIRECT_LABOR: il_exp, C.MAINTENANCE: maint_exp, C.UTILITIES: util_exp,
                    C.CONSUMABLES: consum_exp, C.SCRAP: scrap_exp, C.FREIGHT: freight_exp, C.DEPRECIATION: dep_exp,
                    C.FEES: fees_exp, C.ALLOCATIONS: alloc_exp, C.OTHER: np.full(N, float(p["other"]))}
        for bucket, items in SPIKES.get((plant, sline), {}).items():
            for ym, amount in items:
                i = MONTHS.get_loc(pd.Timestamp(ym + "-01"))
                if amount is None:
                    actual[bucket][i] = -20_000.0 if bucket == C.FEES else 0.0  # reversal or zero month
                else:
                    actual[bucket][i] += amount

        # Spend rows (wide): Act 2024-2026 (2026 through June), AOP and PPF for 2026
        base = {"Plant": plant, "Product_Line": sline, "Currency": cur}
        for bucket in C.STANDARD_BUCKETS:
            series = pd.Series(np.round(actual[bucket], 0), index=MONTHS)
            exp = pd.Series(expected[bucket], index=MONTHS)
            for year in (2024, 2025, 2026):
                vals = series[series.index.year == year].copy()
                vals[vals.index > CUTOFF] = np.nan
                if plant == "US01":
                    for gl, cc, desc, share in US01_GL[bucket]:
                        shares = share * (1 + rng.normal(0, 0.01, len(vals))) if share < 1 else np.ones(len(vals))
                        spend_rows.append({**base, "Version": "Act", "Year": year, "Bucket": "", "GL_Account": gl,
                                           "Cost_Center": cc, "Description": desc,
                                           **dict(zip(C.MONTH_ABBR, np.round(vals.to_numpy() * shares, 0)))})
                else:
                    spend_rows.append({**base, "Version": "Act", "Year": year, "Bucket": bucket, "GL_Account": "",
                                       "Cost_Center": "", "Description": "", **dict(zip(C.MONTH_ABBR, vals.to_numpy()))})
            exp26 = exp[exp.index.year == 2026]
            aop = np.round(exp26.to_numpy() * AOP_FACTOR[bucket], 0)
            ppf = np.where(exp26.index <= pd.Timestamp("2026-05-01"), series[series.index.year == 2026].to_numpy(),
                           np.round(exp26.to_numpy() * (1 + rng.normal(0, 0.03, 12)), 0))
            for version, values in (("AOP", aop), ("PPF", ppf)):
                spend_rows.append({**base, "Version": version, "Year": 2026, "Bucket": bucket, "GL_Account": "",
                                   "Cost_Center": "", "Description": "", **dict(zip(C.MONTH_ABBR, values))})

        # Drivers
        drv_line = sline if plant != "IT01" else ""
        wage_plan = np.full(N, wage[act_mask][-1])
        if plant == "IT01":
            wage_plan[MONTHS >= pd.Timestamp("2026-07-01")] *= 1.025
        util_level = "" if plant == "US01" else drv_line
        add(plant, "" if plant in ("US01", "IT01") else drv_line, C.WAGE, wage, act_mask, "Act")
        add(plant, "" if plant in ("US01", "IT01") else drv_line, C.WAGE, wage_plan, plan_mask, "Plan")
        add(plant, util_level, "Util_Units_Electricity", elec_units, act_mask, "Act")
        add(plant, util_level, "Util_Tariff_Electricity", elec_tariff, act_mask, "Act")
        add(plant, util_level, "Util_Tariff_Electricity", np.full(N, p["plan_tariffs"]["Electricity"]), plan_mask, "Plan")
        if gas_units is not None:
            add(plant, util_level, "Util_Units_Gas", gas_units, act_mask, "Act")
            add(plant, util_level, "Util_Tariff_Gas", gas_tariff, act_mask, "Act")
            add(plant, util_level, "Util_Tariff_Gas", np.full(N, p["plan_tariffs"]["Gas"]), plan_mask, "Plan")

    # Line-level build drivers
    for (plant, line), x in lines.items():
        drv_line = "" if plant == "IT01" else line
        add(plant, drv_line, C.UNITS, x["units"], act_mask, "Act")
        units_plan = np.round(x["units_exp"] * 1.03 * (1 + 0.04 * np.sin(T * 1.7 + len(line))))
        if plant == "MX01" and line == PERIMETER:
            oct_i = MONTHS.get_loc(pd.Timestamp("2026-10-01"))
            units_plan[oct_i:oct_i + 3] = units_plan[oct_i]  # copied plan (integrity check demo)
        add(plant, drv_line, C.UNITS, units_plan, plan_mask, "Plan")
        add(plant, drv_line, C.STD_HOURS, x["std"], act_mask, "Act")
        add(plant, drv_line, C.STD_HOURS, x["std"], plan_mask, "Plan")
        add(plant, drv_line, C.LABOR_HOURS, x["labor_hours"], act_mask, "Act")
        if plant == "MX01" and line == LIQUID:
            add(plant, drv_line, C.EFFICIENCY, np.full(N, 0.86), plan_mask, "Plan")

    spend = pd.DataFrame(spend_rows)
    drivers = pd.DataFrame(driver_rows, columns=["Plant", "Product_Line", "Driver", "Version", "Year", "Month", "Value"])
    mapping = pd.DataFrame([[gl, "", bucket, desc] for bucket, accts in US01_GL.items()
                            for gl, _cc, desc, _s in accts if gl not in NOT_IN_MAPPING],
                           columns=["GL_Account", "Cost_Center", "Bucket", "Description"])
    events = pd.DataFrame(event_rows, columns=["Plant", "Product_Line", "Asset", "Type", "Year", "Month", "Amount", "Source"])
    fx = pd.DataFrame([["MXN", 2026, None, 0.055], ["EUR", 2026, None, 1.08]],
                      columns=["Currency", "Year", "Month", "Rate_to_USD"])
    adjustments = pd.DataFrame([
        ["US01", "", C.MAINTENANCE, 2026, 9, 60_000, "Plant Controller US01", "Test lab chiller loop: scope increase"],
        ["MX01", LIQUID, C.DIRECT_LABOR, 2026, 10, 400_000, "Operations Manager MX01", "Training cohort for second shift"],
    ], columns=["Plant", "Product_Line", "Bucket", "Year", "Month", "Amount", "Owner", "Reason"])
    readme = pd.DataFrame({"Notes": [
        "Synthetic sample data for data center cooling equipment plants. Plants, lines, assets and figures are invented.",
        "US01 (USD): plant-level GL rows; build plan by product line (Perimeter Cooling, Chillers) rolled up to the plant.",
        "MX01 (MXN): spend and drivers by product line (Liquid Cooling, Perimeter Cooling).",
        "IT01 (EUR): plant-level rows and drivers; August summer shutdown.",
        "Actuals run from Jan-2024 to Jun-2026; the model forecasts Jul-2026 to Dec-2026.",
        "Built-in data issues: a reversal (US01 Fees, Oct-24), a zero month (IT01 Other, Aug-24), a copied build plan "
        "(MX01 Perimeter Cooling, Oct-Dec 2026) and two GL accounts missing from Mapping.",
        "FX rates are illustrative AOP rates, not market data.",
    ]})
    return {"README": readme, "Spend": spend, "Drivers": drivers, "Mapping": mapping,
            "Maintenance_Events": events, "FX": fx, "Adjustments": adjustments}


def make_sample_workbook(seed: int = 42) -> bytes:
    return write_workbook(make_sample_frames(seed))
