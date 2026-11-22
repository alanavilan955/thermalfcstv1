"""Static configuration for a data center thermal equipment plant.

Discrete manufacturing of cooling equipment (perimeter cooling units, chillers, liquid cooling
distribution units and similar). Activity is measured in earned standard hours by product line.
"""

from __future__ import annotations

APP_VERSION = "1.1"  # 1.1: flat file layout (all modules next to app.py)

MONTH_ABBR =["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------
ACT, AOP, PPF = "Act", "AOP", "PPF"
SPEND_VERSIONS = (ACT, AOP, PPF)

VERSION_ALIASES = {
    "act": ACT, "actual": ACT, "actuals": ACT, "acts": ACT, "real": ACT, "reales": ACT, "ac": ACT,
    "aop": AOP, "budget": AOP, "presupuesto": AOP, "annual plan": AOP, "plan anual": AOP, "op": AOP,
    "ppf": PPF, "prior forecast": PPF, "previous forecast": PPF, "prev forecast": PPF,
    "prior fcst": PPF, "pronostico anterior": PPF,
}

DRIVER_ACT, DRIVER_PLAN, DRIVER_AOP = "Act", "Plan", "AOP"
DRIVER_VERSION_ALIASES = {
    "act": DRIVER_ACT, "actual": DRIVER_ACT, "actuals": DRIVER_ACT, "real": DRIVER_ACT,
    "plan": DRIVER_PLAN, "forecast": DRIVER_PLAN, "fcst": DRIVER_PLAN, "s&op": DRIVER_PLAN,
    "sop": DRIVER_PLAN, "mps": DRIVER_PLAN, "build plan": DRIVER_PLAN, "pronostico": DRIVER_PLAN,
    "aop": DRIVER_AOP, "budget": DRIVER_AOP, "presupuesto": DRIVER_AOP,
}

# ---------------------------------------------------------------------------
# Spend buckets (conversion cost of a discrete equipment plant)
# ---------------------------------------------------------------------------
DIRECT_LABOR = "Direct Labor"
INDIRECT_LABOR = "Indirect Labor"
MAINTENANCE = "Maintenance"
UTILITIES = "Utilities"
CONSUMABLES = "Consumables & Tooling"
SCRAP = "Scrap & Rework"
FREIGHT = "Freight"
DEPRECIATION = "Depreciation"
FEES = "Fees"
ALLOCATIONS = "Allocations"
OTHER = "Other"

STANDARD_BUCKETS = [
    DIRECT_LABOR, INDIRECT_LABOR, MAINTENANCE, UTILITIES, CONSUMABLES, SCRAP, FREIGHT,
    DEPRECIATION, FEES, ALLOCATIONS, OTHER,
]

BUCKET_SYNONYMS = {
    "direct labor": DIRECT_LABOR, "direct labour": DIRECT_LABOR, "dl": DIRECT_LABOR,
    "mano de obra directa": DIRECT_LABOR, "mod": DIRECT_LABOR, "labor directa": DIRECT_LABOR,
    "indirect labor": INDIRECT_LABOR, "indirect labour": INDIRECT_LABOR, "il": INDIRECT_LABOR,
    "mano de obra indirecta": INDIRECT_LABOR, "moi": INDIRECT_LABOR, "fixed labor": INDIRECT_LABOR,
    "maintenance": MAINTENANCE, "mantenimiento": MAINTENANCE, "repairs and maintenance": MAINTENANCE,
    "repairs & maintenance": MAINTENANCE, "r&m": MAINTENANCE,
    "utilities": UTILITIES, "utility": UTILITIES, "energy": UTILITIES, "energia": UTILITIES,
    "servicios": UTILITIES,
    "consumables & tooling": CONSUMABLES, "consumables and tooling": CONSUMABLES, "consumables": CONSUMABLES,
    "tooling": CONSUMABLES, "supplies": CONSUMABLES, "indirect materials": CONSUMABLES,
    "consumibles": CONSUMABLES, "herramentales": CONSUMABLES, "consumibles y herramentales": CONSUMABLES,
    "materiales indirectos": CONSUMABLES,
    "scrap & rework": SCRAP, "scrap and rework": SCRAP, "scrap": SCRAP, "rework": SCRAP,
    "retrabajo": SCRAP, "scrap y retrabajo": SCRAP, "desperdicio": SCRAP, "merma": SCRAP, "chatarra": SCRAP,
    "cost of poor quality": SCRAP, "costo de no calidad": SCRAP,
    "freight": FREIGHT, "fletes": FREIGHT, "flete": FREIGHT,
    "depreciation": DEPRECIATION, "depreciacion": DEPRECIATION, "d&a": DEPRECIATION,
    "fees": FEES, "fee": FEES, "honorarios": FEES, "professional fees": FEES,
    "allocations": ALLOCATIONS, "allocation": ALLOCATIONS, "asignaciones": ALLOCATIONS,
    "other / allocations": ALLOCATIONS, "other/allocations": ALLOCATIONS, "other allocations": ALLOCATIONS,
    "other": OTHER, "others": OTHER, "otros": OTHER, "otro": OTHER,
}

# Keyword rules to suggest a bucket for GL accounts that are not in the Mapping sheet.
# Each keyword is a regular expression matched at the start of a word ("maint" matches "maintenance").
# Order matters: the first rule with a matching keyword wins.
KEYWORD_RULES = [
    (DEPRECIATION, [r"deprec", r"amortiz"]),
    (ALLOCATIONS, [r"alloc", r"asignac", r"assessment", r"charge in", r"cargo corporativo"]),
    (FREIGHT, [r"freight", r"flete", r"shipping", r"transport", r"outbound", r"inbound", r"expedit"]),
    (SCRAP, [r"scrap", r"rework", r"retrabajo", r"chatarra", r"desperdicio", r"merma", r"quality cost",
             r"no calidad"]),
    (MAINTENANCE, [r"maint", r"repair", r"spare", r"parts\b", r"manten", r"refacc", r"reparac"]),
    (CONSUMABLES, [r"consumab", r"consumib", r"tooling", r"tool\b", r"herramient", r"herramental", r"supplies",
                   r"suministro", r"abrasiv", r"welding", r"brazing", r"soldadura", r"nitrogen", r"nitrogeno",
                   r"argon"]),
    (UTILITIES, [r"electric", r"power\b", r"natural gas", r"gas\b", r"water", r"sewer", r"steam", r"utilit",
                 r"energ", r"agua\b", r"vapor\b", r"luz\b"]),
    (FEES, [r"fees?\b", r"consult", r"professional", r"legal\b", r"audit", r"honorar", r"license", r"licencia",
            r"certif", r"calibrat", r"calibrac"]),
    (DIRECT_LABOR, [r"hourly", r"direct lab", r"overtime", r"temp lab", r"temporary lab", r"agency lab",
                    r"mano de obra directa", r"tiempo extra", r"eventual"]),
    (INDIRECT_LABOR, [r"salar", r"supervis", r"indirect lab", r"benefit", r"bonus", r"mano de obra indirecta",
                      r"sueldo", r"prestacion"]),
]

# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------
M_DEFAULT = "Default"
M_AUTO = "Auto"
M_DRIVER = "Driver"
M_ENSEMBLE = "ETS+ARIMA"
M_SNAIVE = "Seasonal naive"
M_NAIVE = "Naive"
M_RR3 = "Run-rate 3m avg"
M_RR6 = "Run-rate 6m median"
M_REG = "Activity regression"
M_AOPADJ = "AOP adjusted"

METHOD_CHOICES = [M_DEFAULT, M_AUTO, M_DRIVER, M_ENSEMBLE, M_SNAIVE, M_NAIVE, M_RR3, M_RR6, M_REG, M_AOPADJ]

DEFAULT_METHOD = {
    DIRECT_LABOR: M_DRIVER,
    UTILITIES: M_DRIVER,
    MAINTENANCE: M_DRIVER,
    DEPRECIATION: M_RR3,
    FEES: M_RR6,
    INDIRECT_LABOR: M_AUTO,
    CONSUMABLES: M_AUTO,
    SCRAP: M_AUTO,
    FREIGHT: M_AUTO,
    ALLOCATIONS: M_AUTO,
    OTHER: M_AUTO,
}
DRIVER_BUCKETS = {DIRECT_LABOR, UTILITIES, MAINTENANCE}

METHOD_DESCRIPTIONS = {
    M_AUTO: "Lowest backtest error among ETS+ARIMA, naive and seasonal naive "
            "(plus activity regression when the bucket behaves as variable).",
    M_DRIVER: "Driver formula for the bucket (labor, utilities or maintenance).",
    M_ENSEMBLE: "Average of damped-trend exponential smoothing and ARIMA.",
    M_SNAIVE: "Same month of the prior year.",
    M_NAIVE: "Last closed month repeated.",
    M_RR3: "Average of the last 3 clean months.",
    M_RR6: "Median of the last 6 clean months.",
    M_REG: "Fixed part plus a variable rate per earned hour times the activity plan.",
    M_AOPADJ: "AOP phasing scaled by year-to-date actual vs AOP.",
}

# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------
UNITS = "Units"
STD_HOURS = "Std_Hours_per_Unit"
EARNED_HOURS = "Earned_Hours"
LABOR_HOURS = "Labor_Hours"
EFFICIENCY = "Labor_Efficiency"
WAGE = "Wage_Rate"
UTIL_UNITS = "Util_Units"
UTIL_TARIFF = "Util_Tariff"
ACTIVITY = "Activity"  # earned hours when standards are loaded, otherwise units

DRIVER_ALIASES = {
    "units": UNITS, "unidades": UNITS, "units produced": UNITS, "unidades producidas": UNITS,
    "equipos": UNITS, "production units": UNITS, "output units": UNITS, "builds": UNITS,
    "std hours per unit": STD_HOURS, "standard hours per unit": STD_HOURS, "std hrs per unit": STD_HOURS,
    "std hrs/unit": STD_HOURS, "horas estandar por unidad": STD_HOURS, "routing hours": STD_HOURS,
    "earned hours": EARNED_HOURS, "standard hours": EARNED_HOURS, "earned standard hours": EARNED_HOURS,
    "std hours": EARNED_HOURS, "horas ganadas": EARNED_HOURS, "horas estandar": EARNED_HOURS,
    "labor hours": LABOR_HOURS, "direct labor hours": LABOR_HOURS, "dl hours": LABOR_HOURS,
    "actual hours": LABOR_HOURS, "horas mano de obra": LABOR_HOURS, "horas directas": LABOR_HOURS,
    "horas reales": LABOR_HOURS, "clocked hours": LABOR_HOURS,
    "labor efficiency": EFFICIENCY, "efficiency": EFFICIENCY, "eficiencia": EFFICIENCY,
    "eficiencia mano de obra": EFFICIENCY,
    "wage rate": WAGE, "wage": WAGE, "$/hr": WAGE, "rate per hour": WAGE, "labor rate": WAGE,
    "tarifa hora": WAGE, "salario hora": WAGE,
}

# Drivers that are averaged (rates) when duplicated or rolled up; the rest are summed (quantities).
RATE_DRIVERS = {STD_HOURS, EFFICIENCY, WAGE}
QUANTITY_DRIVERS = {UNITS, EARNED_HOURS, LABOR_HOURS}

# ---------------------------------------------------------------------------
# Thresholds and model settings
# ---------------------------------------------------------------------------
VAR_SHARE_VARIABLE = 0.70
VAR_SHARE_FIXED = 0.30
P_VALUE_MAX = 0.05
CV_STABLE = 0.10
CV_VARIABLE = 0.25
INTERMITTENT_ZERO_SHARE = 0.30
OUTLIER_Z = 3.5
OUTLIER_MIN_DEV = 0.15  # an atypical item must also deviate 15% or more from its baseline
OUTLIER_WINDOW = 5
BACKTEST_POINTS = 6
HORIZONS = (1, 3)
ACCURACY_TARGET = 0.90
CALIBRATION_MONTHS = 6
GL_MIX_MONTHS = 6
CLASSIFY_MONTHS = 24
INTERVAL_Z = 1.2816  # two-sided 80% normal interval
MIN_SEASONAL_OBS = 24
MIN_TRAIN_OBS = 12
MIN_HISTORY_FOR_MODELS = 6

# ---------------------------------------------------------------------------
# Palettes (dark is the default theme; light keeps the One View navy and red)
# ---------------------------------------------------------------------------
NAVY = "#132A5E"
RED = "#E4002B"

PALETTES = {
    "dark": {
        "actual": "#7FB2FF", "forecast": "#FF4D6A", "band": "rgba(255,77,106,0.18)",
        "aop": "#AEB7C8", "ppf": "#6E7891", "base": "#8C96AA", "later": "#7FB2FF",
        "bar_actual": "#3E6FD8", "bar_forecast": "#E4002B",
        "font": "#E6EAF2", "title": "#FFFFFF", "grid": "#26314A", "axis": "#3A4660",
        "up": "#FF4D6A", "down": "#3FA7F5", "total": "#7D879C",
        "heat": ["#E4002B", "#5C1A2B", "#1B2438", "#3E8EF7"], "heat_text": "#FFFFFF",
        "template": "plotly_dark", "hover_bg": "#1B2438", "bg": "#0B1220",
    },
    "light": {
        "actual": NAVY, "forecast": RED, "band": "rgba(228,0,43,0.10)",
        "aop": "#8C94A6", "ppf": "#C9CED8", "base": "#8C94A6", "later": NAVY,
        "bar_actual": NAVY, "bar_forecast": RED,
        "font": "#1A1A1A", "title": NAVY, "grid": "#E6E9EF", "axis": "#C9CED8",
        "up": RED, "down": NAVY, "total": "#8C94A6",
        "heat": [RED, "#F6C9D1", "#FFFFFF", NAVY], "heat_text": "#1A1A1A",
        "template": "plotly_white", "hover_bg": "#FFFFFF", "bg": "#FFFFFF",
    },
}
