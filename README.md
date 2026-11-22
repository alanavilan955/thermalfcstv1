# Thermal Plant Spend Forecast

Streamlit app (dark theme) that classifies the conversion spend of plants that build data center cooling
equipment (perimeter cooling units, chillers, liquid cooling distribution units and similar) and forecasts
the remaining months of the year (year-end landing).

Activity is measured in earned standard hours (units x standard hours per unit, by product line), so a
change in product mix does not distort the drivers. Direct Labor, Utilities and Maintenance use driver
formulas, Depreciation and Fees use run-rates, and the other buckets use ETS+ARIMA or an activity
regression chosen by backtest. Accuracy is 1 - WAPE in a rolling backtest at 1 and 3 months, against a
90% target.

The repository contains no data files. The app generates a synthetic sample in memory; its plants, product
lines, assets, accounts and figures are invented and do not represent any company.

Version 1.1 uses a flat layout: every Python file sits next to `app.py` at the repository root, so the app
runs even when the files are uploaded with GitHub's "choose your files" dialog, which does not upload folders.
The sidebar shows the version number, so you can confirm which version Community Cloud is running.

## What the app does

1. Reads an Excel workbook (`.xlsx`, `.xlsm` or `.xlsb`). Months can be columns or rows. Rows can be at
   bucket level or at GL account and cost center level (mapped to buckets), by plant and optional product line.
2. Builds the activity of each plant and product line: earned hours = units x standard hours per unit.
   When spend is at plant level and the build plan is by product line, the lines are summed to the plant
   (rates such as wage or tariffs are weighted by hours or consumption).
3. Classifies every series: bucket mapping for GL accounts, fixed vs variable cost (regression on earned
   hours), forecastability (volatility around a linear trend, so a product ramp is not counted as noise)
   and atypical items.
4. Forecasts the open months, backtests each method and reports accuracy, bias and forecast value added
   (FVA) against a naive forecast.
5. Shows the FY landing against AOP and the prior forecast (PPF), a waterfall by bucket and the
   conversion cost per earned hour by month.
6. Runs scenarios (build plan, labor efficiency, wage rate, energy tariff) and management adjustments.
7. Exports a workbook with the landing, monthly forecast in local currency and USD, GL detail,
   classification, atypical items, accuracy, method selection, backtest points, adjustments, assumptions
   and data checks.

## Methods by bucket

| Bucket | Default method |
|---|---|
| Direct Labor | earned hours of the build plan / labor efficiency x wage rate x calibration factor (actual vs formula, last 6 months). Efficiency = earned / actual labor hours; it is forecast with ETS+ARIMA unless a Labor_Efficiency plan is loaded. Without labor hours, the factor becomes a $ per earned hour. |
| Utilities | consumption per earned hour (ETS+ARIMA) x earned hours x tariff, plus fixed charges (median of the last 6 months). One pair of drivers per utility type (electricity, gas...). |
| Maintenance | unplanned base (ETS+ARIMA on actual minus planned events, atypical months removed) plus the planned events of each month (test stands, brazing furnaces, paint lines, shutdowns). |
| Depreciation | average of the last 3 clean months. |
| Fees | median of the last 6 clean months. Switch to Auto when fees follow a calendar (for example certification testing twice a year). |
| Indirect Labor, Consumables & Tooling, Scrap & Rework, Freight, Allocations, Other | Auto: lowest backtest error among ETS+ARIMA, naive and seasonal naive; an activity regression (fixed + rate per earned hour) also competes when the bucket behaves as variable. |

Every bucket can be switched to another method in the sidebar.

## Atypical items

A month is atypical when its modified z-score against a 5-month rolling median is above 3.5 and it
deviates 15% or more from that median. A month is not flagged when the same calendar month of another year
shows the same deviation: a recurring pattern (August shutdown, December holidays, quarter-end allocations)
is seasonality. Negative values are flagged as possible reversals. Excluded months are replaced by the
rolling median when models are fitted; actuals are not changed. Every flag can be overridden in the app.

## Accuracy

- Accuracy = 1 - WAPE, where WAPE = sum |actual - forecast| / sum |actual|.
- Backtest: each of the last 6 closed months is forecast 1 and 3 months before it, using only data
  available at that time. Plant and network totals use only months where every series has a point.
- Network totals are in USD at the AOP rate (constant currency).
- Driver buckets are backtested with actual units, hours, wage and tariffs because historical plans are
  not loaded. Their score measures the driver formula, not the error of the build plan.
- The approximate 80% interval is +/- 1.28 x the backtest RMSE for each horizon.

## Input workbook

Download the template and the synthetic sample from the app sidebar.

| Sheet | Required | Columns |
|---|---|---|
| Spend | Yes | Plant, Product_Line (optional), Currency, Version (Act, AOP, PPF), Year, Bucket or GL_Account + Cost_Center + Description, then Jan..Dec. Long format: Year, Month, Value. |
| Drivers | No | Plant, Product_Line, Driver, Version (Act or Plan), Year, Month, Value. Drivers: Units, Std_Hours_per_Unit or Earned_Hours, Labor_Hours, Labor_Efficiency (plan), Wage_Rate, Util_Units_<Type>, Util_Tariff_<Type>. |
| Mapping | No | GL_Account and/or Cost_Center, Bucket. |
| Maintenance_Events | No | Plant, Product_Line, Asset, Type, Year, Month, Amount, Source. Include past events too. |
| FX | No | Currency, Year (optional), Month (optional), Rate_to_USD (USD per 1 unit of local currency, AOP rate). |
| Adjustments | No | Plant, Product_Line, Bucket, Year, Month, Amount (plant currency), Owner, Reason. |

Spanish column, bucket and driver names are accepted (Planta, Linea de producto, Rubro, Mes, Monto,
Unidades, Horas estandar, Retrabajo, Consumibles...). Plant-level spend uses plant-level drivers; when a
driver exists only by product line, the lines are summed. Every product line with recent actual activity
needs a build plan, otherwise the plant total is not built and the driver buckets fall back to Auto.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501 in dark mode with the synthetic sample loaded. Upload your workbook
in the sidebar. The theme is set in `.streamlit/config.toml`; charts follow the active theme.

## Deploy on Streamlit Community Cloud

1. In the GitHub repository, open "Add file" > "Upload files" > "choose your files". Open the unzipped
   folder, select every file in it (Ctrl+A on Windows, Cmd+A on Mac) and commit. All files go to the
   repository root, next to `app.py`. Uploading a file with an existing name replaces it.
2. Dark theme: open "Add file" > "Create new file", type `.streamlit/config.toml` as the name (the slash
   creates the folder), paste the content below and commit. Without this file the app follows the browser
   or system theme.
3. New app only: sign in to https://share.streamlit.io with GitHub, create the app from the repository with
   branch `main` and main file path `app.py`, open "Advanced settings", select Python 3.12 and deploy.
   Python 3.12, 3.13 and 3.14 were tested. The Python version can only be chosen there; changing it later
   requires deleting and redeploying the app.
4. Existing app: Community Cloud picks up the new commit by itself. If the old error stays on screen, open
   "Manage app" (lower right corner of the app) and choose "Reboot app".

Content of `.streamlit/config.toml`:

```toml
[theme]
base = "dark"
primaryColor = "#E4002B"
backgroundColor = "#0B1220"
secondaryBackgroundColor = "#151F33"
textColor = "#E6EAF2"
borderColor = "#26314A"
dataframeHeaderBackgroundColor = "#151F33"

[theme.sidebar]
backgroundColor = "#0F1729"
secondaryBackgroundColor = "#1A2540"

[server]
maxUploadSize = 50

[browser]
gatherUsageStats = false
```

## Troubleshooting

- "The app cannot start because the deployment is incomplete": the message lists the files missing next to
  `app.py` or the packages that were not installed. Upload the listed files to the repository root, or check
  that `requirements.txt` is at the root, then reboot the app.
- `ModuleNotFoundError` at `from spendcast import charts`: the repository still runs version 1.0, which needs
  the `spendcast/` folder. Upload the version 1.1 files (they replace `app.py`).

## Data privacy

The app keeps uploaded files in memory and does not save them. On Community Cloud that memory is on
Streamlit's servers, so check your company's data policy before uploading real figures; for confidential
data, run the app locally. If the repository is public, anyone can read its files: never upload workbooks
with real figures to it. `.gitignore` keeps Excel and CSV files out of commits made with git.

## Tests

```bash
pip install pytest
pytest -q
```

23 tests in `test_core.py` cover metrics, period parsing, wide and long formats, double counting between GL and bucket rows,
classification (including recurring shutdown months), earned-hour roll-up from product lines to plant, the
plan completeness rule, scenario arithmetic, the landing identity, cost per earned hour, GL allocation and an
end-to-end run on the sample data.

## Project structure

```
app.py               Streamlit interface; checks that every file below is present before it starts
sc_config.py         buckets, drivers, default methods, thresholds, palettes, version
sc_io_excel.py       workbook reading, format detection and data checks
sc_driverprep.py     earned hours and roll-up of product lines to plant level
sc_classify.py       mapping, fixed vs variable, forecastability, atypical items
sc_models.py         naive, seasonal naive, run-rates, ETS, ARIMA, regression
sc_drivers.py        Direct Labor, Utilities and Maintenance formulas; scenario recalculation
sc_engine.py         backtest, method selection, final forecast and activity table
sc_scenario.py       landing table, adjustments, bridge, GL allocation, cost per earned hour
sc_charts.py         Plotly figures
sc_export.py         output workbook and input template
sc_sample.py         synthetic sample data (three plants, three product lines)
sc_context.py        per-series data shared by the methods (clean history, drivers, events)
sc_fx.py             conversion to USD at the AOP rate
sc_metrics.py        WAPE, accuracy, bias and RMSE
sc_utils.py          text, account code, month and number cleaning helpers
test_core.py         23 tests (pytest)
requirements.txt     pinned packages
.streamlit/config.toml  dark theme (create it in GitHub as described above)
```

## Known limits

- Material cost, material usage variance and purchase price variance are out of scope; the model covers
  conversion spend.
- Run time grows with the number of series; results are cached per file and settings.
- Reading `.xlsb` relies on `pyxlsb`, which returns dates as Excel serial numbers; the app converts them.
  It has not been tested with a real `.xlsb` workbook.
