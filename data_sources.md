# Data Sources: ERCOT Generation vs. Henry Hub Gas Futures

## Project context

We are testing whether intraday changes in ERCOT gas-fired generation, driven by solar output and unplanned plant outages, show up in Henry Hub natural gas futures prices. The signal is built from **surprises** (actual vs. what was forecast or known at the time), so preserving *when* each data point became public is as important as the data itself.

**Sample window:** December 11, 2023 to present. This is the earliest date available through the ERCOT Public API, and it keeps every dataset on the same window.

---

## 1. Henry Hub natural gas futures (intraday prices)

| | |
|---|---|
| **Source** | Databento, dataset `GLBX.MDP3` (CME Globex) |
| **Symbol** | `NG.v.0` with `stype_in="continuous"` (front month, rolls by volume) |
| **Schema** | `ohlcv-1m` (1-minute bars) |
| **Access** | Existing Databento account (has $125 in credits) |

**Notes**
- Run `client.metadata.get_cost(...)` before pulling to confirm the cost. Do **not** pull `mbo`, `trades` or `mbp` schemas, which are far larger and more expensive.
- Keep the instrument ID with each bar. The continuous series jumps on roll days, so returns that span a contract change must be flagged or dropped.
- Bars exist only for minutes with trades. Leave gaps as gaps; do not forward-fill.
- Timestamps are UTC (nanoseconds since epoch).

---

## 2. ERCOT solar and wind: actuals and forecasts

| | |
|---|---|
| **Source** | ERCOT Public API (free registration at apiexplorer.ercot.com, then subscribe to the "Public API" product) |
| **Reports** | Solar Power Production, Hourly Averaged Actual and Forecasted Values (NP4-745-CD) and its system-wide equivalent; the matching wind reports |
| **Frequency** | Hourly, plus 5-minute actuals where available |

**Notes**
- **Keep every forecast vintage, not just the latest.** Each forecast should be stored with its publish timestamp. The analysis needs "what was the forecast as of time T," and overwriting old vintages would make the study impossible.
- Timestamps are Central Prevailing Time. Watch for the duplicated and skipped hours at DST transitions; store a UTC timestamp alongside.

**Day-before solar forecast error** (`pipeline/solar_forecast_error.py` → `data/clean/solar_forecast_error.parquet`; "Solar forecast error" page)
- For each delivery hour:
  - **Forecast:** the latest NP4-737-CD vintage posted at or before **10:00 CT the day before** (Day-Ahead Market close), kept for both `STPPF` and `PVGRPP`. `forecast_posted_ct` records which vintage was used.
  - **Actual:** the final actual.
  - **Error:** `error = actual − forecast`. Positive means more solar than expected.
- **Night:** hours where actual and forecast are both below 100 MW are excluded from stats, z and the HMM.
- The page uses the same z / MW / HMM cutoffs as the ramp pages, on the hourly error. On the Henry Hub chart, large surprises are shaded by direction.
- **Corrupt ERCOT postings:** 25 consecutive NP4-737-CD postings (2024-08-22 15:55 → 2024-08-23 15:55 CT) have every value inflated about 6.5× (actuals up to 134 GW).
  - The build drops any posting with 3 or more hours above a monthly plausibility cap. The cap is 1.6 × the month's p95 of each hour's minimum actual across vintages.
  - Dropped postings are listed in `data/clean/solar_forecast_error_dropped_docs.csv`.
  - As a result, the 2024-08-24 day-before forecast comes from an older posting (flagged as stale).
  - The cleaned `ercot_np4_737_cd.parquet` itself still contains these postings, so filter them in any other use.
- **Single implausible hour:** 2024-04-29 14:00 CT has an actual of 34 GW in every posting (true value about 16.5 GW). It is blanked.
- **Missing hours:** the repeated 1 AM hour on fall-back days is missing from the source report (night, so no effect).
- Full sample (2023-12-12 → 2026-11-04, daytime hours):
  - STPPF: bias −983 MW, MAE 1,610 MW.
  - PVGRPP: bias +2,883 MW (conservative by design).

---

## 3. ERCOT fuel mix (actual generation by fuel)

| | |
|---|---|
| **Source** | ERCOT Public API; yearly Fuel Mix spreadsheets on ercot.com/gridinfo/generation as backup |
| **Frequency** | 15-minute settlement intervals (5-minute where available) |
| **Key column** | Natural gas generation (MW); also solar, wind, storage, coal, nuclear |

**Notes**
- The yearly spreadsheets are formatted for people, not code (sheets by month, separate date and interval columns), so expect cleanup if they're used.
- Same Central-time and DST handling as above.

**As implemented** (`pipeline/clean_fuel_mix.py` → `data/clean/ercot_fuel_mix.parquet`)
- Source: the yearly `IntGenbyFuel<YYYY>.xlsx` workbooks, downloaded by hand into `data/raw/ercot/fuel-mix-*/`. The Public API has no fuel-mix endpoint.
- Long format: one row per 15-minute interval per fuel. Fuels are Biomass, Coal, Gas, Gas-CC, Hydro, Nuclear, Other, Solar, WSL and Wind. **Gas-fired generation = Gas + Gas-CC.** WSL (wholesale storage load, i.e. battery charging) is negative.
- **Units:** the workbook values are **MWh per 15-minute interval**, kept as `mwh`. `mw_avg = mwh × 4`. Checked against NP4-733 5-minute wind: correlation 0.9998, median ratio 1.001.
- **DST:** the workbooks label intervals by ending time, with the fall-back repeat in extra `01:15 (DST)`…`02:00 (DST)` columns (the second, standard-time hour) and blanks for the skipped spring hour. The parser places intervals by position from local midnight, giving 92/96/100 intervals per day, and checks that each day's intervals sum to the sheet's `Total`.
- **No publish timestamp:** this is an after-the-fact settlement report. `settlement_type` is `INITIAL` for recent days and later revised to `FINAL`. The parser keeps FINAL over INITIAL when files overlap.
- **Coverage:** 2023-01-01 to 2026-09-30 as currently downloaded (2026-08-01 onward is INITIAL). **Re-download the 2026 workbook to extend it.** Save it as a new dated copy (e.g. `fuel-mix-2026/IntGenbyFuel2026_2026-12-01.xlsx`) rather than overwriting, so the INITIAL vintages stay reproducible. The parser picks up every `IntGenbyFuel*.xlsx`, and the newer file wins ties.

**Ramp rates** (`pipeline/ramps.py`, `pipeline/ramp_hmm.py`; "Gas ramps" and "Solar ramps" pages in Streamlit)
- Fuels are computed separately: `Gas` and `Gas-CC`, and `Solar`. The forward difference is `d(t) = mw_avg(t+15 min) − mw_avg(t)`, in MW per 15 min (÷15 for MW/min), on the UTC series, so DST days and midnights need no special handling.
- A difference is taken only when the next row is exactly 15 minutes later; anything else is a gap and is never bridged. The last interval of a range is dropped, and no data outside the range is used.
- Built-in data checks:
  - interval count vs expected;
  - non-15-minute steps;
  - duplicates;
  - missing or negative MW;
  - INITIAL days;
  - ramps across yearly-workbook seams;
  - the largest ramps.
- Full dataset (2023-01-01 → 2026-09-30): 0 gaps, 0 duplicates, 0 missing. 61 INITIAL days (Aug–Sep 2026). 3 seams, each with an ordinary-sized ramp.
- **Solar nights:** steps where both ends are below 100 MW are marked `excluded`. They are left out of the stats, histograms, z and the HMM, and drawn gray. Otherwise half the steps would be ~0 and the std would be meaningless.
- CLI: `.venv/bin/python pipeline/ramps.py --fuels gas|solar --start 2026-09-24 --end 2026-09-30` (or `--all`). Add `--hmm 0.95` to fit the regime HMM.
- **Large ramp vs noise (Streamlit cutoff):** a visual first pass. A ramp is "large" if it falls beyond a ± cutoff on the histogram's horizontal axis. The cutoff is set either:
  - in z (default ±2): `z = (d − mean) / std`, using each fuel's own mean and std over the selected range; or
  - in MW per 15 min: `|d| ≥ cutoff`; or
  - as an **HMM regime**: a 3-state (down / flat / up) Gaussian HMM per fuel, numpy only.
    - Transitions are fixed by a stickiness `p_stay` (default 0.95). Emission means and variances are fitted by EM.
    - States are decoded with Viterbi. Up or down = large ramp.
    - It cuts flicker: on 2026-09-24 → 30, class switches drop from 131 to 47 (Gas) and from 170 to 30 (Gas-CC), compared with ±1 z.
    - **Look-back only:** Viterbi uses the whole selected range, including later steps, so it is not a real-time signal. A real-time version would use forward-filtered probabilities.
    - It finds *sustained* ramping, including moderate ramps, so it flags far more steps than ±2 z (about 45–60% vs 7%).
- A z cutoff adapts to the selected range, so the same z gives different MW edges for different ranges. The page shows the MW equivalent.


**Solar-regime volatility** (`pipeline/regime_vol.py`; "Solar regime volatility" page)
- **Question:** is Henry Hub front-month volatility different while ERCOT solar is ramping up or down (solar ramps HMM), compared with everything else (flat daytime + night)?
- **Returns:** 15-minute NG log returns in basis points. The price at each :00/:15/:30/:45 is the last 1-minute close in that window.
  - Returns are kept only between consecutive windows on the same contract, which drops the halt, weekends, no-trade windows and rolls.
  - Each return is matched to the solar step covering the same 15 minutes.
- **Time-of-day adjustment:** returns are divided by the expected RMS return for their 15-minute CT slot.
  - **By month (default):** a profile per calendar month and slot from non-Thursdays, times a per-slot Thursday factor (storage report). This follows the seasonal shift of sunrise and the morning pattern.
  - **Whole range:** one profile per slot, with Thursdays separate.
- **Model:** GARCH(1,1)-X with Student-t errors, hand-written in scipy because `arch` can't put regressors in the variance equation.
  - `h_t = g_t·exp(θ_up·Up + θ_dn·Down)`. The volatility ratio is `exp(θ/2)`, tested with likelihood-ratio tests.
  - A simulation test recovers all parameters within about 1 SE.
- **Model-free check:** RMS ratio with a day-block bootstrap CI.
- **Placebo test (the main check):** the analysis is re-run with each day's solar labels moved by ±1, 2, 3 and 4 weeks. That keeps the weekday, time of day and roughly the season, but uses the wrong day's solar.
  - **Effect beyond placebo** = real ratio ÷ geometric-mean placebo ratio.
  - Solar labels are decoded over the range ±4 weeks, so short ranges still have shifted labels.
  - Without the placebo, the GARCH ratio looked like a 10% up-ramp effect with p < 1e-14, and the wrong-day labels gave the same number.
- **Identification:** only about 40% of up-ramp and down-ramp returns fall in month × slot cells where ramping is neither almost always nor almost never the case. The rest is absorbed by the time-of-day profile by construction.
- **Results, full sample 2023-12-11 → 2026-09-30, p_stay 0.95, by-month profile:**

| | Real | Placebo | Effect beyond placebo |
|---|---|---|---|
| GARCH, up-ramps | 1.040× | 1.034× (1.026–1.038) | 1.006× |
| GARCH, down-ramps | 0.951× | 0.953× | 0.997× |
| Model-free, up-ramps | 0.988× | 0.981× | 1.008× |
| Model-free, down-ramps | 0.984× | 0.976× | 1.009× |

  - Raw (no time-of-day adjustment): up-ramps 1.73×, down-ramps 0.57×, almost all of it time of day.
  - **Conclusion so far:** solar ramps show no volatility effect on Henry Hub beyond about 1%. That is at the edge of the placebo spread.
- **Caveats:**
  - The HMM labels are look-back (Viterbi), so this measures association.
  - The time-of-day profile is estimated over the same range.
  - Returns are chained across gaps.
- CLI: `.venv/bin/python pipeline/regime_vol.py --all --p-stay 0.95 [--source solar|gas|solar_error] [--simple-profile]` (prints identification, adjusted and raw fits, ratios, placebo results and checks).

**Gas regime volatility** (same method; "Gas regime volatility" page; `--source gas`)
- **Regimes:** come from one combined series, `Gas + Gas-CC` (`ramps.combine_fuels`, "Gas total"). An interval is kept only if both fuels are present. The 3-state HMM on its 15-minute changes gives up / flat / down.
- **No night exclusion:** gas runs all day, so out-of-regime = flat.
- **Identification is much better than for solar:** about 90% of gas ramp returns fall in month × slot cells that are sometimes ramping and sometimes not (solar: about 40%).
- **Results, full sample 2023-12-11 → 2026-09-30, p_stay 0.95, by-month profile:**

| | Real | Effect beyond placebo |
|---|---|---|
| GARCH, up-ramps | 0.994× (p = 0.56) | 0.990× |
| GARCH, down-ramps | 0.982× (p = 0.04) | 0.983× |
| Model-free, up-ramps | 1.007× (CI 0.976–1.043) | 1.009× |
| Model-free, down-ramps | 0.993× (CI 0.970–1.019) | 1.002× |

  - Raw (no time-of-day adjustment): up-ramps 0.93×, down-ramps 1.07×.
  - **Conclusion so far:** no volatility effect from gas ramps beyond about ±2%, and the two methods disagree on direction.

**Solar surprise volatility** (same method; "Solar surprise volatility" page; `--source solar_error`)
- **Regimes:** an HMM on the hourly error (actual − day-before 10:00 CT STPPF, from `solar_forecast_error.parquet`) labels each hour above / near / below forecast. Each hour's label applies to its four 15-minute NG returns.
- **Comparison:** "more solar than forecast" and "less solar than forecast" are each compared against near-forecast hours plus night (both < 100 MW).
- **Coverage:** runs to 2026-11-04, past the fuel-mix end.
- **Identification is the best of the three:** 96–99% of surprise returns fall in month × slot cells that are mixed, because surprises don't follow the clock.
- **Results, full sample 2023-12-11 → 2026-11-04, p_stay 0.95, by-month profile (67,682 returns):**

| | Real | Effect beyond placebo |
|---|---|---|
| GARCH, more solar | 0.987× (p = 0.35) | 0.975× |
| GARCH, less solar | 1.003× (p = 0.85) | 1.003× |
| Model-free, more solar | 0.978× (CI 0.94–1.02) | 0.989× |
| Model-free, less solar | 0.998× (CI 0.95–1.04) | 1.013× |

  - Joint LR test p = 0.62.
  - Raw (no time-of-day adjustment): 1.30× and 1.20×, because surprises are largest midday.
- **Robustness:** defining surprises with a cutoff instead of the HMM (|z| ≥ 1, or |z| ≥ 2 for the 347 / 1,421 largest hours) also gives effects beyond placebo of 0.98–1.05×, all inside the placebo spread.
- **Conclusion so far:** day-before solar forecast surprises show no detectable effect on Henry Hub 15-minute volatility.
- **Caveat:** the surprise is measured against the *day-before* forecast. By the delivery hour, ERCOT's intraday forecasts and real-time data have already revealed most of it, so the hour itself may not be when the news arrives. Testing against the latest forecast before each hour (an intraday surprise) is the natural next check.

**Price direction** (`pipeline/price_direction.py`; "Price direction" page)
- **Question:** does NG tend to rise or fall during a regime? Same 15-minute returns and labels as the volatility pages, for any of the three sources.
- **Mean return:** in-regime minus rest (bp per 15 min), with a day-block bootstrap CI, the 8-shift wrong-day placebo, and a by-year breakdown.
- **Event study:** average cumulative return from 2 h before to 4 h after each regime episode starts, against the placebo band.
- **Expected sign for solar surprises:** less solar than forecast → more gas burn → positive.
- **Results, full sample:**
  - **Solar surprises:**
    - *More solar than forecast:* +1.07 bp (CI +0.05 to +2.17, t = 2.0), just outside the placebo range (−0.81 to +0.93). That is the *wrong* sign.
    - *Less solar than forecast:* +0.47 bp (t = 1.0), inside the placebo range.
    - The "more solar" effect is not stable: the first half (Dec 2023 – May 2025) gives +0.24 bp (t = 0.3) and the second half +2.31 bp (t = 2.8). By year: 2024 +0.41, 2025 +1.08, 2026 +1.76, all with CIs that include 0.
    - The event-study paths stay inside the placebo bands.
  - **Solar ramps and gas ramps:** all mean differences fall inside the placebo ranges (|t| ≤ 1.2).
- **Conclusion so far:** no reliable directional effect.
  - The one marginal result (more solar → slightly higher prices, recent period only) has the opposite sign to the gas-burn story.
  - It could be weather confounding: sunnier-than-forecast days are often hotter, so load rises.
  - It is also one of many tests run.
- CLI: `.venv/bin/python pipeline/price_direction.py --source solar_error|solar|gas --all`.

---

## 4. ERCOT outages

### 4a. Hourly Resource Outage Capacity (NP3-233-CD): primary

| | |
|---|---|
| **Source** | ERCOT Public API / Data Access Portal |
| **Content** | Aggregated outaged capacity (MW) by load zone for the next 168 hours, including full outages and derates |
| **Frequency** | Published every hour |

**Notes**
- **Store every hourly publication, not just the latest one.** Each report forecasts the next 168 hours, and the analysis compares each hour's reported outages against what earlier reports predicted for that same hour. An unexpected jump is our proxy for forced outages. This only works if all vintages are kept.
- This report combines planned, forced and maintenance outages; the comparison above is how we separate them.

### 4b. Unplanned Resource Outages Report (NP1-346-ER): validation only

| | |
|---|---|
| **Source** | ercot.com data products page (zip/xlsx) |
| **Content** | Unit-level unplanned outages and derates, including fuel type and MW |
| **Frequency** | Daily, covering the day **three days before** posting |

**Notes**
- Because of the 3-day lag, this data was **not** available to the market in real time. It is used only to check that jumps in 4a correspond to real forced outages of gas units, never as a same-day signal.
- Record both the outage dates and the report's posting date.

---

## General requirements

- **Store raw pulls untouched** (e.g. `data/raw/`), with cleaned versions kept separately. We need to be able to reproduce everything.
- **Every record needs two timestamps where applicable:** the time the data *describes* and the time it was *published*.
- **Normalize to UTC** in cleaned data, keeping the original local time column.
- **Save a pull log** (source, report ID, date range, pull date, row counts) so gaps can be found later.
- **Out of scope:** regional gas hub prices (Waha, Houston Ship Channel) and ERCOT power prices.

---

## Pipeline (how the data is pulled)

Credentials live in `.env` (gitignored). Run everything with the project venv.

| Command | What it does |
|---|---|
| `.venv/bin/python pipeline/pull_databento.py --cost-only` | Prints the Databento cost; pulls nothing |
| `.venv/bin/python pipeline/pull_databento.py` | Cost check (aborts above `--max-cost`, default $5), then pulls NG 1-minute bars |
| `.venv/bin/python pipeline/pull_ercot.py all` | Lists and bulk-downloads every posted ERCOT document (resumable) |
| `.venv/bin/python pipeline/pull_ercot.py all --clean` | Rebuilds the cleaned ERCOT parquet files from raw |
| `.venv/bin/python pipeline/clean_fuel_mix.py` | Parses the fuel-mix workbooks in `data/raw/ercot/fuel-mix-*/` into `ercot_fuel_mix.parquet` |
| `.venv/bin/python pipeline/ramps.py --fuels gas --all` | Prints ramp-rate stats and data checks (`--fuels solar`, `--hmm 0.95`) |
| `.venv/bin/python pipeline/solar_forecast_error.py` | Builds the hourly day-before solar forecast vs actual table (drops corrupt postings) and prints checks |
| `.venv/bin/python pipeline/regime_vol.py --all [--source gas]` | GARCH, model-free and placebo volatility ratios for Henry Hub inside vs outside solar (or combined gas) ramp regimes |
| `.venv/bin/python pipeline/price_direction.py --source solar_error --all` | Mean-return and event-study direction tests against the placebo |
| `.venv/bin/python tools/github_push.py -m "message"` | Commits and pushes to GitHub using `GITHUB_USER`/`GITHUB_TOKEN` from `.env` (`--check` = dry run) |
| `.venv/bin/streamlit run streamlit_app.py` | App with eight pages: "Data check" (one day of each dataset), "Gas ramps", "Solar ramps", "Solar forecast error" (histograms, cutoffs, HMM), "Solar regime volatility", "Gas regime volatility" and "Solar surprise volatility" (GARCH + placebo), and "Price direction" |

- Raw files: `data/raw/databento/` and `data/raw/ercot/<report>/batch_*.zip`. ERCOT batches are saved exactly as the API returns them; each inner file is named with its docId and post time.
- Cleaned files: `data/clean/*.parquet`. Every ERCOT row has `posted_ct`/`posted_utc` (when published) and `interval_start_*` or `interval_end_*` (the time it describes).
- `data/pull_log.csv` has one row per pull.
- **Known caveats:**
  - NP3-233-CD has no DST flag and always lists hour-ending 1–24, so the skipped or repeated hour on DST days gets a `NaT` interval time (~0.02% of rows). Drop those rows or handle them by hand.
  - The hourly solar reports include hour-ending 3 on spring-forward days, an hour that doesn't exist on the clock. Those rows (645, all ~0 MW at night) have a `NaT` interval time.
  - Post times are Central with no DST flag. Fall-back-night postings are resolved by docId order and the visible 01:55 → 01:00 clock jump (`_localize_posts` in `pull_ercot.py`). Every cleaned row has a publish time.
  - ERCOT lists 6 documents in the 5-minute reports that it can't serve (HTTP 400, even individually): 1 in NP4-738-CD and 5 in NP4-733-CD. They are recorded in `data/raw/ercot/<report>/unavailable_docs.csv` and skipped on re-runs. Each is a single 5-minute posting, so the next posting covers the gap.
  - To resume or top up quickly, use `pull_ercot.py all --since-last`. It lists only from the last saved posting minus one day. The default run re-lists the full window, which takes about 50 minutes per 5-minute report.
  - ERCOT sometimes skips an hour and publishes it late (12 times for NP3-233-CD), so **order vintages by `posted_utc`, not `doc_id`**.
  - In Sep 2024 ERCOT renamed the solar and wind actuals columns: `ACTUAL_SYSTEM_WIDE`/`GEN_SYSTEM_WIDE` became `SYSTEM_WIDE_GEN`, and wind's `ACTUAL_LZ_*` became `GEN_LZ_*`. ERCOT also added `SYSTEM_WIDE_HSL`, which has no history before then. Cleaned files merge each renamed column back into its original name (`COLUMN_ALIASES` in `pull_ercot.py`). Raw files keep ERCOT's names.
