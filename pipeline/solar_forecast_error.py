"""Hourly ERCOT system-wide solar: day-before forecast vs actual (NP4-737-CD).

For each delivery hour (hour-beginning, Central):
- forecast_<S> = the latest vintage of forecast S (STPPF, PVGRPP) posted at or before
  10:00 CT on the day before the operating day, i.e. the last forecast published before the
  Day-Ahead Market closes. `forecast_posted_ct` records which vintage that was.
- actual = the final (latest-posted) ACTUAL_SYSTEM_WIDE for the hour.
- error_<S> = actual - forecast_<S>, MW (positive = more solar than forecast).

Input : data/clean/ercot_np4_737_cd.parquet (all vintages)
Output: data/clean/solar_forecast_error.parquet

Usage: .venv/bin/python pipeline/solar_forecast_error.py            # build parquet + checks
       .venv/bin/python pipeline/solar_forecast_error.py --checks   # checks only
"""

import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd

from common import CLEAN, append_pull_log

CT = "America/Chicago"
SERIES = ("STPPF", "PVGRPP")
DA_CUTOFF_HOUR = 10  # 10:00 CT the day before = DAM close
NIGHT_FLOOR = 100.0  # MW; hours where actual and forecast are both below are "night"
CAP_FACTOR = 1.6  # plausibility cap = this x the month's p95 actual
BAD_DOC_MIN_HOURS = 3  # a posting with this many hours above the cap is dropped as corrupt
SRC = CLEAN / "ercot_np4_737_cd.parquet"
OUT = CLEAN / "solar_forecast_error.parquet"
DROPPED = CLEAN / "solar_forecast_error_dropped_docs.csv"


def plausibility_caps(df: pd.DataFrame) -> pd.Series:
    """MW cap per calendar month: 1.6 x the 95th percentile of each hour's *minimum* actual
    across vintages (corrupt postings only ever inflate values, so the minimum is clean)."""
    a = df.dropna(subset=["ACTUAL_SYSTEM_WIDE"])
    hourly_min = a.groupby("interval_start_utc").agg(
        v=("ACTUAL_SYSTEM_WIDE", "min"), month=("month", "first"))
    return hourly_min.groupby("month")["v"].quantile(0.95) * CAP_FACTOR


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (hourly table, dropped postings)."""
    cols = ["doc_id", "posted_ct", "interval_start_utc", "interval_start_ct", "ACTUAL_SYSTEM_WIDE"] + \
           [f"{s}_SYSTEM_WIDE" for s in SERIES]
    df = pd.read_parquet(SRC, columns=cols).dropna(subset=["interval_start_utc"])
    df["operating_date"] = df["interval_start_ct"].dt.date
    df["month"] = df["interval_start_ct"].dt.strftime("%Y-%m")

    # 1) Drop corrupt postings: >= 3 hours with an actual above the month's cap. (Aug 22-23
    #    2024: 25 postings with every value inflated ~6.5x.) Single-hour spikes are handled below.
    df["cap"] = df["month"].map(plausibility_caps(df))
    over = df[df["ACTUAL_SYSTEM_WIDE"] > df["cap"]].groupby(["doc_id", "posted_ct"]).size()
    bad_docs = over[over >= BAD_DOC_MIN_HOURS].rename("hours_over_cap").reset_index()
    df = df[~df["doc_id"].isin(bad_docs["doc_id"])]
    # Day-before 10:00 CT per operating date (built from the date, so DST days are right).
    dates = pd.Series(df["operating_date"].unique())
    naive = pd.to_datetime(dates - timedelta(days=1)) + pd.Timedelta(hours=DA_CUTOFF_HOUR)
    cut = pd.Series(naive.dt.tz_localize(CT).to_numpy(), index=dates.to_numpy())
    df["da_cutoff_ct"] = df["operating_date"].map(cut)
    df = df.sort_values("posted_ct")

    fc = df[df["posted_ct"] <= df["da_cutoff_ct"]]
    fc = fc.dropna(subset=[f"{s}_SYSTEM_WIDE" for s in SERIES], how="all")
    fc = fc.groupby("interval_start_utc").last()  # latest vintage before the cutoff
    act = df.dropna(subset=["ACTUAL_SYSTEM_WIDE"]).groupby("interval_start_utc")["ACTUAL_SYSTEM_WIDE"].last()
    hours = df.groupby("interval_start_utc")[["interval_start_ct", "operating_date", "da_cutoff_ct", "cap"]].first()

    # 2) Single implausible hours (e.g. 2024-04-29 14:00 = 34 GW in every posting): set to NaN.
    out = hours.join(act.rename("actual"))
    out["actual_implausible"] = out["actual"] > out["cap"]
    out.loc[out["actual_implausible"], "actual"] = np.nan
    out["forecast_posted_ct"] = fc["posted_ct"]
    for s in SERIES:
        f = fc[f"{s}_SYSTEM_WIDE"].reindex(out.index)
        out[f"forecast_{s}_implausible"] = f > out["cap"]
        out[f"forecast_{s}"] = f.mask(out[f"forecast_{s}_implausible"])
        out[f"error_{s}"] = out["actual"] - out[f"forecast_{s}"]
    out = out.reset_index().sort_values("interval_start_utc").reset_index(drop=True)
    out = out[["operating_date", "interval_start_utc", "interval_start_ct", "da_cutoff_ct",
               "forecast_posted_ct", "actual", "actual_implausible", "cap"]
              + [f"forecast_{s}" for s in SERIES] + [f"forecast_{s}_implausible" for s in SERIES]
              + [f"error_{s}" for s in SERIES]]
    return out, bad_docs


def load() -> pd.DataFrame:
    return pd.read_parquet(OUT)


def in_range(fe: pd.DataFrame, start: date, end: date, series: str,
             night_floor: float | None = NIGHT_FLOOR) -> pd.DataFrame:
    """Hours in [start, end] with a generic layout: value = error, plus usable/excluded flags."""
    r = fe[(fe["operating_date"] >= start) & (fe["operating_date"] <= end)].copy()
    r["series"] = series
    r["forecast"] = r[f"forecast_{series}"]
    r["error"] = r[f"error_{series}"]
    r["next_interval_utc"] = r["interval_start_utc"].shift(-1)
    r["excluded"] = False
    if night_floor is not None:
        r["excluded"] = (r["actual"] < night_floor) & (r["forecast"] < night_floor)
    r["usable"] = r["error"].notna() & ~r["excluded"]
    return r


def summarize(r: pd.DataFrame) -> pd.DataFrame:
    v = r.loc[r["usable"], "error"]
    qs = [0.01, 0.05, 0.5, 0.95, 0.99]
    q = v.quantile(qs) if len(v) else pd.Series(np.nan, qs)
    row = {"series": r["series"].iloc[0] if len(r) else "", "n": len(v),
           "missing (no actual or forecast)": int(r["error"].isna().sum()),
           "excluded (night)": int((r["excluded"] & r["error"].notna()).sum()),
           "mean (bias)": v.mean(), "MAE": v.abs().mean(), "std": v.std(),
           "p1": q[0.01], "p5": q[0.05], "median": q[0.5], "p95": q[0.95], "p99": q[0.99],
           "min": v.min(), "max": v.max()}
    return pd.DataFrame([row]).set_index("series")


def diagnostics(r: pd.DataFrame, start: date, end: date, top: int = 10) -> dict:
    out = {}
    s = pd.Timestamp(start).tz_localize(CT)
    e = (pd.Timestamp(end) + timedelta(days=1)).tz_localize(CT)
    exp = round((e - s) / pd.Timedelta(hours=1))
    out["Hours present ≠ expected"] = {
        "count": int(len(r) != exp),
        "examples": pd.DataFrame({"hours": [len(r)], "expected": [exp]}) if len(r) != exp else pd.DataFrame()}

    no_act = r[r["actual"].isna()]
    out["Hours without an actual"] = {"count": len(no_act),
                                      "examples": no_act[["interval_start_ct"]].head(top)}
    no_fc = r[r["forecast"].isna()]
    out["Hours without a day-before forecast"] = {
        "count": len(no_fc), "examples": no_fc[["interval_start_ct", "da_cutoff_ct"]].head(top)}

    age = r["da_cutoff_ct"] - r["forecast_posted_ct"]
    stale = r[age > pd.Timedelta(hours=2)]
    out["Stale forecast vintage (> 2 h before the 10:00 cutoff)"] = {
        "count": int(stale["operating_date"].nunique()),
        "examples": stale.drop_duplicates("operating_date")[
            ["operating_date", "da_cutoff_ct", "forecast_posted_ct"]].head(top)}

    imp = r[r["actual_implausible"] | r[f"forecast_{r['series'].iloc[0]}_implausible"]] if len(r) else r
    out["Implausible hours blanked (above the month's cap)"] = {
        "count": len(imp), "examples": imp[["interval_start_ct", "cap"]].head(top)}

    if DROPPED.exists():
        dropped = pd.read_csv(DROPPED, parse_dates=["posted_ct"])
        dropped["posted_ct"] = pd.to_datetime(dropped["posted_ct"], utc=True).dt.tz_convert(CT)
        lo = pd.Timestamp(start).tz_localize(CT) - timedelta(days=8)  # postings cover ~7 days ahead
        hi = (pd.Timestamp(end) + timedelta(days=1)).tz_localize(CT)
        dropped = dropped[(dropped["posted_ct"] >= lo) & (dropped["posted_ct"] < hi)]
        out["Corrupt postings dropped (posted near this range)"] = {"count": len(dropped),
                                                                    "examples": dropped.head(top)}

    neg = r[(r["actual"] < 0) | (r["forecast"] < 0)]
    out["Negative actual or forecast"] = {"count": len(neg),
                                          "examples": neg[["interval_start_ct", "actual", "forecast"]].head(top)}

    big = (r[r["usable"]].assign(abs_e=lambda x: x["error"].abs())
           .sort_values("abs_e", ascending=False).head(top)
           [["interval_start_ct", "actual", "forecast", "error", "forecast_posted_ct"]])
    out["Largest errors"] = {"count": 0, "examples": big.reset_index(drop=True)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", action="store_true", help="skip the build; check the saved file")
    ap.add_argument("--series", choices=SERIES, default="STPPF")
    args = ap.parse_args()

    if not args.checks:
        fe, bad_docs = build()
        fe.to_parquet(OUT, index=False)
        bad_docs.to_csv(DROPPED, index=False)
        append_pull_log(source="derived", report_id="solar_forecast_error",
                        start=str(fe["operating_date"].min()), end=str(fe["operating_date"].max()),
                        rows=len(fe), file=str(OUT.relative_to(CLEAN.parent.parent)),
                        notes=f"NP4-737-CD: day-before 10:00 CT STPPF/PVGRPP vintage vs final actual; "
                              f"{len(bad_docs)} corrupt postings dropped")
        print(f"Wrote {len(fe):,} hours to {OUT}; dropped {len(bad_docs)} corrupt postings "
              f"(listed in {DROPPED.name})")
    fe = load()
    have = fe.dropna(subset=["actual", f"forecast_{args.series}"])
    start, end = have["operating_date"].min(), have["operating_date"].max()
    r = in_range(fe, start, end, args.series)
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print(f"{args.series}, {start} → {end}\n")
    print(summarize(r).round(1).to_string(), "\n")
    for name, chk in diagnostics(r, start, end).items():
        print(f"{name}: {chk['count']}")
        if len(chk["examples"]):
            print(chk["examples"].to_string(index=False), "\n")


if __name__ == "__main__":
    main()
