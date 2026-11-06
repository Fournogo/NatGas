"""Forward derivatives (ramp rates) of ERCOT generation by fuel, from the fuel-mix workbooks.

d(t) = mw_avg(t + 15 min) - mw_avg(t), in MW per 15 min, on the UTC interval-end series.
A difference is kept only when the next row is exactly 15 minutes later; anything else is
a gap (NaN, counted, never bridged). The last interval of the range has no "next" and is
dropped; nothing outside [start, end] is used.

`night_floor` (used for solar): steps where both ends are below the floor (MW) are marked
`excluded`. They keep their d but are left out of stats, histograms, z and the HMM fit.

Usage: .venv/bin/python pipeline/ramps.py --fuels gas --start 2026-09-24 --end 2026-09-30
       .venv/bin/python pipeline/ramps.py --fuels solar --all --hmm 0.95
"""

import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd

from common import CLEAN

CT = "America/Chicago"
GAS = ("Gas", "Gas-CC")
SOLAR = ("Solar",)
GAS_TOTAL = "Gas total"  # Gas + Gas-CC as one series
SOLAR_NIGHT_FLOOR = 100.0  # MW
STEP = pd.Timedelta(minutes=15)
FUEL_MIX = CLEAN / "ercot_fuel_mix.parquet"
COLUMNS = ["operating_date", "interval_end_utc", "fuel", "settlement_type", "mw_avg", "source_file"]


def load_fuel_mix(fuels=GAS) -> pd.DataFrame:
    return pd.read_parquet(FUEL_MIX, columns=COLUMNS, filters=[("fuel", "in", list(fuels))])


def combine_fuels(fm: pd.DataFrame, name: str, fuels=GAS) -> pd.DataFrame:
    """Sum several fuels into one series called `name`. An interval is kept only if every fuel
    is present; it counts as INITIAL if any part is INITIAL."""
    f = fm[fm["fuel"].isin(fuels)]
    out = f.groupby(["operating_date", "interval_end_utc"], as_index=False).agg(
        mw_avg=("mw_avg", "sum"), n=("fuel", "nunique"),
        settlement_type=("settlement_type", "max"),  # "INITIAL" > "FINAL"
        source_file=("source_file", "first"))
    out = out[out["n"] == len(fuels)].drop(columns="n")
    out["fuel"] = name
    return out[COLUMNS]


def in_range(fm: pd.DataFrame, start: date, end: date, fuels=GAS) -> pd.DataFrame:
    """Rows whose operating date (Central) is within [start, end], inclusive."""
    m = (fm["operating_date"] >= start) & (fm["operating_date"] <= end) & fm["fuel"].isin(fuels)
    return fm[m].sort_values(["fuel", "interval_end_utc"]).reset_index(drop=True)


def forward_diffs(fm: pd.DataFrame, start: date, end: date, fuels=GAS,
                  night_floor: float | None = None) -> pd.DataFrame:
    r = in_range(fm, start, end, fuels)
    g = r.groupby("fuel", sort=False)
    r["next_interval_utc"] = g["interval_end_utc"].shift(-1)
    r["next_mw_avg"] = g["mw_avg"].shift(-1)
    step = r["next_interval_utc"] - r["interval_end_utc"]
    r["step_ok"] = step.eq(STEP)
    r["is_last"] = r["next_interval_utc"].isna()
    r["d_mw_15min"] = (r["next_mw_avg"] - r["mw_avg"]).where(r["step_ok"])
    r["d_mw_per_min"] = r["d_mw_15min"] / 15
    if night_floor is None:
        r["excluded"] = False
    else:
        r["excluded"] = (r["mw_avg"] < night_floor) & (r["next_mw_avg"] < night_floor)
    r["interval_end_ct"] = r["interval_end_utc"].dt.tz_convert(CT)
    return r[["interval_end_utc", "interval_end_ct", "next_interval_utc", "fuel", "mw_avg", "next_mw_avg",
              "d_mw_15min", "d_mw_per_min", "step_ok", "is_last", "excluded", "settlement_type",
              "source_file", "operating_date"]]


def usable(d: pd.DataFrame) -> pd.Series:
    """Rows that count for stats, histograms, z and the HMM: a valid step, not excluded."""
    return d["d_mw_15min"].notna() & ~d["excluded"]


def bin_ramps(d: pd.DataFrame, bin_w: float) -> pd.DataFrame:
    """Valid differences (excluded ones included, for drawing) with their histogram bin
    (`bin_start`, width `bin_w`). Index is kept aligned with `d`."""
    out = d.dropna(subset=["d_mw_15min"]).copy()
    out["bin_start"] = np.floor(out["d_mw_15min"] / bin_w) * bin_w
    return out


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    qs = [0.01, 0.05, 0.5, 0.95, 0.99]
    for fuel, g in d.groupby("fuel", sort=False):
        v = g.loc[usable(g), "d_mw_15min"]
        q = v.quantile(qs) if len(v) else pd.Series(np.nan, qs)
        rows.append({
            "fuel": fuel, "n": len(v), "gaps": int((~g["step_ok"] & ~g["is_last"]).sum()),
            "excluded (night)": int((g["excluded"] & g["d_mw_15min"].notna()).sum()),
            "mean": v.mean(), "std": v.std(), "p1": q[0.01], "p5": q[0.05], "median": q[0.5],
            "p95": q[0.95], "p99": q[0.99], "min": v.min(), "max": v.max(),
            "std (MW/min)": v.std() / 15,
        })
    return pd.DataFrame(rows).set_index("fuel")


def expected_intervals(start: date, end: date) -> int:
    """15-minute intervals per fuel in [start, end] local days (92/96/100 on DST days)."""
    s = pd.Timestamp(start).tz_localize(CT)
    e = (pd.Timestamp(end) + timedelta(days=1)).tz_localize(CT)
    return round((e - s) / STEP)


def diagnostics(fm: pd.DataFrame, d: pd.DataFrame, start: date, end: date, fuels=GAS,
                top: int = 10) -> dict:
    """Checks that should be zero/empty on clean data, plus the largest ramps to eyeball.

    Returns {name: {"count": int, "examples": DataFrame}}. "Largest ramps" is informational.
    """
    r = in_range(fm, start, end, fuels)
    exp = expected_intervals(start, end)
    out = {}

    got = r.groupby("fuel")["interval_end_utc"].nunique().reindex(list(fuels), fill_value=0)
    short = got[got != exp]
    out["Interval count ≠ expected"] = {
        "count": len(short),
        "examples": pd.DataFrame({"fuel": short.index, "intervals": short.values, "expected": exp}),
    }

    bad = d[~d["step_ok"] & ~d["is_last"]]
    out["Non-15-minute steps"] = {
        "count": len(bad),
        "examples": bad[["fuel", "interval_end_ct", "next_interval_utc"]].head(top),
    }

    dup = r[r.duplicated(["fuel", "interval_end_utc"], keep=False)]
    out["Duplicate timestamps"] = {"count": len(dup), "examples": dup.head(top)}

    neg = r[r["mw_avg"].isna() | (r["mw_avg"] < 0)]
    out["Missing or negative MW"] = {"count": len(neg), "examples": neg.head(top)}

    init = r[r["settlement_type"] != "FINAL"]
    days = init.drop_duplicates("operating_date")[["operating_date", "settlement_type"]]
    out["INITIAL (unrevised) days"] = {"count": len(days), "examples": days.head(top)}

    seams = []
    for fuel, g in d.groupby("fuel", sort=False):
        chg = g["source_file"].ne(g["source_file"].shift()) & g["source_file"].shift().notna()
        for i in np.flatnonzero(chg.to_numpy()):
            prev = g.iloc[i - 1]
            seams.append({"fuel": fuel, "last_interval_ct": prev["interval_end_ct"],
                          "from_file": prev["source_file"], "to_file": g.iloc[i]["source_file"],
                          "d_mw_15min": prev["d_mw_15min"]})
    out["Workbook seams (check the d at the seam)"] = {"count": len(seams),
                                                       "examples": pd.DataFrame(seams)}

    big = (d[usable(d)]
             .assign(abs_d=lambda x: x["d_mw_15min"].abs())
             .sort_values("abs_d", ascending=False)
             .groupby("fuel", sort=False).head(top)
             [["fuel", "interval_end_ct", "mw_avg", "next_mw_avg", "d_mw_15min"]])
    out["Largest ramps"] = {"count": 0, "examples": big.reset_index(drop=True)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuels", choices=["gas", "solar"], default="gas")
    ap.add_argument("--start", type=date.fromisoformat)
    ap.add_argument("--end", type=date.fromisoformat)
    ap.add_argument("--all", action="store_true", help="whole fuel-mix file")
    ap.add_argument("--hmm", type=float, metavar="P_STAY", help="also fit the regime HMM")
    args = ap.parse_args()
    fuels, floor = (GAS, None) if args.fuels == "gas" else (SOLAR, SOLAR_NIGHT_FLOOR)

    fm = load_fuel_mix(fuels)
    lo, hi = fm["operating_date"].min(), fm["operating_date"].max()
    start = lo if args.all or not args.start else args.start
    end = hi if args.all or not args.end else min(args.end, hi)
    if start > end:
        raise SystemExit(f"Empty range {start} → {end} (data runs {lo} → {hi})")

    d = forward_diffs(fm, start, end, fuels, floor)
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print(f"Range {start} → {end}  ({expected_intervals(start, end):,} expected intervals per fuel)\n")
    print("Forward differences, MW per 15 min")
    print(summarize(d).round(1).to_string(), "\n")
    for name, chk in diagnostics(fm, d, start, end, fuels).items():
        print(f"{name}: {chk['count']}")
        if len(chk["examples"]):
            print(chk["examples"].to_string(index=False), "\n")

    if args.hmm:
        import ramp_hmm
        states, params = ramp_hmm.decode(d, args.hmm)
        print(f"HMM regimes (p_stay {args.hmm})")
        print(params.round(1).to_string(), "\n")
        for fuel in fuels:
            s = states[d["fuel"] == fuel].dropna()
            print(f"{fuel}: {s.value_counts().to_dict()}, "
                  f"switches {int((s != s.shift()).sum() - 1)}")


if __name__ == "__main__":
    main()
