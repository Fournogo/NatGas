"""Parse ERCOT's yearly fuel-mix workbooks (IntGenbyFuel<YYYY>.xlsx) into a long parquet.

Input : data/raw/ercot/fuel-mix-*/IntGenbyFuel*.xlsx  (downloaded by hand from
        ercot.com/gridinfo/generation; left untouched)
Output: data/clean/ercot_fuel_mix.parquet, one row per 15-minute interval per fuel.

Workbook layout (one sheet per month): Date | Fuel | Settlement Type | Total | 0:15 ... 0:00,
plus 01:15 (DST) ... 02:00 (DST) on November sheets. Values are MWh per 15-minute interval
(x4 = average MW). Labels are interval-ending times; "0:00" is midnight at the end of the day.

DST: intervals are placed by position, not by label. Each day's non-blank cells, ordered
0:15 ... 2:00, [01:15-02:00 (DST)], 2:15 ... 0:00, are counted forward from local midnight.
That gives 92 / 96 / 100 intervals on spring-forward / normal / fall-back days. (Checked
against NP4-733 5-minute wind: the "(DST)" columns are the second, standard-time 1 AM hour.)

Usage: .venv/bin/python pipeline/clean_fuel_mix.py
"""

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from common import RAW, append_pull_log, clean_dir

CT = "America/Chicago"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ID_COLS = ["Date", "Fuel", "Settlement Type", "Total"]
DST_COLS = ["01:15 (DST)", "01:30 (DST)", "01:45 (DST)", "02:00 (DST)"]
TOTAL_TOLERANCE = 1e-4  # relative difference allowed between sum of intervals and Total


def interval_columns(columns) -> list[str]:
    """Interval columns in chronological order: regular through 2:00, the DST repeat, the rest."""
    regular = [c for c in columns if c not in ID_COLS and c not in DST_COLS]
    dst = [c for c in DST_COLS if c in columns]
    if not dst:
        return regular
    cut = regular.index("2:00") + 1
    return regular[:cut] + dst + regular[cut:]


def local_day_hours(day) -> int:
    start = pd.Timestamp(day).tz_localize(CT)
    end = (pd.Timestamp(day) + timedelta(days=1)).tz_localize(CT)
    return round((end - start).total_seconds() / 3600)


HEADER_ALIASES = {"Fuel Type": "Fuel"}  # ERCOT's not-yet-filled 2026 sheets use these names


def parse_sheet(df: pd.DataFrame, source: str, sheet: str) -> pd.DataFrame:
    df = df.rename(columns=lambda c: HEADER_ALIASES.get(str(c).strip(), str(c).strip()))
    missing = [c for c in ID_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"{source}/{sheet}: missing columns {missing}; got {list(df.columns[:6])}")
    df = df.dropna(subset=["Date", "Fuel"])
    if df.empty:
        return pd.DataFrame()
    cols = interval_columns(list(df.columns))
    values = df[cols].to_numpy(dtype=float)  # rows x intervals, NaN where blank
    df = df.rename(columns={"Settlement Type": "Settlement"})

    out = []
    for row, vals in zip(df.itertuples(index=False), values):
        day = pd.Timestamp(row.Date).normalize()
        present = vals[~np.isnan(vals)]
        expected = 4 * local_day_hours(day)
        if len(present) != expected:
            raise SystemExit(f"{source}/{sheet} {day.date()} {row.Fuel}: "
                             f"{len(present)} intervals, expected {expected}")
        total = float(row.Total)
        if abs(present.sum() - total) > TOTAL_TOLERANCE * max(abs(total), 1.0):
            raise SystemExit(f"{source}/{sheet} {day.date()} {row.Fuel}: intervals sum "
                             f"{present.sum():.3f} != Total {total:.3f}")
        midnight_utc = day.tz_localize(CT).tz_convert("UTC")
        ends = midnight_utc + pd.to_timedelta(np.arange(1, expected + 1) * 15, unit="min")
        out.append(pd.DataFrame({
            "operating_date": day.date(),
            "interval_end_utc": ends,
            "fuel": row.Fuel,
            "settlement_type": row.Settlement,
            "mwh": present,
        }))
    return pd.concat(out, ignore_index=True)


def parse_workbook(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=[m for m in MONTHS])
    parts = [parse_sheet(df, path.name, name) for name, df in sheets.items()]
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    df["source_file"] = str(path.relative_to(RAW.parent.parent))
    return df


def main() -> None:
    files = sorted(RAW.glob("ercot/fuel-mix-*/IntGenbyFuel*.xlsx"),
                   key=lambda p: p.stat().st_mtime)  # oldest first, so newer files win ties
    if not files:
        raise SystemExit("No IntGenbyFuel*.xlsx files under data/raw/ercot/fuel-mix-*/")

    frames = []
    for i, f in enumerate(files):
        df = parse_workbook(f)
        df["_file_order"] = i
        frames.append(df)
        st = df.drop_duplicates(["operating_date", "fuel"])["settlement_type"].value_counts().to_dict()
        print(f"{f.name}: {len(df):,} rows, {df['operating_date'].min()} -> "
              f"{df['operating_date'].max()}, settlement {st}")
        append_pull_log(
            source="ercot_fuel_mix_xlsx", report_id="IntGenbyFuel",
            start=str(df["operating_date"].min()), end=str(df["operating_date"].max()),
            rows=len(df), file=str(f.relative_to(RAW.parent.parent)),
            notes=f"manual download; settlement day-fuel counts {st}",
        )

    df = pd.concat(frames, ignore_index=True)
    # Overlapping files (e.g. a re-downloaded 2026 workbook): FINAL beats INITIAL, then newer file.
    df["_final"] = df["settlement_type"].eq("FINAL")
    df = (df.sort_values(["_final", "_file_order"])
            .drop_duplicates(["interval_end_utc", "fuel"], keep="last")
            .drop(columns=["_final", "_file_order"])
            .sort_values(["interval_end_utc", "fuel"]).reset_index(drop=True))

    df["interval_end_ct"] = df["interval_end_utc"].dt.tz_convert(CT)
    df["interval_start_utc"] = df["interval_end_utc"] - pd.Timedelta(minutes=15)
    df["mw_avg"] = df["mwh"] * 4
    df = df[["operating_date", "interval_end_utc", "interval_end_ct", "interval_start_utc",
             "fuel", "settlement_type", "mwh", "mw_avg", "source_file"]]

    out = clean_dir() / "ercot_fuel_mix.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df):,} rows ({df['operating_date'].nunique():,} days, "
          f"{df['fuel'].nunique()} fuels) to {out}")


if __name__ == "__main__":
    main()
