"""Pull Henry Hub NG front-month 1-minute bars from Databento (GLBX.MDP3).

Order of operations:
  1. Query the cost (free) and abort if it exceeds --max-cost.
  2. Download the raw DBN file untouched into data/raw/databento/.
  3. Save the continuous -> instrument symbology alongside it.
  4. Write a cleaned parquet to data/clean/ (UTC + Chicago time, contract, roll flag).
  5. Append a row to data/pull_log.csv.

Usage:
  .venv/bin/python pipeline/pull_databento.py                 # cost check, then pull
  .venv/bin/python pipeline/pull_databento.py --cost-only     # just print the cost
  .venv/bin/python pipeline/pull_databento.py --clean-only FILE.dbn.zst
"""

import argparse
import json
from pathlib import Path

import databento as db
import pandas as pd

from common import SAMPLE_START, append_pull_log, clean_dir, env, raw_dir

DATASET = "GLBX.MDP3"
SYMBOL = "NG.v.0"
STYPE_IN = "continuous"
SCHEMA = "ohlcv-1m"  # never mbo / mbp / trades: far larger and more expensive


def contract_names(client: db.Historical, instrument_ids, start: str, end: str) -> dict[int, str]:
    """Map instrument_id -> raw CME symbol (e.g. NGF4) via the free symbology endpoint."""
    res = client.symbology.resolve(
        dataset=DATASET,
        symbols=[str(i) for i in instrument_ids],
        stype_in="instrument_id",
        stype_out="raw_symbol",
        start_date=start[:10],
        end_date=end[:10],
    )
    out = {}
    for iid, intervals in res["result"].items():
        if intervals:
            out[int(iid)] = intervals[-1]["s"]
    return out


def clean(store: db.DBNStore, client: db.Historical, start: str, end: str) -> pd.DataFrame:
    df = store.to_df(price_type="float", pretty_ts=True, map_symbols=True).reset_index()
    df = df.rename(columns={"ts_event": "ts_event_utc", "symbol": "continuous_symbol"})
    df = df.sort_values("ts_event_utc", kind="stable").reset_index(drop=True)

    names = contract_names(client, df["instrument_id"].unique(), start, end)
    df["contract"] = df["instrument_id"].map(names)
    df["ts_event_ct"] = df["ts_event_utc"].dt.tz_convert("America/Chicago")

    # A return that spans a change of underlying contract is not a real price move.
    df["roll_flag"] = df["instrument_id"].ne(df["instrument_id"].shift()) & df.index.to_series().gt(0)

    # Databento marks some days "degraded" (data-quality issues on their side); keep a record.
    cond = client.metadata.get_dataset_condition(dataset=DATASET, start_date=start[:10], end_date=end[:10])
    (raw_dir("databento") / "glbx_dataset_condition.json").write_text(json.dumps(cond, indent=2))
    degraded = {c["date"] for c in cond if c["condition"] != "available"}
    df["degraded_day"] = df["ts_event_utc"].dt.strftime("%Y-%m-%d").isin(degraded)

    cols = [
        "ts_event_utc", "ts_event_ct", "continuous_symbol", "instrument_id", "contract",
        "open", "high", "low", "close", "volume", "roll_flag", "degraded_day",
    ]
    return df[cols]  # gaps (minutes with no trades) are left as gaps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=SAMPLE_START)
    ap.add_argument("--end", default=None, help="default: latest available for the schema")
    ap.add_argument("--max-cost", type=float, default=5.0, help="abort if cost (USD) exceeds this")
    ap.add_argument("--cost-only", action="store_true")
    ap.add_argument("--clean-only", type=Path, help="rebuild clean parquet from an existing raw file")
    args = ap.parse_args()

    client = db.Historical(env("DATABENTO_API_KEY"))

    if args.clean_only:
        store = db.DBNStore.from_file(args.clean_only)
        start, end = str(store.start), str(store.end)
        raw_path = args.clean_only
        cost = None
    else:
        end = args.end or client.metadata.get_dataset_range(dataset=DATASET)["schema"][SCHEMA]["end"]
        start = args.start
        q = dict(dataset=DATASET, symbols=[SYMBOL], stype_in=STYPE_IN, schema=SCHEMA, start=start, end=end)
        cost = client.metadata.get_cost(**q)
        size = client.metadata.get_billable_size(**q)
        print(f"Cost for {SYMBOL} {SCHEMA} {start} -> {end}: ${cost:.4f} ({size / 1e6:.1f} MB)")
        if args.cost_only:
            return
        if cost > args.max_cost:
            raise SystemExit(f"Cost ${cost:.2f} exceeds --max-cost ${args.max_cost:.2f}; aborting.")

        tag = f"{start[:10]}_{end[:10]}"
        raw_path = raw_dir("databento") / f"ng_v0_ohlcv1m_{tag}.dbn.zst"
        print(f"Downloading to {raw_path} ...")
        store = client.timeseries.get_range(**q, path=raw_path)

        sym_path = raw_path.with_name(f"ng_v0_symbology_{tag}.json")
        sym_path.write_text(json.dumps(store.symbology, indent=2, default=str))

    df = clean(store, client, start, end)
    out = clean_dir() / "ng_front_ohlcv1m.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df):,} rows to {out} ({df['roll_flag'].sum()} roll bars flagged)")

    if not args.clean_only:
        append_pull_log(
            source="databento", report_id=f"{DATASET}/{SYMBOL}/{SCHEMA}",
            start=start, end=end, rows=len(df), file=str(raw_path.relative_to(raw_path.parents[3])),
            cost_usd=f"{cost:.4f}", notes="continuous front month, rolls by volume",
        )


if __name__ == "__main__":
    main()
