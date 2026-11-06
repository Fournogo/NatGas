"""Does Henry Hub move in a direction during generation regimes? (companion to regime_vol.py)

Uses the same 15-minute NG returns and regime labels as regime_vol (sources: solar ramps, combined
gas ramps, solar forecast surprises). Two tests, each against the wrong-day placebo (labels moved
by +/-1-4 weeks: same weekday, time of day, nearly the same season):

1. Mean return: mean 15-min return (bp) in each in-regime group minus the mean in the rest.
   CI from resampling whole CT days. Also on vol-scaled returns (ret_adj) for a less noisy t-stat.
2. Event study: average cumulative return from 2 h before to 4 h after each regime episode starts
   (an episode = a run of consecutive 15-min returns in the same group).

Expected sign for solar surprises: less solar than forecast -> more gas burn -> positive returns.

Usage: .venv/bin/python pipeline/price_direction.py --source solar_error --all
"""

import argparse
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd

import regime_vol as rv

STEP = rv.STEP
BEFORE, AFTER = 8, 16  # event-study window in 15-min steps (2 h before, 4 h after)


def mean_diffs(df: pd.DataFrame, col: str = "ret_bp", n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Mean(col | group) - mean(col | out) for each in-group, with a day-block bootstrap CI."""
    day = df["t_end_ct"].dt.date
    grp = df["group"].where(df["group"].isin(rv.IN_GROUPS), "Out")
    keys = list(rv.IN_GROUPS) + ["Out"]
    agg = df[col].groupby([day, grp]).agg(["sum", "size"]).unstack(fill_value=0)
    S = agg["sum"].reindex(columns=keys, fill_value=0).to_numpy()
    N = agg["size"].reindex(columns=keys, fill_value=0).to_numpy()

    def diffs(w):
        m = (w @ S) / np.maximum(w @ N, 1)
        return m[:2] - m[2], m

    point, means = diffs(np.ones(len(S)))
    if n_boot:
        rng = np.random.default_rng(seed)
        W = rng.multinomial(len(S), np.full(len(S), 1 / len(S)), size=n_boot)
        boots = np.array([diffs(w)[0] for w in W])
        lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
        se = np.nanstd(boots, axis=0)
    else:
        lo = hi = se = np.full(2, np.nan)
    n = N.sum(axis=0)
    thin = n[:2] < 50  # too few in-regime returns to say anything
    point, lo, hi, se = (np.where(thin, np.nan, v) for v in (point, lo, hi, se))
    return pd.DataFrame({"mean_in": np.where(thin, np.nan, means[:2]), "mean_out": means[2], "diff": point, "lo": lo, "hi": hi,
                         "t": point / np.where(se > 0, se, np.nan), "n": n[:2], "n_out": n[2]},
                        index=list(rv.IN_GROUPS))


def episode_starts(df: pd.DataFrame) -> pd.DataFrame:
    """First return of each run of consecutive 15-min returns in the same in-group."""
    d = df.sort_values("t_end_utc")
    new = (d["group"] != d["group"].shift()) | (d["t_end_utc"] - d["t_end_utc"].shift() != STEP)
    starts = d[new & d["group"].isin(rv.IN_GROUPS)]
    return starts[["t_end_utc", "group"]].reset_index(drop=True)


def event_paths(r: pd.DataFrame, starts: pd.DataFrame, col: str = "ret_bp") -> pd.DataFrame:
    """Average return at each offset k (15-min steps relative to the episode's first return, k=0)
    and the cumulative path, anchored at 0 at the end of the return just before the episode (k=-1).
    Missing returns (gaps, outside the range) are skipped."""
    ret = r.set_index(r["t_end_utc"].dt.tz_convert(None))[col]
    ret = ret[~ret.index.duplicated()]
    ks = np.arange(-BEFORE, AFTER + 1)
    offsets = pd.to_timedelta(ks * 15, unit="min").to_numpy()
    rows = []
    for g in rv.IN_GROUPS:
        t0 = starts.loc[starts["group"] == g, "t_end_utc"].dt.tz_convert(None).to_numpy()
        if not len(t0):
            continue
        grid = t0[:, None] + offsets[None, :]
        vals = ret.reindex(pd.DatetimeIndex(grid.ravel())).to_numpy().reshape(grid.shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # offsets with no returns -> NaN
            mean_k = np.nanmean(vals, axis=0)
        n_k = (~np.isnan(vals)).sum(axis=0)
        m = np.nan_to_num(mean_k)
        cum = np.empty(len(ks))
        after = ks >= 0
        cum[after] = np.cumsum(m[after])
        before = m[~after]
        rc = np.cumsum(before[::-1])[::-1]  # sum(before[i:])
        cum[~after] = -(rc - before)        # -sum(before[i+1:]); 0 at k = -1
        rows.append(pd.DataFrame({"group": g, "k": ks, "minutes": ks * 15, "mean_bp": mean_k,
                                  "cum_bp": cum, "n": n_k, "episodes": len(t0)}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def placebo(r: pd.DataFrame, groups: pd.DataFrame, shifts=rv.PLACEBO_SHIFTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean differences and event paths with labels moved by whole weeks."""
    md, paths = [], []
    for k in shifts:
        g = groups.assign(t_end_utc=groups["t_end_utc"] + pd.Timedelta(days=k))
        df, _ = rv.align(r, g)
        if len(df) < 50:
            continue
        m = mean_diffs(df, n_boot=0)[["diff"]].assign(shift_days=k)
        md.append(m.rename_axis("group").reset_index())
        p = event_paths(r, episode_starts(df))
        if len(p):
            paths.append(p.assign(shift_days=k))
    return (pd.concat(md, ignore_index=True) if md else pd.DataFrame(),
            pd.concat(paths, ignore_index=True) if paths else pd.DataFrame())


def by_year(df: pd.DataFrame, col: str = "ret_bp") -> pd.DataFrame:
    """mean_diffs per calendar year (CT), to see whether an effect is stable over time."""
    out = []
    for y, g in df.groupby(df["t_end_ct"].dt.year):
        if len(g) < 500:
            continue
        out.append(mean_diffs(g, col, n_boot=200).rename_axis("group").reset_index().assign(year=int(y)))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def analyze(source: str, start: date, end: date, p_stay: float = 0.95, ng=None, data=None) -> dict:
    prep = rv.prepare(start, end, p_stay, ng=ng, fm=data, seasonal=True, source=source)
    df = prep["df"]
    out = {"df": df, "counts": prep["counts"], "n_unmatched": prep["n_unmatched"]}
    if len(df) < 50:
        return out
    out["means_bp"] = mean_diffs(df, "ret_bp")
    out["means_adj"] = mean_diffs(df, "ret_adj")
    starts = episode_starts(df)
    out["episodes"] = starts
    out["paths"] = event_paths(prep["returns"], starts)
    out["placebo_means"], out["placebo_paths"] = placebo(prep["returns"], prep["groups"])
    out["by_year"] = by_year(df)
    out["diag"] = rv.diagnostics(df, prep["counts"], prep["n_unmatched"])
    return out


def placebo_table(res: dict) -> pd.DataFrame:
    m, p = res["means_bp"], res["placebo_means"]
    rows = []
    for g in rv.IN_GROUPS:
        v = p.loc[p["group"] == g, "diff"].to_numpy()
        rows.append({"group": g, "real_diff_bp": m.loc[g, "diff"], "ci_lo": m.loc[g, "lo"], "ci_hi": m.loc[g, "hi"],
                     "t": m.loc[g, "t"], "placebo_mean": v.mean() if len(v) else np.nan,
                     "placebo_min": v.min() if len(v) else np.nan, "placebo_max": v.max() if len(v) else np.nan,
                     "outside_placebo_range": bool(len(v) and (m.loc[g, "diff"] < v.min() or m.loc[g, "diff"] > v.max()))})
    return pd.DataFrame(rows).set_index("group")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(rv.SOURCES), default="solar_error")
    ap.add_argument("--start", type=date.fromisoformat)
    ap.add_argument("--end", type=date.fromisoformat)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--p-stay", type=float, default=0.95)
    args = ap.parse_args()
    data = rv.load_source(args.source)
    lo, hi = rv.date_bounds(args.source, data)
    end = hi if args.all or not args.end else min(args.end, hi)
    start = lo if args.all else (args.start or end - timedelta(days=364))
    res = analyze(args.source, start, end, args.p_stay, data=data)
    names = rv.SOURCES[args.source]["names"]
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print(f"{rv.SOURCES[args.source]['label']}: {start} → {end}, {len(res['df']):,} returns")
    print("Mean 15-min return in regime minus out (bp), day-block CI:")
    print(res["means_bp"].rename(index=names).round(3).to_string())
    print("Same on vol-scaled returns (units of typical 15-min move):")
    print(res["means_adj"].rename(index=names).round(4).to_string())
    print("Against placebo:")
    print(placebo_table(res).rename(index=names).round(3).to_string())
    p, pp = res["paths"], res["placebo_paths"]
    for g in rv.IN_GROUPS:
        real = p[(p["group"] == g)].set_index("k")["cum_bp"]
        band = pp[pp["group"] == g].pivot(index="k", columns="shift_days", values="cum_bp")
        sel = [0, 4, 8, 16]
        print(f"\nEvent study, {names[g]} ({int(p.loc[p['group'] == g, 'episodes'].iloc[0])} episodes): "
              "cumulative bp after 15 min / 1 h / 2 h / 4 h")
        print("  real    ", [round(real[k], 2) for k in sel])
        print("  placebo ", [f"{band.loc[k].min():.2f}..{band.loc[k].max():.2f}" for k in sel])


if __name__ == "__main__":
    main()
