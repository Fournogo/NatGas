"""Henry Hub volatility inside vs outside ERCOT generation ramp regimes (GARCH(1,1)-X).

Sources: "solar" (fuel-mix Solar ramps, night excluded), "gas" (Gas + Gas-CC ramps combined), or
"solar_error" (hourly solar actual - day-before 10:00 CT STPPF forecast; HMM regimes "actual above /
near / below forecast", each hour's label applied to its four 15-minute returns; night excluded).

1. 15-minute NG front-month log returns (bp). Price at grid time t = last 1-minute close whose
   bar ends in (t-15min, t]. A return is kept only for consecutive grid points with the same
   contract (drops no-trade windows, the daily halt, weekends and roll jumps).
2. Solar regimes from the fuel-mix Solar HMM (ramp_hmm): the solar step t -> t+15 is matched to
   the price return ending at t+15. Groups: Up, Down, Flat (daytime), Night.
3. Time-of-day adjustment: r_adj = r / s(slot), s = RMS return per 15-min CT slot, separately
   for Thursdays (EIA storage report) and other days, over the selected range.
4. GARCH(1,1)-X, Student-t, multiplicative regime effects:
       h_t = g_t * exp(th_up*U_t + th_dn*D_t)
       g_t = w + a * r_{t-1}^2 / exp(th.X_{t-1}) + b * g_{t-1}
   exp(th/2) = volatility ratio in that regime vs out (Flat + Night). LR tests vs th = 0.
5. Model-free check: RMS(r_adj | group) / RMS(r_adj | out), day-block bootstrap CI.

Returns are concatenated across dropped gaps (standard for intraday GARCH).
Solar regimes are look-back (Viterbi over the whole range): association, not a trading signal.

Usage: .venv/bin/python pipeline/regime_vol.py --source gas --start 2025-10-01 --end 2026-09-30
       .venv/bin/python pipeline/regime_vol.py --all --p-stay 0.95
"""

import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy import optimize, signal, special, stats

import ramp_hmm
import ramps
from common import CLEAN

CT = "America/Chicago"
STEP = pd.Timedelta(minutes=15)
GROUPS = ("Up", "Down", "Flat", "Night")
IN_GROUPS = ("Up", "Down")
MIN_RETURNS = 2000  # below this the GARCH fit is unreliable
RAMP_NAMES = {"Up": "ramping up", "Down": "ramping down"}
SOURCES = {
    "solar": {"label": "Solar", "kind": "ramp", "fuels": ramps.SOLAR, "combine": None,
              "night_floor": ramps.SOLAR_NIGHT_FLOOR, "names": RAMP_NAMES},
    "gas": {"label": "Gas (Gas + Gas-CC)", "kind": "ramp", "fuels": ramps.GAS, "combine": ramps.GAS_TOTAL,
            "night_floor": None, "names": RAMP_NAMES},
    "solar_error": {"label": "Solar forecast surprise (actual - day-before STPPF)", "kind": "error",
                    "series": "STPPF", "names": {"Up": "more solar than forecast", "Down": "less solar than forecast"}},
}
ERROR_LABELS = ("Below forecast", "Near forecast", "Above forecast")  # HMM states, low -> high error
ERROR_TO_GROUP = {"Below forecast": "Down", "Near forecast": "Flat", "Above forecast": "Up"}
STATE_TO_GROUP = {ramp_hmm.LABELS[0]: "Down", ramp_hmm.LABELS[1]: "Flat", ramp_hmm.LABELS[2]: "Up"}


# ---------------------------------------------------------------- data
def load_ng() -> pd.DataFrame:
    return pd.read_parquet(CLEAN / "ng_front_ohlcv1m.parquet",
                           columns=["ts_event_utc", "close", "contract", "degraded_day"])


def window(start: date, end: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    return (pd.Timestamp(start).tz_localize(CT).tz_convert("UTC"),
            (pd.Timestamp(end) + timedelta(days=1)).tz_localize(CT).tz_convert("UTC"))


def ng_returns_15m(ng: pd.DataFrame, start: date, end: date) -> tuple[pd.DataFrame, dict]:
    """15-minute returns ending in [start, end] (Central days). Also returns drop counts."""
    w0, w1 = window(start, end)
    b = ng[(ng["ts_event_utc"] >= w0 - STEP) & (ng["ts_event_utc"] < w1)].sort_values("ts_event_utc")
    grid = (b["ts_event_utc"] + pd.Timedelta(minutes=1)).dt.ceil("15min")
    p = b.assign(t=grid).groupby("t").agg(close=("close", "last"), contract=("contract", "last"),
                                          degraded=("degraded_day", "max"))
    prev_t = p.index.to_series().shift()
    consecutive = (p.index.to_series() - prev_t).eq(STEP)
    same = p["contract"].eq(p["contract"].shift())
    r = pd.DataFrame({
        "t_end_utc": p.index,
        "ret_bp": 1e4 * np.log(p["close"] / p["close"].shift()).to_numpy(),
        "contract": p["contract"].to_numpy(),
        "degraded": p["degraded"].to_numpy(),
    })
    ok = (consecutive & same).to_numpy()
    counts = {"price points": len(p), "non-consecutive (gap) breaks": int((~consecutive).sum() - 1),
              "dropped at contract rolls": int((consecutive & ~same).sum())}
    r = r[ok & (r["t_end_utc"] > w0) & (r["t_end_utc"] <= w1)].reset_index(drop=True)
    r["t_end_ct"] = r["t_end_utc"].dt.tz_convert(CT)
    return r, counts


def load_source(source: str) -> pd.DataFrame:
    """Rows for the regime source: fuel-mix generation (gas combined into one series), or the hourly
    solar forecast-error table."""
    cfg = SOURCES[source]
    if cfg["kind"] == "error":
        import solar_forecast_error as sfe
        return sfe.load()
    fm = ramps.load_fuel_mix(cfg["fuels"])
    return ramps.combine_fuels(fm, cfg["combine"], cfg["fuels"]) if cfg["combine"] else fm


def date_bounds(source: str, data: pd.DataFrame) -> tuple[date, date]:
    """First and last operating day with regime data."""
    if SOURCES[source]["kind"] == "error":
        data = data.dropna(subset=["actual", f"forecast_{SOURCES[source]['series']}"])
    return max(data["operating_date"].min(), date(2023, 12, 11)), data["operating_date"].max()


def error_regimes(fe: pd.DataFrame, start: date, end: date, p_stay: float, series: str = "STPPF"):
    """HMM regimes on the hourly forecast error; each hour's label is applied to the four 15-minute
    returns ending at :15, :30, :45 and :00 of that hour."""
    import solar_forecast_error as sfe
    r = sfe.in_range(fe, start, end, series)
    states, params = ramp_hmm.decode(r, p_stay, value="error", t="interval_start_utc", group="series",
                                     step=pd.Timedelta(hours=1), labels=ERROR_LABELS)
    g = pd.Series(np.where(r["excluded"], "Night", states.map(ERROR_TO_GROUP)), index=r.index)
    g[r["error"].isna()] = None
    hourly = pd.DataFrame({"hour_start": r["interval_start_utc"], "group": g, "gen_mw": r["actual"],
                           "error_mw": r["error"]}).dropna(subset=["group"])
    parts = [hourly.assign(t_end_utc=hourly["hour_start"] + k * STEP) for k in (1, 2, 3, 4)]
    return pd.concat(parts, ignore_index=True).drop(columns="hour_start"), params


def series_fuels(source: str) -> tuple:
    cfg = SOURCES[source]
    return (cfg["combine"],) if cfg["combine"] else cfg["fuels"]


def regime_groups(d: pd.DataFrame, states: pd.Series) -> pd.DataFrame:
    """Regime group per generation step, keyed by the step's end time (= the return's end time)."""
    g = pd.Series(np.where(d["excluded"], "Night", states.map(STATE_TO_GROUP)), index=d.index)
    g[d["d_mw_15min"].isna()] = None
    out = pd.DataFrame({"t_end_utc": d["next_interval_utc"], "group": g,
                        "gen_mw": d["mw_avg"], "gen_d_mw": d["d_mw_15min"]})
    return out.dropna(subset=["t_end_utc", "group"])


def align(r: pd.DataFrame, groups: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    m = r.merge(groups, on="t_end_utc", how="left")
    n_unmatched = int(m["group"].isna().sum())
    return m.dropna(subset=["group"]).reset_index(drop=True), n_unmatched


def tod_profile(df: pd.DataFrame, seasonal: bool = True, min_n: int = 20) -> pd.DataFrame:
    """Adds slot, is_thu, month, tod_scale (RMS bp expected for that return) and ret_adj.

    seasonal=False: one profile per (Thursday?, 15-min CT slot) over the whole range.
    seasonal=True : base profile per (calendar month, slot) from non-Thursdays, times a per-slot
                    Thursday factor (Thu / other-day variance, pooled over months). Tracks the
                    seasonal shift of the intraday pattern (sunrise moves ~1.5 h over the year).
    Cells with fewer than `min_n` returns fall back to the slot-level value.
    """
    df = df.copy()
    df["slot"] = df["t_end_ct"].dt.strftime("%H:%M")
    df["is_thu"] = df["t_end_ct"].dt.day_of_week.eq(3)
    df["month"] = df["t_end_ct"].dt.month
    sq = df["ret_bp"] ** 2

    def lookup(mask: pd.Series, keys: list[str]) -> np.ndarray:
        """Mean of sq over rows in `mask`, by `keys`, mapped to every row; NaN where n < min_n."""
        agg = sq[mask].groupby([df.loc[mask, k] for k in keys]).agg(["mean", "size"])
        agg.loc[agg["size"] < min_n, "mean"] = np.nan
        idx = pd.MultiIndex.from_frame(df[keys]) if len(keys) > 1 else pd.Index(df[keys[0]])
        return agg["mean"].reindex(idx).to_numpy()

    all_rows = pd.Series(True, index=df.index)
    slot_all = lookup(all_rows, ["slot"])
    slot_all = np.where(np.isnan(slot_all), sq.mean(), slot_all)
    if seasonal:
        other = ~df["is_thu"]
        base = lookup(other, ["month", "slot"])
        base_slot = lookup(other, ["slot"])
        base = np.where(np.isnan(base), np.where(np.isnan(base_slot), slot_all, base_slot), base)
        thu = lookup(df["is_thu"], ["slot"])
        factor = np.where(np.isnan(thu) | np.isnan(base_slot), 1.0, thu / np.where(base_slot > 0, base_slot, np.nan))
        factor = np.where(np.isfinite(factor), factor, 1.0)
        var = np.where(df["is_thu"], base * factor, base)
    else:
        var = lookup(all_rows, ["is_thu", "slot"])
        var = np.where(np.isnan(var), slot_all, var)
    df["tod_scale"] = np.sqrt(np.maximum(var, 1e-12))
    df["ret_adj"] = df["ret_bp"] / df["tod_scale"]
    return df


def identification(df: pd.DataFrame, seasonal: bool = True) -> pd.DataFrame:
    """How much in-regime returns can be told apart from time of day: share of Up / Down returns in
    profile cells (month x slot, or slot) where that regime is 10-90% of returns. Regimes that sit in
    cells that are always in-regime are absorbed by the time-of-day profile."""
    cell = [df["month"], df["slot"]] if seasonal else [df["slot"]]
    rows = {}
    for g in IN_GROUPS:
        share = df["group"].eq(g).groupby(cell).transform("mean")
        mine = df["group"].eq(g)
        rows[g] = {"returns": int(mine.sum()),
                   "in mixed cells": float(((share > 0.1) & (share < 0.9))[mine].mean()) if mine.any() else np.nan}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------- GARCH(1,1)-X
NAMES = ("omega", "alpha", "beta", "nu")


def _unpack(u: np.ndarray, k: int, free: np.ndarray):
    omega = np.exp(u[0])
    pers = special.expit(u[1]) * 0.9999
    share = special.expit(u[2])
    alpha, beta = pers * share, pers * (1 - share)
    nu = 2.05 + np.exp(u[3])
    theta = np.zeros(k)
    theta[free] = u[4:]
    return omega, alpha, beta, nu, theta


def _variance(eps2: np.ndarray, X: np.ndarray, omega, alpha, beta, theta, g0) -> np.ndarray:
    m = np.exp(X @ theta) if X.shape[1] else np.ones(len(eps2))
    x = eps2 / m
    drive = np.empty(len(eps2))
    drive[0] = omega + (alpha + beta) * g0  # g_0 from the unconditional level
    drive[1:] = omega + alpha * x[:-1]
    g = signal.lfilter([1.0], [1.0, -beta], drive)
    return g * m


def _nll(eps2, X, omega, alpha, beta, nu, theta, g0):
    h = _variance(eps2, X, omega, alpha, beta, theta, g0)
    c = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    ll = c - 0.5 * np.log(h) - (nu + 1) / 2 * np.log1p(eps2 / (h * (nu - 2)))
    return -ll.sum()


def fit_garch_x(y: np.ndarray, X: np.ndarray, names: list[str], free=None) -> dict:
    """MLE fit. `free`: boolean mask of which theta are estimated (others fixed at 0)."""
    y = np.asarray(y, float) - np.mean(y)
    eps2 = y ** 2
    k = X.shape[1]
    free = np.ones(k, bool) if free is None else np.asarray(free, bool)
    g0 = eps2.mean()
    u0 = np.r_[np.log(0.05 * g0), special.logit(0.95 / 0.9999), special.logit(0.1), np.log(6.0),
               np.zeros(free.sum())]

    def f(u):
        o, a, b, n, th = _unpack(u, k, free)
        v = _nll(eps2, X, o, a, b, n, th, g0)
        return v if np.isfinite(v) else 1e300

    res = optimize.minimize(f, u0, method="L-BFGS-B")
    o, a, b, n, th = _unpack(res.x, k, free)
    nat = np.r_[o, a, b, n, th[free]]

    # Standard errors: numerical Hessian of the NLL in natural parameters.
    def f_nat(p):
        tt = np.zeros(k)
        tt[free] = p[4:]
        if p[0] <= 0 or p[1] < 0 or p[2] < 0 or p[1] + p[2] >= 1 or p[3] <= 2:
            return np.nan
        return _nll(eps2, X, p[0], p[1], p[2], p[3], tt, g0)

    se = np.full(len(nat), np.nan)
    try:
        H = _hessian(f_nat, nat)
        cov = np.linalg.inv(H)
        se = np.sqrt(np.where(np.diag(cov) > 0, np.diag(cov), np.nan))
    except (np.linalg.LinAlgError, ValueError):
        pass
    labels = list(NAMES) + [f"theta_{nm}" for nm, fr in zip(names, free) if fr]
    h = _variance(eps2, X, o, a, b, th, g0)
    return {"params": pd.DataFrame({"estimate": nat, "se": se}, index=labels), "nll": res.fun,
            "converged": bool(res.success), "theta": th, "h": h, "n": len(y)}


def _hessian(f, x, rel=1e-4):
    n = len(x)
    h = np.maximum(np.abs(x) * rel, 1e-6)
    H = np.empty((n, n))
    f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i], ej[j] = h[i], h[j]
            if i == j:
                v = (f(x + ei) - 2 * f0 + f(x - ei)) / h[i] ** 2
            else:
                v = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
            H[i, j] = H[j, i] = v
    if not np.all(np.isfinite(H)):
        raise ValueError("non-finite Hessian")
    return H


def regime_garch(df: pd.DataFrame, col: str = "ret_adj") -> dict:
    """Full model (theta_up, theta_dn) plus LR tests for each and jointly."""
    X = np.column_stack([df["group"].eq(g).to_numpy(float) for g in IN_GROUPS])
    y = df[col].to_numpy()
    full = fit_garch_x(y, X, list(IN_GROUPS))
    tests = {}
    for label, free in (("Up", [False, True]), ("Down", [True, False]), ("Up and Down jointly", [False, False])):
        restricted = fit_garch_x(y, X, list(IN_GROUPS), free=free)
        lr = 2 * (restricted["nll"] - full["nll"])
        dfree = 2 - sum(free)
        tests[label] = {"LR": lr, "df": dfree, "p_value": stats.chi2.sf(max(lr, 0), dfree)}
    p = full["params"]
    ratios = {}
    for g in IN_GROUPS:
        th, se = p.loc[f"theta_{g}", "estimate"], p.loc[f"theta_{g}", "se"]
        ratios[g] = {"vol_ratio": np.exp(th / 2), "lo": np.exp((th - 1.96 * se) / 2),
                     "hi": np.exp((th + 1.96 * se) / 2), "p_value": tests[g]["p_value"]}
    return {"full": full, "tests": pd.DataFrame(tests).T, "ratios": pd.DataFrame(ratios).T}


# ---------------------------------------------------------------- model-free checks
def bootstrap_ratios(df: pd.DataFrame, col: str = "ret_adj", n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """RMS(col | group) / RMS(col | out = Flat + Night), CI from resampling whole CT days."""
    day = df["t_end_ct"].dt.date
    grp = df["group"].where(df["group"].isin(IN_GROUPS), "Out")
    keys = list(IN_GROUPS) + ["Out"]
    sq = (df[col] ** 2).groupby([day, grp]).agg(["sum", "size"]).unstack(fill_value=0)
    S = sq["sum"].reindex(columns=keys, fill_value=0).to_numpy()
    N = sq["size"].reindex(columns=keys, fill_value=0).to_numpy()

    def ratios(w):
        s, n = w @ S, w @ N
        rms = np.sqrt(s / np.maximum(n, 1))
        return rms[:2] / rms[2]

    point = ratios(np.ones(len(S)))
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(S), np.full(len(S), 1 / len(S)), size=n_boot)
    boots = np.array([ratios(w) for w in W])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    n = N.sum(axis=0)
    return pd.DataFrame({"vol_ratio": point, "lo": lo, "hi": hi, "n": n[:2], "n_out": n[2]},
                        index=list(IN_GROUPS))


def by_hour(df: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """Raw RMS return (bp) by CT hour (of the return's start) for Up / Down / Out."""
    hour = (df["t_end_ct"] - STEP).dt.hour
    grp = df["group"].where(df["group"].isin(IN_GROUPS), "Out")
    t = (df["ret_bp"] ** 2).groupby([hour, grp]).agg(["mean", "size"]).reset_index()
    t.columns = ["hour", "group", "ms", "n"]
    t["rms_bp"] = np.sqrt(t["ms"])
    return t[t["n"] >= min_n].drop(columns="ms")


def diagnostics(df: pd.DataFrame, counts: dict, n_unmatched: int, top: int = 10) -> dict:
    out = {}
    out["Returns without a regime (outside fuel-mix coverage or generation gaps)"] = {
        "count": n_unmatched, "examples": pd.DataFrame()}
    out["Returns dropped at contract rolls"] = {"count": counts["dropped at contract rolls"],
                                                "examples": pd.DataFrame()}
    deg = df[df["degraded"].fillna(False).astype(bool)]
    deg_days = (deg.assign(day=deg["t_end_ct"].dt.date).groupby("day").size()
                .rename("returns").reset_index())
    out["Returns on Databento degraded-quality days"] = {"count": len(deg), "examples": deg_days.head(top)}
    present = [g for g in GROUPS if g != "Night" or (df["group"] == "Night").any()]
    small = df.groupby("group").size().reindex(present, fill_value=0)
    thin = small[small < 200]
    out["Groups with fewer than 200 returns"] = {
        "count": len(thin), "examples": thin.rename("returns").reset_index()}
    big = (df.assign(abs_r=df["ret_bp"].abs()).sort_values("abs_r", ascending=False).head(top)
           [["t_end_ct", "contract", "ret_bp", "ret_adj", "group"]])
    out["Largest 15-minute returns"] = {"count": 0, "examples": big.reset_index(drop=True)}
    return out


# ---------------------------------------------------------------- placebo
PLACEBO_SHIFTS = (-28, -21, -14, -7, 7, 14, 21, 28)  # whole weeks: same weekday and time of day


def rms_ratios(df: pd.DataFrame, col: str = "ret_adj") -> pd.Series:
    out = df["group"].where(df["group"].isin(IN_GROUPS), "Out")
    rms = np.sqrt((df[col] ** 2).groupby(out).mean())
    return pd.Series({g: rms.get(g, np.nan) / rms.get("Out", np.nan) for g in IN_GROUPS})


def garch_ratios(df: pd.DataFrame, col: str = "ret_adj") -> pd.Series:
    X = np.column_stack([df["group"].eq(g).to_numpy(float) for g in IN_GROUPS])
    th = fit_garch_x(df[col].to_numpy(), X, list(IN_GROUPS))["theta"]
    return pd.Series(np.exp(th / 2), index=list(IN_GROUPS))


def placebo(r: pd.DataFrame, groups: pd.DataFrame, seasonal: bool,
            shifts=PLACEBO_SHIFTS) -> pd.DataFrame:
    """Re-run the analysis with each day's solar labels moved by whole weeks (wrong day, same
    weekday, same time of day, nearly the same season). A real solar effect should vanish."""
    rows = []
    for k in shifts:
        g = groups.assign(t_end_utc=groups["t_end_utc"] + pd.Timedelta(days=k))
        df, _ = align(r, g)
        if len(df) < 50:
            continue
        df = tod_profile(df, seasonal)
        for method, vals in (("GARCH", garch_ratios(df)), ("Model-free", rms_ratios(df))):
            for reg, v in vals.items():
                rows.append({"shift_days": k, "method": method, "regime": reg, "ratio": v})
    return pd.DataFrame(rows)


def placebo_summary(real: dict, plc: pd.DataFrame) -> pd.DataFrame:
    """real: {(method, regime): ratio}. Excess = real / mean placebo; rank = how many placebos
    are at least as far from 1 as the real one (in log terms), +1, out of n+1."""
    rows = []
    if plc.empty:
        return pd.DataFrame(columns=["method", "regime", "real", "placebo_mean", "placebo_min", "placebo_max",
                                     "excess", "outside_placebo_range", "n_placebos"])
    for (method, reg), v in real.items():
        p = plc[(plc["method"] == method) & (plc["regime"] == reg)]["ratio"].to_numpy()
        if not len(p):
            continue
        dev = np.abs(np.log(v / np.exp(np.log(p).mean())))
        p_dev = np.abs(np.log(p / np.exp(np.log(p).mean())))
        rows.append({"method": method, "regime": reg, "real": v, "placebo_mean": float(np.exp(np.log(p).mean())),
                     "placebo_min": p.min(), "placebo_max": p.max(), "excess": v / np.exp(np.log(p).mean()),
                     "outside_placebo_range": bool(v < p.min() or v > p.max()),
                     "n_placebos": len(p)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- pipeline
def prepare(start: date, end: date, p_stay: float, ng: pd.DataFrame | None = None,
            fm: pd.DataFrame | None = None, seasonal: bool = True, source: str = "solar") -> dict:
    """Everything up to the aligned, time-of-day-adjusted frame (plus inputs for the placebo)."""
    ng = load_ng() if ng is None else ng
    fm = load_source(source) if fm is None else fm
    # Generation labels cover the range +/- the largest placebo shift, so shifted (wrong-day) labels
    # exist even for short ranges. Real returns are still limited to [start, end].
    pad = timedelta(days=max(abs(k) for k in PLACEBO_SHIFTS))
    lo, hi = fm["operating_date"].min(), fm["operating_date"].max()
    s0, s1 = max(lo, start - pad), min(hi, end + pad)
    cfg = SOURCES[source]
    if cfg["kind"] == "error":
        groups, hmm_params = error_regimes(fm, s0, s1, p_stay, cfg["series"])
    else:
        d = ramps.forward_diffs(fm, s0, s1, series_fuels(source), cfg["night_floor"])
        states, hmm_params = ramp_hmm.decode(d, p_stay)
        groups = regime_groups(d, states)
    r, counts = ng_returns_15m(ng, start, end)
    df, n_unmatched = align(r, groups)
    return {"df": tod_profile(df, seasonal), "counts": counts, "n_unmatched": n_unmatched,
            "hmm": hmm_params, "returns": r, "groups": groups}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat)
    ap.add_argument("--end", type=date.fromisoformat)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--p-stay", type=float, default=0.95)
    ap.add_argument("--source", choices=list(SOURCES), default="solar")
    ap.add_argument("--simple-profile", action="store_true", help="one time-of-day profile, not by month")
    args = ap.parse_args()
    seasonal = not args.simple_profile
    fm = load_source(args.source)
    _, hi = date_bounds(args.source, fm)
    end = hi if args.all or not args.end else min(args.end, hi)
    start = date(2023, 12, 11) if args.all else (args.start or end - timedelta(days=364))

    prep = prepare(start, end, args.p_stay, fm=fm, seasonal=seasonal, source=args.source)
    df, counts, n_unmatched = prep["df"], prep["counts"], prep["n_unmatched"]
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print(f"{SOURCES[args.source]['label']}: {start} → {end}, p_stay {args.p_stay}, {'seasonal' if seasonal else 'simple'} time-of-day "
          f"profile: {len(df):,} returns; {df['group'].value_counts().to_dict()}")
    print("Identification (share of regime returns in mixed time-of-day cells):")
    print(identification(df, seasonal).round(3).to_string())
    for col, lab in (("ret_adj", "time-of-day adjusted"), ("ret_bp", "raw (no time-of-day adjustment)")):
        res = regime_garch(df, col)
        print(f"\n== GARCH(1,1)-X, {lab} ==")
        print(res["full"]["params"].round(4).to_string())
        print(res["ratios"].round(3).to_string())
        print(res["tests"].round(4).to_string())
        print("Model-free RMS ratios (day-block bootstrap):")
        print(bootstrap_ratios(df, col).round(3).to_string())
    plc = placebo(prep["returns"], prep["groups"], seasonal)
    g_adj = regime_garch(df, "ret_adj")["ratios"]["vol_ratio"]
    real = {("GARCH", g): g_adj[g] for g in IN_GROUPS} | {("Model-free", g): v for g, v in rms_ratios(df).items()}
    print("\n== Placebo (labels shifted by whole weeks) ==")
    print(plc.pivot_table(index="shift_days", columns=["method", "regime"], values="ratio").round(3).to_string())
    print(placebo_summary(real, plc).round(3).to_string(index=False))
    for name, chk in diagnostics(df, counts, n_unmatched).items():
        print(f"\n{name}: {chk['count']}")
        if len(chk["examples"]):
            print(chk["examples"].to_string(index=False))


if __name__ == "__main__":
    main()
