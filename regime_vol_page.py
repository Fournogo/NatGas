"""Shared renderer: Henry Hub volatility inside vs outside generation ramp regimes
(GARCH(1,1)-X + model-free check + placebo). Used by the solar and gas regime volatility pages."""

import calendar
from datetime import timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ramp_page import PANEL_H, PRESETS, Y_AXIS, data_checks, henry_hub_panel, time_axis
from app_lib import ct_naive  # ramp_page/app_lib put pipeline/ on sys.path
import regime_vol as rv

STEP = pd.Timedelta(minutes=15)
GROUP_COLORS = alt.Scale(domain=["Up", "Down", "Out"], range=["#f58518", "#4c78a8", "#9aa5b1"])


@st.cache_data(ttl="1h", max_entries=10)
def load_ng() -> pd.DataFrame:
    return rv.load_ng()


@st.cache_data(ttl="1h", max_entries=4)
def load_source(source: str) -> pd.DataFrame:
    return rv.load_source(source)


@st.cache_data(ttl="1h", max_entries=10, show_spinner="Fitting the HMM and GARCH models…")
def analyze(source: str, start, end, p_stay: float, seasonal: bool):
    prep = rv.prepare(start, end, p_stay, ng=load_ng(), fm=load_source(source), seasonal=seasonal, source=source)
    df, counts, n_unmatched, hmm_params = prep["df"], prep["counts"], prep["n_unmatched"], prep["hmm"]
    if len(df) < 50:
        return df, None
    out = {
        "garch_adj": rv.regime_garch(df, "ret_adj"),
        "garch_raw": rv.regime_garch(df, "ret_bp"),
        "boot_adj": rv.bootstrap_ratios(df, "ret_adj"),
        "boot_raw": rv.bootstrap_ratios(df, "ret_bp"),
        "by_hour": rv.by_hour(df),
        "diag": rv.diagnostics(df, counts, n_unmatched),
        "hmm": hmm_params,
        "ident": rv.identification(df, seasonal),
        "placebo": rv.placebo(prep["returns"], prep["groups"], seasonal),
    }
    real = ({("GARCH", g): out["garch_adj"]["ratios"].loc[g, "vol_ratio"] for g in rv.IN_GROUPS}
            | {("Model-free", g): out["boot_adj"].loc[g, "vol_ratio"] for g in rv.IN_GROUPS})
    out["placebo_summary"] = rv.placebo_summary(real, out["placebo"])
    df = df.assign(cond_vol_bp=np.sqrt(out["garch_adj"]["full"]["h"]) * df["tod_scale"])
    return df, out


def fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def render(source: str, title: str, noun: str, ramps_page: str, out_desc: str, intro: str | None = None) -> None:
    """Draw the page for regime `source` (see regime_vol.SOURCES). `noun` names the series in text;
    `intro` replaces the default first caption."""
    Noun = noun[0].upper() + noun[1:]
    KP = f"rv_{source}"
    names = rv.SOURCES[source]["names"]  # {"Up": "...", "Down": "..."}
    band = {g: f"{Noun}: {names[g]}" for g in rv.IN_GROUPS}
    BAND_SCALE = alt.Scale(domain=[band["Up"], band["Down"]], range=["#f58518", "#4c78a8"])

    st.title(title)
    st.caption((intro or f"Is Henry Hub front-month volatility different while ERCOT {noun} is ramping? 15-minute NG "
                f"returns are matched to the {noun} HMM regime ({ramps_page} page) for the same 15 minutes. Up and "
                f"down ramps are each compared against everything else ({out_desc}). ") +
               f"A placebo test re-runs everything "
               f"with each day's {noun} labels moved to a different week (same weekday and time of day); only the part "
               f"of an effect beyond the placebo can be credited to {noun}.")

    fm = load_source(source)
    lo, hi = rv.date_bounds(source, fm)
    if f"{KP}_range" not in st.session_state:
        st.session_state[f"{KP}_range"] = (max(lo, hi - timedelta(days=364)), hi)


    def apply_preset() -> None:
        days = PRESETS.get(st.session_state.get(f"{KP}_preset"))
        start = lo if days is None else max(lo, hi - timedelta(days=days - 1))
        st.session_state[f"{KP}_range"] = (start, hi)


    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.date_input("Date range (Central operating days)", key=f"{KP}_range", min_value=lo, max_value=hi)
        st.segmented_control(f"Quick range (ending on the last day of {noun} data)", list(PRESETS),
                             key=f"{KP}_preset", on_change=apply_preset)
        p_stay = st.slider(f"{Noun} HMM stickiness (p_stay)", 0.80, 0.995, value=0.95, step=0.005, format="%.3f",
                           key=f"{KP}_p_stay")
        profile = st.segmented_control("Time-of-day profile", ["By month", "Whole range"], default="By month",
                                       key=f"{KP}_profile",
                                       help="By month: a separate intraday volatility profile for each calendar month "
                                            "(sunrise and the morning pattern shift through the year). Whole range: one "
                                            "profile for the whole date range.") or "By month"
        seasonal = profile == "By month"

    rng = st.session_state[f"{KP}_range"]
    if len(rng) < 2:
        st.info("Pick an end date.", icon=":material/date_range:")
        st.stop()
    start, end = rng
    st.caption(f"{Noun} regimes run {lo} to {hi} (fuel-mix coverage); NG returns are limited to the same range.")

    df, res = analyze(source, start, end, p_stay, seasonal)
    if res is None:
        st.warning("Too few returns in this range to fit anything.", icon=":material/warning:")
        st.stop()
    if len(df) < rv.MIN_RETURNS:
        st.warning(f"Only {len(df):,} returns in this range. GARCH estimates need a few thousand "
                   f"(about a month or more) to be reliable; treat these numbers as rough.", icon=":material/warning:")

    # ---------- Headline ----------
    ps = res["placebo_summary"].set_index(["method", "regime"])
    with st.container(horizontal=True):
        st.metric("Returns analyzed", f"{len(df):,}", border=True)
        for g, name in (("Up", names["Up"]), ("Down", names["Down"])):
            for method in ("GARCH", "Model-free"):
                if (method, g) not in ps.index:
                    continue
                row = ps.loc[(method, g)]
                ok = np.isfinite(row["excess"])
                st.metric(f"{Noun}: {name}, effect beyond placebo ({method})",
                          f"{row['excess']:.3f}×" if ok else "n/a",
                          (f"real {row['real']:.3f}× vs placebo {row['placebo_mean']:.3f}× "
                           f"({row['placebo_min']:.3f}–{row['placebo_max']:.3f})") if ok
                          else "no returns in this regime in the range",
                          delta_color="off", delta_arrow="off", border=True)
    ident = res["ident"]
    outside = ps[ps["outside_placebo_range"].astype(bool)]
    verdict = ("No placebo could be run for this range." if ps.empty else f"No real ratio falls outside its placebo range: nothing here can be credited to {noun}."
               if outside.empty else
               "Outside the placebo range: " + ", ".join(f"{m}, {names[r]}" for m, r in outside.index)
               + f" (that alone is roughly a 2-in-{int(ps['n_placebos'].max()) + 1} chance).")
    st.caption(f"Effect beyond placebo = real ratio ÷ average ratio with wrong-day labels; 1.000× = no {noun} "
               f"effect. {verdict} Ratios are in-regime ÷ out-of-regime volatility after the time-of-day "
               f"adjustment. {ident.loc['Up', 'in mixed cells']:.0%} of '{names['Up']}' and "
               f"{ident.loc['Down', 'in mixed cells']:.0%} of '{names['Down']}' returns fall in time-of-day cells where "
               "that regime is neither almost always nor almost never the case; only those can separate the regime "
               "from time of day (the rest is absorbed by the time-of-day profile).")

    # ---------- Real vs placebo ----------
    with st.container(border=True):
        st.subheader("Real vs placebo (wrong-day labels)")
        if ps.empty:
            st.info("No placebo results for this range.", icon=":material/info:")
        else:
            plc = res["placebo"].assign(kind="Placebo")
            realr = ps.reset_index()[["method", "regime", "real"]].rename(columns={"real": "ratio"}).assign(kind="Real")
            allr = pd.concat([plc, realr], ignore_index=True)
            allr["row"] = allr["method"] + ", " + allr["regime"].map(names)
            pts = alt.Chart(allr).mark_point(filled=True).encode(
                y=alt.Y("row:N", title=None),
                x=alt.X("ratio:Q", title="In-regime ÷ out-of-regime volatility", scale=alt.Scale(zero=False)),
                color=alt.Color("kind:N", title=None, scale=alt.Scale(domain=["Real", "Placebo"], range=["#e45756", "#9aa5b1"]),
                                legend=alt.Legend(orient="top")),
                size=alt.condition(alt.datum.kind == "Real", alt.value(160), alt.value(50)),
                tooltip=["row", "kind", alt.Tooltip("shift_days:Q", title="Labels shifted (days)"),
                         alt.Tooltip("ratio:Q", format=".3f")],
            )
            one = alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(strokeDash=[4, 3], color="gray").encode(x="x:Q")
            st.altair_chart(alt.layer(one, pts).properties(height=220),
                            alt=f"Real volatility ratios against placebo ratios from shifted {noun} labels")
            st.caption(f"Gray: the same analysis with {noun} labels moved by ±1–4 weeks "
                       f"({len(rv.PLACEBO_SHIFTS)} shifts). If the red point sits among the gray ones, the ratio reflects "
                       f"timing (time of day, weekday, season), not {noun}.")

    # ---------- Ratio comparison ----------
    rows = []
    for key, method in (("garch_adj", "GARCH, time-of-day adjusted"), ("boot_adj", "Model-free, time-of-day adjusted"),
                        ("garch_raw", "GARCH, raw"), ("boot_raw", "Model-free, raw")):
        t = res[key]["ratios"] if key.startswith("garch") else res[key]
        for g in rv.IN_GROUPS:
            rows.append({"Method": method, "Regime": g, "ratio": t.loc[g, "vol_ratio"],
                         "lo": t.loc[g, "lo"], "hi": t.loc[g, "hi"]})
    cmp = pd.DataFrame(rows)
    with st.container(border=True):
        st.subheader("Volatility ratio by method")
        base = alt.Chart(cmp).encode(y=alt.Y("Method:N", title=None, sort=list(dict.fromkeys(cmp["Method"]))),
                                     yOffset=alt.YOffset("Regime:N"),
                                     color=alt.Color("Regime:N", scale=GROUP_COLORS, legend=alt.Legend(orient="top")))
        chart = alt.layer(
            base.mark_rule(strokeWidth=2).encode(x=alt.X("lo:Q", title="In-regime ÷ out-of-regime volatility",
                                                         scale=alt.Scale(type="log")), x2="hi:Q"),
            base.mark_point(filled=True, size=80).encode(
                x="ratio:Q", tooltip=["Method", "Regime", alt.Tooltip("ratio:Q", format=".3f"),
                                      alt.Tooltip("lo:Q", format=".3f"), alt.Tooltip("hi:Q", format=".3f")]),
            alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(strokeDash=[4, 3], color="gray").encode(x="x:Q"),
        ).properties(height=260)
        st.altair_chart(chart, alt="Volatility ratios with 95% intervals, by method and regime")
        st.caption(f"Raw ratios mix any {noun} effect with NG's normal intraday pattern (the 08:00 CT open, the "
                   "13:30 CT settlement). Compare them with the adjusted rows to see how much is just time of day.")

    # ---------- By hour + profile ----------
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.subheader("Raw volatility by hour")
        bh = res["by_hour"]
        st.altair_chart(alt.Chart(bh).mark_line(point=True).encode(
            x=alt.X("hour:O", title="Hour starting (CT)"), y=alt.Y("rms_bp:Q", title="RMS 15-min return (bp)"),
            color=alt.Color("group:N", title=None, scale=GROUP_COLORS, legend=alt.Legend(orient="top")),
            tooltip=["hour", "group", alt.Tooltip("rms_bp:Q", format=".1f"), "n"]),
            alt=f"Raw NG volatility by hour, in vs out of {noun} regimes")
        st.caption("Within the same hour, in-regime vs out-of-regime volatility. Hours with fewer than 30 returns "
                   "in a group are hidden.")
    with c2, st.container(border=True):
        st.subheader("Intraday volatility profile")
        if seasonal:
            prof = (df[~df["is_thu"]].groupby(["month", "slot"])["tod_scale"].first().reset_index()
                      .assign(Month=lambda x: x["month"].map(lambda m: calendar.month_abbr[m])))
            color = alt.Color("month:O", title="Month", scale=alt.Scale(scheme="viridis"),
                              legend=alt.Legend(orient="top", labelExpr="['','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][datum.value]"))
            tip = ["Month", "slot", alt.Tooltip("tod_scale:Q", format=".1f")]
        else:
            prof = (df.groupby(["is_thu", "slot"])["tod_scale"].first().reset_index()
                      .assign(Day=lambda x: np.where(x["is_thu"], "Thursday", "Other days")))
            color = alt.Color("Day:N", title=None, legend=alt.Legend(orient="top"))
            tip = ["Day", "slot", alt.Tooltip("tod_scale:Q", format=".1f")]
        st.altair_chart(alt.Chart(prof).mark_line(strokeWidth=1.2).encode(
            x=alt.X("slot:O", title="15-min slot ending (CT)", axis=alt.Axis(values=[f"{h:02d}:00" for h in range(0, 24, 3)])),
            y=alt.Y("tod_scale:Q", title="RMS 15-min return (bp)"), color=color, tooltip=tip),
            alt="NG intraday volatility profile used for the time-of-day adjustment")
        st.caption("Returns are divided by this profile before modeling. "
                   + ("One line per calendar month (non-Thursdays); Thursdays get a per-slot storage-report factor on top."
                      if seasonal else "Thursdays are separate because of the EIA storage report (09:30 ET)."))

    # ---------- Over time ----------
    with st.container(border=True):
        st.subheader("Volatility and price over the range")
        if (end - start).days + 1 > 31:
            st.info("Shown for ranges of 31 days or less.", icon=":material/zoom_in:")
        else:
            x = time_axis(start, end)
            t = df.assign(t=ct_naive(df["t_end_utc"] - STEP), t_next=ct_naive(df["t_end_utc"]))
            ins = t[t["group"].isin(rv.IN_GROUPS)].sort_values("t")
            bands = []
            if not ins.empty:
                run = ((ins["t"] != ins["t_next"].shift()) | (ins["group"] != ins["group"].shift())).cumsum()
                b = ins.groupby(run).agg(t=("t", "first"), t_next=("t_next", "last"), group=("group", "first"))
                b["band"] = b["group"].map(band)
                bands = [b[["t", "t_next", "band"]]]
            st.markdown("**GARCH conditional volatility** (bp per 15 min, time-of-day profile included)")
            layers = []
            if bands:
                layers.append(alt.Chart(bands[0]).mark_rect(opacity=0.18).encode(
                    x=x, x2="t_next:T", color=alt.Color("band:N", scale=BAND_SCALE, legend=alt.Legend(title=None, orient="top"))))
            layers.append(alt.Chart(t).mark_line(color="#54a24b", strokeWidth=1.2).encode(
                x=x, y=alt.Y("cond_vol_bp:Q", title="bp", axis=alt.Axis(**Y_AXIS)),
                tooltip=[alt.Tooltip("t_next:T", title="Return ending (CT)", format="%b %d %H:%M"),
                         alt.Tooltip("cond_vol_bp:Q", title="Cond. vol (bp)", format=".1f"),
                         alt.Tooltip("ret_bp:Q", title="Return (bp)", format="+.1f"), "group"]))
            st.altair_chart(alt.layer(*layers, height=PANEL_H), alt=f"GARCH conditional volatility {start} to {end}")
            st.markdown("**Henry Hub front month (NG.v.0)**")
            note = henry_hub_panel(start, end, x, bands, BAND_SCALE,
                                   f"Henry Hub front-month prices {start} to {end}, with {noun} regimes shaded")
            st.caption(f"Orange = {names['Up']}, blue = {names['Down']}." + note)

    # ---------- Model details ----------
    with st.container(border=True):
        st.subheader("Model details")
        full = res["garch_adj"]["full"]
        st.markdown("**GARCH(1,1)-X, Student-t, time-of-day adjusted returns**")
        st.dataframe(full["params"].round(4), alt="GARCH parameter estimates and standard errors")
        st.markdown("**Likelihood-ratio tests** (restricted model sets the regime effect to zero)")
        st.dataframe(res["garch_adj"]["tests"].round(4), alt="Likelihood-ratio tests for regime effects")
        st.markdown("**Returns per group**")
        st.dataframe(df["group"].value_counts().reindex(list(rv.GROUPS)).dropna().astype(int).rename("returns").to_frame().T,
                     hide_index=True, alt=f"Returns per {noun} group")
        st.caption(f"h_t = g_t·exp(θ_up·Up + θ_down·Down); g_t = ω + α·r²_(t−1)/exp(θ·X_(t−1)) + β·g_(t−1). "
                   f"Volatility ratio = exp(θ/2). Optimizer converged: {full['converged']}. Returns are chained across "
                   f"gaps (halt, weekends, quiet windows). The {noun} regimes are labeled with hindsight (Viterbi over "
                   "the whole range), so this measures association, not something tradable in real time.")

    data_checks(res["diag"], largest="Largest 15-minute returns",
                note="Returns are dropped across contract rolls and wherever a 15-minute window has no trades. "
                     "Databento flags some days as degraded quality; they are kept but listed here.")
