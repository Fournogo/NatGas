"""Shared renderer for the ramp-rate pages (Gas ramps, Solar ramps)."""

from datetime import timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_lib import CT, ct_naive, load_ng  # app_lib also puts pipeline/ on sys.path
import ramp_hmm
import ramps

PRESETS = {"1 week": 7, "1 month": 30, "1 year": 365, "All": None}
UNITS = ["z (std devs)", "MW per 15 min", "HMM regime"]
LARGE, NOISE = "Large ramp", "Noise"
CLASS_SCALE = alt.Scale(domain=[LARGE, NOISE], range=["#e45756", "#9aa5b1"])
BAND_PALETTE = ["#e45756", "#b279a2", "#f58518"]
PANEL_H = 340
Y_AXIS = dict(minExtent=70, maxExtent=70)  # same y-axis width on every panel so the x axes line up


@st.cache_data(ttl="1h")
def load(fuels: tuple) -> pd.DataFrame:
    return ramps.load_fuel_mix(fuels)


@st.cache_data(ttl="1h", max_entries=20)
def compute(fuels: tuple, start, end, night_floor):
    fm = load(fuels)
    d = ramps.forward_diffs(fm, start, end, fuels, night_floor)
    return d, ramps.summarize(d), ramps.diagnostics(fm, d, start, end, fuels)


@st.cache_data(ttl="1h", max_entries=20, show_spinner="Fitting the regime HMM…")
def hmm_fit(fuels: tuple, start, end, night_floor, p_stay: float):
    d, _, _ = compute(fuels, start, end, night_floor)
    return ramp_hmm.decode(d, p_stay)


def ramp_bands(seg: pd.DataFrame, fuel: str) -> pd.DataFrame:
    """Merge consecutive large-ramp steps of one fuel into [t, t_next) bands."""
    r = seg[(seg["fuel"] == fuel) & (seg["class"] == LARGE)].sort_values("t")
    if r.empty:
        return pd.DataFrame(columns=["t", "t_next", "band"])
    run = (r["t"] != r["t_next"].shift()).cumsum()
    out = r.groupby(run).agg(t=("t", "first"), t_next=("t_next", "last"))
    out["band"] = f"{fuel} large ramp"
    return out


def henry_hub_panel(start, end, x, bands: list[pd.DataFrame], band_scale: alt.Scale, alt_text: str) -> str:
    """Henry Hub front-month closes over [start, end] (Central days) on time encoding `x`, with
    `bands` (frames of t, t_next, band) shaded behind. Returns a caption fragment."""
    w0 = pd.Timestamp(start, tz=CT)
    w1 = pd.Timestamp(end + timedelta(days=1), tz=CT)
    ng = load_ng()
    n = ng[(ng["ts_event_ct"] >= w0) & (ng["ts_event_ct"] < w1)]
    if n.empty:
        st.info("No Henry Hub trades in this range.", icon=":material/event_busy:")
        return ""
    n = n.set_index("ts_event_utc")
    res = "1min" if (end - start).days + 1 <= 7 else "15min"  # keep long ranges light
    if res != "1min":
        n = n.groupby("contract").resample(res, label="right", closed="right").agg(
            close=("close", "last")).dropna(subset=["close"]).reset_index()
    else:
        n = n.reset_index()
    n["t"] = ct_naive(n["ts_event_utc"])
    line = alt.Chart(n).mark_line(strokeWidth=1.4).encode(
        x=x, y=alt.Y("close:Q", title="$/MMBtu", scale=alt.Scale(zero=False), axis=alt.Axis(**Y_AXIS)),
        color=alt.Color("contract:N", legend=alt.Legend(title="Contract", orient="top")),
        tooltip=[alt.Tooltip("t:T", title="Time (CT)", format="%b %d %H:%M"),
                 alt.Tooltip("close:Q", title="Close", format="$.3f"),
                 alt.Tooltip("contract:N", title="Contract")],
    )
    layers = []
    if bands:
        layers.append(alt.Chart(pd.concat(bands, ignore_index=True)).mark_rect(opacity=0.18).encode(
            x=x, x2="t_next:T",
            color=alt.Color("band:N", scale=band_scale, legend=alt.Legend(title=None, orient="top"))))
    layers.append(line)
    st.altair_chart(alt.layer(*layers, height=PANEL_H).resolve_scale(color="independent"), alt=alt_text)
    return (f" Henry Hub: {'1-minute' if res == '1min' else '15-minute'} closes"
            f"{'; colored by contract (the front month rolled in this range)' if n['contract'].nunique() > 1 else ''}."
            " Lines bridge weekends and the daily trading halt.")


def time_axis(start, end) -> alt.X:
    """Shared Central-time x encoding over [start, end] so stacked charts line up."""
    w0 = pd.Timestamp(start, tz=CT)
    w1 = pd.Timestamp(end + timedelta(days=1), tz=CT)
    return alt.X("t:T", title="Time (CT)", scale=alt.Scale(
        domain=[ct_naive(pd.Series([w0])).iloc[0].isoformat(), ct_naive(pd.Series([w1])).iloc[0].isoformat()]))


def data_checks(diag: dict, largest: str = "Largest ramps",
                note: str = "INITIAL days are not errors, but ERCOT will revise them. Workbook seams are where "
                            "one yearly file ends and the next begins; the ramp across the seam should look ordinary.") -> None:
    issues = {k: v for k, v in diag.items() if k != largest and v["count"]}
    exp_label = "Data checks: " + (f"{len(issues)} to review" if issues else "all clear")
    with st.expander(exp_label, icon=":material/warning:" if issues else ":material/check_circle:",
                     expanded=bool(issues)):
        for name, chk in diag.items():
            if name == largest:
                continue
            icon = ":orange[:material/warning:]" if chk["count"] else ":green[:material/check:]"
            st.markdown(f"{icon} **{name}:** {chk['count']:,}")
            if chk["count"] and len(chk["examples"]):
                st.dataframe(chk["examples"], hide_index=True, alt=f"Examples: {name}")
        st.markdown(f"**{largest}** (check these against the chart or other sources)")
        ex = diag[largest]["examples"]
        st.dataframe(ex.round({c: 1 for c in ex.select_dtypes("number").columns}), hide_index=True, alt=largest)
        st.caption(note)


def render(fuels: tuple, title: str, kp: str, night_floor: float | None = None) -> None:
    """Draw a ramp page for `fuels`. `kp` prefixes widget keys; `night_floor` (MW) excludes
    steps where both ends are below it (solar nights)."""
    label = " and ".join(fuels)
    st.title(title)
    st.caption("Forward difference of 15-minute average generation: d(t) = MW(t + 15 min) − MW(t). "
               "Only consecutive 15-minute intervals are differenced; any other step counts as a gap. "
               "Source: ERCOT fuel mix workbooks.")

    if not ramps.FUEL_MIX.exists():
        st.info("Fuel mix not parsed yet. Run `pipeline/clean_fuel_mix.py`.", icon=":material/hourglass_empty:")
        st.stop()

    fm = load(fuels)
    lo, hi = fm["operating_date"].min(), fm["operating_date"].max()
    k_range, k_preset = f"{kp}_range", f"{kp}_preset"
    if k_range not in st.session_state:
        st.session_state[k_range] = (hi - timedelta(days=6), hi)

    def apply_preset() -> None:
        days = PRESETS.get(st.session_state.get(k_preset))
        start = lo if days is None else max(lo, hi - timedelta(days=days - 1))
        st.session_state[k_range] = (start, hi)

    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.date_input("Date range (Central operating days)", key=k_range, min_value=lo, max_value=hi)
        st.segmented_control("Quick range (ending on the last day of data)", list(PRESETS),
                             key=k_preset, on_change=apply_preset)

    rng = st.session_state[k_range]
    if len(rng) < 2:
        st.info("Pick an end date.", icon=":material/date_range:")
        st.stop()
    start, end = rng
    note = f"Fuel mix data runs {lo} to {hi}."
    if night_floor is not None:
        note += (f" Night steps (both ends below {night_floor:,.0f} MW) are left out of the histograms, "
                 "statistics, z and the HMM, and drawn gray on the chart.")
    st.caption(note)

    d, summary, diag = compute(fuels, start, end, night_floor)
    if not ramps.usable(d).any():
        st.warning("No usable consecutive intervals in this range.", icon=":material/warning:")
        st.stop()

    with st.container(horizontal=True, vertical_alignment="bottom"):
        bin_w = st.select_slider("Bin width (MW per 15 min)", [10, 25, 50, 100, 200, 500], value=50,
                                 key=f"{kp}_bin")
        binned = ramps.bin_ramps(d, bin_w)
        ok = ramps.usable(binned)
        unit = st.segmented_control("Cutoff in", UNITS, default=UNITS[0], key=f"{kp}_unit")
        mode = {"MW per 15 min": "mw", "HMM regime": "hmm"}.get(unit, "z")
        if mode == "mw":
            top = float(np.ceil(binned.loc[ok, "d_mw_15min"].abs().max() / 50) * 50)
            cutoff = st.slider("Cutoff ± (MW per 15 min)", 0.0, top, value=min(500.0, top), step=25.0,
                               key=f"{kp}_cut_mw")
        elif mode == "hmm":
            cutoff = st.slider("Stickiness (p_stay)", 0.80, 0.995, value=0.95, step=0.005, format="%.3f",
                               key=f"{kp}_p_stay",
                               help="Probability of staying in the same regime each 15 minutes. "
                                    "Higher = fewer, longer regimes (less flicker).")
        else:
            cutoff = st.slider("Cutoff ± z", 0.0, 6.0, value=2.0, step=0.1, key=f"{kp}_cut_z",
                               help="z = (change − mean) / std, per fuel over the selected range.")
        log_y = st.toggle("Log count axis", key=f"{kp}_log")

    # z per fuel, using that fuel's mean and std over the selected range (usable steps only).
    binned["z"] = ((binned["d_mw_15min"] - binned["fuel"].map(summary["mean"]))
                   / binned["fuel"].map(summary["std"]))
    params = None
    if mode == "hmm":
        states, params = hmm_fit(fuels, start, end, night_floor, cutoff)
        binned["state"] = states.reindex(binned.index)
        large = binned["state"].isin([ramp_hmm.LABELS[0], ramp_hmm.LABELS[2]])
    elif mode == "z":
        large = (binned["z"].abs() >= cutoff) & (cutoff > 0)
    else:
        large = (binned["d_mw_15min"].abs() >= cutoff) & (cutoff > 0)
    binned["class"] = np.where(large & ok, LARGE, NOISE)

    def cutoff_lines(fuel: str) -> list[float]:
        """Where to draw dotted lines on the histogram (MW per 15 min)."""
        if mode == "hmm":
            return [params.loc[fuel, f"mean {ramp_hmm.LABELS[i]}"] for i in (0, 2)] if fuel in params.index else []
        if cutoff <= 0:
            return []
        if mode == "mw":
            return [-cutoff, cutoff]
        m, s = summary.loc[fuel, "mean"], summary.loc[fuel, "std"]
        return [m - cutoff * s, m + cutoff * s]

    def cutoff_text(fuel: str) -> str:
        if mode == "hmm":
            if fuel not in params.index:
                return "HMM not fitted"
            p = params.loc[fuel]
            parts = [f"{l.lower()} {p[f'mean {l}']:+,.0f} ± {p[f'std {l}']:,.0f}" for l in ramp_hmm.LABELS]
            return (f"HMM regime means: {'; '.join(parts)} MW per 15 min "
                    f"({int(p['EM iterations'])} EM iterations). Red dotted lines = up/down regime means")
        lines = cutoff_lines(fuel)
        if not lines:
            return "No cutoff"
        if mode == "z":
            return f"Red dotted lines = cutoff (±{cutoff:.1f} z = {lines[0]:+,.0f} / {lines[1]:+,.0f} MW per 15 min)"
        return f"Red dotted lines = cutoff (±{cutoff:,.0f} MW per 15 min)"

    # ---------- Histograms ----------
    usable_b = binned[ok]
    edge = np.ceil(usable_b["d_mw_15min"].abs().max() / bin_w) * bin_w
    domain = [-edge, edge]

    cols = st.columns(len(fuels))
    for col, fuel in zip(cols, fuels):
        s = summary.loc[fuel]
        fb = usable_b[usable_b["fuel"] == fuel]
        n_large = int((fb["class"] == LARGE).sum())
        with col, st.container(border=True):
            st.subheader(fuel)
            with st.container(horizontal=True):
                st.metric("Intervals", f"{int(s['n']):,}", border=True)
                st.metric("Median", f"{s['median']:+,.0f}", border=True)
                st.metric("5th–95th pct", f"{s['p5']:+,.0f} / {s['p95']:+,.0f}", border=True)
                st.metric("Max down / up", f"{s['min']:+,.0f} / {s['max']:+,.0f}", border=True)
                st.metric("Large ramps", f"{n_large:,} ({n_large / max(len(fb), 1):.0%})", border=True)
            # Pre-bin in pandas so the full dataset doesn't ship 130k points to the browser.
            # A bin straddling the cutoff is split into stacked red/gray parts.
            b = fb.groupby(["bin_start", "class"]).size().reset_index(name="count")
            b["bin_end"] = b["bin_start"] + bin_w
            y = alt.Y("count:Q", title="Intervals", scale=alt.Scale(type="log") if log_y else alt.Undefined)
            bars = alt.Chart(b).mark_bar().encode(
                x=alt.X("bin_start:Q", title="Change in MW over 15 min", scale=alt.Scale(domain=domain)),
                x2="bin_end:Q", y=y,
                color=alt.Color("class:N", scale=CLASS_SCALE, legend=alt.Legend(title=None, orient="top")),
                tooltip=[alt.Tooltip("bin_start:Q", title="From"), alt.Tooltip("bin_end:Q", title="To"),
                         alt.Tooltip("count:Q", title="Intervals"), alt.Tooltip("class:N", title="Class")],
            )
            zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 3], color="gray").encode(x="x:Q")
            layers = [bars, zero]
            if lines := cutoff_lines(fuel):
                layers.append(alt.Chart(pd.DataFrame({"x": lines}))
                              .mark_rule(color="#e45756", strokeDash=[2, 2], strokeWidth=1.5).encode(x="x:Q"))
            st.altair_chart(alt.layer(*layers), alt=f"Histogram of 15-minute {fuel} ramps, {start} to {end}")
            st.caption(f"Mean {s['mean']:+,.0f}, std {s['std']:,.0f} MW per 15 min "
                       f"({s['std (MW/min)']:,.1f} MW/min). {cutoff_text(fuel)}. "
                       "Left of the gray dashed line = output falling.")

    # ---------- Generation, split by the cutoff ----------
    band_scale = alt.Scale(domain=[f"{f} large ramp" for f in fuels], range=BAND_PALETTE[:len(fuels)])
    with st.container(border=True):
        st.subheader("Generation over the range")
        if (end - start).days + 1 > 31:
            st.info("Shown for ranges of 31 days or less.", icon=":material/zoom_in:")
        else:
            x = time_axis(start, end)

            # Each forward difference is drawn as the segment it describes: t -> t + 15 min.
            cols_ = ["fuel", "interval_end_utc", "next_interval_utc", "mw_avg", "next_mw_avg",
                     "d_mw_15min", "z", "class"] + (["state"] if mode == "hmm" else [])
            seg = binned[cols_].assign(t=lambda x: ct_naive(x["interval_end_utc"]),
                                       t_next=lambda x: ct_naive(x["next_interval_utc"]))
            tips = [alt.Tooltip("t:T", title="From (CT)", format="%b %d %H:%M"),
                    alt.Tooltip("mw_avg:Q", title="MW", format=",.0f"),
                    alt.Tooltip("d_mw_15min:Q", title="Change (MW/15 min)", format="+,.0f"),
                    alt.Tooltip("z:Q", title="z", format="+.2f"),
                    alt.Tooltip("class:N", title="Class")]
            if mode == "hmm":
                tips.append(alt.Tooltip("state:N", title="HMM regime"))
            for i, fuel in enumerate(fuels):
                fs = seg[seg["fuel"] == fuel]
                st.markdown(f"**{fuel}**")
                chart = alt.Chart(fs, height=PANEL_H).mark_rule().encode(
                    x=x, x2="t_next:T",
                    y=alt.Y("mw_avg:Q", title="MW", scale=alt.Scale(zero=False), axis=alt.Axis(**Y_AXIS)),
                    y2="next_mw_avg:Q",
                    color=alt.Color("class:N", scale=CLASS_SCALE,
                                    legend=alt.Legend(title=None, orient="top") if i == 0 else None),
                    strokeWidth=alt.condition(alt.datum["class"] == LARGE, alt.value(2.5), alt.value(1.2)),
                    tooltip=tips,
                )
                st.altair_chart(chart, alt=f"{fuel} generation {start} to {end}, colored by ramp class")

            # Henry Hub front month over the same window, with the large-ramp periods shaded.
            st.markdown("**Henry Hub front month (NG.v.0)**")
            shade = st.segmented_control("Shade large ramps from", list(fuels), selection_mode="multi",
                                         default=list(fuels), key=f"{kp}_shade")
            parts = [b for b in (ramp_bands(seg, f) for f in (shade or [])) if not b.empty]
            ng_note = henry_hub_panel(start, end, x, parts, band_scale,
                                      f"Henry Hub front-month prices {start} to {end}, with large {label} ramps shaded")
            how = ("its HMM regime is ramping up or down" if mode == "hmm"
                   else "its change is beyond the ± cutoff")
            hmm_note = (" The HMM labels each step using the whole selected range (including later steps), "
                        "so it is a look-back labeling, not a real-time signal." if mode == "hmm" else "")
            st.caption(f"Each 15-minute generation step is red (a large ramp) if {how}, gray otherwise. "
                       "The same steps are shaded on the price chart." + hmm_note + ng_note)

    # ---------- Stats ----------
    with st.container(border=True):
        st.subheader("Summary (MW per 15 min)")
        st.dataframe(summary.round(1), alt=f"Ramp-rate summary statistics, {label}")
        if params is not None:
            st.markdown("**HMM regimes** (MW per 15 min)")
            st.dataframe(params.round(1), alt="Fitted HMM regime means and standard deviations")

    data_checks(diag)
