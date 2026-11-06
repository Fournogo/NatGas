"""Hourly ERCOT solar: actual vs the day-before (10:00 CT) forecast."""

from datetime import timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ramp_page import (BAND_PALETTE, CLASS_SCALE, LARGE, NOISE, PANEL_H, PRESETS, Y_AXIS,
                       data_checks, henry_hub_panel, time_axis)  # ramp_page puts pipeline/ on sys.path
from app_lib import ct_naive
import ramp_hmm
import solar_forecast_error as sfe

HMM_LABELS = ("Actual below forecast", "Near forecast", "Actual above forecast")
STEP = pd.Timedelta(hours=1)
KP = "sfe"
UNITS = ["z (std devs)", "MW", "HMM regime"]


@st.cache_data(ttl="1h")
def load() -> pd.DataFrame:
    return sfe.load()


@st.cache_data(ttl="1h", max_entries=20)
def compute(start, end, series: str):
    r = sfe.in_range(load(), start, end, series)
    return r, sfe.summarize(r), sfe.diagnostics(r, start, end)


@st.cache_data(ttl="1h", max_entries=20, show_spinner="Fitting the regime HMM…")
def hmm_fit(start, end, series: str, p_stay: float):
    r, _, _ = compute(start, end, series)
    return ramp_hmm.decode(r, p_stay, value="error", t="interval_start_utc", group="series",
                           step=STEP, labels=HMM_LABELS)


st.title("Solar forecast error")
st.caption("Hourly system-wide solar (NP4-737-CD). Error = actual − the day-before forecast, where the "
           "day-before forecast is the latest one ERCOT published at or before 10:00 CT the day before "
           "(Day-Ahead Market close). Positive = more solar than expected, i.e. less gas needed.")

if not sfe.OUT.exists():
    st.info("Not built yet. Run `pipeline/solar_forecast_error.py`.", icon=":material/hourglass_empty:")
    st.stop()

fe = load()
have = fe.dropna(subset=["actual", "forecast_STPPF"])
lo, hi = have["operating_date"].min(), have["operating_date"].max()
if f"{KP}_range" not in st.session_state:
    st.session_state[f"{KP}_range"] = (hi - timedelta(days=6), hi)


def apply_preset() -> None:
    days = PRESETS.get(st.session_state.get(f"{KP}_preset"))
    start = lo if days is None else max(lo, hi - timedelta(days=days - 1))
    st.session_state[f"{KP}_range"] = (start, hi)


with st.container(horizontal=True, vertical_alignment="bottom"):
    st.date_input("Date range (Central operating days)", key=f"{KP}_range", min_value=lo, max_value=hi)
    st.segmented_control("Quick range (ending on the last day of data)", list(PRESETS),
                         key=f"{KP}_preset", on_change=apply_preset)
    series = st.segmented_control("Forecast", list(sfe.SERIES), default="STPPF", key=f"{KP}_series",
                                  help="STPPF: ERCOT's short-term PV power forecast (central). "
                                       "PVGRPP: 80% probability-of-exceedance forecast (conservative, "
                                       "so errors run positive).") or "STPPF"

rng = st.session_state[f"{KP}_range"]
if len(rng) < 2:
    st.info("Pick an end date.", icon=":material/date_range:")
    st.stop()
start, end = rng
st.caption(f"Data runs {lo} to {hi}. Night hours (actual and forecast both below "
           f"{sfe.NIGHT_FLOOR:,.0f} MW) are left out of the histogram, statistics, z and the HMM.")

r, summary, diag = compute(start, end, series)
ok = r["usable"]
if not ok.any():
    st.warning("No usable hours in this range.", icon=":material/warning:")
    st.stop()
s = summary.iloc[0]

with st.container(horizontal=True, vertical_alignment="bottom"):
    bin_w = st.select_slider("Bin width (MW)", [50, 100, 250, 500, 1000], value=250, key=f"{KP}_bin")
    unit = st.segmented_control("Cutoff in", UNITS, default=UNITS[0], key=f"{KP}_unit")
    mode = {"MW": "mw", "HMM regime": "hmm"}.get(unit, "z")
    if mode == "mw":
        top = float(np.ceil(r.loc[ok, "error"].abs().max() / 250) * 250)
        cutoff = st.slider("Cutoff ± (MW)", 0.0, top, value=min(2000.0, top), step=250.0, key=f"{KP}_cut_mw")
    elif mode == "hmm":
        cutoff = st.slider("Stickiness (p_stay)", 0.80, 0.995, value=0.95, step=0.005, format="%.3f",
                           key=f"{KP}_p_stay",
                           help="Probability of staying in the same regime each hour. Higher = fewer, longer regimes.")
    else:
        cutoff = st.slider("Cutoff ± z", 0.0, 6.0, value=2.0, step=0.1, key=f"{KP}_cut_z",
                           help="z = (error − mean) / std over the selected range.")
    log_y = st.toggle("Log count axis", key=f"{KP}_log")

r = r.copy()
r["z"] = (r["error"] - s["mean (bias)"]) / s["std"]
params = None
if mode == "hmm":
    states, params = hmm_fit(start, end, series, cutoff)
    r["state"] = states.reindex(r.index)
    large = r["state"].isin([HMM_LABELS[0], HMM_LABELS[2]])
    lines = [params.iloc[0][f"mean {HMM_LABELS[i]}"] for i in (0, 2)] if len(params) else []
    p = params.iloc[0] if len(params) else None
    cut_txt = ("HMM regime means: " + "; ".join(f"{l.lower()} {p[f'mean {l}']:+,.0f} ± {p[f'std {l}']:,.0f}"
                                                 for l in HMM_LABELS) + " MW. Red dotted lines = outer regime means"
               if p is not None else "HMM not fitted")
elif mode == "z":
    large = (r["z"].abs() >= cutoff) & (cutoff > 0)
    lines = [s["mean (bias)"] - cutoff * s["std"], s["mean (bias)"] + cutoff * s["std"]] if cutoff > 0 else []
    cut_txt = (f"Red dotted lines = cutoff (±{cutoff:.1f} z = {lines[0]:+,.0f} / {lines[1]:+,.0f} MW)"
               if lines else "No cutoff")
else:
    large = (r["error"].abs() >= cutoff) & (cutoff > 0)
    lines = [-cutoff, cutoff] if cutoff > 0 else []
    cut_txt = f"Red dotted lines = cutoff (±{cutoff:,.0f} MW)" if lines else "No cutoff"
r["class"] = np.where(large & ok, LARGE, NOISE)
r["direction"] = np.where(r["error"] >= 0, "Actual above forecast", "Actual below forecast")

# ---------- Histogram ----------
u = r[ok]
n_large = int((u["class"] == LARGE).sum())
with st.container(border=True):
    st.subheader(f"Error distribution ({series})")
    with st.container(horizontal=True):
        st.metric("Hours", f"{int(s['n']):,}", border=True)
        st.metric("Bias (mean)", f"{s['mean (bias)']:+,.0f} MW", border=True)
        st.metric("MAE", f"{s['MAE']:,.0f} MW", border=True)
        st.metric("5th–95th pct", f"{s['p5']:+,.0f} / {s['p95']:+,.0f}", border=True)
        st.metric("Max under / over", f"{s['min']:+,.0f} / {s['max']:+,.0f}", border=True)
        st.metric("Large surprises", f"{n_large:,} ({n_large / max(len(u), 1):.0%})", border=True)
    edge = np.ceil(u["error"].abs().max() / bin_w) * bin_w
    b = (u.assign(bin_start=np.floor(u["error"] / bin_w) * bin_w)
          .groupby(["bin_start", "class"]).size().reset_index(name="count"))
    b["bin_end"] = b["bin_start"] + bin_w
    y = alt.Y("count:Q", title="Hours", scale=alt.Scale(type="log") if log_y else alt.Undefined)
    bars = alt.Chart(b).mark_bar().encode(
        x=alt.X("bin_start:Q", title="Actual − forecast (MW)", scale=alt.Scale(domain=[-edge, edge])),
        x2="bin_end:Q", y=y,
        color=alt.Color("class:N", scale=CLASS_SCALE, legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("bin_start:Q", title="From"), alt.Tooltip("bin_end:Q", title="To"),
                 alt.Tooltip("count:Q", title="Hours"), alt.Tooltip("class:N", title="Class")],
    )
    layers = [bars, alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 3], color="gray").encode(x="x:Q")]
    if lines:
        layers.append(alt.Chart(pd.DataFrame({"x": lines}))
                      .mark_rule(color="#e45756", strokeDash=[2, 2], strokeWidth=1.5).encode(x="x:Q"))
    st.altair_chart(alt.layer(*layers), alt=f"Histogram of hourly solar forecast error ({series}), {start} to {end}")
    st.caption(f"Std {s['std']:,.0f} MW. {cut_txt}. Left of the gray dashed line = less solar than forecast "
               "(more gas needed).")

# ---------- Over time ----------
with st.container(border=True):
    st.subheader("Actual vs forecast over the range")
    if (end - start).days + 1 > 31:
        st.info("Shown for ranges of 31 days or less.", icon=":material/zoom_in:")
    else:
        x = time_axis(start, end)
        t = r.assign(t=ct_naive(r["interval_start_utc"]), t_next=ct_naive(r["interval_start_utc"] + STEP))

        st.markdown("**Solar generation**")
        lv = t.melt(id_vars=["t"], value_vars=["actual", "forecast"], var_name="Series", value_name="MW")
        lv["Series"] = lv["Series"].map({"actual": "Actual", "forecast": f"Day-before {series}"})
        st.altair_chart(alt.Chart(lv, height=PANEL_H).mark_line(strokeWidth=1.6).encode(
            x=x, y=alt.Y("MW:Q", title="MW", axis=alt.Axis(**Y_AXIS)),
            color=alt.Color("Series:N", scale=alt.Scale(range=["#f58518", "#9aa5b1"]),
                            legend=alt.Legend(title=None, orient="top")),
            strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(range=[[1, 0], [5, 3]]), legend=None),
            tooltip=[alt.Tooltip("t:T", title="Hour starting (CT)", format="%b %d %H:%M"),
                     alt.Tooltip("Series:N"), alt.Tooltip("MW:Q", format=",.0f")],
        ), alt=f"Solar actual vs day-before {series} forecast, {start} to {end}")

        st.markdown("**Error (actual − forecast)**")
        tips = [alt.Tooltip("t:T", title="Hour starting (CT)", format="%b %d %H:%M"),
                alt.Tooltip("actual:Q", title="Actual", format=",.0f"),
                alt.Tooltip("forecast:Q", title="Forecast", format=",.0f"),
                alt.Tooltip("error:Q", title="Error", format="+,.0f"),
                alt.Tooltip("z:Q", title="z", format="+.2f"), alt.Tooltip("class:N", title="Class"),
                alt.Tooltip("forecast_posted_ct:T", title="Forecast published", format="%b %d %H:%M")]
        if mode == "hmm":
            tips.append(alt.Tooltip("state:N", title="HMM regime"))
        st.altair_chart(alt.Chart(t.dropna(subset=["error"]), height=PANEL_H).mark_rect().encode(
            x=x, x2="t_next:T", y=alt.Y("error:Q", title="MW", axis=alt.Axis(**Y_AXIS)), y2=alt.datum(0),
            color=alt.Color("class:N", scale=CLASS_SCALE, legend=alt.Legend(title=None, orient="top")),
            tooltip=tips,
        ), alt=f"Hourly solar forecast error, {start} to {end}, colored by class")

        st.markdown("**Henry Hub front month (NG.v.0)**")
        big = t[t["class"] == LARGE].sort_values("t")
        bands = []
        if not big.empty:
            # Merge consecutive large hours with the same sign into one band.
            run = ((big["t"] != big["t_next"].shift()) | (big["direction"] != big["direction"].shift())).cumsum()
            bands.append(big.groupby(run).agg(t=("t", "first"), t_next=("t_next", "last"),
                                              band=("direction", "first")))
        band_scale = alt.Scale(domain=["Actual above forecast", "Actual below forecast"],
                               range=[BAND_PALETTE[2], BAND_PALETTE[0]])
        ng_note = henry_hub_panel(start, end, x, bands, band_scale,
                                  f"Henry Hub front-month prices {start} to {end}, with large solar surprises shaded")
        hmm_note = (" The HMM labels each hour using the whole selected range (including later hours), "
                    "so it is a look-back labeling, not a real-time signal." if mode == "hmm" else "")
        st.caption("Bars are red where the hour's error is a large surprise. On the price chart, orange = "
                   "more solar than forecast, red = less solar than forecast (more gas burn)."
                   + hmm_note + ng_note)

# ---------- Stats ----------
with st.container(border=True):
    st.subheader("Summary (MW)")
    st.dataframe(summary.round(1), alt="Solar forecast error summary statistics")
    if params is not None:
        st.markdown("**HMM regimes** (MW)")
        st.dataframe(params.round(1), alt="Fitted HMM regime means and standard deviations")

data_checks(diag, largest="Largest errors",
            note="Corrupt postings (every value inflated) are dropped before the forecast vintage is picked, "
                 "so the forecast for nearby days may come from an earlier posting; that shows up as a stale "
                 "vintage. Single implausible hours are blanked. The repeated 1 AM hour on fall-back days is "
                 "missing from the source report (night, so no effect).")
