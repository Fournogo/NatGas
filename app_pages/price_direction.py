"""Does Henry Hub move up or down during generation regimes? Mean returns + event study vs placebo."""

from datetime import timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ramp_page import PRESETS, data_checks  # ramp_page puts pipeline/ on sys.path
from regime_vol_page import load_ng, load_source
import price_direction as pdir
import regime_vol as rv

KP = "pdir"
SOURCES = {"Solar surprise": "solar_error", "Solar ramps": "solar", "Gas ramps": "gas"}
EXPECT = {"solar_error": {"Up": "negative (more solar, less gas burn)", "Down": "positive (less solar, more gas burn)"}}
COLORS = alt.Scale(domain=["Up", "Down"], range=["#f58518", "#4c78a8"])


@st.cache_data(ttl="1h", max_entries=10, show_spinner="Labeling regimes and running the placebo…")
def analyze(source: str, start, end, p_stay: float) -> dict:
    return pdir.analyze(source, start, end, p_stay, ng=load_ng(), data=load_source(source))


st.title("Price direction")
st.caption("Does Henry Hub tend to rise or fall during a generation regime? Uses the same 15-minute NG returns "
           "and regime labels as the volatility pages. Each result is compared with a placebo that moves each "
           "day's labels to a different week (same weekday and time of day), which captures ordinary intraday "
           "and weekly drift.")

with st.container(horizontal=True, vertical_alignment="bottom"):
    src_label = st.segmented_control("Regimes from", list(SOURCES), default="Solar surprise", key=f"{KP}_src") \
        or "Solar surprise"
source = SOURCES[src_label]
names = rv.SOURCES[source]["names"]
data = load_source(source)
lo, hi = rv.date_bounds(source, data)
rk = f"{KP}_{source}_range"
if rk not in st.session_state:
    st.session_state[rk] = (lo, hi)  # direction tests need the whole sample


def apply_preset() -> None:
    days = PRESETS.get(st.session_state.get(f"{KP}_{source}_preset"))
    st.session_state[rk] = (lo if days is None else max(lo, hi - timedelta(days=days - 1)), hi)


with st.container(horizontal=True, vertical_alignment="bottom"):
    st.date_input("Date range (Central operating days)", key=rk, min_value=lo, max_value=hi)
    st.segmented_control("Quick range", list(PRESETS), key=f"{KP}_{source}_preset", on_change=apply_preset)
    p_stay = st.slider("HMM stickiness (p_stay)", 0.80, 0.995, value=0.95, step=0.005, format="%.3f",
                       key=f"{KP}_p_stay")

rng = st.session_state[rk]
if len(rng) < 2:
    st.info("Pick an end date.", icon=":material/date_range:")
    st.stop()
start, end = rng
res = analyze(source, start, end, p_stay)
if "means_bp" not in res:
    st.warning("Too few returns in this range.", icon=":material/warning:")
    st.stop()
df = res["df"]
if len(df) < 20000:
    st.warning(f"{len(df):,} returns. Mean 15-minute returns are tiny next to their noise; with less than "
               "about a year of data the intervals will be wide.", icon=":material/warning:")

# ---------- Headline ----------
pt = pdir.placebo_table(res)
with st.container(horizontal=True):
    st.metric("Returns analyzed", f"{len(df):,}", border=True)
    for g in rv.IN_GROUPS:
        row = pt.loc[g]
        ok = np.isfinite(row["real_diff_bp"])
        st.metric(f"Mean return, {names[g]} (vs rest)", f"{row['real_diff_bp']:+.2f} bp" if ok else "n/a",
                  (f"95% CI {row['ci_lo']:+.2f} to {row['ci_hi']:+.2f}; placebo {row['placebo_min']:+.2f} to "
                   f"{row['placebo_max']:+.2f}") if ok else "fewer than 50 returns in this regime",
                  delta_color="off", delta_arrow="off", border=True)
exp = EXPECT.get(source)
st.caption("Mean 15-minute return in the regime minus the mean in all other 15-minute periods, in basis points. "
           "CI from resampling whole days; placebo range from 8 wrong-day label shifts."
           + (f" Expected sign if the market prices the gas-burn effect: {names['Up']} → {exp['Up']}; "
              f"{names['Down']} → {exp['Down']}." if exp else ""))

# ---------- Real vs placebo ----------
c1, c2 = st.columns([2, 3])
with c1, st.container(border=True):
    st.subheader("Mean return vs placebo")
    pm = res["placebo_means"].assign(kind="Placebo")
    real = pt.reset_index()[["group", "real_diff_bp"]].rename(columns={"real_diff_bp": "diff"}).assign(kind="Real")
    allm = pd.concat([pm, real], ignore_index=True)
    allm["Regime"] = allm["group"].map(names)
    ci = pt.reset_index().assign(Regime=lambda x: x["group"].map(names))
    chart = alt.layer(
        alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[4, 3], color="gray").encode(x="x:Q"),
        alt.Chart(ci).mark_rule(color="#e45756", strokeWidth=2).encode(
            y=alt.Y("Regime:N", title=None), x=alt.X("ci_lo:Q", title="bp per 15 min, regime minus rest"), x2="ci_hi:Q"),
        alt.Chart(allm).mark_point(filled=True).encode(
            y="Regime:N", x="diff:Q",
            color=alt.Color("kind:N", title=None, scale=alt.Scale(domain=["Real", "Placebo"], range=["#e45756", "#9aa5b1"]),
                            legend=alt.Legend(orient="top")),
            size=alt.condition(alt.datum.kind == "Real", alt.value(160), alt.value(50)),
            tooltip=["Regime", "kind", alt.Tooltip("shift_days:Q", title="Labels shifted (days)"),
                     alt.Tooltip("diff:Q", format="+.3f")]),
    ).properties(height=180)
    st.altair_chart(chart, alt="Mean return difference in each regime, real vs placebo")
    st.caption("Red line: 95% CI of the real difference. Gray: wrong-day placebos.")

with c2, st.container(border=True):
    st.subheader("Stability by year")
    by = res["by_year"]
    if by.empty:
        st.info("Needs at least one year with 500+ returns.", icon=":material/info:")
    else:
        by = by.assign(Regime=lambda x: x["group"].map(names))
        base = alt.Chart(by).encode(x=alt.X("year:O", title=None), xOffset=alt.XOffset("Regime:N"),
                                    color=alt.Color("group:N", scale=COLORS, legend=None))
        st.altair_chart(alt.layer(
            alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(strokeDash=[4, 3], color="gray").encode(y="y:Q"),
            base.mark_rule(strokeWidth=2).encode(y=alt.Y("lo:Q", title="bp per 15 min, regime minus rest"), y2="hi:Q"),
            base.mark_point(filled=True, size=80).encode(
                y="diff:Q", tooltip=["year", "Regime", alt.Tooltip("diff:Q", format="+.2f"),
                                     alt.Tooltip("lo:Q", format="+.2f"), alt.Tooltip("hi:Q", format="+.2f"), "n"]),
        ).properties(height=180), alt="Mean return difference by calendar year with 95% CIs")
        st.caption("Orange = " + names["Up"] + ", blue = " + names["Down"] + ". A real effect should keep its sign "
                   "across years.")

# ---------- Event study ----------
with st.container(border=True):
    st.subheader("Average cumulative return around regime starts")
    paths, pp = res["paths"], res["placebo_paths"]
    cols = st.columns(len(rv.IN_GROUPS))
    for col, g in zip(cols, rv.IN_GROUPS):
        with col:
            real_p = paths[paths["group"] == g]
            if real_p.empty:
                st.info(f"No '{names[g]}' episodes.", icon=":material/info:")
                continue
            band = (pp[pp["group"] == g].groupby("minutes")["cum_bp"].agg(lo="min", hi="max").reset_index()
                    if not pp.empty else pd.DataFrame(columns=["minutes", "lo", "hi"]))
            st.markdown(f"**{names[g].capitalize()}** ({int(real_p['episodes'].iloc[0]):,} episodes)")
            st.altair_chart(alt.layer(
                alt.Chart(band).mark_area(opacity=0.25, color="#9aa5b1").encode(
                    x=alt.X("minutes:Q", title="Minutes from regime start"), y=alt.Y("lo:Q", title="Cumulative bp"), y2="hi:Q"),
                alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 3], color="gray").encode(x="x:Q"),
                alt.Chart(real_p).mark_line(point=True, color=COLORS.range[rv.IN_GROUPS.index(g)]).encode(
                    x="minutes:Q", y="cum_bp:Q",
                    tooltip=[alt.Tooltip("minutes:Q"), alt.Tooltip("cum_bp:Q", title="Cumulative bp", format="+.2f"),
                             alt.Tooltip("mean_bp:Q", title="Mean return this step", format="+.2f"), "n"]),
            ).properties(height=280), alt=f"Event study around '{names[g]}' regime starts")
    st.caption("Line: average cumulative NG return from 2 h before to 4 h after a regime episode starts, set to 0 "
               "just before the start. Gray band: range across the 8 wrong-day placebos. A real price response "
               "would leave the band after time 0.")

# ---------- Details ----------
with st.container(border=True):
    st.subheader("Details")
    st.markdown("**Mean returns (bp per 15 min)**")
    st.dataframe(res["means_bp"].rename(index=names).round(3), alt="Mean returns in and out of regime")
    st.markdown("**Vol-scaled returns** (units of a typical 15-minute move for that time of day and month)")
    st.dataframe(res["means_adj"].rename(index=names).round(4), alt="Mean vol-scaled returns in and out of regime")
    st.caption("Vol-scaled returns down-weight hectic periods, so their t-statistics are less noisy. Regime labels "
               "are look-back (HMM over the whole range), so this measures association, not a tradable signal. "
               "Many tests have been run across these pages; a single |t| near 2 is expected by chance.")

data_checks(res["diag"], largest="Largest 15-minute returns",
            note="Returns are dropped across contract rolls and wherever a 15-minute window has no trades.")
