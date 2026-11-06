"""Sanity-check page: one day of Henry Hub futures next to ERCOT solar, wind and outages."""

from datetime import timedelta

import pandas as pd
import streamlit as st

from app_lib import CLEAN, CT, ct_naive, load_fuel_mix, load_ng

@st.cache_data(ttl="1h")
def load_ercot(emil_id: str, columns: list[str]) -> pd.DataFrame | None:
    path = CLEAN / f"ercot_{emil_id.replace('-', '_')}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path, columns=columns)


def hourly_actual_vs_forecast(df: pd.DataFrame, day, actual: str, forecast: str) -> pd.DataFrame:
    """For each hour of `day`: the final actual, and the forecast as published at midnight CT
    the day before delivery (i.e. what the market knew going into the day)."""
    day_start = pd.Timestamp(day, tz=CT)
    rows = df[(df["interval_start_ct"] >= day_start) & (df["interval_start_ct"] < day_start + timedelta(days=1))]
    if rows.empty:
        return rows
    final = rows.sort_values("posted_ct").dropna(subset=[actual]).groupby("interval_start_ct")[actual].last()
    known = rows[rows["posted_ct"] <= day_start]
    fcst = known.sort_values("posted_ct").groupby("interval_start_ct")[forecast].last()
    out = pd.DataFrame({"Actual": final, "Forecast as of midnight": fcst}).reset_index()
    out["Hour (CT)"] = out.pop("interval_start_ct").dt.tz_localize(None)
    return out.melt("Hour (CT)", var_name="Series", value_name="MW")


ng = load_ng()
fuel_mix = load_fuel_mix()
solar = load_ercot("np4-737-cd", ["posted_ct", "interval_start_ct", "ACTUAL_SYSTEM_WIDE", "STPPF_SYSTEM_WIDE"])
wind = load_ercot("np4-732-cd", None)
outages = load_ercot("np3-233-cd", None)

st.title("NatGas data check")
st.caption("One day of each dataset, to confirm the pulls worked. All times are Central.")

# Default to the latest day that has both NG trading and ERCOT solar data.
ng_days = set(ng["day"].unique())
if solar is not None:
    # Solar reports include ~7 days of forecasts; only count days that have actuals.
    solar_days = set(solar.dropna(subset=["ACTUAL_SYSTEM_WIDE"])["interval_start_ct"].dt.date.unique())
    both = sorted(ng_days & solar_days)
else:
    both = []
# Prefer the latest *complete* day (today is still in progress).
default_day = both[-2] if len(both) > 1 else (both[-1] if both else max(ng_days))

day = st.date_input("Day", value=default_day, min_value=min(ng_days), max_value=max(ng_days))

# ---------- Henry Hub ----------
with st.container(border=True):
    st.subheader("Henry Hub front-month futures (1-minute)")
    d = ng[ng["day"] == day]
    if d.empty:
        st.info("No NG trades on this day (weekend or holiday).", icon=":material/event_busy:")
    else:
        contract = ", ".join(d["contract"].unique())
        with st.container(horizontal=True):
            st.metric("Contract", contract, border=True)
            st.metric("Open", f"${d['open'].iloc[0]:.3f}", border=True)
            st.metric("Close", f"${d['close'].iloc[-1]:.3f}",
                      f"{d['close'].iloc[-1] - d['open'].iloc[0]:+.3f}", border=True)
            st.metric("Range", f"${d['low'].min():.3f} – ${d['high'].max():.3f}", border=True)
            st.metric("Volume", f"{int(d['volume'].sum()):,}", border=True)
        chart = pd.DataFrame({"Time (CT)": ct_naive(d["ts_event_utc"]), "Close ($/MMBtu)": d["close"]})
        st.line_chart(chart, x="Time (CT)", y="Close ($/MMBtu)", alt=f"NG 1-minute closes on {day}")
        flags = [f for f, on in (("contract roll", d["roll_flag"].any()),
                                 ("Databento degraded-quality day", d["degraded_day"].any())) if on]
        if flags:
            st.warning("This day includes a " + " and a ".join(flags) + ".", icon=":material/flag:")

# ---------- ERCOT fuel mix ----------
with st.container(border=True):
    st.subheader("ERCOT generation by fuel (15-minute)")
    if fuel_mix is None:
        st.info("Not parsed yet. Run `pipeline/clean_fuel_mix.py`.", icon=":material/hourglass_empty:")
    else:
        f = fuel_mix[fuel_mix["operating_date"] == day]
        if f.empty:
            last = fuel_mix["operating_date"].max()
            st.info(f"No fuel mix for this day. The workbooks run through {last}; download a newer "
                    "IntGenbyFuel file from ercot.com to extend them.", icon=":material/event_busy:")
        else:
            gas = f[f["fuel"].isin(["Gas", "Gas-CC"])].groupby("interval_end_utc")["mw_avg"].sum()
            total = f[f["fuel"] != "WSL"].groupby("interval_end_utc")["mw_avg"].sum()
            with st.container(horizontal=True):
                st.metric("Peak gas (Gas + Gas-CC)", f"{gas.max():,.0f} MW", border=True)
                st.metric("Min gas", f"{gas.min():,.0f} MW", border=True)
                st.metric("Gas share of generation", f"{gas.sum() / total.sum():.0%}", border=True)
            stack = f[f["fuel"] != "WSL"].assign(**{"Time (CT)": lambda d: ct_naive(d["interval_end_utc"])})
            stack = stack.rename(columns={"fuel": "Fuel", "mw_avg": "MW"})
            st.area_chart(stack, x="Time (CT)", y="MW", color="Fuel", stack=True,
                          alt=f"ERCOT generation by fuel on {day}")
            notes = ["Average MW per 15-minute interval (ending times). Storage charging (WSL) is "
                     "left out of the stack."]
            if f["settlement_type"].eq("INITIAL").any():
                notes.append("This day is still INITIAL settlement; ERCOT will revise it to FINAL.")
            st.caption(" ".join(notes))

# ---------- ERCOT solar / wind ----------
col1, col2 = st.columns(2)
for col, name, df, actual, fcst in (
    (col1, "ERCOT solar (system-wide)", solar, "ACTUAL_SYSTEM_WIDE", "STPPF_SYSTEM_WIDE"),
    (col2, "ERCOT wind (system-wide)", wind, "ACTUAL_SYSTEM_WIDE", "STWPF_SYSTEM_WIDE"),
):
    with col, st.container(border=True):
        st.subheader(name)
        if df is None or actual not in df or fcst not in df:
            st.info("Not cleaned yet. Run `pull_ercot.py --clean`.", icon=":material/hourglass_empty:")
            continue
        h = hourly_actual_vs_forecast(df, day, actual, fcst)
        if h.empty:
            st.info("No data for this day yet.", icon=":material/hourglass_empty:")
        else:
            st.line_chart(h, x="Hour (CT)", y="MW", color="Series", alt=f"{name} on {day}")

with st.container(border=True):
    st.subheader("ERCOT outaged capacity (NP3-233-CD)")
    if outages is None:
        st.info("Not cleaned yet. Run `pull_ercot.py --clean`.", icon=":material/hourglass_empty:")
    else:
        zones = [c for c in outages if c.startswith("TotalResourceMWZone")]
        day_start = pd.Timestamp(day, tz=CT)
        o = outages[(outages["interval_start_ct"] >= day_start)
                    & (outages["interval_start_ct"] < day_start + timedelta(days=1))].copy()
        if o.empty:
            st.info("No data for this day yet.", icon=":material/hourglass_empty:")
        else:
            o["Total MW"] = o[zones].sum(axis=1)
            o = o.sort_values("posted_ct")
            latest = o.groupby("interval_start_ct")["Total MW"].last().rename("Latest report")
            known = o[o["posted_ct"] <= day_start - timedelta(days=1)]
            prior = known.groupby("interval_start_ct")["Total MW"].last().rename("Reported a day earlier")
            out = pd.concat([latest, prior], axis=1).reset_index()
            out["Hour (CT)"] = out.pop("interval_start_ct").dt.tz_localize(None)
            out = out.melt("Hour (CT)", var_name="Series", value_name="MW")
            st.line_chart(out, x="Hour (CT)", y="MW", color="Series",
                          alt=f"ERCOT total outaged resource capacity on {day}")
            st.caption(f"{o['doc_id'].nunique()} hourly reports cover this day. A gap between the "
                       "two lines is the kind of surprise the study looks for.")
