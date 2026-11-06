"""NatGas Streamlit app (entry point).

Run:  .venv/bin/streamlit run streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="NatGas", page_icon=":material/monitoring:", layout="wide")

pg = st.navigation([
    st.Page("app_pages/data_check.py", title="Data check", icon=":material/fact_check:", default=True),
    st.Page("app_pages/gas_ramps.py", title="Gas ramps", icon=":material/ssid_chart:"),
    st.Page("app_pages/solar_ramps.py", title="Solar ramps", icon=":material/solar_power:"),
    st.Page("app_pages/solar_forecast_error.py", title="Solar forecast error", icon=":material/wb_sunny:"),
    st.Page("app_pages/solar_regime_vol.py", title="Solar regime volatility", icon=":material/stacked_line_chart:"),
    st.Page("app_pages/gas_regime_vol.py", title="Gas regime volatility", icon=":material/local_fire_department:"),
    st.Page("app_pages/solar_error_regime_vol.py", title="Solar surprise volatility", icon=":material/thunderstorm:"),
    st.Page("app_pages/price_direction.py", title="Price direction", icon=":material/swap_vert:"),
])
pg.run()
