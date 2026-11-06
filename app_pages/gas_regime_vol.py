"""Henry Hub volatility inside vs outside gas-generation ramp regimes (Gas + Gas-CC combined)."""

from regime_vol_page import render  # puts pipeline/ on sys.path via ramp_page/app_lib

render("gas", "Gas regime volatility", "gas generation", "Gas ramps (Gas + Gas-CC combined)", "flat periods")
