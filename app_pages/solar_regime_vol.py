"""Henry Hub volatility inside vs outside solar ramp regimes."""

from regime_vol_page import render  # puts pipeline/ on sys.path via ramp_page/app_lib

render("solar", "Solar regime volatility", "solar", "Solar ramps", "flat daytime + night")
