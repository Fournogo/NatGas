"""Distribution of 15-minute ramp rates for ERCOT solar generation (night steps excluded)."""

from ramp_page import ramps, render  # ramp_page puts pipeline/ on sys.path

render(ramps.SOLAR, "Solar ramp rates", "solar", night_floor=ramps.SOLAR_NIGHT_FLOOR)
