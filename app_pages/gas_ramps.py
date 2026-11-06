"""Distribution of 15-minute ramp rates for ERCOT gas generation (Gas vs Gas-CC)."""

from ramp_page import ramps, render  # ramp_page puts pipeline/ on sys.path

render(ramps.GAS, "Gas ramp rates", "gas")
