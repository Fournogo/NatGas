"""Shared helpers for the Streamlit pages."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
CLEAN = ROOT / "data" / "clean"
CT = "America/Chicago"

# Let pages import the pipeline modules (they use `from common import ...`).
if str(ROOT / "pipeline") not in sys.path:
    sys.path.insert(0, str(ROOT / "pipeline"))


@st.cache_data(ttl="1h")
def load_ng() -> pd.DataFrame:
    df = pd.read_parquet(CLEAN / "ng_front_ohlcv1m.parquet")
    df["day"] = df["ts_event_ct"].dt.date
    return df


@st.cache_data(ttl="1h")
def load_fuel_mix() -> pd.DataFrame | None:
    path = CLEAN / "ercot_fuel_mix.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path, columns=["operating_date", "interval_end_utc", "fuel",
                                          "settlement_type", "mw_avg"])


def ct_naive(s: pd.Series) -> pd.Series:
    """Charts render tz-aware times in the viewer's zone; show Central time explicitly."""
    return s.dt.tz_convert(CT).dt.tz_localize(None)
