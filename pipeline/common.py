"""Shared helpers: environment loading, data paths, and the pull log."""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CLEAN = DATA / "clean"
PULL_LOG = DATA / "pull_log.csv"

SAMPLE_START = "2023-12-11"  # earliest date in the ERCOT Public API; shared sample window

LOG_FIELDS = [
    "pulled_at_utc", "source", "report_id", "start", "end",
    "rows", "file", "cost_usd", "notes",
]

load_dotenv(ROOT / ".env")


def env(name: str, required: bool = True) -> str | None:
    val = os.environ.get(name) or None
    if required and not val:
        raise SystemExit(f"Missing {name}: set it in {ROOT / '.env'}")
    return val


def raw_dir(*parts: str) -> Path:
    p = RAW.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_dir() -> Path:
    CLEAN.mkdir(parents=True, exist_ok=True)
    return CLEAN


def append_pull_log(**row) -> None:
    """Append one row to data/pull_log.csv so gaps can be found later."""
    row.setdefault("pulled_at_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    unknown = set(row) - set(LOG_FIELDS)
    if unknown:
        raise ValueError(f"Unknown pull-log fields: {unknown}")
    new = not PULL_LOG.exists()
    PULL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PULL_LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})
