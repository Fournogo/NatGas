"""Progress bar for the ERCOT bulk pull (reads local files only; no API calls).

Usage:
  .venv/bin/python pipeline/ercot_progress.py          # print once
  .venv/bin/python pipeline/ercot_progress.py --watch  # refresh every 30s (Ctrl+C to stop)
"""

import argparse
import re
import time
import zipfile
from datetime import datetime

from common import DATA, RAW, SAMPLE_START
from pull_ercot import REPORTS

LOG = DATA.parent / "logs" / "ercot_pull.log"
WIDTH = 30
_count_cache: dict[str, tuple[float, int]] = {}  # batch path -> (mtime, doc count)


def saved(emil_id: str) -> int:
    n = 0
    for z in (RAW / "ercot" / emil_id).glob("batch_*.zip"):
        mt = z.stat().st_mtime
        hit = _count_cache.get(str(z))
        if not hit or hit[0] != mt:
            with zipfile.ZipFile(z) as zf:
                hit = (mt, len(zf.namelist()))
            _count_cache[str(z)] = hit
        n += hit[1]
    return n


def expected() -> dict[str, tuple[int, bool]]:
    """Total docs per report: exact from the pull log once listed, else an estimate."""
    exact = {}
    if LOG.exists():
        # Keep the largest listing per report; --since-last restarts only list a short window.
        for m in re.finditer(r"(np\d-\d+-cd): ([\d,]+) docs in window", LOG.read_text()):
            exact[m.group(1)] = max(exact.get(m.group(1), 0), int(m.group(2).replace(",", "")))
    hours = (datetime.now() - datetime.fromisoformat(SAMPLE_START)).total_seconds() / 3600
    out = {}
    for r in REPORTS:
        est = int(hours * (12 if r in ("np4-738-cd", "np4-733-cd") else 1))  # 5-min vs hourly
        out[r] = (exact[r], True) if r in exact else (est, False)
    return out


def bar(frac: float) -> str:
    full = int(frac * WIDTH)
    return "█" * full + "░" * (WIDTH - full)


def current_activity() -> str:
    if not LOG.exists():
        return "no log yet"
    lines = [l.strip() for l in LOG.read_text().splitlines() if l.strip()]
    return lines[-1] if lines else "starting"


def render(prev: tuple[float, int] | None) -> tuple[float, int]:
    exp = expected()
    rows, tot_have, tot_exp = [], 0, 0
    for r, (n_exp, exact) in exp.items():
        have = min(saved(r), n_exp)
        tot_have += have
        tot_exp += n_exp
        frac = have / n_exp if n_exp else 0
        mark = "" if exact else " (est. total)"
        rows.append(f"  {r}  {bar(frac)} {frac:6.1%}  {have:>7,}/{n_exp:,}{mark}")

    now = time.time()
    rate_txt = ""
    if prev and now > prev[0] and tot_have > prev[1]:
        rate = (tot_have - prev[1]) / (now - prev[0])  # docs/sec
        eta_h = (tot_exp - tot_have) / rate / 3600
        rate_txt = f"   ~{rate * 60:,.0f} docs/min, ETA ~{eta_h:.1f} h (downloads only; listing adds time)"

    frac = tot_have / tot_exp if tot_exp else 0
    print(f"\nERCOT pull  {datetime.now():%H:%M:%S}")
    print("\n".join(rows))
    print(f"  {'TOTAL':10}  {bar(frac)} {frac:6.1%}  {tot_have:>7,}/{tot_exp:,}{rate_txt}")
    print(f"  now: {current_activity()}")
    return now, tot_have


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--every", type=int, default=30, help="seconds between refreshes")
    args = ap.parse_args()
    prev = render(None)
    while args.watch:
        time.sleep(args.every)
        prev = render(prev)


if __name__ == "__main__":
    main()
