"""Pull ERCOT Public API reports, keeping EVERY published vintage.

How it works:
  1. List every posted document for a report via GET /archive/{emilId} (docId + postDatetime).
  2. Bulk-download them 1000 at a time via POST /archive/{emilId}/download.
  3. Save each batch zip untouched to data/raw/ercot/<emil_id>/batch_<first>_<last>.zip.
     Inside are ERCOT's own per-posting zips, whose names carry the docId and post time
     (e.g. 964571238.cdr.00013103.0000000000000000.20231211.010345177.HRLYRESOUTCAPNP3233_csv.zip),
     so "what was known as of time T" can always be reconstructed.
Re-running is safe: docIds already present in saved batches are skipped.

Credentials (in .env): ERCOT_USERNAME, ERCOT_PASSWORD, ERCOT_SUBSCRIPTION_KEY
  Register at https://apiexplorer.ercot.com and subscribe to the "Public API" product.

Usage:
  .venv/bin/python pipeline/pull_ercot.py --check                  # verify creds + token
  .venv/bin/python pipeline/pull_ercot.py --list                   # show configured reports
  .venv/bin/python pipeline/pull_ercot.py np3-233-cd               # pull one report
  .venv/bin/python pipeline/pull_ercot.py all                      # pull every report
  .venv/bin/python pipeline/pull_ercot.py np3-233-cd --clean       # rebuild clean parquet

Not covered here (not in the Public API, see data_sources.md):
  - Fuel mix (§3): yearly spreadsheets from ercot.com/gridinfo/generation
  - NP1-346-ER unplanned outages (§4b): ercot.com data products page
"""

import argparse
import io
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from common import SAMPLE_START, append_pull_log, clean_dir, env, raw_dir

API = "https://api.ercot.com/api/public-reports"
TOKEN_URL = (
    "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/"
    "B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
)
CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"

# emil_id -> description. Hourly reports first: they carry the forecast/outage vintages.
REPORTS = {
    "np3-233-cd": "Hourly Resource Outage Capacity, next 168h by load zone (§4a)",
    "np4-737-cd": "Solar: hourly averaged actual + forecast, system-wide (§2)",
    "np4-745-cd": "Solar: hourly averaged actual + forecast, by region (§2)",
    "np4-732-cd": "Wind: hourly averaged actual + forecast, system-wide (§2)",
    "np4-742-cd": "Wind: hourly averaged actual + forecast, by region (§2)",
    "np4-738-cd": "Solar: 5-minute actuals, system-wide (§2)",
    "np4-733-cd": "Wind: 5-minute actuals, system-wide (§2)",
}

# ERCOT renamed some columns mid-sample (Sep 2024 for the solar/wind reports). In cleaned data,
# merge the new name into the old one so a single column spans the whole window.
COLUMN_ALIASES = {  # new name -> candidate original names (first one present in the report wins)
    "SYSTEM_WIDE_GEN": ["ACTUAL_SYSTEM_WIDE", "GEN_SYSTEM_WIDE", "SYSTEM_WIDE"],
}
# Plus a general rule applied in clean(): GEN_<X> is merged into ACTUAL_<X> when ACTUAL_<X>
# exists (wind load zones / regions were renamed this way).

PAGE_SIZE = 1000   # archive listing page size
BATCH = 1000       # docIds per bulk download (ERCOT's downloadLimit)
REQUEST_PAUSE_S = 2.1  # stay under ERCOT's ~30 requests/minute limit

# Inner file names: <docId>.cdr.<rtid>.<16 zeros>.<YYYYMMDD>.<HHMMSSmmm>.<friendly>.zip
INNER_RE = re.compile(r"^(\d+)\.cdr\.\d+\.\d+\.(\d{8})\.(\d{6})\d*\.")


class ErcotClient:
    def __init__(self) -> None:
        self.user = env("ERCOT_USERNAME")
        self.password = env("ERCOT_PASSWORD")
        self.sub_key = env("ERCOT_SUBSCRIPTION_KEY")
        self._token = None
        self._token_time = 0.0
        self.s = requests.Session()

    def token(self) -> str:
        # Tokens last 1 hour and can't be refreshed; get a new one after 50 minutes.
        if not self._token or time.time() - self._token_time > 50 * 60:
            r = requests.post(TOKEN_URL, data={
                "username": self.user, "password": self.password, "grant_type": "password",
                "scope": f"openid {CLIENT_ID} offline_access", "client_id": CLIENT_ID,
                "response_type": "id_token",
            }, timeout=30)
            r.raise_for_status()
            body = r.json()
            self._token = body.get("id_token") or body["access_token"]
            self._token_time = time.time()
        return self._token

    def request(self, method: str, url: str, **kw) -> requests.Response:
        for attempt in range(6):
            time.sleep(REQUEST_PAUSE_S)
            try:
                r = self.s.request(method, url, timeout=600, headers={
                    "Authorization": f"Bearer {self.token()}",
                    "Ocp-Apim-Subscription-Key": self.sub_key,
                }, **kw)
            except requests.ConnectionError as e:
                print(f"  connection error ({e}); retrying")
                time.sleep(10 * (attempt + 1))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                wait = int(r.headers.get("Retry-After", 15 * (attempt + 1)))
                print(f"  HTTP {r.status_code}; sleeping {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        raise RuntimeError(f"Gave up after repeated failures: {method} {url}")

    def get(self, url: str, **params) -> requests.Response:
        return self.request("GET", url, params=params)

    def archives(self, emil_id: str, start: str, end: str | None) -> list[tuple[int, str]]:
        """Every archived document (docId, postDatetime) in the window, oldest first."""
        out, page = [], 1
        while True:
            params = {"postDatetimeFrom": f"{start}T00:00:00", "page": page, "size": PAGE_SIZE,
                      "sort": "postDatetime", "dir": "asc"}
            if end:
                params["postDatetimeTo"] = f"{end}T23:59:59"
            body = self.get(f"{API}/archive/{emil_id}", **params).json()
            out += [(a["docId"], a["postDatetime"]) for a in body.get("archives", [])]
            meta = body.get("_meta", {})
            if page == 1 or page % 25 == 0:
                print(f"  {emil_id}: listing page {page}/{meta.get('totalPages', '?')}", flush=True)
            if page >= meta.get("totalPages", 1):
                return out
            page += 1

    def bulk_download(self, emil_id: str, doc_ids: list[int]) -> bytes:
        return self.request("POST", f"{API}/archive/{emil_id}/download",
                            json={"docIds": doc_ids}).content


def _stamp(post_dt: str) -> str:
    return re.sub(r"[^0-9T]", "", post_dt)[:15]  # 2024-01-05T13:02:11 -> 20240105T130211


def saved_doc_ids(folder: Path) -> set[int]:
    have = set()
    for z in folder.glob("batch_*.zip"):
        with zipfile.ZipFile(z) as zf:
            have.update(int(n.split(".", 1)[0]) for n in zf.namelist())
    return have


UNAVAILABLE = "unavailable_docs.csv"  # docs ERCOT lists but refuses to serve (HTTP 400)


def unavailable_doc_ids(folder: Path) -> set[int]:
    f = folder / UNAVAILABLE
    if not f.exists():
        return set()
    return {int(line.split(",")[0]) for line in f.read_text().splitlines()[1:] if line}


def last_saved_post(folder: Path) -> str | None:
    """Latest post date among saved batches (from batch_<first>_<last>.zip names)."""
    stamps = [z.stem.split("_")[-1] for z in folder.glob("batch_*.zip")]
    return max(stamps)[:8] if stamps else None  # YYYYMMDD


def _save_batch(folder: Path, chunk: list[tuple[int, str]], content: bytes) -> int:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:  # validate before saving
        got = len(zf.namelist())
    path = folder / f"batch_{_stamp(chunk[0][1])}_{_stamp(chunk[-1][1])}.zip"
    path.write_bytes(content)  # untouched
    if got != len(chunk):
        print(f"  WARNING: asked for {len(chunk)} docs, got {got}")
    return got


def _download(client: ErcotClient, emil_id: str, folder: Path,
              chunk: list[tuple[int, str]], bad: list[tuple[int, str]]) -> int:
    """Download a chunk; if ERCOT rejects it (400), split it to isolate unservable docs."""
    try:
        return _save_batch(folder, chunk, client.bulk_download(emil_id, [d for d, _ in chunk]))
    except requests.HTTPError as e:
        if e.response is None or e.response.status_code != 400:
            raise
    if len(chunk) == 1:
        print(f"  {emil_id}: doc {chunk[0][0]} (posted {chunk[0][1]}) is listed but "
              f"not downloadable; recording in {UNAVAILABLE}")
        bad.append(chunk[0])
        return 0
    mid = len(chunk) // 2
    return (_download(client, emil_id, folder, chunk[:mid], bad)
            + _download(client, emil_id, folder, chunk[mid:], bad))


def pull(client: ErcotClient, emil_id: str, start: str, end: str | None,
         since_last: bool = False) -> None:
    out = raw_dir("ercot", emil_id)
    if since_last and (last := last_saved_post(out)):
        start = max(start, (pd.Timestamp(last) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        print(f"{emil_id}: --since-last, listing from {start}")
    docs = client.archives(emil_id, start, end)
    have = saved_doc_ids(out)
    skip = unavailable_doc_ids(out)
    todo = [(d, p) for d, p in docs if d not in have and d not in skip]
    print(f"{emil_id}: {len(docs):,} docs in window, {len(have):,} already saved, "
          f"{len(skip):,} known-unavailable, {len(todo):,} to download", flush=True)

    n_new, bad = 0, []
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        n_new += _download(client, emil_id, out, chunk, bad)
        print(f"  {emil_id}: {i + len(chunk):,}/{len(todo):,} downloaded "
              f"(through {chunk[-1][1]})", flush=True)

    if bad:
        f = out / UNAVAILABLE
        new_file = not f.exists()
        with f.open("a") as fh:
            if new_file:
                fh.write("doc_id,post_datetime,recorded_at_utc\n")
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            fh.writelines(f"{d},{p},{now}\n" for d, p in bad)

    append_pull_log(
        source="ercot_public_api", report_id=emil_id, start=start, end=end or "",
        rows=n_new, file=f"data/raw/ercot/{emil_id}/",
        notes=(f"docs={len(docs)} new={n_new} unavailable={len(bad)}"
               f"{' ' + ';'.join(str(d) for d, _ in bad) if bad else ''}; "
               f"first post {docs[0][1] if docs else '-'}; last post {docs[-1][1] if docs else '-'}"),
    )


def _localize_ct(naive: pd.Series, dst_flag: pd.Series | None = None) -> pd.Series:
    """Central Prevailing Time -> tz-aware. DSTFlag 'Y' marks the repeated (standard-time) hour."""
    if dst_flag is not None:
        ambiguous = ~dst_flag.astype(str).str.upper().isin(["Y", "TRUE"])
        return naive.dt.tz_localize("America/Chicago", ambiguous=ambiguous.to_numpy(),
                                    nonexistent="NaT")
    return naive.dt.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="NaT")


def _localize_posts(posts: pd.Series) -> pd.Series:
    """Localize naive Central post times (indexed by docId, sorted ascending).

    In the repeated fall-back hour, each posting could be CDT or CST. Because docIds
    increase with publication, the first k ambiguous postings are CDT and the rest CST.
    Pick the k that makes the UTC sequence increase with the most even spacing.
    """
    as_dst = posts.dt.tz_localize("America/Chicago", ambiguous=True, nonexistent="NaT")
    as_std = posts.dt.tz_localize("America/Chicago", ambiguous=False, nonexistent="NaT")
    out = as_dst.copy()
    amb = as_dst.ne(as_std) & as_dst.notna()
    for _, idx in amb[amb].groupby(posts[amb].dt.date).groups.items():
        pos = [posts.index.get_loc(i) for i in idx]
        # Frequent reports: the clock visibly jumps back (e.g. 01:55 -> 01:00) where CST starts.
        # Small reorderings between neighbouring docIds are common, so require a > 30 min drop.
        naive = posts.iloc[pos]
        drops = [j for j in range(1, len(pos))
                 if (naive.iloc[j - 1] - naive.iloc[j]) > pd.Timedelta(minutes=30)]
        if len(drops) == 1:
            k = drops[0]
            for j, p in enumerate(pos):
                out.iloc[p] = (as_dst if j < k else as_std).iloc[p]
            continue
        # Otherwise (e.g. hourly: 01:01:04 CDT then 01:01:05 CST) choose the split giving the
        # most evenly spaced, strictly increasing UTC sequence.
        lo, hi = max(pos[0] - 1, 0), min(pos[-1] + 2, len(posts))
        best, best_score = None, None
        for k in range(len(pos) + 1):
            trial = out.iloc[lo:hi].copy()
            for j, p in enumerate(pos):
                trial.iloc[p - lo] = (as_dst if j < k else as_std).iloc[p]
            gaps = trial.diff().dropna().dt.total_seconds()
            if (gaps <= 0).any():
                continue  # must increase
            score = gaps.max()
            if best_score is None or score < best_score:
                best, best_score = trial, score
        if best is not None:
            out.iloc[lo:hi] = best
        else:
            out.iloc[pos] = pd.NaT  # can't resolve: leave unknown rather than guess
    return out


def iter_postings(emil_id: str):
    """Yield (doc_id, posted_ct_naive, DataFrame) for every saved posting."""
    for batch in sorted(raw_dir("ercot", emil_id).glob("batch_*.zip")):
        with zipfile.ZipFile(batch) as outer:
            for name in outer.namelist():
                m = INNER_RE.match(name)
                if not m:
                    print(f"  skipping unexpected file {name}")
                    continue
                posted = datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S")
                with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                    for csv in (n for n in inner.namelist() if n.lower().endswith(".csv")):
                        yield int(m.group(1)), posted, pd.read_csv(io.BytesIO(inner.read(csv)))


def clean(emil_id: str) -> pd.DataFrame:
    """Stack every raw vintage into one long table with posted + delivery timestamps."""
    frames = []
    for doc_id, posted, df in iter_postings(emil_id):
        df.insert(0, "doc_id", doc_id)
        df.insert(1, "posted_ct_naive", posted)
        frames.append(df)
    if not frames:
        raise SystemExit(f"No raw files for {emil_id}; pull first.")
    df = pd.concat(frames, ignore_index=True)

    aliases = dict(COLUMN_ALIASES)
    for c in df.columns:
        if c.startswith("GEN_") and f"ACTUAL_{c[4:]}" in df:
            aliases[c] = [f"ACTUAL_{c[4:]}"]
    for new, olds in aliases.items():
        if new in df:
            old = next((o for o in olds if o in df), olds[0])
            if old in df:
                clash = df[new].notna() & df[old].notna()
                if clash.any():
                    raise SystemExit(f"{emil_id}: {new} and {old} both set on {clash.sum()} rows")
                df[old] = df[old].fillna(df[new])
                df = df.drop(columns=new)
            else:
                df = df.rename(columns={new: old})
            print(f"  {emil_id}: merged renamed column {new} into {old}")

    # Post times are Central with no DST flag. On fall-back night the 1 AM hour repeats; docIds
    # increase with time, so localize the postings in docId order and let pandas infer which
    # 1 AM is which.
    posts = (df[["doc_id", "posted_ct_naive"]].drop_duplicates("doc_id")
             .sort_values("doc_id").set_index("doc_id")["posted_ct_naive"])
    df["posted_ct"] = df["doc_id"].map(_localize_posts(posts))
    df = df.drop(columns="posted_ct_naive")
    df["posted_utc"] = df["posted_ct"].dt.tz_convert("UTC")

    dst = next((df[c] for c in ("DSTFlag", "DSTFLAG", "DST_FLAG") if c in df), None)
    date_col = next((c for c in ("DELIVERY_DATE", "DeliveryDate", "Date", "OperatingDate")
                     if c in df), None)
    he_col = next((c for c in ("HOUR_ENDING", "HourEnding") if c in df), None)
    int_col = next((c for c in ("INTERVAL_ENDING", "IntervalEnding") if c in df), None)

    if date_col and he_col:  # hourly: interval start = date + (HE - 1) hours, Central
        he = df[he_col].astype(str).str.extract(r"(\d+)")[0].astype(int)
        start_naive = pd.to_datetime(df[date_col], format="mixed") + pd.to_timedelta(he - 1, unit="h")
        df["interval_start_ct"] = _localize_ct(start_naive, dst)
        df["interval_start_utc"] = df["interval_start_ct"].dt.tz_convert("UTC")
    elif int_col:  # 5-minute: interval ending timestamp, Central
        df["interval_end_ct"] = _localize_ct(pd.to_datetime(df[int_col], format="mixed"), dst)
        df["interval_end_utc"] = df["interval_end_ct"].dt.tz_convert("UTC")
    else:
        print(f"  {emil_id}: no recognised delivery-time columns; columns = {list(df.columns)}")

    out = clean_dir() / f"ercot_{emil_id.replace('-', '_')}.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df):,} rows from {df['doc_id'].nunique():,} vintages to {out}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?", help="emil id (e.g. np3-233-cd) or 'all'")
    ap.add_argument("--start", default=SAMPLE_START)
    ap.add_argument("--end", default=None)
    ap.add_argument("--check", action="store_true", help="verify credentials and token only")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--clean", action="store_true", help="rebuild clean parquet from raw files")
    ap.add_argument("--since-last", action="store_true",
                    help="list only from (last saved post - 1 day); faster restarts")
    args = ap.parse_args()

    if args.list:
        for k, v in REPORTS.items():
            print(f"{k:12} {v}")
        return
    reports = list(REPORTS) if args.report == "all" else [args.report]
    if args.clean:
        for r in reports:
            clean(r)
        return

    client = ErcotClient()  # exits cleanly if credentials are missing
    if args.check:
        client.token()
        print("ERCOT credentials OK; token acquired at", datetime.now(timezone.utc).isoformat())
        return
    if not args.report:
        ap.error("give a report id or 'all'")
    for r in reports:
        pull(client, r, args.start, args.end, since_last=args.since_last)


if __name__ == "__main__":
    main()
