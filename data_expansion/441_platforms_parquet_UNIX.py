#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-converted from Jupyter notebook:
  441 platforms parquet.ipynb
Generated on 2026-02-07T00:57:46.236192Z

Notes:
- This script preserves code cell order from the notebook.
- Markdown cells are included as comments for readability.
"""


"""
Snapshot 441 spaces data collection (proposals + votes) — storage-safe final version

✅ Writes to /snapshot_votes_441 (configurable)
✅ Skips heavy/unreliable spaces (e.g., stgdao.eth)
✅ Avoids skip<=5000 issue using created_lt cursor pagination
✅ Handles 524/timeout by dynamic page_size reduction + multi-round attempts
✅ Disk-space guard (stop cleanly before filling disk)
✅ ML-friendly output:
   - Preferred: Parquet parts (requires pyarrow)
   - Fallback: CSV.GZ parts (no extra deps)
✅ Robust progress / skip logs per space

Output columns per vote-row:
Space, FollowersCount, Proposal ID, Proposal Title, Proposal Body, Created Time,
Voter, Choice, Voting Power, VP Ratio (%), Is Whale, Aligned With Majority, Vote Timestamp
"""

import os
import time
import json
import shutil
import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

SNAPSHOT_URL = "https://hub.snapshot.org/graphql"

# ==========================
# ✅ CONFIG (edit here)
# ==========================
SPACES_FILE = "data/processed/selected_spaces_follower_knee.json"

# ✅ Put outputs on D drive
OUT_DIR = "/home/carlo/data/summer_voters/snapshot_votes_441"

# Output mode:
#   "parquet"  -> requires pyarrow, writes many small parquet parts (safe on crashes)
#   "csv_gz"   -> fallback if pyarrow not installed, writes gzipped csv parts
OUTPUT_MODE = "parquet"  # "parquet" or "csv_gz"

# Space-level skip
SKIP_SPACES = {"stgdao.eth"}

# Proposals paging
PROPOSAL_BATCH = 100

# Votes paging (dynamic downshift under errors)
VOTES_PAGE_SIZE = 1000
MIN_VOTES_PAGE_SIZE = 50
MAX_PROPOSAL_FAILURE_ROUNDS = 3
PAGE_SIZE_LADDER = [1000, 500, 250, 200, 100, 50]

# Rate limiting / retries
SLEEP_BETWEEN_CALLS = 1.6
MAX_RETRIES = 6
BACKOFF_BASE = 2.0
HTTP_TIMEOUT = (10, 200)  # (connect, read)

# Optional: sort spaces by followersCount first (extra API calls)
SORT_BY_FOLLOWERS = True

# Safety: stop if disk free < this threshold
MIN_FREE_GB_TO_CONTINUE = 8.0

# Optional test limits
MAX_SPACES = None
MAX_PROPOSALS_PER_SPACE = None
# ==========================


# ==========================
# ✅ Paths
# ==========================
SPACE_DIR = os.path.join(OUT_DIR, "spaces")              # output parts
PROGRESS_DIR = os.path.join(OUT_DIR, "progress_by_space")
SKIP_DIR = os.path.join(OUT_DIR, "skip_by_space")
LOG_DIR = os.path.join(OUT_DIR, "logs")

for d in [OUT_DIR, SPACE_DIR, PROGRESS_DIR, SKIP_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

FAIL_LOG = os.path.join(LOG_DIR, "failures.log")


# ==========================
# ✅ Optional parquet support
# ==========================
HAVE_ARROW = False
if OUTPUT_MODE == "parquet":
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        HAVE_ARROW = True
    except Exception:
        HAVE_ARROW = False
        OUTPUT_MODE = "csv_gz"
        print("⚠️ pyarrow not available → fallback to OUTPUT_MODE='csv_gz' (still ML-friendly).")


# ==========================
# ✅ Utils
# ==========================
def safe_filename(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def get_free_gb(path: str) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def ensure_disk_space_or_stop(tag: str = ""):
    free_gb = get_free_gb(OUT_DIR)
    if free_gb < MIN_FREE_GB_TO_CONTINUE:
        msg = f"❌ Disk free too low ({free_gb:.2f} GB < {MIN_FREE_GB_TO_CONTINUE} GB). Stop. {tag}"
        print(msg)
        try:
            with open(FAIL_LOG, "a", encoding="utf-8") as f:
                f.write(f"[DISK_STOP] {msg}\n")
        except Exception:
            pass
        raise SystemExit(msg)


def log_failure(tag: str, msg: str):
    line = f"[{tag}] {msg}"
    print(line)
    try:
        with open(FAIL_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # disk full / permission issues should not crash the run
        pass


def progress_path(space: str) -> str:
    return os.path.join(PROGRESS_DIR, f"progress_{safe_filename(space)}.json")


def skip_path(space: str) -> str:
    return os.path.join(SKIP_DIR, f"skip_{safe_filename(space)}.json")


def load_set(path: str) -> set:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_set(path: str, s: set):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(list(s)), f, ensure_ascii=False, indent=2)


def append_skip(space: str, record: Dict[str, Any]):
    p = skip_path(space)
    arr = []
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                arr = json.load(f)
            if not isinstance(arr, list):
                arr = []
        except Exception:
            arr = []
    arr.append(record)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)


def space_outdir(space: str) -> str:
    # Partition folder per space (ML-friendly)
    # e.g., /snapshot_votes_441/spaces/space=uniswap.eth/
    return os.path.join(SPACE_DIR, f"space={space}")


# ==========================
# ✅ GraphQL client (robust)
# ==========================
def post_gql(query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    last_err = None
    last_status = None
    last_text = None

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(SNAPSHOT_URL, json=payload, timeout=HTTP_TIMEOUT)
            last_status = r.status_code
            last_text = r.text

            if r.status_code == 400:
                raise RuntimeError(f"HTTP 400 Bad Request: {r.text[:800]}")

            if r.status_code in (429, 500, 502, 503, 504, 524):
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")

            r.raise_for_status()
            data = r.json()

            if "errors" in data:
                raise RuntimeError(f"GQL errors: {str(data['errors'])[:800]}")

            return data["data"]

        except Exception as e:
            last_err = e
            if isinstance(e, RuntimeError) and "HTTP 400" in str(e):
                print("\n========== SNAPSHOT 400 BAD REQUEST ==========")
                print("Response (first 800 chars):", (last_text or "")[:800])
                print("Query (first 800 chars):", query[:800])
                print("=============================================\n")
                raise

            wait = (BACKOFF_BASE ** attempt) + 0.2 * attempt
            print(f"⚠️ request failure（attempt {attempt+1}/{MAX_RETRIES}）：{e} | backoff {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(
        f"❌ GraphQL failed after {MAX_RETRIES} retries. "
        f"last_status={last_status}, last_error={repr(last_err)}"
    )


# ==========================
# ✅ Snapshot queries
# ==========================
def load_selected_space_ids(spaces_file: str) -> List[str]:
    with open(spaces_file, "r", encoding="utf-8") as f:
        obj = json.load(f)
    ids = obj.get("selected_space_ids", [])
    if not isinstance(ids, list):
        raise ValueError("selected_space_ids is not a list in SPACES_FILE")

    out, seen = [], set()
    for s in ids:
        if isinstance(s, str) and s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def fetch_space_followers(space: str) -> Optional[int]:
    q = f"""
    {{
      space(id: "{space}") {{
        followersCount
      }}
    }}
    """
    data = post_gql(q)
    sp = data.get("space")
    if not sp:
        return None
    fc = sp.get("followersCount")
    try:
        return int(fc) if fc is not None else None
    except Exception:
        return None


def fetch_all_proposals(space: str) -> List[Dict[str, Any]]:
    all_props = []
    skip = 0

    print(f"\n🚀 Fetching proposals for space={space} ...")
    while True:
        q = f"""
        {{
          proposals(first: {PROPOSAL_BATCH}, skip: {skip},
            where: {{ space_in: ["{space}"] }},
            orderBy: "created", orderDirection: desc
          ) {{
            id
            title
            body
            created
          }}
        }}
        """
        data = post_gql(q)
        batch = data.get("proposals", [])
        print(f"📦 {space} proposals: {skip + 1} ~ {skip + len(batch)}")

        if not batch:
            break

        all_props.extend(batch)
        if len(batch) < PROPOSAL_BATCH:
            break

        skip += PROPOSAL_BATCH
        time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"✅ {space} proposals total: {len(all_props)}")
    return all_props


def _is_retryable_heavy_error(e: Exception) -> bool:
    msg = str(e).lower()
    keywords = [
        "524", "504", "502", "503", "500", "server error", "gateway",
        "read timed out", "timeout", "timed out", "connectionpool", "connection aborted"
    ]
    return any(k in msg for k in keywords)


def fetch_all_votes_for_proposal(proposal_id: str, space: str) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Cursor pagination via created_lt (desc).
    Returns (votes, skipped_flag)
    """
    page_size = VOTES_PAGE_SIZE

    for round_idx in range(1, MAX_PROPOSAL_FAILURE_ROUNDS + 1):
        all_votes: List[Dict[str, Any]] = []
        created_lt = int(time.time()) + 10
        seen = set()

        print(f"   ↪️ votes round {round_idx}/{MAX_PROPOSAL_FAILURE_ROUNDS} | page_size={page_size}")

        try:
            while True:
                q = f"""
                {{
                  votes(first: {page_size},
                    where: {{ proposal: "{proposal_id}", created_lt: {created_lt} }},
                    orderBy: "created", orderDirection: desc
                  ) {{
                    voter
                    choice
                    vp
                    created
                  }}
                }}
                """
                data = post_gql(q)
                batch = data.get("votes", [])

                if not batch:
                    break

                for v in batch:
                    key = (v.get("voter"), v.get("created"))
                    if key in seen:
                        continue
                    seen.add(key)
                    all_votes.append(v)

                last_created = batch[-1]["created"]
                if last_created >= created_lt:
                    break
                created_lt = last_created

                if len(batch) < page_size:
                    break

                time.sleep(SLEEP_BETWEEN_CALLS)

            return all_votes, False

        except Exception as e:
            if _is_retryable_heavy_error(e):
                smaller = [x for x in PAGE_SIZE_LADDER if x < page_size]
                if smaller:
                    page_size = max(MIN_VOTES_PAGE_SIZE, smaller[0])
                else:
                    page_size = max(MIN_VOTES_PAGE_SIZE, page_size // 2)

                print(f"⚠️ votes fetch unstable: {e}")
                print(f"   ✅ downshift page_size -> {page_size}, wait then retry ...")
                time.sleep(6.0 + 2.0 * round_idx)
                continue

            raise

    append_skip(space, {
        "proposal_id": proposal_id,
        "reason": f"Failed after {MAX_PROPOSAL_FAILURE_ROUNDS} rounds with page_size reductions (likely heavy 524/timeout)."
    })
    print(f"🚫 Skipped proposal (logged): {proposal_id}")
    return [], True


# ==========================
# ✅ Feature helpers
# ==========================
def compute_majority_choice(votes: List[Dict[str, Any]]) -> Optional[int]:
    counter: Dict[int, int] = {}
    for v in votes:
        c = v.get("choice")
        if isinstance(c, int):
            counter[c] = counter.get(c, 0) + 1
        elif isinstance(c, dict):
            for k, val in c.items():
                try:
                    kk = int(k)
                except Exception:
                    continue
                if val and val > 0:
                    counter[kk] = counter.get(kk, 0) + 1
    if not counter:
        return None
    return max(counter.items(), key=lambda x: x[1])[0]


def normalize_choice_for_output(choice_obj: Any) -> Any:
    if isinstance(choice_obj, int):
        return choice_obj
    if isinstance(choice_obj, dict):
        return json.dumps(choice_obj, ensure_ascii=False)
    return choice_obj


# ==========================
# ✅ Writers (Parquet parts or CSV.GZ parts)
# ==========================
def write_rows(space: str, proposal_id: str, rows: List[Dict[str, Any]]):
    if not rows:
        return

    ensure_disk_space_or_stop(tag=f"before write space={space}")

    out_dir = space_outdir(space)
    os.makedirs(out_dir, exist_ok=True)

    if OUTPUT_MODE == "parquet":
        # Many small parquet parts -> safe + ML-friendly + easy to parallel read
        # filename includes proposal_id to avoid collisions
        fname = f"proposal={safe_filename(proposal_id)}_{int(time.time()*1000)}.parquet"
        out_path = os.path.join(out_dir, fname)

        df = pd.DataFrame(rows)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, out_path, compression="snappy")

        print(f"   💾 wrote parquet part: {out_path} | rows={len(df):,}")
        return

    # Fallback: gzipped csv part
    fname = f"proposal={safe_filename(proposal_id)}_{int(time.time()*1000)}.csv.gz"
    out_path = os.path.join(out_dir, fname)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, compression="gzip", encoding="utf-8")
    print(f"   💾 wrote csv.gz part: {out_path} | rows={len(df):,}")


# ==========================
# ✅ MAIN
# ==========================
def main():
    ensure_disk_space_or_stop("start")

    spaces = load_selected_space_ids(SPACES_FILE)
    print(f"✅ loaded spaces: {len(spaces)}")
    spaces = [s for s in spaces if s not in SKIP_SPACES]
    print(f"⏭️  skipped spaces: {sorted(list(SKIP_SPACES))} | remaining={len(spaces)}")

    if MAX_SPACES is not None:
        spaces = spaces[:MAX_SPACES]
        print(f"🧪 MAX_SPACES={MAX_SPACES} -> running {len(spaces)} spaces")

    # Optional sort by followersCount
    space_items: List[Tuple[str, Optional[int]]] = [(s, None) for s in spaces]
    if SORT_BY_FOLLOWERS:
        print("🔎 Fetch followersCount and sort ...")
        tmp = []
        for i, s in enumerate(spaces, 1):
            try:
                fc = fetch_space_followers(s)
            except Exception as e:
                fc = None
                log_failure("FOLLOWERS_FAIL", f"{s} | {repr(e)}")
            tmp.append((s, fc))
            if i % 25 == 0:
                print(f"   ... followers fetched {i}/{len(spaces)}")
            time.sleep(0.2)
        tmp.sort(key=lambda x: (x[1] is None, -(x[1] or 0)))
        space_items = tmp
        print("✅ top-5 by followers:", space_items[:5])

    for idx, (space, followers) in enumerate(space_items, 1):
        ensure_disk_space_or_stop(tag=f"space={space}")

        print("\n" + "=" * 90)
        print(f"🏷️  Space [{idx}/{len(space_items)}]: {space} | followersCount={followers}")
        print("=" * 90)

        prog_p = progress_path(space)
        completed = load_set(prog_p)
        print(f"📌 completed proposals: {len(completed)} | progress={prog_p}")
        print(f"🧾 skip log: {skip_path(space)}")
        print(f"📁 output dir: {space_outdir(space)}")
        print(f"💽 free GB on D: {get_free_gb(OUT_DIR):.2f}")

        # fetch proposals
        try:
            proposals = fetch_all_proposals(space)
        except Exception as e:
            log_failure("SPACE_FAIL", f"{space} | {repr(e)}")
            continue

        if MAX_PROPOSALS_PER_SPACE is not None:
            proposals = proposals[:MAX_PROPOSALS_PER_SPACE]
            print(f"🧪 MAX_PROPOSALS_PER_SPACE={MAX_PROPOSALS_PER_SPACE} -> running {len(proposals)} proposals")

        # loop proposals
        for i, p in enumerate(proposals, 1):
            pid = p["id"]
            if pid in completed:
                continue

            title_short = (p.get("title") or "")[:80]
            print(f"\n🔍 [{space}] Proposal {i}/{len(proposals)}: {title_short}")

            try:
                votes, skipped = fetch_all_votes_for_proposal(pid, space=space)

                # mark progress early (even skip) to avoid infinite loops
                completed.add(pid)
                save_set(prog_p, completed)

                if skipped or not votes:
                    continue

                total_vp = sum(v.get("vp", 0) for v in votes)
                if total_vp == 0:
                    continue

                majority = compute_majority_choice(votes)
                if majority is None:
                    continue

                # Build rows
                created_time = datetime.utcfromtimestamp(p["created"]).isoformat()
                body_clean = (p.get("body") or "").replace("\n", " ").replace("\r", " ")
                # (optional) cap body length to reduce disk usage
                body_clean = body_clean[:800]

                rows = []
                for v in votes:
                    vp = v.get("vp", 0) or 0
                    vp_ratio = (vp / total_vp) if total_vp else 0.0
                    c = v.get("choice")
                    choice_used = normalize_choice_for_output(c)

                    aligned = False
                    if isinstance(c, int):
                        aligned = (c == majority)
                    elif isinstance(c, dict):
                        aligned = (str(majority) in c and c.get(str(majority), 0) > 0)

                    rows.append({
                        "Space": space,
                        "FollowersCount": followers,
                        "Proposal ID": pid,
                        "Proposal Title": p.get("title", ""),
                        "Proposal Body": body_clean,
                        "Created Time": created_time,
                        "Voter": v.get("voter"),
                        "Choice": choice_used,
                        "Voting Power": vp,
                        "VP Ratio (%)": round(vp_ratio * 100, 6),
                        "Is Whale": vp_ratio > 0.05,
                        "Aligned With Majority": aligned,
                        "Vote Timestamp": datetime.utcfromtimestamp(v["created"]).isoformat()
                    })

                # Write (Parquet parts or CSV.GZ parts)
                write_rows(space=space, proposal_id=pid, rows=rows)

                time.sleep(SLEEP_BETWEEN_CALLS)

            except SystemExit:
                raise
            except Exception as e:
                log_failure("PROPOSAL_FAIL", f"{space} | {pid} | {repr(e)}")
                continue

    print("\n🎉 DONE (or best-effort done).")
    print(f"📁 OUT_DIR: {OUT_DIR}")
    print(f"📁 outputs under: {SPACE_DIR}")
    print(f"📌 progress: {PROGRESS_DIR}")
    print(f"🧾 skips: {SKIP_DIR}")
    print(f"📄 failures log: {FAIL_LOG}")
    print(f"✅ OUTPUT_MODE used: {OUTPUT_MODE} (pyarrow={HAVE_ARROW})")


if __name__ == "__main__":
    main()
