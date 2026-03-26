"""
DAO eligible base reconstruction (Strict, proposal-time, Snapshot-consistent)

For each Snapshot space:
  - Fetch recent proposals: snapshot block + strategies
  - Fetch votes for each proposal (Layer 3)
  - Build candidate pools at snapshot:
      * ERC20 holders at snapshot (Layer 1 candidates)
      * Delegate addresses (if delegation-related strategies detected)
      * Always include voters (for sanity coverage)
  - Compute voting power at snapshot via Snapshot Score API
  - Define 3 layers:
      Layer 1: raw token holders (balance >= tau_balance_token)
      Layer 2: effective voting power holders (vp >= tau_vp_token)  [main eligible definition]
      Layer 3: active participants (voters)
  - Output parquet files + summary CSV

Assumptions:
  - Default: Ethereum mainnet (network="1"). Works for .eth spaces you listed.
  - Candidate enumeration is strict for ERC20 holders (via Transfer log replay) and delegates (via DelegateChanged logs).
  - For staking/LP/multi-asset strategies: we still compute vp for addresses in candidate pool,
    but we warn that holder enumeration may be incomplete (needs strategy-specific pool expansion).

Usage:
  python dao_eligible_reconstruct_full.py --spaces aavedao.eth ens.eth uniswapgovernance.eth --max_proposals 3 --out out_dir

Env:
  RPC_URL (required)
  ETHERSCAN_API_KEY (optional, for auto token_start_block)
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set

import requests
import pandas as pd
from tqdm import tqdm
from web3 import Web3


# =========================
# Constants / Endpoints
# =========================

SNAPSHOT_GRAPHQL = "https://hub.snapshot.org/graphql"
SNAPSHOT_SCORE_API = "https://score.snapshot.org/api/scores"
ETHERSCAN_V2_API = "https://api.etherscan.io/v2/api"

TRANSFER_TOPIC = Web3.keccak(text="Transfer(address,address,uint256)").hex()

# Compound-style delegation events (common in ERC20Votes / UNI / ENS / AAVE-like)
DELEGATE_CHANGED_TOPIC = Web3.keccak(text="DelegateChanged(address,address,address)").hex()
DELEGATE_VOTES_CHANGED_TOPIC = Web3.keccak(text="DelegateVotesChanged(address,uint256,uint256)").hex()

ETH_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")


# =========================
# Config
# =========================

@dataclass
class RunConfig:
    network: str = "1"                  # snapshot network
    chainid: int = 1                    # etherscan chainid (ETH mainnet)
    max_proposals: int = 3
    proposal_batch_votes: int = 1000    # votes per page fetch
    score_batch_size: int = 800
    score_sleep_s: float = 0.15
    logs_chunk_size: int = 20_000
    logs_sleep_s: float = 0.0
    local_score_only: bool = False       # skip Snapshot Score API; use RPC-based local scoring
    # Dust thresholds:
    tau_balance_token: float = 0.0      # for Layer 1 holders
    tau_vp_token: float = 0.0           # for Layer 2 vp-based eligible
    # optional USD-based threshold support (you can set price manually per space)
    tau_usd: Optional[float] = 100.0
    token_price_usd: Optional[float] = None  # if provided, tau_usd -> tau_token conversion for both thresholds
    # Start block discovery:
    use_etherscan_start_block: bool = True
    # Caching:
    cache_dir: str = "cache"
    out_dir: str = "out"


# =========================
# Helpers
# =========================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def looks_like_address(x: Any) -> bool:
    return isinstance(x, str) and ETH_ADDRESS_RE.match(x) is not None

def find_addresses_in_obj(obj: Any) -> List[str]:
    found: List[str] = []
    if isinstance(obj, dict):
        for v in obj.values():
            found.extend(find_addresses_in_obj(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(find_addresses_in_obj(v))
    elif isinstance(obj, str):
        if looks_like_address(obj):
            found.append(obj.lower())
    return found

def chunk_ranges(start: int, end: int, step: int) -> Iterable[Tuple[int, int]]:
    s = start
    while s <= end:
        e = min(s + step - 1, end)
        yield s, e
        s = e + 1

def read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def normalize_addr(a: str) -> str:
    return a.lower()

def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


# =========================
# Snapshot GraphQL
# =========================

def snapshot_graphql(query: str, variables: dict) -> dict:
    """GraphQL call with retry on 5xx/429 (gateway timeout, rate limits, etc.)."""
    max_retries = 5
    backoff_s = 3.0
    backoff_max_s = 60.0
    timeout_s = 90

    last_err: Optional[BaseException] = None
    last_status: Optional[int] = None

    for attempt in range(max_retries):
        try:
            r = requests.post(SNAPSHOT_GRAPHQL, json={"query": query, "variables": variables}, timeout=timeout_s)
            last_status = r.status_code
            retryable = r.status_code == 429 or (500 <= r.status_code < 600)
            if retryable:
                raise RuntimeError(f"Retryable GraphQL HTTP {r.status_code}")
            if r.status_code >= 400:
                raise RuntimeError(f"GraphQL HTTP {r.status_code}: {(r.text or '')[:500]}")
            data = r.json()
            if "errors" in data:
                raise RuntimeError(f"Snapshot GraphQL errors: {data['errors']}")
            return data["data"]
        except requests.exceptions.Timeout as e:
            last_err = e
            retryable = True
        except RuntimeError as e:
            last_err = e
            retryable = "Retryable" in str(e)
        except Exception as e:
            last_err = e
            retryable = True  # retry on connection errors etc.

        if not retryable or attempt >= max_retries - 1:
            raise last_err
        sleep_s = min(backoff_s * (2 ** attempt), backoff_max_s)
        time.sleep(sleep_s)

def fetch_space_network(space: str) -> Optional[str]:
    q = """
    query Space($id: String!) {
      space(id: $id) { id network name }
    }
    """
    d = snapshot_graphql(q, {"id": space})
    sp = d.get("space")
    if not sp:
        return None
    return sp.get("network")

def fetch_recent_proposals(space: str, first: int = 3) -> List[dict]:
    q = """
    query Proposals($space: String!, $first: Int!) {
      proposals(
        first: $first
        where: { space_in: [$space] }
        orderBy: "created"
        orderDirection: desc
      ) {
        id
        title
        snapshot
        created
        strategies { name params }
        space { id }
      }
    }
    """
    d = snapshot_graphql(q, {"space": space, "first": first})
    return d["proposals"]

def fetch_votes(proposal_id: str, first: int = 1000, skip: int = 0) -> List[dict]:
    # votes pagination
    q = """
    query Votes($proposal: String!, $first: Int!, $skip: Int!) {
      votes(
        first: $first
        skip: $skip
        where: { proposal: $proposal }
        orderBy: "created"
        orderDirection: asc
      ) {
        id
        voter
        created
        choice
        vp
      }
    }
    """
    d = snapshot_graphql(q, {"proposal": proposal_id, "first": first, "skip": skip})
    return d["votes"]

def fetch_all_votes(proposal_id: str, page_size: int = 1000, max_pages: int = 200) -> pd.DataFrame:
    rows: List[dict] = []
    skip = 0
    for _ in range(max_pages):
        batch = fetch_votes(proposal_id, first=page_size, skip=skip)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        skip += page_size
    if not rows:
        return pd.DataFrame(columns=["voter", "choice", "created", "vp"])
    df = pd.DataFrame(rows)
    df["voter"] = df["voter"].str.lower()
    df["created"] = df["created"].astype(int)
    return df[["voter", "choice", "created", "vp"]]


# =========================
# Strategy parsing: infer token contract + flags
# =========================

def infer_token_contract_from_strategies(strategies: List[dict]) -> Tuple[Optional[str], dict]:
    """
    Best effort:
      - prefer params keys that look like token address
      - if multiple, pick the one associated with erc20 balance/votes strategy
    """
    debug = {"strategy_names": [s.get("name") for s in strategies], "found_addresses": []}
    preferred_keys = {"address", "token", "tokenAddress", "contractAddress", "token_address"}

    candidates: List[Tuple[int, str]] = []

    for s in strategies:
        name = (s.get("name") or "").lower()
        params = s.get("params") or {}
        found = find_addresses_in_obj(params)
        debug["found_addresses"].append({"name": s.get("name"), "addresses": found, "params": params})

        # score weight by strategy relevance
        weight = 0
        if "erc20" in name:
            weight += 5
        if "balance" in name:
            weight += 5
        if "votes" in name:
            weight += 4
        if "delegation" in name:
            weight += 2

        # prefer explicit keys
        if isinstance(params, dict):
            for k, v in params.items():
                if k in preferred_keys and looks_like_address(v):
                    candidates.append((weight + 10, v.lower()))

        # fallback: any address found
        for addr in found:
            candidates.append((weight, addr))

    if not candidates:
        return None, debug

    # pick best score, then most frequent
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_weight = candidates[0][0]
    top_addrs = [a for w, a in candidates if w == top_weight]
    # frequency tie-break
    from collections import Counter
    c = Counter(top_addrs)
    best = c.most_common(1)[0][0]
    return best, debug

def detect_strategy_flags(strategies: List[dict]) -> dict:
    names = [(s.get("name") or "").lower() for s in strategies]
    flags = {
        "has_erc20_balance": any("erc20" in n and "balance" in n for n in names) or any("erc20-balance-of" in n for n in names),
        "has_erc20_votes": any("votes" in n for n in names),
        "has_delegation": any("delegation" in n for n in names),
        "has_lp_or_staking": any(k in n for n in names for k in ["lp", "pool", "staking", "staked", "farm", "vault"]),
        "has_multi_asset": len(names) > 1,
    }
    return flags


# =========================
# Etherscan: token_start_block (optional)
# =========================

def etherscan_latest_block(chainid: int) -> int:
    api_key = os.environ.get("ETHERSCAN_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ETHERSCAN_API_KEY")
    params = {"chainid": chainid, "module": "proxy", "action": "eth_blockNumber", "apikey": api_key}
    r = requests.get(ETHERSCAN_V2_API, params=params, timeout=30)
    r.raise_for_status()
    return int(r.json()["result"], 16)

def etherscan_has_logs(chainid: int, address: str, topic0: str, from_block: int, to_block: int) -> bool:
    api_key = os.environ.get("ETHERSCAN_API_KEY")
    params = {
        "chainid": chainid,
        "module": "logs",
        "action": "getLogs",
        "fromBlock": from_block,
        "toBlock": to_block,
        "address": address.lower(),
        "topic0": topic0,
        "apikey": api_key,
    }
    r = requests.get(ETHERSCAN_V2_API, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    result = data.get("result")
    if isinstance(result, str):
        msg = result.lower()
        if "no records" in msg or "not found" in msg:
            return False
        if "more than" in msg or "rate limit" in msg:
            return True
        return False
    if isinstance(result, list):
        return len(result) > 0
    return False

def etherscan_find_first_log_block(chainid: int, token_contract: str, topic0: str, step: int = 500_000, sleep_s: float = 0.2) -> Optional[int]:
    """
    Coarse scan then binary search for earliest block that has logs matching topic0 on token_contract.
    Works for Transfer topic to approximate token_start_block.
    """
    api_key = os.environ.get("ETHERSCAN_API_KEY")
    if not api_key:
        return None

    latest = etherscan_latest_block(chainid)
    time.sleep(sleep_s)

    lo = 0
    hi = min(step, latest)
    found_interval = False

    while lo <= latest:
        if etherscan_has_logs(chainid, token_contract, topic0, lo, hi):
            found_interval = True
            break
        lo = hi + 1
        hi = min(hi + step, latest)
        time.sleep(sleep_s)

    if not found_interval:
        return None

    left, right = lo, hi
    while left < right:
        mid = (left + right) // 2
        if etherscan_has_logs(chainid, token_contract, topic0, left, mid):
            right = mid
        else:
            left = mid + 1
        time.sleep(sleep_s)

    return left


# =========================
# On-chain reads: decimals + log replay
# =========================

ERC20_DECIMALS_ABI = [{"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"}]

def get_erc20_decimals(w3: Web3, token_contract: str, default: int = 18) -> int:
    try:
        c = w3.eth.contract(address=Web3.to_checksum_address(token_contract), abi=ERC20_DECIMALS_ABI)
        return int(c.functions.decimals().call())
    except Exception:
        return default

def replay_erc20_balances_from_transfers(
    w3: Web3,
    token_contract: str,
    start_block: int,
    end_block: int,
    chunk_size: int,
    sleep_s: float,
    cache_path: Optional[str] = None,
) -> Dict[str, int]:
    """
    Reconstruct balances at end_block via Transfer logs replay.
    """
    if cache_path and os.path.exists(cache_path):
        cached = read_json(cache_path)
        if cached and cached.get("end_block") == end_block and cached.get("start_block") == start_block and cached.get("token_contract") == token_contract.lower():
            return {k: int(v) for k, v in cached["balances_raw"].items()}

    token = Web3.to_checksum_address(token_contract)
    balances: Dict[str, int] = {}

    base_filter = {"address": token, "topics": [TRANSFER_TOPIC]}

    for b0, b1 in tqdm(list(chunk_ranges(start_block, end_block, chunk_size)), desc=f"Transfer replay {token_contract[:8]}.."):
        flt = dict(base_filter)
        flt["fromBlock"] = b0
        flt["toBlock"] = b1
        logs = w3.eth.get_logs(flt)

        for lg in logs:
            topics = lg["topics"]
            frm = "0x" + topics[1].hex()[-40:]
            to  = "0x" + topics[2].hex()[-40:]
            value_raw = int(lg["data"].hex(), 16)

            if frm != "0x0000000000000000000000000000000000000000":
                balances[frm] = balances.get(frm, 0) - value_raw
            if to != "0x0000000000000000000000000000000000000000":
                balances[to] = balances.get(to, 0) + value_raw

        if sleep_s > 0:
            time.sleep(sleep_s)

    cleaned = {a.lower(): b for a, b in balances.items() if b > 0}
    neg_cnt = sum(1 for b in balances.values() if b < 0)
    if neg_cnt > 0:
        print(f"[WARN] {neg_cnt} negative balances found. start_block may be too late for full reconstruction.")

    if cache_path:
        write_json(cache_path, {
            "token_contract": token_contract.lower(),
            "start_block": start_block,
            "end_block": end_block,
            "balances_raw": {k: int(v) for k, v in cleaned.items()}
        })
    return cleaned

def replay_delegate_addresses(
    w3: Web3,
    token_contract: str,
    start_block: int,
    end_block: int,
    chunk_size: int,
    sleep_s: float,
    cache_path: Optional[str] = None,
) -> Set[str]:
    """
    Collect delegate addresses using DelegateChanged logs (Compound-style).
    This helps include delegates who may have vp>0 without holding tokens directly.
    """
    if cache_path and os.path.exists(cache_path):
        cached = read_json(cache_path)
        if cached and cached.get("end_block") == end_block and cached.get("start_block") == start_block and cached.get("token_contract") == token_contract.lower():
            return set(cached["delegates"])

    token = Web3.to_checksum_address(token_contract)
    delegates: Set[str] = set()

    base_filter = {"address": token, "topics": [DELEGATE_CHANGED_TOPIC]}

    for b0, b1 in tqdm(list(chunk_ranges(start_block, end_block, chunk_size)), desc=f"Delegate replay {token_contract[:8]}.."):
        flt = dict(base_filter)
        flt["fromBlock"] = b0
        flt["toBlock"] = b1
        try:
            logs = w3.eth.get_logs(flt)
        except Exception:
            logs = []

        for lg in logs:
            topics = lg["topics"]
            # DelegateChanged(indexed delegator, indexed fromDelegate, indexed toDelegate)
            delegator = "0x" + topics[1].hex()[-40:]
            from_d    = "0x" + topics[2].hex()[-40:]
            to_d      = "0x" + topics[3].hex()[-40:]
            for a in [delegator, from_d, to_d]:
                if a.lower() != "0x0000000000000000000000000000000000000000":
                    delegates.add(a.lower())

        if sleep_s > 0:
            time.sleep(sleep_s)

    if cache_path:
        write_json(cache_path, {
            "token_contract": token_contract.lower(),
            "start_block": start_block,
            "end_block": end_block,
            "delegates": sorted(list(delegates))
        })
    return delegates


# =========================
# Snapshot Score API
# =========================

def compute_scores_snapshot(space: str, network: str, snapshot: int, strategies: list, addresses: List[str]) -> Dict[str, float]:
    payload = {
        "space": space,
        "network": network,
        "snapshot": snapshot,
        "strategies": strategies,
        "addresses": addresses,
    }

    # The Snapshot Score API occasionally returns transient 5xx/429 errors.
    # We retry those so long batch jobs don't fail mid-run.
    max_retries = 6
    backoff_s = 1.5
    backoff_max_s = 30.0

    last_err: Optional[BaseException] = None
    last_status: Optional[int] = None
    last_text: str = ""

    for attempt in range(max_retries):
        try:
            r = requests.post(SNAPSHOT_SCORE_API, json=payload, timeout=180)
            last_status = r.status_code
            if r.status_code == 429 or (500 <= r.status_code < 600):
                last_text = (r.text or "")[:2000]
                raise RuntimeError(f"Retryable Score API HTTP {r.status_code}: {last_text}")
            if r.status_code >= 400:
                last_text = (r.text or "")[:2000]
                raise RuntimeError(f"Score API HTTP {r.status_code}: {last_text}")

            data = r.json()
            if "result" not in data or "scores" not in data["result"]:
                raise RuntimeError(f"Unexpected score response: {data}")

            scores_by_strategy = data["result"]["scores"]
            total: Dict[str, float] = {a.lower(): 0.0 for a in addresses}
            for strat_scores in scores_by_strategy:
                for a, s in strat_scores.items():
                    total[a.lower()] = total.get(a.lower(), 0.0) + float(s)
            return total
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                sleep_s = min(backoff_s * (2 ** attempt), backoff_max_s)
                time.sleep(sleep_s)
                continue
            break

    raise RuntimeError(
        f"Score API request failed after {max_retries} attempts "
        f"(last_status={last_status}, last_text={last_text!r}): {last_err}"
    )

def batch_compute_scores(space: str, network: str, snapshot: int, strategies: list, addresses: List[str], batch_size: int, sleep_s: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i in tqdm(range(0, len(addresses), batch_size), desc="Score API batches"):
        batch = addresses[i:i+batch_size]
        scores = compute_scores_snapshot(space, network, snapshot, strategies, batch)
        out.update(scores)
        if sleep_s > 0:
            time.sleep(sleep_s)
    return out


# =========================
# Local fallback scoring
# =========================

ERC20_BALANCEOF_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function",
    }
]


def _as_checksum(w3: Web3, addr: str) -> str:
    try:
        return w3.to_checksum_address(addr)
    except Exception:
        return addr


def _call_contract_method_at_block(
    w3: Web3,
    contract_address: str,
    method_abi: dict,
    args: list,
    block: int,
) -> int:
    c = w3.eth.contract(address=_as_checksum(w3, contract_address), abi=[method_abi])
    fn = getattr(c.functions, method_abi["name"])(*args)
    return int(fn.call(block_identifier=block))


def local_compute_scores(
    w3: Web3,
    snapshot_block: int,
    strategies: list,
    addresses: List[str],
) -> Tuple[Dict[str, float], List[str]]:
    """
    Best-effort local score computation for a subset of strategy types using RPC_URL.

    Supported (best-effort):
      - contract-call (methodABI provided)
      - erc20-balance-of (falls back to balanceOf)

    Returns: (scores, warnings)
    """
    warnings: List[str] = []
    totals: Dict[str, float] = {a.lower(): 0.0 for a in addresses}

    # Safety cap: local scoring is RPC-call heavy and can be very slow for huge address sets.
    local_max = safe_int(os.environ.get("LOCAL_SCORE_MAX_ADDRESSES"), default=5000)
    if local_max > 0 and len(addresses) > local_max:
        warnings.append(
            f"local scoring skipped: {len(addresses)} addresses exceeds LOCAL_SCORE_MAX_ADDRESSES={local_max}"
        )
        return totals, warnings

    if snapshot_block <= 0:
        warnings.append("snapshot_block <= 0; local scores forced to 0")
        return totals, warnings

    # Prepare strategy executors
    execs: List[Tuple[str, dict]] = []
    for s in (strategies or []):
        if not isinstance(s, dict):
            continue
        name = str(s.get("name") or "").strip()
        params = s.get("params") if isinstance(s.get("params"), dict) else {}
        execs.append((name, params))

    if not execs:
        warnings.append("no strategies found; local scores forced to 0")
        return totals, warnings

    # For each address, sum strategy contributions.
    for a in tqdm(addresses, desc="Local scoring (RPC)", leave=False):
        a_l = a.lower()
        addr_cs = _as_checksum(w3, a)
        score = 0.0

        for (name, params) in execs:
            try:
                # contract-call: methodABI + address + args with %{address}
                if "methodABI" in params and "address" in params:
                    method_abi = params.get("methodABI")
                    caddr = str(params.get("address"))
                    if not isinstance(method_abi, dict) or not caddr:
                        continue
                    decimals = int(params.get("decimals") or 0)
                    raw_args = params.get("args") or ["%{address}"]
                    if not isinstance(raw_args, list):
                        raw_args = ["%{address}"]
                    call_args = []
                    for x in raw_args:
                        if isinstance(x, str) and x == "%{address}":
                            call_args.append(addr_cs)
                        else:
                            call_args.append(x)
                    val = _call_contract_method_at_block(w3, caddr, method_abi, call_args, snapshot_block)
                    score += float(val) / float(10 ** max(decimals, 0))
                    continue

                # erc20-balance-of: standard ERC20 balanceOf
                if name == "erc20-balance-of" and "address" in params:
                    token = str(params.get("address"))
                    decimals = int(params.get("decimals") or 18)
                    c = w3.eth.contract(address=_as_checksum(w3, token), abi=ERC20_BALANCEOF_ABI)
                    val = int(c.functions.balanceOf(addr_cs).call(block_identifier=snapshot_block))
                    score += float(val) / float(10 ** max(decimals, 0))
                    continue

            except Exception:
                # best-effort: treat as zero for this strategy/address
                continue

        totals[a_l] = score

    return totals, warnings


# =========================
# Metrics
# =========================

def topk_share(values: List[float], k: int) -> float:
    if not values:
        return 0.0
    vals = sorted(values, reverse=True)
    total = sum(vals)
    if total <= 0:
        return 0.0
    return sum(vals[:min(k, len(vals))]) / total

def apply_tau_from_usd(cfg: RunConfig) -> Tuple[float, float]:
    """
    Convert tau_usd to token if token_price_usd is provided; else keep given tau_token thresholds.
    We use the same tau for balance and vp if you want. You can adjust separately later.
    """
    if cfg.tau_usd is not None and cfg.token_price_usd and cfg.token_price_usd > 0:
        tau_token = float(cfg.tau_usd) / float(cfg.token_price_usd)
        return tau_token, tau_token
    return float(cfg.tau_balance_token), float(cfg.tau_vp_token)


# =========================
# Main pipeline per proposal
# =========================

def run_one_proposal(
    w3: Web3,
    cfg: RunConfig,
    space: str,
    proposal: dict,
    token_contract: Optional[str],
    token_start_block: Optional[int],
    token_decimals: int,
    flags: dict,
):
    proposal_id = proposal["id"]
    snapshot_block = safe_int(proposal["snapshot"])
    strategies = proposal["strategies"]
    title = proposal.get("title", "")

    ensure_dir(cfg.cache_dir)
    ensure_dir(cfg.out_dir)

    print(f"\n=== Proposal {proposal_id} | block={snapshot_block} | {title[:80]} ===")

    # --- Fetch votes (Layer 3)
    votes_df = fetch_all_votes(proposal_id, page_size=cfg.proposal_batch_votes)
    voters = sorted(set(votes_df["voter"].tolist())) if not votes_df.empty else []
    print(f"[INFO] votes: n_voters={len(voters)}")

    # Candidate pool always includes voters
    candidate_set: Set[str] = set(voters)

    # --- Enumerate holders (Layer 1 candidates) if token is known
    balances_raw: Dict[str, int] = {}
    holders: List[str] = []
    delegates: Set[str] = set()

    if token_contract and token_start_block is not None and snapshot_block > 0:
        bal_cache = os.path.join(cfg.cache_dir, f"balances_{space}_{proposal_id}.json")
        balances_raw = replay_erc20_balances_from_transfers(
            w3=w3,
            token_contract=token_contract,
            start_block=token_start_block,
            end_block=snapshot_block,
            chunk_size=cfg.logs_chunk_size,
            sleep_s=cfg.logs_sleep_s,
            cache_path=bal_cache,
        )
        holders = list(balances_raw.keys())
        candidate_set.update(holders)
        print(f"[INFO] Layer1 holder candidates: {len(holders)}")

        # Delegation: add delegates if strategy suggests it (or votes-related)
        if flags.get("has_delegation") or flags.get("has_erc20_votes"):
            del_cache = os.path.join(cfg.cache_dir, f"delegates_{space}_{proposal_id}.json")
            delegates = replay_delegate_addresses(
                w3=w3,
                token_contract=token_contract,
                start_block=token_start_block,
                end_block=snapshot_block,
                chunk_size=cfg.logs_chunk_size,
                sleep_s=cfg.logs_sleep_s,
                cache_path=del_cache,
            )
            candidate_set.update(delegates)
            print(f"[INFO] Delegation candidates added: {len(delegates)}")

    else:
        print("[WARN] token_contract or token_start_block missing; cannot enumerate strict holders. Candidate pool = voters only (plus any addresses in strategies).")

    # Strategy params might include extra contract addresses; we do NOT add them as voters, but we keep for debugging
    # If strategies are LP/staking, eligible base might need strategy-specific address enumeration.
    if flags.get("has_lp_or_staking"):
        print("[WARN] LP/Staking-like strategy detected. Strict eligible base may be incomplete unless you enumerate participants of those contracts.")

    # --- Compute voting power (Layer 2 basis) via Score API
    candidates = sorted(candidate_set)
    print(f"[INFO] Total candidate pool size for score: {len(candidates)}")
    score_api_error: Optional[str] = None
    local_score_warnings: Optional[List[str]] = None
    local_score_used: bool = False

    if cfg.local_score_only:
        scores, local_score_warnings = local_compute_scores(
            w3=w3,
            snapshot_block=snapshot_block,
            strategies=strategies,
            addresses=candidates,
        )
        local_score_used = True
        if local_score_warnings:
            print(f"[WARN] Local scoring warnings: {local_score_warnings}")
    else:
        try:
            scores = batch_compute_scores(
                space=space,
                network=cfg.network,
                snapshot=snapshot_block,
                strategies=strategies,
                addresses=candidates,
                batch_size=cfg.score_batch_size,
                sleep_s=cfg.score_sleep_s,
            )
        except Exception as e:
            score_api_error = str(e)
            print(f"[WARN] Score API failed; trying local RPC scoring fallback. Error: {score_api_error}")
            try:
                scores, local_score_warnings = local_compute_scores(
                    w3=w3,
                    snapshot_block=snapshot_block,
                    strategies=strategies,
                    addresses=candidates,
                )
                local_score_used = True
                if local_score_warnings:
                    print(f"[WARN] Local scoring warnings: {local_score_warnings}")
            except Exception as e2:
                print(f"[WARN] Local scoring also failed; proceeding with vp=0 for all candidates. Error: {e2}")
                scores = {a.lower(): 0.0 for a in candidates}

    # Thresholds
    tau_balance_token, tau_vp_token = apply_tau_from_usd(cfg)

    # Build Layer 1 table (raw holders)
    layer1_rows = []
    if balances_raw:
        for a, b_raw in balances_raw.items():
            bal = b_raw / (10 ** token_decimals)
            if bal >= tau_balance_token:
                layer1_rows.append((a, bal))
    df_l1 = pd.DataFrame(layer1_rows, columns=["address", "balance_token"])
    df_l1["proposal_id"] = proposal_id
    df_l1["space"] = space
    df_l1["snapshot_block"] = snapshot_block
    df_l1 = df_l1.sort_values("balance_token", ascending=False).reset_index(drop=True)

    # Build Layer 2 table (effective voting power holders)
    layer2_rows = []
    for a in candidates:
        vp = float(scores.get(a.lower(), 0.0))
        if vp >= tau_vp_token:
            layer2_rows.append((a, vp))
    df_l2 = pd.DataFrame(layer2_rows, columns=["address", "vp"])
    df_l2["proposal_id"] = proposal_id
    df_l2["space"] = space
    df_l2["snapshot_block"] = snapshot_block
    df_l2 = df_l2.sort_values("vp", ascending=False).reset_index(drop=True)

    # Build Layer 3 table (active participants)
    df_l3 = votes_df.copy()
    df_l3["proposal_id"] = proposal_id
    df_l3["space"] = space
    df_l3["snapshot_block"] = snapshot_block

    # --- Metrics / participation
    # Unweighted turnout
    n_voters = len(voters)
    n_l1 = len(df_l1)
    n_l2 = len(df_l2)

    turnout_l1 = (n_voters / n_l1) if n_l1 > 0 else None
    turnout_l2 = (n_voters / n_l2) if n_l2 > 0 else None

    # Weighted turnout (using vp): numerator=sum vp of voters, denominator=sum vp of layer2 eligible
    voters_vp_sum = 0.0
    for v in voters:
        voters_vp_sum += float(scores.get(v.lower(), 0.0))
    eligible_vp_sum = float(df_l2["vp"].sum()) if n_l2 > 0 else 0.0
    w_turnout = (voters_vp_sum / eligible_vp_sum) if eligible_vp_sum > 0 else None

    # Concentration stats on voting power (Layer 2)
    vp_vals = df_l2["vp"].tolist() if n_l2 > 0 else []
    top1 = topk_share(vp_vals, 1)
    top10 = topk_share(vp_vals, 10)
    top100 = topk_share(vp_vals, 100)

    summary = {
        "space": space,
        "proposal_id": proposal_id,
        "snapshot_block": snapshot_block,
        "n_candidates_scored": len(candidates),
        "n_voters": n_voters,
        "n_layer1_holders": n_l1,
        "n_layer2_vp_holders": n_l2,
        "score_api_error": score_api_error,
        "local_score_used": local_score_used,
        "local_score_warnings": local_score_warnings,
        "turnout_unweighted_layer1": turnout_l1,
        "turnout_unweighted_layer2": turnout_l2,
        "turnout_weighted_vp": w_turnout,
        "voters_vp_sum": voters_vp_sum,
        "eligible_vp_sum": eligible_vp_sum,
        "tau_balance_token": tau_balance_token,
        "tau_vp_token": tau_vp_token,
        "token_contract": token_contract or "",
        "token_start_block": token_start_block,
        "token_decimals": token_decimals,
        "flags": flags,
        "vp_top1_share": top1,
        "vp_top10_share": top10,
        "vp_top100_share": top100,
    }

    # --- Save outputs
    base = os.path.join(cfg.out_dir, space.replace(".", "_"))
    ensure_dir(base)

    df_l1.to_parquet(os.path.join(base, f"{proposal_id}_layer1_raw_holders.parquet"), index=False)
    df_l2.to_parquet(os.path.join(base, f"{proposal_id}_layer2_effective_vp.parquet"), index=False)
    df_l3.to_parquet(os.path.join(base, f"{proposal_id}_layer3_active_voters.parquet"), index=False)

    # Save summary json for proposal
    write_json(os.path.join(base, f"{proposal_id}_summary.json"), summary)

    print(f"[OK] Saved layers for {proposal_id}: L1={n_l1}, L2={n_l2}, L3(voters)={n_voters}")
    if turnout_l2 is not None and turnout_l2 > 1:
        print("[WARN] turnout_l2 > 1 — indicates candidate pool for Layer2 might be incomplete (missing eligible addresses), or threshold too high/low.")
    return summary


# =========================
# Per space driver
# =========================

def run_space(w3: Web3, cfg: RunConfig, space: str) -> List[dict]:
    ensure_dir(cfg.cache_dir)
    ensure_dir(cfg.out_dir)

    # Fetch network from Snapshot (should be "1" for ETH mainnet for your spaces)
    net = fetch_space_network(space)
    if net and str(net) != str(cfg.network):
        print(f"[WARN] Space {space} network={net} differs from cfg.network={cfg.network}. Using cfg.network.")
    proposals = fetch_recent_proposals(space, first=cfg.max_proposals)
    if not proposals:
        print(f"[WARN] No proposals found for space={space}")
        return []

    # Infer token from strategies of first proposal (usually stable within space)
    token_contract, debug = infer_token_contract_from_strategies(proposals[0]["strategies"])
    flags = detect_strategy_flags(proposals[0]["strategies"])

    # If token is missing, still proceed (Layer2 from voters only)
    token_start_block = None
    if token_contract and (not cfg.use_etherscan_start_block):
        print("[INFO] Etherscan start-block discovery disabled (--no_etherscan).")
    elif token_contract and cfg.use_etherscan_start_block and os.environ.get("ETHERSCAN_API_KEY"):
        # cache token start block per space
        start_cache = os.path.join(cfg.cache_dir, f"token_start_{space.replace('.','_')}_{token_contract[:10]}.json")
        cached = read_json(start_cache)
        if cached and cached.get("token_contract") == token_contract.lower():
            token_start_block = cached.get("token_start_block")
        else:
            print(f"[INFO] Finding token_start_block via Etherscan logs for {token_contract} ...")
            # earliest Transfer log block
            tb = etherscan_find_first_log_block(cfg.chainid, token_contract, topic0=TRANSFER_TOPIC, step=500_000, sleep_s=0.2)
            token_start_block = tb
            write_json(start_cache, {"token_contract": token_contract.lower(), "token_start_block": token_start_block})
    else:
        if token_contract:
            print("[WARN] ETHERSCAN_API_KEY missing or disabled; token_start_block will be None unless you set manually in cache/config.")
        else:
            print("[WARN] Could not infer token_contract from strategies; will run with voters-only candidate pool.")

    token_decimals = 18
    if token_contract:
        token_decimals = get_erc20_decimals(w3, token_contract, default=18)

    # Save per-space debug
    space_dir = os.path.join(cfg.out_dir, space.replace(".", "_"))
    ensure_dir(space_dir)
    write_json(os.path.join(space_dir, "_space_strategy_debug.json"), {
        "space": space,
        "token_contract": token_contract,
        "token_decimals": token_decimals,
        "token_start_block": token_start_block,
        "flags": flags,
        "debug": debug,
    })

    # Run each proposal
    summaries = []
    for p in proposals:
        # load / resume: if summary exists, skip
        summary_path = os.path.join(space_dir, f"{p['id']}_summary.json")
        if os.path.exists(summary_path):
            print(f"[SKIP] Existing summary found for {p['id']}")
            summaries.append(read_json(summary_path))
            continue

        s = run_one_proposal(
            w3=w3,
            cfg=cfg,
            space=space,
            proposal=p,
            token_contract=token_contract,
            token_start_block=token_start_block,
            token_decimals=token_decimals,
            flags=flags
        )
        summaries.append(s)

    # Save summary CSV
    df = pd.DataFrame([x for x in summaries if x is not None])
    df.to_csv(os.path.join(space_dir, "_space_summary.csv"), index=False)
    return summaries


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spaces", nargs="+", required=True, help="Snapshot space IDs, e.g. aavedao.eth ens.eth ...")
    ap.add_argument("--max_proposals", type=int, default=3)
    ap.add_argument("--out", default="out")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--tau_usd", type=float, default=100.0, help="USD dust threshold (optional; requires token_price_usd)")
    ap.add_argument("--token_price_usd", type=float, default=None, help="Constant token price for USD->token conversion (optional)")
    ap.add_argument("--tau_balance_token", type=float, default=0.0, help="Token threshold for Layer1 holders (if not using USD)")
    ap.add_argument("--tau_vp_token", type=float, default=0.0, help="Token threshold for Layer2 vp (if not using USD)")
    ap.add_argument("--no_etherscan", action="store_true", help="Disable Etherscan start-block discovery")
    ap.add_argument("--local_score_only", action="store_true", help="Skip Snapshot Score API and compute scores locally via RPC (slower)")
    args = ap.parse_args()

    rpc_url = os.environ.get("RPC_URL")
    if not rpc_url:
        raise RuntimeError("Missing RPC_URL env var. Set it to an Ethereum RPC endpoint.")

    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 180}))
    if not w3.is_connected():
        raise RuntimeError("RPC not connected. Check RPC_URL.")

    cfg = RunConfig(
        max_proposals=args.max_proposals,
        out_dir=args.out,
        cache_dir=args.cache,
        tau_usd=args.tau_usd,
        token_price_usd=args.token_price_usd,
        tau_balance_token=args.tau_balance_token,
        tau_vp_token=args.tau_vp_token,
        use_etherscan_start_block=(not args.no_etherscan),
        local_score_only=args.local_score_only,
    )

    ensure_dir(cfg.out_dir)
    ensure_dir(cfg.cache_dir)

    all_summaries = []
    for sp in args.spaces:
        print(f"\n\n#############################\n### SPACE: {sp}\n#############################")
        sums = run_space(w3, cfg, sp)
        all_summaries.extend([s for s in sums if s is not None])

    # global summary
    if all_summaries:
        df_all = pd.DataFrame(all_summaries)
        df_all.to_csv(os.path.join(cfg.out_dir, "_ALL_SPACES_SUMMARY.csv"), index=False)
        print(f"\n[OK] Global summary saved: {os.path.join(cfg.out_dir, '_ALL_SPACES_SUMMARY.csv')}")

if __name__ == "__main__":
    main()
