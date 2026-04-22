"""
Standardize raw proposal-level vote rows to a fixed canonical schema.
Handles mixed Choice types (int codes vs weighted JSON strings).
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

import pandas as pd

from .config import CANONICAL_MASTER_COLUMNS, CHOICE_INT_TO_NORM, RAW_TO_CANONICAL, log


def _safe_json_loads(s: str) -> Optional[dict]:
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # try replacing single quotes (rare)
        try:
            return json.loads(s.replace("'", '"'))
        except json.JSONDecodeError:
            return None


def normalize_choice(choice_val: Any) -> str:
    """
    Map raw Choice to one of: for, against, abstain, other, unknown.
    Weighted JSON: assign to category with max weight; ties -> other.
    """
    if choice_val is None or (isinstance(choice_val, float) and pd.isna(choice_val)):
        return "unknown"
    # integer / float code
    if isinstance(choice_val, (int, float)) and not isinstance(choice_val, bool):
        code = int(choice_val)
        return CHOICE_INT_TO_NORM.get(code, "other")
    s = str(choice_val).strip()
    if not s:
        return "unknown"
    if s.startswith("{") and s.endswith("}"):
        d = _safe_json_loads(s)
        if not d:
            return "other"
        # keys may be str ints "1","2"
        weights = {}
        for k, v in d.items():
            try:
                kk = int(str(k))
                weights[kk] = float(v)
            except (ValueError, TypeError):
                continue
        if not weights:
            return "other"
        best_k = max(weights, key=lambda x: weights[x])
        return CHOICE_INT_TO_NORM.get(best_k, "other")
    # plain digit string
    if re.fullmatch(r"-?\d+", s):
        return CHOICE_INT_TO_NORM.get(int(s), "other")
    low = s.lower()
    if "for" in low and "against" not in low:
        return "for"
    if "against" in low or "reject" in low:
        return "against"
    if "abstain" in low:
        return "abstain"
    return "other"


def standardize_vote_dataframe(
    df: pd.DataFrame,
    space: str,
    proposal_id: str,
) -> pd.DataFrame:
    """
    Rename known raw columns to canonical names, add space/proposal_id, normalize choice.
    """
    if df.empty:
        return pd.DataFrame(columns=CANONICAL_MASTER_COLUMNS)

    renamed = {}
    for raw_col, std_col in RAW_TO_CANONICAL.items():
        if raw_col in df.columns:
            renamed[raw_col] = std_col
    part = df.rename(columns=renamed)

    out = pd.DataFrame(index=part.index)
    for c in CANONICAL_MASTER_COLUMNS:
        out[c] = pd.NA

    for c in part.columns:
        if c in CANONICAL_MASTER_COLUMNS:
            out[c] = part[c]

    out["space"] = space
    out["proposal_id"] = proposal_id

    if "choice_raw" in out.columns:
        # Mixed int / JSON-string in raw Snapshot exports — store as string for Parquet schema
        out["choice_raw"] = out["choice_raw"].apply(lambda x: x if pd.isna(x) else str(x))
        out["choice_norm"] = out["choice_raw"].map(normalize_choice)
    else:
        out["choice_norm"] = "unknown"

    # Coerce booleans / numeric helpers
    if "is_whale" in out.columns:
        out["is_whale"] = out["is_whale"].map(_coerce_bool)
    if "aligned_with_majority" in out.columns:
        out["aligned_with_majority"] = out["aligned_with_majority"].map(_coerce_bool)

    for c in ("voting_power", "vp_ratio_pct", "followers_count"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.reindex(columns=CANONICAL_MASTER_COLUMNS)
    return out


def _coerce_bool(x: Any) -> Optional[bool]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None
