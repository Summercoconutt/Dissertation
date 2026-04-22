"""
Recursively scan Snapshot proposal parquet files and build a unified master vote table.
Writes Parquet incrementally to control memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import (
    CANONICAL_MASTER_COLUMNS,
    DATA_PROCESSED_DIR,
    MASTER_CSV_SAMPLE_ROWS,
    MAX_PARQUET_FILES,
    PARQUET_READ_BATCH,
    SNAPSHOT_SPACES_ROOT,
    log,
)
from .standardize_votes import standardize_vote_dataframe


def iter_space_proposal_parquets(root: Path) -> Iterator[Tuple[str, str, Path]]:
    """Yield (space, proposal_id, parquet_path)."""
    for space_dir in sorted(root.glob("space=*")):
        if not space_dir.is_dir():
            continue
        name = space_dir.name
        space = name.split("=", 1)[1] if "=" in name else name
        for pq_path in sorted(space_dir.glob("proposal=*.parquet")):
            stem = pq_path.name
            if not stem.endswith(".parquet"):
                continue
            prop_part = stem[:-8]
            proposal_id = prop_part.split("=", 1)[1] if prop_part.startswith("proposal=") else prop_part
            yield space, proposal_id, pq_path


def _write_csv_sample_from_parquet(parquet_path: Path, csv_path: Path, max_rows: int) -> None:
    """Read only the first row groups up to max_rows (avoids loading huge file)."""
    pf = pq.ParquetFile(parquet_path)
    chunks = []
    n = 0
    for batch in pf.iter_batches(batch_size=min(50_000, max_rows)):
        t = pa.Table.from_batches([batch]).to_pandas()
        chunks.append(t)
        n += len(t)
        if n >= max_rows:
            break
    if not chunks:
        return
    df = pd.concat(chunks, ignore_index=True).head(max_rows)
    df.to_csv(csv_path, index=False)
    log(f"Sample CSV saved: {csv_path} (rows={len(df)})")


def build_master_votes(
    snapshot_root: Optional[Path] = None,
    out_parquet: Optional[Path] = None,
    out_csv_sample: Optional[Path] = None,
) -> Path:
    """
    Scan parquet files, standardize, write master_votes.parquet (+ optional CSV sample).
    Returns path to main parquet output.
    """
    root = snapshot_root or SNAPSHOT_SPACES_ROOT
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_parquet = out_parquet or (DATA_PROCESSED_DIR / "master_votes.parquet")
    out_csv_sample = out_csv_sample or (DATA_PROCESSED_DIR / "master_votes_sample.csv")

    if not root.exists():
        raise FileNotFoundError(f"Snapshot root not found: {root}")

    if out_parquet.exists():
        out_parquet.unlink()
        log(f"Removed existing {out_parquet} for a clean rebuild.")

    log(f"Scanning parquet under: {root}")
    paths: List[Tuple[str, str, Path]] = []
    for tup in iter_space_proposal_parquets(root):
        paths.append(tup)
        if MAX_PARQUET_FILES is not None and len(paths) >= MAX_PARQUET_FILES:
            break

    log(f"Queued {len(paths)} proposal parquet files (MAX_PARQUET_FILES={MAX_PARQUET_FILES}).")

    writer: Optional[pq.ParquetWriter] = None
    batch_frames: List[pd.DataFrame] = []
    total_rows = 0

    def flush_batch() -> None:
        nonlocal writer, total_rows, batch_frames
        if not batch_frames:
            return
        big = pd.concat(batch_frames, ignore_index=True)
        batch_frames.clear()
        # Stable dtypes across batches for ParquetWriter schema
        for c in ("voting_power", "vp_ratio_pct", "followers_count"):
            if c in big.columns:
                big[c] = pd.to_numeric(big[c], errors="coerce").astype("float64")
        table = pa.Table.from_pandas(big, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(out_parquet), table.schema, compression="snappy")
        writer.write_table(table)
        total_rows += len(big)
        log(f"Written cumulative rows: {total_rows}")

    for i, (space, proposal_id, pq_path) in enumerate(paths):
        try:
            raw = pd.read_parquet(pq_path)
        except Exception as e:
            log(f"SKIP read error {pq_path}: {e}")
            continue
        std = standardize_vote_dataframe(raw, space=space, proposal_id=proposal_id)
        batch_frames.append(std)
        if len(batch_frames) >= PARQUET_READ_BATCH:
            flush_batch()
        if (i + 1) % 500 == 0:
            log(f"Processed {i+1}/{len(paths)} files")

    flush_batch()
    if writer is not None:
        writer.close()
        log(f"Master votes parquet saved: {out_parquet} (rows={total_rows})")
    else:
        empty = pd.DataFrame(columns=CANONICAL_MASTER_COLUMNS)
        empty.to_parquet(out_parquet, index=False)
        log(f"No rows read; wrote empty parquet: {out_parquet}")

    if MASTER_CSV_SAMPLE_ROWS and total_rows > 0 and out_parquet.exists():
        try:
            _write_csv_sample_from_parquet(out_parquet, out_csv_sample, MASTER_CSV_SAMPLE_ROWS)
        except Exception as e:
            log(f"Could not write CSV sample: {e}")

    return out_parquet
