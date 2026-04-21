"""
Run voter clustering without needing `python -m` (adds src/ to path).

Usage (from Voter_Clustering directory):
  python run_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from voter_clustering.main_voter_clustering import main  # noqa: E402

if __name__ == "__main__":
    main()
