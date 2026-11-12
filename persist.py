"""CSV persistence for the Auditor: local file always, private GitHub Gist when GITHUB_GIST_TOKEN is set.

Safety (same rule as the prop apps): a file is never pushed before it has been read from the Gist once. If the first
read finds saved data that the app never loaded, `save` merges it into the current data (current rows win by key) and
saves again, so a restart can never overwrite history.
"""
from __future__ import annotations

import io
import os
from typing import List, Optional, Sequence

import pandas as pd

import cloud_store

DATA_DIR = os.environ.get("AUDITOR_DATA_DIR", ".")


def local_path(name: str) -> str:
    """Local copy of a stored file."""
    return os.path.join(DATA_DIR, name)


def parse(text: Optional[str], columns: Sequence[str]) -> pd.DataFrame:
    """CSV text -> DataFrame with exactly `columns` (missing ones added empty; unreadable text -> empty frame)."""
    cols: List[str] = list(columns)
    if not text or not text.strip():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception:
        return pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[cols]


def load(name: str, columns: Sequence[str]) -> pd.DataFrame:
    """Gist copy if reachable (mirrored to disk), else the local file, else an empty frame. Never raises."""
    try:
        return parse(cloud_store.load_text(name, local_path(name)), columns)
    except Exception:
        return pd.DataFrame(columns=list(columns))


def save(name: str, df: pd.DataFrame, key_cols: Sequence[str], columns: Sequence[str], force: bool = False) -> pd.DataFrame:
    """Save `df`; returns the data to keep in memory (merged with a recovered copy when one turned up). Never raises."""
    try:
        recovered = cloud_store.save_text(name, df.to_csv(index=False), local_path(name), force)
        if recovered:
            merged = pd.concat([parse(recovered, columns), df[list(columns)]], ignore_index=True)
            merged = merged.drop_duplicates(list(key_cols), keep="last").reset_index(drop=True)
            cloud_store.save_text(name, merged.to_csv(index=False), local_path(name), force)
            return merged
    except Exception:
        pass
    return df
