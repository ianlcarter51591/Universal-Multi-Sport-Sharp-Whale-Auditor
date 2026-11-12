"""
Split history: remembers each selection's ticket / handle readings so a "whale" signal must PERSIST before it is trusted.

Math
----
A reading is a money signal when gap = handle% - tickets% >= SIGNAL_GAP and tickets% >= MIN_TICKET_SHARE.
Persistence for a selection that shows the signal NOW:
  held   = minutes since the start of the unbroken run of signal readings that ends now. A run breaks at a reading without the signal
           or across a hole of more than MAX_GAP_HOURS between readings (the app was closed and we can't know what happened).
  watched = minutes since the first saved reading of the selection.
  True  if held >= HOLD_MINUTES;   False if watched >= HOLD_MINUTES but held < HOLD_MINUTES (the signal is new or flickering);
  None  if we have not watched long enough to say.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

SPLIT_NAME = "auditor_split_history.csv"
COLUMNS = ["TS", "EVENT_ID", "MARKET", "SELECTION", "TICKETS", "HANDLE", "COMMENCE"]
KEY = ["EVENT_ID", "MARKET", "SELECTION"]
MIN_GAP_MINUTES = 30.0
HOLD_MINUTES = 30.0
MAX_GAP_HOURS = 4.0
KEEP_DAYS = 7
SIGNAL_GAP = 15.0
MIN_TICKET_SHARE = 15.0


def is_signal(ticket: Optional[float], handle: Optional[float]) -> bool:
    """True when the readings show a money signal (big enough gap on a real share of the tickets)."""
    return ticket is not None and handle is not None and handle - ticket >= SIGNAL_GAP and ticket >= MIN_TICKET_SHARE


def rows_from_sides(all_sides: pd.DataFrame, ts: str) -> pd.DataFrame:
    """Readings worth saving: every selection that has real splits."""
    if all_sides is None or all_sides.empty:
        return pd.DataFrame(columns=COLUMNS)
    s = all_sides[all_sides["TICKETS_%"].notna() & all_sides["HANDLE_%"].notna()]
    return pd.DataFrame({"TS": ts, "EVENT_ID": s["EVENT_ID"].to_numpy(), "MARKET": s["MARKET"].to_numpy(), "SELECTION": s["SELECTION"].to_numpy(),
                         "TICKETS": s["TICKETS_%"].to_numpy(), "HANDLE": s["HANDLE_%"].to_numpy(), "COMMENCE": s["COMMENCE"].to_numpy()}, columns=COLUMNS)


def append_new(history: pd.DataFrame, new: pd.DataFrame, min_gap_minutes: float = MIN_GAP_MINUTES) -> Tuple[pd.DataFrame, int]:
    """Add readings that are first for their selection, changed, or older than the gap. (frame, rows added)."""
    if new is None or new.empty:
        return history, 0
    if history is None or history.empty:
        return new.reset_index(drop=True), len(new)
    h = history.copy()
    h[["TICKETS", "HANDLE"]] = h[["TICKETS", "HANDLE"]].apply(pd.to_numeric, errors="coerce")
    last = h.sort_values("TS").groupby(KEY, as_index=False).tail(1)[KEY + ["TS", "TICKETS", "HANDLE"]]
    last = last.rename(columns={"TS": "L_TS", "TICKETS": "L_T", "HANDLE": "L_H"})
    m = new.merge(last, on=KEY, how="left")
    age = (pd.to_datetime(m["TS"], utc=True) - pd.to_datetime(m["L_TS"], utc=True)).dt.total_seconds() / 60.0
    same = np.isclose(m["TICKETS"], m["L_T"].fillna(-999.0), atol=0.05) & np.isclose(m["HANDLE"], m["L_H"].fillna(-999.0), atol=0.05)
    keep = (m["L_TS"].isna() | ~same | (age >= min_gap_minutes)).to_numpy()
    if not keep.any():
        return history, 0
    return pd.concat([history, new[keep]], ignore_index=True), int(keep.sum())


def prune(history: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Drop games that started more than KEEP_DAYS ago."""
    if history.empty:
        return history
    start = pd.to_datetime(history["COMMENCE"], utc=True, errors="coerce")
    return history[~(start < now - pd.Timedelta(days=KEEP_DAYS))].reset_index(drop=True)


def persistence(history: pd.DataFrame, event_id: str, market: str, selection: str, now: pd.Timestamp) -> Optional[bool]:
    """Has the money signal held for HOLD_MINUTES? Call only when the signal is present in the current reading."""
    if history is None or history.empty:
        return None
    s = history[(history["EVENT_ID"] == event_id) & (history["MARKET"] == market) & (history["SELECTION"] == selection)]
    if s.empty:
        return None
    s = s.assign(_t=pd.to_datetime(s["TS"], utc=True)).sort_values("_t")
    t = pd.to_numeric(s["TICKETS"], errors="coerce").to_numpy()
    h = pd.to_numeric(s["HANDLE"], errors="coerce").to_numpy()
    times = s["_t"].tolist() + [now]
    flagged = (h - t >= SIGNAL_GAP) & (t >= MIN_TICKET_SHARE)
    hole = np.diff(np.array([x.value for x in times], dtype="int64")) > MAX_GAP_HOURS * 3.6e12      # nanoseconds
    broken = ~flagged | hole
    start_i = int(np.flatnonzero(broken)[-1]) + 1 if broken.any() else 0
    run_start = now if start_i >= len(flagged) else times[start_i]
    held = (now - run_start).total_seconds() / 60.0
    watched = (now - times[0]).total_seconds() / 60.0
    return True if held >= HOLD_MINUTES else False if watched >= HOLD_MINUTES else None
