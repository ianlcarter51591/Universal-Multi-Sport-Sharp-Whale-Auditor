"""
Line history: the app remembers every line it sees, so movement is measured instead of typed in.

Math
----
Movement is measured from the selection's own point of view; NEGATIVE = the market moved TOWARD the selection:
    spreads  move = line_now - line_first                        (a 3-pt favourite going -3 -> -4 is -1, toward)
    totals   Over:  move = -(total_now - total_first)            (total rising is toward the Over)
             Under: move = +(total_now - total_first)
    h2h      move = -(implied_now - implied_first) * 100         (implied = raw implied probability of the price, in
                                                                  probability points; negative = price shortened)
"First" is the first reading this app saved, NOT the true opener (The Odds API does not return it), so the history
gets better the longer the app has been open on game days.
Early move:  movement between readings taken at least EARLY_HOURS before the start (needs two such readings, else N/A).
Late move:   inside CLOSING_HOURS of the start, movement since the last reading taken before that window (else N/A);
             outside the window it is simply "not in the closing window".
Thresholds per market (steam, early, late) live in MOVE_SCALE; points for spreads/totals, probability points for h2h.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from auditor_core import FEATURED_MARKETS, PREFERRED_BOOK, market_outcomes, pick_bookmaker

HISTORY_NAME = "auditor_line_history.csv"
COLUMNS = ["TS", "EVENT_ID", "LEAGUE", "GAME", "COMMENCE", "MARKET", "SELECTION", "LINE", "PRICE"]
KEY = ["EVENT_ID", "MARKET", "SELECTION"]
MIN_GAP_MINUTES = 30.0            # save an unchanged line again only after this long
KEEP_DAYS = 7                     # forget games that started more than this long ago
EARLY_HOURS = 24.0
CLOSING_HOURS = 3.0
MOVE_SCALE: Dict[str, Tuple[float, float, float]] = {      # (steam, early, late)
    "spreads": (1.0, 0.5, 0.5), "totals": (1.0, 0.5, 0.5), "h2h": (3.0, 1.5, 1.5), "team_totals": (1.0, 0.5, 0.5),
}


def implied_prob(american: Any) -> float:
    """Raw implied probability of American odds (nan if invalid)."""
    try:
        o = float(american)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(o) or abs(o) < 100:
        return float("nan")
    return -o / (-o + 100.0) if o < 0 else 100.0 / (o + 100.0)


def toward_move(market: str, selection: str, line0: Any, price0: Any, line1: Any, price1: Any) -> float:
    """Movement from reading 0 to reading 1, negative = toward the selection (nan if not measurable)."""
    if market == "h2h":
        return float(-(implied_prob(price1) - implied_prob(price0)) * 100.0)
    try:
        d = float(line1) - float(line0)
    except (TypeError, ValueError):
        return float("nan")
    if market in ("totals", "team_totals"):
        return -d if str(selection).strip().lower().endswith("over") else d
    return d


def rows_from_games(games: Sequence[Mapping[str, Any]], league: str, ts: str, preferred: str = PREFERRED_BOOK) -> pd.DataFrame:
    """One row per selection of every featured market, from the preferred book of each game."""
    out = []
    for g in games:
        book = pick_bookmaker(g.get("bookmakers", []), preferred)
        if book is None:
            continue
        for market, outcomes in market_outcomes(book).items():
            if market not in FEATURED_MARKETS:
                continue
            for o in outcomes:
                out.append({"TS": ts, "EVENT_ID": g.get("id"), "LEAGUE": league, "GAME": f"{g['away_team']} @ {g['home_team']}",
                            "COMMENCE": g.get("commence_time"), "MARKET": market, "SELECTION": o.get("name"),
                            "LINE": o.get("point"), "PRICE": o.get("price")})
    df = pd.DataFrame(out, columns=COLUMNS)
    df[["LINE", "PRICE"]] = df[["LINE", "PRICE"]].apply(pd.to_numeric, errors="coerce")
    return df


def append_new(history: pd.DataFrame, new: pd.DataFrame, min_gap_minutes: float = MIN_GAP_MINUTES) -> Tuple[pd.DataFrame, int]:
    """Add readings that are the first for their selection, differ from the last one, or are older than the gap. (frame, rows added)."""
    if new is None or new.empty:
        return history, 0
    if history is None or history.empty:
        return new.reset_index(drop=True), len(new)
    h = history.copy()
    h[["LINE", "PRICE"]] = h[["LINE", "PRICE"]].apply(pd.to_numeric, errors="coerce")
    last = h.sort_values("TS").groupby(KEY, as_index=False).tail(1)[KEY + ["TS", "LINE", "PRICE"]]
    last = last.rename(columns={"TS": "L_TS", "LINE": "L_LINE", "PRICE": "L_PRICE"})
    m = new.merge(last, on=KEY, how="left")
    age_min = (pd.to_datetime(m["TS"], utc=True) - pd.to_datetime(m["L_TS"], utc=True)).dt.total_seconds() / 60.0
    same = np.isclose(m["LINE"].fillna(-999.0), m["L_LINE"].fillna(-999.0)) & np.isclose(m["PRICE"].fillna(0.0), m["L_PRICE"].fillna(0.0))
    keep = (m["L_TS"].isna() | ~same | (age_min >= min_gap_minutes)).to_numpy()
    if not keep.any():
        return history, 0
    return pd.concat([history, new[keep]], ignore_index=True), int(keep.sum())


def prune(history: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Drop games that started more than KEEP_DAYS ago."""
    if history.empty:
        return history
    start = pd.to_datetime(history["COMMENCE"], utc=True, errors="coerce")
    return history[~(start < now - pd.Timedelta(days=KEEP_DAYS))].reset_index(drop=True)


@dataclass(frozen=True)
class Movement:
    """What the saved readings say about one selection. Moves: negative = toward the selection; None = can't tell."""

    has_history: bool
    n_points: int
    open_ts: Optional[str]
    open_line: Optional[float]
    move: Optional[float]
    early_move: Optional[float]
    late_move: Optional[float]
    in_closing: bool
    hours_to_start: float


def _num(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def movement(history: pd.DataFrame, event_id: str, market: str, selection: str, commence_iso: str,
             now: pd.Timestamp, now_line: Any, now_price: Any) -> Movement:
    """Movement of one selection from saved readings plus the current reading."""
    start = pd.to_datetime(commence_iso, utc=True)
    hours_now = float((start - now).total_seconds() / 3600.0)
    empty = Movement(False, 0, None, None, None, None, None, hours_now <= CLOSING_HOURS, hours_now)
    if history is None or history.empty:
        return empty
    s = history[(history["EVENT_ID"] == event_id) & (history["MARKET"] == market) & (history["SELECTION"] == selection)]
    if s.empty:
        return empty
    s = s.assign(_t=pd.to_datetime(s["TS"], utc=True)).sort_values("_t")
    pts = pd.DataFrame({"t": list(s["_t"]) + [now], "line": list(pd.to_numeric(s["LINE"], errors="coerce")) + [_num(now_line)],
                        "price": list(pd.to_numeric(s["PRICE"], errors="coerce")) + [_num(now_price)]})
    pts["h"] = (start - pts["t"]).dt.total_seconds() / 3600.0
    if len(pts) < 2:
        return empty

    def mv(a: int, b: int) -> Optional[float]:
        v = toward_move(market, selection, pts["line"].iloc[a], pts["price"].iloc[a], pts["line"].iloc[b], pts["price"].iloc[b])
        return None if not np.isfinite(v) else float(v)

    early_idx = np.flatnonzero(pts["h"].to_numpy() >= EARLY_HOURS)
    early = mv(int(early_idx[0]), int(early_idx[-1])) if len(early_idx) >= 2 else None
    in_closing = hours_now <= CLOSING_HOURS
    if in_closing:
        base = np.flatnonzero(pts["h"].to_numpy() >= CLOSING_HOURS)
        late = mv(int(base[-1]), len(pts) - 1) if len(base) else None
    else:
        late = 0.0
    return Movement(True, len(pts), s["TS"].iloc[0], _num(pts["line"].iloc[0]), mv(0, len(pts) - 1), early, late, in_closing, hours_now)
