"""
Board scan: score EVERY selection of every upcoming game from data the app already has (no extra API credits).

Inputs: the bulk odds payload (one call per league), the DraftKings Network splits rows (one page fetch), and the saved
line history. Output: `all_sides` (one row per selection) and `best` (one row per game+market: the side with the higher
tier / score, which in a two-way market is the whale side, so each market appears once).
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from auditor_core import FEATURED_MARKETS, PREFERRED_BOOK, format_commence, format_price, market_outcomes, match_split_rows, pick_bookmaker, read_split
from line_history import movement
from scoring import TIER_RANK, pair_warning, score_selection

SIDE_COLUMNS = ["EVENT_ID", "GAME", "COMMENCE", "WHEN", "MARKET", "SELECTION", "PICK", "LINE", "PRICE", "TICKETS_%", "HANDLE_%", "GAP",
                "TOWARD", "SCORE", "TIER", "MONEY", "REACTION", "TIMING", "STRUCTURE", "NOTE"]


def pick_label(market: str, selection: str, line: Optional[float], price: Any) -> str:
    """'Boston Celtics -3.5 (-110)', 'Over 221.5 (-105)' or 'New York Knicks (+145)'."""
    pt = "" if line is None or pd.isna(line) else (f" {line:+g}" if market == "spreads" else f" {line:g}")
    return f"{selection}{pt} ({format_price(price)})"


def scan_board(games: Sequence[Mapping[str, Any]], league: str, split_rows: Sequence[Mapping[str, Any]], history: pd.DataFrame,
               now: pd.Timestamp, preferred: str = PREFERRED_BOOK) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(best, all_sides) for every game that has not started."""
    rec: List[Dict[str, Any]] = []
    for g in games:
        start = pd.to_datetime(g["commence_time"], utc=True)
        book = pick_bookmaker(g.get("bookmakers", []), preferred)
        if book is None or start <= now:
            continue
        away, home = g["away_team"], g["home_team"]
        event_rows = match_split_rows(split_rows, away, home, g["commence_time"]) if split_rows else []
        for market, outcomes in market_outcomes(book).items():
            if market not in FEATURED_MARKETS:
                continue
            reads = [read_split(event_rows, market, o.get("name"), away, home) if event_rows else None for o in outcomes]
            warn = pair_warning([r.ticket_pct for r in reads if r], [r.handle_pct for r in reads if r]) if all(reads) and len(reads) == 2 else None
            for o, reading in zip(outcomes, reads):
                line, price, name = o.get("point"), o.get("price"), o.get("name")
                mv = movement(history, g["id"], market, name, g["commence_time"], now, line, price)
                t, h = (reading.ticket_pct, reading.handle_pct) if reading else (None, None)
                sc = score_selection(league, market, line, t, h, mv)
                rec.append({"EVENT_ID": g["id"], "GAME": f"{away} @ {home}", "COMMENCE": g["commence_time"], "WHEN": format_commence(g["commence_time"]),
                            "MARKET": market, "SELECTION": name, "PICK": pick_label(market, name, line, price), "LINE": line, "PRICE": price,
                            "TICKETS_%": t, "HANDLE_%": h, "GAP": None if t is None else round(h - t, 1),
                            "TOWARD": None if sc.toward is None else round(sc.toward, 2), "SCORE": round(sc.score, 1), "TIER": sc.tier,
                            "MONEY": sc.money, "REACTION": sc.reaction, "TIMING": sc.timing, "STRUCTURE": sc.structure,
                            "NOTE": " ".join(([warn] if warn else []) + sc.notes[:1])})
    all_sides = pd.DataFrame(rec, columns=SIDE_COLUMNS)
    if all_sides.empty:
        return all_sides.copy(), all_sides
    rank = all_sides["TIER"].map(TIER_RANK)
    ordered = all_sides.assign(_rank=rank).sort_values(["_rank", "SCORE"], ascending=False)
    best = ordered.drop_duplicates(["EVENT_ID", "MARKET"], keep="first").drop(columns="_rank")
    return best.reset_index(drop=True), all_sides


def pair_table(all_sides: pd.DataFrame, event_id: str, market: str) -> pd.DataFrame:
    """Both sides of one market next to each other, with the whale side marked."""
    p = all_sides[(all_sides["EVENT_ID"] == event_id) & (all_sides["MARKET"] == market)]
    if p.empty:
        return pd.DataFrame(columns=["PICK", "TICKETS_%", "HANDLE_%", "GAP", "TOWARD", "SCORE", "TIER", "WHALE SIDE"])
    gap = pd.to_numeric(p["GAP"], errors="coerce")
    whale = gap.eq(gap.max()) & (gap >= 5.0)
    out = p[["PICK", "TICKETS_%", "HANDLE_%", "GAP", "TOWARD", "SCORE", "TIER"]].copy()
    out["WHALE SIDE"] = ["◀ money is here" if w else "" for w in whale]
    return out.reset_index(drop=True)
