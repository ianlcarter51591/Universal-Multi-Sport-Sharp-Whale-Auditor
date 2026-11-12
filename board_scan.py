"""
Board scan: score EVERY selection of every upcoming game from data the app already has (no extra API credits).

Inputs: the bulk odds payload (one call per league), the DraftKings Network splits rows (one page fetch), the saved line
history and split history, and the games the user marked as news-driven. Output: `all_sides` (one row per selection) and
`best` (one row per game+market: the side with the higher tier / score, which in a two-way market is the whale side).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from auditor_core import FEATURED_MARKETS, PREFERRED_BOOK, SplitReading, format_commence, format_price, market_outcomes, match_split_rows, pick_bookmaker, read_split
from line_history import Movement, movement
from scoring import TIER_RANK, TOTAL_CHECKS, Context, Scored, pair_warning, score_selection
from split_history import persistence

SIDE_COLUMNS = ["EVENT_ID", "GAME", "COMMENCE", "WHEN", "MARKET", "SELECTION", "PICK", "LINE", "PRICE", "TICKETS_%", "HANDLE_%", "GAP", "BET_SIZE_X",
                "TOWARD", "CHECKS", "SCORE", "TIER", "MONEY", "NOTE"]


@dataclass(frozen=True)
class Evaluated:
    """One selection after scoring."""

    outcome: Mapping[str, Any]
    reading: Optional[SplitReading]
    movement: Movement
    scored: Scored


def _note(sc: Scored) -> str:
    """One line for the Board: the filter that blocked the pick, else what a LEAN still needs for SHARP, else the first note."""
    if sc.tier != "FILTERED" and sc.needs:
        return next((n for n in sc.notes if n.startswith("To become SHARP")), "")
    return sc.notes[0] if sc.notes else ""


def pick_label(market: str, selection: str, line: Optional[float], price: Any) -> str:
    """'Boston Celtics -3.5 (-110)', 'Over 221.5 (-105)' or 'New York Knicks (+145)'."""
    pt = "" if line is None or pd.isna(line) else (f" {line:+g}" if market == "spreads" else f" {line:g}")
    return f"{selection}{pt} ({format_price(price)})"


def evaluate_market(game: Mapping[str, Any], market: str, outcomes: Sequence[Mapping[str, Any]], event_rows: Sequence[Mapping[str, Any]],
                    history: pd.DataFrame, split_hist: Optional[pd.DataFrame], now: pd.Timestamp, league: str,
                    news_ids: Collection[str] = ()) -> List[Evaluated]:
    """Score every selection of one market, with pair-aware context (other side's splits, clean-splits check, persistence)."""
    away, home = game["away_team"], game["home_team"]
    reads: List[Optional[SplitReading]] = [read_split(event_rows, market, o.get("name"), away, home) if event_rows else None for o in outcomes]
    have = [r for r in reads if r is not None]
    complete = len(have) == len(reads) and len(reads) >= 2
    degenerate = any(r.ticket_pct in (0.0, 100.0) or r.handle_pct in (0.0, 100.0) for r in have)
    pair_bad = bool(complete and len(reads) == 2 and pair_warning([r.ticket_pct for r in have], [r.handle_pct for r in have]))
    out: List[Evaluated] = []
    for i, (o, reading) in enumerate(zip(outcomes, reads)):
        line, price, name = o.get("point"), o.get("price"), o.get("name")
        mv = movement(history, game["id"], market, name, game["commence_time"], now, line, price)
        opp = None
        if reading is not None and complete:
            others = [r for j, r in enumerate(reads) if j != i]
            opp = (sum(r.ticket_pct for r in others), sum(r.handle_pct for r in others))
        t, h = (reading.ticket_pct, reading.handle_pct) if reading else (None, None)
        persist = persistence(split_hist, game["id"], market, name, now) if reading is not None and split_hist is not None else None
        ctx = Context(price=None if price is None else float(price), opp=opp, degenerate=degenerate, pair_bad=pair_bad, persistence=persist,
                      news=game["id"] in set(news_ids))
        out.append(Evaluated(o, reading, mv, score_selection(league, market, line, t, h, mv, None, ctx)))
    return out


def scan_board(games: Sequence[Mapping[str, Any]], league: str, split_rows: Sequence[Mapping[str, Any]], history: pd.DataFrame,
               now: pd.Timestamp, split_hist: Optional[pd.DataFrame] = None, news_ids: Collection[str] = (),
               preferred: str = PREFERRED_BOOK) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(best, all_sides) for every game that has not started."""
    rec: List[Dict[str, Any]] = []
    for g in games:
        start = pd.to_datetime(g["commence_time"], utc=True)
        book = pick_bookmaker(g.get("bookmakers", []), preferred)
        if book is None or start <= now:
            continue
        event_rows = match_split_rows(split_rows, g["away_team"], g["home_team"], g["commence_time"]) if split_rows else []
        for market, outcomes in market_outcomes(book).items():
            if market not in FEATURED_MARKETS:
                continue
            for ev in evaluate_market(g, market, outcomes, event_rows, history, split_hist, now, league, news_ids):
                o, r, sc = ev.outcome, ev.reading, ev.scored
                line, price, name = o.get("point"), o.get("price"), o.get("name")
                t, h = (r.ticket_pct, r.handle_pct) if r else (None, None)
                rec.append({"EVENT_ID": g["id"], "GAME": f"{g['away_team']} @ {g['home_team']}", "COMMENCE": g["commence_time"],
                            "WHEN": format_commence(g["commence_time"]), "MARKET": market, "SELECTION": name, "PICK": pick_label(market, name, line, price),
                            "LINE": line, "PRICE": price, "TICKETS_%": t, "HANDLE_%": h, "GAP": None if t is None else round(h - t, 1),
                            "BET_SIZE_X": None if sc.bet_size is None else round(min(sc.bet_size, 99.0), 1),
                            "TOWARD": None if sc.toward is None else round(sc.toward, 2), "CHECKS": f"{sc.points}/{TOTAL_CHECKS}", "SCORE": float(sc.points),
                            "TIER": sc.tier, "MONEY": sc.money, "NOTE": _note(sc)})
    all_sides = pd.DataFrame(rec, columns=SIDE_COLUMNS)
    if all_sides.empty:
        return all_sides.copy(), all_sides
    ordered = all_sides.assign(_rank=all_sides["TIER"].map(TIER_RANK), _m=all_sides["MONEY"].astype(float).fillna(0.0)).sort_values(["_rank", "SCORE", "_m"], ascending=False)
    best = ordered.drop_duplicates(["EVENT_ID", "MARKET"], keep="first").drop(columns=["_rank", "_m"])
    return best.reset_index(drop=True), all_sides


def pair_table(all_sides: pd.DataFrame, event_id: str, market: str) -> pd.DataFrame:
    """Both sides of one market next to each other, with the whale side marked."""
    cols = ["PICK", "TICKETS_%", "HANDLE_%", "GAP", "BET_SIZE_X", "TOWARD", "CHECKS", "TIER"]
    p = all_sides[(all_sides["EVENT_ID"] == event_id) & (all_sides["MARKET"] == market)]
    if p.empty:
        return pd.DataFrame(columns=cols + ["WHALE SIDE"])
    gap = pd.to_numeric(p["GAP"], errors="coerce")
    whale = gap.eq(gap.max()) & (gap >= 5.0)
    out = p[cols].copy()
    out["WHALE SIDE"] = ["◀ money is here" if w else "" for w in whale]
    return out.reset_index(drop=True)
