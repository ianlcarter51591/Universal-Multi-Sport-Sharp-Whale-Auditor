"""
Off-market check: is DraftKings paying more than the rest of the market says the side is worth?

Math
----
For every other book b that lists the same market AND the same line as DraftKings:
    q_b(side)  = implied(price_b(side)) / sum over that book's sides of implied(price_b(.))      (multiplicative devig)
    p_fair     = sum_b w_b * q_b / sum_b w_b                                                       (weighted mean; Pinnacle weighs 3, others 1)
    EV_%       = 100 * (p_fair * decimal(price_DraftKings) - 1)                                    (what a unit bet at DraftKings is worth if p_fair is right)
A side is OFF-MARKET when EV_% >= OFF_MARKET_EV and the reference weight is at least MIN_REF_WEIGHT.
Limits: US books copy each other, so p_fair is a consensus, not the truth; books on a different line are skipped, not adjusted; in the last
STALE_WINDOW_HOURS before the start a reference book that has not updated for STALE_MINUTES is ignored.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from line_history import implied_prob

OFF_MARKET_EV = 2.0
MIN_REF_WEIGHT = 3.0
STALE_MINUTES = 60.0
STALE_WINDOW_HOURS = 6.0
BOOK_WEIGHT: Dict[str, float] = {"pinnacle": 3.0}
EXCLUDED_BOOKS = ("betfair", "matchbook", "smarkets", "betopenly", "novig", "prophetx")      # exchanges and lay prices are not offers you could take


def merge_books(games: Sequence[Dict[str, Any]], extra_games: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Add the bookmakers of `extra_games` (same event ids) to `games` when a game does not already list that book."""
    by_id = {g.get("id"): g for g in extra_games}
    out: List[Dict[str, Any]] = []
    for g in games:
        have = {b.get("key") for b in g.get("bookmakers", [])}
        add = [b for b in by_id.get(g.get("id"), {}).get("bookmakers", []) if b.get("key") not in have]
        out.append({**g, "bookmakers": list(g.get("bookmakers", [])) + add} if add else g)
    return out


def decimal_odds(american: Any) -> float:
    """Decimal odds of an American price (nan if invalid)."""
    try:
        o = float(american)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(o) or abs(o) < 100:
        return float("nan")
    return 1.0 + (o / 100.0 if o > 0 else 100.0 / -o)


def _same_point(a: Any, b: Any) -> bool:
    """Lines match (both missing counts as a match, as in the moneyline)."""
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return a is None and b is None or (pd.isna(a) and pd.isna(b))


def _novig(market: Mapping[str, Any], name: Any, point: Any) -> Optional[float]:
    """No-vig probability of (name, point) inside one book's market, or None when it can't be read."""
    outs = market.get("outcomes", [])
    imp = [implied_prob(o.get("price")) for o in outs]
    if len(outs) < 2 or any(np.isnan(i) for i in imp):
        return None
    total = float(sum(imp))
    for o, i in zip(outs, imp):
        if o.get("name") == name and _same_point(o.get("point"), point):
            return i / total
    return None


def _age_ok(book: Mapping[str, Any], market: Mapping[str, Any], commence: Any, now: Optional[pd.Timestamp]) -> bool:
    """False when the game is close and this book's line has not been refreshed recently."""
    if now is None or commence is None:
        return True
    start = pd.to_datetime(commence, utc=True, errors="coerce")
    upd = pd.to_datetime(market.get("last_update") or book.get("last_update"), utc=True, errors="coerce")
    if pd.isna(start) or pd.isna(upd) or (start - now).total_seconds() / 3600.0 > STALE_WINDOW_HOURS:
        return True
    return (now - upd).total_seconds() / 60.0 <= STALE_MINUTES


def reference_rows(game: Mapping[str, Any], market: str, name: Any, point: Any, preferred: str, now: Optional[pd.Timestamp] = None) -> List[Dict[str, Any]]:
    """One row per other book: its price, no-vig probability, weight and whether it was used (with the reason when not)."""
    rows: List[Dict[str, Any]] = []
    for b in game.get("bookmakers", []):
        key = str(b.get("key", ""))
        if key == preferred:
            continue
        row: Dict[str, Any] = {"BOOK": b.get("title") or key, "KEY": key, "PRICE": None, "NOVIG_%": None, "WEIGHT": BOOK_WEIGHT.get(key, 1.0), "USED": False, "WHY": ""}
        m = next((m for m in b.get("markets", []) if m.get("key") == market), None)
        if key.startswith(EXCLUDED_BOOKS):
            row["WHY"] = "exchange / lay price"
        elif m is None:
            row["WHY"] = "market not listed"
        else:
            o = next((o for o in m.get("outcomes", []) if o.get("name") == name and _same_point(o.get("point"), point)), None)
            q = _novig(m, name, point)
            if o is None or q is None:
                row["WHY"] = "different line"
            elif not _age_ok(b, m, game.get("commence_time"), now):
                row["WHY"] = "stale price"
            else:
                row.update(PRICE=o.get("price"), **{"NOVIG_%": 100.0 * q}, USED=True)
        rows.append(row)
    return rows


def consensus(rows: Sequence[Mapping[str, Any]]) -> Tuple[float, float, int]:
    """(weighted fair probability in [0, 1] or nan, total weight, number of books) from reference_rows."""
    used = [r for r in rows if r["USED"]]
    w = float(sum(r["WEIGHT"] for r in used))
    if not used or w <= 0:
        return float("nan"), 0.0, 0
    return float(sum(r["WEIGHT"] * r["NOVIG_%"] for r in used) / w / 100.0), w, len(used)


def edge(game: Mapping[str, Any], market: str, name: Any, point: Any, dk_price: Any, preferred: str, now: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
    """FAIR_% (consensus), VS_MKT_EV_%, REF_BOOKS and OFF_MARKET for one DraftKings selection."""
    p, w, n = consensus(reference_rows(game, market, name, point, preferred, now))
    dec = decimal_odds(dk_price)
    if not np.isfinite(p) or not np.isfinite(dec):
        return {"FAIR_%": None, "VS_MKT_EV_%": None, "REF_BOOKS": n, "OFF_MARKET": False}
    ev = 100.0 * (p * dec - 1.0)
    return {"FAIR_%": round(100.0 * p, 1), "VS_MKT_EV_%": round(ev, 1), "REF_BOOKS": n, "OFF_MARKET": bool(ev >= OFF_MARKET_EV and w >= MIN_REF_WEIGHT)}
