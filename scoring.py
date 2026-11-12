"""
Sharp / whale checklist: five yes/no checks, a tier, and a plain "needs" note. Pure functions, no Streamlit.

The five checks (each is Yes, No, or Not yet known; only Yes counts toward the "x/5"):
  1. Money gap        handle% - tickets% >= 15 points
  2. Bigger bets      R >= 1.5, where R = (handle/tickets of this side) / (handle/tickets of the other sides)
  3. Line moved       the line moved toward the pick by at least the steam size (spreads / totals 1.0 pt, moneylines 3 prob. pts)
  4. Held 30 minutes  the money signal (check 1 with >= 15% of tickets) has been there for 30+ minutes of saved readings
  5. A market tell    any one of: reverse line movement (<= 40% of tickets and the line moved toward the pick by half the steam
                      size), an early move (24h+ out) or late move (closing window) toward the pick, or a spread crossing a key number

Tiers:  NO DATA   no real splits (the app never invents numbers)
        FILTERED  it would be LEAN / SHARP but failed a noise filter (below)
        SHARP     checks 1, 3, 4 and 5 are all Yes: the money is real, the market agrees, it held, and there is a tell
        LEAN      check 1 is Yes and at least one other check is Yes (usually bigger bets)
        PASS      everything else, including money the line moved AGAINST
Noise filters:  price band (h2h -300..+250, spreads / totals -130..+110) | >= 15% of tickets | clean splits (no 0% / 100%, two-way sides
        add to ~100%) | signal held 30 min once watched that long | > 30 minutes to the start | game not marked as news.
Ties are broken by money strength (the gap and bet-size strengths below), never shown as a score.
  gap_strength = clip((gap - 5) / 30, 0, 1);   size_strength = clip(R - 1, 0, 1);   money = mean of the two
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from auditor_core import SPORT_CONFIG
from line_history import MOVE_SCALE, Movement
from split_history import MIN_TICKET_SHARE, SIGNAL_GAP, is_signal

TOTAL_CHECKS = 5
GAP_FLOOR, GAP_FULL = 5.0, 35.0
SIZE_MET = 1.5
RLM_TICKETS_MAX = 40.0
CLOSE_TO_START_HOURS = 0.5
PAIR_TOLERANCE = 4.0
PRICE_BANDS: Dict[str, Tuple[float, float]] = {"h2h": (-300.0, 250.0), "spreads": (-130.0, 110.0), "totals": (-130.0, 110.0)}
TIER_RANK: Dict[str, int] = {"SHARP": 4, "FILTERED": 3, "LEAN": 3, "PASS": 2, "NO DATA": 0}      # a filtered whale side outranks the quiet side
YES, NO, UNKNOWN = "✅ Yes", "❌ No", "⏳ Not yet known"


@dataclass(frozen=True)
class Context:
    """Everything the filters need besides the splits and line movement."""

    price: Optional[float] = None
    opp: Optional[Tuple[float, float]] = None        # other sides' summed (tickets %, handle %); None = assume a two-way complement
    degenerate: bool = False                          # any side of the market reads exactly 0% or 100%
    pair_bad: bool = False                            # a two-way market's sides don't add to ~100%
    persistence: Optional[bool] = None                # money signal held >= 30 min (None = can't tell yet / no signal now)
    news: bool = False                                # user marked this game as injury / lineup news


@dataclass(frozen=True)
class Scored:
    """Result for one selection. `score` is the number of checks met (0-5), kept as a float for sorting and the signal log."""

    score: float
    tier: str
    money: Optional[float]                    # tie-break strength 0-1, not shown as a score
    toward: Optional[float]                   # line move toward the pick (positive = toward), None without history
    rows: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    filters: List[Dict[str, str]] = field(default_factory=list)
    blocked: List[str] = field(default_factory=list)
    bet_size: Optional[float] = None          # R, how many times bigger this side's average bet is than the other side's
    points: int = 0
    needs: List[str] = field(default_factory=list)


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(x, lo, hi))


def money_strength(ticket: float, handle: float) -> float:
    """0-1 strength of the handle-over-tickets gap."""
    return _clip((handle - ticket - GAP_FLOOR) / (GAP_FULL - GAP_FLOOR))


def relative_bet_size(ticket: float, handle: float, opp_ticket: float, opp_handle: float) -> Optional[float]:
    """R = (handle/tickets) / (opp handle/opp tickets); None when it can't be computed."""
    if ticket is None or opp_ticket is None or ticket <= 0 or opp_ticket <= 0:
        return None
    if opp_handle <= 0:
        return float("inf") if handle > 0 else None
    return float((handle / ticket) / (opp_handle / opp_ticket))


def size_strength(ratio: Optional[float]) -> Optional[float]:
    """0-1 strength of the relative bet size."""
    return None if ratio is None else _clip(ratio - 1.0)


def crossed_key(open_line: Optional[float], line: Optional[float], key_numbers: Sequence[float]) -> bool:
    """True if a spread moved toward the pick across (or onto) a key number: some signed key s with line <= s < open."""
    if open_line is None or line is None:
        return False
    k = np.asarray(key_numbers, dtype=float)
    s = np.concatenate([k, -k])
    return bool(np.any((float(line) <= s) & (s < float(open_line))))


def _filter_rows(market: str, ticket: Optional[float], handle: Optional[float], hours: Optional[float], ctx: Context) -> Tuple[List[Dict[str, str]], List[str]]:
    """Every filter with its verdict, plus the messages of the ones that failed."""
    rows: List[Dict[str, str]] = []
    blocked: List[str] = []

    def add(name: str, ok: Optional[bool], ok_text: str, bad_text: str) -> None:
        rows.append({"Filter": name, "Status": "➖ N/A" if ok is None else "✅ Pass" if ok else "⛔ Fail", "Detail": ok_text if ok in (True, None) else bad_text})
        if ok is False:
            blocked.append(bad_text)

    band = PRICE_BANDS.get(market)
    p = ctx.price
    add("Price band", None if band is None or p is None else bool(band[0] <= p <= band[1]),
        "n/a" if band is None or p is None else f"{p:+.0f} is inside {band[0]:+.0f} to {band[1]:+.0f}",
        "" if band is None or p is None else f"Price {p:+.0f} is outside the {band[0]:+.0f} to {band[1]:+.0f} band: longshot / heavy-favourite splits are mostly noise.")
    add("Ticket share", None if ticket is None else bool(ticket >= MIN_TICKET_SHARE),
        "n/a" if ticket is None else f"{ticket:.0f}% of tickets (needs {MIN_TICKET_SHARE:.0f}%+)",
        "" if ticket is None else f"Only {ticket:.0f}% of tickets are on this side (needs {MIN_TICKET_SHARE:.0f}%+): a handful of bets can't be a whale.")
    clean = None if ticket is None else not (ctx.degenerate or ctx.pair_bad)
    add("Clean splits", clean, "No 0% / 100% readings and the sides add to about 100%",
        "The splits look half-posted (a 0% / 100% reading, or the two sides don't add to about 100%).")
    add("Signal stability", ctx.persistence if (ticket is not None and is_signal(ticket, handle)) else None,
        "Signal held 30+ minutes, or not enough watching time yet",
        "The money signal is new or flickering (it hasn't held for 30 minutes): wait for it to persist.")
    add("Start window", None if hours is None else bool(hours > CLOSE_TO_START_HOURS),
        "n/a" if hours is None else f"{hours:.1f} hours to the start",
        "" if hours is None else f"Only {max(hours, 0.0) * 60:.0f} minutes to the start: late lineup news makes splits unreliable.")
    add("No news flag", not ctx.news, "Not marked as news-driven", "You marked this game as injury / lineup news: moves after news are everyone's money, not sharp money.")
    return rows, blocked


def _dir(toward: float) -> str:
    return "moving toward" if toward > 0 else "moving away from" if toward < 0 else "not moving on"


def _unit(market: str) -> str:
    return "prob. pts" if market == "h2h" else "pts"


def _verdict(met: Optional[bool]) -> str:
    return UNKNOWN if met is None else YES if met else NO


def score_selection(sport: str, market: str, line: Optional[float], ticket: Optional[float], handle: Optional[float],
                    mv: Optional[Movement], manual_move: Optional[float] = None, ctx: Optional[Context] = None) -> Scored:
    """Run the five checks and the filters for one selection. `manual_move` (negative = toward) is used only when no line history exists."""
    ctx = ctx or Context()
    steam_t, early_t, late_t = MOVE_SCALE.get(market, MOVE_SCALE["totals"])
    notes: List[str] = []
    move = mv.move if (mv is not None and mv.has_history) else None
    if move is None and manual_move is not None:
        move = float(manual_move)
        notes.append("Line movement was typed in by hand (no saved line history for this selection).")
    toward = None if move is None else -float(move)

    have = ticket is not None and handle is not None
    opp = ctx.opp if ctx.opp is not None else ((100.0 - ticket, 100.0 - handle) if have else None)
    ratio = relative_bet_size(ticket, handle, opp[0], opp[1]) if have and opp is not None else None
    gap = None if not have else float(handle) - float(ticket)
    gap_s, size_s = (None if not have else money_strength(ticket, handle)), size_strength(ratio)
    money = None if gap_s is None else (gap_s if size_s is None else 0.5 * (gap_s + size_s))

    # --- the five checks: True / False / None (not yet known)
    c1 = None if gap is None else bool(gap >= SIGNAL_GAP)
    c2 = None if ratio is None else bool(ratio >= SIZE_MET)
    c3 = None if toward is None else bool(toward >= steam_t)
    c4 = (False if not (have and is_signal(ticket, handle)) else ctx.persistence)
    early = None if mv is None or mv.early_move is None else bool(-mv.early_move >= early_t)
    late = None if mv is None or mv.late_move is None else bool(mv.in_closing and -mv.late_move >= late_t)
    cfg = SPORT_CONFIG.get(sport)
    crossed = bool(market == "spreads" and cfg is not None and mv is not None and mv.has_history and crossed_key(mv.open_line, line, cfg.key_numbers))
    rlm = None if toward is None or ticket is None else bool(ticket <= RLM_TICKETS_MAX and toward >= steam_t / 2.0)
    tells = [rlm, early, late, crossed if (mv is not None and mv.has_history) else None]
    c5 = True if any(t is True for t in tells) else None if any(t is None for t in tells) else False
    checks = [c1, c2, c3, c4, c5]
    points = int(sum(c is True for c in checks))

    hours = None if mv is None else mv.hours_to_start
    filters, blocked = _filter_rows(market, ticket, handle, hours, ctx)
    against = toward is not None and toward <= -steam_t
    if c1 is None:
        tier = "NO DATA"
        notes.append("No real handle/ticket splits for this selection, so there is nothing to score.")
    elif c1 and c3 and c4 and c5 is True and not against:
        tier = "SHARP"
    elif c1 and points >= 2 and not against:
        tier = "LEAN"
    else:
        tier = "PASS"
    if tier in ("SHARP", "LEAN") and blocked:
        tier = "FILTERED"
        notes.insert(0, blocked[0])

    needs: List[str] = []
    if tier in ("LEAN", "FILTERED") and c1:
        if not c3:
            needs.append("the line to move toward it" if c3 is False else "line history (the app has not seen it move yet)")
        if not c4:
            needs.append("the signal to hold 30 minutes" if c4 is None else "a steadier signal (it is new or flickering)")
        if c5 is not True:
            needs.append("one market tell (reverse movement, an early / late move, or a key-number cross)")
    if tier == "LEAN" and needs:
        notes.append("To become SHARP it needs: " + "; ".join(needs) + ".")
    if against and c1:
        notes.append("The money is on this side but the line moved the other way: treated as PASS.")

    rows = [
        {"Check": "1. Money gap", "Result": _verdict(c1), "Detail": "no splits" if gap is None else f"Handle {handle:.0f}% vs bets {ticket:.0f}%: gap {gap:+.1f} points (needs {SIGNAL_GAP:.0f}+)"},
        {"Check": "2. Bigger bets", "Result": _verdict(c2), "Detail": "no splits" if ratio is None else f"The average bet on this side is {'many' if np.isinf(ratio) else format(ratio, '.1f')}x the other side's (needs {SIZE_MET}x)"},
        {"Check": "3. Line moved toward it", "Result": _verdict(c3), "Detail": "needs line history" if toward is None else f"Moved {abs(toward):.1f} {_unit(market)} {_dir(toward)} this pick (needs {steam_t:g} toward)"},
        {"Check": "4. Held 30 minutes", "Result": _verdict(c4),
         "Detail": "no money signal right now (check 1 with 15%+ of tickets)" if not (have and is_signal(ticket, handle)) else "not watched for 30 minutes yet" if ctx.persistence is None
         else "the signal has held for 30+ minutes" if ctx.persistence else "the signal is new or flickering"},
        {"Check": "5. A market tell", "Result": _verdict(c5),
         "Detail": "needs line history" if c5 is None and not (mv is not None and mv.has_history) else "; ".join(
             [x for x, ok in (("reverse line movement", rlm), ("early move toward it", early), ("late move toward it", late), ("crossed a key number", crossed)) if ok]) or "none of: reverse movement, early / late move, key-number cross"},
    ]
    return Scored(float(points), tier, money, toward, rows, notes, filters, blocked, ratio, points, needs)


def pair_warning(tickets: Sequence[float], handles: Sequence[float]) -> Optional[str]:
    """For a two-way market, a message when the two sides' splits don't each add up to about 100%."""
    if len(tickets) != 2 or len(handles) != 2:
        return None
    t, h = float(np.sum(tickets)), float(np.sum(handles))
    if abs(t - 100.0) > PAIR_TOLERANCE or abs(h - 100.0) > PAIR_TOLERANCE:
        return f"The two sides' splits add up to {t:.0f}% of bets and {h:.0f}% of handle (should be about 100%): the page may be mid-update."
    return None
