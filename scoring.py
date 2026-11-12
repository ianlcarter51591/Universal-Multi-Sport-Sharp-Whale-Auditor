"""
Two-layer sharp score. Pure functions, no Streamlit.

Layer 1: noise FILTERS (a pick that would otherwise be LEAN / SHARP but fails any filter becomes FILTERED)
  price band        h2h -300..+250, spreads / totals -130..+110      (kills the +5000 "whale on a longshot")
  ticket share      the pick needs >= 15% of the tickets             (a few bets can't be a whale)
  clean splits      no 0% / 100% reading, two-way sides add to ~100% (half-posted or placeholder pages)
  signal stability  money signal held >= 30 min (split_history.persistence) when we have watched that long
  start window      more than 30 minutes before the start
  news              the game is not marked "injury / lineup news"    (news moves are everyone's money, not sharp money)

Layer 2: evidence, grouped so overlapping checks count once. Every group value is a strength in [0, 1] (reaction -1..1):
  money      m = mean(gap_strength, size_strength)
             gap_strength  = clip((gap - 5) / 30, 0, 1),            gap = handle% - tickets%
             size_strength = clip((R - 1) / (2 - 1), 0, 1),         R = (handle/tickets of this side) / (handle/tickets of the others)
                             = how many times bigger the average bet is on this side (R >= 1.5 counts as "met")
  reaction   toward = -move (positive = line moved toward the pick), T = steam threshold of the market
             steam = clip(toward / 2T, 0, 1), negative when the line moved AWAY;  rlm = clip((50 - tickets%) / 20, 0, 1) * clip(toward / T, 0, 1)
             r = steam if steam < 0 else max(steam, rlm)
  timing     1 if an early (24h+ out) or late (closing window) move >= its threshold went toward the pick
  structure  spreads: 1.0 if the move crossed or reached a key number toward the pick, 0.3 if the line merely sits on one, else 0
  score = 100 * sum(w_g * v_g) / sum(w_g of groups that can be measured), clipped to 0-100;
  w = money .40, reaction .35, timing .15, structure .10 (assumptions: the signal log is how to test them).
Tier: NO DATA (no real splits) | FILTERED | SHARP (score >= 60, money >= .30, reaction > 0, signal held >= 30 min, and one of
      reverse movement / timing / key-number crossing) | LEAN (score >= 40, money >= .20) | PASS.
Two-way markets mirror each other, so only the side with the positive gap earns money credit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from auditor_core import SPORT_CONFIG, key_number_hit
from line_history import MOVE_SCALE, Movement
from split_history import MIN_TICKET_SHARE, SIGNAL_GAP, is_signal

W_MONEY, W_REACTION, W_TIMING, W_STRUCTURE = 0.40, 0.35, 0.15, 0.10
GAP_FLOOR, GAP_FULL = 5.0, 35.0
SIZE_FLOOR, SIZE_FULL, SIZE_MET = 1.0, 2.0, 1.5
RLM_PUBLIC_LOW = 30.0
SHARP_SCORE, LEAN_SCORE = 60.0, 40.0
SHARP_MIN_MONEY, LEAN_MIN_MONEY = 0.30, 0.20
SIT_ON_KEY = 0.3
CLOSE_TO_START_HOURS = 0.5
PAIR_TOLERANCE = 4.0
PRICE_BANDS: Dict[str, Tuple[float, float]] = {"h2h": (-300.0, 250.0), "spreads": (-130.0, 110.0), "totals": (-130.0, 110.0)}
TIER_RANK: Dict[str, int] = {"SHARP": 4, "FILTERED": 3, "LEAN": 3, "PASS": 2, "NO DATA": 0}      # a filtered whale side outranks the quiet side of its market
GROUP_LABEL = {"money": f"Money ({W_MONEY:.0%})", "reaction": f"Market reaction ({W_REACTION:.0%})", "timing": f"Timing ({W_TIMING:.0%})",
               "structure": f"Structure ({W_STRUCTURE:.0%})", "gate": "Needed for SHARP"}


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
    """Result for one selection. Group values are strengths (reaction can be negative); None = could not be measured."""

    score: float
    tier: str
    money: Optional[float]
    reaction: Optional[float]
    timing: Optional[float]
    structure: Optional[float]
    toward: Optional[float]                   # line move toward the pick (positive = toward), None without history
    rows: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    filters: List[Dict[str, str]] = field(default_factory=list)
    blocked: List[str] = field(default_factory=list)
    bet_size: Optional[float] = None          # R, how many times bigger this side's average bet is than the other side's


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
    return None if ratio is None else _clip((ratio - SIZE_FLOOR) / (SIZE_FULL - SIZE_FLOOR))


def reaction_parts(ticket: float, toward: float, steam_thr: float) -> Dict[str, float]:
    """Steam (signed) and reverse-line-movement strengths for a move toward the pick of `toward` points."""
    steam = _clip(toward / (2.0 * steam_thr)) if toward > 0 else -_clip(-toward / (2.0 * steam_thr))
    rlm = _clip((50.0 - ticket) / 20.0) * _clip(toward / steam_thr)
    return {"steam": steam, "rlm": rlm}


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


def score_selection(sport: str, market: str, line: Optional[float], ticket: Optional[float], handle: Optional[float],
                    mv: Optional[Movement], manual_move: Optional[float] = None, ctx: Optional[Context] = None) -> Scored:
    """Score one selection. `manual_move` (legacy sign, negative = toward) is used only when no line history exists."""
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
    gap_s = None if not have else money_strength(ticket, handle)
    size_s = size_strength(ratio)
    money = None if gap_s is None else (gap_s if size_s is None else 0.5 * (gap_s + size_s))

    parts = None if toward is None else reaction_parts(float(ticket) if ticket is not None else 50.0, toward, steam_t)
    reaction = None if parts is None else (parts["steam"] if parts["steam"] < 0 else max(parts["steam"], parts["rlm"]))   # moving away is never hidden by rlm = 0
    early = None if mv is None or mv.early_move is None else bool(-mv.early_move >= early_t)
    late = None if mv is None or mv.late_move is None else bool(mv.in_closing and -mv.late_move >= late_t)
    timing = None if early is None and late is None else float(bool(early) or bool(late))
    cfg = SPORT_CONFIG.get(sport)
    structure, crossed = None, False
    if market == "spreads" and line is not None and cfg is not None:
        crossed = crossed_key(mv.open_line if mv is not None and mv.has_history else None, line, cfg.key_numbers)
        structure = 1.0 if crossed else SIT_ON_KEY if key_number_hit(line, cfg.key_numbers, cfg.key_tolerance) else 0.0

    groups = {"money": (money, W_MONEY), "reaction": (reaction, W_REACTION), "timing": (timing, W_TIMING), "structure": (structure, W_STRUCTURE)}
    live = {k: (v, w) for k, (v, w) in groups.items() if v is not None}
    score = _clip(100.0 * sum(v * w for v, w in live.values()) / sum(w for _, w in live.values()) if live else 0.0, 0.0, 100.0)

    hours = None if mv is None else mv.hours_to_start
    filters, blocked = _filter_rows(market, ticket, handle, hours, ctx)
    extras = [bool(parts and parts["rlm"] > 0), bool(timing), bool(crossed)]
    if money is None:
        tier = "NO DATA"
        notes.append("No real handle/ticket splits for this selection, so there is nothing to score.")
    elif score >= SHARP_SCORE and money >= SHARP_MIN_MONEY and reaction is not None and reaction > 0 and ctx.persistence is True and any(extras):
        tier = "SHARP"
    elif score >= LEAN_SCORE and money >= LEAN_MIN_MONEY:
        tier = "LEAN"
    else:
        tier = "PASS"
    if tier in ("SHARP", "LEAN") and blocked:
        tier = "FILTERED"
        notes.insert(0, blocked[0])
    if money is not None and reaction is None:
        notes.append("No line history yet, so the market can't confirm the money: the best possible tier is LEAN.")
    if money is not None and reaction is not None and money >= LEAN_MIN_MONEY and reaction < 0:
        notes.append("Money says one thing and the line moved the other way: treat with caution.")
    if tier == "LEAN" and money is not None and money >= SHARP_MIN_MONEY and reaction is not None and reaction > 0 and not (ctx.persistence is True and any(extras)):
        notes.append("Looks sharp, but it needs the signal to hold 30 minutes and one more confirmation (reverse movement, timing or a key-number cross) for SHARP.")

    gap = None if not have else handle - ticket
    rows = [
        _row("1. Ticket % vs Handle % divergence", "money", None if gap is None else gap >= SIGNAL_GAP, gap_s,
             "no splits" if gap is None else f"Handle {handle:.0f}% vs bets {ticket:.0f}%: gap {gap:+.1f} pts (full strength at {GAP_FULL:.0f})"),
        _row("2. Relative bet size", "money", None if ratio is None else bool(ratio >= SIZE_MET), size_s,
             "no splits" if ratio is None else f"Average bet on this side is {'many' if np.isinf(ratio) else format(ratio, '.1f')}x the other side's (met at {SIZE_MET}x)"),
        _row("3. Steam toward the money", "reaction", None if toward is None else bool(toward >= steam_t), None if parts is None else parts["steam"],
             "needs line history" if toward is None else f"Line moved {abs(toward):.1f} {_unit(market)} {_dir(toward)} this pick (steam = {steam_t:g})"),
        _row("4. Reverse line movement", "reaction", None if parts is None or ticket is None else bool(ticket <= RLM_PUBLIC_LOW and toward > 0),
             None if parts is None else parts["rlm"],
             "needs line history" if parts is None else f"{'n/a' if ticket is None else format(ticket, '.0f')}% of bets with the line {_dir(toward)} this pick"),
        _row("5. Signal persistence", "gate", ctx.persistence if have and is_signal(ticket, handle) else None,
             None if ctx.persistence is None else float(ctx.persistence),
             "no money signal now, or not watched for 30 minutes yet" if ctx.persistence is None else "Signal has held for 30+ minutes" if ctx.persistence else "Signal is new or flickering"),
        _row("6. Early opener action", "timing", early, None if early is None else float(early),
             "needs two readings 24h+ before the game" if early is None else f"Moved {-mv.early_move:+.1f} {_unit(market)} toward the pick while 24h+ out"),
        _row("7. Late market limit steam", "timing", late, None if late is None else float(late),
             "not measurable yet" if late is None else ("Inside the closing window" if mv and mv.in_closing else "Not inside the closing window")),
        _row("8. Key number crossing", "structure", None if structure is None else bool(crossed), structure,
             "spreads only" if structure is None else "Moved across or onto a key number toward the pick" if crossed
             else "Sits on a key number but did not move across one" if structure else "Not near a key number"),
    ]
    return Scored(score, tier, money, reaction, timing, structure, toward, rows, notes, filters, blocked, ratio)


def _dir(toward: float) -> str:
    return "moving toward" if toward > 0 else "moving away from" if toward < 0 else "not moving on"


def _unit(market: str) -> str:
    return "prob. pts" if market == "h2h" else "pts"


def _row(name: str, group: str, met: Optional[bool], strength: Optional[float], detail: str) -> Dict[str, Any]:
    status = "➖ N/A" if met is None else "✅ Met" if met else "❌ Failed"
    return {"Criterion": name, "Status": status, "Strength %": None if strength is None else round(100.0 * strength, 0),
            "Counts in": GROUP_LABEL[group], "Detail": detail}


def pair_warning(tickets: Sequence[float], handles: Sequence[float]) -> Optional[str]:
    """For a two-way market, a message when the two sides' splits don't each add up to about 100%."""
    if len(tickets) != 2 or len(handles) != 2:
        return None
    t, h = float(np.sum(tickets)), float(np.sum(handles))
    if abs(t - 100.0) > PAIR_TOLERANCE or abs(h - 100.0) > PAIR_TOLERANCE:
        return f"The two sides' splits add up to {t:.0f}% of bets and {h:.0f}% of handle (should be about 100%): the page may be mid-update."
    return None
