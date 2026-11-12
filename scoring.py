"""
Weighted sharp score (replaces "count the checks"). Pure functions, no Streamlit.

Math
----
The six checks overlap (a big handle gap, steam and reverse movement mostly measure the same event), so they are grouped
and each group counts once. Every group value is a strength, not a yes/no:
  money      m = clip((gap - 5) / 30, 0, 1)           gap = handle% - tickets%; 5 pts = nothing, 35 pts = full strength
  reaction   toward = -move (pts, positive = line moved toward the pick), steam threshold T for the market
             steam = clip(toward / 2T, 0, 1) if toward > 0 else -clip(-toward / 2T, 0, 1)    (moving AWAY is negative)
             rlm   = clip((50 - tickets%) / 20, 0, 1) * clip(toward / T, 0, 1)               (line moves toward the minority)
             r = max(steam, rlm)
  timing     t = 1 if the early move (>= early threshold, toward) or the late move (>= late threshold, toward) happened
  structure  k = 1 if a spread sits on / within a hook of a key number (spreads only)
  score = 100 * sum(w_g * v_g) / sum(w_g over groups that can be measured), clipped to 0-100,
  w = money .40, reaction .35, timing .15, structure .10   (assumptions; the signal log shows whether they should change).
Groups that cannot be measured (no line history yet, no key numbers for the market) are left out and the rest re-weighted.
Tier:  NO DATA  no real splits (the app never invents numbers);
       SHARP    score >= 60, money >= 0.30 and the market confirmed it (reaction > 0);
       LEAN     score >= 40 and money >= 0.20;     otherwise PASS.
Without line history the best possible tier is LEAN: money alone is not confirmation.
Pair rule: in a two-way market the sides' splits mirror each other, so the whale side is the one with the positive gap; the
other side's money group is 0. `pair_warning` flags splits that don't add up to ~100%.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from auditor_core import SPORT_CONFIG, key_number_hit
from line_history import MOVE_SCALE, Movement

W_MONEY, W_REACTION, W_TIMING, W_STRUCTURE = 0.40, 0.35, 0.15, 0.10
GAP_FLOOR, GAP_FULL = 5.0, 35.0
WHALE_GAP_PCT = 15.0                 # the old "whale divergence" line, still used for the Met / Failed label
RLM_PUBLIC_LOW = 30.0
SHARP_SCORE, LEAN_SCORE = 60.0, 40.0
SHARP_MIN_MONEY, LEAN_MIN_MONEY = 0.30, 0.20
PAIR_TOLERANCE = 4.0
TIER_RANK: Dict[str, int] = {"SHARP": 3, "LEAN": 2, "PASS": 1, "NO DATA": 0}
GROUP_LABEL = {"money": f"Money ({W_MONEY:.0%})", "reaction": f"Market reaction ({W_REACTION:.0%})",
               "timing": f"Timing ({W_TIMING:.0%})", "structure": f"Structure ({W_STRUCTURE:.0%})"}


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


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(x, lo, hi))


def money_strength(ticket: float, handle: float) -> float:
    """0-1 strength of the handle-over-tickets gap."""
    return _clip((handle - ticket - GAP_FLOOR) / (GAP_FULL - GAP_FLOOR))


def reaction_parts(ticket: float, toward: float, steam_thr: float) -> Dict[str, float]:
    """Steam (signed) and reverse-line-movement strengths for a move toward the pick of `toward` points."""
    steam = _clip(toward / (2.0 * steam_thr)) if toward > 0 else -_clip(-toward / (2.0 * steam_thr))
    rlm = _clip((50.0 - ticket) / 20.0) * _clip(toward / steam_thr)
    return {"steam": steam, "rlm": rlm}


def score_selection(sport: str, market: str, line: Optional[float], ticket: Optional[float], handle: Optional[float],
                    mv: Optional[Movement], manual_move: Optional[float] = None) -> Scored:
    """Score one selection. `manual_move` (legacy sign, negative = toward) is used only when no line history exists."""
    steam_t, early_t, late_t = MOVE_SCALE.get(market, MOVE_SCALE["totals"])
    notes: List[str] = []
    move = mv.move if (mv is not None and mv.has_history) else None
    if move is None and manual_move is not None:
        move = float(manual_move)
        notes.append("Line movement was typed in by hand (no saved line history for this selection).")
    toward = None if move is None else -float(move)

    money = None if ticket is None or handle is None else money_strength(ticket, handle)
    parts = None if toward is None else reaction_parts(float(ticket) if ticket is not None else 50.0, toward, steam_t)
    reaction = None if parts is None else (parts["steam"] if parts["steam"] < 0 else max(parts["steam"], parts["rlm"]))   # moving away is never hidden by rlm = 0
    early = None if mv is None or mv.early_move is None else bool(-mv.early_move >= early_t)
    late = None if mv is None or mv.late_move is None else bool(mv.in_closing and -mv.late_move >= late_t)
    timing = None if early is None and late is None else float(bool(early) or bool(late))
    cfg = SPORT_CONFIG.get(sport)
    structure = None
    if market == "spreads" and line is not None and cfg is not None:
        structure = float(key_number_hit(line, cfg.key_numbers, cfg.key_tolerance))

    groups = {"money": (money, W_MONEY), "reaction": (reaction, W_REACTION), "timing": (timing, W_TIMING), "structure": (structure, W_STRUCTURE)}
    live = {k: (v, w) for k, (v, w) in groups.items() if v is not None}
    score = 100.0 * sum(v * w for v, w in live.values()) / sum(w for _, w in live.values()) if live else 0.0
    score = _clip(score, 0.0, 100.0)

    if money is None:
        tier = "NO DATA"
        notes.append("No real handle/ticket splits for this selection, so there is nothing to score.")
    elif score >= SHARP_SCORE and money >= SHARP_MIN_MONEY and reaction is not None and reaction > 0:
        tier = "SHARP"
    elif score >= LEAN_SCORE and money >= LEAN_MIN_MONEY:
        tier = "LEAN"
    else:
        tier = "PASS"
    if money is not None and reaction is None:
        notes.append("No line history yet, so the market can't confirm the money: the best possible tier is LEAN.")
    if money is not None and reaction is not None and money >= LEAN_MIN_MONEY and reaction < 0:
        notes.append("Money says one thing and the line moved the other way: treat with caution.")

    gap = None if ticket is None or handle is None else handle - ticket
    rows = [
        _row("1. Ticket % vs Handle % divergence", "money", None if gap is None else gap >= WHALE_GAP_PCT, money,
             "no splits" if gap is None else f"Handle {handle:.0f}% vs bets {ticket:.0f}%: gap {gap:+.1f} pts (full strength at {GAP_FULL:.0f})"),
        _row("2. Reverse line movement", "reaction", None if parts is None or ticket is None else bool(
            ticket <= RLM_PUBLIC_LOW and toward > 0),                    # the public-side mirror is judged on the other side
             None if parts is None else parts["rlm"],
             "needs line history" if parts is None else f"{'n/a' if ticket is None else format(ticket, '.0f')}% of bets with the line {_dir(toward)} this pick"),
        _row("3. Steam move", "reaction", None if toward is None else bool(toward >= steam_t),
             None if parts is None else parts["steam"],
             "needs line history" if toward is None else f"Line moved {abs(toward):.1f} {_unit(market)} {_dir(toward)} this pick (steam = {steam_t:g})"),
        _row("4. Early opener action", "timing", early, None if early is None else float(early),
             "needs two readings 24h+ before the game" if early is None else f"Moved {-mv.early_move:+.1f} {_unit(market)} toward the pick while 24h+ out"),
        _row("5. Late market limit steam", "timing", late, None if late is None else float(late),
             "not measurable yet" if late is None else ("Inside the closing window" if mv and mv.in_closing else "Not inside the closing window")),
        _row("6. Key number resistance", "structure", None if structure is None else bool(structure), structure,
             "spreads only" if structure is None else "Spread sits on or beside a key number for this sport"),
    ]
    return Scored(score, tier, money, reaction, timing, structure, toward, rows, notes)


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
