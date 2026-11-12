"""
Parlay / same-game-parlay evaluator (shared by the NBA and NFL apps).

Math
----
Decimal price of American odds o:   d = 1 + 100/|o| if o < 0 else 1 + o/100.
Independent legs:                   P(all hit) = prod(p_i);  a standard parlay pays prod(d_i).
Same-game legs are NOT independent. They are joined with a Gaussian copula: draw z ~ N(0, C) and call leg i a hit when
    z_i < Phi^-1(p_i).  C has 1 on the diagonal, rho between two legs of the same game, 0 between different games.
    rho > 0 means the legs tend to hit together (e.g. a QB's passing yards over and his receiver's yards over);
    rho < 0 means they fight each other. rho is an assumption you set; the result is shown at several rho values so you
    can see how much the answer depends on it. Each leg keeps its own probability exactly; only the overlap changes.
Boost: a "+30% parlay boost" multiplies the PROFIT, so payout_decimal = 1 + (d_offered - 1) * (1 + boost).
EV per unit staked = P * (payout_decimal - 1) - (1 - P).   Break-even probability = 1 / payout_decimal.
Kelly fraction = (b*P - (1 - P)) / b with b = payout_decimal - 1; the apps stake a quarter of it, capped at 5%.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

N_SIMS = 200_000
RHO_GRID: Sequence[float] = (-0.3, 0.0, 0.3, 0.6)
KELLY_FRACTION, KELLY_CAP = 0.25, 0.05


@dataclass(frozen=True)
class Leg:
    """One parlay leg. p_model and p_market are probabilities in [0, 1]; odds are American (None if unknown)."""

    label: str
    p_model: float
    odds: Optional[float] = None
    p_market: Optional[float] = None
    game: str = ""


def to_decimal(odds: float) -> float:
    """American odds -> decimal price."""
    o = float(odds)
    if abs(o) < 100:
        raise ValueError(f"{o:g} is not valid American odds (must be <= -100 or >= +100)")
    return 1.0 + (100.0 / -o if o < 0 else o / 100.0)


def to_american(decimal: float) -> float:
    """Decimal price -> American odds."""
    d = float(decimal)
    if d <= 1.0:
        return float("nan")
    return (d - 1.0) * 100.0 if d >= 2.0 else -100.0 / (d - 1.0)


def corr_matrix(games: Sequence[str], rho: float) -> np.ndarray:
    """Correlation matrix: rho between legs that share a non-empty game tag, 0 otherwise (kept positive-definite)."""
    g = np.array([str(x).strip() for x in games], dtype=object)
    same = (g[:, None] == g[None, :]) & (g[:, None] != "")
    sizes = np.array([int(same[i].sum()) for i in range(len(g))]) if len(g) else np.array([1])
    k = int(sizes.max()) if len(g) else 1
    lo = -1.0 / (k - 1) + 0.02 if k > 1 else -0.99        # an equicorrelated block of k legs needs rho > -1/(k-1)
    r = float(np.clip(rho, lo, 0.98))
    c = np.where(same, r, 0.0).astype(float)
    np.fill_diagonal(c, 1.0)
    return c


def joint_probability(p: Sequence[float], corr: np.ndarray, n_sims: int = N_SIMS, seed: int = 7) -> float:
    """P(every leg hits) under a Gaussian copula. Exact product when the legs are uncorrelated."""
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    if len(p) == 0:
        return float("nan")
    off = corr - np.diag(np.diag(corr))
    if len(p) == 1 or not np.any(np.abs(off) > 1e-12):
        return float(np.prod(p))
    thresh = np.array([NormalDist().inv_cdf(float(x)) for x in p])
    z = np.random.default_rng(seed).standard_normal((n_sims, len(p))) @ np.linalg.cholesky(corr).T
    return float(np.mean(np.all(z < thresh, axis=1)))


def evaluate(legs: Sequence[Leg], offered_odds: Optional[float] = None, boost_pct: float = 0.0, rho: float = 0.0) -> Dict[str, Any]:
    """Price a parlay. `offered_odds`: the book's parlay price before any boost (None = product of the leg prices)."""
    if not legs:
        return {"error": "Add at least one leg."}
    p = np.array([l.p_model for l in legs], dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p <= 0) | (p >= 1)):
        return {"error": "Every leg needs a model probability between 0 and 100%."}
    games = [l.game for l in legs]
    if offered_odds is None or not np.isfinite(float(offered_odds)) or float(offered_odds) == 0:
        if any(l.odds is None for l in legs):
            return {"error": "Enter the parlay's offered odds, or give every leg its own odds."}
        try:
            base_decimal, price_source = float(np.prod([to_decimal(l.odds) for l in legs])), "product of the leg prices"
        except ValueError as e:
            return {"error": str(e)}
    else:
        try:
            base_decimal, price_source = to_decimal(float(offered_odds)), "offered parlay price"
        except ValueError as e:
            return {"error": str(e)}
    payout = 1.0 + (base_decimal - 1.0) * (1.0 + float(boost_pct) / 100.0)
    b = payout - 1.0
    c = corr_matrix(games, rho)
    p_joint, p_indep = joint_probability(p, c), float(np.prod(p))
    ev = p_joint * b - (1.0 - p_joint)
    kelly = max(0.0, (b * p_joint - (1.0 - p_joint)) / b) if b > 0 else 0.0
    pm = [l.p_market for l in legs]
    p_market = joint_probability(pm, c) if all(x is not None and np.isfinite(x) and 0 < x < 1 for x in pm) else float("nan")
    same_game = len(set(g for g in games if str(g).strip())) < len([g for g in games if str(g).strip()])
    grid = sorted(set(list(RHO_GRID) + [round(float(rho), 2)]))
    sens = pd.DataFrame([{"SAME_GAME_CORRELATION": r, "CHANCE_ALL_HIT_%": 100.0 * (pj := joint_probability(p, corr_matrix(games, r))),
                          "EV_PER_UNIT_%": 100.0 * (pj * b - (1.0 - pj))} for r in grid]) if same_game else pd.DataFrame()
    return {
        "n_legs": len(legs), "same_game": same_game, "price_source": price_source,
        "base_american": to_american(base_decimal), "boosted_american": to_american(payout), "payout_decimal": payout,
        "p_joint": p_joint, "p_independent": p_indep, "p_market": p_market, "breakeven": 1.0 / payout,
        "fair_american": to_american(1.0 / p_joint) if p_joint > 0 else float("nan"),
        "ev_pct": ev * 100.0, "kelly_pct": min(kelly * KELLY_FRACTION, KELLY_CAP) * 100.0,
        "verdict": "PLAY" if ev > 0 else "PASS", "sensitivity": sens,
        "legs": pd.DataFrame([{"LEG": l.label, "GAME": l.game, "MODEL_%": l.p_model * 100.0,
                               "MARKET_%": l.p_market * 100.0 if l.p_market is not None else np.nan, "ODDS": l.odds} for l in legs]),
    }


def legs_from_snapshots(snaps: pd.DataFrame) -> pd.DataFrame:
    """Pending snapshots as pickable legs, one row per side: LABEL, GAME, P_MODEL, P_MARKET, ODDS (probabilities in [0, 1])."""
    cols = ["LABEL", "GAME", "P_MODEL", "P_MARKET", "ODDS"]
    need = ["STATUS", "PLAYER", "STAT", "LINE", "P_OVER", "P_UNDER", "OVER_ODDS", "UNDER_ODDS"]
    if snaps is None or snaps.empty or any(c not in snaps.columns for c in need):
        return pd.DataFrame(columns=cols)
    s = snaps[snaps["STATUS"] == "Pending"]
    game = s["GAME"] if "GAME" in s else pd.Series("", index=s.index)
    parts: List[pd.DataFrame] = []
    for side, pc, mc, oc in (("Over", "P_OVER", "PM_OVER", "OVER_ODDS"), ("Under", "P_UNDER", "PM_UNDER", "UNDER_ODDS")):
        line = s["LINE"].map(lambda x: f"{x:g}")
        parts.append(pd.DataFrame({
            "LABEL": s["PLAYER"] + " " + side + " " + line + " " + s["STAT"].astype(str), "GAME": game.fillna("").astype(str),
            "P_MODEL": pd.to_numeric(s[pc], errors="coerce") / 100.0,
            "P_MARKET": pd.to_numeric(s[mc], errors="coerce") / 100.0 if mc in s else np.nan,
            "ODDS": pd.to_numeric(s[oc], errors="coerce")}))
    out = pd.concat(parts, ignore_index=True).dropna(subset=["P_MODEL"])
    return out[(out["P_MODEL"] > 0) & (out["P_MODEL"] < 1)].sort_values("LABEL").reset_index(drop=True)[cols]
