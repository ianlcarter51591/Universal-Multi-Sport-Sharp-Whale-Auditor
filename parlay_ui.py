"""
Parlay builder for the Auditor: pick sides from the board, set your own chance for each, price the ticket.

Math
----
This app has no player model, so a leg's default chance is the market's NO-VIG probability of the side:
  p_i = implied(price_i) / sum over the market's sides of implied(price_j)        (multiplicative devig, per game and market)
You can overwrite it with your own number when you think the whale side is better than the price says.
Everything else (copula for same-game legs, boost, EV, Kelly) is the shared `parlay.py`.
At the market's own probabilities a parlay is always -EV by about the vig of each leg; a play needs real edge in your numbers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import bet_links as bl
import parlay as pl
from line_history import implied_prob

POOL_COLUMNS = ["LABEL", "GAME", "WHEN", "TIER", "CHECKS", "P_MARKET", "ODDS", "LINK", "LINK_KIND"]


def legs_from_board(all_sides: pd.DataFrame) -> pd.DataFrame:
    """Every scanned selection as a pickable leg with its no-vig market probability (in [0, 1])."""
    if all_sides is None or all_sides.empty:
        return pd.DataFrame(columns=POOL_COLUMNS)
    s = all_sides.copy()
    s["_imp"] = s["PRICE"].map(implied_prob)
    s = s[s["_imp"].notna()]
    total = s.groupby(["EVENT_ID", "MARKET"])["_imp"].transform("sum")
    s["P_MARKET"] = s["_imp"] / total
    s["LABEL"] = s["GAME"] + " | " + s["PICK"]
    if "LINK" not in s.columns:
        s["LINK"], s["LINK_KIND"] = bl.DK_HOME, "home"
    out = s.rename(columns={"PRICE": "ODDS"})
    out = out[(out["P_MARKET"] > 0) & (out["P_MARKET"] < 1)]
    return out[POOL_COLUMNS].drop_duplicates("LABEL").reset_index(drop=True)


def _num(x: Any, default: float) -> float:
    """float(x) or the default (widgets can return None)."""
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _odds_text(x: Optional[float]) -> str:
    """American odds as text, e.g. +581 / -110."""
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.0f}"


def _legs(st: Any) -> List[Dict[str, Any]]:
    ss = st.session_state
    if "parlay_legs" not in ss:
        ss["parlay_legs"] = []
    return ss["parlay_legs"]


def add_leg(st: Any, label: str, p_model: float, odds: Optional[float], p_market: Optional[float], game: str,
            link: Optional[str] = None, kind: str = "home") -> bool:
    """Append a leg unless the same label is already on the ticket."""
    legs = _legs(st)
    if not label or any(l["label"] == label for l in legs):
        return False
    legs.append({"label": label, "p_model": float(p_model), "odds": odds, "p_market": p_market, "game": str(game or "").strip(),
                 "link": link or bl.DK_HOME, "kind": kind if link else "home"})
    return True


def render_slip(st: Any, legs: List[Dict[str, Any]]) -> None:
    """DraftKings buttons for the ticket: one link per leg, plus a button that opens them all. The app never places the bet."""
    st.markdown("**Take it to DraftKings**")
    links = [(l.get("link") or bl.DK_HOME, l.get("kind", "home")) for l in legs]
    for l, (url, kind) in zip(legs, links):
        st.markdown(f"- {l['label']}: [Bet on DraftKings ↗]({url}) ({bl.KIND_LABEL.get(kind, kind)})")
    n_real = bl.real_links(links)
    try:
        from streamlit.components.v1 import html as _components_html
        _components_html(bl.open_all_html([u for u, _ in links], f"➕ Add slip: open all {len(links)} legs"), height=60)
    except Exception:
        pass       # no components support (tests): the per-leg links above still work
    st.caption(f"{n_real} of {len(links)} legs have a DraftKings link that adds the pick to your bet slip or opens the market; "
               "the rest open the DraftKings sportsbook. Each link opens in its own tab (allow pop-ups for this site once). DraftKings decides "
               "what the link does, so check the slip: the price and line there are DraftKings' current ones, and you confirm the bet yourself.")


def render_parlay(st: Any, all_sides: Optional[pd.DataFrame]) -> None:
    """Build a ticket from the scanned board or by hand, then price it."""
    ss = st.session_state
    legs = _legs(st)
    st.markdown("Pick sides from the board, set your own chance for each, and price the ticket. Legs from the same game are adjusted for "
                "correlation (for example a favourite's spread and the under often fight each other).")
    pool = legs_from_board(all_sides) if all_sides is not None else pd.DataFrame(columns=POOL_COLUMNS)
    if not pool.empty:
        show = st.selectbox("Show", ["LEAN and SHARP", "Everything"], key="parlay_show")
        view = pool if show == "Everything" else pool[pool["TIER"].isin(["LEAN", "SHARP"])]
        if view.empty:
            st.info("No LEAN or SHARP sides right now. Choose Everything to pick any side.")
        else:
            pick = st.selectbox("Add a leg from the board", view["LABEL"].tolist(), key="parlay_pick")
            r = view[view["LABEL"] == pick].iloc[0]
            c1, c2, c3 = st.columns([2, 2, 1])
            c1.caption(f"{r['WHEN']} · {r['TIER']} · checks {r['CHECKS']} · market no-vig chance {r['P_MARKET'] * 100:.1f}%")
            mine = c2.number_input("Your chance %", min_value=1.0, max_value=99.0, value=round(float(r["P_MARKET"]) * 100.0, 1), step=0.5,
                                   key=f"parlay_mine_{pick}")
            c3.write("")
            if c3.button("➕ Add leg", key="parlay_add"):
                ok = add_leg(st, r["LABEL"], _num(mine, 50.0) / 100.0, None if pd.isna(r["ODDS"]) else float(r["ODDS"]),
                             float(r["P_MARKET"]), r["GAME"], r["LINK"], r["LINK_KIND"])
                (st.toast if ok else st.warning)("Leg added" if ok else "That leg is already on the ticket.")
    else:
        st.caption("No scanned games yet. Add legs by hand below.")
    with st.expander("Add a leg by hand"):
        m1, m2, m3, m4 = st.columns([3, 1, 1, 2])
        label = m1.text_input("Leg", key="parlay_m_label", placeholder="Boston Celtics -3.5")
        pm = m2.number_input("Your chance %", min_value=1.0, max_value=99.0, value=52.0, step=1.0, key="parlay_m_p")
        od = m3.number_input("Leg odds (0 = none)", value=0, step=5, key="parlay_m_odds")
        game = m4.text_input("Game (same text = same game)", key="parlay_m_game", placeholder="BOS @ MIA")
        if st.button("➕ Add this leg", key="parlay_m_add"):
            o = _num(od, 0.0)
            ok = add_leg(st, str(label or "").strip(), _num(pm, 52.0) / 100.0, o if abs(o) >= 100 else None, None, str(game or ""))
            (st.toast if ok else st.warning)("Leg added" if ok else "Give the leg a name that isn't already on the ticket.")
    if not legs:
        st.info("Your ticket is empty.")
        return
    st.markdown("**Your ticket**")
    st.dataframe(pd.DataFrame([{"LEG": l["label"], "GAME": l["game"], "YOUR_CHANCE_%": round(l["p_model"] * 100.0, 1),
                                "MARKET_%": None if l["p_market"] is None else round(l["p_market"] * 100.0, 1),
                                "ODDS": l["odds"]} for l in legs]), use_container_width=True, hide_index=True)
    r1, r2 = st.columns([4, 1])
    drop = r1.selectbox("Remove a leg", [l["label"] for l in legs], key="parlay_drop")
    r2.write("")
    if r2.button("🗑️ Remove", key="parlay_remove"):
        ss["parlay_legs"] = [l for l in legs if l["label"] != drop]
        st.rerun()
    if st.button("Clear ticket", key="parlay_clear"):
        ss["parlay_legs"] = []
        st.rerun()
    render_slip(st, legs)
    p1, p2, p3 = st.columns(3)
    offered = _num(p1.number_input("Parlay odds offered, before any boost (0 = multiply the leg odds)", value=0, step=5, key="parlay_offered"), 0.0)
    boost = _num(p2.number_input("Boost %", min_value=0.0, max_value=200.0, value=0.0, step=5.0, key="parlay_boost"), 0.0)
    rho = _num(p3.slider("Same-game correlation", -0.5, 0.8, 0.0, 0.05, key="parlay_rho"), 0.0)
    res = pl.evaluate([pl.Leg(l["label"], l["p_model"], l["odds"], l["p_market"], l["game"]) for l in legs],
                      offered if abs(offered) >= 100 else None, boost, rho)
    if "error" in res:
        st.warning(res["error"])
        return
    for col, lab, value in zip(st.columns(5), ("ALL LEGS HIT", "BREAK-EVEN", "EV PER UNIT", "PRICE YOU GET", "FAIR PRICE"),
                               (f"{res['p_joint'] * 100:.1f}%", f"{res['breakeven'] * 100:.1f}%", f"{res['ev_pct']:+.1f}%",
                                _odds_text(res["boosted_american"]), _odds_text(res["fair_american"]))):
        with col:
            st.metric(lab, value)
    verdict = (f"**{res['verdict']}**: your legs hit together {res['p_joint'] * 100:.1f}% of the time by your numbers; the price needs "
               f"{res['breakeven'] * 100:.1f}%. Suggested stake {res['kelly_pct']:.2f}% of bankroll (quarter Kelly, capped at 5%).")
    (st.success if res["verdict"] == "PLAY" else st.warning)(verdict)
    notes = [f"Price used: {res['price_source']} ({_odds_text(res['base_american'])})" + (f", boosted {boost:g}% to {_odds_text(res['boosted_american'])}." if boost else ".")]
    if np.isfinite(res["p_market"]):
        notes.append(f"The market's own no-vig chance for these legs is {res['p_market'] * 100:.1f}%.")
    if res["same_game"]:
        notes.append("Some legs share a game, so the answer depends on the correlation you set. The table shows how much.")
    st.caption(" ".join(notes))
    if res["same_game"] and not res["sensitivity"].empty:
        st.dataframe(res["sensitivity"].round(2), use_container_width=True, hide_index=True)
    st.caption("Limits: this app has no player or team model, so each leg starts at the market's no-vig chance and a parlay of those is -EV by "
               "the vig. It only shows PLAY when YOUR chances beat the price, so only raise a leg's chance when you have a real reason (for example a "
               "SHARP signal that held). The correlation is your assumption. Books price same-game parlays below the product of the leg odds, so "
               "enter the price the book actually offers.")
