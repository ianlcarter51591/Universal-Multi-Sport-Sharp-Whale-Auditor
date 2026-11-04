"""Universal Sharp & Whale Action Auditor: Streamlit UI.

Run with:  streamlit run app.py
Requires auditor_core.py in the same folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import streamlit as st

from auditor_core import (
    EVENT_ONLY_MARKETS,
    FEATURED_MARKETS,
    SPORT_CONFIG,
    OddsAPIError,
    audit_market,
    checklist_table,
    fetch_event_market,
    fetch_featured_odds,
    format_price,
    game_label,
    market_outcomes,
    outcome_label,
    pick_bookmaker,
)

CSS = """
<style>
.main, [data-testid="stAppViewContainer"] { background-color: #FFFFFF; color: #1A1A1A; }
.stSidebar, [data-testid="stSidebar"] { background-color: #F0F4F8; }
h1, h2, h3 { color: #1A365D; }
</style>
"""


@dataclass(frozen=True)
class SplitInputs:
    """Manual market-split inputs from the sidebar."""

    ticket_pct: float
    handle_pct: float
    line_movement: float
    is_opening_move: bool
    is_closing_window: bool


# --- Cached API wrappers (exceptions are NOT cached by Streamlit) -------------
@st.cache_data(ttl=600, show_spinner="Pulling live odds...")
def cached_featured(
    sport_api_key: str, api_key: str
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Cached bulk odds pull (10 min)."""
    return fetch_featured_odds(sport_api_key, api_key)


@st.cache_data(ttl=600, show_spinner="Pulling team totals...")
def cached_event_market(
    sport_api_key: str, event_id: str, api_key: str, market: str
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Cached single-event market pull (costs extra API credits)."""
    return fetch_event_market(sport_api_key, event_id, api_key, market)


def default_api_key() -> str:
    """Read ODDS_API_KEY from Streamlit secrets if a secrets file exists."""
    try:
        return str(st.secrets.get("ODDS_API_KEY", ""))
    except Exception:  # no secrets.toml configured
        return ""


def render_sidebar() -> tuple[str, str, SplitInputs]:
    """Draw the sidebar and return (league name, api key, split inputs)."""
    with st.sidebar:
        st.header("⚙️ Configuration & Feed")
        api_key = st.text_input(
            "The Odds API Key", value=default_api_key(), type="password"
        )
        league = st.selectbox("Select League / Sport", list(SPORT_CONFIG.keys()))

        st.divider()
        st.markdown("### 📊 Market Split Parameters")
        st.caption("Enter these from your splits source for the selection below.")
        ticket_pct = st.slider(
            "Public Ticket % (% of Bets)", min_value=0.0, max_value=100.0, value=30.0
        )
        handle_pct = st.slider(
            "Public Handle % (% of Money)", min_value=0.0, max_value=100.0, value=85.0
        )
        line_movement = st.number_input(
            "Net Line Movement (pts from open)",
            value=-1.0,
            step=0.5,
            help="Negative = market moved TOWARD this selection; positive = away.",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            is_opening_move = st.checkbox("Early Opener Action")
        with col_b:
            is_closing_window = st.checkbox("Closing Window / Limit Time")

    return (
        league,
        api_key,
        SplitInputs(
            ticket_pct, handle_pct, line_movement, is_opening_move, is_closing_window
        ),
    )


def market_label(market: str) -> str:
    """Selector label; flags markets that cost an extra API call."""
    return f"{market} (extra API credit)" if market in EVENT_ONLY_MARKETS else market


def main() -> None:
    """Page entry point."""
    st.set_page_config(
        page_title="Universal Sharp & Whale Action Auditor",
        page_icon="⚡",
        layout="wide",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("⚡ Universal Sharp & Whale Action Auditor")
    st.markdown(
        "Scan live odds across **Spreads, Totals, Moneylines, and Team Totals** "
        "for major North American leagues and European football competitions."
    )

    league, api_key, inputs = render_sidebar()
    cfg = SPORT_CONFIG[league]

    if not api_key:
        st.warning(
            "⚠️ Please enter your Odds API Key in the sidebar or configure it in "
            "your Streamlit secrets."
        )
        return

    try:
        games, remaining = cached_featured(cfg.api_key, api_key)
    except OddsAPIError as exc:
        st.error(str(exc))
        return
    if remaining is not None:
        st.sidebar.caption(f"API credits remaining: {remaining}")

    if not games:
        st.info(
            f"No upcoming games found for {league}. Check API quota or schedule."
        )
        return

    st.subheader(f"Upcoming Live Markets: {league}")
    game_idx = st.selectbox(
        "Select Matchup to Audit",
        range(len(games)),
        format_func=lambda i: game_label(games[i]),
    )
    game = games[game_idx]

    book = pick_bookmaker(game.get("bookmakers", []))
    if book is None:
        st.error("No active bookmaker odds available for this game.")
        return
    st.caption(
        f"Active Book Feed: **{book['title']}** | Last Updated: "
        f"{book.get('last_update', 'N/A')}"
    )

    market_dict = market_outcomes(book)
    markets = [m for m in FEATURED_MARKETS if m in market_dict] + list(
        EVENT_ONLY_MARKETS
    )
    market_type = st.selectbox("Select Market Type", markets, format_func=market_label)

    if market_type in EVENT_ONLY_MARKETS:
        try:
            ev_books, _ = cached_event_market(
                cfg.api_key, game["id"], api_key, market_type
            )
        except OddsAPIError as exc:
            st.error(str(exc))
            return
        ev_book = pick_bookmaker(ev_books)
        outcomes = market_outcomes(ev_book).get(market_type, []) if ev_book else []
        if not outcomes:
            st.warning(f"No {market_type} currently posted for this game.")
            return
    else:
        outcomes = market_dict[market_type]

    chosen_idx = st.selectbox(
        "Select Specific Selection",
        range(len(outcomes)),
        format_func=lambda i: outcome_label(outcomes[i]),
    )
    chosen = outcomes[chosen_idx]
    line_val = chosen.get("point")
    odds_price = chosen.get("price")

    result = audit_market(
        league,
        market_type,
        line_val,
        inputs.ticket_pct,
        inputs.handle_pct,
        inputs.line_movement,
        inputs.is_opening_move,
        inputs.is_closing_window,
    )

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Auditor Verdict",
            value="🔥 WHALE / SHARP PLAY" if result.is_sharp else "🚫 PASS",
            delta=f"{result.passed}/{result.applicable} Criteria Met",
            delta_color="off",
        )
    with col2:
        st.metric(
            label="Handle vs. Ticket Gap",
            value=f"{inputs.handle_pct:.0f}% Handle",
            delta=f"{inputs.handle_pct - inputs.ticket_pct:+.1f}% vs Public Bets",
        )
    with col3:
        st.metric(
            label="Current Line & Price",
            value=format_price(odds_price),
            delta=f"Line/Spread: {line_val if line_val is not None else 'N/A'}",
            delta_color="off",
        )

    st.markdown("### 📋 Universal Whale Checklist Breakdown")
    st.table(
        checklist_table(
            result,
            league,
            inputs.ticket_pct,
            inputs.handle_pct,
            inputs.line_movement,
            inputs.is_opening_move,
            inputs.is_closing_window,
        )
    )


if __name__ == "__main__":
    main()
