"""Universal Sharp & Whale Action Auditor: Streamlit UI.

Run with:  streamlit run app.py
Requires auditor_core.py in the same folder.

Keys are loaded automatically on open (first match wins):
    1. .streamlit/secrets.toml  (ODDS_API_KEY / SHARP_API_KEY)
    2. environment variables    (ODDS_API_KEY / SHARP_API_KEY)
The sidebar key boxes are optional overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import streamlit as st

from auditor_core import (
    EVENT_ONLY_MARKETS,
    FEATURED_MARKETS,
    SPORT_CONFIG,
    OddsAPIError,
    SharpAPIError,
    SplitReading,
    audit_market,
    checklist_table,
    fetch_event_market,
    fetch_featured_odds,
    fetch_splits,
    format_price,
    game_label,
    market_outcomes,
    match_split_rows,
    outcome_label,
    pick_bookmaker,
    read_split,
)

CSS = """
<style>
.main, [data-testid="stAppViewContainer"] { background-color: #FFFFFF; color: #1A1A1A; }
.stSidebar, [data-testid="stSidebar"] { background-color: #F0F4F8; }
h1, h2, h3 { color: #1A365D; }
</style>
"""


@dataclass(frozen=True)
class ManualInputs:
    """Sidebar inputs: manual split fallback plus timing flags."""

    ticket_pct: float
    handle_pct: float
    force_manual: bool
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


@st.cache_data(ttl=300, show_spinner="Pulling handle / bet splits...")
def cached_splits(league: str, sharp_key: str) -> list[dict[str, Any]]:
    """Cached SharpAPI splits pull (source refreshes about every 5 minutes)."""
    return fetch_splits(league, sharp_key)


def load_secret(name: str) -> str:
    """Read a key from Streamlit secrets, then the environment, else ''."""
    try:
        value = st.secrets.get(name, "")
    except Exception:  # no secrets.toml configured
        value = ""
    return str(value or os.environ.get(name, ""))


def render_sidebar() -> tuple[str, str, str, ManualInputs]:
    """Draw the sidebar; return (league, odds key, sharp key, manual inputs)."""
    odds_default = load_secret("ODDS_API_KEY")
    sharp_default = load_secret("SHARP_API_KEY")

    with st.sidebar:
        st.header("⚙️ Configuration & Feed")
        league = st.selectbox("Select League / Sport", list(SPORT_CONFIG.keys()))

        with st.expander("🔑 API keys", expanded=not (odds_default and sharp_default)):
            st.caption(
                "Auto-loaded from secrets / environment when set. "
                "Type here only to override."
            )
            odds_key = st.text_input(
                "The Odds API Key", value=odds_default, type="password"
            )
            sharp_key = st.text_input(
                "SharpAPI Key", value=sharp_default, type="password"
            )

        st.divider()
        st.markdown("### 📊 Splits & Line Movement")
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

        with st.expander("✍️ Manual splits (fallback / override)"):
            force_manual = st.checkbox("Always use manual splits")
            ticket_pct = st.slider(
                "Public Ticket % (% of Bets)", min_value=0.0, max_value=100.0, value=30.0
            )
            handle_pct = st.slider(
                "Public Handle % (% of Money)", min_value=0.0, max_value=100.0, value=85.0
            )

    return (
        league,
        odds_key,
        sharp_key,
        ManualInputs(
            ticket_pct, handle_pct, force_manual, line_movement,
            is_opening_move, is_closing_window,
        ),
    )


def market_label(market: str) -> str:
    """Selector label; flags markets that cost an extra API call."""
    return f"{market} (extra API credit)" if market in EVENT_ONLY_MARKETS else market


def resolve_splits(
    league: str,
    sharp_key: str,
    game: dict[str, Any],
    market_type: str,
    selection: str,
) -> tuple[Optional[SplitReading], str]:
    """Look up live splits for the chosen selection.

    Returns:
        (reading or None, human-readable status for the caption)
    """
    if not sharp_key:
        return None, "No SharpAPI key set: using manual splits."
    try:
        rows = cached_splits(SPORT_CONFIG[league].splits_league, sharp_key)
    except SharpAPIError as exc:
        return None, f"SharpAPI unavailable ({exc}). Using manual splits."

    event_rows = match_split_rows(
        rows, game["away_team"], game["home_team"], game["commence_time"]
    )
    if not event_rows:
        return None, "No SharpAPI splits matched this game. Using manual splits."
    reading = read_split(
        event_rows, market_type, selection, game["away_team"], game["home_team"]
    )
    if reading is None:
        return None, (
            f"SharpAPI has no ticket+handle splits for {market_type} on this "
            "selection. Using manual splits."
        )
    stamp = f" | fetched {reading.fetched_at}" if reading.fetched_at else ""
    return reading, f"Splits: **{reading.source}** via SharpAPI{stamp}"


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
        "Live odds from **The Odds API** plus handle / ticket splits from "
        "**SharpAPI**, scored on a 6-point whale checklist."
    )

    league, odds_key, sharp_key, manual = render_sidebar()
    cfg = SPORT_CONFIG[league]

    if not odds_key:
        st.warning(
            "⚠️ No Odds API key found. Add ODDS_API_KEY to .streamlit/secrets.toml "
            "(or your environment), or enter it under 🔑 API keys in the sidebar."
        )
        return

    try:
        games, remaining = cached_featured(cfg.api_key, odds_key)
    except OddsAPIError as exc:
        st.error(str(exc))
        return
    if remaining is not None:
        st.sidebar.caption(f"Odds API credits remaining: {remaining}")

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
                cfg.api_key, game["id"], odds_key, market_type
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

    # --- Splits: SharpAPI first, manual sliders as fallback / override ---------
    reading: Optional[SplitReading] = None
    if manual.force_manual:
        status = "Splits: **manual** (override on)."
    else:
        reading, status = resolve_splits(
            league, sharp_key, game, market_type, chosen["name"]
        )
    ticket_pct = reading.ticket_pct if reading else manual.ticket_pct
    handle_pct = reading.handle_pct if reading else manual.handle_pct
    if reading is None and not manual.force_manual:
        st.warning(status)
    else:
        st.caption(status)

    result = audit_market(
        league,
        market_type,
        line_val,
        ticket_pct,
        handle_pct,
        manual.line_movement,
        manual.is_opening_move,
        manual.is_closing_window,
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
            label=f"Handle vs. Ticket Gap ({ticket_pct:.0f}% bets)",
            value=f"{handle_pct:.0f}% Handle",
            delta=f"{handle_pct - ticket_pct:+.1f}% vs Public Bets",
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
            ticket_pct,
            handle_pct,
            manual.line_movement,
            manual.is_opening_move,
            manual.is_closing_window,
        )
    )


if __name__ == "__main__":
    main()
