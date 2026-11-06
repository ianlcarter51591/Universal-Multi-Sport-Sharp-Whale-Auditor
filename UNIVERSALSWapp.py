"""Universal Sharp & Whale Action Auditor: Streamlit UI.

Run with:  streamlit run app.py
Requires auditor_core.py and dk_splits.py in the same folder.

Odds come from The Odds API. Handle / bet splits come from the free DraftKings
Network splits page (no key needed). The Odds API key loads automatically on open
(first match wins):
    1. .streamlit/secrets.toml  (ODDS_API_KEY)
    2. environment variable     (ODDS_API_KEY)
The sidebar key box is an optional override.
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
    SplitReading,
    audit_market,
    checklist_table,
    fetch_event_market,
    fetch_featured_odds,
    format_commence,
    format_price,
    game_label,
    market_outcomes,
    match_split_rows,
    outcome_label,
    pick_bookmaker,
    read_split,
)
from dk_splits import (
    DKResult,
    DKSplitsError,
    coverage_report,
    describe_report,
    fetch_dk_splits,
)

# --- Theme palette (swap these to restyle the whole app) ---------------------
WHITE = "#FFFFFF"
EMERALD = "#047857"        # body text on white (contrast ~5.5:1)
EMERALD_DEEP = "#065F46"   # alerts / emphasis on pale green
EMERALD_LIGHT = "#34D399"  # text on onyx surfaces (contrast ~9:1)
ONYX = "#0F0F0F"           # highlight surfaces: inputs, table header, accents
MINT = "#ECFDF5"           # soft hover / card tint


def build_css() -> str:
    """Return the global stylesheet: white page, emerald text, onyx highlights."""
    return f"""
<style>
/* Page + sidebar + top bar: force light surfaces regardless of system theme */
.stApp, [data-testid="stAppViewContainer"], .main,
[data-testid="stSidebar"], [data-testid="stHeader"] {{
    background-color: {WHITE} !important;
}}
[data-testid="stSidebar"] {{ border-right: 2px solid {ONYX}; }}
[data-testid="stHeader"] * {{ color: {ONYX} !important; fill: {ONYX} !important; }}

/* Emerald text everywhere on white */
.stApp, .stApp p, .stApp span, .stApp label, .stApp li, .stApp small,
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stWidgetLabel"] *, [data-testid="stCaptionContainer"] *,
[data-testid="stMetricLabel"] *, [data-testid="stMetricValue"] *,
[data-testid="stMarkdownContainer"] * {{
    color: {EMERALD} !important;
}}
.stApp h1, .stApp h2, .stApp h3 {{ color: {EMERALD_DEEP} !important; }}

/* Onyx inputs / dropdowns with light-emerald text */
.stApp [data-baseweb="select"] > div,
.stApp [data-baseweb="input"] > div,
.stApp [data-baseweb="base-input"],
.stApp [data-testid="stNumberInput"] input,
.stApp [data-testid="stTextInput"] input {{
    background-color: {ONYX} !important;
    border-color: {ONYX} !important;
}}
.stApp [data-baseweb="select"] *,
.stApp [data-baseweb="input"] *,
.stApp [data-baseweb="base-input"] *,
.stApp [data-testid="stNumberInput"] *,
.stApp [data-testid="stTextInput"] input {{
    color: {EMERALD_LIGHT} !important;
    fill: {EMERALD_LIGHT} !important;
}}
/* Open dropdown list */
[data-baseweb="popover"] ul, [data-baseweb="popover"] li,
[data-baseweb="menu"], [data-baseweb="menu"] li {{
    background-color: {ONYX} !important;
    color: {EMERALD_LIGHT} !important;
}}
[data-baseweb="popover"] li:hover, [data-baseweb="menu"] li[aria-selected="true"] {{
    background-color: {EMERALD} !important;
    color: {WHITE} !important;
}}

/* Sliders, checkboxes, expanders */
[data-testid="stSlider"] [role="slider"] {{ background-color: {ONYX} !important; }}
[data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] * {{
    color: {ONYX} !important;
}}
[data-testid="stExpander"] details {{
    border: 1px solid {ONYX} !important; background: {WHITE} !important;
}}
[data-testid="stExpander"] summary * {{ color: {EMERALD} !important; }}

/* Alerts: pale mint card, onyx edge, readable emerald text */
[data-testid="stAlert"] {{
    background-color: {MINT} !important; border-left: 5px solid {ONYX} !important;
}}
[data-testid="stAlert"] * {{ color: {EMERALD_DEEP} !important; }}

/* Code blocks (diagnostics): onyx surface, light-emerald text */
[data-testid="stCode"], [data-testid="stCode"] pre, [data-testid="stCode"] code {{
    background-color: {ONYX} !important; color: {EMERALD_LIGHT} !important;
}}

/* Metric cards with an onyx highlight bar */
[data-testid="stMetric"] {{
    background: {MINT}; border-left: 5px solid {ONYX};
    padding: 0.6rem 0.9rem; border-radius: 4px;
}}

/* Checklist table: onyx header, emerald rows */
[data-testid="stTable"] table {{ border-collapse: collapse; width: 100%; }}
[data-testid="stTable"] th, [data-testid="stTable"] th * {{
    background-color: {ONYX} !important; color: {WHITE} !important;
}}
[data-testid="stTable"] td, [data-testid="stTable"] td * {{
    color: {EMERALD} !important; border-color: {ONYX}33 !important;
}}
[data-testid="stTable"] tr:hover td {{ background-color: {MINT} !important; }}
</style>
"""


CSS = build_css()


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


@st.cache_data(ttl=300, show_spinner="Pulling DraftKings handle / bet splits...")
def cached_dk_splits(label: str, event_group: int) -> DKResult:
    """Cached DraftKings Network splits pull (page refreshes every few minutes)."""
    return fetch_dk_splits(label, event_group)


def load_secret(name: str) -> str:
    """Read a key from Streamlit secrets, then the environment, else ''."""
    try:
        value = st.secrets.get(name, "")
    except Exception:  # no secrets.toml configured
        value = ""
    return str(value or os.environ.get(name, ""))


def render_sidebar() -> tuple[str, str, ManualInputs]:
    """Draw the sidebar; return (league, odds key, manual inputs)."""
    odds_default = load_secret("ODDS_API_KEY")

    with st.sidebar:
        st.header("⚙️ Configuration & Feed")
        league = st.selectbox("Select League / Sport", list(SPORT_CONFIG.keys()))

        with st.expander("🔑 API key", expanded=not odds_default):
            st.caption(
                "Auto-loaded from secrets / environment when set. "
                "Type here only to override."
            )
            odds_key = st.text_input(
                "The Odds API Key", value=odds_default, type="password"
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
    game: dict[str, Any],
    market_type: str,
    selection: str,
) -> tuple[Optional[SplitReading], str, str]:
    """Look up live DraftKings Network splits for the chosen selection.

    Returns:
        (reading or None, status caption, diagnostics text)
    """
    cfg = SPORT_CONFIG[league]
    try:
        result = cached_dk_splits(cfg.dk_label, cfg.dk_event_group)
    except DKSplitsError as exc:
        return None, f"Splits unavailable: {exc} Using manual splits.", describe_report(exc.report)

    diag = describe_report(result.report)
    rows = result.rows
    event_rows = match_split_rows(
        rows, game["away_team"], game["home_team"], game["commence_time"]
    )
    if not event_rows:
        wanted = f"{game['away_team']} @ {game['home_team']} ({format_commence(game['commence_time'])})"
        listing = "\n".join(
            f"  {r['away_team']} @ {r['home_team']}  {r.get('event_start_time', '?')}" for r in rows[:40]
        )
        diag += f"\nLooking for: {wanted}\nGames read from the page:\n{listing}"
        span = f" The page covers {result.report.span}." if result.report.span else ""
        sample = "; ".join(f"{r['away_team']} @ {r['home_team']}" for r in rows[:3])
        return None, (
            f"No DraftKings Network splits matched {wanted} ({len(rows)} games read, "
            f"e.g. {sample}).{span} It may not be posted yet. Using manual splits."
        ), diag
    reading = read_split(
        event_rows, market_type, selection, game["away_team"], game["home_team"]
    )
    if reading is None:
        return None, (
            f"DraftKings Network has no handle + bets splits for {market_type} on this "
            "selection. Using manual splits."
        ), diag
    stamp = f" | updated {reading.fetched_at}" if reading.fetched_at else ""
    return reading, f"Splits: **DraftKings Network** (all jurisdictions){stamp}", diag


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
        "Live odds from **The Odds API** plus handle / bet splits from "
        "**DraftKings Network**, scored on a 6-point whale checklist."
    )

    league, odds_key, manual = render_sidebar()
    cfg = SPORT_CONFIG[league]

    with st.expander("🧪 DraftKings league coverage check"):
        st.caption("Tries every league on the DraftKings Network page and shows what it returns.")
        if st.button("Check every league"):
            with st.spinner("Checking leagues..."):
                st.table(coverage_report(
                    {n: (c.dk_label, c.dk_event_group) for n, c in SPORT_CONFIG.items()},
                    fetch=cached_dk_splits,
                ))

    if not odds_key:
        st.warning(
            "⚠️ No Odds API key found. Add ODDS_API_KEY to .streamlit/secrets.toml "
            "(or your environment), or enter it under 🔑 API key in the sidebar."
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

    # --- Splits: DraftKings Network first, manual sliders as fallback / override
    reading: Optional[SplitReading] = None
    diag = ""
    if manual.force_manual:
        status = "Splits: **manual** (override on)."
    else:
        reading, status, diag = resolve_splits(
            league, game, market_type, chosen["name"]
        )
    ticket_pct = reading.ticket_pct if reading else manual.ticket_pct
    handle_pct = reading.handle_pct if reading else manual.handle_pct
    if reading is None and not manual.force_manual:
        st.warning(status)
    else:
        st.caption(status)
    if diag:
        with st.expander("🔧 Splits diagnostics"):
            st.code(diag)

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
