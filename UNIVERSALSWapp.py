"""Universal Sharp & Whale Action Auditor: Streamlit UI.

Tabs: Board (every game scored at once), Audit a game (one selection in detail, both sides of the market), Signal log
(does LEAN / SHARP beat the closing line and the result?).

Odds: The Odds API. Handle / bet splits: free DraftKings Network page. Line history and the signal log are saved to a
private GitHub Gist when GITHUB_GIST_TOKEN is set (otherwise to local files that reset when Streamlit Cloud restarts).
Secrets (first match wins): .streamlit/secrets.toml, then environment variables: ODDS_API_KEY, GITHUB_GIST_TOKEN.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

REQUIRED_FILES = ("auditor_core.py", "dk_splits.py", "cloud_store.py", "persist.py", "line_history.py", "scoring.py", "board_scan.py", "audit_log.py", "split_history.py")
_missing = [f for f in REQUIRED_FILES if not (Path(__file__).resolve().parent / f).exists()]
if _missing:
    st.error("These files are missing from the app folder in GitHub: " + ", ".join(_missing) + ". Upload them next to this file and reboot the app.")
    st.stop()

import audit_log
import cloud_store
import persist
import split_history as sh
from auditor_core import (
    EVENT_ONLY_MARKETS,
    FEATURED_MARKETS,
    SPORT_CONFIG,
    OddsAPIError,
    SplitReading,
    fetch_event_market,
    fetch_featured_odds,
    fetch_scores,
    format_commence,
    format_price,
    game_label,
    market_outcomes,
    match_split_rows,
    outcome_label,
    pick_bookmaker,
    read_split,
)
from board_scan import evaluate_market, pair_table, scan_board
from dk_splits import DKResult, DKSplitsError, coverage_report, describe_report, fetch_dk_splits, parse_dk_override
from line_history import COLUMNS as HIST_COLUMNS, HISTORY_NAME, KEY as HIST_KEY, append_new, movement, prune, rows_from_games
from scoring import TOTAL_CHECKS, Context, pair_warning, score_selection

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


TIER_ICON = {"SHARP": "🔥 SHARP", "LEAN": "👀 LEAN", "PASS": "🚫 PASS", "FILTERED": "🧹 FILTERED", "NO DATA": "⛔ NO DATA"}


@dataclass(frozen=True)
class ManualInputs:
    """Sidebar overrides. Nothing is used unless its box is ticked: the app never invents splits or movement."""

    use_splits: bool
    ticket_pct: float
    handle_pct: float
    use_move: bool
    line_movement: float


# --- Cached API wrappers (exceptions are NOT cached by Streamlit) -------------
@st.cache_data(ttl=600, show_spinner="Pulling live odds...")
def cached_featured(sport_api_key: str, api_key: str) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Cached bulk odds pull (10 min)."""
    return fetch_featured_odds(sport_api_key, api_key)


@st.cache_data(ttl=600, show_spinner="Pulling team totals...")
def cached_event_market(sport_api_key: str, event_id: str, api_key: str, market: str) -> tuple[list[dict[str, Any]], Optional[str]]:
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


def render_sidebar() -> tuple[str, str, str, ManualInputs]:
    """Draw the sidebar; return (league, odds key, DraftKings override text, manual inputs)."""
    odds_default = load_secret("ODDS_API_KEY")
    with st.sidebar:
        st.header("⚙️ Configuration & Feed")
        league = st.selectbox("Select League / Sport", list(SPORT_CONFIG.keys()))
        with st.expander("🔑 API key", expanded=not odds_default):
            st.caption("Auto-loaded from secrets / environment when set. Type here only to override.")
            odds_key = st.text_input("The Odds API Key", value=odds_default, type="password")
        override = st.text_input("DraftKings page URL or tb_eg id (optional)", value="",
                                 help="Only needed if a league's splits don't load: pick the league on the DraftKings Network page and paste the address.")
        with st.expander("✍️ Manual overrides (only when you tick them)"):
            use_splits = st.checkbox("Use manual splits")
            ticket_pct = st.slider("Public Ticket % (% of Bets)", min_value=0.0, max_value=100.0, value=50.0)
            handle_pct = st.slider("Public Handle % (% of Money)", min_value=0.0, max_value=100.0, value=50.0)
            use_move = st.checkbox("Use manual line movement")
            move = st.number_input("Line movement from open (negative = toward this pick)", value=0.0, step=0.5)
        status = cloud_store.status()
        (st.caption if status["level"] == "ok" else st.warning)(("💾 " if status["level"] == "ok" else "⚠️ ") + status["text"])
    return league, odds_key, override, ManualInputs(bool(use_splits), float(ticket_pct), float(handle_pct), bool(use_move), float(move))


def market_label(market: str) -> str:
    """Selector label; flags markets that cost an extra API call."""
    return f"{market} (extra API credit)" if market in EVENT_ONLY_MARKETS else market


# --- Splits -------------------------------------------------------------------
def load_splits(league: str, override: str) -> tuple[list[dict[str, Any]], str, str]:
    """Rows from the DraftKings Network page for the league. (rows, status text, diagnostics); rows empty on failure."""
    cfg = SPORT_CONFIG[league]
    label, eg, note = cfg.dk_label, cfg.dk_event_group, ""
    if override.strip():
        parsed = parse_dk_override(override, cfg.dk_label)
        if parsed is None:
            note = "Could not find a tb_eg id in the pasted text; using the built-in league id. "
        else:
            label, eg = parsed
    try:
        result = cached_dk_splits(label, eg)
    except DKSplitsError as exc:
        return [], f"{note}Splits unavailable: {exc}", describe_report(exc.report)
    stamp = next((r.get("fetched_at") for r in result.rows if r.get("fetched_at")), None)
    return list(result.rows), f"{note}Splits: **DraftKings Network** (all jurisdictions)" + (f" | updated {stamp}" if stamp else ""), describe_report(result.report)


def reading_for(rows: list[dict[str, Any]], game: dict[str, Any], market: str, selection: str) -> tuple[Optional[SplitReading], str]:
    """Splits for one selection, or (None, why not)."""
    if not rows:
        return None, "No splits were loaded."
    event_rows = match_split_rows(rows, game["away_team"], game["home_team"], game["commence_time"])
    if not event_rows:
        return None, f"No DraftKings Network splits matched {game['away_team']} @ {game['home_team']} ({format_commence(game['commence_time'])}). It may not be posted yet."
    reading = read_split(event_rows, market, selection, game["away_team"], game["home_team"])
    if reading is None:
        return None, f"DraftKings Network has no handle + bets splits for {market} on this selection."
    return reading, ""


# --- Stored data ----------------------------------------------------------------
def get_history() -> pd.DataFrame:
    """Saved line readings (loaded once per session)."""
    if "aud_history" not in st.session_state:
        st.session_state["aud_history"] = persist.load(HISTORY_NAME, HIST_COLUMNS)
    return st.session_state["aud_history"]


def get_signals() -> pd.DataFrame:
    """Saved signal log (loaded once per session)."""
    if "aud_signals" not in st.session_state:
        st.session_state["aud_signals"] = audit_log.normalize(persist.load(audit_log.SIGNALS_NAME, audit_log.COLUMNS))
    return st.session_state["aud_signals"]


def get_split_history() -> pd.DataFrame:
    """Saved ticket / handle readings (loaded once per session)."""
    if "aud_splits" not in st.session_state:
        st.session_state["aud_splits"] = persist.load(sh.SPLIT_NAME, sh.COLUMNS)
    return st.session_state["aud_splits"]


def store_split_history(history: pd.DataFrame) -> None:
    """Save ticket / handle readings."""
    st.session_state["aud_splits"] = persist.save(sh.SPLIT_NAME, history, ["TS"] + sh.KEY, sh.COLUMNS)


def store_history(history: pd.DataFrame) -> None:
    """Save line readings (permanent storage merges anything it already held)."""
    st.session_state["aud_history"] = persist.save(HISTORY_NAME, history, ["TS"] + HIST_KEY, HIST_COLUMNS)


def store_signals(signals: pd.DataFrame, force: bool = False) -> None:
    """Save the signal log."""
    st.session_state["aud_signals"] = audit_log.normalize(persist.save(audit_log.SIGNALS_NAME, signals, ["SIGNAL_ID"], audit_log.COLUMNS, force))


# --- Tabs -----------------------------------------------------------------------
def render_board(best: pd.DataFrame, split_status: str, split_rows: list, diag: str, hist_rows: int) -> None:
    """Ranked table of every game / market."""
    (st.caption if split_rows else st.warning)(split_status)
    if best.empty:
        st.info("No upcoming games to scan.")
        return
    tiers = best["TIER"].value_counts()
    cols5 = st.columns(5)
    for col, label, value in zip(cols5, ("SHARP", "LEAN", "Filtered as noise", "Markets scanned", "Line readings saved"),
                                 (int(tiers.get("SHARP", 0)), int(tiers.get("LEAN", 0)), int(tiers.get("FILTERED", 0)), len(best), hist_rows)):
        with col:
            st.metric(label, value)
    if not split_rows:
        st.warning("No real splits loaded, so every market shows NO DATA. Nothing is scored from made-up numbers.")
    show = st.selectbox("Show", ["LEAN and SHARP", "Filtered as noise", "Everything"])
    view = best if show == "Everything" else best[best["TIER"].isin(["FILTERED"] if show == "Filtered as noise" else ["LEAN", "SHARP"])]
    if view.empty and split_rows:
        st.info("Nothing reaches LEAN right now. Choose Everything to see the full board.")
    cols = ["WHEN", "GAME", "MARKET", "PICK", "TICKETS_%", "HANDLE_%", "GAP", "BET_SIZE_X", "TOWARD", "CHECKS", "TIER", "NOTE"]
    st.dataframe(view[cols], use_container_width=True, hide_index=True)
    st.caption("How to read it: one row per game and market, showing the side with the money. CHECKS = how many of 5 yes/no checks are met: "
               "(1) money gap of 15+ points, (2) bigger bets, (3) the line moved toward the pick, (4) the signal held 30 minutes, (5) a market tell "
               "(reverse movement, an early / late move, or a key-number cross). LEAN = the money is on this side. SHARP = money, line move, "
               "30-minute hold and a tell all agree. FILTERED = it would be LEAN / SHARP but failed a noise filter. NOTE says what a LEAN still needs. "
               "TOWARD = how far the line has moved toward the pick since this app first saw it (negative = away). BET_SIZE_X = how many times bigger "
               "the average bet is on this side than the other.")
    if diag:
        with st.expander("🔧 Splits diagnostics"):
            st.code(diag)
    with st.expander("🧪 DraftKings league coverage check"):
        if st.button("Check every league"):
            with st.spinner("Checking leagues..."):
                st.table(coverage_report({n: (c.dk_label, c.dk_event_group) for n, c in SPORT_CONFIG.items()}, fetch=cached_dk_splits))


def render_audit(league: str, games: list[dict[str, Any]], odds_key: str, split_rows: list, all_sides: pd.DataFrame,
                 history: pd.DataFrame, split_hist: pd.DataFrame, news_ids: list, manual: ManualInputs, now: pd.Timestamp) -> None:
    """One selection in detail."""
    cfg = SPORT_CONFIG[league]
    game_idx = st.selectbox("Select Matchup to Audit", range(len(games)), format_func=lambda i: game_label(games[i]))
    game = games[game_idx]
    book = pick_bookmaker(game.get("bookmakers", []))
    if book is None:
        st.error("No active bookmaker odds available for this game.")
        return
    st.caption(f"Active Book Feed: **{book['title']}** | Last Updated: {book.get('last_update', 'N/A')}")
    market_dict = market_outcomes(book)
    markets = [m for m in FEATURED_MARKETS if m in market_dict] + list(EVENT_ONLY_MARKETS)
    market = st.selectbox("Select Market Type", markets, format_func=market_label)
    if market in EVENT_ONLY_MARKETS:
        try:
            ev_books, _ = cached_event_market(cfg.api_key, game["id"], odds_key, market)
        except OddsAPIError as exc:
            st.error(str(exc))
            return
        ev_book = pick_bookmaker(ev_books)
        outcomes = market_outcomes(ev_book).get(market, []) if ev_book else []
        if not outcomes:
            st.warning(f"No {market} currently posted for this game.")
            return
    else:
        outcomes = market_dict[market]
    chosen_idx = st.selectbox("Select Specific Selection", range(len(outcomes)), format_func=lambda i: outcome_label(outcomes[i]))
    chosen = outcomes[chosen_idx]
    line, price, name = chosen.get("point"), chosen.get("price"), chosen["name"]

    featured = market in FEATURED_MARKETS and not manual.use_splits
    if featured:
        event_rows = match_split_rows(split_rows, game["away_team"], game["home_team"], game["commence_time"]) if split_rows else []
        ev = evaluate_market(game, market, outcomes, event_rows, history, split_hist, now, league, news_ids)[chosen_idx]
        reading, mv, sc = ev.reading, ev.movement, ev.scored
        ticket, handle = (reading.ticket_pct, reading.handle_pct) if reading else (None, None)
        if reading is not None:
            source = "Splits: **DraftKings Network**" + (f" | updated {reading.fetched_at}" if reading.fetched_at else "")
        else:
            source = reading_for(split_rows, game, market, name)[1]
    else:
        ticket, handle = (manual.ticket_pct, manual.handle_pct) if manual.use_splits else (None, None)
        source = "Splits: **manual** (override on)." if manual.use_splits else "No splits are published for team totals; tick manual splits in the sidebar to score one."
        mv = None if market in EVENT_ONLY_MARKETS or manual.use_move else movement(history, game["id"], market, name, game["commence_time"], now, line, price)
        sc = score_selection(league, market, line, ticket, handle, mv, manual.line_movement if manual.use_move else None,
                             Context(price=None if price is None else float(price), news=game["id"] in news_ids))
    (st.caption if ticket is not None else st.warning)(source)

    st.divider()
    c1, c2, c3 = st.columns(3)
    gap_text = "n/a" if ticket is None else f"{handle - ticket:+.1f}% vs Public Bets"
    with c1:
        st.metric(label="Auditor Verdict", value=TIER_ICON[sc.tier], delta=f"{sc.points}/{TOTAL_CHECKS} checks met", delta_color="off")
    with c2:
        st.metric(label="Handle vs. Ticket Gap" + ("" if ticket is None else f" ({ticket:.0f}% bets)"),
                  value="n/a" if handle is None else f"{handle:.0f}% Handle", delta=gap_text)
    with c3:
        st.metric(label="Current Line & Price", value=format_price(price), delta=f"Line/Spread: {line if line is not None else 'N/A'}", delta_color="off")

    if market in FEATURED_MARKETS and not manual.use_splits:
        pair = pair_table(all_sides, game["id"], market)
        if not pair.empty:
            st.markdown("### ⚖️ Both sides of this market")
            st.dataframe(pair, use_container_width=True, hide_index=True)
            warn = pair_warning(pd.to_numeric(pair["TICKETS_%"], errors="coerce").dropna().tolist(), pd.to_numeric(pair["HANDLE_%"], errors="coerce").dropna().tolist())
            st.caption(warn or "In a two-way market the sides mirror each other: the money is on the side with the positive gap.")
    st.markdown("### 🧹 Noise filters")
    st.table(pd.DataFrame(sc.filters))
    st.markdown("### 📋 The five checks")
    st.table(pd.DataFrame(sc.rows))
    for note in sc.notes:
        st.caption("• " + note)
    st.caption("SHARP = checks 1, 3, 4 and 5 all met. LEAN = check 1 plus at least one more. A pick that fails a noise filter shows as FILTERED.")
    if mv is not None and mv.has_history:
        st.caption(f"First line this app saw: {mv.open_line} at {mv.open_ts} ({mv.n_points} readings). That is not necessarily the true opener.")


def render_log(league: str, odds_key: str, now: pd.Timestamp) -> None:
    """Signal log, validation tables and grading."""
    signals = get_signals()
    st.markdown("### 📒 Does the Auditor beat the market?")
    for line in audit_log.verdict(signals):
        st.info(line)
    if not signals.empty:
        st.markdown("**By tier** (LOW / HIGH = 95% range; BEAT_CLOSE_% = picks whose line was better than the last line the app saw)")
        st.dataframe(audit_log.summary(signals, "TIER").round(1), use_container_width=True, hide_index=True)
        st.markdown("**By market**")
        st.dataframe(audit_log.summary(signals, "MARKET").round(1), use_container_width=True, hide_index=True)
    if st.button(f"Grade finished {league} games (uses 2 Odds API credits)"):
        try:
            scores, _ = fetch_scores(SPORT_CONFIG[league].api_key, odds_key)
        except OddsAPIError as exc:
            st.error(str(exc))
        else:
            graded, n = audit_log.grade(signals, scores, now)
            store_signals(graded, force=True)
            st.success(f"Graded {n} pick(s).")
    if not signals.empty:
        st.dataframe(signals.sort_values("FIRST_SEEN", ascending=False).head(200), use_container_width=True, hide_index=True)
        st.download_button("Download signal log (CSV)", signals.to_csv(index=False), file_name="auditor_signals.csv")
    st.caption("Picks are logged only while the app is open on a league, and the 'closing line' is the last line the app saw. "
               "Results come from The Odds API scores (finished games from the last 3 days only: grade often).")


def main() -> None:
    """Page entry point."""
    st.set_page_config(page_title="Universal Sharp & Whale Action Auditor", page_icon="⚡", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("⚡ Universal Sharp & Whale Action Auditor")
    st.markdown("Live odds from **The Odds API** plus handle / bet splits from **DraftKings Network**, scored on a five-check whale checklist.")

    cloud_store.configure(load_secret("GITHUB_GIST_TOKEN"))
    league, odds_key, override, manual = render_sidebar()
    cfg = SPORT_CONFIG[league]
    if not odds_key:
        st.warning("⚠️ No Odds API key found. Add ODDS_API_KEY to .streamlit/secrets.toml (or your environment), or enter it under 🔑 API key in the sidebar.")
        return
    try:
        games, remaining = cached_featured(cfg.api_key, odds_key)
    except OddsAPIError as exc:
        st.error(str(exc))
        return
    if remaining is not None:
        st.sidebar.caption(f"Odds API credits remaining: {remaining}")
    if not games:
        st.info(f"No upcoming games found for {league}. Check API quota or schedule.")
        return

    now = pd.Timestamp.now(tz="UTC")
    history, added = append_new(get_history(), rows_from_games(games, league, now.isoformat()))
    if added:
        store_history(prune(history, now))
        history = get_history()
    with st.sidebar:
        labels = {g["id"]: game_label(g) for g in games}
        news_ids = list(st.multiselect("Games with injury / lineup news", list(labels), format_func=lambda i: labels[i],
                                       help="Moves after news are everyone's money, not sharp money: marked games are filtered out.") or [])
    split_rows, split_status, diag = load_splits(league, override)
    split_hist = get_split_history()
    best, all_sides = scan_board(games, league, split_rows, history, now, split_hist, news_ids)
    split_hist, n_split = sh.append_new(split_hist, sh.rows_from_sides(all_sides, now.isoformat()))
    if n_split:
        store_split_history(sh.prune(split_hist, now))

    before = get_signals()
    signals, n_new = audit_log.record(before, best, league, now.isoformat())
    signals = audit_log.finalize_clv(audit_log.update_closing(signals, all_sides, now), now)
    if n_new or not signals.equals(before):
        store_signals(signals)

    tab_board, tab_audit, tab_log = st.tabs(["🏆 Board", "🔍 Audit a game", "📒 Signal log"])
    with tab_board:
        render_board(best, split_status, split_rows, diag, len(history))
    with tab_audit:
        render_audit(league, games, odds_key, split_rows, all_sides, history, split_hist, news_ids, manual, now)
    with tab_log:
        render_log(league, odds_key, now)
    cloud_store.flush()


if __name__ == "__main__":
    main()
