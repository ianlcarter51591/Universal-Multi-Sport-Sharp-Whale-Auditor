"""Core logic for the Universal Sharp & Whale Action Auditor.

Pure-Python / pandas / numpy / requests only (no Streamlit), so it can be unit
tested without a running UI.

Model (all thresholds are module constants):
    gap        = handle% - ticket%                  -> whale if gap >= WHALE_GAP_PCT
    RLM        = (tickets <= 30% and move < 0) or (tickets >= 70% and move > 0)
    steam      = |move| >= STEAM_PTS
    late limit = in closing window and |move| >= LATE_MOVE_PTS
    key number = min_k | |line| - k | <= tolerance   (spreads only)
    verdict    = whale and (criteria passed) >= MIN_CRITERIA_FOR_SHARP

Line-movement sign convention (from the selected side's perspective):
    negative = market moved TOWARD this selection (price/spread got more expensive)
    positive = market moved AWAY from this selection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import requests

# --- API constants -----------------------------------------------------------
BASE_URL = "https://api.the-odds-api.com/v4/sports"
REQUEST_TIMEOUT_S = 15
FEATURED_MARKETS = ("spreads", "totals", "h2h")
EVENT_ONLY_MARKETS = ("team_totals",)  # not allowed on the bulk /odds endpoint
PREFERRED_BOOK = "draftkings"
DISPLAY_TZ = "America/New_York"

# --- Model thresholds --------------------------------------------------------
WHALE_GAP_PCT = 15.0
RLM_PUBLIC_LOW = 30.0
RLM_PUBLIC_HIGH = 70.0
STEAM_PTS = 1.0
LATE_MOVE_PTS = 0.5
MIN_CRITERIA_FOR_SHARP = 2  # whale divergence + at least one corroborating signal


@dataclass(frozen=True)
class SportConfig:
    """Static per-league configuration."""

    api_key: str
    key_numbers: tuple[float, ...]
    kind: str
    key_tolerance: float = 0.0  # 0.5 = "on the key or a half-point hook"


SPORT_CONFIG: dict[str, SportConfig] = {
    "NBA": SportConfig("basketball_nba", (3, 4, 5, 7, 10), "basketball", 0.5),
    "NFL": SportConfig(
        "americanfootball_nfl", (3, 7, 6, 10, 4, 14, 17, 21), "football", 0.5
    ),
    "MLB": SportConfig("baseball_mlb", (1.5,), "baseball", 0.0),
    "NHL": SportConfig("icehockey_nhl", (1.5,), "hockey", 0.0),
    "NCAA Football": SportConfig(
        "americanfootball_ncaaf", (3, 7, 10, 6, 14, 21, 24), "football", 0.5
    ),
    "NCAA Basketball": SportConfig(
        "basketball_ncaab", (3, 4, 5, 7, 10), "basketball", 0.5
    ),
    "EPL": SportConfig("soccer_epl", (0.25, 0.5, 0.75, 1.25, 1.5), "soccer"),
    "UCL": SportConfig(
        "soccer_uefa_champs_league", (0.25, 0.5, 0.75, 1.25, 1.5), "soccer"
    ),
    "Bundesliga": SportConfig(
        "soccer_germany_bundesliga", (0.25, 0.5, 0.75, 1.25, 1.5), "soccer"
    ),
    "La Liga": SportConfig(
        "soccer_spain_la_liga", (0.25, 0.5, 0.75, 1.25, 1.5), "soccer"
    ),
}

CHECK_LABELS: dict[str, str] = {
    "whale_divergence": "1. Ticket % vs Handle % Divergence (>= 15%)",
    "reverse_line_movement": "2. Reverse Line Movement (RLM)",
    "steam_movement": "3. Steam Move (>= 1.0 pt)",
    "early_opener": "4. Early Opener Hunting",
    "late_market_limit": "5. Late Market Limit Steam",
    "key_number_resistance": "6. Key Number Resistance (spreads only)",
}


# --- Odds API client ---------------------------------------------------------
class OddsAPIError(Exception):
    """Raised for any failure talking to The Odds API."""


def _request(url: str, params: Mapping[str, Any]) -> tuple[Any, Optional[str]]:
    """GET `url` and return (parsed JSON, remaining-credits header).

    Raises:
        OddsAPIError: on network failure, non-200 status, or invalid JSON.
    """
    try:
        resp = requests.get(url, params=dict(params), timeout=REQUEST_TIMEOUT_S)
    except requests.RequestException as exc:
        raise OddsAPIError(f"Could not reach The Odds API: {exc}") from exc

    if resp.status_code != 200:
        try:
            body = resp.json()
            msg = body.get("message") if isinstance(body, dict) else None
        except ValueError:
            msg = None
        raise OddsAPIError(
            f"API error ({resp.status_code}): "
            f"{msg or (resp.text or '')[:200] or 'check API key / quota.'}"
        )
    try:
        return resp.json(), resp.headers.get("x-requests-remaining")
    except ValueError as exc:
        raise OddsAPIError("The Odds API returned invalid JSON.") from exc


def fetch_featured_odds(
    sport_api_key: str, api_key: str
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Fetch spreads/totals/moneyline for every upcoming game in a sport."""
    data, remaining = _request(
        f"{BASE_URL}/{sport_api_key}/odds/",
        {
            "apiKey": api_key,  # NOTE: camelCase is required by the API
            "regions": "us",
            "markets": ",".join(FEATURED_MARKETS),
            "oddsFormat": "american",  # NOTE: camelCase is required by the API
        },
    )
    if not isinstance(data, list):
        raise OddsAPIError("Unexpected response shape from The Odds API.")
    return data, remaining


def fetch_event_market(
    sport_api_key: str, event_id: str, api_key: str, market: str
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Fetch one additional market (e.g. team_totals) for a single event.

    Returns:
        (bookmakers list, remaining credits)
    """
    data, remaining = _request(
        f"{BASE_URL}/{sport_api_key}/events/{event_id}/odds",
        {
            "apiKey": api_key,
            "regions": "us",
            "markets": market,
            "oddsFormat": "american",
        },
    )
    if not isinstance(data, dict):
        raise OddsAPIError("Unexpected response shape from The Odds API.")
    return data.get("bookmakers", []), remaining


# --- Feed helpers ------------------------------------------------------------
def pick_bookmaker(
    bookmakers: Sequence[Mapping[str, Any]], preferred: str = PREFERRED_BOOK
) -> Optional[Mapping[str, Any]]:
    """Return the preferred bookmaker, else the first one, else None."""
    return next(
        (b for b in bookmakers if b.get("key") == preferred),
        bookmakers[0] if bookmakers else None,
    )


def market_outcomes(book: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Map market key -> outcomes for a bookmaker payload."""
    return {m["key"]: m.get("outcomes", []) for m in book.get("markets", [])}


def format_price(price: Any) -> str:
    """Format an American price as +150 / -110; fall back to str()."""
    try:
        return f"{int(round(float(price))):+d}"
    except (TypeError, ValueError):
        return str(price)


def outcome_label(outcome: Mapping[str, Any]) -> str:
    """Human-readable label, including team for team totals."""
    parts = [
        outcome.get("description"),  # team name on team_totals
        outcome.get("name"),
        outcome.get("point"),
    ]
    text = " ".join(str(p) for p in parts if p not in (None, ""))
    return f"{text} ({format_price(outcome.get('price'))})"


def format_commence(commence_iso: str) -> str:
    """Convert an ISO-8601 UTC timestamp to a US Eastern display string."""
    ts = pd.to_datetime(commence_iso, utc=True).tz_convert(DISPLAY_TZ)
    return ts.strftime("%a %b %d, %I:%M %p ET")


def game_label(game: Mapping[str, Any]) -> str:
    """Matchup label for the game selector."""
    return (
        f"{game['away_team']} @ {game['home_team']} "
        f"({format_commence(game['commence_time'])})"
    )


# --- Audit logic -------------------------------------------------------------
@dataclass(frozen=True)
class AuditResult:
    """Outcome of the checklist. `checks` value None means 'not applicable'."""

    is_sharp: bool
    passed: int
    applicable: int
    checks: dict[str, Optional[bool]]


def key_number_hit(
    line: float, key_numbers: Sequence[float], tolerance: float
) -> bool:
    """True if |line| is within `tolerance` of any key number (vectorized)."""
    keys = np.asarray(key_numbers, dtype=float)
    return bool(np.min(np.abs(keys - abs(float(line)))) <= tolerance + 1e-9)


def audit_market(
    sport_name: str,
    market_type: str,
    line_val: Optional[float],
    ticket_pct: float,
    handle_pct: float,
    line_movement: float,
    is_opening_move: bool,
    is_closing_window: bool,
) -> AuditResult:
    """Run the 6-point whale/sharp checklist for one selection."""
    cfg = SPORT_CONFIG[sport_name]
    gap = handle_pct - ticket_pct
    move = abs(line_movement)

    key_check: Optional[bool]
    if market_type == "spreads":
        key_check = line_val is not None and key_number_hit(
            line_val, cfg.key_numbers, cfg.key_tolerance
        )
    else:
        key_check = None  # key numbers only describe point spreads

    checks: dict[str, Optional[bool]] = {
        "whale_divergence": bool(gap >= WHALE_GAP_PCT),
        "reverse_line_movement": bool(
            (ticket_pct <= RLM_PUBLIC_LOW and line_movement < 0)
            or (ticket_pct >= RLM_PUBLIC_HIGH and line_movement > 0)
        ),
        "steam_movement": bool(move >= STEAM_PTS),
        "early_opener": bool(is_opening_move),
        "late_market_limit": bool(is_closing_window and move >= LATE_MOVE_PTS),
        "key_number_resistance": key_check,
    }
    scored = [v for v in checks.values() if v is not None]
    passed = int(sum(scored))
    return AuditResult(
        is_sharp=bool(
            checks["whale_divergence"] and passed >= MIN_CRITERIA_FOR_SHARP
        ),
        passed=passed,
        applicable=len(scored),
        checks=checks,
    )


def checklist_table(
    result: AuditResult,
    sport_name: str,
    ticket_pct: float,
    handle_pct: float,
    line_movement: float,
    is_opening_move: bool,
    is_closing_window: bool,
) -> pd.DataFrame:
    """Build the display table for the checklist breakdown."""
    details = {
        "whale_divergence": (
            f"Handle ({handle_pct:.0f}%) vs Bets ({ticket_pct:.0f}%): "
            f"gap is {handle_pct - ticket_pct:+.1f}%"
        ),
        "reverse_line_movement": (
            f"Tickets {ticket_pct:.0f}% with net move {line_movement:+.1f} pts "
            "(negative = toward this side)"
        ),
        "steam_movement": f"Net movement of {line_movement:+.1f} pts recorded.",
        "early_opener": (
            "Flagged as opening-market action."
            if is_opening_move
            else "Standard market timing."
        ),
        "late_market_limit": (
            "Move inside the closing window."
            if is_closing_window
            else "Not in closing window."
        ),
        "key_number_resistance": (
            f"Evaluated against key margins for {sport_name}."
            if result.checks["key_number_resistance"] is not None
            else "N/A: key numbers apply to point spreads only."
        ),
    }
    status = {True: "✅ Met", False: "❌ Failed", None: "➖ N/A"}
    return pd.DataFrame(
        {
            "Criterion": list(CHECK_LABELS.values()),
            "Status": [status[result.checks[k]] for k in CHECK_LABELS],
            "Detail": [details[k] for k in CHECK_LABELS],
        }
    )
