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

Splits (DraftKings Network page, see dk_splits.py): rows carry bets_pct / handle_pct
as 0.0-1.0 per side; they are scaled to 0-100 and fed into `gap = handle% - ticket%`.

Line-movement sign convention (from the selected side's perspective):
    negative = market moved TOWARD this selection (price/spread got more expensive)
    positive = market moved AWAY from this selection
"""

from __future__ import annotations

import re
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
SPLIT_BOOK_PRIORITY = ("draftkings",)
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
    dk_label: str = ""  # league label on the DraftKings Network splits page
    dk_event_group: int = 0  # that page's tb_eg id (0 = league not published)


_SOCCER_KEYS = (0.25, 0.5, 0.75, 1.25, 1.5)

# dk_event_group: NFL's 88808 is confirmed from the page's own links; the rest are
# DraftKings' league ids and are cross-checked against the page at runtime.
# Bundesliga / La Liga are not offered on the DraftKings Network page.
SPORT_CONFIG: dict[str, SportConfig] = {
    "NBA": SportConfig("basketball_nba", (3, 4, 5, 7, 10), "basketball", 0.5, "NBA", 42648),
    "NFL": SportConfig(
        "americanfootball_nfl", (3, 7, 6, 10, 4, 14, 17, 21), "football", 0.5, "NFL", 88808
    ),
    "MLB": SportConfig("baseball_mlb", (1.5,), "baseball", 0.0, "MLB", 84240),
    "NHL": SportConfig("icehockey_nhl", (1.5,), "hockey", 0.0, "NHL", 42133),
    "NCAA Football": SportConfig(
        "americanfootball_ncaaf", (3, 7, 10, 6, 14, 21, 24), "football", 0.5,
        "NCAA Football", 87637,
    ),
    "NCAA Basketball": SportConfig(
        "basketball_ncaab", (3, 4, 5, 7, 10), "basketball", 0.5, "NCAA Basketball", 92483
    ),
    "EPL": SportConfig(
        "soccer_epl", _SOCCER_KEYS, "soccer", 0.0, "England Premier League", 40253
    ),
    "UCL": SportConfig(
        "soccer_uefa_champs_league", _SOCCER_KEYS, "soccer", 0.0, "Champions League", 40685
    ),
    "Bundesliga": SportConfig("soccer_germany_bundesliga", _SOCCER_KEYS, "soccer"),
    "La Liga": SportConfig("soccer_spain_la_liga", _SOCCER_KEYS, "soccer"),
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
class APIError(Exception):
    """Base class for upstream API failures."""


class OddsAPIError(APIError):
    """Raised for any failure talking to The Odds API."""


def _error_message(resp: Any) -> Optional[str]:
    """Extract a message from either {"message": ..} or {"error": {"message": ..}}."""
    try:
        body = resp.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if isinstance(err, dict) and err.get("message"):
        return str(err["message"])
    return body.get("message") or (err if isinstance(err, str) else None)


def _request(
    url: str,
    params: Mapping[str, Any],
    headers: Optional[Mapping[str, str]] = None,
    error_cls: type[APIError] = OddsAPIError,
    service: str = "The Odds API",
) -> tuple[Any, Optional[str]]:
    """GET `url` and return (parsed JSON, remaining-credits header).

    Raises:
        error_cls: on network failure, non-200 status, or invalid JSON.
    """
    try:
        resp = requests.get(
            url,
            params=dict(params),
            headers=dict(headers) if headers else None,
            timeout=REQUEST_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        raise error_cls(f"Could not reach {service}: {exc}") from exc

    if resp.status_code != 200:
        msg = _error_message(resp)
        raise error_cls(
            f"{service} error ({resp.status_code}): "
            f"{msg or (resp.text or '')[:200] or 'check API key / plan / quota.'}"
        )
    try:
        return resp.json(), resp.headers.get("x-requests-remaining")
    except ValueError as exc:
        raise error_cls(f"{service} returned invalid JSON.") from exc


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


# --- Splits matching (rows come from dk_splits.py) ---------------------------
@dataclass(frozen=True)
class SplitReading:
    """Ticket and handle share (0-100) for one selection."""

    ticket_pct: float
    handle_pct: float
    source: str
    fetched_at: Optional[str]


_ALIASES = {"uconn": "connecticut", "umass": "massachusetts", "pitt": "pittsburgh"}


def _norm(name: Any) -> str:
    """Lowercase alphanumeric-only form of a team name.

    Also expands college shorthand: "St" -> "State" (or "Saint" when first),
    plus a few common nicknames of schools (UConn, UMass, Pitt).
    """
    tokens = re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).split()
    tokens = [
        ("saint" if i == 0 else "state") if t == "st" else _ALIASES.get(t, t)
        for i, t in enumerate(tokens)
    ]
    return " ".join(tokens)


def team_match_score(a: Any, b: Any) -> int:
    """Score how well two team names match.

    2 = same name, nickname suffix ("Bears" / "Chicago Bears"), or school name plus
        a one-word nickname ("Virginia" / "Virginia Cavaliers");
    1 = school name plus a longer nickname ("Texas Tech" / "Texas Tech Red Raiders")
        or the same last word;
    0 = no match.
    """
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0
    ta, tb = na.split(), nb.split()
    short, long_ = sorted((ta, tb), key=len)
    if long_[-len(short):] == short:  # equal, or "bears" vs "chicago bears"
        return 2
    if long_[:len(short)] == short:  # DraftKings lists college teams without nicknames
        return 2 if len(long_) - len(short) == 1 else 1
    return int(ta[-1] == tb[-1])


def _start_gap_s(row: Mapping[str, Any], commence_iso: str) -> float:
    """Seconds between a row's event start and the game's start (huge if unknown)."""
    start = row.get("event_start_time")
    if not start:
        return 1e12
    try:
        delta = pd.to_datetime(start, utc=True) - pd.to_datetime(commence_iso, utc=True)
        return abs(delta.total_seconds())
    except (ValueError, TypeError):
        return 1e12


def match_split_rows(
    rows: Sequence[Mapping[str, Any]], away: str, home: str, commence_iso: str
) -> list[Mapping[str, Any]]:
    """Return all source rows for the splits event that matches this game."""
    scored = []
    for r in rows:
        s_away = max(team_match_score(r.get(k), away) for k in ("away_team", "away_raw"))
        s_home = max(team_match_score(r.get(k), home) for k in ("home_team", "home_raw"))
        if s_away and s_home:
            scored.append((s_away + s_home, -_start_gap_s(r, commence_iso), r))
    if not scored:
        return []
    best = max(scored, key=lambda t: (t[0], t[1]))
    best_id = best[2].get("event_id")
    return [r for _, _, r in scored if r.get("event_id") == best_id]


def _selection_side(
    market_type: str, selection: str, away: str, home: str
) -> Optional[str]:
    """Map an Odds API selection to a splits side key (away/home/over/under)."""
    if market_type == "totals":
        side = str(selection).strip().lower()
        return side if side in ("over", "under") else None
    if str(selection).strip().lower() == "draw":
        return "draw"
    s_away = team_match_score(selection, away)
    s_home = team_match_score(selection, home)
    if s_away == s_home:
        return None  # no match, or ambiguous
    return "away" if s_away > s_home else "home"


def _scaled(block: Mapping[str, Any], metric: str, side: str) -> Optional[float]:
    """Read block[metric][side] (0-1) as a 0-100 percentage, or None."""
    value = (block.get(metric) or {}).get(side)
    return None if value is None else float(value) * 100.0


def read_split(
    event_rows: Sequence[Mapping[str, Any]],
    market_type: str,
    selection: str,
    away: str,
    home: str,
    preferred: str = PREFERRED_BOOK,
) -> Optional[SplitReading]:
    """Pick the best source row that has BOTH ticket and handle for the selection.

    Returns None for unsupported markets (e.g. team_totals), unmatched
    selections, or when no source publishes both numbers.
    """
    market_key = {"spreads": "spread", "h2h": "moneyline", "totals": "total"}.get(
        market_type
    )
    side = _selection_side(market_type, selection, away, home)
    if market_key is None or side is None:
        return None

    order = (preferred, *[b for b in SPLIT_BOOK_PRIORITY if b != preferred])
    rank = {b: i for i, b in enumerate(order)}
    for row in sorted(event_rows, key=lambda r: rank.get(r.get("sportsbook"), 99)):
        block = row.get(market_key) or {}
        tickets = _scaled(block, "bets_pct", side)
        handle = _scaled(block, "handle_pct", side)
        if tickets is not None and handle is not None:
            return SplitReading(
                ticket_pct=round(tickets, 1),
                handle_pct=round(handle, 1),
                source=str(row.get("sportsbook", "unknown")),
                fetched_at=row.get("fetched_at"),
            )
    return None


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
