"""Fetch and parse handle / bets splits from PlayerProps.ai."""

from __future__ import annotations

import json
import re
from datetime import datetime
from html.parser import HTMLParser
from typing import Any, Optional

import pandas as pd
import requests

from auditor_core import (
    DISPLAY_TZ,
    REQUEST_TIMEOUT_S,
    APIError,
)

PLAYERPROPS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

LEAGUE_PATHS: dict[str, list[str]] = {
    "NFL": ["Football/{date}/NFL"],
    "NCAA Football": ["Football/{date}/CFB", "Football/{date}/NCAAF"],
    "NBA": ["Basketball/{date}/NBA"],
    "NCAA Basketball": ["Basketball/{date}/CBB", "Basketball/{date}/NCAAB"],
    "NHL": ["Basketball/{date}/NHL", "Hockey/{date}/NHL"],
    "MLB": ["Baseball/{date}/MLB"],
    "EPL": ["Soccer/{date}/EPL"],
    "UCL": ["Soccer/{date}/UCL"],
    "Bundesliga": ["Soccer/{date}/BUNDESLIGA"],
    "La Liga": ["Soccer/{date}/LALIGA"],
}


class PlayerPropsSplitsError(APIError):
    """Raised when PlayerProps.ai page cannot be fetched or parsed."""


class _PPTokenParser(HTMLParser):
    """Extract text tokens and script contents (for __NEXT_DATA__)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tokens: list[str] = []
        self.next_data_json: Optional[dict] = None
        self._in_next_data = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        attr_dict = dict(attrs)
        if tag == "script" and attr_dict.get("id") == "__NEXT_DATA__":
            self._in_next_data = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_next_data = False

    def handle_data(self, data: str) -> None:
        if self._in_next_data:
            try:
                self.next_data_json = json.loads(data)
            except Exception:
                pass
        text = " ".join(data.split())
        if text:
            self.tokens.append(text)


def parse_playerprops_html(
    html: str, now: Optional[datetime] = None
) -> list[dict[str, Any]]:
    """Parse playerprops.ai HTML or __NEXT_DATA__ into standardized split rows."""
    parser = _PPTokenParser()
    parser.feed(html)
    rows: list[dict[str, Any]] = []

    if parser.next_data_json:
        rows.extend(_parse_next_data(parser.next_data_json, now))
        if rows:
            return rows

    return _parse_tokens(parser.tokens, now)


def _parse_next_data(
    data: dict[str, Any], now: Optional[datetime] = None
) -> list[dict[str, Any]]:
    """Recursively search Next.js props data for games with betting splits."""
    rows: list[dict[str, Any]] = []

    def find_games(node: Any) -> None:
        if isinstance(node, dict):
            if (
                ("away_team" in node or "awayTeam" in node or "away" in node)
                and ("home_team" in node or "homeTeam" in node or "home" in node)
                and (
                    "spread" in node
                    or "total" in node
                    or "moneyline" in node
                    or "splits" in node
                )
            ):
                row = _build_pp_row_from_dict(node, now)
                if row:
                    rows.append(row)
            for v in node.values():
                find_games(v)
        elif isinstance(node, list):
            for item in node:
                find_games(item)

    find_games(data)
    return rows


def _build_pp_row_from_dict(
    d: dict[str, Any], now: Optional[datetime] = None
) -> Optional[dict[str, Any]]:
    """Convert a dictionary game object from JSON into standardized split row."""
    away = str(d.get("away_team") or d.get("awayTeam") or d.get("away") or "").strip()
    home = str(d.get("home_team") or d.get("homeTeam") or d.get("home") or "").strip()
    if not away or not home:
        return None

    now_iso = (now or datetime.now()).isoformat()
    row: dict[str, Any] = {
        "event_id": f"pp_{away}_{home}".lower().replace(" ", "_"),
        "sportsbook": "playerprops",
        "away_team": away,
        "home_team": home,
        "fetched_at": "PlayerProps.ai",
        "event_start_time": d.get("start_time")
        or d.get("commence_time")
        or d.get("date")
        or now_iso,
    }

    for mkt in ("spread", "total", "moneyline"):
        m_data = (
            d.get(mkt)
            or d.get(f"{mkt}_splits")
            or (d.get("splits") or {}).get(mkt)
        )
        if isinstance(m_data, dict):
            row[mkt] = _format_market_block(mkt, m_data)

    return row if {"moneyline", "spread", "total"} & row.keys() else None


def _format_market_block(market: str, d: dict[str, Any]) -> dict[str, Any]:
    """Format market dictionary to standard shape with handle_pct and bets_pct."""
    block: dict[str, Any] = {"bets_pct": {}, "handle_pct": {}}

    for side_src in ("away", "home", "over", "under"):
        h = (
            d.get(f"{side_src}_handle_pct")
            or d.get(f"{side_src}_handle")
            or (d.get("handle") or {}).get(side_src)
        )
        b = (
            d.get(f"{side_src}_bets_pct")
            or d.get(f"{side_src}_bets")
            or (d.get("bets") or {}).get(side_src)
        )
        if h is not None:
            val = float(h)
            block["handle_pct"][side_src] = val / 100.0 if val > 1.0 else val
        if b is not None:
            val = float(b)
            block["bets_pct"][side_src] = val / 100.0 if val > 1.0 else val

    if market == "spread":
        if "away_line" in d:
            block["away_line"] = float(d["away_line"])
        if "home_line" in d:
            block["home_line"] = float(d["home_line"])
    elif market == "total":
        if "line" in d:
            block["line"] = float(d["line"])

    return block


def _parse_tokens(
    tokens: list[str], now: Optional[datetime] = None
) -> list[dict[str, Any]]:
    """Parse text tokens from HTML if JSON is not embedded."""
    rows: list[dict[str, Any]] = []
    k = 0
    while k < len(tokens):
        if (
            ("@" in tokens[k] or " vs " in tokens[k].lower())
            and k > 0
            and k + 5 < len(tokens)
        ):
            matchup = tokens[k]
            sep = "@" if "@" in matchup else " vs "
            away, _, home = matchup.partition(sep)
            away, home = away.strip(), home.strip()
            if away and home:
                row: dict[str, Any] = {
                    "event_id": f"pp_{away}_{home}".lower().replace(" ", "_"),
                    "sportsbook": "playerprops",
                    "away_team": away,
                    "home_team": home,
                    "fetched_at": "PlayerProps.ai",
                }
                rows.append(row)
        k += 1

    return rows


def fetch_playerprops_splits(
    league: str, now: Optional[datetime] = None
) -> list[dict[str, Any]]:
    """Fetch splits for a league from playerprops.ai."""
    paths = LEAGUE_PATHS.get(league)
    if not paths:
        raise PlayerPropsSplitsError(
            f"PlayerProps.ai splits route not configured for {league}."
        )

    now_dt = now or datetime.now()
    date_str = now_dt.strftime("%Y-%m-%d")

    rows: list[dict[str, Any]] = []
    last_error: Optional[Exception] = None

    for path_template in paths:
        url = f"https://playerprops.ai/betting-splits/{path_template.format(date=date_str)}"
        try:
            resp = requests.get(
                url, headers=PLAYERPROPS_HEADERS, timeout=REQUEST_TIMEOUT_S
            )
            if resp.status_code == 200:
                parsed = parse_playerprops_html(resp.text, now_dt)
                if parsed:
                    rows.extend(parsed)
                    break
            else:
                last_error = PlayerPropsSplitsError(
                    f"PlayerProps.ai returned HTTP {resp.status_code} for {url}"
                )
        except requests.RequestException as exc:
            last_error = PlayerPropsSplitsError(
                f"Could not reach PlayerProps.ai: {exc}"
            )

    if not rows and last_error:
        raise last_error
    elif not rows:
        raise PlayerPropsSplitsError(
            f"No betting splits could be read from PlayerProps.ai for {league}."
        )

    return rows