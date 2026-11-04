"""Free handle / bets splits from the DraftKings Network public splits page.

Source: https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/
The page lists, per game, Odds / Handle % / Bets % for Moneyline, Spread and
Total (all jurisdictions combined; the site refreshes it every few minutes).

Parsing is deliberately DOM-agnostic: the HTML is flattened to a stream of text
tokens and read with anchors that survive markup changes:
    game header  -> "AWAY @ HOME" immediately before a "Sun 11/8 · 4:05 PM ET" line
    market block -> "Moneyline" | "Spread" | "Total"
    selection    -> <name> <odds> <handle%> <bets%>      (column order on the page)

Rows are returned in the same shape as SharpAPI /splits rows (percentages as
0-1 fractions), so `match_split_rows` / `read_split` work on either source.
"""

from __future__ import annotations

import re
from datetime import datetime
from html.parser import HTMLParser
from typing import Any, Mapping, Optional
from urllib.parse import unquote_plus

import pandas as pd
import requests

from auditor_core import (
    DISPLAY_TZ,
    REQUEST_TIMEOUT_S,
    APIError,
    team_match_score,
)

DK_URL = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"
DK_MAX_PAGES = 12
DK_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SharpWhaleAuditor/1.0)",
    "Accept": "text/html,application/xhtml+xml",
}

_DATE_RE = re.compile(
    r"^[A-Za-z]{3,9},?\s+(\d{1,2})/(\d{1,2})\s*[·•|\-–]\s*"
    r"(\d{1,2}):(\d{2})\s*([AP]M)\b",
    re.IGNORECASE,
)
_ODDS_RE = re.compile(r"^[+-]\d{2,5}$")
_PCT_RE = re.compile(r"^(\d{1,3})%$")
_SPREAD_RE = re.compile(r"^(.*?)\s*([+-]\d+(?:\.\d+)?)$")
_TOTAL_RE = re.compile(r"^(over|under)\s+(\d+(?:\.\d+)?)$", re.IGNORECASE)
_ABBREV_RE = re.compile(r"^[A-Z]{2,5}$")
_MARKETS = {"moneyline": "moneyline", "spread": "spread", "total": "total"}
_EG_PATTERNS = (
    re.compile(r"tb_eg=(\d+)(?:&amp;|&)itm_content=([^&\"'\s<>]+)"),
    re.compile(r"itm_content=([^&\"'\s<>]+)(?:&amp;|&)[^\"'\s<>]*?tb_eg=(\d+)"),
)


class DKSplitsError(APIError):
    """Raised when the DraftKings Network page cannot be fetched or parsed."""


class _TokenParser(HTMLParser):
    """Flatten HTML into whitespace-normalised text nodes (no scripts/styles)."""

    _SKIP = {"script", "style", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tokens: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        self._skip_depth += tag in self._SKIP

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if text and not self._skip_depth:
            self.tokens.append(text)


def html_tokens(html: str) -> list[str]:
    """Return the page's text nodes, in document order."""
    parser = _TokenParser()
    parser.feed(html)
    return parser.tokens


def _norm_sign(text: str) -> str:
    """Replace unicode minus/en-dash so '−175' parses as '-175'."""
    return text.replace("\u2212", "-").replace("\u2013", "-").strip()


def strip_abbrev(name: str) -> str:
    """'CHI Bears' -> 'Bears', 'NY Jets' -> 'Jets'; leave single words alone."""
    parts = name.split()
    return " ".join(parts[1:]) if len(parts) > 1 and _ABBREV_RE.match(parts[0]) else name


def parse_start(label: str, now: Optional[datetime] = None) -> Optional[str]:
    """Turn 'Sun 11/8 · 4:05 PM ET' into a UTC ISO string (year inferred)."""
    m = _DATE_RE.match(label)
    if not m:
        return None
    month, day, hour, minute, meridiem = m.groups()
    now = now or datetime.now()
    hour_24 = int(hour) % 12 + (12 if meridiem.upper() == "PM" else 0)
    try:
        ts = pd.Timestamp(
            year=now.year, month=int(month), day=int(day),
            hour=hour_24, minute=int(minute), tz=DISPLAY_TZ,
        )
    except ValueError:
        return None
    if (pd.Timestamp(now, tz=DISPLAY_TZ) - ts).days > 180:  # Dec page showing Jan games
        ts = ts.replace(year=ts.year + 1)
    return ts.tz_convert("UTC").isoformat()


def _header_teams(tokens: list[str], date_idx: int) -> Optional[tuple[str, str]]:
    """Find 'AWAY @ HOME' in the few tokens right before a date line."""
    for start in range(date_idx - 1, max(date_idx - 5, -1), -1):
        if "@" in tokens[start]:
            # "@" can be its own text node: the away team is then the previous node
            first = start - 1 if tokens[start].lstrip().startswith("@") and start else start
            joined = " ".join(tokens[first:date_idx])
            away, _, home = joined.partition("@")
            if away.strip() and home.strip():
                return away.strip(), home.strip()
            return None
    return None


def _read_entries(section: list[str]) -> dict[str, list[tuple[str, int, int]]]:
    """Read (selection, handle%, bets%) triples under each market heading."""
    entries: dict[str, list[tuple[str, int, int]]] = {k: [] for k in _MARKETS}
    market: Optional[str] = None
    k = 0
    while k < len(section):
        heading = _MARKETS.get(section[k].lower())
        if heading:
            market, k = heading, k + 1
            continue
        if market and k + 3 < len(section):
            odds = _norm_sign(section[k + 1])
            handle, bets = _PCT_RE.match(section[k + 2]), _PCT_RE.match(section[k + 3])
            if _ODDS_RE.match(odds) and handle and bets:
                entries[market].append((section[k], int(handle[1]), int(bets[1])))
                k += 4
                continue
        k += 1
    return entries


def _side(name: str, away: str, home: str) -> Optional[str]:
    """Which side ('away'/'home') a selection name belongs to."""
    s_away, s_home = team_match_score(name, away), team_match_score(name, home)
    if s_away == s_home:
        return None
    return "away" if s_away > s_home else "home"


def _build_row(
    away_raw: str,
    home_raw: str,
    start_label: str,
    entries: Mapping[str, list[tuple[str, int, int]]],
    updated: Optional[str],
    now: Optional[datetime],
) -> Optional[dict[str, Any]]:
    """Convert one game's parsed entries into a SharpAPI-style splits row."""
    away, home = strip_abbrev(away_raw), strip_abbrev(home_raw)
    start_iso = parse_start(start_label, now)
    row: dict[str, Any] = {
        "event_id": f"dk_{away}_{home}_{start_iso or start_label}".lower().replace(" ", "_"),
        "sportsbook": "draftkings",
        "away_team": away,
        "home_team": home,
        "fetched_at": updated,
    }
    if start_iso:
        row["event_start_time"] = start_iso

    for market in ("moneyline", "spread"):
        block: dict[str, Any] = {"bets_pct": {}, "handle_pct": {}}
        for raw, handle, bets in entries[market]:
            name, line = raw, None
            if market == "spread":
                m = _SPREAD_RE.match(raw)
                if not m:
                    continue
                name, line = m[1], float(m[2])
            side = _side(strip_abbrev(name), away, home)
            if side is None:
                continue
            block["handle_pct"][side] = handle / 100.0
            block["bets_pct"][side] = bets / 100.0
            if line is not None:
                block[f"{side}_line"] = line
        if block["handle_pct"]:
            row[market] = block

    total: dict[str, Any] = {"bets_pct": {}, "handle_pct": {}}
    for raw, handle, bets in entries["total"]:
        m = _TOTAL_RE.match(raw)
        if m:
            side = m[1].lower()
            total["handle_pct"][side] = handle / 100.0
            total["bets_pct"][side] = bets / 100.0
            total["line"] = float(m[2])
    if total["handle_pct"]:
        row["total"] = total

    return row if {"moneyline", "spread", "total"} & row.keys() else None


def parse_dk_html(html: str, now: Optional[datetime] = None) -> list[dict[str, Any]]:
    """Parse one page of the splits table into SharpAPI-style rows."""
    tokens = html_tokens(html)
    updated = next(
        (t.split(None, 1)[1] for t in tokens if t.lower().startswith("updated ")), None
    )
    date_idx = [i for i, t in enumerate(tokens) if _DATE_RE.match(t)]
    rows: list[dict[str, Any]] = []
    for n, d in enumerate(date_idx):
        teams = _header_teams(tokens, d)
        if teams is None:
            continue
        end = date_idx[n + 1] if n + 1 < len(date_idx) else len(tokens)
        row = _build_row(
            teams[0], teams[1], tokens[d], _read_entries(tokens[d + 1:end]), updated, now
        )
        if row:
            rows.append(row)
    return rows


def discover_event_groups(html: str) -> dict[str, int]:
    """Map lower-cased league label -> tb_eg id from any links on the page."""
    found: dict[str, int] = {}
    for pattern, eg_first in zip(_EG_PATTERNS, (True, False)):
        for a, b in pattern.findall(html):
            eg, label = (a, b) if eg_first else (b, a)
            found[unquote_plus(label).strip().lower()] = int(eg)
    return found


def total_pages(html: str) -> int:
    """Highest tb_page number linked from the pagination (1 if none)."""
    return max([1, *(int(n) for n in re.findall(r"tb_page=(\d+)", html))])


def _get(params: Optional[Mapping[str, Any]] = None) -> str:
    """GET the splits page and return its HTML."""
    try:
        resp = requests.get(
            DK_URL, params=dict(params or {}), headers=DK_HEADERS, timeout=REQUEST_TIMEOUT_S
        )
    except requests.RequestException as exc:
        raise DKSplitsError(f"Could not reach DraftKings Network: {exc}") from exc
    if resp.status_code != 200:
        raise DKSplitsError(f"DraftKings Network returned HTTP {resp.status_code}.")
    return resp.text


def fetch_dk_splits(
    label: str, fallback_eg: int, now: Optional[datetime] = None
) -> list[dict[str, Any]]:
    """Fetch every page of splits for one league (next 7 days, all markets).

    Args:
        label: league label used by the site (e.g. "NBA", "England Premier League").
        fallback_eg: tb_eg id used if the page doesn't expose one for `label`.

    Raises:
        DKSplitsError: league not published, request failure, or nothing parsable.
    """
    if not label and not fallback_eg:
        raise DKSplitsError("DraftKings Network does not publish splits for this league.")
    eg = discover_event_groups(_get()).get(label.lower(), fallback_eg)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    page, last = 1, 1
    while page <= min(last, DK_MAX_PAGES):
        html = _get(
            {"tb_eg": eg, "itm_content": label, "tb_edate": "n7days", "tb_emt": "0", "tb_page": page}
        )
        last = max(last, total_pages(html))
        fresh = [r for r in parse_dk_html(html, now) if r["event_id"] not in seen]
        if not fresh:
            break
        seen.update(r["event_id"] for r in fresh)
        rows.extend(fresh)
        page += 1
    if not rows:
        raise DKSplitsError(
            "No games could be read from the DraftKings Network page "
            "(it may be JavaScript-rendered, the league id may be wrong, or the layout changed)."
        )
    return rows
