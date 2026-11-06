"""Free handle / bets splits from the DraftKings Network public splits page.

Source: https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/
The page lists, per game, Odds / Handle % / Bets % for Moneyline, Spread and
Total (all jurisdictions combined; the site refreshes it every few minutes).

Parsing is deliberately DOM-agnostic: the HTML is flattened to a stream of text
tokens and read with anchors that survive markup changes:
    game header  -> "AWAY @ HOME" immediately before a "Sun 11/8 · 4:05 PM ET" line
    market block -> "Moneyline" | "Spread" | "Total"
    selection    -> <name> <odds> <handle%> <bets%>      (column order on the page)

Rows use a common shape (percentages as
0-1 fractions) consumed by `match_split_rows` / `read_split` in auditor_core.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from typing import Any, Callable, Mapping, Optional
from urllib.parse import parse_qs, unquote_plus, urlparse

import pandas as pd
import requests

from auditor_core import (
    DISPLAY_TZ,
    REQUEST_TIMEOUT_S,
    APIError,
    team_match_score,
)

DK_URL = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"
DK_MAX_PAGES = 20
DK_WINDOWS = ("today", "n7days")  # tb_edate values seen in the site's own URLs
DK_HORIZON_DAYS = 4  # stop paging once a page starts beyond this window
DK_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SharpWhaleAuditor/1.0)",
    "Accept": "text/html,application/xhtml+xml",
}

_DATE_RE = re.compile(
    r"^[A-Za-z]{3,9},?\s+(\d{1,2})/(\d{1,2})\s*[·•|\-–]\s*"
    r"(\d{1,2}):(\d{2})\s*([AP]M)\b",
    re.IGNORECASE,
)
_ODDS_RE = re.compile(r"^(?:[+-]\d{2,5}|EVEN|EV)$", re.IGNORECASE)
_PCT_RE = re.compile(r"^<?\s*(\d{1,3}(?:\.\d+)?)\s*%$")
_SPREAD_RE = re.compile(r"^(.*?)\s*([+-]\d+(?:\.\d+)?)$")
_TOTAL_RE = re.compile(r"^(over|under)\s+(\d+(?:\.\d+)?)$", re.IGNORECASE)
_ABBREV_RE = re.compile(r"^[A-Z]{2,5}$")
# Heading text varies by sport ("Puck Line", "Run Line", "Match Result", ...).
_MARKET_PATTERNS = (
    ("moneyline", re.compile(r"^(money ?line|match (result|winner)|3-?way( money ?line| result)?|1x2)$", re.I)),
    ("spread", re.compile(r"^((point |puck |run |goal )?(spread|line)|(.* )?handicap)$", re.I)),
    ("total", re.compile(r"^(total( (points|goals|runs|games))?|over/under|game total)$", re.I)),
)
_MARKET_KEYS = tuple(k for k, _ in _MARKET_PATTERNS)
_ID_RE = re.compile(r"(?<!\d)(\d{4,6})(?!\d)")
_VOID_TAGS = {"img", "br", "hr", "input", "meta", "link", "source", "area", "base", "col", "embed", "param", "track", "wbr"}
KNOWN_LABELS = (
    "NFL", "NBA", "NHL", "MLB", "WNBA", "UFC", "UFL", "NCAA Football", "NCAA Basketball",
    "NCAA Womens Basketball", "NCAA Baseball", "NCAA Ice Hockey", "England Premier League",
    "Champions League", "Europa League", "MLS",
)
_EG_PATTERNS = (
    re.compile(r"tb_eg=(\d+)(?:&amp;|&)itm_content=([^&\"'\s<>]+)"),
    re.compile(r"itm_content=([^&\"'\s<>]+)(?:&amp;|&)[^\"'\s<>]*?tb_eg=(\d+)"),
)


class DKSplitsError(APIError):
    """Raised when the DraftKings Network page cannot be fetched or parsed."""

    def __init__(self, message: str, report: Optional["DKReport"] = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class DKReport:
    """What a fetch tried: (league id, origin, new games read) per attempt."""

    label: str
    attempts: tuple[tuple[int, str, int], ...]
    clue: str
    pages: int = 0
    span: str = ""


@dataclass(frozen=True)
class DKResult:
    """Parsed rows plus the report describing how they were obtained."""

    rows: list[dict[str, Any]]
    report: DKReport


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


def _market_key(token: str) -> Optional[str]:
    """Normalise a market heading ('Puck Line' -> 'spread'), else None."""
    text = token.strip()
    return next((key for key, pat in _MARKET_PATTERNS if pat.match(text)), None)


def _read_entries(section: list[str]) -> dict[str, list[tuple[str, float, float]]]:
    """Read (selection, handle%, bets%) triples under each market heading."""
    entries: dict[str, list[tuple[str, int, int]]] = {k: [] for k in _MARKET_KEYS}
    market: Optional[str] = None
    k = 0
    while k < len(section):
        heading = _market_key(section[k])
        if heading:
            market, k = heading, k + 1
            continue
        if market and k + 3 < len(section):
            odds = _norm_sign(section[k + 1])
            handle, bets = _PCT_RE.match(section[k + 2]), _PCT_RE.match(section[k + 3])
            if _ODDS_RE.match(odds) and handle and bets:
                entries[market].append((section[k], float(handle[1]), float(bets[1])))
                k += 4
                continue
        k += 1
    return entries


def _side(name: str, away: str, home: str) -> Optional[str]:
    """Which side ('away'/'home'/'draw') a selection name belongs to."""
    if name.strip().lower() == "draw":
        return "draw"
    s_away, s_home = team_match_score(name, away), team_match_score(name, home)
    if s_away == s_home:
        return None
    return "away" if s_away > s_home else "home"


def _build_row(
    away_raw: str,
    home_raw: str,
    start_label: str,
    entries: Mapping[str, list[tuple[str, float, float]]],
    updated: Optional[str],
    now: Optional[datetime],
) -> Optional[dict[str, Any]]:
    """Convert one game's parsed entries into a splits row (0-1 fractions)."""
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
    """Parse one page of the splits table into splits rows."""
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


class _LabelIdParser(HTMLParser):
    """Find numeric ids carried by the elements that wrap a league label.

    Only `value`, `data-*` and `tb_eg=` inside links are trusted; `id` / `class`
    are ignored because CMS menus put unrelated numbers there.
    """

    def __init__(self, labels: tuple[str, ...]) -> None:
        super().__init__(convert_charrefs=True)
        self.labels = {label.lower() for label in labels}
        self.stack: list[list[int]] = []
        self.found: dict[str, int] = {}

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in _VOID_TAGS:
            return
        ids: list[int] = []
        for name, value in attrs:
            if not value:
                continue
            if name in ("href", "data-href", "data-url"):
                ids += [int(n) for n in re.findall(r"tb_eg=(\d+)", value)]
            elif name == "value" or name.startswith("data-"):
                ids += [int(n) for n in _ID_RE.findall(value)]
        self.stack.append(ids)

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        return  # self-closing tags never wrap text

    def handle_endtag(self, tag: str) -> None:
        if self.stack and tag not in _VOID_TAGS:
            self.stack.pop()

    def handle_data(self, data: str) -> None:
        key = " ".join(data.split()).lower()
        if key in self.labels and key not in self.found:
            ids = next((i for i in reversed(self.stack[-3:]) if i), None)
            if ids:
                self.found[key] = ids[0]


def discover_event_groups(html: str, labels: tuple[str, ...] = KNOWN_LABELS) -> dict[str, int]:
    """Map lower-cased league label -> tb_eg id using links, then dropdown markup."""
    found: dict[str, int] = {}
    for pattern, eg_first in zip(_EG_PATTERNS, (True, False)):
        for a, b in pattern.findall(html):
            eg, label = (a, b) if eg_first else (b, a)
            found[unquote_plus(label).strip().lower()] = int(eg)
    parser = _LabelIdParser(labels)
    parser.feed(html)
    return {**parser.found, **found}


def markup_clue(html: str, label: str) -> str:
    """Raw HTML around the league label, preferring the sports dropdown.

    A label also appears in site navigation, so the occurrence that has other
    league names nearby (the dropdown) is chosen over the first one.
    """
    hits = list(re.finditer(rf">\s*{re.escape(label)}\s*<", html, re.IGNORECASE))
    if not hits:
        return "(league label not found in the raw HTML)"

    def neighbours(m: "re.Match[str]") -> int:
        window = html[max(0, m.start() - 400): m.end() + 400].lower()
        return sum(k.lower() in window for k in KNOWN_LABELS if k.lower() != label.lower())

    best = max(hits, key=lambda m: (neighbours(m) >= 2, -m.start()))
    return html[max(0, best.start() - 200): best.end() + 200]


def describe_report(report: Optional[DKReport]) -> str:
    """Plain-text summary of a fetch attempt for the diagnostics panel."""
    if report is None:
        return "No fetch report available."
    lines = [f"League label: {report.label or '(none)'}"]
    lines += [f"Tried league id {eg} ({origin}): {n} new games read" for eg, origin, n in report.attempts]
    if report.pages:
        lines.append(f"Pages read: {report.pages} (date windows: {', '.join(DK_WINDOWS)})")
    if report.span:
        lines.append(f"Games on the page start between: {report.span}")
    lines.append(f"Page markup near the league label:\n{report.clue}")
    return "\n".join(lines)


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


def _beyond_horizon(rows: list[dict[str, Any]], now: datetime) -> bool:
    """True if every dated row on a page starts after the paging horizon."""
    starts = [pd.Timestamp(r["event_start_time"]) for r in rows if r.get("event_start_time")]
    if not starts:
        return False
    cutoff = pd.Timestamp(now, tz=DISPLAY_TZ) + pd.Timedelta(days=DK_HORIZON_DAYS)
    return min(starts) > cutoff


def _fetch_group(
    label: str, eg: int, now: Optional[datetime], seen: set[str]
) -> tuple[list[dict[str, Any]], int]:
    """Read every date window and page for one league id, skipping games in `seen`.

    The site's own links use ``tb_edate=today`` for some leagues and ``n7days``
    for others, and a league can come back empty under one of them, so both
    windows are read and merged. Within a window, pages are requested until one
    yields nothing new (pagination links are not always in the raw HTML) or the
    page starts beyond DK_HORIZON_DAYS.

    Returns:
        (new rows, number of pages that contained games)
    """
    now = now or _now_et()
    rows: list[dict[str, Any]] = []
    pages = 0
    for window in DK_WINDOWS:
        page = 1
        while page <= DK_MAX_PAGES:
            html = _get(
                {"tb_eg": eg, "itm_content": label, "tb_edate": window, "tb_emt": "0", "tb_page": page}
            )
            fresh = [r for r in parse_dk_html(html, now) if r["event_id"] not in seen]
            if not fresh:
                break
            seen.update(r["event_id"] for r in fresh)
            rows.extend(fresh)
            pages += 1
            page += 1
            if _beyond_horizon(fresh, now):
                break
    return rows, pages


def _now_et() -> datetime:
    """Current Eastern wall-clock time as a naive datetime."""
    return pd.Timestamp.now(tz=DISPLAY_TZ).tz_localize(None).to_pydatetime()


def _span(rows: list[dict[str, Any]]) -> str:
    """'first start -> last start' (Eastern) for the rows that have start times."""
    starts = sorted(pd.Timestamp(r["event_start_time"]) for r in rows if r.get("event_start_time"))
    if not starts:
        return ""
    fmt = "%a %b %d %I:%M %p ET"
    return f"{starts[0].tz_convert(DISPLAY_TZ).strftime(fmt)} -> {starts[-1].tz_convert(DISPLAY_TZ).strftime(fmt)}"


def fetch_dk_splits(
    label: str, fallback_eg: int, now: Optional[datetime] = None
) -> DKResult:
    """Fetch splits for one league (next 7 days, all markets).

    The configured league id is tried first; any different id the page itself
    exposes for `label` (links, dropdown values, data-* attributes) is tried as
    well and the games are merged, so a wrong configured id still works.

    Args:
        label: league label used by the site (e.g. "NBA", "England Premier League").
        fallback_eg: configured tb_eg id (0 = unknown, rely on discovery).

    Raises:
        DKSplitsError: league not published, request failure, or nothing parsable
            (the exception carries a `report` of what was tried).
    """
    if not label and not fallback_eg:
        raise DKSplitsError("DraftKings Network does not publish splits for this league.")
    base_html = _get()
    discovered = discover_event_groups(base_html).get(label.lower(), 0)
    candidates = [(eg, origin) for eg, origin in ((fallback_eg, "configured"), (discovered, "discovered")) if eg]
    candidates = [c for i, c in enumerate(candidates) if c[0] not in {x[0] for x in candidates[:i]}]

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts: list[tuple[int, str, int]] = []
    pages = 0
    for eg, origin in candidates:
        got, n_pages = _fetch_group(label, eg, now, seen)
        attempts.append((eg, origin, len(got)))
        rows.extend(got)
        pages += n_pages
    report = DKReport(
        label=label, attempts=tuple(attempts), clue=markup_clue(base_html, label),
        pages=pages, span=_span(rows),
    )
    if not rows:
        raise DKSplitsError(
            "No games could be read from the DraftKings Network page "
            f"for {label!r}: it may have no games in the next 7 days, the league id "
            "may be wrong, or the layout changed.",
            report,
        )
    return DKResult(rows=rows, report=report)


def coverage_report(
    leagues: Mapping[str, tuple[str, int]],
    fetch: Callable[[str, int], DKResult] = fetch_dk_splits,
) -> pd.DataFrame:
    """Try every league and summarise what the DraftKings page returned.

    Args:
        leagues: display name -> (site label, configured league id).
        fetch: injectable fetcher (the app passes its cached one).

    Returns:
        One row per league: ids tried, games read, first matchup, status.
    """
    rows: list[dict[str, Any]] = []
    for name, (label, eg) in leagues.items():
        try:
            res = fetch(label, eg)
            first = res.rows[0]
            rows.append({
                "League": name,
                "League id(s) tried": ", ".join(str(a[0]) for a in res.report.attempts),
                "Games": len(res.rows),
                "First game": f"{first['away_team']} @ {first['home_team']}",
                "Status": "OK",
            })
        except DKSplitsError as exc:
            attempts = exc.report.attempts if exc.report else ()
            rows.append({
                "League": name,
                "League id(s) tried": ", ".join(str(a[0]) for a in attempts) or str(eg or "-"),
                "Games": 0,
                "First game": "",
                "Status": str(exc)[:110],
            })
    return pd.DataFrame(rows)


def parse_dk_override(text: str, fallback_label: str = "") -> Optional[tuple[str, int]]:
    """Read a league id the user copied from the DraftKings page.

    Accepts the page URL after choosing the league in the site's dropdown
    (``...?tb_eg=42133&itm_content=NHL&...``) or just the number.

    Returns:
        (label, tb_eg) or None if no id could be read.
    """
    text = (text or "").strip()
    if not text:
        return None
    if text.isdigit():
        return fallback_label, int(text)
    query = parse_qs(urlparse(text).query)
    eg = next((v for v in query.get("tb_eg", []) if v.isdigit()), None)
    if eg is None:
        m = re.search(r"tb_eg=(\d+)", text)  # tolerate pasted fragments
        eg = m[1] if m else None
    if eg is None:
        return None
    label = unquote_plus((query.get("itm_content") or [fallback_label])[0])
    return label or fallback_label, int(eg)
