"""
Bet links: where a pick sends you on DraftKings.

The Odds API can return links when asked (`includeLinks=true`): one per selection (adds it to the bet slip), one per market and one per
bookmaker event. We use the deepest one that exists, and fall back to the DraftKings sportsbook home page, which the API never returns.
The app never places a bet: a link only opens DraftKings, where you confirm the bet yourself.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

DK_HOME = "https://sportsbook.draftkings.com/"
KIND_LABEL: Dict[str, str] = {"slip": "adds to bet slip", "market": "opens the market", "event": "opens the game", "home": "opens DraftKings"}


def _url(x: Any) -> Optional[str]:
    """x when it is a usable http(s) link, else None."""
    return x.strip() if isinstance(x, str) and x.strip().lower().startswith(("http://", "https://")) else None


def best_link(outcome: Mapping[str, Any], market: Optional[Mapping[str, Any]] = None, book: Optional[Mapping[str, Any]] = None) -> Tuple[str, str]:
    """(url, kind) using the deepest link available: selection > market > game > DraftKings home."""
    for kind, src in (("slip", outcome), ("market", market), ("event", book)):
        u = _url(src.get("link")) if src else None
        if u:
            return u, kind
    return DK_HOME, "home"


def market_map(book: Optional[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    """{market key -> market dict} of one bookmaker payload."""
    return {m.get("key"): m for m in (book or {}).get("markets", [])}


def link_columns(df: pd.DataFrame) -> bool:
    """True when the frame carries bet links."""
    return df is not None and "LINK" in df.columns


def real_links(links: Sequence[Tuple[str, str]]) -> int:
    """How many (url, kind) pairs are deeper than the DraftKings home page."""
    return sum(1 for _, k in links if k != "home")


def open_all_html(urls: Sequence[str], label: str = "Open all legs") -> str:
    """A button that opens every link in its own tab on one click (the browser must allow pop-ups for this site)."""
    import html
    import json
    safe = json.dumps([u for u in urls if _url(u)]).replace("</", "<\\/")
    return (f'<button id="b" style="padding:.5rem 1rem;border-radius:.5rem;border:1px solid #888;cursor:pointer;font-size:1rem">{html.escape(label)}</button>'
            f'<span id="m" style="margin-left:.75rem;font-family:sans-serif;font-size:.85rem"></span>'
            f'<script>const U={safe};document.getElementById("b").onclick=()=>{{let n=0;U.forEach(u=>{{if(window.open(u,"_blank"))n++}});'
            f'document.getElementById("m").textContent=n===U.length?"Opened "+n+" tabs.":"Opened "+n+" of "+U.length+". Allow pop-ups for this site and click again."}};</script>')
