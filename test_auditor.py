"""Tests for the auditor. Run: python -m unittest -v test_auditor"""

from __future__ import annotations

import sys
import types
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import requests

import auditor_core as core
import dk_splits as dk
import playerprops_splits as pp

GAME = {
    "id": "evt1",
    "away_team": "Boston Celtics",
    "home_team": "New York Knicks",
    "commence_time": "2026-11-05T00:30:00Z",
    "bookmakers": [
        {"key": "fanduel", "title": "FanDuel", "last_update": "x", "markets": []},
        {
            "key": "draftkings",
            "title": "DraftKings",
            "last_update": "2026-11-04T12:00:00Z",
            "markets": [
                {
                    "key": "spreads",
                    "outcomes": [
                        {"name": "Boston Celtics", "price": -110, "point": -3.5},
                        {"name": "New York Knicks", "price": -110, "point": 3.5},
                    ],
                },
                {
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "price": -105, "point": 221.5},
                        {"name": "Under", "price": -115, "point": 221.5},
                    ],
                },
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Boston Celtics", "price": -170},
                        {"name": "New York Knicks", "price": 145},
                    ],
                },
            ],
        },
    ],
}
TEAM_TOTALS_BOOKS = [
    {
        "key": "draftkings",
        "title": "DraftKings",
        "markets": [
            {
                "key": "team_totals",
                "outcomes": [
                    {"name": "Over", "description": "Boston Celtics", "price": -115, "point": 112.5},
                    {"name": "Under", "description": "Boston Celtics", "price": -105, "point": 112.5},
                ],
            }
        ],
    }
]


def split_row(book="playerprops", spread=(0.30, 0.62), total=(0.55, 0.40), ml=(0.35, 0.60),
              start="2026-11-05T00:30:00Z", event_id="nba_knicks_celtics_2026-11-04"):
    def blk(pair, a, b):
        return {"bets_pct": {a: pair[0], b: 1 - pair[0]},
                "handle_pct": {a: pair[1], b: 1 - pair[1]}}
    return {
        "event_id": event_id, "sportsbook": book, "league": "nba",
        "away_team": "Boston Celtics", "home_team": "New York Knicks",
        "spread": {"away_line": -3.5, "home_line": 3.5, **blk(spread, "away", "home")},
        "total": {"line": 221.5, **blk(total, "over", "under")},
        "moneyline": {"away_odds": -170, "home_odds": 145, **blk(ml, "away", "home")},
        "fetched_at": "PlayerProps.ai", "event_start_time": start,
    }


SPLITS_ROWS = [
    split_row("playerprops", spread=(0.40, 0.70)),
    split_row("draftkings"),
]


def dk_game_html(away, home, when, ml, spread, total, line, a_line=None):
    def row(name, odds, h, b):
        return (f'<div class="r"><span>{name}</span><a href="https://x/bet">{odds}</a>'
                f"<span>{h}%</span><span>{b}%</span></div>")
    ml_rows = "".join(row(*r) for r in ml)
    sp_rows = "".join(row(*r) for r in spread)
    tt_rows = "".join(row(*r) for r in total)
    return (
        f'<h5><a href="https://sportsbook.draftkings.com/event/1"><img src="a.png"> {away} @ '
        f'<img src="b.png">{home}</a></h5><p>{when}</p>'
        "<div>Selection</div><div>Odds</div><div>Handle</div><div>Bets</div>"
        f"<div>Moneyline</div>{ml_rows}<div>Spread</div>{sp_rows}<div>Total</div>{tt_rows}"
    )


def dk_page(*games, pages=1, updated="Updated 1:47 PM ET"):
    nav = ('<a href="?tb_eg=88808&amp;itm_content=NFL&amp;tb_edate=n7days&amp;tb_emt=0">All</a>'
           "".join(f'<a href="?tb_eg=88808&itm_content=NFL&tb_page={n}">{n}</a>' for n in range(1, pages + 1)))
    return (
        "<html><head><script>var g='CHI @ SEA Mon 11/2 · 8:20 PM ET';</script></head><body>"
        f"<nav>{nav}</nav><p>{updated}</p>" + "".join(games) +
        "<ul><li><a>Bears at Seahawks NFL Player Prop Bets</a></li></ul><p>Advertisement</p></body></html>"
    )


NFL_GAME_1 = dk_game_html(
    "CHI Bears", "SEA Seahawks", "Mon 11/2 · 8:20 PM ET",
    ml=[("SEA Seahawks", "−175", 32, 55), ("CHI Bears", "+145", 68, 45)],
    spread=[("SEA Seahawks -3.5", "−102", 63, 53), ("CHI Bears +3.5", "−118", 37, 47)],
    total=[("Over 46.5", "−110", 63, 57), ("Under 46.5", "−110", 37, 43)], line=46.5)

DK_UI_HTML = dk_page(dk_game_html(
    "BOS Celtics", "NY Knicks", "Wed 11/4 · 7:30 PM ET",
    ml=[("BOS Celtics", "−170", 60, 35), ("NY Knicks", "+145", 40, 65)],
    spread=[("BOS Celtics -3.5", "−110", 62, 30), ("NY Knicks +3.5", "−110", 38, 70)],
    total=[("Over 221.5", "−105", 40, 55), ("Under 221.5", "−115", 60, 45)], line=221.5))


def audit(**kw):
    base = dict(
        sport_name="NBA", market_type="spreads", line_val=-3.5, ticket_pct=30.0,
        handle_pct=85.0, line_movement=-1.0, is_opening_move=False,
        is_closing_window=False,
    )
    base.update(kw)
    return core.audit_market(**base)


class TestAuditLogic(unittest.TestCase):
    def test_default_inputs_are_sharp(self):
        r = audit()
        self.assertTrue(r.is_sharp)
        self.assertTrue(r.checks["whale_divergence"])

    def test_key_numbers_for_all_sports(self):
        for name in core.SPORT_CONFIG:
            self.assertIsInstance(audit(sport_name=name, line_val=-1.5).is_sharp, bool)


class TestPlayerProps(unittest.TestCase):
    @patch("playerprops_splits.requests.get")
    def test_fetch_playerprops_splits(self, mock_get):
        html_payload = """
        <html>
            <script id="__NEXT_DATA__" type="application/json">
            {
                "props": {
                    "pageProps": {
                        "splits": [
                            {
                                "away_team": "Boston Celtics",
                                "home_team": "New York Knicks",
                                "spread": {
                                    "away_handle_pct": 70,
                                    "away_bets_pct": 40,
                                    "away_line": -3.5
                                }
                            }
                        ]
                    }
                }
            }
            </script>
        </html>
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = html_payload
        mock_get.return_value = mock_resp

        rows = pp.fetch_playerprops_splits("NBA")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["away_team"], "Boston Celtics")
        self.assertEqual(rows[0]["spread"]["handle_pct"]["away"], 0.7)


class TestDkParser(unittest.TestCase):
    def test_strip_abbrev(self):
        self.assertEqual(dk.strip_abbrev("CHI Bears"), "Bears")
        self.assertEqual(dk.strip_abbrev("(1) Kansas"), "Kansas")
        self.assertEqual(dk.strip_abbrev("Duke [W]"), "Duke")

    def test_parse_start_without_weekday(self):
        self.assertIsNotNone(dk.parse_start("11/5 · 7:00 PM ET", datetime(2026, 11, 5)))


class FakeSt(types.ModuleType):
    def __init__(self, picks=None, key="KEY", checks=None):
        super().__init__("streamlit")
        self.calls: list[tuple[str, tuple, dict]] = []
        self.picks = picks or {}
        self.keys = {"The Odds API Key": key}
        self.checks = checks or {}
        self.sidebar = MagicMock()
        self.secrets = {}

    def cache_data(self, *a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]

    def _rec(self, name, *a, **k):
        self.calls.append((name, a, k))

    def selectbox(self, label, options, format_func=str, **k):
        opts = list(options)
        labels = [format_func(o) for o in opts]
        self._rec("selectbox", label, labels)
        want = self.picks.get(label)
        return opts[labels.index(want)] if want else opts[0]

    def text_input(self, label, value="", **k):
        return self.keys.get(label, value)

    def expander(self, *a, **k):
        return MagicMock()

    def slider(self, label, min_value=0, max_value=1, value=0, **k):
        return value

    def number_input(self, label, value=0, **k):
        return value

    def checkbox(self, label, **k):
        return self.checks.get(label, False)

    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [MagicMock() for _ in range(n)]

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: self._rec(name, *a, **k)


def run_app(picks=None, key="KEY", get_side_effect=None, checks=None):
    fake = FakeSt(picks, key, checks)
    sys.modules["streamlit"] = fake
    sys.modules.pop("app", None)
    import app

    with patch("auditor_core.requests.get", side_effect=get_side_effect):
        app.main()
    return fake


def router(url, params=None, timeout=None, headers=None):
    if "dknetwork.draftkings.com" in url:
        return resp(text=DK_UI_HTML)
    if "playerprops.ai" in url:
        return resp(text="<html></html>")
    if url.endswith("/odds/") or url.endswith("/odds"):
        if "/events/" in url:
            return resp(payload={"bookmakers": TEAM_TOTALS_BOOKS})
        return resp(payload=[GAME], headers={"x-requests-remaining": "99"})
    return resp(status=404, payload={"message": "nope"})


def resp(status=200, payload=None, headers=None, text="", json_raises=False):
    m = MagicMock()
    m.status_code = status
    m.headers = headers or {}
    m.text = text
    if json_raises:
        m.json.side_effect = ValueError("no json")
    else:
        m.json.return_value = payload
    return m


class TestUiSmoke(unittest.TestCase):
    def test_happy_path(self):
        fake = run_app(get_side_effect=router)
        self.assertIn("table", [c[0] for c in fake.calls])


if __name__ == "__main__":
    unittest.main()
