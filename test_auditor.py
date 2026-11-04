"""Tests for the auditor. Run: python -m unittest -v test_auditor  (or pytest)."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import requests

import auditor_core as core

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


def split_row(book, spread=(0.30, 0.62), total=(0.55, 0.40), ml=(0.35, 0.60),
              start="2026-11-05T00:30:00Z", event_id="nba_knicks_celtics_2026-11-04"):
    """Build a SharpAPI splits row. Tuples are (away bets, away handle)."""
    def blk(pair, a, b):
        return {"bets_pct": {a: pair[0], b: 1 - pair[0]},
                "handle_pct": {a: pair[1], b: 1 - pair[1]}}
    return {
        "event_id": event_id, "sportsbook": book, "league": "nba",
        "away_team": "Boston Celtics", "home_team": "New York Knicks",
        "spread": {"away_line": -3.5, "home_line": 3.5, **blk(spread, "away", "home")},
        "total": {"line": 221.5, **blk(total, "over", "under")},
        "moneyline": {"away_odds": -170, "home_odds": 145, **blk(ml, "away", "home")},
        "fetched_at": "2026-11-04T17:00:00Z", "event_start_time": start,
    }


SPLITS_ROWS = [
    split_row("circa", spread=(0.40, 0.70)),
    split_row("draftkings"),
    {**split_row("betmgm"), "event_id": "nba_other_game_2026-11-04",
     "away_team": "Miami Heat", "home_team": "Chicago Bulls"},
]


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
        self.assertTrue(r.checks["reverse_line_movement"])
        self.assertTrue(r.checks["steam_movement"])
        self.assertTrue(r.checks["key_number_resistance"])
        self.assertEqual((r.passed, r.applicable), (4, 6))

    def test_whale_is_directional(self):
        # Public heavy on tickets AND money is the reverse of a whale signal.
        self.assertFalse(audit(ticket_pct=85, handle_pct=30).checks["whale_divergence"])
        self.assertTrue(audit(ticket_pct=30, handle_pct=45).checks["whale_divergence"])
        self.assertFalse(audit(ticket_pct=30, handle_pct=44.9).checks["whale_divergence"])

    def test_whale_alone_is_not_enough(self):
        r = audit(line_val=-12.5, line_movement=0.0, ticket_pct=50, handle_pct=70)
        self.assertTrue(r.checks["whale_divergence"])
        self.assertEqual(r.passed, 1)
        self.assertFalse(r.is_sharp)

    def test_no_whale_never_sharp(self):
        r = audit(ticket_pct=50, handle_pct=55, is_opening_move=True, line_movement=-2)
        self.assertFalse(r.is_sharp)

    def test_rlm_both_directions(self):
        self.assertTrue(audit(ticket_pct=75, line_movement=0.5).checks["reverse_line_movement"])
        self.assertFalse(audit(ticket_pct=75, line_movement=-0.5).checks["reverse_line_movement"])
        self.assertFalse(audit(ticket_pct=50, line_movement=-2).checks["reverse_line_movement"])
        self.assertFalse(audit(ticket_pct=20, line_movement=0.0).checks["reverse_line_movement"])

    def test_steam_and_late_thresholds(self):
        self.assertTrue(audit(line_movement=-1.0).checks["steam_movement"])
        self.assertFalse(audit(line_movement=-0.5).checks["steam_movement"])
        self.assertTrue(audit(line_movement=-0.5, is_closing_window=True).checks["late_market_limit"])
        self.assertFalse(audit(line_movement=-0.4, is_closing_window=True).checks["late_market_limit"])
        self.assertFalse(audit(line_movement=-2, is_closing_window=False).checks["late_market_limit"])

    def test_key_number_uses_absolute_line(self):
        # Original code compared a signed line, so -3.5 never matched the key 3.
        self.assertTrue(core.key_number_hit(-3.5, (3, 7), 0.5))
        self.assertTrue(core.key_number_hit(3.5, (3, 7), 0.5))
        self.assertTrue(core.key_number_hit(-7, (3, 7), 0.5))
        self.assertFalse(core.key_number_hit(-12.5, (3, 7, 10), 0.5))
        self.assertTrue(core.key_number_hit(-1.5, (1.5,), 0.0))
        self.assertFalse(core.key_number_hit(-2.5, (1.5,), 0.0))

    def test_key_numbers_only_for_spreads(self):
        for mkt in ("totals", "h2h", "team_totals"):
            r = audit(market_type=mkt, line_val=None)
            self.assertIsNone(r.checks["key_number_resistance"])
            self.assertEqual(r.applicable, 5)

    def test_moneyline_no_longer_gets_free_point(self):
        r = audit(market_type="h2h", line_val=None, ticket_pct=50, handle_pct=70,
                  line_movement=0.0)
        self.assertEqual(r.passed, 1)
        self.assertFalse(r.is_sharp)

    def test_spread_without_point(self):
        self.assertFalse(audit(line_val=None).checks["key_number_resistance"])

    def test_all_sports_configured(self):
        for name in core.SPORT_CONFIG:
            self.assertIsInstance(audit(sport_name=name, line_val=-1.5).is_sharp, bool)


class TestHelpers(unittest.TestCase):
    def test_format_price(self):
        self.assertEqual(core.format_price(150), "+150")
        self.assertEqual(core.format_price(-110), "-110")
        self.assertEqual(core.format_price(1.91), "+2")  # decimal-format guard
        self.assertEqual(core.format_price(None), "None")

    def test_commence_is_converted_to_eastern(self):
        self.assertEqual(core.format_commence("2026-11-05T00:30:00Z"), "Wed Nov 04, 07:30 PM ET")
        # DST (EDT) date
        self.assertEqual(core.format_commence("2026-07-01T23:00:00Z"), "Wed Jul 01, 07:00 PM ET")

    def test_labels(self):
        self.assertEqual(core.game_label(GAME), "Boston Celtics @ New York Knicks (Wed Nov 04, 07:30 PM ET)")
        self.assertEqual(core.outcome_label({"name": "Over", "price": -105, "point": 221.5}), "Over 221.5 (-105)")
        self.assertEqual(core.outcome_label({"name": "Boston Celtics", "price": -170}), "Boston Celtics (-170)")
        self.assertEqual(
            core.outcome_label({"name": "Over", "description": "Boston Celtics", "price": -115, "point": 112.5}),
            "Boston Celtics Over 112.5 (-115)",
        )

    def test_pick_bookmaker(self):
        self.assertEqual(core.pick_bookmaker(GAME["bookmakers"])["key"], "draftkings")
        self.assertEqual(core.pick_bookmaker(GAME["bookmakers"][:1])["key"], "fanduel")
        self.assertIsNone(core.pick_bookmaker([]))

    def test_checklist_table_shape(self):
        r = audit(market_type="totals", line_val=221.5)
        df = core.checklist_table(r, "NBA", 30, 85, -1.0, False, False)
        self.assertEqual(df.shape, (6, 3))
        self.assertIn("➖ N/A", set(df["Status"]))


class TestSplits(unittest.TestCase):
    def match(self, rows=SPLITS_ROWS):
        return core.match_split_rows(rows, "Boston Celtics", "New York Knicks", GAME["commence_time"])

    def test_team_matching(self):
        self.assertEqual(core.team_match_score("Boston Celtics", "boston celtics"), 2)
        self.assertEqual(core.team_match_score("LA Clippers", "Los Angeles Clippers"), 1)
        self.assertEqual(core.team_match_score("LA Lakers", "LA Clippers"), 0)
        self.assertEqual(core.team_match_score(None, "x"), 0)

    def test_match_rows_picks_right_event(self):
        rows = self.match()
        self.assertEqual({r["sportsbook"] for r in rows}, {"circa", "draftkings"})

    def test_match_requires_orientation(self):
        self.assertEqual(core.match_split_rows(SPLITS_ROWS, "New York Knicks", "Boston Celtics", GAME["commence_time"]), [])

    def test_doubleheader_uses_start_time(self):
        g1 = split_row("draftkings", start="2026-11-05T00:30:00Z", event_id="g1")
        g2 = split_row("draftkings", start="2026-11-05T04:00:00Z", event_id="g2")
        got = core.match_split_rows([g1, g2], "Boston Celtics", "New York Knicks", "2026-11-05T04:05:00Z")
        self.assertEqual(got[0]["event_id"], "g2")

    def test_read_split_scales_and_prefers_draftkings(self):
        r = core.read_split(self.match(), "spreads", "Boston Celtics", "Boston Celtics", "New York Knicks")
        self.assertEqual((r.ticket_pct, r.handle_pct, r.source), (30.0, 62.0, "draftkings"))

    def test_read_split_home_side_and_total_and_ml(self):
        rows = self.match()
        home = core.read_split(rows, "spreads", "New York Knicks", "Boston Celtics", "New York Knicks")
        self.assertEqual((home.ticket_pct, home.handle_pct), (70.0, 38.0))
        over = core.read_split(rows, "totals", "Over", "Boston Celtics", "New York Knicks")
        self.assertEqual((over.ticket_pct, over.handle_pct), (55.0, 40.0))
        ml = core.read_split(rows, "h2h", "New York Knicks", "Boston Celtics", "New York Knicks")
        self.assertEqual((ml.ticket_pct, ml.handle_pct), (65.0, 40.0))

    def test_unsupported_or_unmatched(self):
        rows = self.match()
        args = ("Boston Celtics", "New York Knicks")
        self.assertIsNone(core.read_split(rows, "team_totals", "Over", *args))
        self.assertIsNone(core.read_split(rows, "h2h", "Draw", *args))
        self.assertIsNone(core.read_split(rows, "totals", "Push", *args))

    def test_betmgm_only_has_no_handle_so_it_is_skipped(self):
        mgm = split_row("betmgm")
        mgm["spread"]["handle_pct"] = {"away": None, "home": None}
        self.assertIsNone(core.read_split([mgm], "spreads", "Boston Celtics", "Boston Celtics", "New York Knicks"))
        dk = split_row("draftkings")
        r = core.read_split([mgm, dk], "spreads", "Boston Celtics", "Boston Celtics", "New York Knicks")
        self.assertEqual(r.source, "draftkings")

    def test_consensus_row_used_when_no_dk(self):
        r = core.read_split([split_row("consensus")], "spreads", "Boston Celtics", "Boston Celtics", "New York Knicks")
        self.assertEqual(r.source, "consensus")

    @patch("auditor_core.requests.get")
    def test_fetch_splits_header_params_pagination(self, get):
        get.side_effect = [
            resp(payload={"data": [SPLITS_ROWS[0]], "pagination": {"has_more": True, "next_offset": 200}}),
            resp(payload={"data": [SPLITS_ROWS[1]], "pagination": {"has_more": False, "next_offset": None}}),
        ]
        rows = core.fetch_splits("nba", "SHARPKEY")
        self.assertEqual(len(rows), 2)
        first = get.call_args_list[0]
        self.assertEqual(first.kwargs["headers"], {"X-API-Key": "SHARPKEY"})
        self.assertEqual(first.kwargs["params"]["league"], "nba")
        self.assertEqual(get.call_args_list[1].kwargs["params"]["offset"], 200)

    @patch("auditor_core.requests.get")
    def test_fetch_splits_tier_error_nested_message(self, get):
        get.return_value = resp(status=403, payload={"error": {"code": "tier_restricted", "message": "Requires Pro tier"}})
        with self.assertRaisesRegex(core.SharpAPIError, "SharpAPI error \\(403\\).*Pro tier"):
            core.fetch_splits("nba", "k")

    @patch("auditor_core.requests.get", side_effect=requests.ConnectionError("down"))
    def test_fetch_splits_network(self, _):
        with self.assertRaises(core.SharpAPIError):
            core.fetch_splits("nba", "k")

    def test_every_league_has_splits_slug(self):
        for name, cfg in core.SPORT_CONFIG.items():
            self.assertTrue(cfg.splits_league, name)


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


class TestApiLayer(unittest.TestCase):
    @patch("auditor_core.requests.get")
    def test_featured_params_and_remaining(self, get):
        get.return_value = resp(payload=[GAME], headers={"x-requests-remaining": "480"})
        games, remaining = core.fetch_featured_odds("basketball_nba", "KEY")
        self.assertEqual((len(games), remaining), (1, "480"))
        _, kwargs = get.call_args
        p = kwargs["params"]
        self.assertEqual(p["apiKey"], "KEY")
        self.assertNotIn("api_key", p)
        self.assertEqual(p["oddsFormat"], "american")
        self.assertNotIn("team_totals", p["markets"])
        self.assertEqual(kwargs["timeout"], core.REQUEST_TIMEOUT_S)

    @patch("auditor_core.requests.get")
    def test_error_message_from_json(self, get):
        get.return_value = resp(status=401, payload={"message": "Invalid key"})
        with self.assertRaisesRegex(core.OddsAPIError, "401.*Invalid key"):
            core.fetch_featured_odds("x", "bad")

    @patch("auditor_core.requests.get")
    def test_error_with_non_json_body(self, get):
        get.return_value = resp(status=502, text="Bad gateway", json_raises=True)
        with self.assertRaisesRegex(core.OddsAPIError, "502.*Bad gateway"):
            core.fetch_featured_odds("x", "k")

    @patch("auditor_core.requests.get", side_effect=requests.ConnectTimeout("slow"))
    def test_network_failure(self, _):
        with self.assertRaisesRegex(core.OddsAPIError, "Could not reach"):
            core.fetch_featured_odds("x", "k")

    @patch("auditor_core.requests.get")
    def test_event_market_url_and_parse(self, get):
        get.return_value = resp(payload={"bookmakers": TEAM_TOTALS_BOOKS})
        books, _ = core.fetch_event_market("basketball_nba", "evt1", "KEY", "team_totals")
        self.assertEqual(books[0]["key"], "draftkings")
        self.assertTrue(get.call_args[0][0].endswith("/basketball_nba/events/evt1/odds"))

    @patch("auditor_core.requests.get")
    def test_unexpected_shape(self, get):
        get.return_value = resp(payload={"oops": 1})
        with self.assertRaises(core.OddsAPIError):
            core.fetch_featured_odds("x", "k")


# --- UI smoke test with a fake `streamlit` module ----------------------------
class FakeSt(types.ModuleType):
    """Minimal recording stand-in for streamlit (real Streamlit isn't installed here)."""

    def __init__(self, picks=None, key="KEY", sharp_key="SHARP", checks=None):
        super().__init__("streamlit")
        self.calls: list[tuple[str, tuple, dict]] = []
        self.picks = picks or {}
        self.keys = {"The Odds API Key": key, "SharpAPI Key": sharp_key}
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
        return self.keys[label] or value

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


def run_app(picks=None, key="KEY", get_side_effect=None, sharp_key="SHARP", checks=None):
    fake = FakeSt(picks, key, sharp_key, checks)
    sys.modules["streamlit"] = fake
    sys.modules.pop("app", None)
    import app  # noqa: WPS433

    with patch("auditor_core.requests.get", side_effect=get_side_effect):
        app.main()
    return fake


def router(url, params=None, timeout=None, headers=None):
    if "sharpapi.io" in url:
        return resp(payload={"data": SPLITS_ROWS, "pagination": {"has_more": False, "next_offset": None}})
    if url.endswith("/odds/") or url.endswith("/odds"):
        if "/events/" in url:
            return resp(payload={"bookmakers": TEAM_TOTALS_BOOKS})
        return resp(payload=[GAME], headers={"x-requests-remaining": "99"})
    return resp(status=404, payload={"message": "nope"})


class TestUiSmoke(unittest.TestCase):
    def names(self, fake):
        return [c[0] for c in fake.calls]

    def test_happy_path_spreads(self):
        fake = run_app(get_side_effect=router)
        self.assertIn("table", self.names(fake))
        self.assertNotIn("error", self.names(fake))
        metrics = [c for c in fake.calls if c[0] == "metric"]
        self.assertEqual(len(metrics), 3)
        self.assertIn("WHALE", metrics[0][2]["value"])
        self.assertEqual(metrics[0][2]["delta"], "4/6 Criteria Met")
        self.assertEqual(metrics[2][2]["value"], "-110")

    def test_totals_market(self):
        fake = run_app(picks={"Select Market Type": "totals"}, get_side_effect=router)
        metrics = [c for c in fake.calls if c[0] == "metric"]
        self.assertTrue(metrics[0][2]["delta"].endswith("/5 Criteria Met"))

    def test_team_totals_market(self):
        fake = run_app(
            picks={"Select Market Type": "team_totals (extra API credit)"},
            get_side_effect=router,
        )
        self.assertIn("table", self.names(fake))
        labels = [c for c in fake.calls if c[0] == "selectbox" and c[1][0] == "Select Specific Selection"]
        self.assertTrue(labels[0][1][1][0].startswith("Boston Celtics Over 112.5"))

    def test_splits_come_from_sharpapi(self):
        fake = run_app(get_side_effect=router)
        metrics = [c for c in fake.calls if c[0] == "metric"]
        self.assertEqual(metrics[1][2]["value"], "62% Handle")
        self.assertIn("30% bets", metrics[1][2]["label"])
        self.assertEqual(metrics[1][2]["delta"], "+32.0% vs Public Bets")
        captions = [c[1][0] for c in fake.calls if c[0] == "caption"]
        self.assertTrue(any("draftkings" in t and "SharpAPI" in t for t in captions))
        self.assertNotIn("warning", self.names(fake))

    def test_splits_follow_selection(self):
        fake = run_app(
            picks={"Select Specific Selection": "New York Knicks 3.5 (-110)"},
            get_side_effect=router,
        )
        metrics = [c for c in fake.calls if c[0] == "metric"]
        self.assertEqual(metrics[1][2]["value"], "38% Handle")

    def test_no_sharp_key_falls_back_to_manual(self):
        fake = run_app(sharp_key="", get_side_effect=lambda url, **k: router(url, **k))
        # text_input falls back to the widget default ("" here) -> manual sliders (30/85)
        metrics = [c for c in fake.calls if c[0] == "metric"]
        self.assertEqual(metrics[1][2]["value"], "85% Handle")
        self.assertIn("warning", self.names(fake))

    def test_sharp_plan_error_falls_back_but_app_still_works(self):
        def route(url, **k):
            if "sharpapi.io" in url:
                return resp(status=403, payload={"error": {"message": "Requires Pro tier"}})
            return router(url, **k)
        fake = run_app(get_side_effect=route)
        warns = [c[1][0] for c in fake.calls if c[0] == "warning"]
        self.assertTrue(any("Pro tier" in w for w in warns))
        metrics = [c for c in fake.calls if c[0] == "metric"]
        self.assertEqual(metrics[1][2]["value"], "85% Handle")
        self.assertIn("table", self.names(fake))

    def test_force_manual_skips_sharpapi(self):
        calls = []
        def route(url, **k):
            calls.append(url)
            return router(url, **k)
        run_app(get_side_effect=route, checks={"Always use manual splits": True})
        self.assertFalse(any("sharpapi.io" in u for u in calls))

    def test_team_totals_uses_manual_splits(self):
        fake = run_app(
            picks={"Select Market Type": "team_totals (extra API credit)"},
            get_side_effect=router,
        )
        metrics = [c for c in fake.calls if c[0] == "metric"]
        self.assertEqual(metrics[1][2]["value"], "85% Handle")
        self.assertIn("warning", self.names(fake))

    def test_missing_key_warns(self):
        fake = run_app(key="")
        self.assertIn("warning", self.names(fake))

    def test_api_error_is_shown(self):
        fake = run_app(get_side_effect=lambda *a, **k: resp(status=401, payload={"message": "bad key"}))
        err = [c for c in fake.calls if c[0] == "error"]
        self.assertIn("bad key", err[0][1][0])

    def test_no_games(self):
        fake = run_app(get_side_effect=lambda *a, **k: resp(payload=[]))
        self.assertIn("info", self.names(fake))


if __name__ == "__main__":
    unittest.main()
