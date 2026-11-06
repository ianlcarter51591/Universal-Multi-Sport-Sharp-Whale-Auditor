"""Tests for the auditor. Run: python -m unittest -v test_auditor  (or pytest)."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import requests

import auditor_core as core
import dk_splits as dk
from datetime import datetime

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


def dk_game_html(away, home, when, ml, spread, total, line, a_line=None):
    """One game block shaped like the live page. Rows are (name, odds, handle, bets)."""
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
NFL_GAME_2 = dk_game_html(
    "JAX Jaguars", "BAL Ravens", "Thu 11/5 · 8:15 PM ET",
    ml=[("BAL Ravens", "+320", 24, 9), ("JAX Jaguars", "−410", 76, 91)],
    spread=[("JAX Jaguars -8.5", "−105", 88, 86), ("BAL Ravens +8.5", "−115", 12, 14)],
    total=[("Over 42.5", "−118", 0, 0), ("Under 42.5", "−102", 100, 100)], line=42.5)
NFL_GAME_3 = dk_game_html(
    "NY Jets", "KC Chiefs", "Sun 11/8 · 1:00 PM ET",
    ml=[("KC Chiefs", "−470", 86, 92), ("NY Jets", "+360", 14, 8)],
    spread=[("KC Chiefs -9.5", "−112", 7, 57), ("NY Jets +9.5", "−108", 93, 43)],
    total=[("Over 46.5", "−105", 0, 0), ("Under 46.5", "−115", 100, 100)], line=46.5)

# Same shape, for the NBA GAME used by the UI tests (Celtics -3.5 spread: 30% bets / 62% handle)
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



NOW = datetime(2026, 11, 4, 12, 0)


class TestDkParser(unittest.TestCase):
    def rows(self):
        return dk.parse_dk_html(dk_page(NFL_GAME_1, NFL_GAME_2, NFL_GAME_3), NOW)

    def test_parses_all_games_ignoring_noise_and_scripts(self):
        rows = self.rows()
        self.assertEqual([(r["away_team"], r["home_team"]) for r in rows],
                         [("Bears", "Seahawks"), ("Jaguars", "Ravens"), ("Jets", "Chiefs")])

    def test_values_map_to_sides_and_scale(self):
        r = self.rows()[0]
        # Page column order is Odds, HANDLE, BETS (not bets first).
        self.assertEqual(r["moneyline"]["handle_pct"], {"home": 0.32, "away": 0.68})
        self.assertEqual(r["moneyline"]["bets_pct"], {"home": 0.55, "away": 0.45})
        self.assertEqual(r["spread"]["home_line"], -3.5)
        self.assertEqual(r["spread"]["away_line"], 3.5)
        self.assertEqual(r["spread"]["handle_pct"], {"home": 0.63, "away": 0.37})
        self.assertEqual(r["total"]["line"], 46.5)
        self.assertEqual(r["total"]["handle_pct"], {"over": 0.63, "under": 0.37})
        self.assertEqual(r["fetched_at"], "1:47 PM ET")

    def test_zero_and_hundred_percent_rows_kept(self):
        r = self.rows()[1]
        self.assertEqual(r["total"]["handle_pct"], {"over": 0.0, "under": 1.0})

    def test_start_time_is_utc_with_dst_handled(self):
        # Nov 2, 2026 is after US DST ends (Nov 1) -> ET = UTC-5
        self.assertEqual(self.rows()[0]["event_start_time"], "2026-11-03T01:20:00+00:00")
        self.assertEqual(dk.parse_start("Fri 9/4 · 7:05 PM ET", datetime(2026, 9, 1)), "2026-09-04T23:05:00+00:00")
        self.assertEqual(dk.parse_start("Sat 1/2 · 1:00 PM ET", datetime(2026, 12, 30)), "2027-01-02T18:00:00+00:00")
        self.assertIsNone(dk.parse_start("garbage"))

    def test_strip_abbrev(self):
        self.assertEqual(dk.strip_abbrev("CHI Bears"), "Bears")
        self.assertEqual(dk.strip_abbrev("NY Jets"), "Jets")
        self.assertEqual(dk.strip_abbrev("POR Trail Blazers"), "Trail Blazers")
        self.assertEqual(dk.strip_abbrev("Arsenal"), "Arsenal")
        self.assertEqual(dk.strip_abbrev("Los Angeles Rams"), "Los Angeles Rams")

    def test_header_split_across_nodes_or_single_node(self):
        single = dk_page(NFL_GAME_1.replace(' @ <img src="b.png">', ' @ <img src="b.png">'))
        self.assertEqual(len(dk.parse_dk_html(single, NOW)), 1)
        merged = NFL_GAME_1.replace("CHI Bears @ <img", "CHI Bears <span>@</span> <img")
        self.assertEqual(dk.parse_dk_html(dk_page(merged), NOW)[0]["home_team"], "Seahawks")

    def test_unicode_minus_and_plain_minus(self):
        html = NFL_GAME_1.replace("−175", "-175")
        self.assertEqual(len(dk.parse_dk_html(dk_page(html), NOW)), 1)

    def test_garbage_and_empty_pages(self):
        self.assertEqual(dk.parse_dk_html("<html></html>", NOW), [])
        self.assertEqual(dk.parse_dk_html("<p>Mon 11/2 · 8:20 PM ET</p>", NOW), [])

    def test_discover_event_groups_both_param_orders(self):
        html = ('<a href="?tb_eg=88808&amp;itm_content=NFL&amp;x=1">a</a>'
                '<a href="?itm_content=NCAA%20Football&tb_edate=n7days&tb_eg=87637">b</a>')
        self.assertEqual(dk.discover_event_groups(html), {"nfl": 88808, "ncaa football": 87637})

    def test_total_pages(self):
        self.assertEqual(dk.total_pages(dk_page(NFL_GAME_1, pages=3)), 3)
        self.assertEqual(dk.total_pages("<p>none</p>"), 1)

    def test_feeds_matching_and_reading_with_oddsapi_names(self):
        rows = self.rows()
        ev = core.match_split_rows(rows, "Chicago Bears", "Seattle Seahawks", "2026-11-03T01:20:00Z")
        self.assertEqual(len(ev), 1)
        args = ("Chicago Bears", "Seattle Seahawks")
        sp = core.read_split(ev, "spreads", "Seattle Seahawks", *args)
        self.assertEqual((sp.ticket_pct, sp.handle_pct, sp.source), (53.0, 63.0, "draftkings"))
        ml = core.read_split(ev, "h2h", "Chicago Bears", *args)
        self.assertEqual((ml.ticket_pct, ml.handle_pct), (45.0, 68.0))
        un = core.read_split(ev, "totals", "Under", *args)
        self.assertEqual((un.ticket_pct, un.handle_pct), (43.0, 37.0))
        # abbreviated rows still match multi-word names
        ev2 = core.match_split_rows(rows, "New York Jets", "Kansas City Chiefs", "2026-11-08T18:00:00Z")
        self.assertEqual(ev2[0]["away_team"], "Jets")


COLLEGE_GAME = dk_game_html(
    "Northeastern", "Virginia", "Fri 11/6 · 5:00 PM ET",
    ml=[("Virginia", "−2500", 90, 80), ("Northeastern", "+1100", 10, 20)],
    spread=[("Virginia -15.5", "−110", 55, 60), ("Northeastern +15.5", "−110", 45, 40)],
    total=[("Over 160.5", "−115", 70, 35), ("Under 160.5", "−105", 30, 65)], line=160.5)
COLLEGE_OTHER = dk_game_html(
    "Virginia Tech", "SMU", "Fri 11/6 · 8:00 PM ET",
    ml=[("SMU", "−200", 60, 55), ("Virginia Tech", "+170", 40, 45)],
    spread=[("SMU -4.5", "−110", 50, 50), ("Virginia Tech +4.5", "−110", 50, 50)],
    total=[("Over 150.5", "−110", 50, 50), ("Under 150.5", "−110", 50, 50)], line=150.5)
COLLEGE_ABBREV = dk_game_html(
    "UNC Wilmington", "Michigan St", "Fri 11/6 · 7:00 PM ET",
    ml=[("Michigan St", "−900", 85, 90), ("UNC Wilmington", "+600", 15, 10)],
    spread=[("Michigan St -12.5", "−110", 52, 58), ("UNC Wilmington +12.5", "−110", 48, 42)],
    total=[("Over 140.5", "−110", 40, 45), ("Under 140.5", "−110", 60, 55)], line=140.5)


class TestCollegeNames(unittest.TestCase):
    """DraftKings lists college teams as bare school names (no nickname)."""

    def rows(self):
        return dk.parse_dk_html(dk_page(COLLEGE_GAME, COLLEGE_OTHER, COLLEGE_ABBREV), NOW)

    def test_school_name_matches_school_plus_nickname(self):
        self.assertEqual(core.team_match_score("Virginia", "Virginia Cavaliers"), 2)
        self.assertEqual(core.team_match_score("Northeastern", "Northeastern Huskies"), 2)
        self.assertEqual(core.team_match_score("Arkansas-Pine Bluff", "Arkansas-Pine Bluff Golden Lions"), 1)
        self.assertEqual(core.team_match_score("Texas Tech", "Texas Tech Red Raiders"), 1)
        self.assertEqual(core.team_match_score("Michigan St", "Michigan State Spartans"), 2)
        self.assertEqual(core.team_match_score("UConn", "Connecticut Huskies"), 2)
        self.assertEqual(core.team_match_score("St John's", "Saint John's Red Storm"), 1)

    def test_virginia_is_not_confused_with_virginia_tech(self):
        self.assertLess(core.team_match_score("Virginia", "Virginia Tech Hokies"),
                        core.team_match_score("Virginia", "Virginia Cavaliers"))
        ev = core.match_split_rows(self.rows(), "Northeastern Huskies", "Virginia Cavaliers", "2026-11-06T22:00:00Z")
        self.assertEqual([(r["away_team"], r["home_team"]) for r in ev], [("Northeastern", "Virginia")])
        self.assertEqual(core.match_split_rows(self.rows(), "Virginia Tech Hokies", "SMU Mustangs", "2026-11-07T01:00:00Z")[0]["away_team"], "Virginia Tech")

    def test_college_readings(self):
        ev = core.match_split_rows(self.rows(), "Northeastern Huskies", "Virginia Cavaliers", "2026-11-06T22:00:00Z")
        args = ("Northeastern Huskies", "Virginia Cavaliers")
        sp = core.read_split(ev, "spreads", "Virginia Cavaliers", *args)
        self.assertEqual((sp.ticket_pct, sp.handle_pct), (60.0, 55.0))
        over = core.read_split(ev, "totals", "Over", *args)
        self.assertEqual((over.ticket_pct, over.handle_pct), (35.0, 70.0))
        ml = core.read_split(ev, "h2h", "Northeastern Huskies", *args)
        self.assertEqual((ml.ticket_pct, ml.handle_pct), (20.0, 10.0))

    def test_abbreviated_college_names_still_match_via_raw_name(self):
        ev = core.match_split_rows(self.rows(), "UNC Wilmington Seahawks", "Michigan State Spartans", "2026-11-07T00:00:00Z")
        self.assertEqual(len(ev), 1)
        sp = core.read_split(ev, "spreads", "Michigan State Spartans", "UNC Wilmington Seahawks", "Michigan State Spartans")
        self.assertEqual((sp.ticket_pct, sp.handle_pct), (58.0, 52.0))


class TestDkFetch(unittest.TestCase):
    @patch("auditor_core.requests.get")
    def test_pagination_and_params(self, get):
        pages = {1: dk_page(NFL_GAME_1, pages=2), 2: dk_page(NFL_GAME_2, NFL_GAME_3, pages=2)}
        def fake(url, params=None, headers=None, timeout=None):
            if not params:
                return resp(text=dk_page(NFL_GAME_1, pages=1))
            return resp(text=pages.get(params["tb_page"], dk_page()))
        get.side_effect = fake
        res = dk.fetch_dk_splits("NFL", 88808, NOW)
        self.assertEqual(len(res.rows), 3)
        calls = [c.kwargs["params"] for c in get.call_args_list]
        self.assertEqual([c.get("tb_page") for c in calls], [None, 1, 2, 3, 1])  # "today" window, then "n7days"
        self.assertEqual([c.get("tb_edate") for c in calls[1:]], ["today"] * 3 + ["n7days"])
        self.assertEqual(res.report.pages, 2)
        self.assertIn("->", res.report.span)
        self.assertEqual(calls[1]["tb_eg"], 88808)
        self.assertEqual(calls[1]["tb_emt"], "0")
        self.assertIn("User-Agent", get.call_args.kwargs["headers"])
        self.assertEqual(res.report.attempts, ((88808, "configured", 3),))

    @patch("auditor_core.requests.get")
    def test_league_that_only_answers_under_one_date_window(self, get):
        for only in ("today", "n7days"):
            def fake(url, params=None, headers=None, timeout=None, only=only):
                if not params:
                    return resp(text="<html></html>")
                return resp(text=dk_page(NFL_GAME_1) if params["tb_edate"] == only else dk_page())
            get.side_effect = fake
            res = dk.fetch_dk_splits("NHL", 42133, NOW)
            self.assertEqual(len(res.rows), 1, only)

    @patch("auditor_core.requests.get")
    def test_discovered_id_is_tried_in_addition(self, get):
        base = '<a href="?tb_eg=12345&itm_content=NBA">x</a>'
        def fake(url, params=None, headers=None, timeout=None):
            if not params:
                return resp(text=base)
            game = NFL_GAME_1 if params["tb_eg"] == 42648 else NFL_GAME_2
            return resp(text=dk_page(game))
        get.side_effect = fake
        res = dk.fetch_dk_splits("NBA", 42648, NOW)
        tried = [c.kwargs["params"]["tb_eg"] for c in get.call_args_list if c.kwargs["params"]]
        self.assertEqual(sorted(set(tried)), [12345, 42648])
        self.assertEqual(len(res.rows), 2)  # rows from both ids are merged
        self.assertEqual([a[:2] for a in res.report.attempts], [(42648, "configured"), (12345, "discovered")])

    @patch("auditor_core.requests.get")
    def test_dropdown_markup_supplies_league_id(self, get):
        base = '<ul><li id="menu-item-123456"><a href="/sport/nhl/">NHL</a></li></ul><select><option value="42999">NHL</option></select>'
        get.side_effect = lambda url, params=None, **k: resp(text=dk_page(NFL_GAME_1) if params else base)
        res = dk.fetch_dk_splits("NHL", 42133, NOW)
        self.assertEqual([a[0] for a in res.report.attempts], [42133, 42999])

    @patch("auditor_core.requests.get")
    def test_repeated_page_stops_loop(self, get):
        get.side_effect = lambda url, params=None, **k: resp(text=dk_page(NFL_GAME_1, pages=9))
        self.assertEqual(len(dk.fetch_dk_splits("NFL", 88808, NOW).rows), 1)
        self.assertLessEqual(get.call_count, 4)

    @patch("auditor_core.requests.get")
    def test_errors_carry_report_and_clue(self, get):
        get.return_value = resp(text='<html><li>NHL</li> js app shell</html>')
        with self.assertRaisesRegex(dk.DKSplitsError, "No games could be read") as ctx:
            dk.fetch_dk_splits("NHL", 42133, NOW)
        self.assertEqual(ctx.exception.report.attempts, ((42133, "configured", 0),))
        self.assertIn("NHL", ctx.exception.report.clue)
        self.assertIn("Tried league id 42133", dk.describe_report(ctx.exception.report))
        get.return_value = resp(status=403)
        with self.assertRaisesRegex(dk.DKSplitsError, "HTTP 403"):
            dk.fetch_dk_splits("NFL", 88808, NOW)
        get.side_effect = requests.Timeout("slow")
        with self.assertRaisesRegex(dk.DKSplitsError, "Could not reach"):
            dk.fetch_dk_splits("NFL", 88808, NOW)
        with self.assertRaisesRegex(dk.DKSplitsError, "does not publish"):
            dk.fetch_dk_splits("", 0, NOW)

    def test_league_coverage_config(self):
        self.assertEqual(core.SPORT_CONFIG["NFL"].dk_event_group, 88808)
        self.assertEqual(core.SPORT_CONFIG["Bundesliga"].dk_event_group, 0)
        self.assertEqual(core.SPORT_CONFIG["La Liga"].dk_label, "")
        for name in ("NBA", "NHL", "MLB", "NCAA Football", "NCAA Basketball", "EPL", "UCL"):
            self.assertTrue(core.SPORT_CONFIG[name].dk_label and core.SPORT_CONFIG[name].dk_event_group, name)


class TestDkOtherSports(unittest.TestCase):
    def test_market_heading_variants(self):
        for text, key in [("Puck Line", "spread"), ("Run Line", "spread"), ("Point Spread", "spread"),
                          ("Asian Handicap", "spread"), ("Match Result", "moneyline"), ("3-Way Moneyline", "moneyline"),
                          ("Moneyline", "moneyline"), ("Total Goals", "total"), ("Total", "total"), ("Selection", None)]:
            self.assertEqual(dk._market_key(text), key, text)

    def test_nhl_puck_line_and_abbreviations(self):
        html = dk_page(dk_game_html(
            "TB Lightning", "NJ Devils", "Thu 11/5 · 7:00 PM ET",
            ml=[("NJ Devils", "−120", 55, 48), ("TB Lightning", "+100", 45, 52)],
            spread=[("NJ Devils -1.5", "+210", 20, 31), ("TB Lightning +1.5", "−260", 80, 69)],
            total=[("Over 6", "−105", 35, 60), ("Under 6", "−115", 65, 40)], line=6))
        rows = dk.parse_dk_html(html.replace("<div>Spread</div>", "<div>Puck Line</div>"), NOW)
        self.assertEqual(len(rows), 1)
        ev = core.match_split_rows(rows, "New Jersey Devils", "Tampa Bay Lightning", "x")
        self.assertEqual(ev, [])  # orientation: Odds API lists TB at NJ here
        ev = core.match_split_rows(rows, "Tampa Bay Lightning", "New Jersey Devils", rows[0]["event_start_time"])
        pl = core.read_split(ev, "spreads", "New Jersey Devils", "Tampa Bay Lightning", "New Jersey Devils")
        self.assertEqual((pl.ticket_pct, pl.handle_pct), (31.0, 20.0))
        self.assertEqual(rows[0]["spread"]["home_line"], -1.5)

    def test_soccer_three_way_draw_is_kept(self):
        html = dk_page(dk_game_html(
            "ARS Arsenal", "CHE Chelsea", "Sat 11/7 · 12:30 PM ET",
            ml=[("ARS Arsenal", "+110", 41, 38), ("Draw", "+240", 12, 18), ("CHE Chelsea", "+260", 47, 44)],
            spread=[], total=[("Over 2.5", "−110", 50, 61), ("Under 2.5", "−110", 50, 39)], line=2.5))
        rows = dk.parse_dk_html(html.replace("<div>Moneyline</div>", "<div>Match Result</div>"), NOW)
        self.assertEqual(rows[0]["moneyline"]["handle_pct"], {"away": 0.41, "draw": 0.12, "home": 0.47})
        ev = [rows[0]]
        d = core.read_split(ev, "h2h", "Draw", "Arsenal", "Chelsea")
        self.assertEqual((d.ticket_pct, d.handle_pct), (18.0, 12.0))
        self.assertNotIn("spread", rows[0])

    def test_markup_discovery_ignores_cms_ids_and_class(self):
        html = ('<li id="menu-item-123456" class="m-99999"><a href="/sport/nba/">NBA</a></li>'
                '<select><option value="42133">NHL</option><option data-league-id="84240">MLB</option></select>'
                '<a href="?x=1&tb_eg=87637&y=2">NCAA Football</a>')
        self.assertEqual(dk.discover_event_groups(html), {"nhl": 42133, "mlb": 84240, "ncaa football": 87637})

    def test_parse_dk_override(self):
        url = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/?itm_content=NCAA+Football&tb_edate=n7days&tb_eg=87637"
        self.assertEqual(dk.parse_dk_override(url, "x"), ("NCAA Football", 87637))
        self.assertEqual(dk.parse_dk_override("?tb_eg=42133&itm_content=NHL", "x"), ("NHL", 42133))
        self.assertEqual(dk.parse_dk_override(" 42133 ", "NHL"), ("NHL", 42133))
        self.assertEqual(dk.parse_dk_override("...tb_eg=555", "NHL"), ("NHL", 555))
        self.assertIsNone(dk.parse_dk_override("https://example.com/?x=1", "NHL"))
        self.assertIsNone(dk.parse_dk_override("", "NHL"))

    def test_markup_clue_prefers_the_sports_dropdown_over_nav(self):
        html = ('<nav><a href="/sport/nba/">NBA</a></nav><p>' + 'filler ' * 120 + '</p>'
                '<ul><li data-eg="42648">NBA</li><li data-eg="42133">NHL</li><li data-eg="84240">MLB</li></ul>')
        self.assertIn("data-eg=\"42648\"", dk.markup_clue(html, "NBA"))
        self.assertNotIn("/sport/nba/", dk.markup_clue(html, "NBA"))

    def test_coverage_report_mixes_ok_and_failures(self):
        ok = dk.DKResult(rows=[{"away_team": "Bruins", "home_team": "Rangers"}],
                         report=dk.DKReport("NHL", ((42133, "configured", 1),), ""))
        def fake(label, eg):
            if label == "NHL":
                return ok
            if not label:
                raise dk.DKSplitsError("DraftKings Network does not publish splits for this league.")
            raise dk.DKSplitsError("No games could be read", dk.DKReport(label, ((eg, "configured", 0),), ""))
        df = dk.coverage_report({"NHL": ("NHL", 42133), "MLB": ("MLB", 84240), "La Liga": ("", 0)}, fetch=fake)
        self.assertEqual(list(df["Games"]), [1, 0, 0])
        self.assertEqual(df.loc[0, "First game"], "Bruins @ Rangers")
        self.assertEqual(df.loc[1, "League id(s) tried"], "84240")
        self.assertEqual(df.loc[0, "Status"], "OK")
        self.assertIn("does not publish", df.loc[2, "Status"])

    def test_markup_clue_and_describe(self):
        html = '<ul><li data-x="1">NHL</li></ul>'
        self.assertIn("NHL", dk.markup_clue(html, "NHL"))
        self.assertIn("not found", dk.markup_clue(html, "MLB"))
        self.assertIn("No fetch report", dk.describe_report(None))


class TestSharpApiRemoved(unittest.TestCase):
    def test_no_sharpapi_anywhere_in_app_code(self):
        import pathlib
        for name in ("app.py", "auditor_core.py", "dk_splits.py", "secrets.toml.example"):
            text = pathlib.Path(name).read_text().lower()
            self.assertNotIn("sharpapi", text, name)
            self.assertNotIn("sharp_api_key", text, name)


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
        self.keys = {"The Odds API Key": key, "The Odds API Key (unused)": sharp_key,
                     "DraftKings page URL or tb_eg id (optional)": ""}
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


def run_app(picks=None, key="KEY", get_side_effect=None, sharp_key="SHARP", checks=None, dk_override=""):
    fake = FakeSt(picks, key, sharp_key, checks)
    fake.keys["DraftKings page URL or tb_eg id (optional)"] = dk_override
    sys.modules["streamlit"] = fake
    sys.modules.pop("app", None)
    import app  # noqa: WPS433

    with patch("auditor_core.requests.get", side_effect=get_side_effect):
        app.main()
    return fake


def router(url, params=None, timeout=None, headers=None):
    if "dknetwork.draftkings.com" in url:
        return resp(text=DK_UI_HTML)
    if url.endswith("/odds/") or url.endswith("/odds"):
        if "/events/" in url:
            return resp(payload={"bookmakers": TEAM_TOTALS_BOOKS})
        return resp(payload=[GAME], headers={"x-requests-remaining": "99"})
    return resp(status=404, payload={"message": "nope"})


def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


class TestTheme(unittest.TestCase):
    def test_readable_color_pairs_meet_wcag_aa(self):
        sys.modules["streamlit"] = FakeSt()
        sys.modules.pop("app", None)
        import app
        pairs = [
            (app.EMERALD, app.WHITE), (app.EMERALD, app.MINT),
            (app.EMERALD_DEEP, app.MINT), (app.EMERALD_LIGHT, app.ONYX),
            (app.WHITE, app.ONYX),
        ]
        for fg, bg in pairs:
            self.assertGreaterEqual(_contrast(fg, bg), 4.5, (fg, bg))

    def test_css_uses_palette_and_no_leftover_dark_defaults(self):
        sys.modules["streamlit"] = FakeSt()
        sys.modules.pop("app", None)
        import app
        for color in (app.EMERALD, app.ONYX, app.WHITE, app.EMERALD_LIGHT):
            self.assertIn(color, app.CSS)
        self.assertNotIn("{{", app.CSS)  # f-string braces fully resolved


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

    def test_splits_follow_selection(self):
        fake = run_app(
            picks={"Select Specific Selection": "New York Knicks 3.5 (-110)"},
            get_side_effect=router,
        )
        metrics = [c for c in fake.calls if c[0] == "metric"]
        self.assertEqual(metrics[1][2]["value"], "38% Handle")

    def test_force_manual_skips_splits_fetch(self):
        calls = []
        def route(url, **k):
            calls.append(url)
            return router(url, **k)
        run_app(get_side_effect=route, checks={"Always use manual splits": True})
        self.assertFalse(any("dknetwork" in u for u in calls))

    def test_team_totals_uses_manual_splits(self):
        fake = run_app(
            picks={"Select Market Type": "team_totals (extra API credit)"},
            get_side_effect=router,
        )
        metrics = [c for c in fake.calls if c[0] == "metric"]
        self.assertEqual(metrics[1][2]["value"], "85% Handle")
        self.assertIn("warning", self.names(fake))

    def test_splits_default_to_draftkings_network(self):
        calls = []
        def route(url, **k):
            calls.append(url)
            return router(url, **k)
        fake = run_app(get_side_effect=route)
        self.assertTrue(any("dknetwork.draftkings.com" in u for u in calls))
        metrics = [c for c in fake.calls if c[0] == "metric"]
        self.assertEqual(metrics[1][2]["value"], "62% Handle")
        captions = [c[1][0] for c in fake.calls if c[0] == "caption"]
        self.assertTrue(any("DraftKings Network" in c and "1:47 PM ET" in c for c in captions))
        self.assertNotIn("warning", self.names(fake))

    def test_dk_total_and_moneyline_selection(self):
        fake = run_app(picks={"Select Market Type": "totals", "Select Specific Selection": "Under 221.5 (-115)"},
                       get_side_effect=router)
        metrics = [c for c in fake.calls if c[0] == "metric"]
        self.assertEqual((metrics[1][2]["value"], metrics[1][2]["label"]), ("60% Handle", "Handle vs. Ticket Gap (45% bets)"))
        fake = run_app(picks={"Select Market Type": "h2h", "Select Specific Selection": "New York Knicks (+145)"},
                       get_side_effect=router)
        metrics = [c for c in fake.calls if c[0] == "metric"]
        self.assertEqual(metrics[1][2]["value"], "40% Handle")

    def test_dk_failure_falls_back_to_manual(self):
        def route(url, **k):
            return resp(status=503) if "dknetwork" in url else router(url, **k)
        fake = run_app(get_side_effect=route)
        warns = [c[1][0] for c in fake.calls if c[0] == "warning"]
        self.assertTrue(any("Splits unavailable" in w and "HTTP 503" in w for w in warns))
        metrics = [c for c in fake.calls if c[0] == "metric"]
        self.assertEqual(metrics[1][2]["value"], "85% Handle")
        self.assertIn("table", self.names(fake))

    def test_unpublished_league_uses_manual(self):
        fake = run_app(picks={"Select League / Sport": "La Liga"}, get_side_effect=router)
        warns = [c[1][0] for c in fake.calls if c[0] == "warning"]
        self.assertTrue(any("does not publish" in w for w in warns))

    def test_unmatched_game_uses_manual(self):
        def route(url, **k):
            if "dknetwork" in url:
                return resp(text=dk_page(NFL_GAME_1))
            return router(url, **k)
        fake = run_app(get_side_effect=route)
        warns = [c[1][0] for c in fake.calls if c[0] == "warning"]
        self.assertTrue(any("No DraftKings Network splits matched" in w for w in warns))

    def test_diagnostics_panel_shown_when_splits_resolved(self):
        fake = run_app(get_side_effect=router)
        codes = [c[1][0] for c in fake.calls if c[0] == "code"]
        self.assertTrue(codes and "Tried league id 42648" in codes[0])

    def test_unmatched_game_message_lists_what_the_page_returned(self):
        def route(url, **k):
            return resp(text=dk_page(NFL_GAME_1)) if "dknetwork" in url else router(url, **k)
        fake = run_app(get_side_effect=route)
        warns = [c[1][0] for c in fake.calls if c[0] == "warning"]
        self.assertTrue(any("Bears @ Seahawks" in w for w in warns))
        self.assertIn("code", [c[0] for c in fake.calls])

    def test_dk_override_id_is_used_for_the_request(self):
        seen = []
        def route(url, params=None, **k):
            if "dknetwork" in url and params:
                seen.append(params)
            return router(url, params=params, **k)
        fake = run_app(get_side_effect=route,
                       dk_override="https://x/?itm_content=NBA&tb_eg=55555")
        self.assertTrue(seen and all(p["tb_eg"] == 55555 for p in seen))
        metrics = [c for c in fake.calls if c[0] == "metric"]
        self.assertEqual(metrics[1][2]["value"], "62% Handle")

    def test_unreadable_dk_override_falls_back_with_message(self):
        fake = run_app(get_side_effect=router, dk_override="not a url")
        warns = [c[1][0] for c in fake.calls if c[0] == "warning"]
        self.assertTrue(any("Could not find a tb_eg id" in w for w in warns))
        metrics = [c for c in fake.calls if c[0] == "metric"]
        self.assertEqual(metrics[1][2]["value"], "85% Handle")

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
