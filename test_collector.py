"""Tests for the free background collector: DK page -> games, scanning and logging, ESPN grading, merging into the app."""
from __future__ import annotations

import os
import tempfile
import types
import unittest
from datetime import datetime

import pandas as pd

TMP = tempfile.mkdtemp()
os.environ["AUDITOR_DATA_DIR"] = TMP

import audit_log
import cloud_store
import collector as col
import dk_splits as dk
import persist
from test_auditor import dk_game_html, dk_page

persist.DATA_DIR = TMP
NOW_ET = datetime(2026, 11, 3, 9, 0)                       # naive ET, as the page parser expects
NOW = pd.Timestamp("2026-11-03T14:00:00Z")
GAME_A = dk_game_html("BOS Celtics", "NY Knicks", "Tue 11/3 · 7:30 PM ET",
                      ml=[("BOS Celtics", "−170", 80, 40), ("NY Knicks", "+145", 20, 60)],
                      spread=[("BOS Celtics -3.5", "−110", 78, 35), ("NY Knicks +3.5", "−110", 22, 65)],
                      total=[("Over 221.5", "−105", 40, 55), ("Under 221.5", "−115", 60, 45)], line=221.5)


def page(spread_line="-3.5", odds="−110"):
    html = GAME_A.replace("BOS Celtics -3.5", f"BOS Celtics {spread_line}").replace(
        "NY Knicks +3.5", f"NY Knicks {'+' + spread_line[1:] if spread_line.startswith('-') else spread_line}")
    return dk_page(html)


def fake_fetch(html):
    return lambda label, eg: types.SimpleNamespace(rows=dk.parse_dk_html(html, NOW_ET))


def espn_get(home_score=110, away_score=100, completed=True):
    def get(url, params=None, timeout=None):
        body = {"events": [{"date": "2026-11-04T00:30Z", "status": {"type": {"completed": completed}},
                            "competitions": [{"competitors": [
                                {"homeAway": "home", "score": str(home_score), "team": {"displayName": "New York Knicks"}},
                                {"homeAway": "away", "score": str(away_score), "team": {"displayName": "Boston Celtics"}}]}]}]}
        return types.SimpleNamespace(json=lambda: body, raise_for_status=lambda: None)
    return get


def clear():
    for f in os.listdir(TMP):
        os.remove(os.path.join(TMP, f))
    cloud_store.configure("")


class TestDkGames(unittest.TestCase):
    def test_full_names_odds_and_lines(self):
        g = col.games_from_dk(dk.parse_dk_html(page(), NOW_ET), "NBA")[0]
        self.assertEqual((g["away_team"], g["home_team"]), ("Boston Celtics", "New York Knicks"))
        mk = {m["key"]: m["outcomes"] for m in g["bookmakers"][0]["markets"]}
        self.assertEqual(mk["h2h"][0], {"name": "Boston Celtics", "price": -170})
        self.assertEqual(mk["spreads"][0]["point"], -3.5)
        self.assertEqual([o["name"] for o in mk["totals"]], ["Over", "Under"])
        self.assertEqual(mk["totals"][1]["price"], -115)
        self.assertEqual(col.canonical("Jets", "NFL"), "New York Jets")
        self.assertEqual(col.canonical("Seahawks", "NBA"), "Seahawks")                    # unknown here: left alone


class TestRun(unittest.TestCase):
    def setUp(self):
        clear()

    def test_two_runs_build_history_log_and_grade(self):
        rep = col.run(["NBA"], NOW, fake_fetch(page()), espn_get())
        self.assertTrue(rep["leagues"]["NBA"]["ok"])
        self.assertEqual(rep["leagues"]["NBA"]["games"], 1)
        sig = audit_log.normalize(persist.load(col.SIGNALS_NAME, audit_log.COLUMNS))
        self.assertGreater(len(sig), 0)
        self.assertTrue((sig["LOGGED_BY"] == "collector").all())
        self.assertTrue(sig["FLAGS"].astype(str).str.len().eq(5).all())
        # 2 hours later the spread moved a full point toward the Celtics: checks 3 can now be Yes
        later = NOW + pd.Timedelta(hours=2)
        col.run(["NBA"], later, fake_fetch(page("-4.5")), espn_get())
        hist = persist.load(col.HISTORY_NAME, col.lh.COLUMNS)
        self.assertEqual(hist[hist["MARKET"] == "spreads"]["TS"].nunique(), 2)
        sig = audit_log.normalize(persist.load(col.SIGNALS_NAME, audit_log.COLUMNS))
        celts = sig[(sig["MARKET"] == "spreads") & (sig["SELECTION"] == "Boston Celtics")].iloc[0]
        self.assertEqual(celts["LAST_LINE"], -4.5)                                       # closing line tracked
        # after the game: graded free from ESPN (Knicks won by 10 -> Celtics -3.5 misses)
        rep = col.run(["NBA"], NOW + pd.Timedelta(hours=16), fake_fetch(dk_page()), espn_get())
        sig = audit_log.normalize(persist.load(col.SIGNALS_NAME, audit_log.COLUMNS))
        celts = sig[(sig["MARKET"] == "spreads") & (sig["SELECTION"] == "Boston Celtics")].iloc[0]
        self.assertEqual((celts["STATUS"], celts["RESULT"]), ("Graded", "Miss"))
        self.assertGreater(celts["CLV"], 0)                                              # took -3.5, closed -4.5
        self.assertGreaterEqual(rep["graded"], 1)
        self.assertEqual(col.main(["--leagues", "NBA"]), 1)                              # no token: refuses to run

    def test_failed_league_is_reported_not_raised(self):
        def boom(label, eg):
            raise dk.DKSplitsError("HTTP 403")
        rep = col.run(["NBA", "NFL"], NOW, boom, espn_get())
        self.assertEqual({v["ok"] for v in rep["leagues"].values()}, {False})
        self.assertIn("403", rep["leagues"]["NBA"]["error"])
        lvl, text = col.status_line(rep, NOW + pd.Timedelta(minutes=5))
        self.assertEqual(lvl, "warn")
        self.assertIn("NBA failed", text)


class TestThin(unittest.TestCase):
    def test_far_games_saved_every_3_hours(self):
        hist = pd.DataFrame({"TS": [NOW.isoformat()] * 2, "EVENT_ID": ["a", "b"], "MARKET": ["h2h"] * 2, "SELECTION": ["X"] * 2,
                             "COMMENCE": [(NOW + pd.Timedelta(days=3)).isoformat(), (NOW + pd.Timedelta(hours=5)).isoformat()]})
        new = hist.assign(TS=(NOW + pd.Timedelta(minutes=30)).isoformat())
        kept = col.thin(new, hist, ["EVENT_ID", "MARKET", "SELECTION"], NOW + pd.Timedelta(minutes=30))
        self.assertEqual(list(kept["EVENT_ID"]), ["b"])                                  # far game waits; near game saved


class TestAppSide(unittest.TestCase):
    def test_remap_and_merge(self):
        games = [{"id": "odds1", "away_team": "Boston Celtics", "home_team": "New York Knicks", "commence_time": "2026-11-04T00:30:00Z"}]
        hist = pd.DataFrame({"TS": [NOW.isoformat()], "EVENT_ID": ["dk_celtics_knicks_x"], "LEAGUE": ["NBA"],
                             "GAME": ["Celtics @ Knicks"], "COMMENCE": ["2026-11-04T00:31:00Z"], "MARKET": ["spreads"],
                             "SELECTION": ["Celtics"], "LINE": [-3.5], "PRICE": [-110]})
        out = col.remap_to_games(hist, games)
        self.assertEqual((out["EVENT_ID"][0], out["SELECTION"][0], out["GAME"][0]), ("odds1", "Boston Celtics", "Boston Celtics @ New York Knicks"))
        base = {"LEAGUE": "NBA", "GAME": "Boston Celtics @ New York Knicks", "COMMENCE": "2026-11-04T00:30:00Z", "MARKET": "spreads",
                "SELECTION": "Boston Celtics", "TIER": "LEAN", "SCORE": 2.0}
        app = pd.DataFrame([{**base, "SIGNAL_ID": "odds1|spreads|Boston Celtics", "EVENT_ID": "odds1", "FIRST_SEEN": "2026-11-03T15:00:00Z",
                             "STATUS": "Pending"}])
        coll = pd.DataFrame([{**base, "SIGNAL_ID": "dk_x|spreads|Boston Celtics", "EVENT_ID": "dk_x", "FIRST_SEEN": "2026-11-03T12:00:00Z",
                              "STATUS": "Graded", "RESULT": "Hit"}])
        m = audit_log.merge_logs(app, coll)
        self.assertEqual(len(m), 1)
        self.assertEqual((m["LOGGED_BY"][0], m["RESULT"][0]), ("collector", "Hit"))   # graded copy wins

    def test_status_line(self):
        self.assertEqual(col.status_line({}, NOW)[0], "none")
        ok = {"last_run": (NOW - pd.Timedelta(minutes=12)).isoformat(), "leagues": {"NBA": {"ok": True, "games": 9}}}
        self.assertEqual(col.status_line(ok, NOW), ("ok", "Background collector: last run 12 min ago: NBA 9 games."))
        self.assertIn("stopped", col.status_line(ok, NOW + pd.Timedelta(hours=3))[1])


class TestByCheck(unittest.TestCase):
    def test_yes_vs_no(self):
        rows = [{"SIGNAL_ID": str(i), "LEAGUE": "NBA", "TIER": "LEAN", "SCORE": 2.0, "MARKET": "spreads", "STATUS": "Graded",
                 "FLAGS": "YYN?N" if i % 2 else "YNN?N", "RESULT": "Hit" if i % 2 else "Miss", "SIG_PRICE": -110} for i in range(10)]
        t = audit_log.by_check(pd.DataFrame(rows)).set_index(["CHECK", "WHEN"])
        self.assertEqual(t.loc[("2. Bigger bets", "Yes"), "HIT_%"], 100.0)
        self.assertEqual(t.loc[("2. Bigger bets", "No"), "HIT_%"], 0.0)
        self.assertAlmostEqual(t.loc[("2. Bigger bets", "Yes"), "ROI_%"], 100 / 110 * 100, places=3)
        self.assertNotIn(("4. Held 30 min", "Yes"), t.index)                              # never known: no row
        self.assertEqual(list(audit_log.by_score(pd.DataFrame(rows))["CHECKS_MET"]), ["2/5"])


if __name__ == "__main__":
    unittest.main()
