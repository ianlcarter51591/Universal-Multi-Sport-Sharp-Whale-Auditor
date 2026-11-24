"""UI smoke tests for UNIVERSALSWapp.py with a fake `streamlit` module (real Streamlit isn't installed here)."""
import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import persist
from test_auditor import DK_UI_HTML, GAME, FakeSt, resp, TEAM_TOTALS_BOOKS

NOW = pd.Timestamp.now(tz="UTC")


def game(hours=30):
    g = copy.deepcopy(GAME)
    g["commence_time"] = (NOW + pd.Timedelta(hours=hours)).isoformat()
    return g


def router(url, params=None, timeout=None, headers=None, dk=DK_UI_HTML, games=None):
    if "dknetwork.draftkings.com" in url:
        return resp(text=dk) if dk is not None else resp(status=503)
    if "site.api.espn.com" in url:
        return resp(payload={"events": [{"date": (NOW - pd.Timedelta(hours=3)).isoformat(), "status": {"type": {"completed": True}},
                                         "competitions": [{"competitors": [
                                             {"homeAway": "home", "score": "100", "team": {"displayName": "New York Knicks"}},
                                             {"homeAway": "away", "score": "110", "team": {"displayName": "Boston Celtics"}}]}]}]})
    if "/scores" in url:
        return resp(payload=[{"id": "evt1", "completed": True, "home_team": "New York Knicks", "away_team": "Boston Celtics",
                              "scores": [{"name": "New York Knicks", "score": "100"}, {"name": "Boston Celtics", "score": "110"}]}])
    if url.endswith("/odds/") or url.endswith("/odds"):
        if "/events/" in url:
            return resp(payload={"bookmakers": TEAM_TOTALS_BOOKS})
        return resp(payload=games if games is not None else [game()], headers={"x-requests-remaining": "99"})
    return resp(status=404, payload={"message": "nope"})


class UiCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = patch.object(persist, "DATA_DIR", self.tmp.name)
        self.p.start()
        persist.cloud_store.configure("")

    def tearDown(self):
        self.p.stop()
        self.tmp.cleanup()

    def run_app(self, picks=None, key="KEY", get=None, checks=None, override="", fake=None):
        fake = fake or FakeSt(picks, key, "x", checks)
        fake.keys["DraftKings page URL or tb_eg id (optional)"] = override
        sys.modules["streamlit"] = fake
        sys.modules.pop("UNIVERSALSWapp", None)
        import UNIVERSALSWapp as app
        with patch("auditor_core.requests.get", side_effect=get or router):
            app.main()
        return fake

    @staticmethod
    def calls(fake, name):
        return [c for c in fake.calls if c[0] == name]


class TestBoardAndAudit(UiCase):
    def test_happy_path_scores_and_saves_history_and_signals(self):
        fake = self.run_app()
        self.assertFalse(self.calls(fake, "error"))
        tables = [c[1][0] for c in self.calls(fake, "dataframe") if "PICK" in c[1][0].columns]
        board = tables[0]
        self.assertIn("LEAN", set(board["TIER"]))
        self.assertTrue((Path(self.tmp.name) / "auditor_line_history.csv").exists())
        sig = pd.read_csv(Path(self.tmp.name) / "auditor_signals.csv")
        self.assertIn("Boston Celtics", set(sig["SELECTION"]))
        self.assertTrue((sig["STATUS"] == "Pending").all())

    def test_audit_tab_verdict_and_pair_table(self):
        fake = self.run_app()
        metrics = {c[2]["label"].split(" (")[0]: c[2] for c in self.calls(fake, "metric") if "label" in c[2]}
        self.assertIn("LEAN", metrics["Auditor Verdict"]["value"])
        self.assertEqual(metrics["Auditor Verdict"]["delta"], "2/5 checks met")
        self.assertEqual(metrics["Handle vs. Ticket Gap"]["value"], "62% Handle")
        pair = [c[1][0] for c in self.calls(fake, "dataframe") if "WHALE SIDE" in c[1][0].columns]
        self.assertEqual(len(pair[0]), 2)
        tables = self.calls(fake, "table")
        self.assertEqual(len(tables[-1][1][0]), 5)               # the five checks
        self.assertEqual(len(tables[-2][1][0]), 6)               # noise filters
        self.assertIn("Price band", set(tables[-2][1][0]["Filter"]))

    def test_splits_down_means_no_data_never_a_made_up_verdict(self):
        fake = self.run_app(get=lambda url, **k: router(url, dk=None, **k))
        warns = [c[1][0] for c in self.calls(fake, "warning")]
        self.assertTrue(any("Splits unavailable" in w for w in warns))
        self.assertTrue(any("NO DATA" in w for w in warns))
        metrics = [c[2] for c in self.calls(fake, "metric") if c[2].get("label") == "Auditor Verdict"]
        self.assertIn("NO DATA", metrics[0]["value"])
        self.assertFalse((Path(self.tmp.name) / "auditor_signals.csv").exists())      # nothing logged from missing data

    def test_manual_splits_only_when_ticked(self):
        fake = self.run_app(get=lambda url, **k: router(url, dk=None, **k), checks={"Use manual splits": True})
        metrics = [c[2] for c in self.calls(fake, "metric") if c[2].get("label") == "Auditor Verdict"]
        self.assertNotIn("NO DATA", metrics[0]["value"])         # 50/50 manual splits: PASS, not a fake whale

    def test_totals_selection_follows_the_pick(self):
        fake = self.run_app(picks={"Select Market Type": "totals", "Select Specific Selection": "Under 221.5 (-115)"})
        m = [c[2] for c in self.calls(fake, "metric") if str(c[2].get("label", "")).startswith("Handle")][0]
        self.assertEqual((m["value"], m["label"]), ("60% Handle", "Handle vs. Ticket Gap (45% bets)"))

    def test_team_totals_has_no_splits(self):
        fake = self.run_app(picks={"Select Market Type": "team_totals (extra API credit)"})
        metrics = [c[2] for c in self.calls(fake, "metric") if c[2].get("label") == "Auditor Verdict"]
        self.assertIn("NO DATA", metrics[0]["value"])

    def test_unreadable_override_is_reported(self):
        fake = self.run_app(override="not a url")
        caps = [c[1][0] for c in self.calls(fake, "caption")]
        self.assertTrue(any("Could not find a tb_eg id" in c for c in caps))

    def test_override_id_is_used(self):
        seen = []

        def route(url, params=None, **k):
            if "dknetwork" in url and params:
                seen.append(params)
            return router(url, params=params, **k)
        self.run_app(get=route, override="https://x/?itm_content=NBA&tb_eg=55555")
        self.assertTrue(seen and all(p["tb_eg"] == 55555 for p in seen))

    def test_started_games_not_logged(self):
        self.run_app(get=lambda url, **k: router(url, games=[game(hours=-5)], **k))
        self.assertFalse((Path(self.tmp.name) / "auditor_signals.csv").exists())

    def test_missing_key_api_error_no_games(self):
        self.assertTrue(self.calls(self.run_app(key=""), "warning"))
        fake = self.run_app(get=lambda *a, **k: resp(status=401, payload={"message": "bad key"}))
        self.assertIn("bad key", self.calls(fake, "error")[0][1][0])
        fake = self.run_app(get=lambda *a, **k: resp(payload=[]))
        self.assertTrue(self.calls(fake, "info"))

    def test_second_load_reuses_saved_history(self):
        self.run_app()
        first = pd.read_csv(Path(self.tmp.name) / "auditor_line_history.csv")
        self.run_app()
        second = pd.read_csv(Path(self.tmp.name) / "auditor_line_history.csv")
        self.assertEqual(len(first), len(second))                 # unchanged lines are not re-saved within 30 minutes


class TestNoiseFilters(UiCase):
    def test_longshot_and_news_game_are_filtered_not_flagged(self):
        g = game()
        g["bookmakers"][1]["markets"][2]["outcomes"] = [{"name": "Boston Celtics", "price": -170}, {"name": "New York Knicks", "price": 5000}]
        html = DK_UI_HTML.replace("+145", "+5000")
        fake = self.run_app(get=lambda url, **k: router(url, dk=html, games=[g], **k), picks={"Select Market Type": "h2h", "Select Specific Selection": "New York Knicks (+5000)"})
        metrics = [c[2] for c in self.calls(fake, "metric") if c[2].get("label") == "Auditor Verdict"]
        self.assertNotIn("SHARP", metrics[0]["value"])
        filt = [c[1][0] for c in self.calls(fake, "table") if "Filter" in c[1][0].columns][0]
        self.assertEqual(filt.loc[filt["Filter"] == "Price band", "Status"].iloc[0], "⛔ Fail")

    def test_news_marked_game_is_filtered(self):
        fake = FakeSt(None, "KEY", "x", None)
        fake.multiselect = lambda label, options, **k: list(options)          # every game marked as news
        self.run_app(fake=fake)
        sig = pd.read_csv(Path(self.tmp.name) / "auditor_signals.csv")
        self.assertTrue((sig["TIER"] == "FILTERED").all())

    def test_split_history_is_saved(self):
        self.run_app()
        sp = pd.read_csv(Path(self.tmp.name) / "auditor_split_history.csv")
        self.assertIn("TICKETS", sp.columns)
        self.assertGreater(len(sp), 0)


class TestLogTab(UiCase):
    def test_grade_button_grades_and_saves(self):
        self.run_app()
        fake = FakeSt(None, "KEY", "x", None)
        fake.button = lambda label, *a, **k: label.startswith("Grade finished")
        fake.session_state = {}
        # make the logged game look finished: rewrite the saved signal's commence into the past
        sp = Path(self.tmp.name) / "auditor_signals.csv"
        df = pd.read_csv(sp)
        df["COMMENCE"] = (NOW - pd.Timedelta(hours=3)).isoformat()
        df.to_csv(sp, index=False)
        self.run_app(fake=fake)
        out = pd.read_csv(sp)
        self.assertIn("Graded", set(out["STATUS"]))
        self.assertFalse([c for c in self.calls(fake, "warning") if "fell back" in str(c[1][0])])   # graded from ESPN, no Odds API fallback

    def test_collector_picks_and_status_show_up(self):
        import json
        sig = pd.DataFrame([{"SIGNAL_ID": "dk_x|totals|Over", "FIRST_SEEN": NOW.isoformat(), "LEAGUE": "NBA", "EVENT_ID": "dk_x",
                             "GAME": "Boston Celtics @ New York Knicks", "COMMENCE": (NOW - pd.Timedelta(days=1)).isoformat(),
                             "MARKET": "totals", "SELECTION": "Over", "TIER": "LEAN", "SCORE": 3.0, "SIG_LINE": 220.5, "SIG_PRICE": -110,
                             "STATUS": "Graded", "RESULT": "Hit", "FLAGS": "YYY?N", "LOGGED_BY": "collector"}])
        sig.to_csv(Path(self.tmp.name) / "collector_signals.csv", index=False)
        (Path(self.tmp.name) / "collector_status.json").write_text(json.dumps(
            {"last_run": NOW.isoformat(), "leagues": {"NBA": {"ok": True, "games": 7}}}))
        fake = self.run_app()
        caps = " ".join(str(c[1][0]) for c in self.calls(fake, "caption"))
        self.assertIn("Background collector: last run 0 min ago: NBA 7 games.", caps)
        self.assertIn("1 by the background collector", caps)
        checks = [c[1][0] for c in self.calls(fake, "dataframe") if "CHECK" in c[1][0].columns]
        self.assertIn("3. Line moved", set(checks[0]["CHECK"]))

    def test_collector_missing_says_so(self):
        fake = self.run_app()
        self.assertTrue(any("not running yet" in str(c[1][0]) for c in self.calls(fake, "info")))

    def test_log_tab_shows_early_warning(self):
        fake = self.run_app()
        infos = [c[1][0] for c in self.calls(fake, "info")]
        self.assertTrue(any("Too early" in i or "No signals" in i for i in infos))


class TestRowClicks(UiCase):
    def clicking(self, table_key, row):
        fake = FakeSt(None, "KEY", "x", None)

        def dataframe(data, **k):
            fake._rec("dataframe", data, **k)
            hit = k.get("on_select") and str(k.get("key", "")).startswith(table_key)
            return {"selection": {"rows": [row] if hit else []}}
        fake.dataframe = dataframe
        return fake

    def choices(self, fake):
        return {c[1][0]: c[1][1] for c in self.calls(fake, "selectbox_choice")}

    def test_board_row_opens_that_pick_in_the_audit_tab(self):
        fake = self.run_app(fake=self.clicking("aud_board", 0))
        board = [c[1][0] for c in self.calls(fake, "dataframe") if "PICK" in c[1][0].columns][0]
        t = fake.session_state["audit_target"]
        self.assertEqual((t["event_id"], t["market"], t["selection"]), ("evt1", board.iloc[0]["MARKET"], fake.session_state["audit_target"]["selection"]))
        self.assertIn(t["selection"], self.choices(fake)["Select Specific Selection"])
        import UNIVERSALSWapp as app
        self.assertEqual(self.choices(fake)["Select Market Type"], app.market_label(t["market"]))
        self.assertTrue(self.calls(fake, "rerun"))
        self.assertIn("Opened", fake.session_state["clicked_msg_aud_board"])

    def test_signal_log_row_opens_too_and_a_gone_game_warns(self):
        self.run_app()                                                     # logs signals
        fake = self.run_app(fake=self.clicking("aud_log", 0))
        self.assertEqual(fake.session_state["audit_target"]["event_id"], "evt1")
        fake2 = FakeSt(None, "KEY", "x", None)
        fake2.session_state["audit_target"] = {"event_id": "gone", "market": "h2h", "selection": "X", "line": None, "game": "A @ B", "pick": "X ML"}
        self.run_app(fake=fake2)
        self.assertTrue(any("no longer on this league's board" in c[1][0] for c in self.calls(fake2, "warning")))

    def test_selection_matches_line_and_name(self):
        import row_click
        outs = [{"name": "Over", "point": 221.5}, {"name": "Under", "point": 221.5}, {"name": "Over", "point": 223.5}]
        self.assertEqual(row_click.outcome_index(outs, "Over", 223.5), 2)
        self.assertEqual(row_click.outcome_index(outs, "Under", 999), 1)
        self.assertEqual(row_click.outcome_index(outs, "Nobody", None), 0)
        self.assertEqual(row_click.target_from_row({"EVENT_ID": "e", "MARKET": "totals", "SELECTION": "Over", "LINE": "221.5"})["line"], 221.5)


class TestStartupGuard(unittest.TestCase):
    def test_every_required_file_ships(self):
        sys.modules["streamlit"] = FakeSt()
        sys.modules.pop("UNIVERSALSWapp", None)
        import UNIVERSALSWapp as app
        here = Path(app.__file__).resolve().parent
        for f in app.REQUIRED_FILES:
            self.assertTrue((here / f).exists(), f)


if __name__ == "__main__":
    unittest.main()
