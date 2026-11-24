"""
Free background collector for the Auditor (run by GitHub Actions every 30 minutes; costs 0 Odds API credits).

Each run, per league:
  1. Read the DraftKings Network splits page (handle %, bets %, line and odds for every side).
  2. Turn it into games shaped like The Odds API payload, with full team names, and save the line and split readings.
  3. Score every side with the same five-check board scan the app uses, log LEAN / SHARP / FILTERED picks,
     and keep each pending pick's latest line (the closing line once the game starts).
  4. Grade finished picks from ESPN's free scoreboard.
It writes only its own files (collector_*.csv / collector_status.json in the Gist), so it can never overwrite the app's
data. The app reads them and merges them in: line and split history feed checks 3-5, and the signal log shows both.

Run locally:  GITHUB_GIST_TOKEN=... python collector.py --leagues NBA,NFL
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

import audit_log
import cloud_store
import line_history as lh
import persist
import split_history as sh
from auditor_core import DISPLAY_TZ, SPORT_CONFIG, team_match_score
from board_scan import scan_board

HISTORY_NAME = "collector_line_history.csv"
SPLITS_NAME = "collector_split_history.csv"
SIGNALS_NAME = "collector_signals.csv"
STATUS_NAME = "collector_status.json"
DEFAULT_LEAGUES = ("NBA", "NFL")
FAR_HOURS = 24.0                 # games further out than this ...
FAR_GAP_MIN = 180.0              # ... are saved at most every 3 hours (they barely move; keeps the Gist small)
KEEP_HOURS_AFTER_START = 24.0    # line / split readings are dropped a day after the game starts (picks keep their closing line)
STALE_MINUTES = 90.0             # the app warns when the last run is older than this
GRADE_AFTER_HOURS = 3.0          # look for a final score this long after the start
MATCH_HOURS = 12.0               # a score / game counts as the same game when starts are this close
ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard"
ESPN_PATH: Dict[str, str] = {"NBA": "basketball/nba", "NFL": "football/nfl", "NHL": "hockey/nhl", "MLB": "baseball/mlb",
                             "NCAA Football": "football/college-football", "NCAA Basketball": "basketball/mens-college-basketball"}

# Full names as The Odds API spells them, so the collector's picks line up with the app's.
TEAMS: Dict[str, Tuple[str, ...]] = {
    "NBA": ("Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls", "Cleveland Cavaliers",
            "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
            "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat", "Milwaukee Bucks", "Minnesota Timberwolves",
            "New Orleans Pelicans", "New York Knicks", "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
            "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors", "Utah Jazz", "Washington Wizards"),
    "NFL": ("Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills", "Carolina Panthers", "Chicago Bears",
            "Cincinnati Bengals", "Cleveland Browns", "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
            "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Kansas City Chiefs", "Las Vegas Raiders",
            "Los Angeles Chargers", "Los Angeles Rams", "Miami Dolphins", "Minnesota Vikings", "New England Patriots",
            "New Orleans Saints", "New York Giants", "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers",
            "San Francisco 49ers", "Seattle Seahawks", "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Commanders"),
}


# --- DraftKings page -> Odds API-shaped games ---------------------------------------------------------------------------
def canonical(name: str, league: str) -> str:
    """'Bears' -> 'Chicago Bears' when exactly one known team matches strongly; otherwise the name unchanged."""
    teams = TEAMS.get(league, ())
    hits = [t for t in teams if team_match_score(name, t) == 2]
    return hits[0] if len(hits) == 1 else str(name)


def games_from_dk(rows: Sequence[Mapping[str, Any]], league: str) -> List[Dict[str, Any]]:
    """
    One game per splits row: {'id', 'away_team', 'home_team', 'commence_time', 'bookmakers': [DraftKings with h2h, spreads,
    totals]}. Sides without odds on the page are left out (no price = nothing to log or grade)."""
    out: List[Dict[str, Any]] = []
    for r in rows:
        start = r.get("event_start_time")
        if not start:
            continue
        away, home = canonical(r.get("away_team", ""), league), canonical(r.get("home_team", ""), league)
        markets: List[Dict[str, Any]] = []
        ml, sp, tt = r.get("moneyline") or {}, r.get("spread") or {}, r.get("total") or {}
        h2h = [{"name": nm, "price": ml[f"{s}_odds"]} for s, nm in (("away", away), ("home", home)) if ml.get(f"{s}_odds") is not None]
        spreads = [{"name": nm, "price": sp[f"{s}_odds"], "point": sp[f"{s}_line"]} for s, nm in (("away", away), ("home", home))
                   if sp.get(f"{s}_odds") is not None and sp.get(f"{s}_line") is not None]
        totals = [{"name": nm, "price": tt[f"{s}_odds"], "point": tt["line"]} for s, nm in (("over", "Over"), ("under", "Under"))
                  if tt.get(f"{s}_odds") is not None and tt.get("line") is not None]
        for key, outs in (("h2h", h2h), ("spreads", spreads), ("totals", totals)):
            if len(outs) == 2:
                markets.append({"key": key, "outcomes": outs})
        if markets:
            out.append({"id": str(r.get("event_id")), "sport_title": league, "away_team": away, "home_team": home,
                        "commence_time": pd.Timestamp(start).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "bookmakers": [{"key": "draftkings", "title": "DraftKings", "markets": markets}]})
    return out


def thin(new: pd.DataFrame, history: pd.DataFrame, key: Sequence[str], now: pd.Timestamp) -> pd.DataFrame:
    """Drop new readings of games more than FAR_HOURS away whose last saved reading is under FAR_GAP_MIN old."""
    if new is None or new.empty or history is None or history.empty:
        return new
    far = (pd.to_datetime(new["COMMENCE"], utc=True, errors="coerce") - now).dt.total_seconds() / 3600.0 > FAR_HOURS
    last = history.assign(_t=pd.to_datetime(history["TS"], utc=True, errors="coerce")).groupby(list(key))["_t"].max().rename("_last")
    m = new.merge(last, left_on=list(key), right_index=True, how="left")
    recent = ((now - m["_last"]).dt.total_seconds() / 60.0 < FAR_GAP_MIN).fillna(False).to_numpy()
    return new[~(far.to_numpy() & recent)].reset_index(drop=True)


def drop_finished(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Readings of games that started more than KEEP_HOURS_AFTER_START ago are no longer needed."""
    if df is None or df.empty:
        return df
    start = pd.to_datetime(df["COMMENCE"], utc=True, errors="coerce")
    return df[~(start < now - pd.Timedelta(hours=KEEP_HOURS_AFTER_START))].reset_index(drop=True)


# --- ESPN grading (free) --------------------------------------------------------------------------------------------------
def espn_events(league: str, dates: Sequence[str], get: Any = None) -> List[Dict[str, Any]]:
    """Finished games from ESPN's scoreboard for YYYYMMDD dates: [{'start', 'home', 'away', 'home_score', 'away_score'}]."""
    path, get = ESPN_PATH.get(league), get or requests.get
    if path is None:
        return []
    out: List[Dict[str, Any]] = []
    for d in dates:
        resp = get(ESPN_URL.format(path=path), params={"dates": d, "limit": 300}, timeout=15)
        resp.raise_for_status()
        for ev in resp.json().get("events", []):
            if not (((ev.get("status") or {}).get("type") or {}).get("completed")):
                continue
            comp = (ev.get("competitions") or [{}])[0]
            sides = {c.get("homeAway"): c for c in comp.get("competitors", [])}
            if {"home", "away"} - sides.keys():
                continue
            try:
                out.append({"start": ev.get("date"), "home": sides["home"]["team"]["displayName"], "away": sides["away"]["team"]["displayName"],
                            "home_score": float(sides["home"]["score"]), "away_score": float(sides["away"]["score"])})
            except (KeyError, TypeError, ValueError):
                continue
    return out


def dates_to_check(log: pd.DataFrame, now: pd.Timestamp, max_days: int = 4) -> Dict[str, List[str]]:
    """Per league, the ET dates (YYYYMMDD) of pending picks whose games should be over."""
    if log is None or log.empty:
        return {}
    start = pd.to_datetime(log["COMMENCE"], utc=True, errors="coerce")
    due = log["STATUS"].eq("Pending") & (start <= now - pd.Timedelta(hours=GRADE_AFTER_HOURS)) & (start >= now - pd.Timedelta(days=max_days))
    d = log[due].assign(_d=start[due].dt.tz_convert(DISPLAY_TZ).dt.strftime("%Y%m%d"))
    return {lg: sorted(g["_d"].unique()) for lg, g in d.groupby("LEAGUE")}


def grade_espn(signals: pd.DataFrame, events: Mapping[str, Sequence[Mapping[str, Any]]], now: pd.Timestamp) -> Tuple[pd.DataFrame, int]:
    """Grade pending picks against ESPN finals (teams matched by name, starts within MATCH_HOURS). (log, number graded)."""
    log = audit_log.finalize_clv(signals, now)
    n = 0
    for i in log.index[log["STATUS"].eq("Pending")]:
        away, _, home = str(log.at[i, "GAME"]).partition(" @ ")
        start = pd.to_datetime(log.at[i, "COMMENCE"], utc=True, errors="coerce")
        if pd.isna(start):
            continue
        best: Optional[Mapping[str, Any]] = None
        for ev in events.get(str(log.at[i, "LEAGUE"]), ()):
            close = abs((pd.to_datetime(ev["start"], utc=True) - start).total_seconds()) <= MATCH_HOURS * 3600
            if close and team_match_score(away, ev["away"]) and team_match_score(home, ev["home"]):
                best = ev
                break
        if best is None:
            continue
        event = {"home_team": home, "away_team": away,
                 "scores": [{"name": home, "score": best["home_score"]}, {"name": away, "score": best["away_score"]}]}
        res = audit_log.result_for(log.at[i, "MARKET"], log.at[i, "SELECTION"], log.at[i, "SIG_LINE"], event)
        if res:
            log.at[i, "RESULT"], log.at[i, "STATUS"] = res, "Graded"
            n += 1
    return log, n


def grade_free(signals: pd.DataFrame, now: pd.Timestamp, get: Any = None) -> Tuple[pd.DataFrame, int, str]:
    """Fetch the ESPN finals the pending picks need and grade them. (log, graded, error text or '')."""
    events: Dict[str, List[Dict[str, Any]]] = {}
    errors: List[str] = []
    for league, dates in dates_to_check(audit_log.normalize(signals), now).items():
        try:
            events[league] = espn_events(league, dates, get)
        except Exception as e:                      # one league failing must not stop the rest
            errors.append(f"{league}: {str(e)[:80]}")
    log, n = grade_espn(audit_log.normalize(signals), events, now)
    return log, n, "; ".join(errors)


# --- What the app uses -----------------------------------------------------------------------------------------------------
def remap_to_games(df: pd.DataFrame, games: Sequence[Mapping[str, Any]], team_selections: bool = True) -> pd.DataFrame:
    """
    Give collector readings (EVENT_ID 'dk_...') the app's Odds API event ids, so line movement and persistence see one
    history per game. Matched by team names and a start within MATCH_HOURS; selections renamed to the app's team names.
    Unmatched rows are left as they are."""
    if df is None or df.empty or not games:
        return df
    out = df.copy()
    keys = out.loc[out["EVENT_ID"].astype(str).str.startswith("dk_"), ["EVENT_ID", "COMMENCE"]].drop_duplicates("EVENT_ID")
    game_of: Dict[str, Mapping[str, Any]] = {}
    for eid, commence in keys.itertuples(index=False):
        start = pd.to_datetime(commence, utc=True, errors="coerce")
        label = out.loc[out["EVENT_ID"] == eid, "GAME"].iloc[0] if "GAME" in out.columns else ""
        away, _, home = str(label).partition(" @ ")
        for g in games:
            near = not pd.isna(start) and abs((pd.to_datetime(g["commence_time"], utc=True) - start).total_seconds()) <= MATCH_HOURS * 3600
            if near and away and team_match_score(away, g["away_team"]) and team_match_score(home, g["home_team"]):
                game_of[str(eid)] = g
                break
    if not game_of:
        return out
    hit = out["EVENT_ID"].astype(str).isin(game_of)
    g_rows = [game_of[str(e)] for e in out.loc[hit, "EVENT_ID"]]
    if team_selections:
        sel = []
        for (mk, s), g in zip(out.loc[hit, ["MARKET", "SELECTION"]].itertuples(index=False), g_rows):
            if mk in ("h2h", "spreads"):
                s = g["away_team"] if team_match_score(s, g["away_team"]) == 2 else g["home_team"] if team_match_score(s, g["home_team"]) == 2 else s
            sel.append(s)
        out.loc[hit, "SELECTION"] = sel
    out.loc[hit, "EVENT_ID"] = [g["id"] for g in g_rows]
    if "GAME" in out.columns:
        out.loc[hit, "GAME"] = [f"{g['away_team']} @ {g['home_team']}" for g in g_rows]
    return out


def add_split_game(splits: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Split readings carry no GAME column: borrow it from the line history so `remap_to_games` can match them."""
    if splits is None or splits.empty:
        return splits
    names = history.drop_duplicates("EVENT_ID").set_index("EVENT_ID")["GAME"] if history is not None and not history.empty else pd.Series(dtype=object)
    return splits.assign(GAME=splits["EVENT_ID"].map(names))


def load_for_app() -> Dict[str, Any]:
    """The collector's files as the app wants them: {'history', 'splits', 'signals', 'status'} (empty frames when absent)."""
    hist = persist.load(HISTORY_NAME, lh.COLUMNS)
    splits = add_split_game(persist.load(SPLITS_NAME, sh.COLUMNS), hist)
    return {"history": hist, "splits": splits, "signals": audit_log.normalize(persist.load(SIGNALS_NAME, audit_log.COLUMNS)),
            "status": load_status()}


def load_status() -> Dict[str, Any]:
    """Last run report ({} when the collector has never run)."""
    try:
        text = cloud_store.load_text(STATUS_NAME, persist.local_path(STATUS_NAME))
        return json.loads(text) if text and text.strip() else {}
    except Exception:
        return {}


def status_line(status: Mapping[str, Any], now: pd.Timestamp) -> Tuple[str, str]:
    """(level 'ok' | 'warn' | 'none', plain-English line) about the collector for the app's sidebar."""
    if not status:
        return "none", "Background collector: not running yet (see the setup steps). Checks 3-5 only see what this app has watched."
    ts = pd.to_datetime(status.get("last_run"), utc=True, errors="coerce")
    age = (now - ts).total_seconds() / 60.0 if not pd.isna(ts) else float("inf")
    parts = [f"{lg} {v.get('games', 0)} games" if v.get("ok") else f"{lg} failed ({v.get('error', '')[:60]})"
             for lg, v in (status.get("leagues") or {}).items()]
    when = "never" if not np.isfinite(age) else f"{age:.0f} min ago" if age < 120 else f"{age / 60:.1f} h ago"
    bad = age > STALE_MINUTES or any(not v.get("ok") for v in (status.get("leagues") or {}).values())
    text = f"Background collector: last run {when}" + (": " + ", ".join(parts) if parts else "") + "."
    if age > STALE_MINUTES:
        text += " It has stopped: check the Actions tab of your GitHub repo."
    return ("warn" if bad else "ok"), text


# --- One run ---------------------------------------------------------------------------------------------------------------
def collect_league(league: str, now: pd.Timestamp, history: pd.DataFrame, splits: pd.DataFrame, signals: pd.DataFrame,
                   fetch: Any) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Read one league's page and update the three frames. fetch(label, eg) -> object with `.rows`."""
    cfg = SPORT_CONFIG[league]
    rows = list(fetch(cfg.dk_label, cfg.dk_event_group).rows)
    games = [g for g in games_from_dk(rows, league) if pd.to_datetime(g["commence_time"], utc=True) > now]
    ts = now.isoformat()
    new_lines = thin(lh.rows_from_games(games, league, ts), history, lh.KEY, now)
    history, _ = lh.append_new(history, new_lines)
    best, all_sides = scan_board(games, league, rows, history, now, splits)
    new_splits = thin(sh.rows_from_sides(all_sides, ts), splits, sh.KEY, now)
    splits, _ = sh.append_new(splits, new_splits)
    signals, added = audit_log.record(signals, best, league, ts, extra=None, logged_by="collector")
    signals = audit_log.update_closing(signals, all_sides, now)
    tiers = best["TIER"].value_counts().to_dict() if not best.empty else {}
    return history, splits, signals, {"ok": True, "games": len(games), "added": added, "tiers": {str(k): int(v) for k, v in tiers.items()}}


def run(leagues: Sequence[str], now: Optional[pd.Timestamp] = None, fetch: Any = None, get: Any = None) -> Dict[str, Any]:
    """One collector run over `leagues`; saves everything and returns the status report."""
    if fetch is None:
        from dk_splits import fetch_dk_splits
        fetch = fetch_dk_splits
    now = now or pd.Timestamp.now(tz="UTC")
    history = persist.load(HISTORY_NAME, lh.COLUMNS)
    splits = persist.load(SPLITS_NAME, sh.COLUMNS)
    signals = audit_log.normalize(persist.load(SIGNALS_NAME, audit_log.COLUMNS))
    report: Dict[str, Any] = {"last_run": now.isoformat(), "leagues": {}}
    for league in leagues:
        try:
            history, splits, signals, info = collect_league(league, now, history, splits, signals, fetch)
        except Exception as e:
            info = {"ok": False, "games": 0, "error": str(e)[:200]}
        report["leagues"][league] = info
    signals = audit_log.finalize_clv(signals, now)
    signals, graded, err = grade_free(signals, now, get)
    report.update({"graded": graded, "grade_error": err, "pending": int(signals["STATUS"].eq("Pending").sum()),
                   "logged": int(len(signals))})
    persist.save(HISTORY_NAME, drop_finished(history, now), ["TS"] + lh.KEY, lh.COLUMNS, force=True)
    persist.save(SPLITS_NAME, drop_finished(splits, now), ["TS"] + sh.KEY, sh.COLUMNS, force=True)
    persist.save(SIGNALS_NAME, signals, ["SIGNAL_ID"], audit_log.COLUMNS, force=True)
    if cloud_store.save_text(STATUS_NAME, json.dumps(report), persist.local_path(STATUS_NAME), True) is not None:
        cloud_store.save_text(STATUS_NAME, json.dumps(report), persist.local_path(STATUS_NAME), True)   # first save only reads; save again
    cloud_store.flush()
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry: exit 1 only when every league failed, so GitHub emails you about a real outage."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--leagues", default=os.environ.get("COLLECT_LEAGUES", ",".join(DEFAULT_LEAGUES)))
    args = ap.parse_args(argv)
    leagues = [x.strip() for x in args.leagues.split(",") if x.strip() in SPORT_CONFIG]
    cloud_store.configure(os.environ.get("GITHUB_GIST_TOKEN", ""))
    if not cloud_store.configured():
        print("GITHUB_GIST_TOKEN is not set: add it under Settings -> Secrets and variables -> Actions.")
        return 1
    report = run(leagues)
    print(json.dumps(report, indent=1))
    if cloud_store.status().get("level") != "ok":
        print("Storage problem:", cloud_store.status().get("text"))
        return 1
    return 0 if any(v.get("ok") for v in report["leagues"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
