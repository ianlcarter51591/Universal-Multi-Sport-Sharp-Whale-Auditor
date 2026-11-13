"""
Signal log: does the Auditor's "LEAN / SHARP" actually beat the market?

Every LEAN or SHARP pick found by a board scan is saved once, with the line and price at that moment (SIG_*). Each later
scan before the game refreshes LAST_* (the last line the app saw, used as the closing line). After the game:
  CLV     = -(movement from the signal line to the closing line), in the selection's frame (see line_history):
            positive = you would have beaten the close; spreads/totals in points, h2h in probability points.
  RESULT  = graded at the SIGNAL line from The Odds API scores: Hit / Miss / Push.
Hit rate gets a 95% Wilson interval; below MIN_GRADED picks everything is labelled "too early". At -110 a bettor needs
52.4% to break even.
Limit: the "closing line" is only the last line this app saw, so it is only as good as how often the app was opened.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from line_history import toward_move

SIGNALS_NAME = "auditor_signals.csv"
COLUMNS = ["SIGNAL_ID", "FIRST_SEEN", "LEAGUE", "EVENT_ID", "GAME", "COMMENCE", "MARKET", "SELECTION", "TIER", "SCORE", "TICKETS", "HANDLE",
           "SIG_LINE", "SIG_PRICE", "LAST_TS", "LAST_LINE", "LAST_PRICE", "STATUS", "RESULT", "CLV", "OFF_EV", "OFF_MARKET"]
SIGNAL_TIERS = ("LEAN", "SHARP", "FILTERED")        # FILTERED picks are logged too, so the log can show whether the filters help
REAL_TIERS = ("LEAN", "SHARP")
MIN_GRADED = 50
BREAKEVEN_PCT = 52.38
Z95 = 1.959964


def empty() -> pd.DataFrame:
    """Empty signal log."""
    return pd.DataFrame(columns=COLUMNS)


def normalize(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Signal log with every column present and numbers as numbers."""
    if df is None or df.empty:
        return empty()
    out = df.copy()
    for c in COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[COLUMNS]
    numeric = ("SCORE", "TICKETS", "HANDLE", "SIG_LINE", "SIG_PRICE", "LAST_LINE", "LAST_PRICE", "CLV", "OFF_EV")
    for c in COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce") if c in numeric else out[c].astype(object)   # text columns stay text even when all empty
    return out.drop_duplicates("SIGNAL_ID", keep="last").reset_index(drop=True)


def signal_id(event_id: Any, market: Any, selection: Any) -> str:
    """Stable id of one pick."""
    return f"{event_id}|{market}|{selection}"


def record(signals: pd.DataFrame, best: pd.DataFrame, league: str, now_iso: str, extra: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, int]:
    """Add LEAN / SHARP / FILTERED picks from a board scan that are not logged yet, plus any OFF_MARKET side in `extra` (all scanned sides) so the
    log can compare the two sources. A pending FILTERED pick that later becomes LEAN / SHARP is upgraded (its signal line and price become the ones
    at that moment); a pending off-market-only (PASS) pick that later becomes LEAN / SHARP only changes label and scores. (log, number added)."""
    log = normalize(signals)
    if best is None or best.empty:
        return log, 0
    pick = best[best["TIER"].isin(SIGNAL_TIERS)].copy()
    if extra is not None and not extra.empty and "OFF_MARKET" in extra.columns:
        off = extra[extra["OFF_MARKET"].fillna(False).astype(bool)]
        if not off.empty:
            pick = pd.concat([pick, off], ignore_index=True)
            pick = pick.drop_duplicates(["EVENT_ID", "MARKET", "SELECTION"], keep="first")
    if pick.empty:
        return log, 0
    pick["_id"] = [signal_id(e, m, s) for e, m, s in zip(pick["EVENT_ID"], pick["MARKET"], pick["SELECTION"])]
    known = set(log["SIGNAL_ID"])
    up = pick[pick["_id"].isin(known) & pick["TIER"].isin(REAL_TIERS)]
    if not up.empty:
        for i in log.index[log["SIGNAL_ID"].isin(up["_id"]) & ~log["TIER"].isin(REAL_TIERS + ("FILTERED",)) & log["STATUS"].eq("Pending")]:
            r = up[up["_id"] == log.at[i, "SIGNAL_ID"]].iloc[0]
            log.loc[i, ["TIER", "SCORE", "TICKETS", "HANDLE"]] = [r["TIER"], r["SCORE"], r["TICKETS_%"], r["HANDLE_%"]]
        idx = log.index[log["SIGNAL_ID"].isin(up["_id"]) & log["TIER"].eq("FILTERED") & log["STATUS"].eq("Pending")]
        for i in idx:
            r = up[up["_id"] == log.at[i, "SIGNAL_ID"]].iloc[0]
            log.loc[i, ["TIER", "SCORE", "TICKETS", "HANDLE", "SIG_LINE", "SIG_PRICE", "FIRST_SEEN"]] = [r["TIER"], r["SCORE"], r["TICKETS_%"], r["HANDLE_%"], r["LINE"], r["PRICE"], now_iso]
    fresh = pick[~pick["_id"].isin(known)]
    if fresh.empty:
        return log, 0
    add = pd.DataFrame({
        "SIGNAL_ID": fresh["_id"].to_numpy(),
        "FIRST_SEEN": now_iso, "LEAGUE": league, "EVENT_ID": fresh["EVENT_ID"].to_numpy(), "GAME": fresh["GAME"].to_numpy(),
        "COMMENCE": fresh["COMMENCE"].to_numpy(), "MARKET": fresh["MARKET"].to_numpy(), "SELECTION": fresh["SELECTION"].to_numpy(),
        "TIER": fresh["TIER"].to_numpy(), "SCORE": fresh["SCORE"].to_numpy(), "TICKETS": fresh["TICKETS_%"].to_numpy(),
        "HANDLE": fresh["HANDLE_%"].to_numpy(), "SIG_LINE": fresh["LINE"].to_numpy(), "SIG_PRICE": fresh["PRICE"].to_numpy(),
        "LAST_TS": now_iso, "LAST_LINE": fresh["LINE"].to_numpy(), "LAST_PRICE": fresh["PRICE"].to_numpy(),
        "STATUS": "Pending", "RESULT": pd.NA, "CLV": np.nan,
        "OFF_EV": fresh["VS_MKT_EV_%"].to_numpy() if "VS_MKT_EV_%" in fresh.columns else np.nan,
        "OFF_MARKET": np.where(fresh["OFF_MARKET"].fillna(False).astype(bool), "Y", "") if "OFF_MARKET" in fresh.columns else ""})
    return normalize(pd.concat([log, add], ignore_index=True)), len(add)


def update_closing(signals: pd.DataFrame, all_sides: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Refresh LAST_* of pending signals whose game has not started, from the latest scan."""
    log = normalize(signals)
    if log.empty or all_sides is None or all_sides.empty:
        return log
    cur = all_sides[["EVENT_ID", "MARKET", "SELECTION", "LINE", "PRICE"]].rename(columns={"LINE": "N_LINE", "PRICE": "N_PRICE"})
    m = log.merge(cur, on=["EVENT_ID", "MARKET", "SELECTION"], how="left")
    live = ((pd.to_datetime(log["COMMENCE"], utc=True, errors="coerce") > now) & log["STATUS"].eq("Pending")).to_numpy() & m["N_PRICE"].notna().to_numpy()
    if not live.any():
        return log
    log.loc[live, "LAST_LINE"] = m.loc[live, "N_LINE"].to_numpy()
    log.loc[live, "LAST_PRICE"] = m.loc[live, "N_PRICE"].to_numpy()
    log.loc[live, "LAST_TS"] = now.isoformat()
    return log


def _clv(row: pd.Series) -> float:
    """Closing-line value of one pick (positive = beat the close)."""
    mv = toward_move(row["MARKET"], row["SELECTION"], row["SIG_LINE"], row["SIG_PRICE"], row["LAST_LINE"], row["LAST_PRICE"])
    return float(-mv) if np.isfinite(mv) else float("nan")


def finalize_clv(signals: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Compute CLV for every pick whose game has started."""
    log = normalize(signals)
    if log.empty:
        return log
    started = (pd.to_datetime(log["COMMENCE"], utc=True, errors="coerce") <= now).to_numpy()
    if started.any():
        log.loc[started, "CLV"] = log[started].apply(_clv, axis=1).to_numpy()
    return log


def _points(event: Mapping[str, Any], team: str) -> Optional[float]:
    for s in event.get("scores") or []:
        if s.get("name") == team:
            try:
                return float(s.get("score"))
            except (TypeError, ValueError):
                return None
    return None


def result_for(market: str, selection: str, line: Any, event: Mapping[str, Any]) -> Optional[str]:
    """Hit / Miss / Push for a pick at its signal line, or None when it can't be graded."""
    home, away = event.get("home_team"), event.get("away_team")
    hs, as_ = _points(event, home), _points(event, away)
    if hs is None or as_ is None:
        return None
    sel = str(selection)
    if market == "spreads":
        if sel == home:
            margin = hs - as_
        elif sel == away:
            margin = as_ - hs
        else:
            return None
        edge = margin + float(line)
    elif market in ("totals", "team_totals"):
        low = sel.strip().lower()
        edge = (hs + as_) - float(line) if low.endswith("over") else float(line) - (hs + as_) if low.endswith("under") else None
        if edge is None:
            return None
    elif market == "h2h":
        if sel.lower() == "draw":
            return "Hit" if hs == as_ else "Miss"
        if sel not in (home, away):
            return None
        edge = (hs - as_) if sel == home else (as_ - hs)
        return "Hit" if edge > 0 else "Miss"
    else:
        return None
    return "Hit" if edge > 0 else "Miss" if edge < 0 else "Push"


def grade(signals: pd.DataFrame, scores: Sequence[Mapping[str, Any]], now: pd.Timestamp) -> Tuple[pd.DataFrame, int]:
    """Grade pending picks from completed events. (log, number graded)."""
    log = finalize_clv(signals, now)
    if log.empty or not scores:
        return log, 0
    done = {e["id"]: e for e in scores if e.get("completed") and e.get("id")}
    n = 0
    for i in log.index[log["STATUS"].eq("Pending")]:
        ev = done.get(log.at[i, "EVENT_ID"])
        if ev is None:
            continue
        res = result_for(log.at[i, "MARKET"], log.at[i, "SELECTION"], log.at[i, "SIG_LINE"], ev)
        if res:
            log.at[i, "RESULT"], log.at[i, "STATUS"] = res, "Graded"
            n += 1
    return log, n


def wilson(hits: float, n: float) -> Tuple[float, float]:
    """95% Wilson interval for a hit rate, as percentages."""
    if n <= 0:
        return float("nan"), float("nan")
    h, z = hits / n, Z95
    denom = 1.0 + z ** 2 / n
    centre = (h + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(h * (1 - h) / n + z ** 2 / (4 * n ** 2)) / denom
    return 100.0 * (centre - half), 100.0 * (centre + half)


def _summarize(group: pd.DataFrame) -> Dict[str, float]:
    decided = group[group["RESULT"].isin(["Hit", "Miss"])]
    hits, n = float((decided["RESULT"] == "Hit").sum()), float(len(decided))
    lo, hi = wilson(hits, n)
    clv = pd.to_numeric(group["CLV"], errors="coerce").dropna()
    return {"SIGNALS": float(len(group)), "GRADED": n, "HIT_%": 100.0 * hits / n if n else np.nan, "LOW_%": lo, "HIGH_%": hi,
            "BEAT_CLOSE_%": 100.0 * float((clv > 0).mean()) if len(clv) else np.nan, "CLV_COUNT": float(len(clv))}


def source(log: pd.DataFrame) -> pd.Series:
    """Why each pick was logged: both, off-market only (money gap did not qualify), whale only, or filtered as noise."""
    off = log["OFF_MARKET"].eq("Y")
    real = log["TIER"].isin(REAL_TIERS)
    return pd.Series(np.select([off & real, off & ~real, real, log["TIER"].eq("FILTERED")], ["Off-market + whale", "Off-market only", "Whale only", "Filtered"], "Other"),
                     index=log.index)


def summary(signals: pd.DataFrame, by: str = "TIER") -> pd.DataFrame:
    """Hit rate with interval and share of picks that beat the close, by TIER or MARKET."""
    log = normalize(signals)
    cols = [by, "SIGNALS", "GRADED", "HIT_%", "LOW_%", "HIGH_%", "BEAT_CLOSE_%", "CLV_COUNT"]
    if log.empty:
        return pd.DataFrame(columns=cols)
    if by == "SOURCE":
        log = log.assign(SOURCE=source(log))
    rows = [{by: k, **_summarize(g)} for k, g in log.groupby(by, sort=True)]
    return pd.DataFrame(rows, columns=cols)


def verdict(signals: pd.DataFrame) -> List[str]:
    """Cautious plain-English reading of the log."""
    all_log = normalize(signals)
    log = all_log[all_log["TIER"].isin(REAL_TIERS)]
    if log.empty:
        return ["No signals logged yet. Open the Board tab on game days: every LEAN / SHARP pick is saved automatically."]
    s = _summarize(log)
    out: List[str] = []
    if s["GRADED"] < MIN_GRADED:
        out.append(f"Too early to judge: {int(s['GRADED'])} graded pick(s). Treat the tables as noise until about {MIN_GRADED}.")
    if s["CLV_COUNT"] >= 20 and np.isfinite(s["BEAT_CLOSE_%"]):
        out.append(f"{s['BEAT_CLOSE_%']:.0f}% of {int(s['CLV_COUNT'])} picks beat the closing line (50% is what luck gives; sustained 55%+ is a real edge).")
    if s["GRADED"] >= MIN_GRADED:
        if s["LOW_%"] > BREAKEVEN_PCT:
            out.append(f"Hit rate {s['HIT_%']:.1f}% with a 95% range of {s['LOW_%']:.1f}-{s['HIGH_%']:.1f}%: above the {BREAKEVEN_PCT:.1f}% break-even at -110.")
        elif s["HIGH_%"] < BREAKEVEN_PCT:
            out.append(f"Hit rate {s['HIT_%']:.1f}% (range {s['LOW_%']:.1f}-{s['HIGH_%']:.1f}%): clearly below the {BREAKEVEN_PCT:.1f}% break-even. The weights may need changing.")
        else:
            out.append(f"Hit rate {s['HIT_%']:.1f}% (range {s['LOW_%']:.1f}-{s['HIGH_%']:.1f}%): can't be told apart from the {BREAKEVEN_PCT:.1f}% break-even yet.")
    src = source(all_log)
    om, wh = _summarize(all_log[src.isin(["Off-market + whale", "Off-market only"])]), _summarize(all_log[src.eq("Whale only")])
    if om["CLV_COUNT"] >= 20 and wh["CLV_COUNT"] >= 20:
        out.append(f"Beat-the-close rate: off-market picks {om['BEAT_CLOSE_%']:.0f}% ({int(om['CLV_COUNT'])}) vs whale-only picks {wh['BEAT_CLOSE_%']:.0f}% ({int(wh['CLV_COUNT'])}): "
                   + ("off-market is the stronger source so far." if om["BEAT_CLOSE_%"] > wh["BEAT_CLOSE_%"] else "the whale signal is holding up at least as well."))
    f = _summarize(all_log[all_log["TIER"].eq("FILTERED")])
    if f["GRADED"] >= MIN_GRADED / 2 and s["GRADED"] >= MIN_GRADED / 2 and np.isfinite(f["HIT_%"]):
        out.append(f"Filtered-out picks hit {f['HIT_%']:.0f}% ({int(f['GRADED'])} graded) vs {s['HIT_%']:.0f}% for the picks that passed: "
                   + ("the filters are removing weaker picks." if f["HIT_%"] < s["HIT_%"] else "the filters are not helping yet; consider loosening them."))
    return out
