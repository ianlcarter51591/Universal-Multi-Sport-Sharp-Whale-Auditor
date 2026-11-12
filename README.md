[README.md](https://github.com/user-attachments/files/34721118/README.md)
# Universal Multi-Sport Sharp & Whale Auditor

Streamlit app. Main file: `UNIVERSALSWapp.py`. Upload EVERY file in this folder to GitHub (the app names any missing file on screen).

## Setup
1. Streamlit Cloud -> Settings -> Secrets: `ODDS_API_KEY = "..."`.
2. Optional but recommended: `GITHUB_GIST_TOKEN = "..."` (classic GitHub token with only the "gist" permission). Line history and the
   signal log then survive restarts. Without it the app works but forgets everything when Streamlit restarts it.

## Tabs
- **Board**: every game and market scored at once (no extra API credits), one row per market: the side with the money.
- **Audit a game**: one selection in detail, both sides of the market, and the checklist breakdown.
- **Signal log**: every LEAN / SHARP pick is saved with its line; after the games, grade them to see hit rate with a 95% range and how
  often the pick beat the closing line.

## Score (0-100)
Overlapping checks are grouped so each idea counts once. Money 40% = clip((handle% - tickets% - 5) / 30, 0, 1). Market reaction 35% =
the larger of steam (line moved toward the pick, 2x the threshold = full) and reverse line movement (line moved toward the minority
side); a move AWAY from the pick is negative. Timing 15% = an early or late move toward the pick. Key numbers 10% (spreads only).
Groups that can't be measured yet are dropped and the rest re-weighted. NO DATA = no real splits. SHARP needs score >= 60 and the line
confirming the money; LEAN needs score >= 40. The weights are assumptions: the signal log is how you check them.

## Limits
- Line movement is measured from the first line this app saw, not the true opener, and only for leagues you open.
- DraftKings Network gives percentages only (no dollar volume), so a thin market can look as strong as a deep one.
- The "closing line" is the last line the app saw; Odds API scores only reach back 3 days, so grade often.
- `playerprops_splits.py` from the old version is not used by anything and can be deleted from the repo.
