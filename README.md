[README.md](https://github.com/user-attachments/files/34769333/README.md)
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

## How a pick is judged
Five yes/no checks, shown as "CHECKS 3/5" (a check can also be "not yet known" until the app has watched long enough):
1. **Money gap**: handle% minus tickets% is 15 points or more.
2. **Bigger bets**: the average bet on this side is 1.5x or more the other side's.
3. **Line moved toward it**: by at least 1 point (spreads / totals) or 3 probability points (moneylines) since the app first saw the line.
4. **Held 30 minutes**: the money signal has been there for 30+ minutes of saved readings.
5. **A market tell**: reverse line movement, an early (24h+ out) or late (closing window) move toward the pick, or a spread crossing a key number.

**LEAN** = check 1 plus at least one more (the money is on this side). **SHARP** = checks 1, 3, 4 and 5: the money is real, the market agrees,
it held, and there is a tell. **PASS** = anything else, including money the line moved AGAINST. **NO DATA** = no real splits.
The NOTE column says what a LEAN still needs to become SHARP. Ties on the Board are broken by how strong the gap and bet size are.

**Noise filters.** A pick that would be LEAN / SHARP but fails one becomes **FILTERED** (Board: "Filtered as noise"): price band (moneylines
-300 to +250, spreads / totals -130 to +110, which removes the +5000 longshot), at least 15% of the tickets, clean splits (no 0% / 100%, two-way
sides add to about 100%), the signal not new or flickering, more than 30 minutes to the start, and games you mark as injury / lineup news.
The thresholds are assumptions; the signal log (which also logs FILTERED picks) shows whether to change them. The log's SCORE column is the
number of checks met (older rows from the 0-100 version still show their old numbers).

## Limits
- Line movement is measured from the first line this app saw, not the true opener, and only for leagues you open.
- DraftKings Network gives percentages only (no dollar volume), so a thin market can look as strong as a deep one.
- The "closing line" is the last line the app saw; Odds API scores only reach back 3 days, so grade often.
- `playerprops_splits.py` from the old version is not used by anything and can be deleted from the repo.
