"""
Durable storage for the apps' CSV files in a private GitHub Gist (Streamlit Cloud wipes the server disk on restart).

How it works
------------
* One secret is needed: GITHUB_GIST_TOKEN (a GitHub token with only the "gist" permission).
* The module finds (or creates) one private gist whose description is GIST_DESCRIPTION and keeps each CSV as a file in it.
* `save_text` always writes the local file first, then pushes to the gist. Pushes are throttled (a board scan saves many
  times in a row) and `flush()` sends whatever is still waiting.
* `load_text` prefers the gist copy (the local disk may have been wiped) and falls back to the local file.
* With no token, or when GitHub is unreachable, everything silently keeps working on local files; `status()` says which.
No function here raises: storage problems must never break an evaluation.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

import requests

API = "https://api.github.com"
GIST_DESCRIPTION = "prop-evaluator-data"
MIN_PUSH_SECONDS = 20.0
TIMEOUT = 12

_STATE: Dict[str, Any] = {"token": "", "gist_id": None, "pending": {}, "synced": set(), "last_push": 0.0, "last_error": "", "last_ok": 0.0}


def configure(token: Optional[str]) -> None:
    """Set (or clear) the GitHub token. Changing the token forgets the cached gist id."""
    token = (token or "").strip()
    if token != _STATE["token"]:
        _STATE.update({"token": token, "gist_id": None, "last_error": "", "pending": {}, "synced": set()})


def configured() -> bool:
    """True when a token is set."""
    return bool(_STATE["token"])


def _headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {_STATE['token']}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


def _gist_id() -> Optional[str]:
    """Id of the data gist: cached, else found by description, else created (private)."""
    if _STATE["gist_id"]:
        return _STATE["gist_id"]
    resp = requests.get(f"{API}/gists", headers=_headers(), params={"per_page": 100}, timeout=TIMEOUT)
    resp.raise_for_status()
    for g in resp.json():
        if g.get("description") == GIST_DESCRIPTION:
            _STATE["gist_id"] = g["id"]
            return g["id"]
    resp = requests.post(f"{API}/gists", headers=_headers(), timeout=TIMEOUT,
                         json={"description": GIST_DESCRIPTION, "public": False,
                               "files": {"README.md": {"content": "Data files for the prop evaluator apps. Do not edit by hand."}}})
    resp.raise_for_status()
    _STATE["gist_id"] = resp.json()["id"]
    return _STATE["gist_id"]


def _read_local(path: str) -> Optional[str]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read()
    except Exception:
        pass
    return None


def _write_local(path: str, text: str) -> None:
    try:
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except Exception:
        pass


def _pull(name: str) -> Tuple[bool, Optional[str]]:
    """(reached GitHub?, text or None). A file that simply isn't in the gist yet is (True, None)."""
    if not configured():
        return False, None
    try:
        resp = requests.get(f"{API}/gists/{_gist_id()}", headers=_headers(), timeout=TIMEOUT)
        resp.raise_for_status()
        f = (resp.json().get("files") or {}).get(name)
        if not f:
            return True, None
        if f.get("truncated") and f.get("raw_url"):        # files over ~1 MB come back cut off: fetch the full copy
            raw = requests.get(f["raw_url"], headers=_headers(), timeout=TIMEOUT)
            raw.raise_for_status()
            return True, raw.text
        return True, f.get("content")
    except Exception as e:
        _STATE["last_error"] = f"read failed: {str(e)[:120]}"
        return False, None


def pull(name: str) -> Optional[str]:
    """Text of `name` in the gist, or None (no token, file absent, or GitHub unreachable)."""
    ok, text = _pull(name)
    if ok:
        _STATE["synced"].add(name)
    return text


def flush() -> bool:
    """Push every waiting file now. True if nothing was waiting or the push succeeded."""
    if not _STATE["pending"]:
        return True
    if not configured():
        _STATE["pending"].clear()
        return True
    try:
        ready = {n: t for n, t in _STATE["pending"].items() if n in _STATE["synced"]}   # never push a file we haven't read
        if not ready:
            return False
        files = {n: {"content": t if t.strip() else "\n"} for n, t in ready.items()}
        resp = requests.patch(f"{API}/gists/{_gist_id()}", headers=_headers(), json={"files": files}, timeout=TIMEOUT)
        resp.raise_for_status()
        for n in ready:
            _STATE["pending"].pop(n, None)
        _STATE.update({"last_push": time.time(), "last_ok": time.time(), "last_error": ""})
        return True
    except Exception as e:
        _STATE.update({"last_push": time.time(), "last_error": f"save failed: {str(e)[:120]}"})   # keep pending; retry later
        return False


def save_text(name: str, text: str, local_path: Optional[str] = None, force: bool = False) -> Optional[str]:
    """Write locally, queue for the gist, and push if forced or the throttle window has passed.

    Safety: a file is never pushed before it has been read from the gist once (otherwise a start-up with GitHub unreachable
    would overwrite saved history with a near-empty file). If that first read happens here and finds saved data, NOTHING is
    pushed and the saved text is returned: the caller must merge it into its data and call save_text again.
    """
    _write_local(local_path or name, text)
    if not configured():
        return None
    if name not in _STATE["synced"]:
        ok, remote = _pull(name)
        if not ok:
            _STATE["pending"][name] = text            # kept; pushed once GitHub is reachable again
            return None
        _STATE["synced"].add(name)
        if remote is not None and remote.strip():
            return remote
    _STATE["pending"][name] = text
    if force or time.time() - _STATE["last_push"] >= MIN_PUSH_SECONDS:
        flush()
    return None


def load_text(name: str, local_path: Optional[str] = None) -> Optional[str]:
    """Gist copy if available (and mirror it to disk), else the local file, else None."""
    remote = pull(name)
    if remote is not None and remote.strip():
        _write_local(local_path or name, remote)
        return remote
    return _read_local(local_path or name)


def status() -> Dict[str, Any]:
    """{'mode': 'cloud'|'local', 'text': plain-English line for the UI, 'level': 'ok'|'warn'}."""
    if not configured():
        return {"mode": "local", "level": "warn",
                "text": "Saved on the app server only (it resets on restart). Add GITHUB_GIST_TOKEN to your secrets for permanent storage."}
    if _STATE["last_error"]:
        return {"mode": "cloud", "level": "warn", "text": f"Permanent storage problem ({_STATE['last_error']}). Your data is still saved locally; it will retry."}
    waiting = len(_STATE["pending"])
    return {"mode": "cloud", "level": "ok",
            "text": "Saved permanently to your private GitHub Gist." + (f" ({waiting} file(s) waiting to sync)" if waiting else "")}
