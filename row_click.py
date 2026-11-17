"""Clickable tables: click a row and it opens in the Audit tab (same idea as the NBA / NFL prop apps)."""
from __future__ import annotations

import zlib
from typing import Any, Callable, Mapping

import pandas as pd

HINT = "👆 Click a row (the box at its left edge) to open that pick in the **🔍 Audit a game** tab."


def selected_row(event: Any) -> Any:
    """Index of the picked row in a st.dataframe selection event, or None."""
    sel = getattr(event, "selection", None)
    if sel is None and isinstance(event, Mapping):
        sel = event.get("selection")
    rows = getattr(sel, "rows", None)
    if rows is None and isinstance(sel, Mapping):
        rows = sel.get("rows")
    return int(rows[0]) if rows else None


def clickable_table(st: Any, shown: Any, payload: pd.DataFrame, open_fn: Callable[[Mapping[str, Any]], str], key: str,
                    hint: str = HINT, **kwargs: Any) -> None:
    """A st.dataframe whose rows can be clicked: the clicked `payload` row (same order as `shown`) goes to `open_fn(row) -> message`,
    then the page reruns. Each table remembers its own last click, so an old selection never re-opens anything."""
    ss = st.session_state
    try:
        ident = key + "_" + str(zlib.crc32(payload.to_csv(index=False).encode()))      # new rows -> fresh selection
    except Exception:
        ident = key
    if hint:
        st.caption(hint)
    try:
        event = st.dataframe(shown, on_select="rerun", selection_mode="single-row", key=ident, **kwargs)
    except TypeError:                                    # Streamlit older than 1.35: no row clicks, plain table
        st.dataframe(shown, **kwargs)
        return
    i = selected_row(event)
    if i is None or i >= len(payload):
        return
    token = (ident, i)
    if ss.get("clicked_" + key) == token:
        if ss.get("clicked_msg_" + key):
            st.success(ss["clicked_msg_" + key])
        return
    ss["clicked_" + key] = token
    try:
        ss["clicked_msg_" + key] = open_fn(payload.iloc[i].to_dict())
    except Exception as e:
        ss["clicked_msg_" + key] = ""
        st.warning(f"Couldn't open that pick: {str(e)[:120]}")
        return
    st.rerun()


def target_from_row(row: Mapping[str, Any]) -> dict:
    """What the Audit tab should select for a board / signal-log row."""
    line = pd.to_numeric(pd.Series([row.get("LINE")]), errors="coerce").iloc[0]
    return {"event_id": str(row.get("EVENT_ID", "")), "market": str(row.get("MARKET", "")), "selection": str(row.get("SELECTION", "")),
            "line": None if pd.isna(line) else float(line), "game": str(row.get("GAME", "")), "pick": str(row.get("PICK", "") or row.get("SELECTION", ""))}


def outcome_index(outcomes: list, selection: str, line: Any) -> int:
    """Index of the outcome named `selection` (same line when given; else the first with that name); 0 if none."""
    named = [i for i, o in enumerate(outcomes) if str(o.get("name")) == selection or str(o.get("description", "")) == selection]
    if line is not None:
        exact = [i for i in named if o_point(outcomes[i]) == float(line)]
        if exact:
            return exact[0]
    return named[0] if named else 0


def o_point(o: Mapping[str, Any]) -> Any:
    try:
        return float(o.get("point"))
    except (TypeError, ValueError):
        return None
