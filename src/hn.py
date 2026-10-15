"""Hacker News comments via the public Algolia HN Search API (no key needed).

Docs: https://hn.algolia.com/api . Each query returns at most 1,000 hits, so windows are split
recursively until each window holds <= 1,000 hits.
"""
import time
from datetime import datetime, timezone

import requests

API = "https://hn.algolia.com/api/v1/search_by_date"


def _ts(y, m=1, d=1):
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp())


def _get(params, tries=5):
    for k in range(tries):
        try:
            r = requests.get(API, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 ** k)
    raise RuntimeError(f"HN API failed for {params}")


def count(query, t0, t1):
    p = {"query": query, "tags": "comment", "hitsPerPage": 0,
         "numericFilters": f"created_at_i>={t0},created_at_i<{t1}"}
    return _get(p)["nbHits"]


def fetch_window(query, t0, t1, cap=None, sleep=0.2):
    """All comment hits for query in [t0, t1), splitting windows above 1,000 hits. cap limits total."""
    out = []
    n = count(query, t0, t1)
    if n == 0:
        return out
    if n > 1000 and (t1 - t0) > 3600:
        mid = (t0 + t1) // 2
        out += fetch_window(query, t0, mid, cap, sleep)
        if cap is None or len(out) < cap:
            out += fetch_window(query, mid, t1, None if cap is None else cap - len(out), sleep)
        return out
    p = {"query": query, "tags": "comment", "hitsPerPage": 1000,
         "numericFilters": f"created_at_i>={t0},created_at_i<{t1}"}
    hits = _get(p).get("hits", [])
    time.sleep(sleep)
    for h in hits:
        out.append({
            "id": h.get("objectID"),
            "author": h.get("author"),
            "thread": h.get("story_id"),
            "created_at_i": h.get("created_at_i"),
            "year": datetime.fromtimestamp(h.get("created_at_i", 0), tz=timezone.utc).year,
            "html": h.get("comment_text") or "",
        })
    return out if cap is None else out[:cap]


def year_bounds(y):
    return _ts(y), _ts(y + 1)
