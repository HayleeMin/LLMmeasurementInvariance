"""Fetch the pilot corpus: Hacker News comments that mention burnout (2012-2022), yearly comment totals,
and a random background sample (for the centering mean).

Usage:  python scripts/s02_fetch_pilot_corpus.py
Output: data/pilot/burnout_comments.jsonl, data/pilot/year_totals.csv, data/pilot/background.jsonl

Alternative corpus: set pilot_corpus.source: csv and csv_path in config.yaml (columns id,date,author,thread,text)
and skip this script; s03 reads the CSV directly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import hn  # noqa: E402
from src.common import DATA, burnout_pattern, clean_html, load_config, write_jsonl  # noqa: E402

QUERIES = ["burnout", "burn-out", "burned out", "burnt out"]

if __name__ == "__main__":
    cfg = load_config()
    y0, y1 = cfg["pilot_corpus"]["years"]
    pat = burnout_pattern(cfg)
    out = DATA / "pilot"
    docs = {}
    totals = []
    for y in range(y0, y1 + 1):
        t0, t1 = hn.year_bounds(y)
        totals.append({"year": y, "comments": hn.count("", t0, t1)})
        for q in QUERIES:
            for h in hn.fetch_window(q, t0, t1):
                h["text"] = clean_html(h.pop("html"))
                if pat.search(h["text"]):
                    docs[h["id"]] = h
        print(f"{y}: {sum(1 for d in docs.values() if d['year'] == y)} burnout comments so far")
    write_jsonl(out / "burnout_comments.jsonl", list(docs.values()))
    pd.DataFrame(totals).to_csv(out / "year_totals.csv", index=False)

    # Background sample: 300 random one-hour windows across the period, up to 40 comments each.
    r = np.random.default_rng(cfg["seed"])
    lo, hi = hn.year_bounds(y0)[0], hn.year_bounds(y1)[1]
    bg = []
    for t in r.integers(lo, hi - 3600, size=300):
        for h in hn.fetch_window("", int(t), int(t) + 3600, cap=40):
            h["text"] = clean_html(h.pop("html"))
            if h["text"]:
                bg.append(h)
    write_jsonl(out / "background.jsonl", bg)
    print(f"Saved {len(docs)} burnout comments, {len(bg)} background comments.")
