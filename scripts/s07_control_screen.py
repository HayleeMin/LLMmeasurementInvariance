"""Control-word screen in the pilot corpus (protocol §2 and the Table S2 rule).

For each candidate control term:
  (a) yearly document frequency (comments containing the term per million comments), 2012-2022
  (b) remote-work co-occurrence share in 2019 vs 2021 (usage window = sentence + adjacent sentence)
  (c) exclusion if share at least doubles AND rises >= 1 percentage point
Final set: passes (c), has >= min_usages_per_year in every year, and log10 frequency within +/-1 of burnout's
(widened to +/-1.5 if fewer than control_screen.min_final remain).

Usage:  python scripts/s07_control_screen.py
Output: outputs/control_screen.csv, outputs/final_controls.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import hn  # noqa: E402
from src.common import DATA, MATERIALS, OUT, burnout_pattern, clean_html, extract_usages, load_config, read_jsonl  # noqa: E402


def term_pattern(term):
    return re.compile(r"\b" + re.escape(term) + r"(s|es)?\b", re.IGNORECASE)


def cooccurrence_share(texts, tpat, lex_pat):
    n = hit = 0
    for t in texts:
        for usage, _, _ in extract_usages(t, tpat):
            n += 1
            hit += bool(lex_pat.search(usage))
    return (hit / n if n else np.nan), n


def screen_row(term, docs_by_year, totals, cfg):
    cs = cfg["control_screen"]
    tpat = term_pattern(term)
    lex_pat = re.compile(r"\b(" + "|".join(map(re.escape, cs["lexicon"])) + r")\b", re.IGNORECASE)
    row = {"term": term}
    for y, docs in docs_by_year.items():
        exact = [d for d in docs if tpat.search(d)]
        row[f"docs_{y}"] = len(exact)
        row[f"per_million_{y}"] = 1e6 * len(exact) / totals[y] if totals.get(y) else np.nan
    for y in (2019, 2021):
        share, n = cooccurrence_share(docs_by_year.get(y, []), tpat, lex_pat)
        row[f"cooc_{y}"], row[f"cooc_n_{y}"] = share, n
    s19, s21 = row["cooc_2019"], row["cooc_2021"]
    ratio = np.inf if (s19 == 0 and s21 > 0) else (s21 / s19 if s19 else np.nan)
    row["cooc_ratio"], row["cooc_rise_pts"] = ratio, 100 * (s21 - s19)
    row["fails_remote_screen"] = bool(ratio >= cs["min_ratio"] and row["cooc_rise_pts"] >= cs["min_rise_points"])
    return row


def finalize(df, burnout_logfreq, cfg):
    cs = cfg["control_screen"]
    years = [c.split("_")[-1] for c in df.columns if c.startswith("per_million_")]
    df["log10_per_million"] = np.log10(df[[f"per_million_{y}" for y in years]].mean(1))
    df["min_docs_any_year"] = df[[f"docs_{y}" for y in years]].min(1)
    base = (~df.fails_remote_screen) & (df.min_docs_any_year >= cs["min_usages_per_year"])
    band = cs["frequency_band_log10"]
    keep = base & ((df.log10_per_million - burnout_logfreq).abs() <= band)
    widened = False
    if cs.get("min_final") and keep.sum() < cs["min_final"]:
        keep, widened = base & ((df.log10_per_million - burnout_logfreq).abs() <= 1.5), True
    df["in_final_set"] = keep
    return df, widened


if __name__ == "__main__":
    cfg = load_config()
    cs = cfg["control_screen"]
    y0, y1 = cfg["pilot_corpus"]["years"]
    totals = pd.read_csv(DATA / "pilot" / "year_totals.csv").set_index("year")["comments"].to_dict()
    burn = pd.DataFrame(read_jsonl(DATA / "pilot" / "burnout_comments.jsonl"))
    bpm = burn.groupby("year").size() / pd.Series(totals) * 1e6
    burnout_logfreq = float(np.log10(bpm.loc[y0:y1].mean()))
    terms = pd.read_csv(MATERIALS / "candidate_controls.csv")["term"].tolist()
    rows = []
    for term in terms:
        docs_by_year = {}
        for y in range(y0, y1 + 1):
            t0, t1 = hn.year_bounds(y)
            cap = cs["usages_per_term_year"] if y in (2019, 2021) else 300
            hits = hn.fetch_window(term, t0, t1, cap=cap)
            n_all = hn.count(term, t0, t1)
            texts = [clean_html(h["html"]) for h in hits]
            # Scale API hit count by the share of sampled hits that are exact (word-boundary) matches.
            exact_share = np.mean([bool(term_pattern(term).search(t)) for t in texts]) if texts else 0
            docs_by_year[y] = texts
            rows_y = {"term": term, "year": y, "api_hits": n_all, "exact_share": exact_share}
            rows.append(rows_y)
        r = screen_row(term, docs_by_year, totals, cfg)
        for y in range(y0, y1 + 1):  # replace sample counts with scaled full-corpus estimates
            api = next(x for x in rows if x["term"] == term and x["year"] == y)
            est = api["api_hits"] * api["exact_share"]
            r[f"docs_{y}"], r[f"per_million_{y}"] = est, 1e6 * est / totals[y]
        rows.append(r)
        print(f"{term}: cooc 2019={r['cooc_2019']:.3f} 2021={r['cooc_2021']:.3f} fails={r['fails_remote_screen']}")
    screen = pd.DataFrame([r for r in rows if "cooc_ratio" in r])
    screen, widened = finalize(screen, burnout_logfreq, cfg)
    screen.to_csv(OUT / "control_screen.csv", index=False)
    screen[screen.in_final_set][["term", "log10_per_million"]].to_csv(OUT / "final_controls.csv", index=False)
    print(f"burnout log10 per-million = {burnout_logfreq:.2f}; final controls: {int(screen.in_final_set.sum())} "
          f"(band widened: {widened}); failed remote screen: {int(screen.fails_remote_screen.sum())}")
