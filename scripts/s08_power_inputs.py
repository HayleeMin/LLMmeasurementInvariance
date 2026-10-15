"""Collect the pilot quantities the sample-size simulation needs (protocol §8). No temporal effects are estimated.

Usage:  python scripts/s08_power_inputs.py [--all-usages]
  --all-usages  also embeds every pilot usage with the selected model to (a) estimate author/thread ICCs on
                predicted categories and (b) estimate the noise-floor SD of drift from random split-halves.
Output: outputs/power_inputs.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATA, OUT, burnout_pattern, find_span, load_anchors, load_config, read_jsonl, rng  # noqa: E402
from src.drift import prepare, usage_drift  # noqa: E402
from src.stats import centroid_features, icc1  # noqa: E402

COMPS = ["exhaustion", "cynicism", "reduced_efficacy", "depressive_symptoms", "ordinary_tiredness"]


def coded_sample_inputs(labels):
    shares = labels["final"].value_counts(normalize=True).to_dict()
    iccs = {}
    for grp in ["author", "thread"]:
        g = labels[grp].fillna("NA").astype(str)
        iccs[grp] = {c: icc1((labels["final"] == c).astype(int), g) for c in sorted(labels["final"].unique())}
    return shares, iccs


def corpus_inputs(usages, totals):
    df = pd.DataFrame(usages)
    per_year = df.groupby("year").agg(usages=("usage_id", "size"), comments=("doc_id", "nunique"),
                                      authors=("author", "nunique"), threads=("thread", "nunique"))
    per_year["usages_per_million_comments"] = 1e6 * per_year.usages / pd.Series(totals)
    return per_year.reset_index().to_dict("records")


def all_usage_inputs(emb, sel, labels, usages, background, cfg):
    """ICCs on predicted categories over all usages, and split-half noise-floor SD of drift."""
    pat = burnout_pattern(cfg)
    anchors = load_anchors()
    L = int(sel["layer"]) - 1
    center, _ = emb.random_token_mean(background, cfg["embedding"]["centering_tokens"], rng(cfg))
    a_txt = [s for c in COMPS for s in anchors[c]]
    a_comp = [c for c in COMPS for _ in anchors[c]]
    A = prepare(emb.span_vectors(a_txt, [find_span(t, pat) for t in a_txt])[:, L], center[L])
    by = {k: A[[i for i, x in enumerate(a_comp) if x == k]] for k in COMPS}
    U_lab = prepare(emb.span_vectors(labels.text.tolist(), list(zip(labels.start, labels.end)))[:, L], center[L])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(centroid_features(U_lab, by, COMPS), labels["final"])
    df = pd.DataFrame(usages)
    V = emb.span_vectors(df.text.tolist(), list(zip(df.start, df.end)))[:, L]
    ok = ~np.isnan(V).any(1)
    df, V = df[ok].reset_index(drop=True), prepare(V[ok], center[L])
    df["pred"] = clf.predict(centroid_features(V, by, COMPS))
    iccs = {grp: {c: icc1((df.pred == c).astype(int), df[grp].fillna("NA").astype(str)) for c in sorted(df.pred.unique())}
            for grp in ["author", "thread"]}
    r = rng(cfg)
    null = []
    for _ in range(200):
        idx = r.permutation(len(V))
        h = len(idx) // 2
        null.append(usage_drift(V[idx[:h]], V[idx[h:2 * h]]))
    return {"icc_predicted": iccs, "null_drift_sd": float(np.std(null)), "null_drift_mean": float(np.mean(null)),
            "n_usages_embedded": int(len(V)), "predicted_shares": df.pred.value_counts(normalize=True).to_dict()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-usages", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    labels = pd.read_csv(DATA / "pilot" / "final_labels.csv")
    usages = read_jsonl(DATA / "pilot" / "usages.jsonl")
    totals = pd.read_csv(DATA / "pilot" / "year_totals.csv").set_index("year")["comments"].to_dict()
    sel = json.load(open(OUT / "selected_model.json"))
    conf = json.load(open(OUT / "gate2_confusion.json"))
    shares, iccs = coded_sample_inputs(labels)
    res = {"selected_model": sel, "category_shares_coded": shares, "icc_coded_sample": iccs,
           "per_year": corpus_inputs(usages, totals),
           "confusion_selected": conf.get(f"{sel['model']}|{sel['layer']}") if sel else None}
    if args.all_usages and sel:
        from src.embedder import HFEmbedder
        e = cfg["embedding"]
        emb = HFEmbedder(sel["model"], sel.get("revision"), e["max_length"], e["batch_size"])
        bg = [d["text"] for d in read_jsonl(DATA / "pilot" / "background.jsonl")]
        res.update(all_usage_inputs(emb, sel, labels, usages, bg, cfg))
    json.dump(res, open(OUT / "power_inputs.json", "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "per_year"}, indent=2, default=float)[:3000])
