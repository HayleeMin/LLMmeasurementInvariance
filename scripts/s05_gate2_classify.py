"""Gate 2: cross-fitted classification of coded burnout usages from anchor-centroid similarities.

Usage:  python scripts/s05_gate2_classify.py [--models ...]
Needs:  data/pilot/final_labels.csv (from s04 finalize), data/pilot/background.jsonl (centering)
Output: outputs/gate2_results.csv, outputs/gate2_confusion.json, outputs/cache/pilot_vectors_<model>.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATA, OUT, burnout_pattern, find_span, load_anchors, load_config, read_jsonl, rng  # noqa: E402
from src.drift import prepare  # noqa: E402
from src.embedder import HFEmbedder  # noqa: E402
from src.stats import centroid_features, classification_report, crossfit_classifier  # noqa: E402

COMPS = ["exhaustion", "cynicism", "reduced_efficacy", "depressive_symptoms", "ordinary_tiredness"]
CLASSES = COMPS + ["other_none"]


def run(embedder_factory, specs, cfg, labels, background, merge_et=False):
    pat = burnout_pattern(cfg)
    anchors = load_anchors()
    classes = list(CLASSES)
    y_keys = labels["final"].tolist()
    if merge_et:  # coders could not separate exhaustion from tiredness: merge, per protocol
        y_keys = ["exhaustion" if k == "ordinary_tiredness" else k for k in y_keys]
        classes.remove("ordinary_tiredness")
    y = np.array([classes.index(k) for k in y_keys])
    texts = labels["text"].tolist()
    spans = list(zip(labels["start"], labels["end"]))
    a_txt, a_comp = [], []
    for c in COMPS:
        for s in anchors[c]:
            a_txt.append(s)
            a_comp.append(c)
    a_spans = [find_span(t, pat) for t in a_txt]
    assert all(a_spans), "Every anchor must contain the focal word"

    rows, confs = [], {}
    for spec in specs:
        emb = embedder_factory(spec)
        center, _ = emb.random_token_mean(background, cfg["embedding"]["centering_tokens"], rng(cfg))
        U = emb.span_vectors(texts, spans)
        A = emb.span_vectors(a_txt, a_spans)
        ok = ~np.isnan(U).any((1, 2))
        np.savez_compressed(OUT / "cache" / f"pilot_vectors_{spec['name'].replace('/', '_')}.npz", U=U, A=A, center=center)
        layer_keys = list(range(U.shape[1])) + ["last4"]
        for L in layer_keys:
            if L == "last4":
                u, a, c = U[:, -4:].mean(1), A[:, -4:].mean(1), center[-4:].mean(0)
            else:
                u, a, c = U[:, L], A[:, L], center[L]
            u, a = prepare(u[ok], c), prepare(a, c)
            by = {k: a[[i for i, x in enumerate(a_comp) if x == k]] for k in COMPS}
            X = centroid_features(u, by, COMPS)
            probs, k_eff = crossfit_classifier(X, y[ok], len(classes), cfg["gate2_classification"]["k_folds"], cfg["seed"])
            rep = classification_report(probs, y[ok], len(classes))
            lname = L if L == "last4" else L + 1
            confs[f"{spec['name']}|{lname}"] = {"classes": classes, "matrix": rep.pop("confusion")}
            rows.append({"model": spec["name"], "revision": emb.resolved_revision(), "layer": lname,
                         "n": int(ok.sum()), "k_folds": k_eff, **rep})
        del emb
    df = pd.DataFrame(rows)
    g = cfg["gate2_classification"]
    df["passes_gate2"] = (df.macro_f1 >= g["min_macro_f1"]) & (df.ece <= g["max_ece"])
    return df, confs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    args = ap.parse_args()
    cfg = load_config()
    specs = [m for m in cfg["models"] if not args.models or m["name"] in args.models]
    labels = pd.read_csv(DATA / "pilot" / "final_labels.csv")
    bg = [d["text"] for d in read_jsonl(DATA / "pilot" / "background.jsonl")]
    agr = json.load(open(OUT / "agreement.json")) if (OUT / "agreement.json").exists() else {}
    e = cfg["embedding"]
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    factory = lambda s: HFEmbedder(s["name"], s.get("revision"), e["max_length"], e["batch_size"])  # noqa: E731
    df, confs = run(factory, specs, cfg, labels, bg, merge_et=agr.get("merge_exhaustion_tiredness", False))
    df.to_csv(OUT / "gate2_results.csv", index=False)
    json.dump(confs, open(OUT / "gate2_confusion.json", "w"))
    print(df.sort_values("macro_f1", ascending=False).groupby("model").head(1).to_string(index=False))
