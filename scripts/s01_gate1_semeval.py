"""Gate 1: Spearman correlation of usage drift with SemEval-2020 English graded change, per model and layer.

Usage:  python scripts/s01_gate1_semeval.py [--models bert-base-uncased ...]
Output: outputs/gate1_results.csv
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import OUT, ROOT, load_config, rng  # noqa: E402
from src.drift import apd, prepare, usage_drift  # noqa: E402
from src.embedder import HFEmbedder  # noqa: E402
from src.semeval import collect_usages, load_targets  # noqa: E402


def run(embedder_factory, model_specs, cfg, data_dir, out_path):
    g = cfg["gate1_semeval"]
    targets, truth = load_targets(data_dir)
    cache = OUT / "cache" / "semeval_usages.pkl"
    if cache.exists():
        usages, bg, info = pickle.load(open(cache, "rb"))
    else:
        r = rng(cfg)
        u1, i1, b1 = collect_usages(data_dir, 1, targets, g["max_usages_per_period"], r)
        u2, i2, b2 = collect_usages(data_dir, 2, targets, g["max_usages_per_period"], r)
        usages, bg, info = (u1, u2), b1 + b2, (i1, i2)
        cache.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump((usages, bg, info), open(cache, "wb"))
    print(f"Targets: {len(targets)}; misaligned lines skipped: {info}")

    rows = []
    for spec in model_specs:
        emb = embedder_factory(spec)
        center, n_tok = emb.random_token_mean(bg, cfg["embedding"]["centering_tokens"], rng(cfg))
        print(f"[{spec['name']}] centering mean from {n_tok} tokens")
        drift_by_layer, apd_by_layer, used = {}, {}, []
        for t in targets:
            p1, p2 = usages[0][t], usages[1][t]
            if len(p1) < 2 or len(p2) < 2:
                print(f"  skip {t}: too few usages ({len(p1)}, {len(p2)})")
                continue
            v1 = emb.span_vectors([x[0] for x in p1], [(x[1], x[2]) for x in p1])
            v2 = emb.span_vectors([x[0] for x in p2], [(x[1], x[2]) for x in p2])
            v1, v2 = v1[~np.isnan(v1).any((1, 2))], v2[~np.isnan(v2).any((1, 2))]
            n_layers = v1.shape[1]
            layer_keys = list(range(n_layers)) + ["last4"]
            for L in layer_keys:
                if L == "last4":
                    a, b = v1[:, -4:].mean(1), v2[:, -4:].mean(1)
                    c = center[-4:].mean(0)
                else:
                    a, b, c = v1[:, L], v2[:, L], center[L]
                a, b = prepare(a, c), prepare(b, c)
                drift_by_layer.setdefault(L, []).append(usage_drift(a, b))
                apd_by_layer.setdefault(L, []).append(apd(a, b))
            used.append(t)
        gold = [truth[t] for t in used]
        for L in drift_by_layer:
            rho = spearmanr(drift_by_layer[L], gold).statistic
            rho_apd = spearmanr(apd_by_layer[L], gold).statistic
            rows.append({"model": spec["name"], "revision": emb.resolved_revision(),
                         "layer": L if L == "last4" else L + 1, "spearman_drift": rho,
                         "spearman_apd": rho_apd, "n_targets": len(used)})
        del emb
    df = pd.DataFrame(rows)
    df["passes_gate1"] = df["spearman_drift"] >= g["min_spearman"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    best = df.sort_values("spearman_drift", ascending=False).groupby("model").head(1)
    print(best.to_string(index=False))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    args = ap.parse_args()
    cfg = load_config()
    specs = [m for m in cfg["models"] if not args.models or m["name"] in args.models]
    e = cfg["embedding"]
    factory = lambda s: HFEmbedder(s["name"], s.get("revision"), e["max_length"], e["batch_size"])  # noqa: E731
    run(factory, specs, cfg, ROOT / cfg["gate1_semeval"]["data_dir"], OUT / "gate1_results.csv")
