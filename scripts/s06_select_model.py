"""Apply the locked selection rule: among model-layer combinations passing both gates, choose the highest
Gate 1 Spearman correlation. Writes outputs/model_selection.md and outputs/selected_model.json.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import OUT, load_config  # noqa: E402


def select(g1, g2, cfg):
    g1, g2 = g1.copy(), g2.copy()
    g1["layer"], g2["layer"] = g1.layer.astype(str), g2.layer.astype(str)
    m = g1.merge(g2, on=["model", "layer"], how="outer", suffixes=("_g1", "_g2"))
    m["passes_gate1"] = m.passes_gate1.fillna(False).astype(bool)
    m["passes_gate2"] = m.passes_gate2.fillna(False).astype(bool)
    m["passes_both"] = m.passes_gate1 & m.passes_gate2
    primary = m[m.layer != "last4"]
    passing = primary[primary.passes_both].sort_values("spearman_drift", ascending=False)
    chosen = passing.iloc[0].to_dict() if len(passing) else None
    return m, chosen


if __name__ == "__main__":
    cfg = load_config()
    g1, g2 = pd.read_csv(OUT / "gate1_results.csv"), pd.read_csv(OUT / "gate2_results.csv")
    m, chosen = select(g1, g2, cfg)
    m.to_csv(OUT / "gate_summary.csv", index=False)
    lines = ["# Pilot model selection", "",
             f"Gate 1: Spearman >= {cfg['gate1_semeval']['min_spearman']}. "
             f"Gate 2: macro-F1 >= {cfg['gate2_classification']['min_macro_f1']} and ECE <= {cfg['gate2_classification']['max_ece']}.", ""]
    for model, g in m[m.layer != "last4"].groupby("model"):
        b1 = g.sort_values("spearman_drift", ascending=False).iloc[0]
        b2 = g.sort_values("macro_f1", ascending=False).iloc[0]
        lines.append(f"- **{model}**: best Gate 1 rho = {b1.spearman_drift:.3f} (layer {b1.layer}); "
                     f"best Gate 2 macro-F1 = {b2.macro_f1:.3f}, ECE = {b2.ece:.3f} (layer {b2.layer}); "
                     f"layers passing both gates: {int(g.passes_both.sum())}")
    lines.append("")
    if chosen:
        lines.append(f"**Selected:** {chosen['model']}, layer {chosen['layer']} "
                     f"(rho = {chosen['spearman_drift']:.3f}, macro-F1 = {chosen['macro_f1']:.3f}, ECE = {chosen['ece']:.3f}).")
        sel = {k: chosen[k] for k in ["model", "layer", "spearman_drift", "macro_f1", "ece", "brier"]}
        sel["revision"] = chosen.get("revision_g1")
        r = m[(m.model != chosen["model"]) & (m.layer != "last4") & m.passes_both].sort_values("spearman_drift", ascending=False)
        sel["runner_up"] = None if r.empty else {"model": r.iloc[0].model, "layer": r.iloc[0].layer}
    else:
        lines.append("**No model-layer combination passes both gates. Per protocol, H2 is not tested.**")
        sel = None
    (OUT / "model_selection.md").write_text("\n".join(lines))
    json.dump(sel, open(OUT / "selected_model.json", "w"), indent=2, default=str)
    print("\n".join(lines))
