"""Shared helpers: config, paths, text processing."""
import html
import json
import re
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
MATERIALS = ROOT / "materials"

CATEGORY_LABELS = {
    "exhaustion": "Exhaustion",
    "cynicism": "Cynicism / depersonalization",
    "reduced_efficacy": "Reduced professional efficacy",
    "depressive_symptoms": "Depressive symptoms",
    "ordinary_tiredness": "Ordinary tiredness",
    "other_none": "Other / none",
}
LABEL_TO_KEY = {v: k for k, v in CATEGORY_LABELS.items()}


def load_config():
    return yaml.safe_load(open(ROOT / "config.yaml"))


def load_anchors():
    a = yaml.safe_load(open(MATERIALS / "anchors.yaml"))
    return {k: v["anchors"] for k, v in a.items()}


def burnout_pattern(cfg=None):
    cfg = cfg or load_config()
    return re.compile(cfg["pilot_corpus"]["burnout_regex"], re.IGNORECASE)


_TAG = re.compile(r"<[^>]+>")


def clean_html(s):
    if not s:
        return ""
    s = s.replace("<p>", "\n").replace("</p>", "\n")
    s = _TAG.sub(" ", s)
    s = html.unescape(s)
    return re.sub(r"[ \t]+", " ", s).strip()


_SENT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])|\n+")


def split_sentences(text):
    return [s.strip() for s in _SENT.split(text) if s and s.strip()]


def extract_usages(text, pattern):
    """Return usages: (usage_text, span_start, span_end) where the span marks the matched focal form.

    A usage is the sentence containing the match plus the next sentence (or the previous one
    if there is no next sentence; or the whole text if it is a single sentence)."""
    sents = split_sentences(text)
    out = []
    for i, s in enumerate(sents):
        m = pattern.search(s)
        if not m:
            continue
        if i + 1 < len(sents):
            usage, offset = s + " " + sents[i + 1], 0
        elif i > 0:
            usage = sents[i - 1] + " " + s
            offset = len(sents[i - 1]) + 1
        else:
            usage, offset = s, 0
        out.append((usage, offset + m.start(), offset + m.end()))
    return out


def find_span(text, pattern):
    m = pattern.search(text)
    return (m.start(), m.end()) if m else None


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def rng(cfg=None):
    cfg = cfg or load_config()
    return np.random.default_rng(cfg["seed"])
