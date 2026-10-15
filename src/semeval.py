"""Loader for the SemEval-2020 Task 1 English data (graded change, Subtask 2).

Expected layout (official release, unzipped into config gate1_semeval.data_dir):
  targets.txt                      e.g. "attack_nn"
  truth/graded.txt                 "attack_nn<TAB>0.144"
  corpus1/lemma/*.txt(.gz)         lemmatized, POS-tagged targets (attack_nn)
  corpus1/token/*.txt(.gz)         tokenized running text, line- and token-aligned with lemma
  corpus2/... same for the later period
"""
import gzip
from pathlib import Path


def _open(p):
    return gzip.open(p, "rt", encoding="utf-8") if str(p).endswith(".gz") else open(p, encoding="utf-8")


def _one(d, sub):
    files = sorted((Path(d) / sub).glob("*.txt*"))
    if not files:
        raise FileNotFoundError(f"No corpus file in {Path(d) / sub}")
    return files[0]


def load_targets(data_dir):
    d = Path(data_dir)
    targets = [l.strip() for l in open(d / "targets.txt") if l.strip()]
    truth = {}
    for l in open(d / "truth" / "graded.txt"):
        if l.strip():
            t, v = l.split()
            truth[t] = float(v)
    return targets, truth


def collect_usages(data_dir, period, targets, max_per_target, rng):
    """Reservoir-sample up to max_per_target usages per target from one period.

    Returns {target: [(text, start, end), ...]}, {"skipped_misaligned": n, "lines_with_targets": n}, and background sentences."""
    d = Path(data_dir) / f"corpus{period}"
    lemma_f, token_f = _one(d, "lemma"), _one(d, "token")
    tset = set(targets)
    seen = {t: 0 for t in targets}
    res = {t: [] for t in targets}
    background, skipped, with_targets = [], 0, 0
    with _open(lemma_f) as fl, _open(token_f) as ft:
        for ln, (ll, lt) in enumerate(zip(fl, ft)):
            lt_tokens = lt.split()
            if ln % 200 == 0 and len(background) < 20000 and lt_tokens:
                background.append(" ".join(lt_tokens))
            lemmas = ll.split()
            hits = [i for i, w in enumerate(lemmas) if w in tset]
            if not hits:
                continue
            with_targets += 1
            # The lemma files have no punctuation; the token files keep it. Map lemma positions to token
            # positions directly if lengths match, otherwise via the token file's non-punctuation tokens.
            if len(lemmas) == len(lt_tokens):
                pos_map = list(range(len(lt_tokens)))
            else:
                pos_map = [j for j, w in enumerate(lt_tokens) if any(ch.isalnum() for ch in w)]
                if len(pos_map) != len(lemmas):
                    skipped += 1
                    continue
            text = " ".join(lt_tokens)
            starts = []
            pos = 0
            for w in lt_tokens:
                starts.append(pos)
                pos += len(w) + 1
            for i in hits:
                t = lemmas[i]
                j = pos_map[i]
                item = (text, starts[j], starts[j] + len(lt_tokens[j]))
                seen[t] += 1
                if len(res[t]) < max_per_target:
                    res[t].append(item)
                else:
                    j = rng.integers(0, seen[t])
                    if j < max_per_target:
                        res[t][j] = item
    return res, {"skipped_misaligned": skipped, "lines_with_targets": with_targets}, background
