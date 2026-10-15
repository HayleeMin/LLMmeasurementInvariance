# Burnout semantic-drift RR: pilot pipeline

This runs the Stage 1 pilot: Gate 1 (SemEval benchmark), Gate 2 (classifying coded *burnout* usages), model selection, the control-word screen, and the inputs for the sample-size simulation. The pipeline was tested end to end on synthetic data (`python tests/test_end_to_end.py`). The steps that download models or data have not been run yet, because they need internet access and ideally a GPU.

## 0. Lock first (before anyone looks at pilot data)

1. Review `materials/anchors.yaml` and `materials/Pilot_Lock_Document.docx` and edit them as needed. Fill the model commit hashes in `config.yaml`. Get each hash from the model page on Hugging Face under "Files and versions" (the latest commit).
2. Send `materials/Expert_Anchor_Sort.xlsx` to 3–5 experts. Keep `materials/anchor_key.csv` private; it is the answer key.
3. When the sorts come back, run `python scripts/score_anchor_sort.py returned/*.xlsx`. Revise any anchor placed correctly by fewer than 80% of experts. If you edit `anchors.yaml`, rerun `python scripts/make_anchor_sort.py`.
4. Post `config.yaml`, `anchors.yaml`, and the lock document on OSF as a time-stamped registration. That date is the "[date]" in the manuscript.

## Setup

Use Colab with a GPU runtime (open `pilot_colab.ipynb`), a lab GPU server, or a laptop (CPU works, but the 0.6B model is slow).

```
pip install -r requirements.txt
```

## Run order

| Step | Command | Needs | Time (GPU) |
|---|---|---|---|
| Gate 1 | `python scripts/s01_gate1_semeval.py` | SemEval-2020 Task 1 English data unzipped to `data/semeval2020_ulscd_eng/` (targets.txt, truth/graded.txt, corpus1/2 lemma+token) | ~20–40 min for 3 models |
| Pilot corpus | `python scripts/s02_fetch_pilot_corpus.py` | internet (Hacker News search API) | ~15–30 min |
| Coding sheets | `python scripts/s03_make_coding_sheets.py` | step above | 1 min |
| **Human coding** | two coders fill `materials/Coding_Sheet_CoderA.xlsx` / `CoderB.xlsx` independently | coding guide (sheet "Guide" + lock doc §2) | ~3–4 h each for 400 items |
| Agreement | `python scripts/s04_agreement.py check A.xlsx B.xlsx` | both sheets | 1 min |
| **Resolve** | discuss disagreements, fill `resolved` in `materials/Disagreements.xlsx` | | ~1 h |
| Final labels | `python scripts/s04_agreement.py finalize A.xlsx B.xlsx materials/Disagreements.xlsx` | | 1 min |
| Gate 2 | `python scripts/s05_gate2_classify.py` | final labels | ~10 min |
| Selection | `python scripts/s06_select_model.py` | gate 1 + 2 results | 1 min |
| Control screen | `python scripts/s07_control_screen.py` | internet | ~30–60 min (API calls) |
| Power inputs | `python scripts/s08_power_inputs.py --all-usages` | selected model | ~10 min |

Run the Gate 1 and corpus steps while the anchor sort is out with the experts. Gate 1 never touches the pilot corpus, and the corpus fetch doesn't involve coding.

## What to send back to me

Send these files from `outputs/` and I'll write them into the manuscript (§7, §8, and the pilot paragraph):
- `gate1_results.csv`
- `gate2_results.csv`
- `agreement.json`
- `model_selection.md`
- `selected_model.json`
- `control_screen.csv`
- `final_controls.csv`
- `power_inputs.json`
- `anchor_sort_results.csv`

## Notes

- Gate 2 merges exhaustion and tiredness automatically if their agreement is α < .667, as the protocol requires.
- If no model–layer combination passes both gates, `s06` says so. Under the protocol, H2 is then not tested.
- To use a different public corpus instead of Hacker News, set `pilot_corpus.source: csv` and `csv_path` in `config.yaml` (columns: `id, date, author, thread, text`), skip `s02`, and supply your own `data/pilot/year_totals.csv` and `data/pilot/background.jsonl`.
- The Hacker News search API returns at most 1,000 hits per query, so the scripts split queries into smaller date windows automatically. Control-term frequencies are estimated from API hit counts, scaled by the share of exact word matches in a sample.
