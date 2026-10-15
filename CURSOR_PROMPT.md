# Instructions for the coding agent: run the burnout RR pilot

You are helping a researcher run the pilot study for a Stage 1 Registered Report (*Psychological Science*) on whether the everyday meaning of "burnout" changed between 2012 and 2022. This repository already contains a complete, tested pipeline. Your job is to **run it, check every step, fix only mechanical bugs, and report the results**. You are not designing the study.

Read `README.md` and `config.yaml` first, then follow this file step by step. After each step, tell the researcher what happened, with exact numbers, before moving on.

---

## Ground rules (preregistration: these are binding)

1. **Do not change any locked choice:**
   - Anything in `config.yaml` except the `revision` hashes and `pilot_corpus.csv_path`.
   - `materials/anchors.yaml`.
   - The gate thresholds (Gate 1: Spearman ≥ .40; Gate 2: macro-F1 ≥ .70 and ECE ≤ .05).
   - The candidate model list.
   - The classifier (standardized features plus multinomial logistic regression, C = 1, stratified 5-fold cross-fitting).
   - The drift formula.
   - The vector pipeline: mean over subword tokens, L2-normalize, then subtract the mean of 100,000 random normalized tokens.
   - The sampling rules or the random seed.
2. **Never tune for better results.**
   - Do not try other layers, models, prompts, features, regularization, or thresholds because a gate failed.
   - Do not drop or relabel coded items.
   - A failed gate is a valid result: report it.
3. **Never use Glassdoor data.** The pilot uses only SemEval-2020 English and the public Hacker News corpus.
4. **Order matters.** Before running anything that touches pilot data (s02 onward), and before running Gate 1, ask the researcher to confirm that `config.yaml`, `materials/anchors.yaml`, and the lock document are posted on OSF as a time-stamped registration. Do not proceed without that confirmation.
5. **Allowed fixes.** You may fix mechanical problems that do not change what is computed, for example:
   - file paths, encodings, or data-layout differences;
   - library version incompatibilities;
   - API pagination or rate-limit handling;
   - device selection (CUDA, Apple MPS, or CPU);
   - out-of-memory problems, by lowering `embedding.batch_size`. This is the only config value you may change without asking.

   Log every fix in a new file, `FIXES_LOG.md`, with the date, file, what was wrong, what you changed, and why it does not change results.
6. **Ask before** anything not covered above, including:
   - a model that will not load;
   - SemEval data that cannot be found;
   - an API that is blocked;
   - any result that looks implausible.

   Do not substitute models, corpora, or methods on your own.
7. **Do not invent numbers.** Every number you report must come from a file in `outputs/`.

---

## Step 0: Environment and self-test

1. Create a virtual environment with Python 3.10–3.12 and run `pip install -r requirements.txt`.
2. Report which device will be used: CUDA GPU, Apple Silicon MPS, or CPU.
   - `src/embedder.py` currently chooses only between `cuda` and `cpu`.
   - On a Mac with Apple Silicon, add `mps` support (`torch.backends.mps.is_available()`) using float32 on MPS, and log it in `FIXES_LOG.md`.
   - On CPU only, warn the researcher that the Qwen 0.6B model will be slow, and suggest Colab (`pilot_colab.ipynb`) for Gate 1 and Gate 2.
3. Run `python tests/test_end_to_end.py`. It must print `ALL TESTS PASSED`. If it fails, fix the cause (mechanical only) before going further.

## Step 1: Pin model versions

For each model in `config.yaml` (`bert-base-uncased`, `roberta-large`, `Qwen/Qwen3-Embedding-0.6B`):

1. Get the current commit hash with `huggingface_hub.model_info(name).sha`.
2. Write it into `revision`.
3. Confirm each model loads with `transformers.AutoModel` and that its tokenizer is a fast tokenizer that returns `offset_mapping`. Qwen3 needs `transformers>=4.51`.

Show the researcher the three hashes. They go into the OSF lock, so ask the researcher to confirm the lock (ground rule 4) after this step.

## Step 2: Gate 1 (SemEval-2020 Task 1, English)

1. Check that `data/semeval2020_ulscd_eng/` exists, containing:
   - `targets.txt`
   - `truth/graded.txt`
   - `corpus1/lemma/*`, `corpus1/token/*`, `corpus2/lemma/*`, `corpus2/token/*` (`.txt` or `.txt.gz`)

   If the data are missing, ask the researcher to download the official English release from the SemEval-2020 Task 1 page (IMS Stuttgart) and unzip it there. If the folder layout differs, adapt `src/semeval.py` paths only, and log the change.
2. Run `python scripts/s01_gate1_semeval.py`.
3. Sanity checks to report:
   - `n_targets` should be 37 (English). If lower, say which targets were skipped and why.
   - The misaligned-line count printed at the start should be small relative to the corpus. If it is large, the lemma and token files are not aligned token by token: stop and report.
   - Published BERT results on this task with average pairwise distance are roughly .46–.57. Values far outside the range of about −.1 to .7, or every layer at the same value, suggest a bug in span location or centering. Check `span_vectors` on three or four example sentences and print the decoded subword tokens for the focal span.
4. Report a table of the best layer per model: Spearman for drift, Spearman for APD, and whether each passes Gate 1. Point to the full results in `outputs/gate1_results.csv`.

## Step 3: Pilot corpus (Hacker News)

1. Run `python scripts/s02_fetch_pilot_corpus.py`. It uses the public Algolia HN Search API, needs no key, and returns at most 1,000 hits per query, which the script handles by splitting date windows.
   - If `hitsPerPage=0` is rejected in `hn.count`, use `hitsPerPage=1` and log it.
   - If the API rate-limits, add backoff. The script is not resumable, so consider caching each year to `data/pilot/raw_<year>.jsonl` and log that change.
2. Report:
   - burnout comments per year;
   - total comments per year (`data/pilot/year_totals.csv`);
   - the background sample size.

   Flag any year with fewer than about 100 burnout comments.

## Step 4: Coding sheets, then STOP for human coding

1. Run `python scripts/s03_make_coding_sheets.py`.
2. Report usages per year and the coding sample size, which should be 400, year-stratified, with at least 20 per year.
3. Check that `materials/Coding_Sheet_CoderA.xlsx` and `CoderB.xlsx` open, contain 400 items, have a working category drop-down, and **show no dates or years**.
4. **Stop.** Tell the researcher the sheets are ready for two independent coders. The coding guide is in the "Guide" sheet and in section 2 of the lock document. Do not code any items yourself, and do not use an LLM to code. Labels must come from humans.

## Step 5: Agreement and resolution, with a STOP for discussion

1. When the researcher provides the two completed sheets, run `python scripts/s04_agreement.py check <A.xlsx> <B.xlsx>`.
2. Report:
   - α overall;
   - α on the exhaustion-vs-tiredness subset;
   - raw agreement;
   - percentages of items marked mixed and unsure.

   If α overall is below .667, tell the researcher the protocol requires retraining and a new sample. Do not continue to Gate 2. If α for the exhaustion-vs-tiredness subset is below .667, note that Gate 2 will merge those two categories automatically, as the protocol requires.
3. **Stop** while the coders resolve the disagreements in `materials/Disagreements.xlsx`.
4. Then run `python scripts/s04_agreement.py finalize <A.xlsx> <B.xlsx> materials/Disagreements.xlsx` and report the final category counts.

## Step 6: Gate 2 and model selection

1. Run `python scripts/s05_gate2_classify.py`, then `python scripts/s06_select_model.py`.
2. Report, for each model's best layer: macro-F1, ECE, Brier, accuracy, and whether it passes Gate 2.
3. Report the selected model and layer, or that none passes both gates. Under the protocol, if none passes, H2 is not tested: report that plainly.
4. Show the confusion matrix for the selected model and layer from `outputs/gate2_confusion.json`, with class names.

## Step 7: Control-word screen

1. Run `python scripts/s07_control_screen.py`. It makes many API calls, so add backoff if needed and log it.
2. Report:
   - burnout's log10 frequency per million comments;
   - how many of the 96 candidates fail the remote-work screen (list them);
   - how many fail the frequency or minimum-usage rules;
   - the size of the final control set.

   `control_screen.min_final` is null. Tell the researcher the final count and ask what minimum they want to register. Do not set it yourself.

## Step 8: Power-simulation inputs

1. Run `python scripts/s08_power_inputs.py --all-usages`.
2. Report:
   - category shares in the coded sample;
   - author- and thread-level ICCs (coded and predicted);
   - the null drift SD;
   - usages per year.

   Note that ICCs from 400 coded items are noisy.

## Step 9: Write `PILOT_RESULTS.md` and package the outputs

Create `PILOT_RESULTS.md` containing only facts taken from the output files. Report each number to three decimals with its source file.

1. **Models:** names and commit hashes.
2. **Gate 1:** best layer and Spearman per model; pass or fail.
3. **Coding:** n, α overall, α for exhaustion vs. tiredness, raw agreement, and whether exhaustion and tiredness were merged.
4. **Gate 2:** for each model's best layer, macro-F1, ECE, Brier, and pass or fail; then the confusion matrix for the selected model.
5. **Selected model:** model, layer, and the runner-up model.
6. **Control screen:** counts at each stage, and the final list.
7. **Power inputs:** category shares, ICCs, null drift SD, and per-year usage counts.
8. **Fixes:** a summary of `FIXES_LOG.md`.
9. **Deviations from the lock document:** this should be none. If there are any, list them explicitly.

Then zip `outputs/` (excluding `outputs/cache/`), together with `PILOT_RESULTS.md` and `FIXES_LOG.md`, into `pilot_outputs.zip`. The researcher will use these files to fill in the manuscript.
