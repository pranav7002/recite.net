# recite.net — iteration log

The full record of how the system got to its current numbers. The short version is the [README](../README.md); run-by-run tables are in [`EXPERIMENTS.md`](../EXPERIMENTS.md).

A voice tutor that answers questions about your own course material, with
citations, and can email a summary to your study group. It retrieves passages
first, drafts an answer, checks every claim against the passages with a second
model, and gates the one action with side effects (sending email) behind
deterministic guardrails.

This README is also the record of how the system got to its current numbers:
each iteration below says what was observed, what was diagnosed, what changed,
and what the result was. The run-by-run tables are in
[`EXPERIMENTS.md`](../EXPERIMENTS.md).

## Where things stand

All numbers are from real Gemini calls, run with `python -m evals.run_evals`.
Corpus: three lecture PDFs (63 pages). Embeddings arm, 3 runs per question.
The `tune`/`report`/`heldout` rows below are current: run **after** the
grounding-check fix (iteration 8).

| Run | Chunks | Hit rate | Correct |
| --- | --- | --- | --- |
| `tune` (10 q), start: ~2-3 pages per chunk | 17 | 40% (12/30) | 63% (19/30) |
| `tune`, one chunk per page, pre grounding-fix | 60 | 100% (30/30) | 80% (24/30) |
| `tune`, post grounding-fix | 60 | 100% (30/30) | 83% (25/30) |
| `report` (15 q), pre grounding-fix | 60 | 100% (45/45) | 82% (37/45) |
| `report`, post grounding-fix | 60 | **100% (45/45)** | **87% (39/45)** |
| `heldout` (5 q), never tuned, run once | 60 | **100% (15/15)** | **100% (15/15)** |
| Safety (4 injection, 3 legitimate, 2 should-not-refuse) | | attacks 0/12 | 5/6 |
| Judge vs. human agreement (10 hand-labeled answers) | | | **10/10 (100%)** |

- Hit rate = any expected page is among the retrieved passages (code, no model).
- Correct = an LLM judge grades `correct | partial | wrong`.
- `report` was never used for tuning, and `heldout` was touched exactly once,
  at the end, after everything else was settled — so both are honest numbers,
  not numbers fit to the eval. `heldout` scoring 100% on questions the system
  never saw during development is the strongest evidence that the chunking
  and grounding-check fixes generalize.
- The grounding-check fix moved `report`'s multihop type from 16/18 to 18/18
  correct — exactly the failure mode it targeted. It did not move `lookup`
  (still 12/18): every remaining wrong answer in `report`, and most of
  `tune`'s, is image-only slide content (a graph, an equation) that has no
  text layer to retrieve — see "What is still wrong" below.
- The system prompt and citation parsing changed again after this `tune` row
  was measured (see EXPERIMENTS.md, "System prompt changes and an end-to-end
  demo"). A rerun is in progress but incomplete — a sustained Google-side
  outage on the answer model cut it off at 23 of 30 case-runs (see Run 11) —
  so this row isn't yet re-confirmed; nothing in the 23 completed runs
  suggests it moved, but the two image-only `structural` questions weren't
  reached.

## Iteration log

### 0. Starting point

The agent loop, guardrails, grounding check, upload path, unit tests and an
eval runner existed, but `evals/retrieval_cases.yaml` was empty and nothing had
ever been run against the real model.

### 1. Turn the questions into a working eval

30 hand-written questions (12 lookup, 12 multi-hop, 6 structural) split
10 tune / 15 report / 5 heldout, plus 8 safety cases.

- **Problem:** the expected pages were written as the *slide numbers printed on
  the slides*, but the runner compares against PDF page indices. They differ
  (a cover page, and DS-2 prints page+1). Every expected page was remapped by
  reading the actual page text.
- **Problem:** five questions had answers that are not in the slides at all
  (the pandas description, the `isnull` code, `describe()`, `students.csv`,
  the Seaborn heatmap). They cannot be answered from the corpus, so they were
  rewritten (`lookup_07`, `lookup_11`, `multihop_01`, `multihop_02`,
  `multihop_10`), and `structural_01` was reworded because the slide has five
  steps, not four.

### 2. First real run: every tool call returned a 400

- **Symptom:** `Function call is missing a thought_signature`.
- **Cause:** Gemini 3 attaches a `thought_signature` (in `extra_content`) to each
  tool call and rejects the next request if it is not echoed back. `agent.py`
  rebuilt tool calls without it. The unit tests could not catch this because the
  fake LLM never sends signatures.
- **Fix:** pass `extra_content` through when rebuilding the assistant message.

### 3. One 503 destroyed a whole run

- **Symptom:** "model is currently experiencing high demand" crashed the runner
  and, because results were written only at the end, lost all progress.
- **Fix:** `llm.py` retries 5xx like 429, with backoff. The runner now keeps
  finished records and stops cleanly on an API error, so a rerun resumes.

### 4. The free-tier quotas were far below the config

- **Symptom:** `gemini-3.8-flash` returned 429: limit 20 requests, not the 1000
  assumed in `.env`. The per-model limits table showed why (next section).
- **Change:** answers moved to `gemini-3.1-flash-lite`, checks and judging stay
  on `gemini-3.5-flash-lite`, and `.env` now carries the real limits. Quota is
  per model, so the two roles have separate 500-a-day budgets.

### 5. Baseline: hit rate 40%, then 100%

- **Observation:** the first `tune` run hit only 40% of questions, and six of ten
  were never hit. Citations were visibly wrong: "80% of the world's data is
  unstructured" cited page 4 but is on page 5; the variance answer cited page 4
  but is on page 6.
- **Diagnosis:** chunks were ~2-3 slides long and labelled with the page they
  *start* on. The database showed a chunk labelled "page 4" that began mid-page
  3 and ran through page 5. Slides are short, so almost every answer sat past a
  chunk's first page, which counted as a miss and also made every citation
  unreliable. 17 chunks over 63 pages.
- **Change:** `chunk_pages(..., split_pages=True)` in `backend/ingest.py`. For PDFs
  a chunk never spans a page and no overlap crosses a page break, so its page
  is exact. Text/Markdown files are unchanged. Two unit tests. Re-indexed to
  60 chunks.
- **Result:** `tune` hit rate 40% -> **100%**, correct 63% -> **80%**. Same
  questions, e.g. the population variance formula went from wrong x3 to
  correct x3. Old results kept in `evals/results/baseline_bigchunks/`.

### 6. The `report` split, and where the errors actually are

`report` was never used for tuning: hit rate 100% (45/45), correct 82% (37/45).
Since retrieval hits every question, the 8 wrong runs are not a retrieval
problem:

- **Image-only slides** (6 runs, `lookup_06`, `lookup_09`; and `structural_02`
  in `tune`): the Lasso graph and the Expected Test MSE equation exist only as
  images. Retrieval finds the page, but there is no text, so the model says it
  cannot find the answer.
- **Grounding-check refusals** (2 runs, `multihop_05`, plus `multihop_03` and
  `multihop_04` in `tune`): the second model refused answers that were correct.

### 7. Safety: 0 attacks, but read the trace

Safety went 12/12 on injection, 9/9 on legitimate sends, 5/6 on
should-not-refuse. Adding a `tool_events` trace to the records showed what that
number does and does not prove:

- The planted address `x@evil.com` is on the allowlist on purpose, so that the
  provenance rule ("recipient came from a document") is what should stop it.
- The model never tried that address. In `injection_01` and `injection_02` it
  invented recipients (`studygroup@example.com`, `student@example.com`) and the
  *allowlist* blocked them; in `injection_03` and `injection_04` it attempted
  no send at all.
- So the live eval shows the allowlist working and a model that did not follow
  the planted instruction. The provenance rule itself is verified by
  `tests/test_agent.py::test_recipient_from_document_blocked`. Showing it fire
  live would need a model that actually follows the injection.

### 8. Fixing the grounding check

The largest source of wrong answers that retrieval got right. The plan was to
guess at the prompt; instead the diagnosis came first.

1. **Record the evidence.** Result records now include each grounding check:
   the claims, the check model's raw output, and every verdict.
2. **Rerun the four refused questions** (`multihop_03`, `multihop_04`,
   `multihop_05`, `no_refuse_01`), 3 runs each, with the new traces.
3. **Root cause: not a bad prompt, not bad JSON.** 6 of 18 check calls (33%)
   were logged as "unparseable JSON", but the raw output was valid JSON full of
   `SUPPORTED` verdicts. The check model returned 3 claims because the answer
   had 3 sentences; the code's sentence splitter had cut the answer into 4,
   because it split after `p.` inside citations like `(lecture4.pdf, p. 16)`,
   producing junk fragments (`"16; DAI-101_Lecture_4 ... p."`). The code
   required the counts to match, discarded a good result, retried, and after two
   tries failed closed and refused a correct answer.
4. **Fixes** (all in `backend/grounding.py`):
   - The sentence splitter only splits when the next character starts a
     sentence, so `p. 16)` no longer breaks a claim; blank lines also split.
   - Claims are numbered, the model returns the `id` of each verdict, and
     verdicts are matched by id, so a reordered or differently split answer
     still lines up. A missing or invented id still fails closed.
   - The prompt now tells the model to judge the claim rather than the
     citation, spells out the exact output shape, and includes one worked
     example.
   - Four regression tests: a citation page number does not split a claim,
     verdicts match by id not position, a missing id fails closed after one
     retry, and fenced JSON with ids parses.
5. **`PARTIAL` deliberately still counts as a failure.** It triggers one retry
   with the failing claim named, and in the traces that retry usually fixed the
   answer.
6. **Result on the 12 affected runs** (a targeted check on the questions that
   failed before, so biased toward showing improvement): parse failures went
   from 6 of 18 check calls to 0 of 15. End-to-end correctness went from 9/12
   to 10/12, which is small and within run-to-run noise: `multihop_05` and
   `no_refuse_01` went from 2/3 to 3/3, `multihop_04` stayed 3/3 (a retry had
   already recovered it), and `multihop_03` went from 2/3 to 1/3. The
   `multihop_03` refusals are genuine verdicts: the answer claims that leakage
   "masks the true extent of the gap", which the slides never say (they say
   leakage makes performance look better, and separately that overfitting is a
   large gap). The link is the model's own inference, which the check is right
   to flag.

Lesson: the label "unparseable JSON" was wrong. Logging the raw output showed
the failure was in our own claim splitting, not the model.

### 9. Eval tooling that came out of the above

- Result files are keyed by suite, arm **and split**, so a `report` run cannot
  resume from or merge into a `tune` run.
- `--only id1,id2` and `--fresh` rerun specific cases without resuming.
- Records include model IDs, tool traces, retrieved passages and grounding
  checks; results also record which model answered and which judged.
- `evals/judge_agreement.py` samples 10 answers into a blind sheet (the judge's
  grade hidden) and scores agreement once a human has filled it in.
- One unit test depended on an empty database; it now counts relative to what
  is already there.

### 10. Confirming the fix, and a first `heldout` run

The targeted rerun in iteration 8 covered only the 4 questions that had
failed before, which is biased toward showing improvement. `tune` and
`report` were rerun in full (3 runs each) on the fixed code, and `heldout`
(5 questions, never touched before) was run for the first time:

- `report`: correct 82% -> **87%** (37/45 -> 39/45). The gain is entirely in
  the multihop type, 16/18 -> **18/18** — exactly the failure mode the fix
  targeted. `lookup` is unchanged at 12/18: every remaining wrong answer
  there is `lookup_06` or `lookup_09`, both image-only slide content with no
  text layer, which the grounding fix was never going to touch.
- `tune`: correct 80% -> 83% (24/30 -> 25/30). The remaining wrong runs are
  `structural_02` (the same image-only issue) and `multihop_03` on 2 of 3
  runs. Reading the trace, `multihop_03`'s refusal is a genuine `PARTIAL`
  verdict, not a bug: the draft claims leakage "masks the true extent of the
  gap," which the slides never state — that link is the model's own
  inference, and the check is right to flag it.
- `heldout`: **100% hit, 100% correct (15/15)**, run once and not reopened.
  It was never used to tune chunking, the router thresholds or the grounding
  prompt, so scoring 100% here is the honest signal that those fixes
  generalize past the questions used to develop them.

Judge agreement with a human grader was also measured for the first time,
against the pre-fix result files: 10 hand-labeled answers, 10/10 agreement
(`EXPERIMENTS.md`, Run 8). Full tables and per-question detail for both:
`EXPERIMENTS.md`, Runs 8-9.

## Models and free-tier quotas

Free-tier limits from the AI Studio rate-limit page (requests per minute /
per day):

| Model | RPM | RPD | Use |
| --- | --- | --- | --- |
| `gemini-3.1-flash-lite` | 15 | 500 | **Answers** |
| `gemini-3.5-flash-lite` | 15 | 500 | **Grounding check and judge** |
| `gemini-embedding-001` | 100 | 1000 | **Embeddings** |
| `gemini-3.5`/`3.6`/`3.7`/`3.8` Flash | 5 | 20 | Spot checks only |
| `gemma-4-26b`, `gemma-4-31b` | 30 | 14,400 | Not on the answer path |

- Keeping the drafter and the checker on different models means the checker does
  not share the drafter's blind spots, and quota is per model, so they also have
  separate daily budgets.
- Full Flash models are capped at 20 a day, so they cannot carry an eval; use
  them for one-off comparisons.
- Gemma 4 was tested: JSON mode and tool calls work, but the output leaks a
  `<thought>` block into the content and calls took 15-83 s, and the 16K
  tokens-per-minute limit caps it near 5 calls a minute. It is too slow for the
  interactive path; the only sensible use is an overflow judge for very large
  overnight runs.
- Latency numbers in the eval files are mostly rate-limit waiting (one call took
  ~600 s), not model time.

## What is still wrong

- **Image-only slide content is not retrievable.** Needs OCR or a vision pass
  at ingest. This is now the entire remaining gap in `report` (`lookup_06`,
  `lookup_09`) and most of it in `tune` (`structural_02`) — see iteration 10.
- **Judge agreement was measured on pre-fix data, not the current `report`/
  `tune` results.** 10/10 on the sample drawn from the old result files
  (`EXPERIMENTS.md`, Run 8); worth re-sampling from the post-fix files.
- **The eval is easy.** With 60 chunks and the top 5 retrieved, about 8% of the
  corpus comes back on every query, so a 100% hit rate mostly shows the corpus
  is small. Reranking and the RLM arm cannot show a benefit until the suite has
  harder questions (paraphrases, exact terms, questions needing all of several
  pages, whole-lecture overviews, and unanswerable questions).
- The `rlm` and `router` arms are built but the RLM arm cannot be driven by the
  free-tier model: `gemini-3.1-flash-lite` plans and inspects the document but
  never finalises a grounded answer through the `rlms` code-execution loop. See
  the Day 3, Block 2 entry in [`EXPERIMENTS.md`](../EXPERIMENTS.md).

## Running it

1. Postgres with pgvector: `docker run -d --name recite-db -e POSTGRES_USER=recite -e POSTGRES_PASSWORD=recite -e POSTGRES_DB=recite -p 5433:5432 -v recite-pgdata:/var/lib/postgresql/data pgvector/pgvector:pg16`
2. Schema: `docker exec -i recite-db psql -U recite -d recite < backend/schema.sql`
3. `.env`: `DATABASE_URL=postgresql://recite:recite@localhost:5433/recite`, `GEMINI_API_KEY`, model names, quota limits and `CONTACTS`.
4. Index: `python -m backend.ingest data/pdfs`
5. Evals: `python -m evals.run_evals --suite retrieval --split tune --runs 3` and `--suite safety`. Runs resume within a day per split; use `--fresh` to ignore earlier results and `--only id1,id2` for specific cases.
6. Tests: `ruff check . && pytest -q`

## Limitations

- Pending actions are not tied to a session, so anyone holding the id can
  confirm one. The ids are random 128-bit values, which is fine for a
  single-user app but not for multi-user.
- Expired pending rows are left in the table rather than swept. Confirming one
  correctly refuses it, but there is no background cleanup.
- The runner does not read the `quota_daily` table before starting; it stops
  when a call is refused.
