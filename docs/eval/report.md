# ResearchPulse evaluation report

Generated 2026-10-14 15:20 IST by `python -m rp.evaluate`. All numbers measured by this script.

## A. Scenario corpus (model off, deterministic)

| Measure | Result |
|---|---|
| Change classification | 16/16 correct (100%) |
| Off-topic item rejected | yes |
| False contradictions | 0 |
| Duplicate grouping (pairwise) | precision 100%, recall 100% (0 false merges, 0 missed) |
| Alerts | precision 100%, recall 100% |
| Trend status | 4/4 correct |
| Evidence excerpts verified verbatim | 23/23 |
| Prompt injection obeyed | no (priority given: REVIEW) |
| Rerun of identical data | new rows [0, 0, 0] (idempotent) |
| Run time per run (s) | 0.55, 0.33, 0.07, 0.08, 0.06 |

## B. Claim extraction (labelled sentences)

| Measure | Result |
|---|---|
| Pattern extractor | precision 100%, recall 100% over 10 sentences |
| Qualifiers / negations captured | 2/2 |

## C. Live runs (real arXiv, Crossref, OpenAlex; read-only)

| Measure | Result |
|---|---|
| Runs / fetched / relevant | 1 / 57 / 34 |
| Model-proposed claims kept | 24 kept, 26 rejected (48% accepted) |
| Kept claims that are exact slices of the source | 24/24 |
| Model narration passing the grounding gate | 11/14 |
| Model usage | 52 calls, 78,576 tokens (counted for runs since usage tracking was added) |
| Run time (s) | 161.8 |

## What this does not show

- The scenario corpus is small and was written alongside the system; it checks the six brief scenarios, not general accuracy.
- Claim recall on real abstracts is not measured: there is no hand-labelled set of real abstracts yet.
- Confidence is an explained rule, not a calibrated probability.
