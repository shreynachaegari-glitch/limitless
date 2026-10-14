# ResearchPulse

An autonomous research-monitoring agent (DejaVu Hackathon · Advanced Agentic AI · problem 5). It watches arXiv, OpenAlex and RSS for a topic, keeps every version of what it has read, and tells you **what changed, why it matters and how sure it is**, with every claim linked to the exact sentence in the source that supports it.

It is built around one idea: *a fluent summary is not the product; a defensible claim about what changed is.*

## What it does

| Brief requirement | How ResearchPulse does it |
|---|---|
| Topic-aware monitoring | Natural-language topic + concepts/synonyms/exclusions; calibrated semantic relevance (bge-small) + concept hits; reports inherit relevance from the original they cite |
| Duplicate detection | Identity (DOI / arXiv id / canonical URL), content hash, then citation links, similarity and matching claims. Guards stop independent studies merging (different model or different authors) |
| Change detection | `NEW_FINDING · DUPLICATE_REPORT · MINOR_UPDATE · MATERIAL_REVISION · CONTRADICTION · RELATED_DEVELOPMENT · INSUFFICIENT_EVIDENCE` |
| Historical comparison | Claims (entity · metric · value · unit · benchmark · method/batch/context) are diffed across versions and works; numbers are compared only when conditions match |
| Trend analysis | Distinct developments per concept per 14-day window vs a 42-day baseline; `INSUFFICIENT_HISTORY` when it can't tell |
| Significance ranking | Six separate scores (relevance, novelty, materiality, impact, evidence, urgency) with reasons; priority HIGH/MEDIUM/LOW/NONE/REVIEW |
| Source reliability | PRIMARY / SECONDARY / UNVERIFIED; "3 reports, based on 1 original source"; hype language and missing conditions flagged |
| Alert control | Backfill run sends nothing; duplicates and minor edits never alert; per-topic threshold; idempotency keys; repeat suppression |
| Traceable summaries | Every excerpt is verified verbatim against stored text; LLM prose must use only numbers present in the evidence, or the template is kept |

## How it works

```
Limitless UI (Vite/React/three) ──REST + NDJSON──► FastAPI
                                                     │ jobs table
worker (FOR UPDATE SKIP LOCKED, leases, scheduler) ◄─┘
  Scout → Resolver → Analyst → Verifier → Sentinel → Trend
  (fetch)  (relevance, (compare     (evidence   (alerts)  (distinct
            clusters)   history)     gate)                  clusters)
                 │
Postgres 17 + pgvector (HNSW) + tsvector FTS
```

Deterministic code owns fetching, identity, persistence, comparisons, scoring and alert policy. The model (NVIDIA NIM, OpenAI-compatible) only writes the prose, through a bounded agent loop (≤3 read-only tool calls, deadline, grounding check), and adjudicates ambiguous duplicate pairs. Source text is treated as data: instruction-like sentences are quarantined before a model sees them, and they cannot change a priority.

## Run it

Needs podman (or Docker), Python 3.12 + uv, Node 22.

```bash
# one-time
cd backend && uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -r requirements.txt && cd ..
(cd frontend && npm install)

./run.sh          # Postgres (podman), API :8000, worker, UI :5181
```

Optional live narration: create `backend/.env` with `NVIDIA_API_KEY=...` (from build.nvidia.com) and optionally `NVIDIA_MODEL=...`, then restart. Without a key everything works and the UI says the narrator is "template". `OFFLINE=1` forces templates.

## Demo

Two topics are seeded:
- **Efficient LLM inference (demo scenarios)**: labelled synthetic fixtures (URLs on the reserved `.invalid` domain, tagged "demo fixture" in the UI) that replay the six evaluation scenarios, one batch per run: baseline → news/contradiction/trend/weak claim/injection → minor edit → value revision → unchanged rerun.
- **Efficient LLM inference (live)**: real arXiv + OpenAlex, polled hourly by the worker.

## Tests

```bash
cd backend && .venv/bin/python -m pytest -q tests     # uses database researchpulse_test
```
The six brief scenarios, look-alike studies not merged, different conditions ≠ contradiction, prompt injection, idempotent rerun, failed source = partial success, evidence gate, SSRF guard, grounding, and the bounded agent loop with a scripted model.

## Credits

Ported ideas and code from the team's CodeRush 2.0 project (provider chain and retries, injection patterns, unit-aware contradiction rules, OpenAlex/arXiv parsing). Frontend is the team's Limitless site.
