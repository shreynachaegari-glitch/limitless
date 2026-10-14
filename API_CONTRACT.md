# ResearchPulse API contract (v1)

Base: `/api` (Vite dev server proxies `/api` → `http://127.0.0.1:8000`). JSON everywhere; timestamps ISO-8601 UTC; ids are integers.

## Enums
- `classification`: `NEW_FINDING | DUPLICATE_REPORT | MINOR_UPDATE | MATERIAL_REVISION | CONTRADICTION | RELATED_DEVELOPMENT | INSUFFICIENT_EVIDENCE`
- `priority`: `HIGH | MEDIUM | LOW | NONE | REVIEW`
- `confidence`: `LOW | MEDIUM | HIGH` (an evidence-confidence indicator, NOT a probability; never show as a %)
- `source_role`: `PRIMARY | SECONDARY | UNVERIFIED | UNKNOWN`
- `trend_status`: `INSUFFICIENT_HISTORY | NO_SIGNAL | TENTATIVE | EMERGING | ESTABLISHED` (NO_SIGNAL = enough history, no change in activity)
- `run_status`: `QUEUED | RUNNING | SUCCEEDED | PARTIAL_SUCCESS | FAILED`

## GET /api/overview
```json
{
  "topics": 1, "active_topics": 1,
  "documents": 42, "clusters": 17,
  "unique_developments_7d": 9,
  "priority_counts": {"HIGH": 2, "MEDIUM": 3, "LOW": 4, "NONE": 6, "REVIEW": 2},
  "contradictions": 1, "trends": {"EMERGING": 1, "TENTATIVE": 1, "INSUFFICIENT_HISTORY": 2},
  "unread_alerts": 3,
  "last_run": {"id": 4, "status": "SUCCEEDED", "finished_at": "2026-10-14T06:10:00Z", "counters": {"fetched": 12, "new_versions": 3, "unchanged": 9}},
  "source_problems": 0,
  "llm": {"provider": "nvidia-nim", "model": "meta/llama-3.3-70b-instruct", "live": true},
  "agents": {
    "scout":    {"label": "items fetched",       "value": 42, "detail": "4 sources, 0 failing"},
    "resolver": {"label": "duplicate reports grouped", "value": 6, "detail": "17 clusters"},
    "analyst":  {"label": "meaningful changes",  "value": 5, "detail": "1 contradiction, 1 revision"},
    "verifier": {"label": "claims verified",     "value": 23, "detail": "2 downgraded"},
    "trend":    {"label": "trend signals",       "value": 2, "detail": "1 emerging"},
    "sentinel": {"label": "alerts sent",         "value": 3, "detail": "11 suppressed"}
  }
}
```

## Topics
`GET /api/topics` → `Topic[]`; `POST /api/topics` body `{name, description, concepts?: string[], exclusions?: string[], source_ids?: number[], relevance_threshold?: number, alert_threshold?: "HIGH"|"MEDIUM"|"LOW", poll_minutes?: number}` → `Topic`; `PATCH /api/topics/{id}` same fields + `active: boolean`.
```json
{"id": 1, "name": "Efficient LLM inference", "description": "Quantisation, speculative decoding and KV-cache work that makes LLM serving cheaper or faster",
 "concepts": ["quantization", "speculative decoding", "kv cache"], "exclusions": ["image generation"],
 "relevance_threshold": 0.45, "alert_threshold": "MEDIUM", "poll_minutes": 60, "active": true,
 "source_ids": [1, 2, 3], "last_run_at": "2026-10-14T06:10:00Z"}
```

## Runs
`POST /api/topics/{id}/run` → `{"run_id": 5}`
`GET /api/runs/{id}` → `{"id":5,"topic_id":1,"status":"RUNNING","phase":"analyze","started_at":"...","finished_at":null,"counters":{...},"error":null}`
`GET /api/runs/{id}/events?after=0` → `[{"seq":1,"ts":"...","agent":"scout","kind":"tool","message":"arXiv: 8 items","status":"done","ms":812}]`

## POST /api/agent  (Limitless console)
Body `{prompt, agent?}`. Starts a run for the topic named in the prompt (or the first active topic) and streams NDJSON `AgentEvent`s exactly as in `frontend/src/agent/types.ts` (`state`, `route`, `plan`, `step`, `tool`, `token`, `final`, `error`). `route.agentId` is one of `scout|resolver|analyst|verifier|trend|sentinel`.

## GET /api/feed?topic_id=&priority=&classification=&limit=50
```json
[{
  "id": 31, "topic_id": 1, "cluster_id": 7,
  "headline": "Nova-7B speedup revised from 2.4× to 1.9×",
  "classification": "MATERIAL_REVISION", "priority": "HIGH", "confidence": "HIGH",
  "what_changed": "The authors' v3 reports a 1.9× decode speedup on A100, down from 2.4× in v1.",
  "why_it_matters": "The headline speedup that 3 outlets repeated is now 21% smaller.",
  "scores": {"relevance": 82, "novelty": 40, "materiality": 90, "impact": 74, "evidence": 88, "urgency": 55},
  "source_count": 3, "independent_sources": 1, "independence_note": "3 reports, based on 1 original source",
  "sources": [{"document_id": 4, "title": "...", "url": "https://...", "publisher": "arXiv", "role": "PRIMARY", "published_at": "...", "retrieved_at": "..."}],
  "published_at": "2026-10-01T00:00:00Z", "first_seen_at": "2026-10-14T06:10:00Z",
  "synthetic": true, "read": false, "narrator": "template"
}]
```
`synthetic: true` means the item came from the labelled demo-scenario fixture source; the UI must show a visible "demo fixture" tag.
`narrator`: `"llm"` or `"template"` (says who wrote what_changed/why_it_matters).

## GET /api/changes/{id}  (Investigation view)
Feed item fields plus:
```json
{
  "comparisons": [{
    "entity": "Nova-7B", "metric": "decode speedup", "context": "A100",
    "old": {"value": 2.4, "unit": "x", "text": "2.4x decode speedup on A100", "document_id": 4, "version": 1, "source": "arXiv", "role": "PRIMARY", "published_at": "..."},
    "new": {"value": 1.9, "unit": "x", "text": "1.9x decode speedup on A100", "document_id": 4, "version": 3, "source": "arXiv", "role": "PRIMARY", "published_at": "..."},
    "delta": {"abs": -0.5, "rel_pct": -20.8}, "comparable": true, "comparability_note": "same entity, metric, unit and hardware"
  }],
  "evidence": [{"document_id": 4, "version": 3, "role": "supports", "excerpt": "...", "url": "https://...", "verified": true}],
  "score_reasons": ["materiality 90: reported value changed by 20.8% in the same work", "evidence 88: primary source, excerpt verified"],
  "uncertainty": ["Only one independent source"],
  "evidence_gaps": [],
  "related": [{"id": 12, "headline": "...", "classification": "DUPLICATE_REPORT"}],
  "flags": ["prompt_injection_quarantined"],
  "audit": {"policy_version": "p1", "model": "template", "tool_calls": 2, "validation": "passed"}
}
```

## GET /api/clusters/{id}
`{"id":7,"title":"...","members":[{"document_id":4,"title":"...","url":"...","publisher":"...","role":"PRIMARY","relation":"origin","published_at":"..."}],"independent_sources":1,"assessments":[{"id":31,"classification":"...","headline":"..."}]}`

## GET /api/trends?topic_id=
```json
[{"id": 2, "concept": "speculative decoding", "status": "EMERGING", "window_days": 14,
  "current_count": 4, "baseline_count": 1, "baseline_windows": 3, "article_count": 4,
  "summary": "4 distinct developments in the last 14 days vs 1 in the prior 42 days.",
  "limitations": ["Counts distinct clusters, not articles", "Co-occurrence only; no causal claim"],
  "series": [{"window_start": "...", "count": 0}, {"window_start": "...", "count": 1}],
  "evidence": [{"cluster_id": 9, "title": "...", "published_at": "..."}], "contradicting": []}]
```

## Sources
`GET /api/sources` → `[{"id":1,"kind":"fixture|arxiv|openalex|rss","name":"Demo scenarios (synthetic)","url":null,"role_default":"UNKNOWN","enabled":true,"health":{"status":"ok|failing|never_run","last_checked":"...","last_success":"...","error":null,"items_last_run":5}}]`
`POST /api/sources` `{kind, name, url?, query?}`; `PATCH /api/sources/{id}` `{enabled}`.

## Alerts
`GET /api/alerts?unread=true` → `[{"id":3,"assessment_id":31,"priority":"HIGH","headline":"...","created_at":"...","read":false,"suppressed_repeats":2}]`
`POST /api/alerts/{id}/read` → `{"ok":true}`

## Feedback
`POST /api/feedback` `{assessment_id, verdict: "useful"|"not_useful"|"wrong_cluster"|"wrong_classification", note?}` → `{"ok":true}`

## Demo
`POST /api/demo/reset` → wipes data and reseeds the demo topic + sources (`{"ok":true,"topic_id":1}`).
